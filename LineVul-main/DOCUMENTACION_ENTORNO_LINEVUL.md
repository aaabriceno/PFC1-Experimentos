# LineVul en Docker con GPU RTX 5060 — Documentación paso a paso

Guía de todo lo que se hizo para poder probar el modelo LineVul en esta máquina,
con la explicación del **porqué** de cada decisión.

Última actualización: 2026-10-01 (re-corrida del test completo el 2026-09-30,
ver la sección 11).

---

## 1. ¿Qué es LineVul y qué se necesita para probarlo?

LineVul es un modelo Transformer (CodeBERT, arquitectura RoBERTa) que predice si
una función escrita en C/C++ es vulnerable, y además señala las líneas
sospechosas usando los pesos de atención. Para **probarlo en modo inferencia**
(sin re-entrenar) se necesitan 3 piezas:

| Pieza | Qué es | Tamaño aprox. |
|---|---|---|
| Dependencias Python | torch, transformers, tokenizers, etc. | ~4 GB instaladas |
| `test.csv` | Dataset de prueba Big-Vul (funciones C/C++ etiquetadas) | 981 MB |
| `12heads_linevul_model.bin` | Pesos del modelo ya entrenado | 952 MB |

El script principal es `linevul/linevul_main.py`. Con la bandera `--do_test`
solo lee `test.csv` (los argumentos `--train_data_file` y `--eval_data_file`
son obligatorios en la línea de comandos pero **no se leen** en modo test —
se verificó en el código, líneas 57–64 y 1243–1248).

---

## 2. Diagnóstico de esta máquina

Comandos usados y hallazgos:

```bash
python3 --version          # Python 3.12.3
nvidia-smi                 # RTX 5060 Laptop 8GB, driver 595.71.05, CUDA 13.2
docker --version           # Docker 29.1.3
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
# -> Error: could not select device driver "" with capabilities: [[gpu]]
```

Conclusiones:

1. **Python 3.12 del host es demasiado nuevo** para el `requirements.txt`
   original (torch 1.13.1 solo existe hasta Python 3.10).
2. **La GPU es Blackwell (sm_120)** — ver sección 3.
3. **El driver 595 es suficiente** (soporta CUDA 12.8+, que es lo que
   necesitamos dentro del contenedor).
4. **Falta `nvidia-container-toolkit`**: Docker funciona, pero no puede pasar
   la GPU a los contenedores todavía. Ver sección 8.

---

## 3. El problema central: arquitectura Blackwell vs torch 1.13

Cada generación de GPU NVIDIA tiene una *compute capability* (código `sm_XX`).
PyTorch **no compila nada en tu máquina**: los binarios de pip traen kernels
CUDA precompilados solo para las arquitecturas que existían cuando se publicó
esa versión.

| Generación | GPU ejemplo | Compute capability | Soportada desde |
|---|---|---|---|
| Ampere (2020) | RTX 3060 | sm_86 | CUDA 11 / torch 1.8 |
| Ada (2022) | RTX 4060 | sm_89 | CUDA 11.8 / torch 2.0 |
| **Blackwell (2025)** | **RTX 5060** | **sm_120** | **CUDA 12.8 / torch 2.7** |

`torch==1.13.1` (diciembre 2022) trae kernels hasta sm_86/sm_90. Para la
RTX 5060 **no hay nada que "instalar aparte"**: los kernels sm_120 no existen
dentro de ese binario. Si se fuerza, cualquier operación en GPU lanza:

```
CUDA error: no kernel image is available for execution on the device
```

**Por qué Docker no arregla esto por sí solo**: Docker aísla el *userland*
(Python, librerías), pero la GPU no se virtualiza — el contenedor accede a la
RTX 5060 física con el driver del host. Un torch 1.13 dentro del contenedor
sigue sin kernels sm_120.

| Problema | ¿Docker lo resuelve? |
|---|---|
| Python 3.12 del host vs 3.10 requerido | ✅ sí — imagen base `python:3.10` |
| torch 1.13 sin kernels para RTX 5060 | ❌ no — hay que usar torch ≥ 2.7 igualmente |

**Decisión**: Docker con `python:3.10-slim` + torch 2.7.1 build cu128,
manteniendo el resto de dependencias con los pins exactos del paper
(`transformers==4.26.0`, `tokenizers==0.13.2`), que son las que definen el
comportamiento del tokenizador y del modelo. Cambiar torch afecta el motor de
ejecución, no la lógica del modelo — los resultados numéricos no cambian.

---

## 4. Diseño del Dockerfile — decisiones y porqués

Archivo: [`Dockerfile`](Dockerfile). Decisiones:

### 4.1 Imagen = solo entorno; el proyecto se monta con `-v`

La imagen NO copia el código ni los datos (`COPY . .` no existe). Se ejecuta con
`-v $(pwd):/app`. Razones:

- Editar código sin reconstruir la imagen.
- Los ~2 GB de dataset + modelo no engordan la imagen ni el build context.
- Los resultados que el script escribe quedan en tu disco al salir.

### 4.2 Base `python:3.10-slim`, NO `nvidia/cuda`

- `tokenizers==0.13.2` y `pandas==1.5.2` publican *wheels* precompilados para
  CPython 3.10; en 3.12 habría que compilarlos desde fuente (Rust/C) y torch
  1.x directamente no existe.
- No hace falta imagen `nvidia/cuda`: el wheel de torch cu128 **ya empaqueta
  sus propias librerías CUDA** (cublas, cudnn, etc. llegan como dependencias
  `nvidia-*-cu12` de pip). Del host solo se necesita el **driver**, que entra
  al contenedor vía nvidia-container-toolkit con `--gpus all`.

### 4.3 torch en su propia capa, primero

Es la capa más pesada (~3 GB descargados). Al ir primero y sola, Docker la
cachea: cambiar cualquier otra dependencia no obliga a re-descargarla.

### 4.4 `ENV HF_HOME=/app/.hf_cache`

`linevul_main.py` descarga `microsoft/codebert-base` (~500 MB) de HuggingFace
en el primer arranque. Al apuntar la caché dentro de `/app` (que está montado
desde el host), se descarga **una sola vez** y persiste entre ejecuciones.

### 4.5 `requirements-docker.txt`

Copia del `requirements.txt` original con dos cambios documentados en el propio
archivo: se quitan `torch==1.13.1` y las librerías `nvidia-*-cu11` (van en la
capa de torch cu128), y todo lo demás mantiene el pin exacto del paper.

---

### 4.6 Actualizar pip antes de instalar (bug encontrado durante el build)

El primer intento de build falló con:

```
ERROR: ... has inconsistent Name: expected 'typing-extensions', but metadata has 'typing_extensions'
ERROR: No matching distribution found for typing-extensions>=4.10.0
```

**Causa**: el pip que trae la imagen `python:3.10-slim` (23.0.x) tiene un bug
de normalización de nombres de paquetes (guión vs guión bajo) que lo hace
rechazar wheels válidos del índice de PyTorch. Corregido en pip ≥ 23.3.
**Solución**: capa `RUN pip install --upgrade pip` antes de instalar torch.

## 5. Construcción de la imagen

```bash
cd LineVul-main
docker build -t linevul:cu128 .
```

---

## 6. Descarga del modelo y del dataset

Los enlaces vienen del README original (Google Drive, vía `gdown`):

```bash
# Modelo entrenado (952 MB) -> linevul/saved_models/checkpoint-best-f1/
mkdir -p linevul/saved_models/checkpoint-best-f1
cd linevul/saved_models/checkpoint-best-f1
gdown "https://drive.google.com/uc?id=1oodyQqRb9jEcvLMVVKILmu8qHyNwd-zH"
cd ../../..

# Dataset de test (981 MB) -> data/big-vul_dataset/
cd data/big-vul_dataset
gdown "https://drive.google.com/uc?id=1h0iFJbc5DGXCXXvvR6dru_Dms_b2zW4V"
cd ../..
```

**Por qué la ruta exacta importa**: en modo `--do_test`, el script busca el
modelo en `<output_dir>/checkpoint-best-f1/<model_name>`
(línea 1244 de `linevul_main.py`), es decir
`./saved_models/checkpoint-best-f1/12heads_linevul_model.bin`.

Nota: `gdown` puede fallar con "Too many users have viewed or downloaded this
file recently" — es un límite de Google Drive, no un error tuyo. Esperar unas
horas o descargar manualmente desde el navegador con el mismo ID.

---

## 7. Riesgo de compatibilidad detectado: `torch.load` en torch ≥ 2.6

Desde torch 2.6, `torch.load()` usa `weights_only=True` por defecto (medida de
seguridad contra pickles maliciosos). El código de LineVul (línea 1246) carga
el `.bin` sin ese argumento. Como el archivo es un `state_dict` puro (solo
tensores), la carga funciona con el nuevo default. Se verificó en el smoke
test de la sección 9. Si algún otro modelo `.bin` fallara con
`UnpicklingError`, la solución sería añadir `weights_only=False` en esa línea
(solo con archivos de origen confiable).

---

## 8. Habilitar la GPU en Docker: nvidia-container-toolkit

Sin este componente, `docker run --gpus all` falla con:

```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

**Qué es**: un runtime hook que monta el driver de NVIDIA del host (las
librerías `libcuda.so`, los device nodes `/dev/nvidia*`) dentro del contenedor
al arrancarlo. Sin él, el contenedor no ve la GPU aunque el host la tenga.

Instalación (Ubuntu 24.04, requiere sudo):

```bash
# 1. Añadir el repositorio oficial de NVIDIA
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. Instalar y configurar el runtime de Docker
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verificación:

```bash
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
# Debe mostrar la RTX 5060
```

---

## 9. Smoke test en CPU (validación del pipeline sin GPU)

Antes de tener la GPU habilitada se validó el pipeline completo (tokenizador,
carga del modelo, inferencia, métricas) con una muestra pequeña en CPU. Esto
descarta problemas de compatibilidad de librerías independientemente del
toolkit de NVIDIA.

```bash
# Muestra de 200 funciones (dentro del contenedor, pandas maneja bien los
# CSV con código multilínea; un "head -200" cortaría filas por la mitad)
docker run --rm --user $(id -u):$(id -g) -v "$(pwd)":/app linevul:cu128 python -c "
import pandas as pd
df = pd.read_csv('../data/big-vul_dataset/test.csv')
df.head(200).to_csv('../data/big-vul_dataset/test_sample.csv', index=False)
print(df.shape, '->', 200)
"

# Inferencia CPU sobre la muestra
docker run --rm --user $(id -u):$(id -g) -v "$(pwd)":/app linevul:cu128 python linevul_main.py \
  --model_name=12heads_linevul_model.bin \
  --output_dir=./saved_models \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_test \
  --train_data_file=../data/big-vul_dataset/train.csv \
  --eval_data_file=../data/big-vul_dataset/val.csv \
  --test_data_file=../data/big-vul_dataset/test_sample.csv \
  --block_size 512 \
  --eval_batch_size 8
```

**Por qué `--user $(id -u):$(id -g)`**: sin esto el contenedor corre como
root, y todo archivo que escriba en el volumen montado (muestras, resultados,
caché de HuggingFace) queda como propiedad de root en tu disco — luego no
puedes borrarlo sin sudo.

Resultado real (2026-07-08): **funcionó a la primera** —

```
***** Test results *****
  test_accuracy  = 0.99
  test_f1        = 0.9286
  test_precision = 0.9286
  test_recall    = 0.9286
```

35 segundos para 200 funciones en CPU. La muestra de 200 (14 vulnerables)
da métricas consistentes con el paper (F1 0.91 sobre el test completo de
18,864 funciones). El test completo en CPU tomaría ~55 min (extrapolado de
esta muestra, no medido); en GPU tomó 14 min 56 s (sección 11).
También confirma que el riesgo de la sección 7 (`torch.load` con
`weights_only=True`) no se materializó: el `.bin` es un state_dict puro.

---

## 10. Ejecución final con GPU (test set completo)

Una vez instalado nvidia-container-toolkit:

```bash
cd LineVul-main
docker run --rm --gpus all --user $(id -u):$(id -g) -v "$(pwd)":/app linevul:cu128 python linevul_main.py \
  --model_name=12heads_linevul_model.bin \
  --output_dir=./saved_models \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_test \
  --train_data_file=../data/big-vul_dataset/train.csv \
  --eval_data_file=../data/big-vul_dataset/val.csv \
  --test_data_file=../data/big-vul_dataset/test.csv \
  --block_size 512 \
  --eval_batch_size 32
```

**Por qué `--eval_batch_size 32` y no 512 como en el README**: el README asume
GPUs de servidor (32–48 GB). Cada muestra son 512 tokens; un batch de 512
muestras necesita ~30+ GB para las activaciones de RoBERTa-base. Con 8 GB de
VRAM, 32 es un valor seguro (~3–4 GB). Si sobra memoria (ver `nvidia-smi`
durante la ejecución), se puede subir a 64.

**Resultados esperados** (paper, RQ1): F1 = 0.91, Precision = 0.97,
Recall = 0.86.

**Rutas y directorio de trabajo.** El comando se lanza desde `LineVul-main/`,
pero el `WORKDIR` final de la imagen es `/app/linevul` (último `WORKDIR` del
`Dockerfile`). Por eso, dentro del contenedor, `../data/...` apunta a
`LineVul-main/data/...` y `./saved_models` a `linevul/saved_models/`.
`train.csv` y `val.csv` no existen en disco y no hacen falta: en `--do_test`
no se leen (sección 1).

**No hay barra de progreso durante la inferencia.** Después de
`Num examples = 18864` y `Batch size = 32` no se imprime nada durante unos
15 min, hasta `Test results`. Para confirmar que sigue vivo: `nvidia-smi`
(GPU cerca del 100 %) o `docker ps`. Con `--rm` y el contenedor en primer
plano, cerrar la terminal o pulsar Ctrl+C lo mata y se pierde la corrida.

---

## 11. Bitácora de resultados

Cronología real de la puesta en marcha (2026-07-08):

1. **Diagnóstico**: Python 3.12 host, RTX 5060 sm_120, driver 595 OK,
   Docker OK, nvidia-container-toolkit ausente.
2. **Build intento 1**: falló — pip 23.0 de la imagen base rechaza wheels del
   índice PyTorch (sección 4.6). Fix: `pip install --upgrade pip` como primera
   capa.
3. **Descargas en paralelo**: `test.csv` (981 MB) y
   `12heads_linevul_model.bin` (952 MB) vía gdown, sin problemas de cuota.
4. **Build final**: imagen `linevul:cu128` = **7.27 GB** (de los cuales ~6 GB
   son torch + librerías CUDA cu12 empaquetadas en el wheel).
5. **Smoke test CPU** (muestra de 200 funciones, 14 vulnerables):
   accuracy 0.99, F1/precision/recall 0.9286, en 35 s. Pipeline completo
   validado sin GPU: tokenizador BPE, carga del state_dict con torch 2.7,
   transformers 4.26 + torch 2.7.1 conviven sin errores.
6. **nvidia-container-toolkit instalado** (por el usuario, sección 8).
   Verificación dentro de la imagen: torch 2.7.1+cu128 detecta la RTX 5060,
   compute capability (12, 0) = sm_120, matmul en GPU OK.
7. **Test completo en GPU** (18,864 funciones, batch 32): tokenización 13 s,
   inferencia ~15 min, GPU al 98% con solo 2.3 GB / 8 GB de VRAM usados.

   ```
   ***** Test results *****
     test_accuracy  = 0.9909
     test_f1        = 0.9142
     test_precision = 0.9712
     test_recall    = 0.8635
   ```

   **Comparación con el paper (RQ1):**

   | Métrica | Paper | Replicado aquí |
   |---|---|---|
   | F1 | 0.91 | 0.9142 ✅ |
   | Precision | 0.97 | 0.9712 ✅ |
   | Recall | 0.86 | 0.8635 ✅ |

   Replicación exacta. Confirma que cambiar torch 1.13 → 2.7.1 (necesario
   para Blackwell) no altera los resultados: los pesos del modelo y el
   tokenizador (transformers/tokenizers pinneados) definen la predicción;
   torch solo es el motor de ejecución.

   Nota: con VRAM tan holgada (2.3 GB usados), `--eval_batch_size 64` es
   viable y acortaría la inferencia.

   El log completo de esta corrida quedó en `resultados_gpu_test.log` (raíz
   del proyecto). Para futuras corridas conviene capturar el log directo al
   proyecto con `... 2>&1 | tee resultados_FECHA.log` — con `--rm` el
   contenedor se borra al terminar y `docker logs` deja de estar disponible.

8. **Re-corrida del test completo (2026-09-30)**, con el mismo comando de la
   sección 10, lanzado desde `LineVul-main/`. Resultado igual al del punto 7
   hasta el cuarto decimal:

   ```
   ***** Test results *****
     test_accuracy = 0.9909
     test_f1 = 0.9142
     test_precision = 0.9712
     test_recall = 0.8635
     test_threshold = 0.5
   ```

   La inferencia fue de 20:18:48 a 20:33:44 en el log del contenedor, que
   escribe en UTC (el host está en UTC-5, o sea 15:18 a 15:33 hora local):
   14 min 56 s, la misma duración que en julio (03:43:32 a 03:58:28).
   Evidencia: captura `evidenciasLineVul/testPruebaLineVul.png`. Esta vez
   no se usó `tee`, así que no hay `.log` nuevo; la salida del test original
   sigue en `resultados_gpu_test.log`. La corrida también regeneró
   `linevul/test_precision_recall_12heads_linevul_model.bin.pdf` (curva
   precisión-recall).

---

## 12. Solución de problemas

| Síntoma | Causa | Solución |
|---|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | Falta nvidia-container-toolkit | Sección 8 |
| `CUDA error: no kernel image is available` | torch sin kernels sm_120 | Usar la imagen `linevul:cu128` (torch 2.7.1+cu128) |
| `CUDA out of memory` | Batch demasiado grande para 8 GB | Bajar `--eval_batch_size` (32 → 16 → 8) |
| `gdown` falla con "Too many users..." | Cuota de Google Drive | Esperar o descargar manual desde navegador |
| `FileNotFoundError: .../checkpoint-best-f1/...` | Modelo en ruta incorrecta | El `.bin` debe estar en `linevul/saved_models/checkpoint-best-f1/` |
| `UnpicklingError` al cargar `.bin` | Default `weights_only=True` de torch ≥ 2.6 | Sección 7 |
| No se imprime nada durante ~15 min tras `Batch size = 32` | Es normal: la fase de test no muestra progreso | Mirar `nvidia-smi` (GPU ~100 %) o `docker ps`; no interrumpir |
