# LineVul en Docker con GPU RTX 5060 — Documentación paso a paso

Guía de todo lo que se hizo para poder probar el modelo LineVul en esta máquina,
con la explicación del **porqué** de cada decisión.

Última actualización: 2026-10-28 (entrenamiento de LineVul sobre FFmpeg+QEMU,
sección 13; re-corrida del test completo de Big-Vul el 2026-09-30, sección 11).

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

9. **Entrenamiento desde `codebert-base` sobre FFmpeg+QEMU (2026-10-27/28)**:
   terminado, con resultado propio. Exactitud 0.6354, precisión 0.5932, recall
   0.6363, F1 0.614 en un test de 2,721 funciones. Detalle, tablas y evidencia
   en la sección 13.6.

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
| `KeyError: 'flaw_line_index'` al final del test | El CSV de test no trae las columnas de líneas vulnerables (FFmpeg+QEMU no las tiene) | Sección 13.3 |
| `CUDA out of memory` al **entrenar** con `--train_batch_size 16` | 8 GB no alcanzan para batch 16 con 512 tokens | Sección 13.2: batch 4 con `--gradient_accumulation_steps 4` |
| La barra de entrenamiento muestra una pérdida de ~0.17 y parece demasiado baja | Con acumulación, el código divide la pérdida entre N antes de mostrarla (`linevul_main.py` líneas 181-182) | La pérdida real es lo que muestra × `--gradient_accumulation_steps` (0.169 × 4 = 0.675) |
| Se renombró mal o se pisó `results/raw_preds.csv` | El programa siempre escribe con ese nombre (`write_raw_preds_csv`, línea 587) | Renombrarlo al terminar cada corrida (ver 13.6) |

---

## 13. Entrenar LineVul desde `codebert-base` sobre FFmpeg+QEMU (2026-10-25 a 2026-10-28)

Hasta aquí LineVul solo se evaluó con el checkpoint de los autores (sobre
Big-Vul). Para compararlo con AMPLE en igualdad de condiciones, se entrena
desde `microsoft/codebert-base` con el mismo reparto 80/10/10 que AMPLE. Esta
sección documenta la preparación (13.1 a 13.5), la corrida terminada el
2026-10-28 con sus resultados (13.6) y unas mediciones complementarias (13.7).
"Desde cero" significa desde los pesos preentrenados de CodeBERT; la cabeza de
clasificación (592,130 parámetros) sí parte de pesos aleatorios.

### 13.1. Primera modificación al código de LineVul (fix #1)

Archivo: `linevul/linevul_main.py`, función `train()`. Hecha por el alumno el
2026-10-25; el original quedó comentado en el código.

| Línea (original → actual) | Antes | Después |
|---|---|---|
| 121 → 126 | `args.max_steps = args.epochs * len(train_dataloader)` | `args.max_steps = args.epochs * (len(train_dataloader) // args.gradient_accumulation_steps)` |
| 123 → 136 | `args.save_steps = len(train_dataloader)` | `args.save_steps = len(train_dataloader) // args.gradient_accumulation_steps` |

**Razón.** Con `--gradient_accumulation_steps N > 1`, los pesos se actualizan
una vez cada N micro-lotes y `global_step` cuenta esas actualizaciones. Pero
`max_steps` y `save_steps` se calculaban con el número de micro-lotes, N veces
mayor. Consecuencias del código original con N = 4: (a) la validación y el
guardado del mejor checkpoint se disparaban cada N épocas en vez de cada una
(con 10 épocas, solo en la 4.ª y la 8.ª); (b) el calendario de la tasa de
aprendizaje (`warmup_steps = max_steps // 5`, decaimiento lineal hasta
`max_steps`) se planeaba para N veces más pasos de los que se dan: el
calentamiento duraba 8 épocas en vez de 2 y la tasa terminaba en ~94 % del
máximo sin decaer (cálculo sobre el código, no una corrida larga).
`warmup_steps` no se tocó: depende de `max_steps`. Con N = 1 (el uso del
paper) el cambio no altera nada.

**Verificación (corrida corta real, sin log archivado):** 320 funciones, 2
épocas, batch 4, acumulación 4. Con el código original, 1 época mostraba
`Total optimization steps = 80` y **ninguna** validación ni checkpoint. Con el
fix: `Total optimization steps = 40` (2 épocas × 20) y una validación
(`eval_f1`) al final de cada época, con `Best f1` y `Saving model checkpoint`
cuando mejora. Los F1 de esa prueba (0.56 y 0.41 con 64 funciones de valid)
no significan nada: solo prueban el mecanismo.

### 13.2. Memoria y velocidad en la RTX 5060 (8 GB)

Prueba corta de 1 época sobre 320 funciones de FFmpeg+QEMU, imagen
`linevul:cu128`, fp32 (el código no usa precisión mixta), `--block_size 512`.
Pico de VRAM medido con `nvidia-smi` cada 1 s. **(sin log archivado)**

| `--train_batch_size` | `--gradient_accumulation_steps` | Resultado |
|---|---|---|
| 16 | 1 | CUDA out of memory |
| 8 | 2 | CUDA out of memory |
| 4 | 4 (lote efectivo 16, como el paper) | Funciona: pico 5,459 MiB; 80 micro-lotes en ~27.7 s (2.89 it/s ≈ 11.6 funciones/s) |

ESTIMACIÓN hecha entonces (no medida a escala): con 21,758 funciones de train a
11.6 funciones/s, una época son ~31 min y 10 épocas ~5.2 h, más la validación de
cada época. **Medido después en la corrida real: ver 13.6** (31 min 13 s de
entrenamiento y 2 min 9 s de validación por época; 5 h 37 min en total).

### 13.3. El test de LineVul exige columnas de líneas vulnerables

Al terminar, `test()` llama a `generate_result_df()`, que lee `flaw_line_index`
y `flaw_line` del CSV de test. FFmpeg+QEMU solo tiene etiqueta por función.
Prueba con un test de juguete (64 funciones), entrenando 1 época y luego
`--do_test --write_raw_preds` **(sin log archivado)**:

| CSV de test | Resultado |
|---|---|
| Sin las dos columnas | `KeyError: 'flaw_line_index'`: las métricas se calculan pero el programa muere antes de escribir las predicciones |
| Con `flaw_line` y `flaw_line_index` **vacías** | Termina bien: `test_accuracy`, `test_f1` y `raw_preds.csv` |

Solución sin tocar el código: `protocolo_comparacion/preparar_test_linevul.py`
copia `test.csv` agregando esas dos columnas vacías. Resultado verificado:
`conjuntos_de_datos/ffmpeg_qemu/split_semilla42/test_linevul.csv`, 2,721 filas,
mismo orden y contenido que `test.csv` en las 5 columnas originales, 1,240
vulnerables. El `test.csv` original no se modifica (lo usa AMPLE). Las
predicciones por función (`raw_preds`) se alinean con las de AMPLE por `id`.
`train.csv` y `val.csv` no necesitan columnas extra (LineVul solo lee
`processed_func` y `target`).

### 13.4. Datos y reparto

Fuente: `conjuntos_de_datos/ffmpeg_qemu/function.json` (Devign; 27,318
funciones; enlace verificado en el repositorio oficial de AMPLE y en la página
de Devign). Reparto común a LineVul y AMPLE, generado por
`protocolo_comparacion/dividir_ffmpeg_qemu.py` (semilla 42, estratificado por
proyecto y etiqueta; descarta las 120 filas cuyo texto aparece con etiquetas
contradictorias): train 21,758 / val 2,719 / test 2,721 funciones. Detalle y
resultados en `../PROGRESS.md`.

### 13.5. Comando y lanzamiento

Script: `protocolo_comparacion/entrenar_linevul_ffmpeg_qemu.sh`. Parte del
comando del README de LineVul (líneas 222-241, "To retrain the RQ1 model") y
cambia solo lo necesario:

| Parámetro | README (paper) | Nuestro script | Motivo |
|---|---|---|---|
| `--train_batch_size` | 16 | 4 | Con 16 y con 8 la GPU se queda sin memoria (13.2) |
| `--gradient_accumulation_steps` | (no se pasa, vale 1) | 4 | Lote efectivo 16, como el paper; requiere el fix #1 (13.1) |
| `--train/eval/test_data_file` | `../data/big-vul_dataset/{train,val,test}.csv` | `/datos/ffmpeg_qemu/split_semilla42/{train,val,test_linevul}.csv` | Nuestro reparto; el test lleva las 2 columnas vacías (13.3) |
| `--output_dir`, `--model_name` | `./saved_models`, `model.bin` | `./saved_models_ffmpeg_qemu`, `linevul_ffmpeg_qemu.bin` | No mezclar con el checkpoint de los autores |
| `--write_raw_preds` | no | sí | Predicciones por función, para alinear con AMPLE |

Igual que el paper: `--epochs 10`, `--block_size 512`, `--eval_batch_size 16`,
`--learning_rate 2e-5`, `--max_grad_norm 1.0`, `--evaluate_during_training`,
`--seed 123456`, tokenizador y modelo `microsoft/codebert-base`.

El script monta `conjuntos_de_datos/` en solo lectura como `/datos` con un
segundo `-v` (el contenedor solo ve `LineVul-main/` y un enlace simbólico hacia
afuera no funciona), usa `--gpus all` (la GPU sí se usa dentro de Docker; la
imagen solo aporta torch 2.7.1+cu128, necesario para Blackwell) y guarda la
salida en `LineVul-main/entrenamiento_ffmpeg_qemu_<fecha>.log`. Antes de empezar
comprueba que existan los tres CSV, que estén las dos líneas del fix #1 y que la
GPU tenga menos de 1.5 GB en uso.

Lanzamiento (2026-10-27, 19:59 hora local), desde la raíz del repositorio y
fuera de VS Code:

```bash
setsid nohup bash protocolo_comparacion/entrenar_linevul_ffmpeg_qemu.sh > /tmp/lanzador_linevul.txt 2>&1 &
```

Con `setsid` el proceso queda en su propia sesión, sin terminal y con systemd
como padre; el contenedor lo gestiona dockerd. No hay reanudación si se corta.

### 13.6. Resultados de la corrida (terminada el 2026-10-28)

Evidencia: `entrenamiento_ffmpeg_qemu_2026-10-27_1959.log` (9.2 MB; el log del
contenedor va en UTC, 5 h adelantado), `linevul/results/raw_preds_ffmpeg_qemu.csv`
y el checkpoint `linevul/saved_models_ffmpeg_qemu/checkpoint-best-f1/linevul_ffmpeg_qemu.bin`
(997,357,647 bytes).

**Cabecera del log (coincide con lo calculado):** `Num examples = 21758`,
`Num Epochs = 10`, `Total train batch size = 16`, `Gradient Accumulation
steps = 4`, `Total optimization steps = 13600` (10 × 1,360). Con el código
original sin el fix #1 esta última cifra y el calendario habrían sido otros
(13.1).

**Tiempos (log).** Inicio 00:59:17 UTC y resultados del test 06:36:16 UTC: 5 h
37 min (19:59 a 01:36 hora local). Cada época: 31 min 13 s de entrenamiento más
2 min 9 s de validación (la barra cierra en 33:22 a 33:27 con la validación
incluida). VRAM pico observada 5,867 MiB (con `nvidia-smi`, **sin log
archivado**).

**Validación por época** (F1, precisión y recall con umbral 0.5; la pérdida real
es la registrada en la barra × 4, porque el código la divide entre la
acumulación):

| Época | F1 val | Precisión val | Recall val | Pérdida real de entrenamiento |
|---|---|---|---|---|
| 1 | 0.6168 | 0.5065 | 0.7887 | 0.675 |
| 2 | 0.3702 | 0.7798 | 0.2427 | 0.641 |
| 3 | 0.5631 | 0.6656 | 0.4879 | 0.614 |
| **4** | **0.6250** | 0.6019 | 0.6500 | 0.581 |
| 5 | 0.5610 | 0.6849 | 0.4750 | 0.542 |
| 6 | 0.5461 | 0.7082 | 0.4444 | 0.514 |
| 7 | 0.5806 | 0.6688 | 0.5129 | 0.481 |
| 8 | 0.5635 | 0.6730 | 0.4847 | 0.460 |
| 9 | 0.5861 | 0.6646 | 0.5242 | 0.431 |
| 10 | 0.5806 | 0.6595 | 0.5185 | 0.411 |

El mejor F1 de validación fue el de la época 4 (0.625); el checkpoint se guardó
a las 03:13:27 UTC (22:13 hora local) y es el que se usó en el test. Solo se
guarda el mejor por F1 (`linevul_main.py` línea 208), así que no se pueden
probar las demás épocas sin repetir el entrenamiento. La tasa de aprendizaje
alcanza su máximo (2e-5) tras 2,720 actualizaciones, al final de la época 2
(cálculo sobre el código).

**Test** (2,721 funciones, 1,240 vulnerables = 45.57 %, checkpoint de la época 4,
umbral 0.5), líneas `Test results` del log:

```
test_accuracy  = 0.6354
test_f1        = 0.614
test_precision = 0.5932
test_recall    = 0.6363
test_threshold = 0.5
```

Recalculado desde `raw_preds_ffmpeg_qemu.csv` (las 2,721 filas, `id` alineados
con `split_semilla42/test.csv`, mismas etiquetas): TP 789, FP 541, FN 451, TN
940, que reproducen las cuatro métricas del log. Predijo "vulnerable" en 1,330
funciones (48.9 %). Exactitud por proyecto: FFmpeg 0.6047 (969 funciones) y
QEMU 0.6524 (1,752).

**Referencias por cálculo** (no son resultados del modelo): con 45.57 % de
vulnerables, un predictor que dijera siempre "vulnerable" tendría F1 0.626 y
exactitud 0.456; uno que dijera siempre "no vulnerable", exactitud 0.544. El F1
de 0.614 queda a la altura del primero; la exactitud (0.635) y la precisión
(0.593 frente a una base de 0.456) sí muestran una señal por encima del azar,
aunque modesta.

**Frente al log de entrenamiento de los autores** (Big-Vul, 150,908 funciones de
train y 18,864 de validación; `linevul/train_logs/linvul_train.log`), mismo
código e hiperparámetros: F1 de validación por época 0.878, 0.888, 0.906, 0.915,
0.912, **0.917**, 0.916, 0.913, 0.912, 0.913 (test final 0.9142). La diferencia
con la nuestra (0.37 a 0.625) no se puede atribuir a una sola causa: cambian el
tamaño de entrenamiento (150,908 frente a 21,758), la proporción de vulnerables
(5.8 % frente a 45.6 %) y la longitud de las funciones (13.7).

**Cautelas.** Es una sola corrida con una semilla (123456); no se sabe cuánto
varía. No es comparable con el F1 de 56.54 % que el paper de AMPLE reporta para
LineVul en FFmpeg+Qemu: lo obtuvieron los autores de AMPLE con su propio
pipeline y 22,361 funciones (las nuestras son 27,198). La pérdida de
entrenamiento bajó de 0.675 a 0.411 mientras el F1 de validación no mejoró desde
la época 4: posible sobreajuste (hipótesis; no se midió la pérdida de
validación). Además, la acumulación de gradiente no es idéntica a un lote de 16
directo, porque el código recorta el gradiente en cada micro-lote
(`linevul_main.py` línea 185); no se midió su efecto.

**Archivos generados:** el log; el checkpoint (ignorado por git, `*.bin`);
`raw_preds_ffmpeg_qemu.csv` (el programa lo escribe como `results/raw_preds.csv`
y se renombró el 2026-10-28; ignorado por git, `*.csv`); las curvas
`linevul/eval_precision_recall_linevul_ffmpeg_qemu.bin.pdf` y
`linevul/test_precision_recall_linevul_ffmpeg_qemu.bin.pdf` (ignoradas por git);
y la captura `evidenciasLineVul/evidenciaEntrenamiento-epoch0.png`.

### 13.7. Mediciones complementarias (2026-10-28, **sin log archivado**)

Hechas con comandos sueltos en el contenedor `linevul:cu128`, sin guardar el
script ni su salida: sirven para orientar, **no citarlas en el artículo** hasta
archivarlas (pendiente, ver 13.8).

**Límite de la ventana.** El `config.json` de CodeBERT dice
`max_position_embeddings = 514` (tabla de posiciones de 514 × 768). Prueba en CPU
con `RobertaModel.from_pretrained('microsoft/codebert-base')`: 512 tokens
funciona, 513 da `IndexError: index out of range in self` y 600 da
`RuntimeError`. El límite es del modelo, no un parámetro de LineVul: `--block_size`
(`linevul_main.py` línea 101, `[:args.block_size-2]`) solo permite bajarlo.

**Longitud de las funciones** (tokenizador de `codebert-base` sobre
`processed_func`; se cortan las de más de 510 tokens):

| Conjunto | Funciones | Mediana de tokens | Más de 510 tokens | AUC usando solo la longitud |
|---|---|---|---|---|
| FFmpeg+QEMU, train | 21,758 | 480 | 48.0 % | no medido |
| FFmpeg+QEMU, val | 2,719 | 494 | 49.2 % | no medido |
| FFmpeg+QEMU, test | 2,721 | 484 | 48.5 % | 0.537 |
| Big-Vul, test | 18,864 | 167 | 18.7 % | 0.689 |

Por clase (test): en Big-Vul las vulnerables tienen mediana de 390 tokens y 40.4 %
de cortadas, frente a 160 y 17.4 % en las no vulnerables; en FFmpeg+QEMU, 534 y
51.5 % frente a 444 y 45.9 %. Las funciones largas pierden su parte final, y que
eso explique parte del rendimiento menor en FFmpeg+QEMU es una **hipótesis sin
probar**.

### 13.8. Pendientes de esta sección

- Archivar el script y la salida de las mediciones de 13.7 en
  `protocolo_comparacion/`.
- Medir el rendimiento de este checkpoint por separado en las funciones que caben
  (≤510 tokens) y en las cortadas (>510), con `raw_preds_ffmpeg_qemu.csv` y la
  cuenta de tokens, para contrastar la hipótesis del corte.
- Entrenar y probar AMPLE sobre este mismo reparto para tener la comparación.
- Material de estudio del código: `GUIA_ESTUDIO_CODIGO_LINEVUL.md`.
