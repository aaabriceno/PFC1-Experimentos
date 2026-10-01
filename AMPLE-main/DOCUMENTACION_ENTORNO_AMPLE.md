# AMPLE en Docker con GPU RTX 5060 — Documentación paso a paso

Guía de todo lo que se hizo para poder probar el modelo AMPLE en esta máquina,
con la explicación del **porqué** de cada decisión. Mismo espíritu que
[`LineVul-main/DOCUMENTACION_ENTORNO_LINEVUL.md`](../LineVul-main/DOCUMENTACION_ENTORNO_LINEVUL.md),
para que ambos modelos queden documentados con el mismo nivel de detalle.

Última actualización: 2026-10-01. El estado actual (qué está hecho y qué falta)
y los tiempos medidos están en [`../PROGRESS.md`](../PROGRESS.md).

---

## 1. ¿Qué es AMPLE y qué se necesita para probarlo?

AMPLE es un modelo GNN (gr**A**ph si**M**plification and enhanced Graph
re**P**resentation **LE**arning) que detecta vulnerabilidades en código
C/C++ representando cada función como un grafo (AST + CFG + DFG) en vez de
una secuencia de texto. A diferencia de LineVul, **no publica un checkpoint
pre-entrenado** — hay que entrenarlo desde cero sobre el dataset elegido.
Para correrlo hacen falta 4 piezas, ninguna lista para usar directamente:

| Pieza | Qué es | Estado |
|---|---|---|
| Joern | Extrae AST/CFG/DFG de cada función `.c` | Binario precompilado vía Zenodo, ya extraído. Corre en el host, no en Docker: la imagen `ample-env` no trae Java |
| word2vec | Embeddings iniciales de los nodos del grafo | Descargado de Zenodo, 3 variantes pre-entrenadas; en la práctica se entrena uno propio por corrida (§8) |
| Extractor de subset | AMPLE no trae uno, solo espera `.c` ya separados | Escrito a medida: `extract_subset.py` (subset aleatorio) y `extract_from_linevul_split.py` (usa el test exacto de LineVul, §6.1) |
| Dependencias Python | torch, dgl, gensim, etc. | Resuelto en imagen Docker `ample-env` (ver §3-4) |

El dataset base es Fan et al. 2020 (Big-Vul), el mismo que usa LineVul —
descargado en formato crudo (`MSR_data_cleaned.csv`, 10.7GB) en
`docker-labs/vulnerability-detection/shared_datasets/fan_et_al/`.

---

## 2. Diagnóstico de esta máquina

Mismo hardware que la corrida de LineVul (ver su documentación, §2):
Python 3.12.3 en el host, RTX 5060 Laptop 8GB (Blackwell, **sm_120**),
driver NVIDIA 595.x (CUDA 13.2), Docker 29.1.3.

---

## 3. El problema central: dgl sin kernels sm_120 (peor que el caso de LineVul)

LineVul se resolvió actualizando a `torch>=2.7+cu128`, que sí trae kernels
Blackwell. Con AMPLE **no hay salida equivalente**, porque el problema no es
solo de torch sino de la compatibilidad ABI entre torch y dgl:

- `dgl==2.1.0` (paquete genérico de PyPI) + cualquier torch con soporte
  sm_120 (≥2.4): **crash al hacer `dlopen` del `.so` nativo de dgl** al
  importar. El mensaje de error es engañoso (`"getCount is
  non-monotonic"`, parece un bug de reloj del sistema) pero la causa real
  es incompatibilidad binaria de ABI entre esa build de dgl y ese torch.
- `dgl==2.4.0+cu124` (del índice propio de `data.dgl.ai`, no está en PyPI)
  + `torch==2.4.0+cu121`: importa limpio y hasta crea grafos en GPU sin
  error — pero **cualquier operación de cómputo real**
  (`dgl.nn.GraphConv.forward`, hasta un simple `in_degrees()`/`arange()`
  interno) falla con `CUDA error: no kernel image is available for
  execution on the device`. Confirma que torch 2.4.x de verdad no trae
  kernels sm_120; el import limpio no era un falso positivo.
- No existe ninguna build de dgl más nueva que 2.4.0 con wheel
  Linux/cu12x en ningún índice (ni PyPI, ni conda-forge/dglteam, ni
  GitHub releases). `data.dgl.ai` (bucket S3/CloudFront) da **403 Access
  Denied** en `2.5.0` con cualquier user-agent probado; `2.4.0` sí
  responde 200. Causa del bloqueo en `2.5.0` no investigada — el
  proyecto dgl lleva desde 2024 prácticamente sin actividad.
- `torch>=2.7+cu128` (el que SÍ soporta Blackwell y se usa en LineVul) no
  tiene ninguna build de dgl compatible en ABI — dgl.ai no publicó nada
  más nuevo que 2.4.0 con wheels Linux/cu12x, y ese está atado a torch
  2.4.x.

**Conclusión confirmada empíricamente (7 ago 2026): AMPLE corre en CPU en
esta máquina con las builds pre-compiladas de dgl (PyPI / data.dgl.ai). La
GPU queda bloqueada mientras el proyecto dgl no publique un build
pre-compilado contra torch ≥2.7.** No es una limitación de configuración
del entorno, es una limitación real de lo que dgl distribuye — pero **sí
existe una vía alternativa: compilar dgl desde source**, ver §3.1 (23 sep
2026).

**Solución que funciona en CPU (sin compilar nada)**: `torch==2.2.1+cpu` +
`dgl==2.1.0`. Ojo: dgl 2.1.0 tampoco acepta cualquier torch en CPU — su
módulo `graphbolt` busca en runtime un `.so` nombrado
`libgraphbolt_pytorch_<versión exacta>.so`, y el wheel solo trae los de
torch 2.0.0/2.0.1/2.1.0/2.1.1/2.1.2/2.2.0/2.2.1 — con torch 2.5.1 tira
`FileNotFoundError` buscando un `.so` que no existe. 2.2.1 es la más nueva
de las soportadas. Verificado con un forward pass real
(`dgl.nn.GraphConv` sobre un grafo de prueba), no solo con un import limpio.
Este sigue siendo el entorno usado en `ample-env` (Docker, §4) para todo el
pipeline de datos y el entrenamiento en CPU documentado en §10.

### 3.1. GPU sí es viable compilando dgl desde source (23 sep 2026)

Corrección al punto anterior: el 23 sep 2026 se investigó si existía algún
workaround para GPU, motivado por una pregunta directa del alumno sobre por
qué no se podía usar GPU. Se encontró
[dmlc/dgl#7917](https://github.com/dmlc/dgl/issues/7917) — un usuario
reportó haber compilado dgl desde source contra una RTX 5070 Ti (mismo
Blackwell/sm_120) con torch 2.10 + CUDA 13. Se replicó el proceso en esta
máquina con **éxito total, incluyendo forward pass real en GPU**, no solo
import limpio (que es donde había fallado el intento de agosto con
`dgl==2.4.0+cu124`).

**Chequeo de seguridad previo** (regla del proyecto: código externo no
confiable por defecto): se revisaron los 2 patches del issue línea por
línea (solo cambios de configuración CMake, sin código ejecutable
inyectado ni descargas ocultas) y el repo `dmlc/dgl` (14.3k★, org legítima,
submódulos apuntando a fuentes conocidas — google, NVIDIA, KarypisLab,
axboe). Sin hallazgos.

**Entorno usado**: venv nuevo y aislado
`dgl_build_experimental/../.venv-ample-gpu-experimental` (no
`.venv-ample`, que sigue obsoleto; no se tocó `ample-env` ni el resto del
entorno CPU que ya funciona) con `torch==2.11.0+cu128` (el que sí trae
kernels sm_120). Fuente de dgl clonada en `dgl_build_experimental/dgl/`,
commit base `3d16000b4170` (2025-08-01), que es el HEAD de `master`
(verificado el 2026-09-30 con un clon limpio).

**Pasos seguidos (basados en el issue, con fixes propios donde el proceso
del issue no alcanzaba)**:

1. `git clone --recurse-submodules https://github.com/dmlc/dgl.git`.
2. Aplicados los 2 patches del issue a `CMakeLists.txt` y
   `cmake/modules/CUDA.cmake`: cambian METIS/GKlib a libs del sistema en
   vez de las embebidas, hacen `HugeCTR` opcional vía flag, fuerzan
   `CUDA_ARCH_BIN=120`/`CUDA_ARCH_PTX=120` en vez de auto-detección.
3. Deshabilitados (renombrados a `.disabled`) `metis_partition.cc` y
   `metis_partition_hetero.cc` como indica el issue.
4. GKlib y METIS compilados aparte con prefix local
   (`dgl_build_experimental/local_install/`, no `/usr/local` — no hace
   falta sudo).
5. `cmake .. -DUSE_CUDA=ON -DUSE_OPENMP=ON -DBUILD_TORCH=ON -DUSE_METIS=ON
   -DBUILD_HUGECTR=ON -DCMAKE_BUILD_TYPE=Release
   -DCMAKE_PREFIX_PATH=<local_install>` (nota: `BUILD_HUGECTR=ON`, no
   `OFF` como en el primer intento — ver fix 3 abajo).
6. `make graphbolt tensoradapter_pytorch -j1` y luego `make dgl -j1` (ver
   fix 1 abajo, sobre por qué no más paralelismo). Un `make -j1` completo con
   `BUILD_HUGECTR=OFF` dejó `libdgl.so` con símbolos sin resolver (fix 3).
   Los cambios quedaron como patch en `patches/dgl/` del repositorio de
   evidencia; se verificó con `git apply --check` que aplica limpio sobre un
   clon nuevo del commit base.

**3 bugs reales de dgl encontrados y arreglados en el proceso** (no
inventados por el fix del issue, confirmados leyendo el código):

1. **Consumo de RAM sin límite en `graphbolt/build.sh`**: el script
   (invocado como proceso externo por `add_custom_target(graphbolt ...)`
   en el `CMakeLists.txt` padre) hace `make -j` sin número — usa todos los
   cores sin límite de RAM por job, y no hereda el `-j` del make padre por
   ser un proceso `bash` separado. En esta máquina (32 cores, 30GB RAM)
   varias unidades de compilación CUDA/CCCL pesadas en paralelo agotaron
   la RAM y dispararon el OOM killer del sistema dos veces durante el
   intento (mató un proceso no relacionado — `brave` — por tener mayor
   RSS, lo que además cerró el editor). Fix: variable de entorno nueva
   `GRAPHBOLT_MAKE_JOBS` en `graphbolt/build.sh` (default vacío =
   comportamiento original) que se pasa a `make -j $MAKE_JOBS`.
   Usada como `export GRAPHBOLT_MAKE_JOBS=1` en esta máquina.
2. **Regex de filtro de arquitectura CUDA roto para arquitecturas de 3
   dígitos** (`graphbolt/CMakeLists.txt`): `list(FILTER
   CMAKE_CUDA_ARCHITECTURES_FILTERED EXCLUDE REGEX "[2-6][0-9]")` hace
   match parcial (no full-match) en CMake — con `CUDAARCHS=120`
   (Blackwell), el regex matcheaba la substring `"20"` dentro de `"120"`
   y lo trataba como si fuera sm_20-sm_69, dejando el filtro vacío y
   cayendo al fallback hardcoded `"70"` (Volta), que el `nvcc` de CUDA
   13.2 ni siquiera soporta (`nvcc fatal: Unsupported gpu architecture
   'compute_70'`). Fix: ancla el regex a `"^[2-6][0-9]$"` (full-match).
3. **Include path de `nv_gpu_cache.hpp` amarrado incorrectamente a
   `BUILD_HUGECTR`** (`CMakeLists.txt` principal, línea ~388): el archivo
   `src/runtime/cuda/gpu_cache.cu` (parte de `DGL_SRC`, se compila
   SIEMPRE, sin relación con `BUILD_HUGECTR`) incluye
   `nv_gpu_cache.hpp`, pero el patch del issue solo agregaba ese include
   dir al target `dgl` dentro del bloque `if(USE_CUDA AND BUILD_HUGECTR)`.
   Con `BUILD_HUGECTR=OFF` (la config usada primero, siguiendo el
   ejemplo del issue) esto compilaba pero fallaba en el LINK: `libdgl.so`
   quedaba con símbolos `gpu_cache::gpu_cache<...>` sin resolver (la
   implementación vive en `nv_gpu_cache.cu`, que solo se compila con
   `BUILD_HUGECTR=ON`). Import de dgl funcionaba igual (el símbolo solo
   se resuelve en runtime al llamar esa función), pero cualquier intento
   de cargar el `.so` fallaba con `OSError: undefined symbol`. Dos vías
   posibles: (a) mover el include dir fuera del `if BUILD_HUGECTR` para
   que solo declare el include sin la implementación, o (b) compilar
   también `nv_gpu_cache.cu` con `BUILD_HUGECTR=ON`. Se usó (b) — más
   simple y sin riesgo, la clase gpu_cache es funcionalidad real de dgl
   (cache de embeddings en GPU), no algo a descartar.

**Verificación real (no solo import)**: forward pass completo con
`dgl.nn.GraphConv` sobre un grafo movido a `cuda:0`, incluyendo
`in_degrees()` (operación de cómputo GPU real) y `torch.cuda.synchronize()`
sin error — el mismo tipo de prueba donde el intento de agosto con
`dgl==2.4.0+cu124` fallaba (`CUDA error: no kernel image is available`).

```
torch: 2.11.0+cu128 cuda available: True
dgl backend: pytorch
grafo en GPU: cuda:0
in_degrees (GPU compute real): tensor([1, 1, 1, 1], device='cuda:0')
GraphConv forward pass en GPU OK, output shape: torch.Size([4, 4])
output device: cuda:0
```

**Gotcha de path**: correr el script de prueba desde dentro de
`dgl_build_experimental/` resuelve `import dgl` a un namespace package
vacío (la carpeta del repo clonado sombrea el paquete instalado, porque
Python agrega el cwd al `sys.path`) — correr desde otro directorio (o
cualquier proyecto real) para que resuelva al paquete instalado de verdad.

**Estado**: entorno experimental, aislado, no reemplaza `ample-env` (CPU).
Ya se usó para entrenar AMPLE en GPU (§10, entradas del 23 y 29 sep): con
este venv `torch.cuda.is_available()` es `True` y `main.py` usa la GPU sin
flags extra, gracias a los fixes de §9. VRAM pico medida: 687 MiB de 8,151 MiB
(sin log archivado). El venv depende de `dgl_build_experimental/`
(instalación editable): borrar esa carpeta lo rompe.

---

## 4. Diseño del Dockerfile — decisiones y porqués

Archivo: [`Dockerfile`](Dockerfile). Mismo patrón que LineVul: la imagen trae
solo el entorno, el código/dataset/grafos se montan del host con `-v`.

- **Capa 1 (torch + dgl)**: `torch==2.2.1` build CPU +
  `dgl==2.1.0` + dependencias transitivas que dgl no declara bien en su
  metadata y hay que instalar a mano: `packaging`, `torchdata==0.7.1`
  (versión vieja específica — 0.11.0 no trae el módulo `.datapipes` que dgl
  usa), `pyyaml`, `pydantic`.
- **Paquetes de sistema**: `build-essential` + `graphviz` vía `apt`.
  `gensim==4.3.2` no publica wheel precompilado para py3.12/linux → pip
  compila `word2vec_inner.c` desde source, necesita `gcc`/`libc6-dev`
  (no alcanza con `gcc` suelto). El paquete Python `graphviz` además
  necesita el binario `dot` del sistema aparte del paquete pip.
- **Capa 2 (resto de dependencias)**: `numpy==1.26.4`, `pandas==2.2.2`,
  `scikit-learn==1.5.0`, `scipy==1.12.0` (fijado — versiones más nuevas
  quitaron `triu` de `scipy.linalg` y rompen `gensim`), `gensim==4.3.2`,
  `graphviz==0.20.3`, `nltk==3.8.1`. Estas tres últimas **no están en el
  README de AMPLE** pero el código real las importa (`utils.py`,
  `word2vec.py`) — confirmado leyendo `data_processing/*.py` y
  `AMPLE_code/*.py` directamente, no asumido.
- **Imagen final**: tageada `ample-env`, resuelta tras 9 iteraciones de
  build (mismo patrón `-v $(pwd):/app` que LineVul, código/datos quedan en
  el host).

---

## 5. Dataset: mismo origen que LineVul, formato distinto (resuelto el 23 sep 2026)

`shared_datasets/fan_et_al/MSR_data_cleaned.csv` (10.7GB) es el dataset
crudo completo de Fan et al./Big-Vul, columnas `func_before`, `func_after`,
`vul` (label) más metadata (CVE, CWE, etc). **Es el mismo dataset fuente
que usa LineVul**, pero LineVul lo usa ya separado en
`train.csv`/`val.csv`/`test.csv` (981MB de test, 18,864 funciones), mientras
que aquí se parte del CSV crudo sin splits predefinidos.

Para que el número de AMPLE sea comparable función por función con el de
LineVul, el test de AMPLE tiene que contener las mismas funciones que el
`test.csv` de LineVul, no solo salir del mismo CSV de origen. **Resuelto**:
`extract_from_linevul_split.py` (§6.1) recupera exactamente las 18,864
funciones de ese `test.csv`, uniéndolas por la columna `index`. Verificado: la
fila con `index=73752` de `test.csv` tiene el mismo `func_before` y la misma
etiqueta que la fila 73752 de `MSR_data_cleaned.csv`. Se recuperaron 18,864
de 18,864.

**Limitación nueva**: el filtro de `cpg_GS` (§9.1) descarta los grafos de más
de 500 nodos, así que AMPLE no se evalúa sobre las 18,864 funciones sino sobre
14,772, y el descarte no es parejo entre clases. Detalle en §10 (entrada del
29 sep). Hasta decidir cómo tratarlo, el resultado de AMPLE no es
directamente comparable con el de LineVul.

---

## 6. Extractor propio: `extract_subset.py`

AMPLE **no trae** un extractor de funciones individuales desde el CSV crudo
— solo espera `.c` ya separados más un JSON de splits. Por eso se escribió
`AMPLE-main/extract_subset.py` (script nuevo, no existía en el repo
original; consistente con la regla del proyecto de escribir solo el paso
faltante del pipeline, no reimplementar el modelo). Extrae un subset
aleatorio, separa en `.c` individuales
(`devign_dataset/devign_raw_code/func_<idx>_<label>.c`), genera splits
train/valid/test 70/15/15 en `devign_dataset/devign_data_split/*.json`.

**Bug encontrado y corregido (7 ago, tarde)**: la v1 tomaba una ventana
ciega de las primeras `n*20` filas del CSV antes de mezclar. El CSV
`MSR_data_cleaned.csv` no viene mezclado — las primeras ~10000+ filas son
todas `vul=0` — así que el subset de 500 inicial tenía **0 ejemplos
vulnerables**, y el modelo colapsó a predecir siempre una sola clase
(Accuracy 100% / Precision/Recall/F1 = 0.0, una señal de alarma, no un
resultado bueno). Fix: reescrito con **reservoir sampling estratificado por
clase**, streaming del CSV completo (10.7GB) una sola vez sin cargarlo
entero en memoria. También se subió `csv.field_size_limit` de 10MB a
`sys.maxsize` (10MB no alcanzaba para algún campo del CSV real, probablemente
`patch`).

**Lección para datasets futuros**: verificar la distribución de clases del
subset ANTES de correr el pipeline completo (Joern es el paso más lento),
no después de ver métricas sospechosas.

### 6.1. Extractor para el test exacto de LineVul: `extract_from_linevul_split.py`

Script nuevo (23 sep 2026) que se usa en lugar del anterior cuando se quiere la
comparación con LineVul. Cómo funciona:

- **Test**: lee `LineVul-main/data/big-vul_dataset/test.csv` (18,864 filas),
  toma su columna `index` (la posición de la fila en `MSR_data_cleaned.csv`) y
  recorre el CSV crudo una sola vez en streaming para recuperar el
  `func_before` y la etiqueta de esas filas.
- **Train y valid**: reservoir sampling balanceado por clase sobre las filas
  del CSV crudo que **no** están en el test de LineVul, para no contaminarlo.
- **Salida**: un `.c` por función (`<split>_<index>_<label>.c`) y los JSON de
  split, con el campo `code` incluido (lo exigen `word2vec.py` y `cpg_GS.py`).
- **Argumentos**: `--test-n` (0 = test completo; N > 0 = subset estratificado
  de N), `--train-n`, `--valid-n`, `--seed`, `--out-dir` y `--root` (raíz de
  `vulnerability-detection/`; vale `/vulndet` dentro del contenedor).

El `--root` se agregó porque el script resolvía rutas con `expanduser("~")`,
que dentro del contenedor es `/root` y no el home del host.

Comando de la corrida a escala completa (ver §12 para la secuencia completa):

```bash
docker run --rm --user $(id -u):$(id -g) -v "$(pwd)":/vulndet -w /vulndet/AMPLE-main ample-env \
  python3 extract_from_linevul_split.py --test-n 0 --train-n 13000 --valid-n 2000 \
  --out-dir devign_dataset_full
```

Resultado: test 18,864/18,864 recuperadas, train 13,000, valid 2,000. El
prefijo `devign_` del nombre de carpeta es heredado del código de AMPLE: el
contenido es Fan et al. / Big-Vul, no Devign.

---

## 7. Joern: extracción de grafos

Joern viene **precompilado** vía Zenodo (no hay que compilar nada):
`https://zenodo.org/api/records/7323504/files/joern.zip/content` (94.8MB,
verificado con md5), extraído en `AMPLE-main/joern_bin/joern_extracted/`.
Dos gotchas corregidos: el zip tiene un nivel de anidamiento extra
(`joern/joern/joern-parse`) y `slicer.sh` usa sintaxis `[[ ]]` (bash-only)
pero a veces se invoca con `sh` — correrlo explícitamente con `bash
slicer.sh ...`.

Script batch creado: `AMPLE-main/run_joern_batch.sh` — corre `joern-parse`
función por función, limpia `tmp/` y la carpeta de salida antes de cada
iteración (Joern lanza `RuntimeException: Output directory already exists`
si no), copia `nodes.csv`/`edges.csv` a
`devign_dataset/devign_parsed/<archivo.c>/`.

**Resultado del batch sobre el subset balanceado (última corrida)**:
495/500 (99%) parseadas correctamente.

### 7.1. Joern en el host, y en paralelo (23 y 29 sep 2026)

Joern corre en el **host**, no dentro de `ample-env`, porque esa imagen no
trae Java (`which java` no encuentra nada).

- `run_joern_batch_param.sh`: la misma lógica que `run_joern_batch.sh`, pero
  con el directorio del dataset como argumento (`bash run_joern_batch_param.sh
  devign_dataset_full`).
- `run_joern_batch_parallel.sh <dataset_dir> [n_workers]` (default 28): reparte
  la **lista de archivos** entre N workers; cada `.c` va entero a un solo
  worker, nunca se parte. Cada worker usa su **propia copia completa** de
  `joern_extracted/` (unos 111 MB cada una, se borran al terminar), porque
  `joern-parse` crea, además de `tmp/`, una carpeta de salida con el nombre
  del archivo al lado de `tmp/`; con un directorio compartido, dos workers
  podrían pisarse. Salta los archivos que ya tienen salida, así que se puede
  relanzar sin repetir trabajo. Acepta rutas del dataset relativas o absolutas.

**Verificación contra el secuencial**, sobre 50 archivos procesados por ambos:
47 idénticos byte a byte y 3 con las mismas líneas en distinto orden en
`edges.csv`; ordenados, `nodes.csv` y `edges.csv` dan 0 diferencias en esos 3.
El paralelo no pierde ni mezcla información.

**Tiempos** (dataset completo, 29 sep; salen de las fechas de modificación de
los directorios de salida):

| Modo | Archivos | Tiempo | Por archivo |
|---|---|---|---|
| Secuencial | 7,292 | 28.5 min | 0.235 s |
| Paralelo, 28 workers | 26,572 | 23.8 min | 0.054 s (4.4x más rápido) |

Total: 33,864 de 33,864 archivos parseados. El tiempo del paralelo va del
primer al último archivo escrito y no incluye la preparación de las copias.

Si se interrumpe una corrida secuencial a mitad de un archivo, quedan `tmp/` y
una carpeta `<archivo>.c/` dentro del directorio de Joern; hay que borrarlas a
mano antes de relanzar.

---

## 8. word2vec

Descargado de Zenodo (`word2vec.zip`, 104.9MB, checksum
`787b68e86fee49a861006d28444cd86f` verificado), extraído en
`AMPLE-main/word2vec_model/word2vec/`. Contiene 3 modelos pre-entrenados
(`fan_train_subtoken_data`, `devign_train_subtoken_data`,
`reveal_train_subtoken_data`).

**`fan_train_subtoken_data` no se puede cargar.** gensim 4.3.2 falla con
`FileNotFoundError: fan_train_subtoken_data.wv.vectors.npy` (verificado el
2026-09-30). El zip original de Zenodo trae solo el archivo principal, sin ese
`.npy`, así que no es un problema de la extracción. `devign_train_subtoken_data`
(75,311 palabras) y `reveal_train_subtoken_data` (59,953) sí cargan, ambos de
dimensión 100.

Por eso, en las corridas de este documento se usó un word2vec **propio**,
entrenado sobre el train set de cada corrida (`word2vec.py`, parámetros
`-ep 5 -eb 100`):

| Corrida | Funciones de train | Tiempo |
|---|---|---|
| Validación n=500 | 352 | no registrado |
| Subset mediano | 2,000 | 3.5 s (`time`, sin log archivado) |
| Dataset completo | 13,000 | 19 s (`time`, sin log archivado) |

---

## 9. Fixes de compatibilidad aplicados al código de `AMPLE_code/` y `data_processing/`

El repo original está pinneado a `dgl==0.7.2`; correrlo con `dgl==2.1.0`
rompió varias APIs. Estos son ajustes de compatibilidad, **no cambios a la
arquitectura/lógica del modelo** — documentados inline con comentarios en
cada archivo tocado, quedan permanentes en el repo, no hay que rehacerlos
en corridas futuras:

1. `data_loader/dataset.py`: `graph.add_edge(s, t, data=...)` (API vieja,
   un par de nodos a la vez) → `graph.add_edges(src_list, dst_list,
   data=...)` (API nueva, batch).
2. `graph_transformer_layers.py`: `RelGraphConv(..., low_mem=False, ...)`
   → kwarg `low_mem` ya no existe en dgl≥1.0, removido.
3. `graph_transformer_layers.py`: `fn.src_mul_edge`/`fn.copy_edge`
   (nombres viejos) → `fn.u_mul_e`/`fn.copy_e` (mismos nombres nuevos,
   misma firma posicional).
4. `graph_transformer_layers.py`: 4 tensores creados con
   `device=torch.device('cuda:0')` hardcoded en la función de
   normalización → cambiados a `device=tensor.device`.
5. `modules/model.py`: `graph.to(torch.device('cuda:0'))` hardcoded en
   `forward()` → condicional según el flag `cuda` que la función ya
   recibía pero no usaba para esto.
6. `modules/model.py`: dos operaciones in-place
   (`outputs += self.RepLK(outputs)`, `outputs += self.ConvFFN(outputs)`)
   rompían el grafo de autograd en torch moderno → cambiadas a
   `outputs = outputs + ...` (misma operación matemática, no in-place).
   Encontrado con `torch.autograd.set_detect_anomaly(True)` temporal.
7. `data_loader/batch_graph.py`: `add_subgraph()` tenía otro `cuda:0`
   hardcoded sin flag condicional → agregado chequeo
   `torch.cuda.is_available()`.
8. `main.py`: `model.cuda()`/`loss_function.cuda()` incondicionales →
   envueltos en `if torch.cuda.is_available()`.
9. `trainer.py`: 3 lugares con `targets.cuda()` +
   `model(graph, cuda=True)` hardcoded (el parámetro `cuda=False` de la
   firma de `evaluate_loss` se ignoraba, bug preexistente del repo) →
   reemplazados por chequeo `torch.cuda.is_available()` real.
10. `data_processing/cpg_GS.py` (línea ~651, 23 sep 2026): el argumento
    `--json_files` no declaraba `nargs`, y argparse no aceptaba pasar los tres
    JSON (train, test, valid) como valores separados. Se agregó `nargs=3`.

Con los fixes 4, 5, 7, 8 y 9, el mismo código corre en CPU y en GPU sin
cambiar nada: usa la GPU solo si `torch.cuda.is_available()` es `True`.

Se comparó cada archivo de `AMPLE-main/` contra el repositorio original
(commit `688776e`, 2022-11-19): los archivos modificados son exactamente 7
(`batch_graph.py`, `dataset.py`, `graph_transformer_layers.py`, `main.py`,
`modules/model.py`, `trainer.py` y `cpg_GS.py`), que son los de estos 10 fixes.

### 9.1. `cpg_GS_parallel.py`: la generación de grafos en paralelo

Script nuevo (23 sep 2026), en `data_processing/`. Usa las mismas funciones de
`cpg_GS.py` sin modificarlas (`graphGeneration`, `read_csv`, `word2vec`) y los
mismos flags, pero reparte los archivos de cada split entre N procesos con
`multiprocessing.Pool` (agrega `--workers`; por defecto, los cores menos 2).
Cada proceso carga el modelo word2vec una sola vez (`initializer`). Motivo:
`cpg_GS.py` secuencial tarda ~1 s por archivo (220 archivos en 3 min 40 s,
sin log archivado), y con 32 cores era el cuello de botella.

**Qué se descarta.** `graphGeneration` (`cpg_GS.py`, línea 545) devuelve
`None` si el grafo tiene 0 nodos o **más de 500 nodos**, y esos archivos no
entran al JSON de salida. Es lo que hace que cada split quede con menos
funciones que las extraídas (§10, entrada del 29 sep).

**Verificación.** Sobre el subset de timing (76/17/73) da los mismos conteos
que la versión secuencial: 76 grafos de train, 73 de test y 17 de valid.

**Limitación conocida.** Usa `pool.imap(..., chunksize=8)`, que entrega los
resultados **en orden**. Un archivo lento bloquea la entrega de los demás: en
el train de la corrida completa, casi todos los workers quedaron sin trabajo
unos 14 min esperando a uno (su tiempo de CPU acumulado casi no avanzaba).
No se pierde trabajo ni se corrompe nada, solo se retrasa;
`imap_unordered` lo evitaría (no aplicado).

Comando (sobre el dataset completo, 28 workers):

```bash
docker run --rm --user $(id -u):$(id -g) -v "$(pwd)":/vulndet -w /vulndet/AMPLE-main/data_processing ample-env \
  python3 cpg_GS_parallel.py \
    --csv ../devign_dataset_full/devign_parsed/ \
    --json_files ../devign_dataset_full/devign_data_split/train_raw_code.json \
                 ../devign_dataset_full/devign_data_split/test_raw_code.json \
                 ../devign_dataset_full/devign_data_split/valid_raw_code.json \
    --wv ../devign_dataset_full/devign_wv_models/full_train_subtoken_data \
    --output_dir ../devign_dataset_full/devign_cpg_full \
    --workers 28
```

---

## 10. Bitácora de resultados

Cronología real (7 ago 2026):

1. Dataset Fan et al. descargado (`MSR_data_cleaned.csv`, 10.7GB) en
   `shared_datasets/fan_et_al/`.
2. `extract_subset.py` v1 escrito, subset de 500 extraído — **inválido**
   (0 ejemplos vulnerables, ver §6). Reescrito con reservoir sampling
   estratificado. Nuevo subset n=500: train 178/172 (vul/no-vul), valid
   33/42, test 39/36 — balanceado real.
3. Joern sobre el subset balanceado: 495/500 (99%) parseadas.
4. word2vec propio reentrenado sobre el train set balanceado.
5. `cpg_GS.py`: 252 grafos de train, 55 de valid, 59 de test generados.
   Verificado que los `targets` del JSON de salida mezclan ambas clases.
6. Imagen Docker `ample-env` resuelta tras 9 iteraciones de build (ver §3-4).
7. Los 9 fixes de compatibilidad aplicados a `AMPLE_code/` (§9).
8. Entrenamiento: 100 epochs, CPU, subset n=500 balanceado.

   ```
   Train: Accuracy 96.83%, Precision 99.04%, Recall 93.64%, F1 96.26%
   (overfitting esperable con solo 252 ejemplos de train)

   Test: Accuracy 56.36%, Precision 50.00%, Recall 37.50%, F1 42.86%
   ```

   Sin colapso a una sola clase (a diferencia del intento con el subset
   desbalanceado) — resultado con sentido para un GNN entrenado desde cero
   en un subset tan chico. **Este NO es un resultado final de AMPLE para el
   artículo** — es la prueba de que el pipeline completo (Joern → word2vec
   → cpg → entrenamiento) funciona de punta a punta en esta máquina. Mucho
   peor que LineVul (F1 91.4%) es esperable: LineVul usa un checkpoint ya
   entrenado en el dataset completo, mientras que aquí AMPLE se entrenó
   desde cero en apenas 500 funciones (AMPLE no publica checkpoint,
   a diferencia de LineVul).

9. **23 sep 2026 — GPU y pruebas de rendimiento.** Con dgl compilado desde
   source (§3.1) se entrenó en GPU, y el mismo subset mediano también en CPU.
   100 épocas, `--batch_size 32`. Los tiempos salen del primer y el último
   timestamp de cada log, archivado en
   `AMPLE_code/models/<corrida>/devign_FFmpeg.log`:

   | Corrida | Train / Valid / Test (con grafo) | Tiempo | Test Acc / Prec / Rec / F1 (%) |
   |---|---|---|---|
   | `timing_test_gpu` | 76 / 17 / 73 | 42.6 s | 58.90 / 52.94 / 56.25 / 54.55 |
   | `medium_test_gpu` | 1,428 / 279 / 368 | 9 min 15 s | 55.16 / 49.66 / 45.12 / 47.28 |
   | `medium_test_cpu` | 1,428 / 279 / 368 | 13 min 16 s | 52.45 / 45.99 / 38.41 / 41.86 |

   En el subset mediano la GPU fue 1.43x más rápida que la CPU. Cautelas:
   una sola corrida por configuración; la GPU usa torch 2.11 con dgl compilado
   y la CPU usa torch 2.2.1 con dgl 2.1.0, así que las diferencias de métricas
   no se pueden atribuir al dispositivo; el log de CPU está en UTC (dice
   09:58 a 10:11, que son 04:58 a 05:11 hora local); y con 73 o 368 funciones
   de test estas métricas no son un resultado de AMPLE, solo validan el
   pipeline. El subset mediano tiene 2,900 funciones (2,000 train, 400 valid,
   500 test): Joern secuencial tardó 12 min 34 s y `cpg_GS_parallel.py` con
   28 workers 9 min 14 s (ambos `time`, sin log archivado).

10. **29 sep 2026 — dataset a escala completa** (`devign_dataset_full/`; el
    prefijo `devign_` es heredado, el contenido es Big-Vul). Extracción: test
    18,864 (las del `test.csv` de LineVul), train 13,000, valid 2,000. Joern:
    33,864 de 33,864 (§7.1). word2vec propio sobre el train: 19 s (sin log
    archivado). `cpg_GS_parallel.py` con 28 workers, 30 min 32 s en total
    (barra `tqdm`), 18.5 archivos/s:

    | Split | Funciones | Con grafo | Retención | Tiempo |
    |---|---|---|---|---|
    | train | 13,000 | 9,233 | 71.0 % | 22 min 52 s |
    | test | 18,864 | 14,772 | 78.3 % | 4 min 43 s |
    | valid | 2,000 | 1,433 | 71.7 % | 2 min 57 s |

    **El filtro de 500 nodos (§9.1) no descarta parejo entre clases:** pierde
    más funciones vulnerables, porque las funciones largas pesan más.

    | Split | Vulnerables: original → con grafo | Se conservan | No vulnerables que se conservan |
    |---|---|---|---|
    | train | 6,523 (50.2 %) → 4,097 (44.4 %) | 62.8 % | 79.3 % |
    | valid | 977 (48.9 %) → 640 (44.7 %) | 65.5 % | 77.5 % |
    | test | 1,055 (5.59 %) → 688 (4.66 %) | 65.2 % | 79.1 % |

    Fuente: `devign_cpg_full/devign-<split>-file.json` contra
    `devign_data_split/<split>_raw_code.json` (los nombres terminan en
    `_<label>.c`). No se separó cuántos descartes son por más de 500 nodos y
    cuántos por otros motivos. Consecuencia: **el test de AMPLE queda en
    14,772 funciones, no en las 18,864 de LineVul**, sesgado hacia funciones
    más chicas y con menos vulnerables. Sus métricas no son directamente
    comparables con las de LineVul hasta decidir cómo tratarlo (§11).

    **Entrenamiento a escala completa: sin resultado.** Se lanzó el 29 sep a
    las 23:01 con 9,233 / 1,433 / 14,772 y se cortó a propósito pocos minutos
    después, antes de completar la primera época. Su log
    (`AMPLE_code/models/full_test_gpu/devign_FFmpeg.log`) termina en
    `Start training!`. No guarda checkpoints intermedios, así que hay que
    relanzarlo desde cero.

**Cómo leer el log de entrenamiento** (código original de `trainer.py`, no
modificado por nosotros aparte de los `.cuda()` del fix 9):

- `main.py` escribe el log **en vivo**, una línea por época para `Train`,
  `Valid` y `Test`. Se sigue con `tail -f AMPLE_code/models/<dataset>/devign_FFmpeg.log`,
  y sirve para proyectar el tiempo total a partir de las primeras épocas.
- El modelo se **elige por el F1 de valid**: se guarda el checkpoint cuando
  mejora ese F1 y, al terminar, se carga el mejor para calcular el resultado.
- Las líneas `Test` de cada época son solo monitoreo y **no son el
  resultado**: el código pasa `initialize_valid_batch()` como número de lotes,
  así que evalúan solo tantos lotes de test como lotes tiene valid, o sea una
  parte del test. **El resultado es la última línea**
  (`models/<dataset>/DevignModel  ----Test----`), que evalúa todo el test con
  el mejor checkpoint de valid.
- `max_patience=100` con 100 épocas: en la práctica el entrenamiento siempre
  corre las 100 épocas, sin parada anticipada.

---

## 11. Pendiente

- **Entrenar a escala completa en GPU** (9,233 / 1,433 / 14,772, ya
  preprocesado): es el bloqueador para tener un resultado real de AMPLE.
  Comando en §12. Tiempo: **no medido**; una extrapolación lineal desde el
  subset mediano (289 lotes de train frente a 45) da del orden de 1 h. Lo más
  fiable es leer el tiempo de las primeras épocas en el log (§10) y
  multiplicarlo por 100.
- **Decidir cómo comparar con LineVul.** El test de AMPLE queda en 14,772 de
  las 18,864 funciones y pierde más vulnerables (§10, entrada del 29 sep).
  Opciones: (a) calcular las métricas de LineVul solo sobre esas mismas 14,772
  funciones (requiere predicciones por función de LineVul; `linevul_main.py`
  tiene la opción `write_raw_preds`); (b) reportar ambos resultados dejando
  explícita la diferencia de test set.
- **Decidir el tamaño de train.** Se usaron 13,000 por tiempo de máquina; el
  CSV crudo permite mucho más. Más datos probablemente mejoran a AMPLE; hay
  que declarar la limitación.
- **Reintentar GPU ya no está bloqueado**: la conclusión de §3 (solo CPU) se
  corrigió en §3.1 con la compilación de dgl desde source.
- Los 10 fixes de compatibilidad (§9) y la imagen `ample-env` no hay que
  rehacerlos: quedaron aplicados y resueltos.
- Opcional: cambiar `pool.imap` por `imap_unordered` en `cpg_GS_parallel.py`
  para evitar el bloqueo por un archivo lento (§9.1).

---

## 12. Comandos exactos de la corrida a escala completa

Todo desde `docker-labs/vulnerability-detection/`, salvo donde se indica.
Los pasos 1, 3 y 4 usan el contenedor `ample-env` con `--user` para que los
archivos generados queden a nombre del usuario y no de root.

```bash
# 1. Extraer test (18,864 de LineVul) + train 13,000 + valid 2,000 (§6.1)
docker run --rm --user $(id -u):$(id -g) -v "$(pwd)":/vulndet -w /vulndet/AMPLE-main ample-env \
  python3 extract_from_linevul_split.py --test-n 0 --train-n 13000 --valid-n 2000 \
  --out-dir devign_dataset_full

# 2. Joern en paralelo, en el host (§7.1)
cd AMPLE-main && bash run_joern_batch_parallel.sh devign_dataset_full 28 && cd ..

# 3. word2vec propio sobre el train (§8)
docker run --rm --user $(id -u):$(id -g) -v "$(pwd)":/vulndet -w /vulndet/AMPLE-main/data_processing ample-env \
  python3 word2vec.py --data_paths ../devign_dataset_full/devign_data_split/train_raw_code.json \
  -bin ../devign_dataset_full/devign_wv_models -n full_train_subtoken_data -ep 5 -eb 100

# 4. Grafos en paralelo (§9.1): comando completo en esa sección
#    -> escribe AMPLE-main/devign_dataset_full/devign_cpg_full/*.json

# 5. Dejar los grafos donde los espera main.py
mkdir -p AMPLE-main/devign_dataset_full_input/devign_cpg_v2_1
cp AMPLE-main/devign_dataset_full/devign_cpg_full/*.json AMPLE-main/devign_dataset_full_input/devign_cpg_v2_1/
```

Entrenar en **GPU** (entorno de §3.1). Hay que correrlo desde `AMPLE_code/`,
no desde dentro de `dgl_build_experimental/` (el cwd sombrea el paquete
`dgl` instalado):

```bash
cd AMPLE-main/AMPLE_code
source ../../.venv-ample-gpu-experimental/bin/activate
export LD_LIBRARY_PATH="$(realpath ../../dgl_build_experimental/dgl/build):$LD_LIBRARY_PATH"
export HOME=/tmp DGLBACKEND=pytorch
python3 main.py --dataset full_test_gpu --input_dir ../devign_dataset_full_input/ --batch_size 32
```

Alternativa en **CPU** (imagen `ample-env`). El `-e HOME=/tmp` es necesario:
con `--user`, dgl intenta escribir su configuración en `/.dgl` y falla con
`PermissionError`:

```bash
cd AMPLE-main/AMPLE_code
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v "$(pwd)/../..":/vulndet \
  -w /vulndet/AMPLE-main/AMPLE_code ample-env \
  python3 main.py --dataset full_test_cpu --input_dir ../devign_dataset_full_input/ --batch_size 32
```
