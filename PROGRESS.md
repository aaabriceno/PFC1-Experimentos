# PROGRESS.md — vulnerability-detection (bitácora de experimentos)

Última actualización: 2026-10-28.

Esta es la bitácora DETALLADA de lo que pasa en esta carpeta (comandos,
errores, fixes, resultados). El `PROGRESS.md` del artículo
(`~/Documentos/Semestre 2026-2/PFC1/PROGRESS.md`) solo guarda un resumen —
consultar ese archivo para el estado del artículo en sí, y este archivo para
el detalle técnico de cada corrida.

Ver el `README.md` de este repositorio para el contexto y las reglas de
trabajo (todo número lleva su fuente).

Convención de este archivo: cada número lleva su fuente. Lo marcado
**(sin log archivado)** salió de la salida de `time` o de un monitor en una
sesión y no quedó guardado en un archivo: sirve de orientación, pero NO debe
citarse en el artículo. Lo marcado **ESTIMACIÓN** no se midió.

## Protocolo de comparación LineVul vs AMPLE sobre FFmpeg+QEMU (2026-10-24)

Decisión (alumno, 2026-10-24): empezar la parte experimental por FFmpeg+QEMU
(Devign), porque es chico y deja entrenar **ambos modelos desde cero con el
mismo reparto 80/10/10**: la única comparación en igualdad de condiciones.
Big-Vul queda como segunda etapa. El alumno lo presentará a sus profesores la
próxima semana (jueves o viernes).

### Qué significa 80/10/10
80 % train (ajusta pesos), 10 % valid (elige cuándo parar o qué checkpoint
guardar, no ajusta pesos), 10 % test (número final, una sola vez). LineVul
guarda el checkpoint con mejor F1 de valid (`linevul_main.py:194`); AMPLE guarda
el mejor por F1 de valid con paciencia (`trainer.py:123`).

### Hechos medidos hoy
- **Big-Vul crudo** (`MSR_data_cleaned.csv`, contado): 188,636 funciones,
  10,900 vulnerables = 5.778 %. LineVul usa las 188,636 (test = 18,864). El
  paper de AMPLE reporta 179,299 (10,547 vul, 5.88 %): 9,337 menos, motivo no
  explicado en el paper. En nuestro pipeline, `cpg_GS.py:545` descarta los
  grafos de más de 500 nodos (comentario del autor: "V1 era 750"), pero eso no
  prueba que sea el motivo del paper. Esto reemplaza la cifra "~187 mil filas
  (no contada)" de más abajo.
- **Fuente de FFmpeg+QEMU verificada**: el enlace del `readme.md` de AMPLE
  (github.com/AMPLE001/AMPLE) y el de la página oficial de Devign
  (sites.google.com/view/devign) son el mismo archivo de Google Drive
  (id `1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF`). Descargado con `gdown` a
  `conjuntos_de_datos/ffmpeg_qemu/function.json` (61,532,917 bytes = 58.7 MiB).
- **Contenido real de `function.json`**: 27,318 funciones (campos `project`,
  `commit_id`, `target`, `func`); 12,460 vulnerables = 45.61 %; FFmpeg 9,769 y
  QEMU 17,549. **El paper de AMPLE reporta 22,361 (10,067 vul)**: 4,957 menos
  (18.1 %), motivo sin explicar. Hipótesis sin probar: el mismo filtro de grafos.
- **Calidad**: 60 textos de función aparecen dos veces con etiquetas distintas
  (120 filas). Se descartan todas: no se sabe cuál etiqueta es la correcta y
  podrían caer en particiones distintas (fuga).
- **Reparto resultante** (`split_semilla42`, semilla 42, estratificado por
  proyecto y etiqueta, 27,198 funciones):

| Partición | Funciones | Vulnerables |
|---|---|---|
| train | 21,758 | 9,920 (45.59 %) |
| val | 2,719 | 1,240 (45.61 %) |
| test | 2,721 | 1,240 (45.57 %) |

  Verificado: ids y textos disjuntos entre particiones; los CSV reproducen
  `function.json` fila por fila (0 diferencias).

### Archivos y scripts nuevos
- Carpeta `shared_datasets/` renombrada a `conjuntos_de_datos/` y
  `fan_et_al/` a `big_vul/`; actualizadas las rutas en
  `extract_from_linevul_split.py`, `extract_subset.py`, la documentación de
  AMPLE y el symlink de `Leev-master/MSR_data_cleaned.csv`.
- `protocolo_comparacion/dividir_ffmpeg_qemu.py`: genera el reparto común.
  Salida en `conjuntos_de_datos/ffmpeg_qemu/split_semilla42/`: `train.csv`,
  `val.csv`, `test.csv` (columnas `id, project, commit_id, target,
  processed_func`, que es lo que lee LineVul) y `resumen.txt`.
- `protocolo_comparacion/convertir_a_ample.py`: pasa esos CSV al formato de
  AMPLE (un `.c` por función + JSON por partición) en
  `AMPLE-main/devign_dataset_ffmpeg_qemu/` (27,198 `.c`; 188 MB).
- `protocolo_comparacion/` quedó en la lista blanca de `.gitignore`
  (2026-10-25), así que sus scripts se versionan.

### Avances del 2026-10-25 (LineVul listo para entrenar; AMPLE sin lanzar)
- **Fix #1 a LineVul** (`linevul_main.py`, líneas 126 y 136, hecho por el
  alumno): `max_steps` y `save_steps` se dividen entre
  `gradient_accumulation_steps`. Sin él, con acumulación 4 la validación y el
  checkpoint se dispararían cada 4 épocas y la tasa de aprendizaje no
  completaría su calendario. Verificado con una corrida corta real (40 pasos de
  optimización, una validación por época). Detalle en
  `LineVul-main/DOCUMENTACION_ENTORNO_LINEVUL.md` §13.1.
- **Memoria/velocidad de LineVul en 8 GB** (prueba corta, sin log archivado):
  batch 16 y batch 8 dan CUDA OOM; batch 4 con acumulación 4 funciona (pico
  5,459 MiB, ~11.6 funciones/s). ESTIMACIÓN: ~5.2 h para 10 épocas con 21,758
  funciones. §13.2.
- **Test de LineVul**: exige `flaw_line` y `flaw_line_index`; sin ellas falla
  con `KeyError` al final. Se creó `protocolo_comparacion/preparar_test_linevul.py`
  y `split_semilla42/test_linevul.csv` (2,721 filas, columnas nuevas vacías,
  verificado idéntico al original en el resto). §13.3.
- **Limpieza (nivel 1)**: borrados `AMPLE-main/tmp/` (copia perdida de un
  directorio temporal), `joern_bin/joern.zip`, `joern_bin/test_input`,
  `.DS_Store` y los `__pycache__` (quedan 2 con archivos de root). ~540 MB.
  Pendiente, a decisión del alumno: nivel 2 (`word2vec_model`,
  `attention_weight`, `bow_rf`, `cppcheck`, `logo`, `LineVul_Presentation.pdf`),
  nivel 3 (`devign_dataset`, `_timing*`, `_medium*`, ~1.05 GB) y la
  reorganización de carpetas.
- **Aún no lanzado**: Joern/grafos de AMPLE sobre FFmpeg+QEMU y ningún
  entrenamiento. Antes de reutilizar un `*_input/`, borrar su `devign.bin`
  (caché que `main.py` lee ignorando los JSON).

### Decisión (2026-10-25): orden de las dos etapas y partición de Big-Vul
**Orden:** 1.º FFmpeg+QEMU con el reparto propio (ensayo barato, ambos modelos
desde cero); 2.º Big-Vul usando las particiones de LineVul para los dos modelos.

**Qué publica cada repositorio** (nombres leídos en las páginas de Drive; los
tamaños no se muestran):
- AMPLE enlaza el Big-Vul **completo, sin partir**: `MSR_data_cleaned.zip`
  (`1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X`), el mismo nombre que el zip que ya
  tenemos en `conjuntos_de_datos/big_vul/`. El reparto 8:1:1 lo hacen ellos al
  azar (paper); no publican la partición.
- LineVul publica 4 archivos ya procesados: `train.csv`
  (`1ldXyFvHG41VMrm260cK_JEPYqeb6e6Yw`), `val.csv`
  (`1yggncqivMcP0tzbh8-8Eu02Edwcs44WZ`), `test.csv`
  (`1h0iFJbc5DGXCXXvvR6dru_Dms_b2zW4V`, ya descargado) y `processed_data.csv`
  completo (`10-kjbsA806Zdk54Ax8J3WvLKGTzN8CMX`).

**Verificado (2026-10-25):** las 18,864 filas del `test.csv` de LineVul están en
el crudo `MSR_data_cleaned.csv` (188,636 filas, 10,900 vulnerables = 5.778 %),
emparejadas por la columna `index`: `func_before` y etiqueta idénticas en las
18,864 (0 diferencias). El `test.csv` de LineVul añade `index`, `target`,
`processed_func`, `flaw_line` y `flaw_line_index`. No verificados: `train.csv` y
`val.csv` (no están en disco).

**Por qué usar el reparto de LineVul en la etapa 2:** el checkpoint de LineVul
ya se entrenó con ese train y se eligió con ese val, así que **no se reentrena**;
AMPLE se entrena con el mismo train/val y se prueba con el mismo test: ambos ven
las mismas funciones (se resuelve el train desigual de ~150k frente a 13k).
Matices a declarar: (a) AMPLE descarta grafos de más de 500 nodos (21.7 % del
test en la corrida previa), así que se compara sobre la intersección usando las
predicciones por función de LineVul (`--write_raw_preds`); (b) hay que decidir y
declarar qué texto recibe Joern (`func_before` del crudo, como hasta ahora, o
`processed_func`, que es lo que lee LineVul).

**Al empezar la etapa 2:** descargar `val.csv` y `train.csv` con `gdown`
(`.venv-ample/bin/gdown`) y verificarlos contra el crudo por `index`, igual que
el test: disjuntos entre sí, sin funciones perdidas ni repetidas (esperado: train
150,908, val 18,864, test 18,864), y código/etiqueta idénticos. Plan B si la
descarga falla (cuota de Drive): train = índices del crudo menos test menos val;
tiene como riesgo que no verifica que LineVul no haya descartado filas. Costo
estimado de AMPLE a esa escala, sin medir: Joern ~2.8 h, grafos ~2.8 h,
entrenamiento ~18 h (del orden de un día de máquina).

### Plan (en orden)
1. **Hecho:** descarga, verificación de la fuente, limpieza, reparto común y
   conversión al formato de AMPLE.
2. **AMPLE, preprocesamiento (pendiente, NO lanzado):** Joern en paralelo sobre
   los 27,198 `.c` → word2vec solo con el train → `cpg_GS_parallel.py` (descarta
   grafos de más de 500 nodos) → copiar los grafos a `devign_dataset_ffmpeg_qemu_input/`.
   ESTIMACIÓN con el ritmo medido (0.054 s/archivo en paralelo): ~25 min de
   Joern más ~25 min de grafos.
3. **AMPLE, entrenamiento:** `main.py` en GPU sobre ~21.7k de train. El resultado
   es la última línea `----Test----`.
4. **LineVul, entrenamiento desde `microsoft/codebert-base`** con `train.csv` y
   `val.csv`, prueba con `test.csv` (hiperparámetros del paper: 10 épocas, batch
   16, lr 2e-5). **Hecho el 2026-10-27/28** con lote 4 y acumulación 4 (batch 16 y
   8 no caben en 8 GB); resultados en la sección siguiente y en EXP-03.
5. **Comparación:** AMPLE solo evalúa las funciones con grafo, así que se
   reportan ambos modelos sobre esa intersección (con `write_raw_preds` de
   LineVul) y, aparte, LineVul sobre las 2,721 completas, con la cobertura
   explícita.
6. **Comprobación extra:** contar cuántas de las 27,198 funciones sobreviven al
   filtro de 500 nodos y compararlo con las 22,361 del paper de AMPLE.
7. Documentar todo en las dos `DOCUMENTACION_ENTORNO_*.md` y, con resultados
   reales, actualizar el artículo (sección IV y conclusiones).

Después, Big-Vul con el protocolo de la sección siguiente de este archivo.

### Entrenamiento de LineVul sobre FFmpeg+QEMU: TERMINADO (2026-10-27 19:59 a 2026-10-28 01:36, hora local)
Detalle completo, tablas y evidencia en
`LineVul-main/DOCUMENTACION_ENTORNO_LINEVUL.md` §13. Resumen:

- **Cómo:** `protocolo_comparacion/entrenar_linevul_ffmpeg_qemu.sh`, lanzado con
  `setsid nohup ... &`. Desde `microsoft/codebert-base`, 10 épocas, lote 4 con
  acumulación 4 (lote efectivo 16), lr 2e-5, semilla 123456, RTX 5060. Log:
  `LineVul-main/entrenamiento_ffmpeg_qemu_2026-10-27_1959.log` (en UTC, +5 h).
- **Verificado en el log:** `Num examples = 21758`, `Total optimization steps =
  13600` (el fix #1 funciona en la corrida real).
- **Tiempo (log):** 5 h 37 min; por época 31 min 13 s de entrenamiento más 2 min
  9 s de validación.
- **Validación:** F1 por época 0.617, 0.370, 0.563, **0.625**, 0.561, 0.546, 0.581,
  0.564, 0.586, 0.581. Mejor la época 4 (precisión 0.602, recall 0.650); ese
  checkpoint se usó en el test. La pérdida real de entrenamiento bajó de 0.675 a
  0.411 (la registrada × 4).
- **Test (2,721 funciones; 1,240 vulnerables = 45.57 %):** exactitud 0.6354,
  precisión 0.5932, recall 0.6363, **F1 0.614**. Recalculado desde
  `raw_preds_ffmpeg_qemu.csv` (TP 789, FP 541, FN 451, TN 940): coincide con el log.
- **Lectura prudente:** hay señal por encima del azar (exactitud 0.635 frente a
  0.544 de decir siempre "no vulnerable"; precisión 0.593 frente a una base de
  0.456), pero el F1 queda a la altura de decir siempre "vulnerable" (0.626, por
  cálculo): con 45 % de vulnerables el F1 solo engaña.
- **No comparable** con el F1 de 56.54 % de LineVul en FFmpeg+Qemu del paper de
  AMPLE: lo midieron los autores de AMPLE con su pipeline y 22,361 funciones (las
  nuestras son 27,198), y esta es una sola corrida con una semilla.
- **Hipótesis sin probar:** (a) posible sobreajuste (la validación no mejora
  desde la época 4 mientras la pérdida de entrenamiento sigue bajando); (b) el
  corte a 510 tokens: 48.5 % de las funciones del test se cortan, frente al 18.7 %
  en el test de Big-Vul (mediciones del 2026-10-28, **sin log archivado**, no
  citar hasta archivarlas; §13.7 de la documentación).
- **Hecho al terminar:** `raw_preds.csv` renombrado a `raw_preds_ffmpeg_qemu.csv`
  (el programa siempre escribe con el primer nombre y lo pisaría).
- **Archivos nuevos sin versionar:** el log y
  `LineVul-main/evidenciasLineVul/evidenciaEntrenamiento-epoch0.png`. El
  checkpoint (`*.bin`) y las predicciones (`*.csv`) están ignorados por git. No se
  hizo commit.

### Registro de experimentos (para la sección IV del artículo)
Cada resultado lleva su fuente. "Sin log archivado" = no citable aún.

| ID | Experimento | Datos | Resultado | Evidencia | Estado |
|---|---|---|---|---|---|
| EXP-01 | LineVul, réplica del paper con el checkpoint de los autores (solo inferencia) | Test de Big-Vul, 18,864 funciones | Exactitud 0.9909, precisión 0.9712, recall 0.8635, F1 0.9142 (paper: 0.91 / 0.97 / 0.86) | `LineVul-main/resultados_gpu_test.log` (2026-07-09); re-corrida del 2026-09-30 con captura `evidenciasLineVul/testPruebaLineVul.png` | Hecho |
| EXP-02 | AMPLE, pruebas piloto del pipeline y de tiempos GPU/CPU | Subconjuntos de 59 a 368 funciones de test | F1 de 42 % a 55 %: **validan el pipeline, no son resultados de AMPLE** | `AMPLE-main/AMPLE_code/models/*/devign_FFmpeg.log`; detalle en la sección de AMPLE más abajo | Hecho (la corrida a escala Big-Vul se cortó, sin resultado) |
| EXP-03 | LineVul entrenado desde CodeBERT | FFmpeg+QEMU, reparto 80/10/10 semilla 42 (21,758 / 2,719 / 2,721) | Exactitud 0.6354, precisión 0.5932, recall 0.6363, F1 0.614; mejor época la 4 | Log `entrenamiento_ffmpeg_qemu_2026-10-27_1959.log`; `raw_preds_ffmpeg_qemu.csv`; documentación §13.6 | Hecho |
| EXP-04 | Longitud en tokens, truncado a 510 y longitud como predictor; límite de 514 posiciones de CodeBERT | Test de Big-Vul y FFmpeg+QEMU | 48.5 % de funciones cortadas frente a 18.7 %; AUC solo con longitud 0.537 frente a 0.689; 512 tokens funciona y 513 falla | Documentación §13.7, **sin log archivado** | Hecho, falta archivar script y salida |
| EXP-05 | Rendimiento de EXP-03 en funciones ≤510 tokens frente a >510 | Test de FFmpeg+QEMU | — | Saldría de `raw_preds_ffmpeg_qemu.csv` y la cuenta de tokens | Pendiente (barato, no requiere reentrenar) |
| EXP-06 | AMPLE sobre el mismo reparto | FFmpeg+QEMU, mismos 27,198 `.c` ya convertidos | — | — | Pendiente: Joern, word2vec, grafos, entrenamiento; ESTIMACIÓN ~50 min de preparación y ~2 h de entrenamiento |
| EXP-07 | Comparación sobre Big-Vul con las particiones de LineVul para los dos modelos | Big-Vul, 188,636 funciones | — | — | Pendiente (ver "Decisión (2026-10-25)") |

## Estado por modelo (2026-09-30)

### LineVul — DONE, resultado final (reconfirmado el 2026-09-30)
Corrida completa en GPU (imagen `linevul:cu128`), test set completo de Fan
et al./Big-Vul (18,864 funciones), checkpoint pre-entrenado del paper.

```
test_accuracy  = 0.9909
test_f1        = 0.9142
test_precision = 0.9712
test_recall    = 0.8635
test_threshold = 0.5
```

Replica el paper (F1 0.91 / Prec 0.97 / Rec 0.86).

- Corrida original (2026-07-09): log crudo en
  `LineVul-main/resultados_gpu_test.log`. Inferencia de 03:43:32 a 03:58:28
  (14 min 56 s).
- Re-corrida (2026-09-30): mismos cuatro números y misma duración,
  14 min 56 s (20:18:48 a 20:33:44 UTC en el log del contenedor, que va en
  UTC; el host está en UTC-5). Evidencia: captura
  `LineVul-main/evidenciasLineVul/testPruebaLineVul.png`. La salida no se
  guardó en un `.log` (el contenedor corre con `--rm`).
- Durante el test no se imprime progreso: tras `Num examples = 18864` no
  aparece nada hasta los resultados. Es normal, no es un cuelgue.

Detalle del entorno: `LineVul-main/DOCUMENTACION_ENTORNO_LINEVUL.md`.

Reentrenamiento desde CodeBERT sobre FFmpeg+QEMU (2026-10-27/28, exactitud
0.6354, F1 0.614): ver "Entrenamiento de LineVul sobre FFmpeg+QEMU: TERMINADO" y
el registro de experimentos, más arriba en este archivo.

### AMPLE — preprocesamiento a escala completa LISTO; entrenamiento final PENDIENTE
**No hay todavía ningún resultado de AMPLE a escala completa.** La corrida de
entrenamiento con el dataset completo se lanzó el 2026-09-29 a las 23:01 y se
cortó a los pocos minutos, antes de completar la primera época (su log,
`AMPLE_code/models/full_test_gpu/devign_FFmpeg.log`, termina en
`Start training!`). No guarda checkpoints intermedios: hay que relanzarla desde
cero.

#### Entornos
- CPU: imagen Docker `ample-env` (torch 2.2.1+cpu, dgl 2.1.0). Sin cambios.
- GPU (nuevo, 2026-09-23): dgl compilado desde source contra torch
  2.11.0+cu128 para Blackwell (sm_120). Venv
  `.venv-ample-gpu-experimental/` + código/binarios en
  `dgl_build_experimental/`. El venv depende de esa carpeta (instalación
  editable): borrar una rompe la otra. Comprobado con un forward pass real de
  `dgl.nn.GraphConv` en `cuda:0`. Se encontraron y arreglaron 3 bugs de dgl
  en el camino; detalle en `AMPLE-main/DOCUMENTACION_ENTORNO_AMPLE.md` §3.1.
- Corre con `LD_LIBRARY_PATH` apuntando a `dgl_build_experimental/dgl/build`,
  `HOME=/tmp` y `DGLBACKEND=pytorch`. No ejecutar desde dentro de
  `dgl_build_experimental/` (el cwd sombrea el paquete instalado y `import
  dgl` da un módulo vacío).

#### Dataset "full" (carpeta `AMPLE-main/devign_dataset_full/`)
El nombre `devign_*` es heredado del código de AMPLE. El contenido NO es
Devign: es Fan et al./Big-Vul (`MSR_data_cleaned.csv`).

- Test: las 18,864 funciones de `LineVul-main/data/big-vul_dataset/test.csv`,
  unidas por la columna `index` con el CSV crudo. Se recuperaron 18,864 de
  18,864.
- Train 13,000 y valid 2,000: muestreados con reservoir sampling balanceado por
  clase sobre las filas del CSV crudo que NO están en el test de LineVul (para
  no contaminarlo). Script: `extract_from_linevul_split.py`.
- Se eligieron 13k/2k por tiempo de máquina, no porque el dataset no dé más:
  el CSV crudo trae unas ~187 mil filas (cifra aproximada, no contada
  localmente). Entrenar con más datos probablemente mejora a AMPLE; esta
  limitación hay que declararla en el artículo.

#### Preprocesamiento sobre el dataset "full" (2026-09-29)
| Paso | Resultado | Tiempo | Fuente |
|---|---|---|---|
| Joern secuencial | 7,292 archivos | 28.5 min (0.235 s/archivo) | mtimes de `devign_parsed/` |
| Joern paralelo, 28 workers | 26,572 archivos | 23.8 min (0.054 s/archivo, 18.6 archivos/s) | mtimes de `devign_parsed/` |
| word2vec propio (train) | 13,000 funciones | 19 s | `time` (sin log archivado) |
| `cpg_GS_parallel.py`, 28 workers | ver abajo | 30 min 32 s en total | barra `tqdm` de la corrida |

- Joern: 33,864 de 33,864 archivos parseados. El paralelo fue 4.4x más rápido
  por archivo que el secuencial. El tiempo del paralelo va del primer al último
  archivo escrito; no incluye la preparación de las 28 copias de Joern.
- Validación del Joern paralelo contra el secuencial, sobre 50 archivos:
  47 idénticos byte a byte; 3 difieren solo en el orden de las líneas de
  `edges.csv`. Ordenadas, `nodes.csv` y `edges.csv` dan 0 diferencias en esos 3.
  Se da por bueno: el paralelo no pierde ni mezcla información.
- `cpg_GS_parallel.py`, por split:

| Split | Funciones | Con grafo | Retención | Tiempo |
|---|---|---|---|---|
| train | 13,000 | 9,233 | 71.0 % | 22 min 52 s |
| test | 18,864 | 14,772 | 78.3 % | 4 min 43 s |
| valid | 2,000 | 1,433 | 71.7 % | 2 min 57 s |

  Total: 33,864 archivos en 30 min 32 s = 18.5 archivos/s. Frente a ~1 s por
  archivo en el `cpg_GS.py` secuencial (220 archivos en 3 min 40 s, **sin log
  archivado**) sale unas 18x más rápido. El tiempo de train incluye un
  estancamiento de ~14 min (ver "Problemas conocidos").

#### Comparabilidad con LineVul: hallazgo importante, sin resolver
`cpg_GS` descarta los grafos de más de 500 nodos (y los fallos de parseo). Eso
descartó el 21.7 % del test: **AMPLE se evaluaría sobre 14,772 funciones, no
sobre las 18,864 de LineVul**. Además el descarte NO es parejo entre clases;
pierde más funciones vulnerables (las funciones largas pesan más):

| Split | Vulnerables: original → con grafo | No vulnerables: retención |
|---|---|---|
| train | 6,523 (50.2 %) → 4,097 (44.4 %), se conserva 62.8 % | 79.3 % |
| valid | 977 (48.9 %) → 640 (44.7 %), se conserva 65.5 % | 77.5 % |
| test | 1,055 (5.59 %) → 688 (4.66 %), se conserva 65.2 % | 79.1 % |

Fuente: `devign_cpg_full/devign-{split}-file.json` contra
`devign_data_split/{split}_raw_code.json` (los nombres de archivo terminan en
`_<label>.c`). No se separó cuántos descartes son por nodos > 500 y cuántos por
fallo de parseo.

Consecuencias para el informe:
- El test de AMPLE queda sesgado hacia funciones más chicas y con menos
  prevalencia de vulnerables. Sus métricas no son directamente comparables con
  las de LineVul sobre las 18,864.
- Opciones abiertas (decisión pendiente): (a) calcular las métricas de LineVul
  solo sobre las mismas 14,772 funciones (requiere predicciones por función de
  LineVul; `linevul_main.py` tiene la opción `write_raw_preds`); (b) reportar
  ambos resultados dejando explícita la diferencia de test set.

#### Pruebas de rendimiento GPU vs CPU (validación del pipeline, NO resultados finales)
Misma configuración en todas: 100 épocas, `--batch_size 32`. Fuente: logs de
entrenamiento archivados en `AMPLE_code/models/<corrida>/devign_FFmpeg.log`
(del primer timestamp al último).

| Corrida | Train / Valid / Test | Tiempo | Test Acc / Prec / Rec / F1 |
|---|---|---|---|
| `timing_test_gpu` | 76 / 17 / 73 | 42.6 s | 58.90 / 52.94 / 56.25 / 54.55 |
| `medium_test_gpu` | 1,428 / 279 / 368 | 9 min 15 s | 55.16 / 49.66 / 45.12 / 47.28 |
| `medium_test_cpu` | 1,428 / 279 / 368 | 13 min 16 s | 52.45 / 45.99 / 38.41 / 41.86 |

- En el dataset mediano la GPU fue 1.43x más rápida que la CPU. La GPU no es el
  cuello de botella del pipeline: Joern y `cpg_GS` corren en CPU.
- VRAM pico con `nvidia-smi -l 1`: 651 MiB (chico) y 687 MiB (mediano) de
  8,151 MiB. **(sin log archivado)**
- Cautelas: una sola corrida por configuración; la GPU usa torch 2.11/dgl
  compilado y la CPU usa torch 2.2.1/dgl 2.1.0, así que las diferencias de
  métricas entre ambas no se pueden atribuir al dispositivo. Esos F1 (42-55 %)
  salen de subsets de 73 y 368 funciones de test: no son el resultado de
  AMPLE.
- El tiempo de CPU del subset chico (~50 s) no tiene log archivado y no se
  usa.
- Tiempo de entrenamiento a escala completa (9,233 de train): **no medido**.
  ESTIMACIÓN, extrapolando linealmente desde la corrida mediana: del orden de
  1 h sin contar la evaluación.

#### Escalar a todas las filas disponibles fuera del test (~168-170k)
ESTIMACIÓN, no medida. Con los ritmos medidos arriba: Joern ~2.8 h,
`cpg_GS_parallel` ~2.8 h, entrenamiento ~18 h (este último es extrapolación
sin medir). Total del orden de 1 día de máquina. Todo el tiempo de GPU está en
esa última parte.

#### Scripts nuevos en `AMPLE-main/` (documentados en `DOCUMENTACION_ENTORNO_AMPLE.md`, §6.1, §7.1 y §9.1)
- `extract_from_linevul_split.py`: arma test/train/valid como se describe
  arriba. Acepta `--test-n`, `--train-n`, `--valid-n`, `--seed`, `--out-dir`,
  `--root`.
- `run_joern_batch_param.sh`: Joern secuencial parametrizado por directorio.
- `run_joern_batch_parallel.sh`: Joern paralelo. Reparte la lista de archivos
  (cada `.c` va entero a un solo worker) y cada worker usa su propia copia
  completa de `joern_extracted/` para no pisarse con `tmp/` ni con la carpeta
  de salida que crea `joern-parse`. Salta los archivos que ya tienen salida.
- `data_processing/cpg_GS_parallel.py`: misma lógica que `cpg_GS.py`, repartida
  con `multiprocessing.Pool`.
- Fix #10 en `data_processing/cpg_GS.py` (línea ~651): se agregó `nargs=3` a
  `--json_files`.

#### Problemas conocidos
- `cpg_GS_parallel.py` usa `pool.imap(..., chunksize=8)`, que entrega los
  resultados en orden. Un archivo lento bloquea la entrega de los demás: en
  train, casi todos los workers quedaron sin trabajo unos 14 min esperando a
  uno (su tiempo de CPU acumulado casi no avanzaba). No se pierde trabajo ni
  se corrompe nada; solo retrasa. `imap_unordered` lo evitaría (no aplicado).
- VSCode: `C_Cpp.intelliSenseEngine: disabled` en `.vscode/settings.json` no se
  aplica en caliente. Si `cpptools-srv2` vuelve a indexar y come RAM, hay que
  hacer "Developer: Reload Window".

### Literatura: qué comparan VELVET, VulBERTa y VulDeBERT
Verificado el 2026-09-30 leyendo los PDFs de Zotero (texto completo y lista de
referencias: 60, 40 y 23 entradas). Ninguno de los tres compara contra LineVul
ni menciona IVDetect.

| Modelo | Año | Baselines / comparaciones |
|---|---|---|
| VELVET (Ding et al.) | 2022 | Infer, FlawFinder, RATS; variantes GGNN / Transformer / Transformer-Sandwich / GREAT / Hoppity |
| VulBERTa (Hanif y Maffeis) | 2022 | VulDeePecker, Draper, ReVeal, muVulDeePecker; benchmarks CodeXGLUE (Devign) y D2A |
| VulDeBERT (Kim et al.) | 2022 | VulDeePecker |

Sí existe una comparación publicada LineVul contra AMPLE, pero es parcial: el
paper de AMPLE (ICSE 2023) compara con LineVul solo en FFMPeg+Qemu y Reveal
(F1 56.54 % y 44.79 % para LineVul frente a 66.94 % y 48.48 % para AMPLE) y no
reporta LineVul sobre Fan et al. El 91 % de F1 de LineVul sobre Fan et al. es de
su propio paper (con su propio split) y el 32.11 % de AMPLE es de la Tabla III
de su paper (con su propio pipeline); juntarlos no es una comparación en
igualdad de condiciones. Verificado el 2026-10-06 contra los PDFs. Esas cifras
son de tablas ajenas; nuestros números salen de corridas propias.

Sobre el uso de Joern: VELVET y Leev (`create_graphs.py`, `joern-export --repr
cpg14`) construyen grafos con Joern igual que AMPLE. VulBERTa lo usa solo para
extraer funciones (tokeniza con Clang, sin grafos). VulDeBERT no usa Joern ni
grafos (code gadgets).

## Pendiente inmediato

(Actualizado 2026-10-28: la prioridad actual es la etapa FFmpeg+QEMU, EXP-05 y
EXP-06 del registro de experimentos. Los puntos de abajo son de la etapa Big-Vul y
se conservan como referencia.)

1. Relanzar el entrenamiento de AMPLE en GPU sobre el dataset completo
   (9,233 / 1,433 / 14,772). Es el bloqueador para tener un número real:

   ```bash
   cd AMPLE-main/AMPLE_code
   rm -rf models/full_test_gpu
   source ../../.venv-ample-gpu-experimental/bin/activate
   export LD_LIBRARY_PATH="$(realpath ../../dgl_build_experimental/dgl/build):$LD_LIBRARY_PATH"
   export HOME=/tmp DGLBACKEND=pytorch
   python3 main.py --dataset full_test_gpu --input_dir ../devign_dataset_full_input/ --batch_size 32
   ```

   Tarda del orden de 1 h (ESTIMACIÓN). El progreso se ve en vivo con
   `tail -f AMPLE_code/models/full_test_gpu/devign_FFmpeg.log` (una línea por
   época). El resultado es la última línea `----Test----`, no las líneas `Test`
   de cada época (evalúan solo una parte del test). El proceso debe quedar
   corriendo: si se apaga la máquina se pierde todo.
2. Decidir cómo manejar la comparabilidad del test (14,772 frente a 18,864;
   ver arriba) antes de redactar la tabla comparativa.
3. Decidir si se sube el tamaño de train por encima de 13k o se declara como
   limitación.
4. Hecho el 2026-10-01: `DOCUMENTACION_ENTORNO_AMPLE.md` y
   `DOCUMENTACION_ENTORNO_LINEVUL.md` actualizadas con los scripts nuevos, el
   fix #10, las corridas y los comandos exactos. Falta añadir el resultado
   final de AMPLE cuando exista. Es requisito de las reglas del proyecto antes de citar
   cualquier número en el artículo.
5. Con el resultado de AMPLE, actualizar:
   - Sección IV del artículo con la tabla comparativa LineVul vs. AMPLE.
   - Conclusiones del artículo (hoy preliminares, sin experimentos; ver
     `conclusiones-preliminares.md` en la memoria del proyecto del artículo).
6. Decidir si se reporta en el artículo el hallazgo de que GPU sí es viable
   para AMPLE compilando dgl desde source (la documentación previa decía que
   no): es un resultado reproducible con evidencia.

### Resto de modelos (Leev, ReVeal, VulBERTa, VulDeBERT, VELVET)
Fuera del plan de comparación actual (ver
`~/Documentos/Semestre 2026-2/PFC1/biblio/notas.md` para el historial de
por qué se descartaron o quedaron en pausa). No tocar sin que el alumno lo
pida explícitamente.
