# PROGRESS.md — vulnerability-detection (bitácora de experimentos)

Última actualización: 2026-10-01.

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

Sí existe una comparación publicada LineVul contra AMPLE, pero está en el
propio paper de AMPLE (ICSE 2023, Tabla III; ver el `PROGRESS.md` de PFC1).
Sobre Fan et al. reporta F1 91.0 % para LineVul y 32.11 % para AMPLE. Esa cifra
es de una tabla ajena; nuestros números salen de corridas propias.

Sobre el uso de Joern: VELVET y Leev (`create_graphs.py`, `joern-export --repr
cpg14`) construyen grafos con Joern igual que AMPLE. VulBERTa lo usa solo para
extraer funciones (tokeniza con Clang, sin grafos). VulDeBERT no usa Joern ni
grafos (code gadgets).

## Pendiente inmediato

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
