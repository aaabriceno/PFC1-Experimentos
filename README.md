# PFC1 - Experimentos

Evidencia de los experimentos del PFC1 (UCSP, curso CS402) de Anthony Briceño
Quiroz. Consiste en ejecutar y comparar dos modelos de detección de
vulnerabilidades en código C/C++, **LineVul** y **AMPLE**, sobre el mismo
dataset fuente (Fan et al. / Big-Vul, también llamado MSR_20), en una laptop
con GPU RTX 5060 (arquitectura Blackwell, sm_120).

> **El código de LineVul y de AMPLE no es mío.** Pertenece a sus autores y
> conserva su licencia, o su falta de licencia. Está en este repositorio solo
> como evidencia académica de la reproducción. Detalle de origen, licencia y
> modificaciones en [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Qué hay aquí

| Ruta | Qué es | Autoría |
|---|---|---|
| `LineVul-main/` | LineVul sin modificar, más Dockerfile, requirements, documentación y evidencia de las corridas | Código: Fu y Tantithamthavorn (MIT). Agregados: propios |
| `AMPLE-main/` | AMPLE con 7 archivos modificados para que corra con Python 3.12 / torch 2.x / dgl 2.x y sin GPU fija, más scripts propios | Código original: autores de AMPLE (sin licencia). Agregados: propios |
| `patches/dgl/` | Patch para compilar dgl con soporte de GPU Blackwell (no incluye el código de dgl) | Derivado de dgl (Apache-2.0) |
| `environment/` | Lista de paquetes del entorno GPU experimental | Propia |
| `PROGRESS.md` | Bitácora detallada: cada número con su fuente | Propia |

Cada modelo tiene su propia documentación del entorno, con los problemas de
compatibilidad encontrados y cómo se resolvieron:

- [LineVul-main/DOCUMENTACION_ENTORNO_LINEVUL.md](LineVul-main/DOCUMENTACION_ENTORNO_LINEVUL.md)
- [AMPLE-main/DOCUMENTACION_ENTORNO_AMPLE.md](AMPLE-main/DOCUMENTACION_ENTORNO_AMPLE.md)

## Qué no está incluido

Este repositorio solo tiene código, documentación y logs (unos 5 MB). Se dejó
fuera a propósito todo lo que pesa o no es mío:

- **Datasets**: `MSR_data_cleaned.csv` (~10.7 GB), el `test.csv` de LineVul
  (~1 GB) y los datasets derivados que generó el pipeline (grafos, CSV de
  Joern, etc.).
- **Pesos de modelos**: el checkpoint `12heads_linevul_model.bin` (~950 MB),
  los modelos word2vec preentrenados de AMPLE y los checkpoints `.bin` de las
  corridas de AMPLE.
- **Binarios y entornos**: Joern, los entornos virtuales, el clon y la carpeta
  de compilación de dgl, cachés de HuggingFace y `__pycache__`.

Los datasets y los pesos originales se obtienen siguiendo los README de cada
proyecto:
[LineVul](https://github.com/awsm-research/LineVul) y
[AMPLE](https://github.com/AMPLE001/AMPLE).

## Resultados

Cada número sale de un log o captura incluida en el repositorio. El detalle
completo está en [PROGRESS.md](PROGRESS.md).

**LineVul** (test completo de 18,864 funciones, checkpoint preentrenado del
paper, GPU). Se corrió dos veces, el 2026-07-09 y el 2026-09-30, con
resultados idénticos:

| Métrica | Valor |
|---|---|
| Accuracy | 0.9909 |
| F1 | 0.9142 |
| Precision | 0.9712 |
| Recall | 0.8635 |

Fuentes: `LineVul-main/resultados_gpu_test.log` y
`LineVul-main/evidenciasLineVul/testPruebaLineVul.png`.

**AMPLE**: el preprocesamiento a escala completa está terminado, pero **todavía
no hay resultado final de entrenamiento**. Las corridas con subsets chicos
(hasta 368 funciones de test) solo validan que el pipeline funciona y miden
tiempos; no son resultados del modelo. Tampoco son comparables todavía con los
de LineVul, porque el filtro de grafos deja el test de AMPLE en 14,772 de las
18,864 funciones (ver `PROGRESS.md`).

## Cómo reproducir

1. Conseguir los datasets y pesos indicados arriba y colocarlos donde
   describen las documentaciones de cada modelo.
2. LineVul: imagen Docker `linevul:cu128` (ver su documentación, sección 10,
   para el comando exacto).
3. AMPLE: para usar GPU Blackwell hay que compilar dgl con el patch de
   `patches/dgl/` (instrucciones en su README). En CPU basta con la imagen
   `ample-env` del `Dockerfile`.

## Aviso

- Este trabajo es académico y no tiene fines comerciales. No tiene relación con
  los autores de LineVul, AMPLE o dgl, ni cuenta con su respaldo.
- El código de terceros se rige por sus propias licencias. Para AMPLE no hay
  licencia declarada: la copia incluida no otorga ningún derecho sobre ese
  código y no debe redistribuirse. Si algún autor pide que se retire, se
  elimina.
- Se entrega tal cual, sin garantías. Los modelos de detección de
  vulnerabilidades son material de investigación y no sustituyen una auditoría
  de seguridad.

## Cómo citar los trabajos originales

Las referencias completas están en [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Si usas estos resultados, cita los papers de LineVul, AMPLE y del dataset de
Fan et al., no este repositorio.
