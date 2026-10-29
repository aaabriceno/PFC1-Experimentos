# Avisos de terceros

Este repositorio incluye o deriva de trabajo de otras personas. Este archivo
dice qué es de quién, bajo qué licencia, y qué se cambió. Los datos de
licencia se verificaron el 2026-10-01 consultando la API de GitHub y los
archivos de cada repositorio original.

## 1. Componentes incluidos

| Carpeta | Origen | Versión comparada | Licencia |
|---|---|---|---|
| `LineVul-main/` | https://github.com/awsm-research/LineVul | commit `9401ec6` (2024-08-06) | MIT |
| `AMPLE-main/` | https://github.com/AMPLE001/AMPLE | commit `688776e` (2022-11-19) | **Ninguna declarada** |

### LineVul (MIT)

Texto de la licencia en `LineVul-main/LICENSE`, que debe conservarse con
cualquier copia del código. Titular según ese archivo: "anon-ai-research".

**Archivos originales modificados: ninguno.** Se comparó cada archivo contra el
commit indicado y el código de LineVul es idéntico al original. Se
agregaron (propios): `Dockerfile`, `requirements-docker.txt`,
`DOCUMENTACION_ENTORNO_LINEVUL.md`, `resultados_gpu_test.log` y
`evidenciasLineVul/`.

### AMPLE (sin licencia)

El repositorio original no tiene archivo de licencia y GitHub no detecta
ninguna. Sin licencia, por defecto todos los derechos pertenecen a los
autores, y terceros no tienen permiso de copiar, modificar ni redistribuir el
código. Solo como evidencia académica del trabajo de reproducción. No se concede
ningún derecho sobre ese código. Si los autores lo solicitan, se retira.

**Archivos originales modificados (7):** cambios de compatibilidad con
Python 3.12, torch 2.x y dgl 2.x, y para no fijar la GPU (`cuda:0`). El detalle
línea por línea está en la sección 9 de
`AMPLE-main/DOCUMENTACION_ENTORNO_AMPLE.md`.

- `AMPLE_code/data_loader/batch_graph.py`
- `AMPLE_code/data_loader/dataset.py`
- `AMPLE_code/graph_transformer_layers.py`
- `AMPLE_code/main.py`
- `AMPLE_code/modules/model.py`
- `AMPLE_code/trainer.py`
- `data_processing/cpg_GS.py` (agrega `nargs=3` a `--json_files`)

**Archivos agregados (propios):** `Dockerfile`, `DOCUMENTACION_ENTORNO_AMPLE.md`,
`extract_subset.py`, `extract_from_linevul_split.py`, `run_joern_batch.sh`,
`run_joern_batch_param.sh`, `run_joern_batch_parallel.sh`,
`data_processing/cpg_GS_parallel.py` y los logs de entrenamiento en
`AMPLE_code/models/*/devign_FFmpeg.log`.

## 2. Obras derivadas

### dgl (Apache-2.0)

`patches/dgl/dgl-blackwell-sm120.patch` modifica 4 archivos de
https://github.com/dmlc/dgl (commit base en `patches/dgl/BASE_COMMIT.txt`,
licencia Apache-2.0). Reúne los dos parches publicados en el issue
[dmlc/dgl#7917](https://github.com/dmlc/dgl/issues/7917) y tres correcciones
propias. No se incluye el código de dgl: solo el patch. Se rige por la
licencia Apache-2.0 de dgl, no por la de este repositorio.

## 3. Lo que se usó pero no se redistribuye

| Recurso | Origen |
|---|---|
| Dataset Fan et al. / Big-Vul (`MSR_data_cleaned.csv`) | https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset |
| `test.csv` y checkpoint `12heads_linevul_model.bin` de LineVul | Enlaces del README de LineVul |
| Datasets y modelos word2vec de AMPLE | Enlaces del README de AMPLE |
| Joern (versión compilada de los autores de AMPLE) | https://zenodo.org/record/7323504 |

## 4. Contribuciones propias

Solo estos archivos están bajo la licencia del archivo `LICENSE` de la raíz:

- `README.md`, `THIRD_PARTY_NOTICES.md`, `PROGRESS.md`
- `environment/`
- Archivos agregados de `LineVul-main/` y `AMPLE-main/` listados en la
  sección 1

No se aplica a LineVul, a AMPLE (incluidos los 7 archivos modificados), ni al
patch de dgl.

## 5. Referencias

```bibtex
@inproceedings{fu2022linevul,
  title={LineVul: A Transformer-based Line-Level Vulnerability Prediction},
  author={Fu, Michael and Tantithamthavorn, Chakkrit},
  booktitle={2022 IEEE/ACM 19th International Conference on Mining Software Repositories (MSR)},
  year={2022},
  organization={IEEE}
}
```

- AMPLE: Wen et al., "Vulnerability Detection with Graph Simplification and
  Enhanced Graph Representation Learning", ICSE 2023.
  arXiv:[2302.04675](https://arxiv.org/abs/2302.04675).
- Dataset: Fan et al., MSR 2020.
  DOI: [10.1145/3379597.3387501](https://dl.acm.org/doi/10.1145/3379597.3387501).
