# Patch de dgl para GPU Blackwell (sm_120)

`dgl-blackwell-sm120.patch` permite compilar [dgl](https://github.com/dmlc/dgl)
desde el código fuente con soporte para la GPU RTX 5060 (arquitectura
Blackwell, sm_120) contra torch 2.11.0+cu128. Las builds precompiladas de dgl
no traen kernels para esa arquitectura.

No incluye el código de dgl, solo los cambios. dgl es Apache-2.0; ver
[../../THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).

- Commit base: el de `BASE_COMMIT.txt` (`3d16000b…`, 2025-08-01).
- Verificado: `git apply --check` pasa sobre una copia limpia de ese commit.
- Afecta 4 archivos: `CMakeLists.txt`, `cmake/modules/CUDA.cmake`,
  `graphbolt/CMakeLists.txt` y `graphbolt/build.sh`.

## Qué contiene

Los dos parches del issue
[dmlc/dgl#7917](https://github.com/dmlc/dgl/issues/7917) (usar METIS y GKlib
del sistema, hacer HugeCTR opcional y forzar `CUDA_ARCH_BIN=120`), más tres
correcciones propias de bugs de dgl que aparecieron al compilar:

1. `graphbolt/build.sh` ejecutaba `make -j` sin límite y agotaba la RAM. Se
   agregó la variable `GRAPHBOLT_MAKE_JOBS`.
2. `graphbolt/CMakeLists.txt`: el filtro de arquitecturas CUDA usaba el regex
   `[2-6][0-9]` sin anclar, y filtraba `120` por contener `20`. Se anclaron
   los extremos.
3. `CMakeLists.txt`: el directorio de includes de `nv_gpu_cache.hpp` solo se
   agregaba con `BUILD_HUGECTR=ON`, pero `src/runtime/cuda/gpu_cache.cu` se
   compila siempre. Sin esto, `libdgl.so` quedaba con símbolos sin resolver.

Detalle y contexto en la sección 3.1 de
[../../AMPLE-main/DOCUMENTACION_ENTORNO_AMPLE.md](../../AMPLE-main/DOCUMENTACION_ENTORNO_AMPLE.md).

## Cómo aplicarlo

Requiere CUDA toolkit 13.2, torch 2.11.0+cu128 y GKlib y METIS compilados en un
prefijo local.

```bash
git clone --recurse-submodules https://github.com/dmlc/dgl.git
cd dgl
git checkout 3d16000b4170fa741ed9e9667f22ba84d3493026
git apply /ruta/a/dgl-blackwell-sm120.patch

# Dos archivos que el issue pide desactivar (no van en el patch):
mv src/graph/metis_partition.cc src/graph/metis_partition.cc.disabled
mv src/graph/transform/metis_partition_hetero.cc src/graph/transform/metis_partition_hetero.cc.disabled

mkdir build && cd build
export CUDAARCHS=120 GRAPHBOLT_MAKE_JOBS=1
cmake .. -DUSE_CUDA=ON -DUSE_OPENMP=ON -DBUILD_TORCH=ON -DUSE_METIS=ON \
  -DBUILD_HUGECTR=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=<prefijo con GKlib y METIS>
make graphbolt tensoradapter_pytorch -j1
make dgl -j1
cd ../python && pip install -e .
```

`-j1` y `GRAPHBOLT_MAKE_JOBS=1` evitan agotar la RAM (máquina con 30 GB). En
esta máquina la compilación tardó del orden de una hora en total. Se
construyeron solo esos tres targets; no hace falta `runUnitTests` (tests de
C++). Con `BUILD_HUGECTR=OFF` ese target falló al enlazar, y con `ON` no se
probó.

Para probarlo, correr desde un directorio fuera del clon de dgl (si no, Python
resuelve `import dgl` a un paquete vacío), con `LD_LIBRARY_PATH` apuntando a
`dgl/build`.
