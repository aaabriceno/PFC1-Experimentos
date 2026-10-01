# LineVul - entorno de replicacion para GPU Blackwell (RTX 5060)
#
# Filosofia: la imagen contiene SOLO el entorno (Python + librerias).
# El codigo, dataset, modelo y resultados viven en el host y se montan
# con "-v $(pwd):/app" al ejecutar. Ventajas:
#   - editar codigo sin reconstruir la imagen
#   - los ~2 GB de dataset/modelo no engordan la imagen
#   - los resultados quedan en tu disco al salir del contenedor
#
# Base python:3.10-slim y no una imagen nvidia/cuda porque:
#   - tokenizers==0.13.2 y pandas==1.5.2 (pins del paper) publican wheels
#     precompilados para CPython 3.10; en 3.12 tocaria compilar desde
#     fuente (Rust/C) y en 3.12 torch 1.x ni existe.
#   - el wheel de torch cu128 ya empaqueta sus propias librerias CUDA
#     (nvidia-cublas-cu12, cudnn, etc.), asi que no hace falta el toolkit
#     CUDA en la imagen. Solo se necesita el DRIVER del host, que entra
#     al contenedor via nvidia-container-toolkit (--gpus all).
FROM python:3.10-slim

WORKDIR /app

# El pip que trae python:3.10-slim (23.0.x) rechaza los wheels del indice de
# PyTorch: "has inconsistent Name: expected 'typing-extensions', but metadata
# has 'typing_extensions'". Es un bug de normalizacion de nombres corregido
# en pip >= 23.3, asi que se actualiza pip antes de instalar nada.
RUN pip install --no-cache-dir --upgrade pip

# Capa 1 (la mas pesada, ~3 GB): torch con kernels sm_120.
# torch >= 2.7 + build cu128 es el primer binario oficial que soporta
# Blackwell. Va en su propia capa para que Docker la cachee y no se
# re-descargue cuando cambien las demas dependencias.
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Capa 2: resto de dependencias con los pins originales del paper.
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements-docker.txt

# Cache de HuggingFace dentro de /app: al montar el proyecto en /app,
# el modelo microsoft/codebert-base (~500 MB) se descarga UNA sola vez
# y persiste en el host entre ejecuciones.
ENV HF_HOME=/app/.hf_cache

# Al ejecutar con "--user $(id -u)" no hay $HOME escribible dentro del
# contenedor; sin esto matplotlib avisa en cada arranque que no puede crear
# su cache de configuracion.
ENV MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app/linevul

CMD ["bash"]
