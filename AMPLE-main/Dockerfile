# AMPLE - entorno de replicacion para GPU Blackwell (RTX 5060)
#
# Mismo patron que LineVul/Dockerfile: la imagen trae solo el entorno, el
# codigo/dataset/grafos viven en el host y se montan con "-v $(pwd):/app".
#
# torch==1.9.0 y dgl==0.7.2 (pins originales del paper) no tienen wheel para
# Python 3.12 y no traen kernels para Blackwell (sm_120). Se usa el mismo
# truco que en LineVul: torch/dgl modernos en capa aparte, resto de deps con
# pin mas cercano posible al original. dgl solo publica wheel Linux/py3.12
# para cu121 (no cu128 como LineVul) -> torch tambien se instala en build
# cu121 aqui para que ambos usen las mismas libs CUDA en runtime. El driver
# 595.84 (CUDA 13.2) soporta cu121 por retrocompatibilidad.
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# Capa 1: torch + dgl, ambos build cu121 para que compartan runtime CUDA.
# dgl.ai (data.dgl.ai, CloudFront) devuelve 403 Access Denied en los wheels
# con sufijo +cuXXX (bloqueo de bucket, no problema de red local - probado
# con distintos user-agents y sigue denegando) - se usa el paquete generico
# de PyPI en su lugar, que autodetecta el runtime CUDA de torch en el import.
#
# CONFIRMADO EMPIRICAMENTE (7 ago 2026): no existe combinacion torch+dgl con
# ABI compatible entre si que TAMBIEN tenga kernels para Blackwell (sm_120).
# - dgl 2.1.0 (PyPI generico) + torch >=2.4 con soporte sm_120: crash al
#   dlopen ("getCount is non-monotonic", en realidad ABI mismatch).
# - dgl 2.4.0+cu124 (indice propio de dgl.ai) + torch 2.4.0+cu121: importa y
#   crea grafos en GPU sin crash, pero CUALQUIER operacion de computo real
#   (dgl.nn.GraphConv.forward, hasta un simple in_degrees()/arange interno)
#   falla con "CUDA error: no kernel image is available for execution on the
#   device" - torch 2.4.x no tiene kernels sm_120, el warning al importar
#   NO era falso positivo.
# - torch >=2.7+cu128 (el que SI soporta Blackwell, usado en LineVul) no
#   tiene ninguna build de dgl compatible en ABI - dgl.ai no publico nada mas
#   nuevo que 2.4.0 con wheels Linux/cu12x, y ese esta atado a torch 2.4.x.
# Conclusion: AMPLE corre en CPU aqui. GPU queda bloqueado mientras el
# proyecto dgl (practicamente sin actividad desde 2024) no publique un build
# contra torch >=2.7.
#
# dgl 2.1.0 tampoco es plug-and-play con cualquier torch en CPU: su modulo
# graphbolt busca en runtime un .so nombrado "libgraphbolt_pytorch_<version
# exacta>.so" y el wheel solo trae los de torch 2.0.0/2.0.1/2.1.0/2.1.1/
# 2.1.2/2.2.0/2.2.1 (confirmado listando el paquete instalado) - con
# torch 2.5.1 tira FileNotFoundError buscando un .so que no existe. Se fija
# torch==2.2.1 (la mas nueva de las soportadas) en build CPU puro.
RUN pip install --no-cache-dir torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir dgl==2.1.0 packaging torchdata==0.7.1 pyyaml pydantic

# gensim==4.3.2 no publica wheel precompilado para py3.12/linux -> pip cae a
# compilar sus extensiones C (word2vec_inner.c) desde source, necesita gcc.
# graphviz Python necesita el binario 'dot' del sistema aparte del paquete pip.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential graphviz \
    && rm -rf /var/lib/apt/lists/*

# Capa 2: resto de dependencias. graphviz/nltk/gensim no estan en el README
# de AMPLE pero el codigo real los importa (utils.py, word2vec.py) -
# confirmado leyendo data_processing/*.py y AMPLE_code/*.py.
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.5.0 \
    scipy==1.12.0 \
    tqdm \
    gensim==4.3.2 \
    graphviz==0.20.3 \
    nltk==3.8.1

WORKDIR /app

CMD ["bash"]
