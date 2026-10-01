#!/bin/bash
# Version paralela de run_joern_batch_param.sh.
#
# Cada archivo .c es una unidad de trabajo COMPLETA e independiente (nunca se
# parte un archivo entre workers) - lo que se distribuye entre workers es la
# LISTA de archivos, no el contenido de ninguno de ellos.
#
# Aislamiento: cada worker corre en su propia copia completa del directorio
# de Joern (joern_extracted/), no solo su propio tmp/. Esto es necesario
# porque joern-parse crea, ademas de tmp/, una carpeta de salida nombrada
# como el archivo (ej. "test_165531_0.c/") al lado de tmp/ - si dos workers
# compartieran el mismo directorio de Joern, podrian pisarse esa carpeta de
# salida entre si aunque cada uno tenga su propio tmp/. Con una copia
# completa por worker, dos workers nunca tocan el mismo archivo en disco.
#
# Salta archivos que ya tienen output en devign_parsed/ (igual que el
# script secuencial), asi que se puede correr de nuevo sobre lo que falta
# sin reprocesar lo ya hecho.
set -e

DATASET_DIR="${1:?Uso: run_joern_batch_parallel.sh <dataset_dir> [n_workers]}"
N_WORKERS="${2:-28}"

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOERN_SRC="$BASE/joern_bin/joern_extracted"
# Acepta DATASET_DIR relativo (a este script) o absoluto.
if [[ "$DATASET_DIR" = /* ]]; then
    DATASET_ABS="$DATASET_DIR"
else
    DATASET_ABS="$BASE/$DATASET_DIR"
fi
RAW_DIR="$DATASET_ABS/devign_raw_code"
OUT_DIR="$DATASET_ABS/devign_parsed"
WORK_ROOT="$DATASET_ABS/.joern_parallel_work"

mkdir -p "$OUT_DIR"
rm -rf "$WORK_ROOT"
mkdir -p "$WORK_ROOT"

echo "Preparando $N_WORKERS copias aisladas de Joern (una por worker)..."
for i in $(seq 0 $((N_WORKERS - 1))); do
    cp -r "$JOERN_SRC" "$WORK_ROOT/worker_$i"
done

# Lista de archivos pendientes (salta los que ya tienen salida).
PENDING_LIST="$WORK_ROOT/pending_files.txt"
> "$PENDING_LIST"
for cfile in "$RAW_DIR"/*.c; do
    fname=$(basename "$cfile")
    if [[ ! -f "$OUT_DIR/$fname/nodes.csv" ]]; then
        echo "$cfile" >> "$PENDING_LIST"
    fi
done

total_pending=$(wc -l < "$PENDING_LIST")
echo "Archivos pendientes: $total_pending"

if [[ "$total_pending" -eq 0 ]]; then
    echo "Nada que hacer, todos los archivos ya tienen salida."
    rm -rf "$WORK_ROOT"
    exit 0
fi

process_worker() {
    local worker_id="$1"
    local worker_dir="$WORK_ROOT/worker_$worker_id/joern/joern"
    local file_list="$2"
    local count=0
    local total
    total=$(wc -l < "$file_list")

    cd "$worker_dir"
    while IFS= read -r cfile; do
        fname=$(basename "$cfile")
        count=$((count + 1))

        rm -rf tmp "$fname"
        mkdir -p tmp
        cp "$cfile" "tmp/$fname"

        ./joern/joern-parse tmp "$fname" > /dev/null 2>&1 || true

        if [[ -f "$fname/tmp/$fname/nodes.csv" ]]; then
            mkdir -p "$OUT_DIR/$fname"
            cp "$fname/tmp/$fname/nodes.csv" "$OUT_DIR/$fname/nodes.csv"
            cp "$fname/tmp/$fname/edges.csv" "$OUT_DIR/$fname/edges.csv"
        fi

        rm -rf "tmp" "$fname"

        if (( count % 20 == 0 )); then
            echo "[worker $worker_id] [$count/$total] procesado: $fname"
        fi
    done < "$file_list"
    echo "[worker $worker_id] terminado, $count archivos."
}
export -f process_worker
export WORK_ROOT OUT_DIR

# Reparte la lista de pendientes en N_WORKERS listas, sin solaparse.
split -n l/"$N_WORKERS" -d -a 2 "$PENDING_LIST" "$WORK_ROOT/split_"

pids=()
i=0
for split_file in "$WORK_ROOT"/split_*; do
    process_worker "$i" "$split_file" &
    pids+=($!)
    i=$((i + 1))
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "Limpiando copias de trabajo de Joern..."
rm -rf "$WORK_ROOT"

final_count=$(ls "$OUT_DIR" | wc -l)
echo "Listo. Total en $OUT_DIR: $final_count archivos."
