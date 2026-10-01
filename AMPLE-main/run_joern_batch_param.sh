#!/bin/bash
# Version parametrizada de run_joern_batch.sh: acepta el directorio dataset
# como argumento en vez de tener devign_dataset/ hardcoded, para poder correr
# sobre devign_dataset_timing/, devign_dataset_full/, etc. sin pisar el subset
# original n=500 (devign_dataset/). Misma logica interna, sin cambios.
set -e

DATASET_DIR="${1:?Uso: run_joern_batch_param.sh <dataset_dir>}"

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOERN_DIR="$BASE/joern_bin/joern_extracted/joern/joern"
RAW_DIR="$BASE/$DATASET_DIR/devign_raw_code"
OUT_DIR="$BASE/$DATASET_DIR/devign_parsed"

mkdir -p "$OUT_DIR"
cd "$JOERN_DIR"

count=0
total=$(ls "$RAW_DIR"/*.c | wc -l)
for cfile in "$RAW_DIR"/*.c; do
    fname=$(basename "$cfile")
    name_noext="${fname%.c}"
    count=$((count+1))

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

    if (( count % 50 == 0 )); then
        echo "[$count/$total] procesado: $fname"
    fi
done

echo "Listo. $count archivos procesados."
