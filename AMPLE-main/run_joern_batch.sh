#!/bin/bash
# Corre joern-parse sobre cada .c en devign_raw_code, deja nodes.csv/edges.csv
# en devign_parsed/<nombre_sin_ext>/ (formato que espera cpg_GS.py --csv).
set -e

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOERN_DIR="$BASE/joern_bin/joern_extracted/joern/joern"
JOERN_BIN="$JOERN_DIR/joern/joern-parse"
RAW_DIR="$BASE/devign_dataset/devign_raw_code"
OUT_DIR="$BASE/devign_dataset/devign_parsed"

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
