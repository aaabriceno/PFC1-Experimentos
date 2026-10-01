"""
Extrae un subset del CSV crudo Fan et al. (MSR_data_cleaned.csv) a archivos .c
individuales + splits train/test/valid en formato JSON, listo para Joern +
cpg_GS.py de AMPLE.

Uso: python3 extract_subset.py --n 500
"""
import argparse
import csv
import json
import os
import random
import sys

csv.field_size_limit(sys.maxsize)

SHARED_CSV = os.path.expanduser(
    "~/Documentos/docker-labs/vulnerability-detection/shared_datasets/fan_et_al/MSR_data_cleaned.csv"
)
RAW_CODE_DIR = "devign_dataset/devign_raw_code"
SPLIT_DIR = "devign_dataset/devign_data_split"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="cantidad de funciones a extraer")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(RAW_CODE_DIR, exist_ok=True)
    os.makedirs(SPLIT_DIR, exist_ok=True)

    # El CSV NO esta mezclado: las primeras ~10000+ filas son todas vul=0
    # (confirmado empiricamente) - una ventana ciega de las primeras N*20 filas
    # nunca encuentra positivos. Reservoir sampling separado por clase mientras
    # se hace streaming del CSV completo, para garantizar ambas clases en el
    # subset sin cargar los 10.7GB en memoria de una vez.
    n_per_class = args.n // 2
    reservoir_0, reservoir_1 = [], []
    seen_0, seen_1 = 0, 0
    with open(SHARED_CSV, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("vul", "").strip()
            if label == "0":
                seen_0 += 1
                if len(reservoir_0) < n_per_class:
                    reservoir_0.append(row)
                else:
                    j = random.randint(0, seen_0 - 1)
                    if j < n_per_class:
                        reservoir_0[j] = row
            elif label == "1":
                seen_1 += 1
                if len(reservoir_1) < n_per_class:
                    reservoir_1.append(row)
                else:
                    j = random.randint(0, seen_1 - 1)
                    if j < n_per_class:
                        reservoir_1[j] = row

    rows = reservoir_0 + reservoir_1
    random.shuffle(rows)

    entries = []
    for idx, row in enumerate(rows):
        func = row.get("func_before", "").strip()
        label = row.get("vul", "0").strip()
        if not func or label not in ("0", "1"):
            continue
        file_name = f"func_{idx}_{label}"
        c_path = os.path.join(RAW_CODE_DIR, file_name + ".c")
        with open(c_path, "w", encoding="utf-8") as cf:
            cf.write(func)
        entries.append({"file_path": file_name + ".c", "label": label, "code": func})

    random.shuffle(entries)
    n = len(entries)
    n_train = int(n * 0.7)
    n_valid = int(n * 0.15)
    train = entries[:n_train]
    valid = entries[n_train : n_train + n_valid]
    test = entries[n_train + n_valid :]

    for name, split in [("train", train), ("valid", valid), ("test", test)]:
        with open(os.path.join(SPLIT_DIR, f"{name}_raw_code.json"), "w") as f:
            json.dump(split, f, indent=2)

    print(f"Total extraido: {n} (train={len(train)}, valid={len(valid)}, test={len(test)})")
    print(f".c files en: {RAW_CODE_DIR}")
    print(f"splits en: {SPLIT_DIR}")


if __name__ == "__main__":
    main()
