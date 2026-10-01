"""
Extrae funciones del CSV crudo Fan et al. (MSR_data_cleaned.csv) usando EL MISMO
split de test que ya uso LineVul (LineVul-main/data/big-vul_dataset/test.csv),
para que la comparacion entre ambos modelos sea funcion-por-funcion y no solo
"mismo dataset de origen".

Como funciona el join: test.csv de LineVul tiene una columna 'index' que es la
posicion de fila original en MSR_data_cleaned.csv (columna sin nombre, primera
columna del CSV crudo). Verificado manualmente: test.csv fila con index=73752
tiene el mismo func_before y mismo label que la fila 73752 de
MSR_data_cleaned.csv. Por eso se puede recuperar el test set exacto de LineVul
leyendo MSR_data_cleaned.csv una vez y filtrando por ese set de indices, en vez
de re-samplear al azar.

AMPLE no tiene train.csv/val.csv propios en este repo (solo test.csv quedo
disponible localmente de la corrida de LineVul), asi que train/valid de AMPLE
se generan por separado con reservoir sampling estratificado sobre las filas
de MSR_data_cleaned.csv que NO estan en el test set de LineVul (para no
contaminar el test set con filas usadas en train/valid).

Uso:
  python3 extract_from_linevul_split.py --test-n 0 --train-n 2000 --valid-n 400
    --test-n 0 (default) usa el test set COMPLETO de LineVul (18,864 funciones).
    --test-n N usa un subset estratificado de N funciones del test set de
    LineVul (sigue siendo un subconjunto real, join exacto por index).
"""
import argparse
import csv
import json
import os
import random
import sys

import pandas as pd

csv.field_size_limit(sys.maxsize)

# Paths relativos a /vulndet (mount del directorio padre vulnerability-detection/
# dentro del container ample-env; ver DOCUMENTACION_ENTORNO_AMPLE.md para el
# comando docker run exacto). Con --root se puede apuntar a otra raiz (ej. para
# correr el script fuera de Docker apuntando al host directamente).
ap_root_default = "/vulndet"
SHARED_CSV_REL = "shared_datasets/fan_et_al/MSR_data_cleaned.csv"
LINEVUL_TEST_CSV_REL = "LineVul-main/data/big-vul_dataset/test.csv"


def stratified_subset(indices_labels, n, seed):
    """indices_labels: list of (index, label). Devuelve subset balanceado de tamano n."""
    if n <= 0 or n >= len(indices_labels):
        return indices_labels
    rng = random.Random(seed)
    by_label = {}
    for idx, label in indices_labels:
        by_label.setdefault(label, []).append((idx, label))
    n_per_class = n // len(by_label)
    out = []
    for label, items in by_label.items():
        rng.shuffle(items)
        out.extend(items[:n_per_class])
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-n", type=int, default=0,
                     help="0 = test set completo de LineVul (18864). N>0 = subset estratificado de N.")
    ap.add_argument("--train-n", type=int, default=2000)
    ap.add_argument("--valid-n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="devign_dataset_full",
                     help="Directorio nuevo para no pisar devign_dataset/ (subset n=500 anterior).")
    ap.add_argument("--root", type=str, default=ap_root_default,
                     help="Raiz de vulnerability-detection/ (default /vulndet dentro del container).")
    args = ap.parse_args()

    shared_csv = os.path.join(args.root, SHARED_CSV_REL)
    linevul_test_csv = os.path.join(args.root, LINEVUL_TEST_CSV_REL)

    raw_dir = os.path.join(args.out_dir, "devign_raw_code")
    split_dir = os.path.join(args.out_dir, "devign_data_split")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    print("Leyendo test.csv de LineVul para obtener el set de indices de test...")
    linevul_test = pd.read_csv(linevul_test_csv)
    test_indices_labels = list(zip(linevul_test["index"].astype(str), linevul_test["target"].astype(str)))
    test_index_set = set(i for i, _ in test_indices_labels)
    print(f"  test.csv de LineVul: {len(test_indices_labels)} funciones, "
          f"{sum(1 for _, l in test_indices_labels if l == '1')} vulnerables.")

    if args.test_n > 0:
        test_indices_labels = stratified_subset(test_indices_labels, args.test_n, args.seed)
        test_index_set = set(i for i, _ in test_indices_labels)
        print(f"  Subset de test estratificado: {len(test_indices_labels)} funciones.")

    print("Streaming MSR_data_cleaned.csv completo (10.7GB) para: "
          "(a) recuperar func_before de las filas de test, (b) muestrear train/valid "
          "de filas NO usadas en test...")

    reservoir_0, reservoir_1 = [], []
    seen_0, seen_1 = 0, 0
    n_per_class_train_valid = (args.train_n + args.valid_n) // 2
    rng = random.Random(args.seed)

    test_rows = {}  # index -> {func_before, label}

    with open(shared_csv, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_index = row.get("", "").strip()
            label = row.get("vul", "").strip()

            if row_index in test_index_set:
                test_rows[row_index] = {
                    "func": row.get("func_before", "").strip(),
                    "label": label,
                }
                continue  # filas de test no entran al pool de train/valid

            if label not in ("0", "1"):
                continue
            func = row.get("func_before", "").strip()
            if not func:
                continue

            if label == "0":
                seen_0 += 1
                if len(reservoir_0) < n_per_class_train_valid:
                    reservoir_0.append((row_index, func, label))
                else:
                    j = rng.randint(0, seen_0 - 1)
                    if j < n_per_class_train_valid:
                        reservoir_0[j] = (row_index, func, label)
            else:
                seen_1 += 1
                if len(reservoir_1) < n_per_class_train_valid:
                    reservoir_1.append((row_index, func, label))
                else:
                    j = rng.randint(0, seen_1 - 1)
                    if j < n_per_class_train_valid:
                        reservoir_1[j] = (row_index, func, label)

    print(f"  Filas de test recuperadas del CSV crudo: {len(test_rows)}/{len(test_index_set)} "
          f"(algunas pueden faltar si el index no aparece en MSR_data_cleaned.csv).")

    missing = test_index_set - set(test_rows.keys())
    if missing:
        print(f"  ADVERTENCIA: {len(missing)} indices de test.csv no se encontraron en "
              f"MSR_data_cleaned.csv, se excluyen del test set de AMPLE.")

    train_valid_pool = reservoir_0 + reservoir_1
    rng.shuffle(train_valid_pool)
    train_pool = train_valid_pool[:args.train_n]
    valid_pool = train_valid_pool[args.train_n: args.train_n + args.valid_n]

    def write_split(name, items_are_test_rows, seq):
        entries = []
        for i, item in enumerate(seq):
            if items_are_test_rows:
                idx, data = item
                func, label = data["func"], data["label"]
            else:
                idx, func, label = item
            if not func or label not in ("0", "1"):
                continue
            file_name = f"{name}_{idx}_{label}"
            c_path = os.path.join(raw_dir, file_name + ".c")
            with open(c_path, "w", encoding="utf-8") as cf:
                cf.write(func)
            entries.append({"file_path": file_name + ".c", "label": label, "code": func})
        with open(os.path.join(split_dir, f"{name}_raw_code.json"), "w") as f:
            json.dump(entries, f, indent=2)
        return len(entries)

    n_test = write_split("test", True, list(test_rows.items()))
    n_train_written = write_split("train", False, train_pool)
    n_valid_written = write_split("valid", False, valid_pool)

    print(f"Total: train={n_train_written}, valid={n_valid_written}, test={n_test}")
    print(f".c files en: {raw_dir}")
    print(f"splits en: {split_dir}")


if __name__ == "__main__":
    main()
