#!/usr/bin/env python3
"""Convierte train/val/test.csv (salida de dividir_ffmpeg_qemu.py) al formato
de entrada de AMPLE: un .c por funcion y un JSON por particion.

AMPLE llama "valid" a lo que aqui es "val". Archivo .c: <particion>_<id>_<etiqueta>.c
(cpg_GS.py lee la etiqueta del final del nombre). El JSON lleva el campo `code`,
que exigen word2vec.py y cpg_GS.py.

Uso:
  python3 convertir_a_ample.py --split ../conjuntos_de_datos/ffmpeg_qemu/split_semilla42 \
      --salida ../AMPLE-main/devign_dataset_ffmpeg_qemu
"""
import argparse
import csv
import json
import os
import sys

csv.field_size_limit(sys.maxsize)
NOMBRE_AMPLE = {"train": "train", "val": "valid", "test": "test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--salida", required=True)
    args = ap.parse_args()

    raw = os.path.join(args.salida, "devign_raw_code")
    div = os.path.join(args.salida, "devign_data_split")
    os.makedirs(raw, exist_ok=True)
    os.makedirs(div, exist_ok=True)
    for origen, destino in NOMBRE_AMPLE.items():
        entradas = []
        with open(os.path.join(args.split, f"{origen}.csv"), newline="", encoding="utf-8") as fh:
            for fila in csv.DictReader(fh):
                etiqueta = fila["target"]
                assert etiqueta in ("0", "1"), etiqueta
                nombre = f"{destino}_{fila['id']}_{etiqueta}"
                with open(os.path.join(raw, nombre + ".c"), "w", encoding="utf-8") as cf:
                    cf.write(fila["processed_func"])
                entradas.append({"file_path": nombre + ".c", "label": etiqueta,
                                 "code": fila["processed_func"]})
        with open(os.path.join(div, f"{destino}_raw_code.json"), "w") as f:
            json.dump(entradas, f, indent=2)
        v = sum(e["label"] == "1" for e in entradas)
        print(f"{destino}: {len(entradas)} funciones, {v} vulnerables")


if __name__ == "__main__":
    main()
