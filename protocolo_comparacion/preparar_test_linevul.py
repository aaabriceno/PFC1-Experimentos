#!/usr/bin/env python3
"""Prepara el test de FFmpeg+QEMU para LineVul.

Al terminar el test, LineVul calcula estadisticas de localizacion de linea y
exige las columnas `flaw_line` y `flaw_line_index` (linevul_main.py,
generate_result_df). FFmpeg+QEMU solo trae etiqueta por funcion, asi que se
agregan las dos columnas VACIAS. Con ellas vacias el test llega hasta el final y
escribe las predicciones (comprobado en una prueba corta, 2026-10-25).

No cambia ninguna funcion ni el orden de las filas: asi las predicciones de
LineVul (raw_preds) se pueden alinear fila a fila con las de AMPLE por `id`.
El test.csv original no se modifica.

Uso:
  python3 preparar_test_linevul.py \
      --entrada ../conjuntos_de_datos/ffmpeg_qemu/split_semilla42/test.csv \
      --salida  ../conjuntos_de_datos/ffmpeg_qemu/split_semilla42/test_linevul.csv
"""
import argparse
import csv
import sys

csv.field_size_limit(sys.maxsize)
COLUMNAS_NUEVAS = ["flaw_line", "flaw_line_index"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", required=True)
    ap.add_argument("--salida", required=True)
    args = ap.parse_args()

    with open(args.entrada, newline="", encoding="utf-8") as fin:
        lector = csv.DictReader(fin)
        campos = lector.fieldnames
        assert not set(COLUMNAS_NUEVAS) & set(campos), "el CSV ya tiene esas columnas"
        filas = list(lector)

    with open(args.salida, "w", newline="", encoding="utf-8") as fout:
        escritor = csv.DictWriter(fout, fieldnames=campos + COLUMNAS_NUEVAS)
        escritor.writeheader()
        for fila in filas:
            escritor.writerow({**fila, **{c: "" for c in COLUMNAS_NUEVAS}})

    print(f"{len(filas)} filas; columnas: {campos + COLUMNAS_NUEVAS}")


if __name__ == "__main__":
    main()
