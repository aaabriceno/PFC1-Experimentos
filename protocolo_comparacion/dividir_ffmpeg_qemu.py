#!/usr/bin/env python3
"""Divide FFmpeg+QEMU (Devign, function.json) en train/val/test 80/10/10.

Un solo reparto, con semilla fija, que usan los dos modelos (LineVul y AMPLE),
para que ambos entrenen, validen y prueben con exactamente las mismas funciones.

Pasos:
  1. Lee function.json (project, commit_id, target, func). El id de cada
     funcion es su posicion en ese archivo.
  2. Texto de funcion repetido: si las copias tienen la misma etiqueta se deja
     una; si tienen etiquetas distintas no se puede saber cual es la correcta
     y ademas podrian caer en particiones distintas (fuga), asi que se
     descartan todas las copias.
  3. Reparto estratificado por (project, target) con random.Random(semilla),
     80/10/10.
  4. Escribe train.csv, val.csv, test.csv (columnas: id, project, commit_id,
     target, processed_func; `processed_func` es el nombre que lee LineVul) y
     resumen.txt con los conteos reales.

Uso:
  python3 dividir_ffmpeg_qemu.py --entrada ../conjuntos_de_datos/ffmpeg_qemu/function.json \
      --salida ../conjuntos_de_datos/ffmpeg_qemu/split_semilla42
"""
import argparse
import collections
import csv
import json
import os
import random
import sys

csv.field_size_limit(sys.maxsize)
CAMPOS = ["id", "project", "commit_id", "target", "processed_func"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    args = ap.parse_args()

    datos = json.load(open(args.entrada, encoding="utf-8"))
    filas = [dict(id=i, project=d["project"], commit_id=d["commit_id"],
                  target=int(d["target"]), processed_func=d["func"])
             for i, d in enumerate(datos)]
    log = [f"filas en function.json: {len(filas)}"]

    # 2. repetidos por texto
    por_texto = collections.defaultdict(list)
    for f in filas:
        por_texto[f["processed_func"]].append(f)
    conservadas, conflicto, mismas = [], 0, 0
    for grupo in por_texto.values():
        etiquetas = {g["target"] for g in grupo}
        if len(grupo) == 1:
            conservadas.append(grupo[0])
        elif len(etiquetas) == 1:
            conservadas.append(grupo[0])
            mismas += len(grupo) - 1
        else:
            conflicto += len(grupo)
    log.append(f"copias con la misma etiqueta descartadas (se deja una): {mismas}")
    log.append(f"filas descartadas por mismo texto con etiquetas distintas: {conflicto}")
    log.append(f"funciones tras limpiar: {len(conservadas)}")

    # 3. reparto estratificado por (project, target)
    rng = random.Random(args.semilla)
    estratos = collections.defaultdict(list)
    for f in conservadas:
        estratos[(f["project"], f["target"])].append(f)
    particiones = {"train": [], "val": [], "test": []}
    for clave in sorted(estratos):
        grupo = sorted(estratos[clave], key=lambda f: f["id"])
        rng.shuffle(grupo)
        n = len(grupo)
        n_train = round(n * args.train)
        n_val = round(n * args.val)
        particiones["train"] += grupo[:n_train]
        particiones["val"] += grupo[n_train:n_train + n_val]
        particiones["test"] += grupo[n_train + n_val:]

    # comprobaciones antes de escribir
    ids = [set(f["id"] for f in p) for p in particiones.values()]
    textos = [set(f["processed_func"] for f in p) for p in particiones.values()]
    for a in range(3):
        for b in range(a + 1, 3):
            assert not (ids[a] & ids[b]), "ids repetidos entre particiones"
            assert not (textos[a] & textos[b]), "texto repetido entre particiones"
    assert sum(len(p) for p in particiones.values()) == len(conservadas)

    # 4. salida
    os.makedirs(args.salida, exist_ok=True)
    log.append(f"semilla: {args.semilla}")
    for nombre, filas_p in particiones.items():
        filas_p.sort(key=lambda f: f["id"])
        with open(os.path.join(args.salida, f"{nombre}.csv"), "w",
                  newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CAMPOS)
            w.writeheader()
            w.writerows(filas_p)
        v = sum(f["target"] for f in filas_p)
        por_proy = collections.Counter(f["project"] for f in filas_p)
        log.append(f"{nombre}: {len(filas_p)} funciones, {v} vulnerables "
                   f"({100 * v / len(filas_p):.2f} %), por proyecto {dict(por_proy)}")
    texto = "\n".join(log)
    open(os.path.join(args.salida, "resumen.txt"), "w", encoding="utf-8").write(texto + "\n")
    print(texto)


if __name__ == "__main__":
    main()
