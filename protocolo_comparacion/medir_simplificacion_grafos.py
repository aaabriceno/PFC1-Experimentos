#!/usr/bin/env python3
"""Mide cuanto reduce cada simplificacion de AMPLE el grafo de una funcion.

AMPLE trae cuatro variantes del generador de grafos en data_processing/, que
solo cambian los pasos de simplificacion dentro de graphGeneration():

  cpg_original.py  sin simplificar (grafo del CPG de Joern, solo nodos CFG y su AST)
  cpg_TGS.py       solo TGS: fusion de nodos por tipo (ast_prune, 7 reglas)
  cpg_VGS.py       solo VGS: fusion de variables con el mismo texto (var_combine)
  cpg_GS.py        TGS + VGS (la que se usa en cpg_GS_parallel.py)

Sobre una muestra aleatoria (semilla fija) de funciones del test de FFmpeg+QEMU
se corre graphGeneration() de las cuatro, con los mismos nodes.csv/edges.csv que
dejo Joern, y se cuentan nodos y aristas. Tambien cuantas funciones pasan el
filtro de <= 500 nodos en cada variante.

Uso (desde la raiz del repositorio, dentro del contenedor ample-env):
  docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v "$(pwd)":/vulndet \
    -w /vulndet/AMPLE-main/data_processing ample-env \
    python3 ../../protocolo_comparacion/medir_simplificacion_grafos.py \
      --muestra 600 --workers 28 --salida ../../protocolo_comparacion/resultados_simplificacion_grafos.json
"""
import argparse
import importlib.util
import json
import os
import random
import statistics as st
import sys
from multiprocessing import Pool

DP = os.getcwd()  # debe ser AMPLE-main/data_processing
PARSED = "../devign_dataset_ffmpeg_qemu/devign_parsed"
TEST_JSON = "../devign_dataset_ffmpeg_qemu/devign_data_split/test_raw_code.json"
VARIANTES = [("original", "cpg_original"), ("TGS", "cpg_TGS"), ("VGS", "cpg_VGS"), ("TGS+VGS", "cpg_GS")]
_mods = {}


def cargar(nombre):
    if nombre not in _mods:
        sys.path.insert(0, DP)
        spec = importlib.util.spec_from_file_location("m_" + nombre, os.path.join(DP, nombre + ".py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _mods[nombre] = m
    return _mods[nombre]


def procesa(archivo):
    salida = {"archivo": archivo}
    for etiqueta, nombre in VARIANTES:
        m = cargar(nombre)
        nodes = m.read_csv(os.path.join(PARSED, archivo, "nodes.csv"))
        edges = m.read_csv(os.path.join(PARSED, archivo, "edges.csv"))
        index_map, graph, n, extra = m.graphGeneration(nodes, edges, m.edgeType_reduced, m.ver_edge_type_reduced)
        if extra is True:
            salida[etiqueta] = {"estado": "sin_ast"}
        elif index_map is None:
            salida[etiqueta] = {"estado": "vacio_o_mas_de_500_nodos"}
        else:
            aristas = sum(1 for e in graph if e[1] != m.edgeType_reduced["Self_loop"])
            salida[etiqueta] = {"estado": "ok", "nodos": n, "aristas": aristas}
    return salida


def resumen(valores):
    return {"media": round(st.mean(valores), 1), "mediana": st.median(valores), "max": max(valores)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muestra", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--salida", required=True)
    args = ap.parse_args()

    todas = [e["file_path"] for e in json.load(open(TEST_JSON))]
    random.Random(args.semilla).shuffle(todas)
    muestra = todas[: args.muestra]
    with Pool(args.workers) as p:
        filas = p.map(procesa, muestra, chunksize=4)

    res = {"muestra": len(filas), "semilla": args.semilla, "variantes": {}}
    for etiqueta, _ in VARIANTES:
        c = {}
        for f in filas:
            c[f[etiqueta]["estado"]] = c.get(f[etiqueta]["estado"], 0) + 1
        res["variantes"][etiqueta] = {"estados": c}
    # comparacion justa: solo funciones que producen grafo en las cuatro variantes
    comunes = [f for f in filas if all(f[e]["estado"] == "ok" for e, _ in VARIANTES)]
    res["funciones_con_grafo_en_las_4"] = len(comunes)
    for etiqueta, _ in VARIANTES:
        res["variantes"][etiqueta]["nodos"] = resumen([f[etiqueta]["nodos"] for f in comunes])
        res["variantes"][etiqueta]["aristas_sin_autolazo"] = resumen([f[etiqueta]["aristas"] for f in comunes])
        res["variantes"][etiqueta]["nodos_total"] = sum(f[etiqueta]["nodos"] for f in comunes)
    base = res["variantes"]["original"]["nodos_total"]
    for etiqueta, _ in VARIANTES:
        res["variantes"][etiqueta]["reduccion_nodos_pct"] = round(100 * (1 - res["variantes"][etiqueta]["nodos_total"] / base), 1)
    json.dump({"resumen": res, "filas": filas}, open(args.salida, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
