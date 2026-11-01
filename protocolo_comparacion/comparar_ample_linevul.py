#!/usr/bin/env python3
"""Compara LineVul y AMPLE sobre FFmpeg+QEMU (reparto 80/10/10, semilla 42).

AMPLE solo evalua las funciones del test cuyo grafo se pudo generar (cpg_GS.py
descarta las de mas de 500 nodos). Para comparar en igualdad de condiciones,
las metricas de LineVul se recalculan sobre esas mismas funciones, usando
las predicciones por funcion de LineVul (raw_preds_ffmpeg_qemu.csv).

Las metricas de AMPLE se leen de la ultima linea "Test" del log de
entrenamiento; la matriz de confusion de AMPLE se reconstruye a partir de
recall, precision y las clases del test (no se guardan predicciones por
funcion) y se comprueba contra las cuatro metricas del log.

Uso, desde la raiz del repositorio:
  python3 protocolo_comparacion/comparar_ample_linevul.py
"""
import csv
import json
import os
import re
import sys
from math import sqrt

csv.field_size_limit(sys.maxsize)

PREDS = "LineVul-main/linevul/results/raw_preds_ffmpeg_qemu.csv"
TEST_AMPLE = "AMPLE-main/devign_dataset_ffmpeg_qemu/devign_cpg_ffmpeg_qemu/devign-test-file.json"
LOG_AMPLE = "AMPLE-main/AMPLE_code/models/ffmpeg_qemu_gpu/devign_FFmpeg.log"


def metricas(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)
    mcc = (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    base = (tp + fn) / n
    return {"n": n, "vul": tp + fn, "tasa_vul": base, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": (tp + tn) / n, "precision": prec, "recall": rec, "f1": f1, "mcc": mcc,
            "f1_siempre_vulnerable": 2 * base / (1 + base)}


def imprime(titulo, m):
    print(f"{titulo}: n={m['n']} vulnerables={m['vul']} ({100 * m['tasa_vul']:.2f} %)")
    print(f"  TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}")
    print(f"  accuracy {m['accuracy']:.4f}  precision {m['precision']:.4f}  recall {m['recall']:.4f}  "
          f"F1 {m['f1']:.4f}  MCC {m['mcc']:.3f}  (F1 de 'siempre vulnerable': {m['f1_siempre_vulnerable']:.4f})")


def linevul(filas):
    y = [int(r["target"]) for r in filas]
    p = [1 if r["raw_preds"] == "True" else 0 for r in filas]
    tp = sum(a & b for a, b in zip(y, p))
    fp = sum((1 - a) & b for a, b in zip(y, p))
    fn = sum(a & (1 - b) for a, b in zip(y, p))
    return metricas(tp, fp, fn, len(y) - tp - fp - fn)


def main():
    filas = list(csv.DictReader(open(PREDS, encoding="utf-8")))
    en_ample = set()
    for nombre in json.load(open(TEST_AMPLE)):
        _, id_, etiqueta = os.path.basename(nombre)[:-2].split("_")
        en_ample.add((id_, etiqueta))
    con_grafo = [r for r in filas if (r["id"], r["target"]) in en_ample]
    sin_grafo = [r for r in filas if (r["id"], r["target"]) not in en_ample]
    assert len(con_grafo) == len(en_ample), (len(con_grafo), len(en_ample))

    ultima = [l for l in open(LOG_AMPLE, encoding="utf-8") if "DevignModel" in l and "----Test----" in l][-1]
    campos = dict(re.findall(r"(Accuracy|Precision|Recall|F1): ([0-9.]+)", ultima))
    acc, prec, rec, f1 = (float(campos[k]) for k in ("Accuracy", "Precision", "Recall", "F1"))

    n, vul = len(con_grafo), sum(int(r["target"]) for r in con_grafo)
    tp = round(rec / 100 * vul)
    fp = round(tp / (prec / 100)) - tp
    m_ample = metricas(tp, fp, vul - tp, n - vul - fp)
    for nombre, calc, log in (("accuracy", m_ample["accuracy"], acc), ("precision", m_ample["precision"], prec),
                              ("recall", m_ample["recall"], rec), ("f1", m_ample["f1"], f1)):
        assert abs(100 * calc - log) < 5e-3, (nombre, 100 * calc, log)

    print("== AMPLE (log: ultima linea Test, mejor checkpoint por F1 de valid) ==")
    imprime("AMPLE en sus 1,908 funciones", m_ample)
    print("   (matriz reconstruida; coincide con las 4 metricas del log a 4 decimales)")
    print()
    print("== LineVul sobre las mismas funciones ==")
    imprime("LineVul en las funciones con grafo", linevul(con_grafo))
    imprime("LineVul en las funciones SIN grafo", linevul(sin_grafo))
    imprime("LineVul en todo el test", linevul(filas))


if __name__ == "__main__":
    main()
