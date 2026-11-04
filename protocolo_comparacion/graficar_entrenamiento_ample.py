#!/usr/bin/env python3
"""Grafica las curvas de entrenamiento de AMPLE (FFmpeg+QEMU) a partir de su log.

Lee protocolo_comparacion/ample_ffmpeg_qemu_devign_FFmpeg.log (lineas Train y
Valid de cada epoca; las lineas Test por epoca no se usan: son monitoreo parcial).

Uso: python3 protocolo_comparacion/graficar_entrenamiento_ample.py <salida.png>
"""
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "protocolo_comparacion/ample_ffmpeg_qemu_devign_FFmpeg.log"
train, valid = {}, {}
for l in open(LOG, encoding="utf-8"):
    m = re.search(r"Epoch (\d+)\s+---Train---.*?\sLoss:\s+([\d.]+)\s+Accuracy: ([\d.]+)", l)
    if m:
        train[int(m.group(1))] = float(m.group(3))
    m = re.search(r"Epoch (\d+)\s+----Valid---- Loss:\s+([\d.]+)\s+Accuracy: ([\d.]+)\s+Precision: ([\d.]+)\s+Recall: ([\d.]+)\s+F1: ([\d.]+)", l)
    if m:
        valid[int(m.group(1))] = (float(m.group(3)), float(m.group(6)))
assert len(train) == 100 and len(valid) == 100, (len(train), len(valid))
ep = list(range(1, 101))
mejor = max(valid, key=lambda e: valid[e][1])

fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.3), dpi=200)
ax[0].plot(ep, [valid[e][1] for e in ep], color="#1f5fa8", lw=1.2)
ax[0].scatter([mejor], [valid[mejor][1]], color="#c0392b", zorder=3)
ax[0].annotate(f"mejor: época {mejor}\nF1 = {valid[mejor][1]:.2f}", (mejor, valid[mejor][1]),
               xytext=(mejor + 14, 67), arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8)
ax[0].set_xlabel("Época"); ax[0].set_ylabel("F1 de validación (%)"); ax[0].grid(alpha=0.3)
ax[0].set_ylim(25, 72)
ax[1].plot(ep, [train[e] for e in ep], color="#2e8b57", lw=1.2, label="entrenamiento")
ax[1].plot(ep, [valid[e][0] for e in ep], color="#1f5fa8", lw=1.2, label="validación")
ax[1].set_xlabel("Época"); ax[1].set_ylabel("Exactitud (%)"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(sys.argv[1])
print("mejor epoca", mejor, valid[mejor], "| epoca 60: train", train[60], "valid", valid[60][0], "| epoca 100: train", train[100], "valid", valid[100][0])
