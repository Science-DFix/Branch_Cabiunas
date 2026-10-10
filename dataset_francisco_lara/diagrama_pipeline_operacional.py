"""Diagrama de arquitetura -- gerado do zero para o relatorio de
documentacao operacional da pipeline (nao reaproveita figura de
relatorios anteriores).

Uso:
    python3 dataset_francisco_lara/diagrama_pipeline_operacional.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))

fig, ax = plt.subplots(figsize=(19, 12.2))
ax.set_xlim(0, 19.6)
ax.set_ylim(-1.0, 13.0)
ax.axis("off")


def box(x, y, w, h, text, facecolor, textcolor="black", fontsize=8.8, fontweight="normal", edgecolor="#3a3a3a"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.12",
                        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.3, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=textcolor, fontweight=fontweight, zorder=3)
    return (x, y, w, h)


def arrow(b1, b2, color="#3a3a3a", lw=1.6, connectionstyle="arc3,rad=0.0"):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    p1 = (x1 + w1 / 2, y1)
    p2 = (x2 + w2 / 2, y2 + h2)
    if y1 < y2:
        p1 = (x1 + w1 / 2, y1 + h1)
        p2 = (x2 + w2 / 2, y2)
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=15, color=color, lw=lw,
                         connectionstyle=connectionstyle, zorder=1)
    ax.add_patch(a)


ax.text(9.8, 12.6, "Pipeline operacional de detecção de anomalias -- visão geral end-to-end",
        ha="center", fontsize=14, fontweight="bold")

COLW = 4.5
GAP = 0.25
X0 = [0.2 + i * (COLW + GAP) for i in range(4)]

ETAPAS_Y = {
    "sensores": 11.0,
    "treino": 9.3,
    "portoes": 7.6,
    "canal": 6.0,
}

titulo_col = ["CANAL 1 -- TEMPERATURA", "CANAL 2 -- VIBRAÇÃO", "CANAL 3 -- ÓLEO", "CANAL 4 -- ALARME"]
for x, t in zip(X0, titulo_col):
    ax.text(x + COLW / 2, 11.75, t, ha="center", fontsize=10, fontweight="bold", color="#222222")

b_temp_s = box(X0[0], ETAPAS_Y["sensores"], COLW, 1.15, "TC382_03_A\nT5_AVG_A", "#EAF1FB")
b_vib_s = box(X0[1], ETAPAS_Y["sensores"], COLW, 1.15, "10 canais TV_*\n(vibração de mancal)", "#EAF1FB")
b_oleo_s = box(X0[2], ETAPAS_Y["sensores"], COLW, 1.15, "PI_0308\nPDIT_0305", "#EAF1FB")
b_alarm_s = box(X0[3], ETAPAS_Y["sensores"], COLW, 1.15, "5 tags de catálogo\n(sem modelo)", "#FDECD2", textcolor="#B06A00")

b_temp_m = box(X0[0], ETAPAS_Y["treino"], COLW, 1.15, "OCSVM\nnu=0,05 · gamma=scale\nlimiar: AutoML p97,5", "#D6E4F7", fontweight="bold")
b_vib_m = box(X0[1], ETAPAS_Y["treino"], COLW, 1.15, "OCSVM\nnu=0,05 · gamma=scale\nlimiar: AutoML p95,0", "#D6E4F7", fontweight="bold")
b_oleo_m = box(X0[2], ETAPAS_Y["treino"], COLW, 1.15, "OCSVM\nnu=0,05 · gamma=scale\nlimiar: AutoML p99,9", "#D6E4F7", fontweight="bold")
b_alarm_m = box(X0[3], ETAPAS_Y["treino"], COLW, 1.15, "proximidade temporal\n\"houve alarme em 24h?\"", "#FDECD2", textcolor="#B06A00", fontweight="bold")

arrow(b_temp_s, b_temp_m)
arrow(b_vib_s, b_vib_m)
arrow(b_oleo_s, b_oleo_m)
arrow(b_alarm_s, b_alarm_m)

b_temp_g = box(X0[0], ETAPAS_Y["portoes"], COLW, 1.15, "duração mín. 4,5min\nrampa de carga · degrau\nveto sensor congelado", "#D6E4F7")
b_vib_g = box(X0[1], ETAPAS_Y["portoes"], COLW, 1.15, "duração mín. 4,5min\nrampa de carga · degrau\nveto sensor congelado", "#D6E4F7")
b_oleo_g = box(X0[2], ETAPAS_Y["portoes"], COLW, 1.15, "duração mín. 4,5min\nveto sensor congelado", "#D6E4F7")

arrow(b_temp_m, b_temp_g)
arrow(b_vib_m, b_vib_g)
arrow(b_oleo_m, b_oleo_g)

b_c1 = box(X0[0], ETAPAS_Y["canal"], COLW, 0.95, "canal binário 0/1", "#C3D9F5", fontweight="bold")
b_c2 = box(X0[1], ETAPAS_Y["canal"], COLW, 0.95, "canal binário 0/1", "#C3D9F5", fontweight="bold")
b_c3 = box(X0[2], ETAPAS_Y["canal"], COLW, 0.95, "canal binário 0/1", "#C3D9F5", fontweight="bold")
b_c4 = box(X0[3], ETAPAS_Y["canal"], COLW, 0.95, "canal binário 0/1", "#FBE0B0", textcolor="#B06A00", fontweight="bold")

arrow(b_temp_g, b_c1)
arrow(b_vib_g, b_c2)
arrow(b_oleo_g, b_c3)
arrow(b_alarm_m, b_c4)

b_voto = box(3.4, 4.65, 12.8, 1.1, "SOMA DE VOTOS ENTRE OS 4 CANAIS  →  VOTAÇÃO ≥ 2 CONCORDANDO",
             "#E4E4E4", fontweight="bold", fontsize=10.8)
arrow(b_c1, b_voto, connectionstyle="arc3,rad=-0.15")
arrow(b_c2, b_voto, connectionstyle="arc3,rad=-0.05")
arrow(b_c3, b_voto, connectionstyle="arc3,rad=0.05")
arrow(b_c4, b_voto, connectionstyle="arc3,rad=0.15")

b_dur = box(3.4, 3.2, 12.8, 1.1, "FILTRO DE DURAÇÃO MÍNIMA 45min -- descarta coincidência pontual entre canais",
            "#E4E4E4", fontweight="bold", fontsize=10.8)
arrow(b_voto, b_dur)

b_refrat = box(3.4, 1.75, 12.8, 1.1, "REFRATÁRIO 48h -- suprime reativação do mesmo evento",
               "#E4E4E4", fontweight="bold", fontsize=10.8)
arrow(b_dur, b_refrat)

b_final = box(6.6, -0.15, 6.4, 1.15, "DECISÃO FINAL DA PIPELINE\n8 de 8 TRIPs · 2,88 FP/mês · antecedência média 23,8h",
              "#DFF5DF", textcolor="#0CA30C", fontweight="bold", fontsize=10.2)
arrow(b_refrat, b_final)

fig.tight_layout()
out_path = os.path.join(OUT, "diagrama_pipeline_operacional.png")
fig.savefig(out_path, dpi=170, facecolor="white")
print("figura salva em", out_path)
