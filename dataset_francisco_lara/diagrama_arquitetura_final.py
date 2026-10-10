"""Diagrama esquematico da PIPELINE UNIFICADA FINAL: 4 canais (temperatura,
vibracao, oleo, alarme de processo) votando >=2 + refratario 48h.

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/dataset_francisco_lara/diagrama_arquitetura_final.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = os.path.dirname(os.path.abspath(__file__))

fig, ax = plt.subplots(figsize=(19, 11.8))
ax.set_xlim(0, 19.6)
ax.set_ylim(-0.6, 12.5)
ax.axis("off")


def box(x, y, w, h, text, facecolor, textcolor="black", fontsize=8.6, fontweight="normal", edgecolor="#444444"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.12",
                        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=textcolor, fontweight=fontweight, zorder=3)
    return (x, y, w, h)


def arrow(b1, b2, color="#444444", lw=1.5, connectionstyle="arc3,rad=0.0"):
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


ax.text(9.8, 12.1, "Pipeline unificada final -- 4 canais, votação ≥2 + refratário 48h (config recomendada)",
        ha="center", fontsize=13.5, fontweight="bold")

COLW = 4.5
GAP = 0.25
X0 = [0.2 + i * (COLW + GAP) for i in range(4)]

# --- linha 1: sensores brutos, separados em 4 grupos ---
b_temp_sensors = box(X0[0], 10.2, COLW, 1.3, "SENSORES → MODELO\nTC382_03_A, T5_AVG_A\n(temperatura de mancal)", "#EAF1FB")
b_vib_sensors = box(X0[1], 10.2, COLW, 1.3, "SENSORES → MODELO\n10 canais TV_*\n(vibração de mancal)", "#EAF1FB")
b_oleo_sensors = box(X0[2], 10.2, COLW, 1.3, "SENSORES → MODELO\nPI_0308, PDIT_0305\n(pressão de óleo)", "#EAF1FB")
b_alarm_sensors = box(X0[3], 10.2, COLW, 1.3, "SEM MODELO -- estatística pura\n5 tags de alarme de processo\n(PI_6240319_AL, PAL_6240315, ...)",
                      "#FDECD2", textcolor="#B06A00")

# --- linha 2: modelos ---
b_temp_model = box(X0[0], 8.5, COLW, 1.2, "OCSVM (EXP33)\n+ features multiescala\nlimiar: busca AutoML", "#D6E4F7", fontweight="bold")
b_vib_model = box(X0[1], 8.5, COLW, 1.2, "OCSVM (EXP34)\n+ features multiescala\nlimiar: grade p69-p99,9", "#D6E4F7", fontweight="bold")
b_oleo_model = box(X0[2], 8.5, COLW, 1.2, "OCSVM (EXP38)\n+ features multiescala\nlimiar: grade p69-p99,9", "#D6E4F7", fontweight="bold")
b_alarm_proximity = box(X0[3], 8.5, COLW, 1.2, "Proximidade temporal\n\"alarme nas últimas 24h?\"\n(sem parâmetro aprendido)",
                        "#FDECD2", textcolor="#B06A00", fontweight="bold")

arrow(b_temp_sensors, b_temp_model)
arrow(b_vib_sensors, b_vib_model)
arrow(b_oleo_sensors, b_oleo_model)
arrow(b_alarm_sensors, b_alarm_proximity)

# --- linha 3: portoes de producao (so nos 3 canais com modelo) ---
b_temp_gates = box(X0[0], 6.8, COLW, 1.2, "Portões de produção:\nduração mínima, rampa de carga,\ndegrau, veto sensor congelado", "#D6E4F7")
b_vib_gates = box(X0[1], 6.8, COLW, 1.2, "Portões de produção:\nduração mínima, rampa de carga,\ndegrau, veto sensor congelado", "#D6E4F7")
b_oleo_gates = box(X0[2], 6.8, COLW, 1.2, "Portões de produção:\nduração mínima,\nveto sensor congelado", "#D6E4F7")

arrow(b_temp_model, b_temp_gates)
arrow(b_vib_model, b_vib_gates)
arrow(b_oleo_model, b_oleo_gates)

# --- linha 4: 4 canais binarios ---
b_canal_temp = box(X0[0], 5.1, COLW, 1.0, "Canal 1: temperatura\n(binário, já gateado)", "#C3D9F5", fontweight="bold")
b_canal_vib = box(X0[1], 5.1, COLW, 1.0, "Canal 2: vibração\n(binário, já gateado)", "#C3D9F5", fontweight="bold")
b_canal_oleo = box(X0[2], 5.1, COLW, 1.0, "Canal 3: pressão de óleo\n(binário, já gateado)", "#C3D9F5", fontweight="bold")
b_canal_alarme = box(X0[3], 5.1, COLW, 1.0, "Canal 4: alarme de processo\n(binário, sem modelo)", "#FBE0B0", textcolor="#B06A00", fontweight="bold")

arrow(b_temp_gates, b_canal_temp)
arrow(b_vib_gates, b_canal_vib)
arrow(b_oleo_gates, b_canal_oleo)
arrow(b_alarm_proximity, b_canal_alarme)

# --- linha 5: camada de decisao (regras, sem treino) ---
b_voto = box(3.4, 3.2, 12.8, 1.3, "VOTAÇÃO ≥ 2 de 4 canais concordando\n(regra fixa -- sem parâmetro aprendido)", "#E4E4E4", fontweight="bold", fontsize=10.5)
arrow(b_canal_temp, b_voto, connectionstyle="arc3,rad=-0.15")
arrow(b_canal_vib, b_voto, connectionstyle="arc3,rad=-0.05")
arrow(b_canal_oleo, b_voto, connectionstyle="arc3,rad=0.05")
arrow(b_canal_alarme, b_voto, connectionstyle="arc3,rad=0.15")

b_refrat = box(3.4, 1.5, 12.8, 1.3, "REFRATÁRIO 48h\n(suprime reativação do mesmo evento -- regra fixa)", "#E4E4E4", fontweight="bold", fontsize=10.5)
arrow(b_voto, b_refrat)

b_final = box(6.6, -0.5, 6.4, 1.1, "DECISÃO FINAL (pipeline unificada)\n8 de 8 TRIPs · 6,59 FP/mês",
              "#DFF5DF", textcolor="#0CA30C", fontweight="bold", fontsize=10.5)
arrow(b_refrat, b_final)

fig.tight_layout()
out_path = os.path.join(OUT, "diagrama_arquitetura_final.png")
fig.savefig(out_path, dpi=170, facecolor="white")
print("figura salva em", out_path)
