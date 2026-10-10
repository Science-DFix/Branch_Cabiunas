"""Diagrama esquematico da arquitetura EXP37: como os sensores foram
separados (o que vai pra modelo, o que vai sem modelo) e como os 3
canais se combinam ate a decisao final.

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/dataset_francisco_lara/diagrama_arquitetura_exp37.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = os.path.dirname(os.path.abspath(__file__))

fig, ax = plt.subplots(figsize=(16, 11.6))
ax.set_xlim(0, 16)
ax.set_ylim(-0.6, 12.5)
ax.axis("off")


def box(x, y, w, h, text, facecolor, textcolor="black", fontsize=9.2, fontweight="normal", edgecolor="#444444"):
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


ax.text(8, 12.1, "EXP37 -- arquitetura de decisão multi-canal (config recomendada)",
        ha="center", fontsize=14, fontweight="bold")

# --- linha 1: sensores brutos, separados em 3 grupos ---
b_temp_sensors = box(0.3, 10.2, 4.6, 1.3,
                      "SENSORES → MODELO\nTC382_03_A, T5_AVG_A\n(temperatura de mancal)",
                      "#EAF1FB")
b_vib_sensors = box(5.6, 10.2, 4.6, 1.3,
                     "SENSORES → MODELO\n10 canais TV_* \n(vibração)",
                     "#EAF1FB")
b_alarm_sensors = box(10.9, 10.2, 4.8, 1.3,
                      "SEM MODELO -- estatística pura\n5 tags de alarme de processo\n(PI_6240319_AL, PAL_6240315, ...)",
                      "#FDECD2", textcolor="#B06A00")

# --- linha 2: modelos ---
b_temp_model = box(0.3, 8.5, 4.6, 1.2,
                    "OCSVM (EXP33)\n+ features multiescala\nlimiar: busca AutoML",
                    "#D6E4F7", fontweight="bold")
b_vib_model = box(5.6, 8.5, 4.6, 1.2,
                   "OCSVM (EXP34)\n+ features multiescala\nlimiar: grade ampla (p69-p99,9)",
                   "#D6E4F7", fontweight="bold")
b_alarm_proximity = box(10.9, 8.5, 4.8, 1.2,
                        "Proximidade temporal\n\"alarme nas últimas 24h?\"\n(sem parâmetro aprendido)",
                        "#FDECD2", textcolor="#B06A00", fontweight="bold")

arrow(b_temp_sensors, b_temp_model)
arrow(b_vib_sensors, b_vib_model)
arrow(b_alarm_sensors, b_alarm_proximity)

# --- linha 3: portoes de producao (so nos 2 canais com modelo) ---
b_temp_gates = box(0.3, 6.8, 4.6, 1.2,
                    "Portões de produção:\nduração mínima, rampa de carga,\ndegrau, veto sensor congelado",
                    "#D6E4F7")
b_vib_gates = box(5.6, 6.8, 4.6, 1.2,
                   "Portões de produção:\nduração mínima, rampa de carga,\ndegrau, veto sensor congelado",
                   "#D6E4F7")

arrow(b_temp_model, b_temp_gates)
arrow(b_vib_model, b_vib_gates)

# --- linha 4: 3 canais binarios ---
b_canal_temp = box(0.3, 5.1, 4.6, 1.0, "Canal 1: temperatura\n(binário, já gateado)", "#C3D9F5", fontweight="bold")
b_canal_vib = box(5.6, 5.1, 4.6, 1.0, "Canal 2: vibração\n(binário, já gateado)", "#C3D9F5", fontweight="bold")
b_canal_alarme = box(10.9, 5.1, 4.8, 1.0, "Canal 3: alarme de processo\n(binário, sem modelo)", "#FBE0B0",
                     textcolor="#B06A00", fontweight="bold")

arrow(b_temp_gates, b_canal_temp)
arrow(b_vib_gates, b_canal_vib)
arrow(b_alarm_proximity, b_canal_alarme)

# --- linha 5: camada de decisao (regras, sem treino) ---
b_voto = box(3.8, 3.2, 8.4, 1.3,
             "VOTAÇÃO ≥ 2 de 3 canais concordando\n(regra fixa -- sem parâmetro aprendido)",
             "#E4E4E4", fontweight="bold")
arrow(b_canal_temp, b_voto, connectionstyle="arc3,rad=-0.1")
arrow(b_canal_vib, b_voto)
arrow(b_canal_alarme, b_voto, connectionstyle="arc3,rad=0.1")

b_refrat = box(3.8, 1.5, 8.4, 1.3,
               "REFRATÁRIO 48h\n(suprime reativação do mesmo evento -- regra fixa)",
               "#E4E4E4", fontweight="bold")
arrow(b_voto, b_refrat)

b_final = box(5.3, -0.5, 5.4, 1.1,
              "DECISÃO FINAL (EXP37)\n8 de 8 TRIPs · 6,39 FP/mês",
              "#DFF5DF", textcolor="#0CA30C", fontweight="bold", fontsize=10.5)
arrow(b_refrat, b_final)

ax.text(8, 12.45, "", fontsize=1)  # spacer
fig.tight_layout()
out_path = os.path.join(OUT, "diagrama_arquitetura_exp37.png")
fig.savefig(out_path, dpi=170, facecolor="white")
print("figura salva em", out_path)
