"""Grafico comparativo: TRIPs detectados x FP/mes, todas as estrategias
testadas nesta investigacao -- do modelo unico (EXP30) ate a pipeline
final unificada de 4 canais. Mesmo espirito da fronteira custo x
deteccao do relatorio do Thallys (DOC_EQUIPE/RELATORIO_DECISAO_DETECTOR.pdf).

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/dataset_francisco_lara/figura_comparativo_estrategias.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# (nome, trips, fp_por_mes, cor, marcador)
PONTOS = [
    ("EXP30\n(modelo único)", 5, 8.72, "#898781", "o"),
    ("EXP33\n(só temperatura)", 7, 24.52, "#2A78D6", "s"),
    ("EXP34\n(só vibração)", 7, 21.15, "#2A78D6", "^"),
    ("EXP38\n(só óleo)", 4, 3.57, "#2A78D6", "v"),
    ("EXP33 OU EXP34\n(união ingênua)", 8, 26.99, "#D03B3B", "D"),
    ("EXP33 E EXP34\n(interseção ingênua)", 6, 14.15, "#D03B3B", "X"),
    ("EXP37\n(3 canais, ≥2 + refratário)", 8, 6.39, "#0CA30C", "*"),
    ("pipeline final\n(4 canais, ≥2 + refratário)", 8, 6.59, "#0B6FA8", "P"),
    ("pipeline final\n(4 canais, ≥3 + refratário)", 5, 3.09, "#898781", "p"),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, xlabel in zip(axes, ["falso positivo por mês (episódios)", "falso positivo por mês (episódios) -- zoom"]):
    for nome, trips, fp, cor, marker in PONTOS:
        size = 380 if marker == "*" else 220
        ax.scatter(fp, trips, s=size, c=cor, marker=marker, edgecolor="black", linewidth=0.8, zorder=3)
        ax.annotate(nome, (fp, trips), textcoords="offset points", xytext=(8, 6), fontsize=8.3)
    ax.set_ylim(4.3, 8.7)
    ax.set_yticks(range(5, 9))
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_ylabel("TRIPs detectados (de 8)", fontsize=9.5)
    ax.grid(alpha=0.3)

axes[0].set_xlim(-1, 29)
axes[1].set_xlim(-1, 10)
axes[0].set_title("Todas as estratégias", fontsize=10.5)
axes[1].set_title("Zoom -- região de baixo custo", fontsize=10.5)

fig.suptitle("Detecção de TRIP × custo de falso positivo -- canto superior esquerdo é o melhor lugar",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])

out_path = os.path.join(OUT, "comparativo_estrategias.png")
fig.savefig(out_path, dpi=170, facecolor="white")
print("figura salva em", out_path)
