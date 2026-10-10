"""Grafico: taxa de FP explicados por OUTRO alarme de processo (nao os
5 ja usados no canal 4) vs baseline aleatorio (controle negativo) --
argumento de que uma parte dos FP e sinal fisico real respondendo a
outros alarmes do processo, nao ruido.

Uso:
    python3 dataset_francisco_lara/figura_alarmes_proximos_42fp.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(OUT, "alarmes_proximos_42fp.csv"))

taxa_fp = 100 * df["explicado_por_outro_alarme"].mean()
taxa_baseline = 100 * 0.0344
enrichment = taxa_fp / taxa_baseline

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

ax0 = axes[0]
bars = ax0.bar(["42 FP\n(pós-filtro 45min)", "baseline aleatório\n(controle negativo)"],
                [taxa_fp, taxa_baseline], color=["#D67A2A", "#898781"], width=0.55, edgecolor="black", linewidth=0.6)
for b, v in zip(bars, [taxa_fp, taxa_baseline]):
    ax0.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax0.set_ylabel("% com outro alarme de processo em ±2h", fontsize=10)
ax0.set_ylim(0, max(taxa_fp, taxa_baseline) * 1.35)
ax0.set_title(f"Enriquecimento: {enrichment:.1f}x\n(exclui os 5 tags já usados no canal 4)", fontsize=10.5)
ax0.grid(alpha=0.25, axis="y")

ax1 = axes[1]
tags = df.loc[df["explicado_por_outro_alarme"], "tag_alarme_proximo"].value_counts()
ax1.barh(tags.index[::-1], tags.values[::-1], color="#2A78D6", edgecolor="black", linewidth=0.6)
ax1.set_xlabel("nº de episódios explicados", fontsize=10)
ax1.set_title(f"Quais tags explicam os {int(df['explicado_por_outro_alarme'].sum())} de 42 episódios\ncorroborados por outro alarme", fontsize=10.5)
ax1.grid(alpha=0.25, axis="x")
ax1.set_xticks(range(0, int(tags.max()) + 2))

fig.suptitle("Parte dos FP remanescentes responde a OUTROS alarmes de processo, não a ruído",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])

out_path = os.path.join(OUT, "alarmes_proximos_42fp.png")
fig.savefig(out_path, dpi=170, facecolor="white")
print("figura salva em", out_path)
