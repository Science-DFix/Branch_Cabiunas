"""Diagrama esquematico do fluxo operacional proposto: onde a anotacao
de contexto (ENABLE_ALERT_CATALOG_CONTEXT, EXP30/31) entra na pipeline
e como ela deveria orientar a priorizacao do operador -- SEM nunca
suprimir ou decidir nada sozinha (ver docs/analise_automl_exp10.md,
secao "Supressao cirurgica... REJEITADA").

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/dataset_francisco_lara/gen_diagrama_operacional.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = os.path.dirname(os.path.abspath(__file__))

fig, ax = plt.subplots(figsize=(15, 9.5))
ax.set_xlim(0, 15)
ax.set_ylim(0, 11)
ax.axis("off")


def box(x, y, w, h, text, facecolor, textcolor="black", fontsize=9.5, fontweight="normal"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.12",
                        facecolor=facecolor, edgecolor="#444444", linewidth=1.1, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=textcolor, fontweight=fontweight, zorder=3, wrap=True)
    return (x, y, w, h)


def arrow(b1, b2, style="-|>", color="#444444", lw=1.4, connectionstyle="arc3,rad=0.0"):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    p1 = (x1 + w1 / 2, y1)
    p2 = (x2 + w2 / 2, y2 + h2)
    if y1 < y2:
        p1 = (x1 + w1 / 2, y1 + h1)
        p2 = (x2 + w2 / 2, y2)
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14, color=color, lw=lw,
                         connectionstyle=connectionstyle, zorder=1)
    ax.add_patch(a)


# --- coluna principal (pipeline existente, FIXA) ---
b_sensores = box(5.2, 9.4, 4.6, 1.0, "Sensores brutos (30s)\nmancal + óleo", "#EAF1FB")
b_modelo = box(5.2, 7.7, 4.6, 1.2,
               "Modelo AutoML (OCSVM/iforest)\n+ portões causais\n(rampa, volatilidade, degrau, congelamento)",
               "#EAF1FB")
b_isanom = box(5.2, 6.2, 4.6, 1.0, "is_anom_point decidido\n(EXP28/29 -- config de produção, INALTERADA)",
               "#D6E4F7", fontweight="bold")
b_context = box(5.2, 4.7, 4.6, 1.1,
                "Anotação de contexto (NOVO -- EXP30/31)\ncruza cada ponto contra catálogo de 47 tags,\njanela ±24h -- 100% informativo",
                "#FFF3D6", fontweight="bold")

arrow(b_sensores, b_modelo)
arrow(b_modelo, b_isanom)
arrow(b_isanom, b_context)

# --- ramificacao em 3 saidas (dashboard/triagem) ---
b_verde = box(0.3, 2.6, 4.0, 1.3,
              "próximo de TRIP curado\n(ground truth conhecida)\n→ PRIORIDADE MÁXIMA\nescalar imediato",
              "#DFF5DF", textcolor="#0CA30C", fontweight="bold")
b_laranja = box(5.2, 2.6, 4.6, 1.5,
                "explicado por outro alarme do catálogo\n→ investigar a TAG:\nrotineira/frequente → prioridade BAIXA\ncrítica/pouco frequente → prioridade ALTA",
                "#FDECD2", textcolor="#B06A00", fontweight="bold")
b_preto = box(10.7, 2.6, 4.0, 1.3,
              "isolado\n(nenhum alarme do catálogo em ±24h)\n→ PRIORIDADE MÉDIA\nchecagem manual (z-score)",
              "#EDEDED", textcolor="black", fontweight="bold")

arrow(b_context, b_verde, connectionstyle="arc3,rad=-0.15")
arrow(b_context, b_laranja)
arrow(b_context, b_preto, connectionstyle="arc3,rad=0.15")

b_dash = box(4.2, 0.4, 6.6, 1.3,
             "Dashboard do operador / simpred-cabiunas\n(cada alerta chega com: score, tag mais próxima,\ndistância em horas e prioridade sugerida)",
             "#E4E4E4", fontweight="bold")
arrow(b_verde, b_dash, connectionstyle="arc3,rad=0.15")
arrow(b_laranja, b_dash)
arrow(b_preto, b_dash, connectionstyle="arc3,rad=-0.15")

ax.text(7.5, 10.75, "Fluxo operacional: da série bruta até a priorização no dashboard",
        ha="center", fontsize=13, fontweight="bold")
ax.text(7.5, 0.05,
        "Importante: a anotação NUNCA decide sozinha -- é contexto para o operador. Achado empírico (EXP30): "
        "TRIPs reais também disparam alarme do catálogo em 100% dos casos verificados,\n"
        "então \"explicado_catalogo\" não pode ser lido como \"seguro ignorar\" -- por isso a ramificação acima "
        "sempre olha primeiro pra proximidade de um TRIP conhecido, não pra tag isolada.",
        ha="center", fontsize=8.3, style="italic", color="#444444")

fig.tight_layout()
out_path = os.path.join(OUT, "diagrama_fluxo_operacional.png")
fig.savefig(out_path, dpi=170, facecolor="white")
print("figura salva em", out_path)
