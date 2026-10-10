"""Figura operacional: 3 janelas reais, zoom, mostrando o que o
ENABLE_ALERT_CATALOG_CONTEXT (EXP30) produz na pratica:

  1. Precursor de TRIP real (07/04/2025) -- is_anom_point=1, alert_confidence
     = "explicado_catalogo" TAMBEM (achado importante: TRIPs reais quase
     sempre disparam outro alarme do catalogo ao mesmo tempo -- ver texto).
  2. Alerta isolado explicado por um alarme de processo rotineiro
     (PI_6240319_AL, 08/01/2025) -- mesma tag "explicado_catalogo", mas
     SEM relacao com nenhum TRIP.
  3. Alerta genuinamente isolado (27/08/2025, o unico "mistério" que
     restou de toda a investigacao) -- alert_confidence = "isolado".

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/dataset_francisco_lara/gen_figura_operacional.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os
from clearml import Dataset

OUT = os.path.dirname(os.path.abspath(__file__))
POINT_CSV = "/home/dvar/.clearml/cache/storage_manager/global/0dc793c090afef5709a6c3a685c2e021.point_anomalies_all.csv"
SENSORS = ["TC382_03_A", "T5_AVG_A"]

EXAMPLES = [
    {
        "titulo": "1. Precursor de TRIP real (mancal, 07/04/2025)",
        "win": (pd.Timestamp("2025-04-06 18:00"), pd.Timestamp("2025-04-08 03:00")),
        "trip_time": pd.Timestamp("2025-04-07 21:18:00"),
        "acao": "PRIORIDADE MÁXIMA -- escalar imediato\n(mesmo já corroborado pelo catálogo)",
        "cor_caixa": "#0CA30C",
    },
    {
        "titulo": "2. Alerta explicado por alarme de processo rotineiro (08/01/2025)",
        "win": (pd.Timestamp("2025-01-08 07:00"), pd.Timestamp("2025-01-08 14:00")),
        "trip_time": None,
        "acao": "PRIORIDADE BAIXA -- registrar, não escalar\n(tag PI_6240319_AL é rotineira,\nnunca associada a um TRIP real)",
        "cor_caixa": "#EB9C34",
    },
    {
        "titulo": "3. Alerta genuinamente isolado (27/08/2025, o único caso sem explicação)",
        "win": (pd.Timestamp("2025-08-26 21:00"), pd.Timestamp("2025-08-27 04:00")),
        "trip_time": None,
        "acao": "PRIORIDADE MÉDIA -- checagem manual\n(sem correlação em nenhum sensor\nou alarme próximo -- ver Seção 6)",
        "cor_caixa": "black",
    },
]

print("lendo point_anomalies_all.csv (EXP30)...", flush=True)
df_point = pd.read_csv(POINT_CSV, index_col=0, parse_dates=True, low_memory=False)

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
raw_path = os.path.join(root, "sensores_full_2024_2026_30s.csv")
print("lendo series brutas...", flush=True)
df_raw = pd.read_csv(raw_path, usecols=["data_datetime"] + SENSORS)
df_raw["data_datetime"] = pd.to_datetime(df_raw["data_datetime"], errors="coerce")
for c in SENSORS:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
df_raw = df_raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()

fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

for ax, ex in zip(axes, EXAMPLES):
    w0, w1 = ex["win"]
    seg_raw = df_raw.loc[w0:w1]
    seg_point = df_point.loc[w0:w1]

    ax.plot(seg_raw.index, seg_raw["T5_AVG_A"], color="#184F95", linewidth=0.8, label="T5_AVG_A")

    if ex["trip_time"] is not None:
        ax.axvline(ex["trip_time"], color="#7A1FA2", linewidth=1.8, linestyle="--", label="TRIP curado")

    anom = seg_point.index[seg_point["is_anom_point"] == 1]
    anom = anom.intersection(seg_raw.index)
    if len(anom) > 0:
        vals = seg_raw["T5_AVG_A"].reindex(anom)
        ax.scatter(vals.index, vals.values, color=ex["cor_caixa"] if ex["cor_caixa"] != "black" else "black",
                   s=45, zorder=5, edgecolor="white", linewidth=0.6, label="ponto detectado (is_anom_point=1)")
        # anota o alarme do catalogo mais proximo, se houver
        row = seg_point.loc[anom[0]]
        tag = row.get("alert_catalog_tag")
        dist = row.get("alert_catalog_distance_h")
        conf = row.get("alert_confidence")
        if pd.notna(tag):
            info = f"catálogo: {tag}\ndistância: {dist:.2f}h\nconfiança: {conf}"
        else:
            info = f"confiança: {conf}\n(nenhum alarme do catálogo\ndentro de ±24h)"
        ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=8.5, va="bottom", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))

    ax.set_title(ex["titulo"], fontsize=9.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7.5)
    ax.legend(loc="upper left", fontsize=7)
    ax.text(0.5, -0.30, ex["acao"], transform=ax.transAxes, fontsize=9, ha="center", va="top",
            fontweight="bold", color=ex["cor_caixa"] if ex["cor_caixa"] != "black" else "black")

fig.suptitle("Como a anotação de contexto (EXP30) aparece na prática -- 3 casos reais", fontsize=12)
fig.tight_layout(rect=[0, 0.08, 1, 0.95])

out_path = os.path.join(OUT, "figura_operacional_exemplos.png")
fig.savefig(out_path, dpi=170, facecolor="white")
print("\nfigura salva em", out_path)
