"""Serie bruta completa + TODAS as anomalias detectadas pelo EXP22 (pontos
finais, is_anom_point=1) + os 40 alarmes reais avaliados (TC382_03_A/
T5_AVG_A) marcados como linhas verticais -- visao direta de "quanto as
deteccoes se alinham com os alarmes reais", sem separar em categorias.

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/relatorio_exp21_portao_pressao/gen_figura_anomalias_vs_alarmes.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from clearml import Dataset

OUT = os.path.dirname(os.path.abspath(__file__))
POINT_CSV = "/home/dvar/.clearml/cache/storage_manager/global/950100358f8e4a8dcba408279af95237.point_anomalies_all.csv"
DATA_END = pd.Timestamp("2026-04-20 23:59:59")
OOS_START = pd.Timestamp("2025-07-01")

print("lendo point_anomalies_all.csv (EXP22)...", flush=True)
df_point = pd.read_csv(POINT_CSV, index_col=0, parse_dates=True)
df_point = df_point.loc[df_point.index <= DATA_END]
anom_times = df_point.index[df_point["is_anom_point"] == 1]
print(f"anomalias detectadas (finais): {len(anom_times)}", flush=True)

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
raw_path = os.path.join(root, "sensores_full_2024_2026_30s.csv")
print("lendo serie bruta do sensor alvo...", flush=True)
df_raw = pd.read_csv(raw_path, usecols=["data_datetime", "TC382_03_A"])
df_raw["data_datetime"] = pd.to_datetime(df_raw["data_datetime"], errors="coerce")
df_raw["TC382_03_A"] = pd.to_numeric(df_raw["TC382_03_A"], errors="coerce")
df_raw = df_raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
df_raw = df_raw.loc[(df_raw.index >= "2024-07-01") & (df_raw.index <= DATA_END)]

alarm_path = os.path.join(root, "alarmes_selecionados_turbina_a.csv")
alarm = pd.read_csv(alarm_path)
alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrência"], errors="coerce")
alarm["Tag"] = alarm["Tag Alarme"]
alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].copy()
alarm = alarm.dropna(subset=["Data da Ocorrencia"]).sort_values("Data da Ocorrencia").reset_index(drop=True)

sub2 = alarm[alarm["Tag"].isin(["TC382_03_A", "T5_AVG_A"])].sort_values("Data da Ocorrencia")
alarm_eval = sub2.loc[sub2["Data da Ocorrencia"] >= OOS_START]
print(f"alarmes avaliados (2 tags, OOS): {len(alarm_eval)}", flush=True)

# hit por alarme: ja marcado +-1440min (24h) na pipeline real
WIN = pd.Timedelta(minutes=1440)
hits, misses = [], []
for t in alarm_eval["Data da Ocorrencia"]:
    if df_point.loc[(df_point.index >= t - WIN) & (df_point.index <= t + WIN), "is_anom_point"].sum() > 0:
        hits.append(t)
    else:
        misses.append(t)
print(f"acertados: {len(hits)}  perdidos: {len(misses)}", flush=True)

fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(df_raw.index, df_raw["TC382_03_A"], color="#184F95", linewidth=0.35, zorder=1, label="TC382_03_A (bruto)")

vals_anom = df_raw["TC382_03_A"].reindex(anom_times.intersection(df_raw.index))
ax.scatter(vals_anom.index, vals_anom.values, color="#D03B3B", s=9, zorder=3,
           label=f"anomalias detectadas (n={len(anom_times)})")

ymin, ymax = df_raw["TC382_03_A"].min(), df_raw["TC382_03_A"].max()
for i, t in enumerate(hits):
    ax.axvline(t, color="#0CA30C", linewidth=1.1, alpha=0.7, zorder=2,
               label="alarme real -- acertado" if i == 0 else None)
for i, t in enumerate(misses):
    ax.axvline(t, color="#7A1FA2", linewidth=1.3, linestyle="--", alpha=0.85, zorder=2,
               label="alarme real -- perdido" if i == 0 else None)

ax.set_xlim(df_raw.index.min(), df_raw.index.max())
ax.set_ylabel("Temperatura (°C)")
ax.set_title(f"EXP22 -- anomalias detectadas x alarmes reais (hit_rate={len(hits)}/{len(alarm_eval)})")
ax.legend(loc="upper left", fontsize=9, frameon=True)
fig.tight_layout()

out_path = os.path.join(OUT, "serie_anomalias_vs_alarmes_exp22.png")
fig.savefig(out_path, dpi=180)
print("\nfigura salva em", out_path)
