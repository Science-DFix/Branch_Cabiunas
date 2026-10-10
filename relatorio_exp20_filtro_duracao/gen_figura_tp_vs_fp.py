"""Figura 3 revisada: cruza serie bruta + anomalias detectadas + alarmes
reais, distinguindo anomalias PROXIMAS de um alarme real (candidatas a
precursor genuino) de anomalias ISOLADAS (nenhum alarme dos tags
TC382_03_A/T5_AVG_A em +-24h -- e o que realmente conta como falso
positivo na metrica normal_alert_rate).

Uso:
    python3 gen_figura_tp_vs_fp.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from clearml import Dataset

OUT = os.path.dirname(os.path.abspath(__file__))
POINT_CSV = "/home/dvar/.clearml/cache/storage_manager/global/18605388c830fc41b0cfa3439a4d5249.point_anomalies_all.csv"
WIN_MIN = 1440

print("lendo point_anomalies_all.csv...", flush=True)
df_point = pd.read_csv(POINT_CSV, index_col=0, parse_dates=True)

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
raw_path = os.path.join(root, "sensores_full_2024_2026_30s.csv")
print("lendo serie bruta do sensor alvo...", flush=True)
df_raw = pd.read_csv(raw_path, usecols=["data_datetime", "TC382_03_A"])
df_raw["data_datetime"] = pd.to_datetime(df_raw["data_datetime"], errors="coerce")
df_raw["TC382_03_A"] = pd.to_numeric(df_raw["TC382_03_A"], errors="coerce")
df_raw = df_raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
df_raw = df_raw.loc[(df_raw.index >= "2024-07-01") & (df_raw.index <= "2026-04-30 23:59:59")]

alarm_path = os.path.join(root, "alarmes_selecionados_turbina_a.csv")
alarm = pd.read_csv(alarm_path)
alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrência"], errors="coerce")
alarm["Tag"] = alarm["Tag Alarme"]
alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].copy()
# TODOS os alarmes destes 2 tags, historico completo (nao so os 40 OOS) --
# e essa a mesma base usada pelo near_alarm_mask do pipeline pra decidir
# o que conta como "normal" na metrica de FP.
sub_all = alarm[alarm["Tag"].isin(["TC382_03_A", "T5_AVG_A"])].sort_values("Data da Ocorrencia").reset_index(drop=True)
oos = sub_all[sub_all["Data da Ocorrencia"] >= "2025-07-01"].reset_index(drop=True)
print(f"alarmes (historico completo, 2 tags): {len(sub_all)}  |  no periodo avaliado (OOS): {len(oos)}")

# --- classifica cada ponto anomalo: perto de QUALQUER alarme (historico completo) ou isolado ---
anom_times = df_point.index[df_point["is_anom_point"] == 1]
win = pd.Timedelta(minutes=WIN_MIN)

alarm_times_sorted = sub_all["Data da Ocorrencia"].sort_values().values.astype("datetime64[ns]")
near_flags = np.zeros(len(anom_times), dtype=bool)
# busca binaria: pra cada ponto anomalo, checa se existe algum alarme dentro de +-24h
pos = np.searchsorted(alarm_times_sorted, anom_times.values)
for i in range(len(anom_times)):
    p = pos[i]
    cands = []
    if p > 0:
        cands.append(alarm_times_sorted[p - 1])
    if p < len(alarm_times_sorted):
        cands.append(alarm_times_sorted[p])
    t = anom_times.values[i]
    near_flags[i] = any(abs((t - c)) <= np.timedelta64(WIN_MIN, "m") for c in cands)

anom_near = anom_times[near_flags]
anom_isolado = anom_times[~near_flags]
print(f"anomalias totais: {len(anom_times)}  |  perto de algum alarme: {len(anom_near)}  |  isoladas (FP disperso): {len(anom_isolado)}")

# --- figura ---
fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax = axes[0]
ax.plot(df_raw.index, df_raw["TC382_03_A"], color="#184F95", linewidth=0.35, label="TC382_03_A (bruto)", zorder=1)

vals_near = df_raw["TC382_03_A"].reindex(anom_near.intersection(df_raw.index))
vals_iso = df_raw["TC382_03_A"].reindex(anom_isolado.intersection(df_raw.index))
ax.scatter(vals_iso.index, vals_iso.values, color="#D03B3B", s=9, zorder=4,
           label=f"anomalia ISOLADA -- FP disperso (n={len(anom_isolado)})")
ax.scatter(vals_near.index, vals_near.values, color="#0CA30C", s=9, zorder=5,
           label=f"anomalia perto de alarme real (n={len(anom_near)})")

data_start, data_end = df_raw.index.min(), df_raw.index.max()
alarms_in_range = sub_all.loc[
    (sub_all["Data da Ocorrencia"] >= data_start) & (sub_all["Data da Ocorrencia"] <= data_end), "Data da Ocorrencia"
]
for t in alarms_in_range:
    ax.axvline(t, color="#898781", linewidth=0.5, alpha=0.35, zorder=0)

ax.set_ylabel("Temperatura (°C)")
ax.set_title("TC382_03_A -- anomalias detectadas: perto de alarme real (verde) vs. isoladas (vermelho)")
ax.legend(loc="upper left", fontsize=8, frameon=True)
ax.set_xlim(data_start, data_end)

ax2 = axes[1]
hit_times = []
miss_times = []
for _, r in oos.iterrows():
    t = r["Data da Ocorrencia"]
    t0, t1 = t - win, t + win
    hit = (anom_times[(anom_times >= t0) & (anom_times <= t1)]).size > 0
    (hit_times if hit else miss_times).append(t)

pre_oos_alarms = alarms_in_range[alarms_in_range < "2025-07-01"]
ax2.eventplot(pre_oos_alarms, lineoffsets=2, colors="#898781", linewidths=1.0)
ax2.eventplot(hit_times if hit_times else [], lineoffsets=1, colors="#0CA30C", linewidths=1.5)
ax2.eventplot(miss_times if miss_times else [], lineoffsets=0, colors="#D03B3B", linewidths=1.5)
ax2.set_yticks([0, 1, 2])
ax2.set_yticklabels(["perdido\n(OOS)", "acertado\n(OOS)", "alarme\n(pré-corte)"])
ax2.set_ylim(-0.5, 2.5)
ax2.set_xlabel("tempo")
ax2.axvline(pd.Timestamp("2025-07-01"), color="black", linestyle="--", linewidth=1.0)
ax2.annotate("corte OOS\n(2025-07-01)", xy=(pd.Timestamp("2025-07-01"), 2.3), fontsize=8, ha="left")
ax2.set_xlim(data_start, data_end)

fig.tight_layout()
out_path = os.path.join(OUT, "serie_completa_tp_vs_fp.png")
fig.savefig(out_path, dpi=180)
print("figura salva em", out_path)
