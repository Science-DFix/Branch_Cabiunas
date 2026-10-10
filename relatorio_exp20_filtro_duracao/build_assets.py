import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os
from clearml import Dataset

OUT = "/tmp/claude-1000/-home-dvar-REPO-CABIUNAS/85b4269a-b5f0-450b-9615-b38e2751d331/scratchpad/relatorio_exp20"

POINT_CSV = "/home/dvar/.clearml/cache/storage_manager/global/18605388c830fc41b0cfa3439a4d5249.point_anomalies_all.csv"

print("lendo point_anomalies_all.csv...", flush=True)
df_point = pd.read_csv(POINT_CSV, index_col=0, parse_dates=True)
print(df_point.shape, df_point.columns.tolist())

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
sub = alarm[alarm["Tag"].isin(["TC382_03_A", "T5_AVG_A"])].sort_values("Data da Ocorrencia")
oos = sub[sub["Data da Ocorrencia"] >= "2025-07-01"].reset_index(drop=True)
print("alarmes OOS:", len(oos))

win = pd.Timedelta(minutes=1440)
anom_times = df_point.index[df_point["is_anom_point"] == 1]

rows = []
for _, r in oos.iterrows():
    t = r["Data da Ocorrencia"]
    tag = r["Tag"]
    t0, t1 = t - win, t + win
    window_anoms = anom_times[(anom_times >= t0) & (anom_times <= t1)]
    hit = len(window_anoms) > 0
    lead = None
    lead_kind = "sem_deteccao"
    if hit:
        before = window_anoms[window_anoms <= t]
        after = window_anoms[window_anoms > t]
        if len(before) > 0:
            lead = (t - before.max()).total_seconds() / 3600.0
            lead_kind = "antecedencia"
        else:
            lead = (after.min() - t).total_seconds() / 3600.0
            lead_kind = "apos_alarme"
    rows.append({
        "alarme": t, "tag": tag, "hit": hit,
        "lead_horas": lead, "tipo": lead_kind,
    })

df_res = pd.DataFrame(rows)
df_res.to_csv(os.path.join(OUT, "tabela_acertos.csv"), index=False)
print(df_res.to_string())

n_hit = df_res["hit"].sum()
print(f"\nhit_rate = {n_hit}/{len(df_res)} = {n_hit/len(df_res)*100:.1f}%")
lead_hits = df_res[(df_res["hit"]) & (df_res["tipo"] == "antecedencia")]["lead_horas"]
print(f"antecedencia media (so onde houve antecedencia real): {lead_hits.mean():.1f}h, "
      f"mediana: {lead_hits.median():.1f}h, min: {lead_hits.min():.2f}h, max: {lead_hits.max():.1f}h")
print(f"n alarmes com antecedencia real: {len(lead_hits)} / {n_hit} hits")

# --- figura: serie completa bruta com anomalias e alarmes ---
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax = axes[0]
ax.plot(df_raw.index, df_raw["TC382_03_A"], color="#184F95", linewidth=0.4, label="TC382_03_A (bruto)")

anom_in_range = anom_times[(anom_times >= df_raw.index.min()) & (anom_times <= df_raw.index.max())]
anom_vals = df_raw["TC382_03_A"].reindex(anom_in_range)
ax.scatter(anom_in_range, anom_vals, color="#D03B3B", s=4, zorder=5, label="anomalia detectada")

for _, r in oos.iterrows():
    color = "#0CA30C" if df_res.loc[df_res["alarme"] == r["Data da Ocorrencia"], "hit"].any() else "#EB6834"
ax.set_ylabel("Temperatura (°C)")
ax.set_title("TC382_03_A — série bruta completa, anomalias detectadas (EXP20, filtro 4,5min)")
ax.legend(loc="upper left", fontsize=8, frameon=False)

ax2 = axes[1]
hit_mask = df_res["hit"].values
ax2.eventplot(oos.loc[hit_mask.astype(bool), "Data da Ocorrencia"] if hit_mask.any() else [],
              lineoffsets=1, colors="#0CA30C", linewidths=1.5, label="alarme acertado")
ax2.eventplot(oos.loc[~hit_mask.astype(bool), "Data da Ocorrencia"] if (~hit_mask.astype(bool)).any() else [],
              lineoffsets=0, colors="#D03B3B", linewidths=1.5, label="alarme perdido")
ax2.set_yticks([0, 1])
ax2.set_yticklabels(["perdido", "acertado"])
ax2.set_ylim(-0.5, 1.5)
ax2.set_xlabel("tempo")
ax2.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "serie_completa_exp20.png"), dpi=180)
print("figura salva.")
