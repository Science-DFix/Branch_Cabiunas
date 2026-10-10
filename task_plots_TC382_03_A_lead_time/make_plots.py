import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

OUT_DIR = Path(__file__).parent
SCRATCH = Path("/tmp/claude-1000/-home-dvar-REPO-CABIUNAS-cabiunas-models/aad24b91-d1c4-46b2-8b59-487614051d5a/scratchpad")

series = pd.read_pickle(SCRATCH / "tc382_raw_series.pkl")
pts = pd.read_csv(SCRATCH / "exp5_seed_sweep/TC382_03_A_univariado__csv__point_anomalies_all.csv",
                   parse_dates=["data_datetime"], index_col="data_datetime")

alarm = pd.read_csv(
    "/home/dvar/.clearml/cache/storage_manager/datasets/ds_424e5b589e13402d9d95371a317e85c9/alarmes_record_2025_tags_modelo.csv"
).drop_duplicates()
alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrencia"], errors="coerce")
alarm_all = alarm.loc[alarm["Tag"] == "TC382_03_A", "Data da Ocorrencia"].dropna().sort_values()
oos_start = pd.Timestamp("2025-05-01")
alarm_eval = alarm_all[alarm_all >= oos_start]

# ---------- recalcula lead time (deduplicado) ----------
win = pd.Timedelta(minutes=1440)
anom_idx = pts.index[pts["is_anom_point"] == 1]
rows = []
for t in alarm_eval:
    window_pts = anom_idx[(anom_idx >= t - win) & (anom_idx <= t + win)]
    if len(window_pts) == 0:
        rows.append({"alarm_time": t, "hit": False, "first_anom": None, "lead_minutes": None})
        continue
    first = window_pts.min()
    rows.append({"alarm_time": t, "hit": True, "first_anom": first, "lead_minutes": (t - first).total_seconds() / 60.0})
lead_detail = pd.DataFrame(rows)
lead_detail.to_csv(OUT_DIR / "lead_time_detail.csv", index=False)
print("alarmes OOS (dedup):", len(lead_detail), "| preditivos:",
      ((lead_detail.hit) & (lead_detail.lead_minutes > 0)).sum())

# ---------- episodios de falso positivo ----------
near_alarm = pd.Series(False, index=pts.index)
for t in alarm_all:
    near_alarm |= (pts.index >= t - win) & (pts.index <= t + win)
eval_mask = pts.index >= oos_start
fp_mask = (pts["is_anom_point"] == 1) & (~near_alarm) & (pts["operational_state"] == "on") & eval_mask
fp_times = pts.index[fp_mask]
gaps = pd.Series(fp_times).diff() > pd.Timedelta(minutes=5)
episode_id = gaps.cumsum()
ep_df = pd.DataFrame({"time": fp_times, "episode": episode_id.values})
fp_episodes = ep_df.groupby("episode")["time"].agg(["min", "max", "count"])
fp_episodes["duration_min"] = (fp_episodes["max"] - fp_episodes["min"]).dt.total_seconds() / 60 + 0.5
fp_episodes = fp_episodes.sort_values("duration_min", ascending=False)
fp_episodes.to_csv(OUT_DIR / "fp_episodes.csv")
print("episodios FP:", len(fp_episodes))

STATE_COLORS = {"on": None, "off_longo": "#d0d0d0", "off_curto": "#e8c48a", "transiente": "#f2a6a6"}


def shade_state(ax, state_series, start, end):
    st = state_series.loc[(state_series.index >= start) & (state_series.index <= end)]
    if st.empty:
        return
    changes = st.ne(st.shift()).cumsum()
    for _, grp in st.groupby(changes):
        s, e = grp.index.min(), grp.index.max()
        color = STATE_COLORS.get(grp.iloc[0])
        if color:
            ax.axvspan(s, e, color=color, alpha=0.35, zorder=0)


# ---------- 1. Overview ----------
# Nota: off_curto/transiente sao milhares de blocos curtos (~1h) espalhados
# pelos 6 meses -- desenhados como axvspan nessa escala de tempo, ficam tao
# proximos uns dos outros que parecem uma faixa solida (artefato de
# renderizacao, nao do dado). Aqui so sombreamos off_longo (poucos blocos
# grandes, sombra real) e marcamos off_curto/transiente como uma rug strip
# fina no rodape, sem inflar visualmente a area coberta.
ov = series.loc[oos_start:].resample("15min").mean()
ov_state = pts["operational_state"].loc[oos_start:]
anom_ov = pts.loc[(pts.index >= oos_start) & (pts["is_anom_point"] == 1)]
anom_vals_ov = series.reindex(anom_ov.index)

fig, ax = plt.subplots(figsize=(20, 5))
off_longo_state = ov_state.where(ov_state == "off_longo")
shade_state(ax, off_longo_state, oos_start, series.index.max())
ax.plot(ov.index, ov.values, color="#1f77b4", linewidth=0.7, label="TC382_03_A (raw, media 15min)")
ax.scatter(anom_vals_ov.index, anom_vals_ov.values, color="red", s=4, alpha=0.5, label="Pontos anomalos (iforest)")
for t in alarm_eval:
    ax.axvline(t, color="green", linestyle="--", linewidth=0.8, alpha=0.6)
short_mask = ov_state.isin(["off_curto", "transiente"])
short_times = ov_state.index[short_mask]
if len(short_times):
    y_rug = series.min() - 0.03 * (series.max() - series.min())
    ax.scatter(short_times, [y_rug] * len(short_times), marker="|", color="#c07a1e", s=8, alpha=0.5,
               label="off_curto / transiente (marcador, nao sombra)")
ax.set_title("TC382_03_A -- periodo OOS completo (2025-05-01 a 2025-10-31)\n"
             "cinza=off_longo (sombra real), tracos laranja no rodape=off_curto/transiente (~1h cada, "
             "muito curtos p/ sombrear nessa escala), linhas verdes=alarmes reais")
ax.set_ylabel("TC382_03_A")
ax.grid(alpha=0.2)
ax.legend(loc="upper right", fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
fig.savefig(OUT_DIR / "00_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("overview salvo")

# ---------- 2. Histograma de duracao FP ----------
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(fp_episodes["duration_min"], bins=[0, 1, 2, 5, 10, 30, 60, 120, 300, 600, 2500], color="#d62728", alpha=0.75)
ax.set_xscale("symlog")
ax.set_xlabel("duracao do episodio (min, escala log)")
ax.set_ylabel("quantidade de episodios")
ax.set_title(f"Distribuicao de duracao dos {len(fp_episodes)} episodios de falso positivo")
ax.grid(alpha=0.2)
fig.savefig(OUT_DIR / "01_fp_duration_histogram.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("histograma salvo")

# ---------- 3. Zoom nos casos preditivos (antecedencia real) ----------
predictive = lead_detail[(lead_detail["hit"]) & (lead_detail["lead_minutes"] > 0)].sort_values("alarm_time")
for i, (_, row) in enumerate(predictive.iterrows(), start=1):
    t = row["alarm_time"]
    lead_h = row["lead_minutes"] / 60
    start = t - pd.Timedelta(hours=30)
    end = t + pd.Timedelta(hours=6)
    s = series.loc[start:end]
    p = pts.loc[start:end]
    anom = p[p["is_anom_point"] == 1]
    anom_v = series.reindex(anom.index)

    fig, ax = plt.subplots(figsize=(12, 4))
    shade_state(ax, p["operational_state"], start, end)
    ax.plot(s.index, s.values, color="#1f77b4", linewidth=1.0)
    ax.scatter(anom_v.index, anom_v.values, color="red", s=14, zorder=3, label="Anomalia")
    ax.axvline(t, color="green", linestyle="--", linewidth=1.5, label=f"Alarme ({t:%d/%m %H:%M})")
    ax.axvline(row["first_anom"], color="darkred", linestyle=":", linewidth=1.5,
               label=f"1a anomalia ({row['first_anom']:%d/%m %H:%M})")
    ax.set_title(f"Antecedencia: {lead_h:.1f}h antes do alarme de {t:%d/%m/%Y %H:%M}")
    ax.set_ylabel("TC382_03_A")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")
    fname = f"pred_{i:02d}_{t:%Y%m%d_%H%M}.png"
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
print("zooms preditivos salvos:", len(predictive))

# ---------- 4. Zoom nos episodios de FP mais longos ----------
top_fp = fp_episodes.head(15)
for i, (_, row) in enumerate(top_fp.iterrows(), start=1):
    start = row["min"] - pd.Timedelta(hours=2)
    end = row["max"] + pd.Timedelta(hours=2)
    s = series.loc[start:end]
    p = pts.loc[start:end]
    anom = p[p["is_anom_point"] == 1]
    anom_v = series.reindex(anom.index)

    fig, ax = plt.subplots(figsize=(12, 4))
    shade_state(ax, p["operational_state"], start, end)
    ax.plot(s.index, s.values, color="#1f77b4", linewidth=1.0)
    ax.scatter(anom_v.index, anom_v.values, color="red", s=10, zorder=3, label="Anomalia (falso positivo)")
    ax.axvspan(row["min"], row["max"], color="red", alpha=0.08, zorder=0)
    ax.set_title(f"FP episodio {row['min']:%d/%m %H:%M} -> {row['max']:%d/%m %H:%M}  "
                 f"(duracao {row['duration_min']:.0f} min, {int(row['count'])} pontos)")
    ax.set_ylabel("TC382_03_A")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")
    fname = f"fp_{i:02d}_{row['min']:%Y%m%d_%H%M}.png"
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
print("zooms de falso positivo salvos:", len(top_fp))

report = {
    "n_alarms_oos_dedup": int(len(lead_detail)),
    "n_hit": int(lead_detail["hit"].sum()),
    "n_predictive": int(((lead_detail.hit) & (lead_detail.lead_minutes > 0)).sum()),
    "n_reactive_only": int(((lead_detail.hit) & (lead_detail.lead_minutes <= 0)).sum()),
    "n_missed": int((~lead_detail["hit"]).sum()),
    "n_fp_episodes": int(len(fp_episodes)),
}
(OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(json.dumps(report, indent=2, ensure_ascii=False))
