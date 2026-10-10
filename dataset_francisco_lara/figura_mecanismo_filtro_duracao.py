"""Figura pedagogica: mostra, num exemplo real, o MECANISMO do filtro
de duracao minima -- uma coincidencia pontual entre 2 canais (30s)
que cruza o limiar de votacao mas nao passa pelo filtro de 45min, vs
um caso sustentado (real) que passa.

Uso:
    PYTHONPATH=. python3 dataset_francisco_lara/figura_mecanismo_filtro_duracao.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from clearml import Task, Dataset

sys.path.insert(0, "/home/dvar/REPO_CABIUNAS/cabiunas-models")
from src.cnn1d_ae.scoring import combine_channels_vote, apply_min_duration_filter

OUT = os.path.dirname(os.path.abspath(__file__))

TID_TEMP = "805fbf34f99f4a889dbdcca7185f20a1"
TID_VIB = "7815d2cf0d07491eb1c949d555cb5de7"
TID_OLEO = "18a61687eb78412ead48c9ce31109b67"
MIN_VOTE_DURATION_MINUTES = 45.0

# exemplo curto (removido), bem isolado: 14/08/2025 14:13, T+V,
# 304h sem nenhum outro episodio removido antes e 76h depois
EXEMPLO_CURTO = pd.Timestamp("2025-08-14 14:13:00")
WIN1_START = EXEMPLO_CURTO - pd.Timedelta(hours=2.5)
WIN1_END = EXEMPLO_CURTO + pd.Timedelta(hours=2.5)

print("baixando canais...", flush=True)
t_temp = Task.get_task(task_id=TID_TEMP)
df_temp = pd.read_csv(t_temp.artifacts["mancal_temperatura_isolada/csv/point_anomalies_all.csv"].get_local_copy(), index_col=0, parse_dates=True, low_memory=False)
t_vib = Task.get_task(task_id=TID_VIB)
df_vib = pd.read_csv(t_vib.artifacts["mancal_vibracao_isolada/csv/point_anomalies_all.csv"].get_local_copy(), index_col=0, parse_dates=True, low_memory=False)
t_oleo = Task.get_task(task_id=TID_OLEO)
df_oleo = pd.read_csv(t_oleo.artifacts["oleo_pressao_isolada/csv/point_anomalies_all.csv"].get_local_copy(), index_col=0, parse_dates=True, low_memory=False)

idx = df_temp.index.intersection(df_vib.index).intersection(df_oleo.index)
canal_temp = df_temp.loc[idx, "is_anom_point"].astype(bool)
canal_vib = df_vib.loc[idx, "is_anom_point"].astype(bool)
canal_oleo = df_oleo.loc[idx, "is_anom_point"].astype(bool)

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
alarm = pd.read_csv(os.path.join(root, "alarmes_selecionados_turbina_a.csv"))
alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrência"], errors="coerce")
alarm["Tag"] = alarm["Tag Alarme"]
alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].dropna(subset=["Data da Ocorrencia"])
tags = ["PI_6240319_AL", "PAL_6240315", "PDAL_6240302", "TC382_05_A", "PAH_6240319"]
alarm_times = alarm.loc[alarm["Tag"].isin(tags), "Data da Ocorrencia"]


def alarm_channel_bool(index, alarm_times, window_hours):
    times = np.sort(pd.DatetimeIndex(pd.Series(alarm_times).dropna()).values.astype("datetime64[ns]"))
    t_arr = index.values.astype("datetime64[ns]")
    out = np.zeros(len(t_arr), dtype=bool)
    pos = np.searchsorted(times, t_arr, side="right") - 1
    valid = pos >= 0
    dt_hours = (t_arr[valid] - times[pos[valid]]).astype("timedelta64[s]").astype(np.float64) / 3600.0
    out[valid] = dt_hours <= float(window_hours)
    return pd.Series(out, index=index)


canal_alarme = alarm_channel_bool(idx, alarm_times, 24.0)
canais = {"temperatura": canal_temp, "vibracao": canal_vib, "oleo_pressao": canal_oleo, "alarme_processo": canal_alarme}
voto = combine_channels_vote(canais, min_votes=2)
df_voto = pd.DataFrame({"is_anom_point": voto.astype(int)}, index=idx)
voto_f = apply_min_duration_filter(df_voto, MIN_VOTE_DURATION_MINUTES)["is_anom_point"].astype(bool)

mask = (idx >= WIN1_START) & (idx <= WIN1_END)
idx_w = idx[mask]

fig, axes = plt.subplots(3, 1, figsize=(13, 7.5), sharex=True, gridspec_kw={"height_ratios": [1.6, 1.0, 1.0]})
fig.patch.set_facecolor("white")

labels = ["temperatura", "vibração", "óleo", "alarme processo"]
series_list = [canal_temp, canal_vib, canal_oleo, canal_alarme]
colors = ["#2A78D6", "#D67A2A", "#8E44AD", "#B06A00"]
ax0 = axes[0]
for i, (lab, s, col) in enumerate(zip(labels, series_list, colors)):
    y = i + s.loc[idx_w].astype(int).values * 0.8
    ax0.fill_between(idx_w, i, y, step="post", color=col, alpha=0.75, linewidth=0)
    ax0.text(WIN1_START, i + 0.4, lab, fontsize=9, va="center", ha="left",
              bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
ax0.set_ylim(-0.1, 4.0)
ax0.set_yticks([])
ax0.set_ylabel("canais\n(0/1)", fontsize=9.5)
ax0.set_title("Exemplo real: temperatura e vibração cruzam o limiar ao mesmo tempo por ~1min",
              fontsize=10.5, loc="left")

ax1 = axes[1]
ax1.fill_between(idx_w, 0, voto.loc[idx_w].astype(int).values, step="post", color="#D03B3B", alpha=0.75)
ax1.set_ylim(0, 1.3)
ax1.set_yticks([0, 1])
ax1.set_ylabel("votação ≥2\n(ANTES do filtro)", fontsize=9)
ax1.set_title("Sem o filtro de duração: este instante seria classificado como anomalia (falso positivo)",
              fontsize=10.5, loc="left", color="#D03B3B")

ax2 = axes[2]
ax2.fill_between(idx_w, 0, voto_f.loc[idx_w].astype(int).values, step="post", color="#0CA30C", alpha=0.75)
ax2.set_ylim(0, 1.3)
ax2.set_yticks([0, 1])
ax2.set_ylabel("votação ≥2\n(DEPOIS do filtro)", fontsize=9)
ax2.set_title(f"Com o filtro de {MIN_VOTE_DURATION_MINUTES:.0f}min: a coincidência pontual é descartada (fica em zero)",
              fontsize=10.5, loc="left", color="#0CA30C")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh%M"))
ax2.set_xlabel("tempo")

for ax in axes:
    ax.set_xlim(WIN1_START, WIN1_END)
    ax.grid(alpha=0.25, axis="x")

fig.suptitle("O que o filtro de duração mínima elimina -- mecanismo em um exemplo real (14/08/2025)",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])

out_path = os.path.join(OUT, "mecanismo_filtro_duracao.png")
fig.savefig(out_path, dpi=170, facecolor="white")
print("figura salva em", out_path)
