"""Figura pedagogica: passo a passo da camada de decisao pos-treino,
sobre uma janela real (TRIP de 04/11/2025, causa oleo) -- mostra os 4
canais binarios, a contagem de votos, e a decisao final apos
refratario.

Uso:
    PYTHONPATH=. python3 dataset_francisco_lara/figura_passo_a_passo_decisao.py
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
from src.cnn1d_ae.scoring import combine_channels_vote, apply_refractory, apply_min_duration_filter

OUT = os.path.dirname(os.path.abspath(__file__))

TID_TEMP = "805fbf34f99f4a889dbdcca7185f20a1"
KEY_TEMP_POINT = "mancal_temperatura_isolada/csv/point_anomalies_all.csv"
TID_VIB = "7815d2cf0d07491eb1c949d555cb5de7"
KEY_VIB_POINT = "mancal_vibracao_isolada/csv/point_anomalies_all.csv"
TID_OLEO = "18a61687eb78412ead48c9ce31109b67"
KEY_OLEO_POINT = "oleo_pressao_isolada/csv/point_anomalies_all.csv"

ALARM_CATALOG_DATASET_ID = "a97ba56ba14840fbb1125c2a82f883c9"
ALARM_CATALOG_FILE = "alarmes_selecionados_turbina_a.csv"
ALARM_CHANNEL_TAGS = ["PI_6240319_AL", "PAL_6240315", "PDAL_6240302", "TC382_05_A", "PAH_6240319"]
ALARM_WINDOW_HOURS = 24.0
REFRACTORY_HOURS = 48.0
MIN_VOTES = 2
MIN_VOTE_DURATION_MINUTES = 45.0

TRIP_TIME = pd.Timestamp("2025-11-04 06:22:00")
WIN_START = TRIP_TIME - pd.Timedelta(hours=90)
WIN_END = TRIP_TIME + pd.Timedelta(hours=12)


def alarm_channel_bool(index, alarm_times, window_hours):
    times = np.sort(pd.DatetimeIndex(pd.Series(alarm_times).dropna()).values.astype("datetime64[ns]"))
    t_arr = index.values.astype("datetime64[ns]")
    out = np.zeros(len(t_arr), dtype=bool)
    if len(times) == 0:
        return pd.Series(out, index=index)
    pos = np.searchsorted(times, t_arr, side="right") - 1
    valid = pos >= 0
    dt_hours = (t_arr[valid] - times[pos[valid]]).astype("timedelta64[s]").astype(np.float64) / 3600.0
    out[valid] = dt_hours <= float(window_hours)
    return pd.Series(out, index=index)


print("baixando canais...", flush=True)
t_temp = Task.get_task(task_id=TID_TEMP)
df_temp = pd.read_csv(t_temp.artifacts[KEY_TEMP_POINT].get_local_copy(), index_col=0, parse_dates=True, low_memory=False)
t_vib = Task.get_task(task_id=TID_VIB)
df_vib = pd.read_csv(t_vib.artifacts[KEY_VIB_POINT].get_local_copy(), index_col=0, parse_dates=True, low_memory=False)
t_oleo = Task.get_task(task_id=TID_OLEO)
df_oleo = pd.read_csv(t_oleo.artifacts[KEY_OLEO_POINT].get_local_copy(), index_col=0, parse_dates=True, low_memory=False)

idx = df_temp.index.intersection(df_vib.index).intersection(df_oleo.index)
canal_temp = df_temp.loc[idx, "is_anom_point"].astype(bool)
canal_vib = df_vib.loc[idx, "is_anom_point"].astype(bool)
canal_oleo = df_oleo.loc[idx, "is_anom_point"].astype(bool)

root = Dataset.get(dataset_id=ALARM_CATALOG_DATASET_ID).get_local_copy()
alarm = pd.read_csv(os.path.join(root, ALARM_CATALOG_FILE))
alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrência"], errors="coerce")
alarm["Tag"] = alarm["Tag Alarme"]
alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].dropna(subset=["Data da Ocorrencia"])
alarm_times = alarm.loc[alarm["Tag"].isin(ALARM_CHANNEL_TAGS), "Data da Ocorrencia"]
canal_alarme = alarm_channel_bool(idx, alarm_times, ALARM_WINDOW_HOURS)

canais = {"temperatura": canal_temp, "vibracao": canal_vib, "oleo_pressao": canal_oleo, "alarme_processo": canal_alarme}
voto = combine_channels_vote(canais, min_votes=MIN_VOTES)
df_voto = pd.DataFrame({"is_anom_point": voto.astype(int)}, index=idx)
voto_filtrado = apply_min_duration_filter(df_voto, MIN_VOTE_DURATION_MINUTES)["is_anom_point"].astype(bool)
decisao = apply_refractory(voto_filtrado, refractory_minutes=REFRACTORY_HOURS * 60.0)
n_votos = (canal_temp.astype(int) + canal_vib.astype(int) + canal_oleo.astype(int) + canal_alarme.astype(int))

mask = (idx >= WIN_START) & (idx <= WIN_END)
idx_w = idx[mask]

fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2, 1.0, 1.0]})
fig.patch.set_facecolor("white")

# painel 1: 4 canais binarios empilhados
labels = ["temperatura", "vibração", "óleo", "alarme processo"]
series_list = [canal_temp, canal_vib, canal_oleo, canal_alarme]
colors = ["#2A78D6", "#D67A2A", "#8E44AD", "#B06A00"]
ax0 = axes[0]
for i, (lab, s, col) in enumerate(zip(labels, series_list, colors)):
    y = i + s.loc[idx_w].astype(int).values * 0.8
    ax0.fill_between(idx_w, i, y, step="post", color=col, alpha=0.75, linewidth=0)
    ax0.text(WIN_START, i + 0.4, lab, fontsize=9, va="center", ha="left",
              bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
ax0.set_yticks([])
ax0.set_ylabel("canais (ativo = bloco colorido)", fontsize=9.5)
ax0.set_title("Passo 1: cada canal já gateado, binário (0/1)", fontsize=10.5, loc="left")

# painel 2: contagem de votos
ax1 = axes[1]
ax1.fill_between(idx_w, 0, n_votos.loc[idx_w].values, step="post", color="#555555", alpha=0.6)
ax1.axhline(MIN_VOTES, color="#D03B3B", linewidth=1.6, linestyle="--", label=f"limiar de votação (≥{MIN_VOTES})")
ax1.set_ylim(0, 4.3)
ax1.set_yticks([0, 1, 2, 3, 4])
ax1.set_ylabel("nº de canais\nconcordando", fontsize=9.5)
ax1.legend(loc="upper left", fontsize=8)
ax1.set_title("Passo 2: soma de votos entre os 4 canais", fontsize=10.5, loc="left")

# painel 3: voto apos filtro de duracao minima (pre-refratario)
ax2 = axes[2]
ax2.fill_between(idx_w, 0, voto_filtrado.loc[idx_w].astype(int).values, step="post", color="#8A6D3B", alpha=0.8)
ax2.set_ylim(0, 1.3)
ax2.set_yticks([0, 1])
ax2.set_ylabel("voto após\nfiltro 45min", fontsize=9.5)
ax2.set_title(f"Passo 3: filtro de duração mínima ({MIN_VOTE_DURATION_MINUTES:.0f}min) -- descarta coincidências pontuais",
              fontsize=10.5, loc="left")

# painel 4: decisao final apos refratario + TRIP
ax3 = axes[3]
ax3.fill_between(idx_w, 0, decisao.loc[idx_w].astype(int).values, step="post", color="#0CA30C", alpha=0.8)
ax3.axvline(TRIP_TIME, color="#7A1FA2", linewidth=1.8, linestyle="--", label="TRIP real")
ax3.set_ylim(0, 1.3)
ax3.set_yticks([0, 1])
ax3.set_ylabel("decisão\nfinal", fontsize=9.5)
ax3.set_title("Passo 4: refratário 48h aplicado → decisão final", fontsize=10.5, loc="left")
ax3.legend(loc="upper left", fontsize=8)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
ax3.set_xlabel("tempo")

for ax in axes:
    ax.set_xlim(WIN_START, WIN_END)
    ax.grid(alpha=0.25, axis="x")

fig.suptitle("Camada de decisão pós-treino, passo a passo -- janela real ao redor do TRIP de 04/11/2025",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])

out_path = os.path.join(OUT, "passo_a_passo_decisao.png")
fig.savefig(out_path, dpi=170, facecolor="white")
print("figura salva em", out_path)
