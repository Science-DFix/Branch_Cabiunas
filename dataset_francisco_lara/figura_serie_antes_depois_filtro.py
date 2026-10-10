"""Serie bruta com 3 categorias de pontos: acerto de TRIP (verde),
falso positivo remanescente depois do filtro de 45min (preto), e
pontos que ERAM falso positivo antes do filtro e foram eliminados por
ele (laranja) -- visualiza o efeito real do filtro sobre a serie
inteira, nao so um exemplo.

Uso:
    PYTHONPATH=. python3 dataset_francisco_lara/figura_serie_antes_depois_filtro.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from clearml import Task, Dataset

sys.path.insert(0, "/home/dvar/REPO_CABIUNAS/cabiunas-models")
from src.cnn1d_ae.scoring import combine_channels_vote, apply_refractory, apply_min_duration_filter

OUT = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(OUT, "..", "runs_pipeline_unificada_final")
SENSORS = ["954005_624_TI_0305", "TC382_03_A", "T5_AVG_A", "954005_624_PI_0308"]
LABELS = {
    "954005_624_TI_0305": "TI_0305 (°C)",
    "TC382_03_A": "TC382_03_A (°C)",
    "T5_AVG_A": "T5_AVG_A",
    "954005_624_PI_0308": "PI_0308 (pressão óleo)",
}

TID_TEMP = "805fbf34f99f4a889dbdcca7185f20a1"
TID_VIB = "7815d2cf0d07491eb1c949d555cb5de7"
TID_OLEO = "18a61687eb78412ead48c9ce31109b67"
MIN_VOTE_DURATION_MINUTES = 45.0
REFRACTORY_HOURS = 48.0

print("baixando canais e recalculando voto ANTES do filtro...", flush=True)
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
voto_bruto = combine_channels_vote(canais, min_votes=2)
df_voto = pd.DataFrame({"is_anom_point": voto_bruto.astype(int)}, index=idx)
voto_filtrado = apply_min_duration_filter(df_voto, MIN_VOTE_DURATION_MINUTES)["is_anom_point"].astype(bool)

# pontos que eram voto>=2 ANTES do filtro mas foram zerados por ele
removido_pelo_filtro = voto_bruto & ~voto_filtrado

print("lendo decisao final (pos-filtro, pos-refratario)...", flush=True)
df_final = pd.read_csv(os.path.join(RUN_DIR, "point_anomalies_final.csv"), index_col=0, parse_dates=True)
cls = pd.read_csv(os.path.join(RUN_DIR, "episodios_classificados.csv"), parse_dates=["start", "end", "falha_associada"])

root2 = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
raw_path = os.path.join(root2, "sensores_full_2024_2026_30s.csv")
print("lendo series brutas (em chunks, RAM limitada)...", flush=True)
chunks = []
for chunk in pd.read_csv(raw_path, usecols=["data_datetime"] + SENSORS, dtype=str, chunksize=200_000):
    chunk["data_datetime"] = pd.to_datetime(chunk["data_datetime"], errors="coerce")
    for c in SENSORS:
        chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
    chunk = chunk.dropna(subset=["data_datetime"])
    chunk = chunk.loc[(chunk["data_datetime"] >= df_final.index.min()) & (chunk["data_datetime"] <= df_final.index.max())]
    if len(chunk):
        chunks.append(chunk)
df_raw = pd.concat(chunks, ignore_index=True).set_index("data_datetime").sort_index()
del chunks

falhas = pd.read_csv(os.path.join(OUT, "alarmes_francisco_falhas.csv"))
falhas["Data da Ocorrência"] = pd.to_datetime(falhas["Data da Ocorrência"])
trip_times = falhas.loc[falhas["Tag Alarme"] == "FALHA_CURADA", "Data da Ocorrência"].sort_values()
trip_times = trip_times.loc[(trip_times >= df_final.index.min()) & (trip_times <= df_final.index.max())]

det_starts = cls.loc[cls["classe"] == "deteccao", "start"]
fp_starts = cls.loc[cls["classe"] == "falso_positivo", "start"]
fp_ends = cls.loc[cls["classe"] == "falso_positivo", "end"]
det_ends = cls.loc[cls["classe"] == "deteccao", "end"]

anom_times_final = df_final.index[df_final["is_anom_point"] == 1]


def in_any_episode(times, starts, ends):
    starts_arr = starts.values.astype("datetime64[ns]")
    ends_arr = ends.values.astype("datetime64[ns]")
    order = np.argsort(starts_arr)
    starts_arr, ends_arr = starts_arr[order], ends_arr[order]
    t_arr = times.values.astype("datetime64[ns]")
    pos = np.searchsorted(starts_arr, t_arr, side="right") - 1
    out = np.zeros(len(t_arr), dtype=bool)
    valid = pos >= 0
    out[valid] = t_arr[valid] <= ends_arr[pos[valid]]
    return out


is_det = in_any_episode(anom_times_final, det_starts, det_ends) if len(det_starts) else np.zeros(len(anom_times_final), dtype=bool)
is_fp = in_any_episode(anom_times_final, fp_starts, fp_ends) if len(fp_starts) else np.zeros(len(anom_times_final), dtype=bool)

det_times = anom_times_final[is_det]
fp_times = anom_times_final[is_fp & ~is_det]
removido_times = removido_pelo_filtro.loc[idx].index[removido_pelo_filtro.loc[idx].values]
print(f"acerto: {len(det_times)}  FP remanescente: {len(fp_times)}  removido pelo filtro: {len(removido_times)}", flush=True)

off = (df_final["operational_state"] != "on")
off = off.reindex(df_raw.index).fillna(False)
off_change = off.astype(int).diff().fillna(off.astype(int).iloc[0])
starts_off = off.index[off_change == 1]
ends_off = off.index[off_change == -1]
if off.iloc[0]:
    starts_off = starts_off.insert(0, off.index[0])
if len(starts_off) > len(ends_off):
    ends_off = ends_off.append(pd.DatetimeIndex([off.index[-1]]))

fig, axes = plt.subplots(4, 1, figsize=(16, 13), sharex=True)
fig.patch.set_facecolor("white")
for ax, sensor in zip(axes, SENSORS):
    ax.set_facecolor("white")
    ax.plot(df_raw.index, df_raw[sensor], color="#184F95", linewidth=0.3, zorder=1)
    for s, e in zip(starts_off, ends_off):
        ax.axvspan(s, e, color="#B0B0B0", alpha=0.18, zorder=0, linewidth=0)
    for i, t in enumerate(trip_times):
        ax.axvline(t, color="#7A1FA2", linewidth=1.5, linestyle="--", alpha=0.9, zorder=3,
                   label=f"TRIP real (n={len(trip_times)})" if i == 0 else None)

    vals_removido = df_raw[sensor].reindex(removido_times.intersection(df_raw.index))
    ax.scatter(vals_removido.index, vals_removido.values, color="#E8A33D", s=10, zorder=3, alpha=0.65,
               label=f"removido pelo filtro de 45min (n={len(removido_times)})")

    vals_fp = df_raw[sensor].reindex(fp_times.intersection(df_raw.index))
    ax.scatter(vals_fp.index, vals_fp.values, color="black", s=16, zorder=4,
               label=f"falso positivo remanescente (n episódios={cls['classe'].eq('falso_positivo').sum()})")

    vals_ok = df_raw[sensor].reindex(det_times.intersection(df_raw.index))
    ax.scatter(vals_ok.index, vals_ok.values, color="#0CA30C", s=22, zorder=5, edgecolor="black", linewidth=0.4,
               label="acerto de TRIP (8 de 8)")

    ax.set_ylabel(LABELS[sensor], fontsize=9)
    ax.legend(loc="upper left", fontsize=7.2, frameon=True)

axes[0].set_title(
    "Efeito do filtro de duração mínima (45min) sobre a série inteira -- antes (laranja, eliminado) vs depois (preto, remanescente)\n"
    "fundo branco = ligado, cinza claro = desligado | roxo = TRIP real | verde = acerto | laranja = seria FP mas foi filtrado | preto = FP remanescente",
    fontsize=10.0,
)
axes[-1].set_xlabel("tempo")
for ax in axes:
    ax.set_xlim(df_raw.index.min(), df_raw.index.max())
fig.tight_layout()

out_path = os.path.join(OUT, "serie_antes_depois_filtro.png")
fig.savefig(out_path, dpi=170, facecolor="white")
print("\nfigura salva em", out_path)
