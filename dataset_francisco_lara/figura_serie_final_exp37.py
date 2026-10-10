"""Serie bruta (T5_AVG_A, TC382_03_A, TI_0305) classificada com a
DECISAO FINAL do EXP37 (votacao multicanal + refratario) -- mostra
quanto o resultado final ficou mais limpo que qualquer estagio
anterior: 8 TRIPs certos, e so os poucos FP genuinos que sobraram
depois da votacao+refratario.

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/dataset_francisco_lara/figura_serie_final_exp37.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from clearml import Dataset

OUT = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(OUT, "..", "runs_exp37_multicanal")
SENSORS = ["954005_624_TI_0305", "TC382_03_A", "T5_AVG_A"]
LABELS = {"954005_624_TI_0305": "TI_0305 (°C)", "TC382_03_A": "TC382_03_A (°C)", "T5_AVG_A": "T5_AVG_A"}

print("lendo decisao final do EXP37...", flush=True)
df_final = pd.read_csv(os.path.join(RUN_DIR, "point_anomalies_multicanal.csv"), index_col=0, parse_dates=True)
cls = pd.read_csv(os.path.join(RUN_DIR, "episodios_classificados.csv"), parse_dates=["start", "end", "falha_associada"])

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
raw_path = os.path.join(root, "sensores_full_2024_2026_30s.csv")
print("lendo series brutas...", flush=True)
df_raw = pd.read_csv(raw_path, usecols=["data_datetime"] + SENSORS)
df_raw["data_datetime"] = pd.to_datetime(df_raw["data_datetime"], errors="coerce")
for c in SENSORS:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
df_raw = df_raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
df_raw = df_raw.loc[(df_raw.index >= df_final.index.min()) & (df_raw.index <= df_final.index.max())]

falhas = pd.read_csv(os.path.join(OUT, "alarmes_francisco_falhas.csv"))
falhas["Data da Ocorrência"] = pd.to_datetime(falhas["Data da Ocorrência"])
trip_times = falhas.loc[falhas["Tag Alarme"] == "FALHA_CURADA", "Data da Ocorrência"].sort_values()
trip_times = trip_times.loc[(trip_times >= df_final.index.min()) & (trip_times <= df_final.index.max())]

# separa episodios de deteccao (verde, contam pra 8/8) dos falso_positivo (preto)
det_starts = cls.loc[cls["classe"] == "deteccao", "start"]
fp_starts = cls.loc[cls["classe"] == "falso_positivo", "start"]
fp_ends = cls.loc[cls["classe"] == "falso_positivo", "end"]
det_ends = cls.loc[cls["classe"] == "deteccao", "end"]

anom_times = df_final.index[df_final["is_anom_point"] == 1]


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


is_det = in_any_episode(anom_times, det_starts, det_ends) if len(det_starts) else np.zeros(len(anom_times), dtype=bool)
is_fp = in_any_episode(anom_times, fp_starts, fp_ends) if len(fp_starts) else np.zeros(len(anom_times), dtype=bool)

det_times = anom_times[is_det]
fp_times = anom_times[is_fp & ~is_det]
print(f"pontos em episodios de deteccao: {len(det_times)}  pontos em episodios FP: {len(fp_times)}", flush=True)

# blocos de "desligado"
off = (df_final["operational_state"] != "on")
off = off.reindex(df_raw.index).fillna(False)
off_change = off.astype(int).diff().fillna(off.astype(int).iloc[0])
starts_off = off.index[off_change == 1]
ends_off = off.index[off_change == -1]
if off.iloc[0]:
    starts_off = starts_off.insert(0, off.index[0])
if len(starts_off) > len(ends_off):
    ends_off = ends_off.append(pd.DatetimeIndex([off.index[-1]]))

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig.patch.set_facecolor("white")
for ax, sensor in zip(axes, SENSORS):
    ax.set_facecolor("white")
    ax.plot(df_raw.index, df_raw[sensor], color="#184F95", linewidth=0.3, zorder=1)
    for s, e in zip(starts_off, ends_off):
        ax.axvspan(s, e, color="#B0B0B0", alpha=0.18, zorder=0, linewidth=0)
    for i, t in enumerate(trip_times):
        ax.axvline(t, color="#7A1FA2", linewidth=1.5, linestyle="--", alpha=0.9, zorder=3,
                   label=f"ocorrência de TRIP (n={len(trip_times)})" if i == 0 else None)
    vals_fp = df_raw[sensor].reindex(fp_times.intersection(df_raw.index))
    ax.scatter(vals_fp.index, vals_fp.values, color="black", s=16, zorder=4,
               label=f"falso positivo final (n episódios FP={cls['classe'].eq('falso_positivo').sum()})")
    vals_ok = df_raw[sensor].reindex(det_times.intersection(df_raw.index))
    ax.scatter(vals_ok.index, vals_ok.values, color="#0CA30C", s=22, zorder=5, edgecolor="black", linewidth=0.4,
               label=f"acerto de TRIP (8 de 8 episódios)")
    ax.set_ylabel(LABELS[sensor], fontsize=9)
    ax.legend(loc="upper left", fontsize=7.5, frameon=True)

axes[0].set_title(
    "EXP37 -- decisão final (votação ≥2 canais + refratário 48h): fundo branco = ligado, cinza claro = desligado\n"
    "roxo = TRIP curado (8 no período) | verde = acerto | preto = falso positivo remanescente",
    fontsize=10.5,
)
axes[-1].set_xlabel("tempo")
for ax in axes:
    ax.set_xlim(df_raw.index.min(), df_raw.index.max())
fig.tight_layout()

out_path = os.path.join(OUT, "serie_final_exp37.png")
fig.savefig(out_path, dpi=170, facecolor="white")
print("\nfigura salva em", out_path)
