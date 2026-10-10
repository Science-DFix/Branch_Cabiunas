"""3 series brutas empilhadas (TI_0305, TC382_03_A, T5_AVG_A) do nosso
proprio dataset de producao (30s), com:
  - sombreado cinza nos periodos "desligado" (operational_state != on)
  - linha vertical tracejada em cada ocorrencia de TRIP curada pelo
    Francisco (alarmes_francisco_falhas.csv, tag FALHA_CURADA)
  - bolinha verde = ponto de anomalia (is_anom_point=1) perto (+-24h)
    de um TRIP real -- acerto
  - bolinha amarela = ponto de anomalia longe de qualquer TRIP -- FP

Usa o point_anomalies_all.csv ja calculado do EXP28 (mancal, com veto
de sensor congelado) -- mesmo dominio do nosso dado bruto de 30s.

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/dataset_francisco_lara/gen_figura_ti0305_tc382_t5.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from clearml import Dataset

OUT = os.path.dirname(os.path.abspath(__file__))
POINT_CSV = "/home/dvar/.clearml/cache/storage_manager/global/c27dfd135eb394c5511ce610bd890c1e.point_anomalies_all.csv"
SENSORS = ["954005_624_TI_0305", "TC382_03_A", "T5_AVG_A"]
LABELS = {
    "954005_624_TI_0305": "TI_0305 (°C)",
    "TC382_03_A": "TC382_03_A (°C)",
    "T5_AVG_A": "T5_AVG_A",
}
DATA_END = pd.Timestamp("2026-04-20 23:59:59")

print("lendo point_anomalies_all.csv (EXP28)...", flush=True)
df_point = pd.read_csv(POINT_CSV, index_col=0, parse_dates=True)
df_point = df_point.loc[df_point.index <= DATA_END]
anom_times = df_point.index[df_point["is_anom_point"] == 1]
print(f"anomalias detectadas: {len(anom_times)}", flush=True)

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
raw_path = os.path.join(root, "sensores_full_2024_2026_30s.csv")
print("lendo series brutas...", flush=True)
df_raw = pd.read_csv(raw_path, usecols=["data_datetime"] + SENSORS)
df_raw["data_datetime"] = pd.to_datetime(df_raw["data_datetime"], errors="coerce")
for c in SENSORS:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
df_raw = df_raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
df_raw = df_raw.loc[(df_raw.index >= "2024-07-01") & (df_raw.index <= DATA_END)]

falhas = pd.read_csv(os.path.join(OUT, "..", "dataset_francisco_lara", "alarmes_francisco_falhas.csv"))
falhas["Data da Ocorrência"] = pd.to_datetime(falhas["Data da Ocorrência"])
trip_times = falhas.loc[falhas["Tag Alarme"] == "FALHA_CURADA", "Data da Ocorrência"].sort_values()
trip_times = trip_times.loc[(trip_times >= df_raw.index.min()) & (trip_times <= df_raw.index.max())]
print(f"ocorrencias de TRIP curadas no periodo: {len(trip_times)}", flush=True)

alarm = pd.read_csv(os.path.join(root, "alarmes_selecionados_turbina_a.csv"))
alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrência"], errors="coerce")
alarm["Tag"] = alarm["Tag Alarme"]
alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].copy()
alarm = alarm.dropna(subset=["Data da Ocorrencia"]).sort_values("Data da Ocorrencia")
outros_times = alarm["Data da Ocorrencia"]
outros_times = outros_times.loc[(outros_times >= df_raw.index.min()) & (outros_times <= df_raw.index.max())]
print(f"ocorrencias no catalogo completo (47 tags) no periodo: {len(outros_times)}", flush=True)

win = pd.Timedelta(hours=24)
trip_arr = trip_times.values.astype("datetime64[ns]")
outros_arr = outros_times.values.astype("datetime64[ns]")


def near_any(t_arr, ref_sorted, win):
    if len(ref_sorted) == 0 or len(t_arr) == 0:
        return np.zeros(len(t_arr), dtype=bool)
    pos = np.searchsorted(ref_sorted, t_arr)
    out = np.zeros(len(t_arr), dtype=bool)
    for i in range(len(t_arr)):
        p = pos[i]
        cands = []
        if p > 0:
            cands.append(ref_sorted[p - 1])
        if p < len(ref_sorted):
            cands.append(ref_sorted[p])
        out[i] = any(abs(t_arr[i] - c) <= win for c in cands)
    return out


near_trip = near_any(anom_times.values, trip_arr, win)


def nearest_alarm(t_arr, ref_sorted, win):
    """Pra cada tempo em t_arr, retorna o timestamp do alarme mais
    proximo em ref_sorted se estiver dentro de win, senao NaT."""
    out = np.full(len(t_arr), np.datetime64("NaT"), dtype="datetime64[ns]")
    if len(ref_sorted) == 0 or len(t_arr) == 0:
        return out
    pos = np.searchsorted(ref_sorted, t_arr)
    for i in range(len(t_arr)):
        p = pos[i]
        cands = []
        if p > 0:
            cands.append(ref_sorted[p - 1])
        if p < len(ref_sorted):
            cands.append(ref_sorted[p])
        if not cands:
            continue
        best = min(cands, key=lambda c: abs(t_arr[i] - c))
        if abs(t_arr[i] - best) <= win:
            out[i] = best
    return out


nearest_outro = nearest_alarm(anom_times.values, outros_arr, win)
near_outro = ~np.isnat(nearest_outro) & ~near_trip

acerto_times = anom_times[near_trip]
laranja_times = anom_times[near_outro]
fp_times = anom_times[~near_trip & ~near_outro]
print(f"acerto (perto de TRIP): {len(acerto_times)}  laranja (perto de outro alarme): {len(laranja_times)}  FP isolado: {len(fp_times)}", flush=True)

outros_relevantes = pd.DatetimeIndex(nearest_outro[near_outro]).unique().sort_values()
print(f"ocorrencias de 'outros alarmes' relevantes pro grafico: {len(outros_relevantes)}", flush=True)

# blocos de "desligado" (operational_state != 'on')
off = (df_point["operational_state"] != "on")
off = off.reindex(df_raw.index).fillna(False)
off_change = off.astype(int).diff().fillna(off.astype(int).iloc[0])
starts = off.index[off_change == 1]
ends = off.index[off_change == -1]
if off.iloc[0]:
    starts = starts.insert(0, off.index[0])
if len(starts) > len(ends):
    ends = ends.append(pd.DatetimeIndex([off.index[-1]]))

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
for ax, sensor in zip(axes, SENSORS):
    ax.plot(df_raw.index, df_raw[sensor], color="#184F95", linewidth=0.3, zorder=1)

    for s, e in zip(starts, ends):
        ax.axvspan(s, e, color="gray", alpha=0.25, zorder=0, linewidth=0)

    for i, t in enumerate(outros_relevantes):
        ax.axvline(t, color="#D03B3B", linewidth=0.9, linestyle="--", alpha=0.5, zorder=2,
                   label=f"outro alarme do catálogo (n={len(outros_relevantes)})" if i == 0 else None)

    for i, t in enumerate(trip_times):
        ax.axvline(t, color="#7A1FA2", linewidth=1.4, linestyle="--", alpha=0.9, zorder=3,
                   label=f"ocorrência de TRIP (n={len(trip_times)})" if i == 0 else None)

    vals_fp = df_raw[sensor].reindex(fp_times.intersection(df_raw.index))
    ax.scatter(vals_fp.index, vals_fp.values, color="#F5C518", s=12, zorder=4, edgecolor="black", linewidth=0.3,
               label=f"isolado, sem explicação (n={len(fp_times)})")

    vals_laranja = df_raw[sensor].reindex(laranja_times.intersection(df_raw.index))
    ax.scatter(vals_laranja.index, vals_laranja.values, color="#EB9C34", s=12, zorder=4, edgecolor="black", linewidth=0.3,
               label=f"explicado por outro alarme (n={len(laranja_times)})")

    vals_ok = df_raw[sensor].reindex(acerto_times.intersection(df_raw.index))
    ax.scatter(vals_ok.index, vals_ok.values, color="#0CA30C", s=14, zorder=5, edgecolor="black", linewidth=0.3,
               label=f"acerto TRIP (n={len(acerto_times)})")

    ax.set_ylabel(LABELS[sensor], fontsize=9)
    ax.legend(loc="upper left", fontsize=7, frameon=True)

axes[0].set_title("EXP28 -- TI_0305 / TC382_03_A / T5_AVG_A: sombreado=desligado, roxo=TRIP, vermelho=outro alarme, verde=acerto TRIP, laranja=explicado, amarelo=isolado")
axes[-1].set_xlabel("tempo")
axes[0].set_xlim(df_raw.index.min(), df_raw.index.max())
fig.tight_layout()

out_path = os.path.join(OUT, "serie_ti0305_tc382_t5_exp28_v2.png")
fig.savefig(out_path, dpi=170)
print("\nfigura salva em", out_path)
