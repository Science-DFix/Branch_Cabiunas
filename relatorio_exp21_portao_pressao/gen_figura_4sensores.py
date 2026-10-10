"""4 series temporais (TC382_03_A, T5_AVG_A, TV_354Y_A -- mancal com
evento recorrente confirmado, TV_351X_A -- canal "normal" de referencia)
com anomalias classificadas: verde = alarme real (perto dos 2 tags
oficiais, +-24h), preto = FP (is_anom_point final, sem correspondencia
com alarme oficial). Usa o EXP22 (melhor config validada).

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/relatorio_exp21_portao_pressao/gen_figura_4sensores.py
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
WIN_2TAGS_MIN = 1440

SENSORS = ["TC382_03_A", "T5_AVG_A", "TV_354Y_A", "TV_351X_A"]
LABELS = {
    "TC382_03_A": "TC382_03_A (temperatura, °C)",
    "T5_AVG_A": "T5_AVG_A (carga, proxy)",
    "TV_354Y_A": "TV_354Y_A (vibração mancal 354 -- evento recorrente confirmado)",
    "TV_351X_A": "TV_351X_A (vibração mancal 351 -- canal de referência)",
}

print("lendo point_anomalies_all.csv (EXP22)...", flush=True)
df_point = pd.read_csv(POINT_CSV, index_col=0, parse_dates=True)
df_point = df_point.loc[df_point.index <= DATA_END]
anom_times = df_point.index[df_point["is_anom_point"] == 1]
print(f"anomalias detectadas (finais): {len(anom_times)}", flush=True)

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
raw_path = os.path.join(root, "sensores_full_2024_2026_30s.csv")
print("lendo series brutas dos 4 sensores...", flush=True)
df_raw = pd.read_csv(raw_path, usecols=["data_datetime"] + SENSORS)
df_raw["data_datetime"] = pd.to_datetime(df_raw["data_datetime"], errors="coerce")
for c in SENSORS:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
df_raw = df_raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
df_raw = df_raw.loc[(df_raw.index >= "2024-07-01") & (df_raw.index <= DATA_END)]

alarm_path = os.path.join(root, "alarmes_selecionados_turbina_a.csv")
alarm = pd.read_csv(alarm_path)
alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrência"], errors="coerce")
alarm["Tag"] = alarm["Tag Alarme"]
alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].copy()
alarm = alarm.dropna(subset=["Data da Ocorrencia"]).sort_values("Data da Ocorrencia").reset_index(drop=True)
sub2 = alarm[alarm["Tag"].isin(["TC382_03_A", "T5_AVG_A"])].sort_values("Data da Ocorrencia")
alarm_times_2 = sub2["Data da Ocorrencia"].values.astype("datetime64[ns]")
win2 = np.timedelta64(WIN_2TAGS_MIN, "m")


def near_any(t_arr, alarm_times_sorted, win):
    if len(alarm_times_sorted) == 0 or len(t_arr) == 0:
        return np.zeros(len(t_arr), dtype=bool)
    pos = np.searchsorted(alarm_times_sorted, t_arr)
    out = np.zeros(len(t_arr), dtype=bool)
    for i in range(len(t_arr)):
        p = pos[i]
        cands = []
        if p > 0:
            cands.append(alarm_times_sorted[p - 1])
        if p < len(alarm_times_sorted):
            cands.append(alarm_times_sorted[p])
        out[i] = any(abs(t_arr[i] - c) <= win for c in cands)
    return out


near_official = near_any(anom_times.values, alarm_times_2, win2)
verde_times = anom_times[near_official]
preto_times = anom_times[~near_official]
print(f"alarme real: {len(verde_times)}  FP: {len(preto_times)}", flush=True)

fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
for ax, sensor in zip(axes, SENSORS):
    ax.plot(df_raw.index, df_raw[sensor], color="#184F95", linewidth=0.3, zorder=1)

    vals_preto = df_raw[sensor].reindex(preto_times.intersection(df_raw.index))
    ax.scatter(vals_preto.index, vals_preto.values, color="black", s=8, zorder=3,
               label=f"FP (n={len(preto_times)})")

    vals_verde = df_raw[sensor].reindex(verde_times.intersection(df_raw.index))
    ax.scatter(vals_verde.index, vals_verde.values, color="#0CA30C", s=10, zorder=4,
               label=f"alarme real (n={len(verde_times)})")

    for i, t in enumerate(sub2["Data da Ocorrencia"]):
        ax.axvline(t, color="#0CA30C", linewidth=1.0, linestyle="--", alpha=0.6, zorder=2,
                   label=f"ocorrência de alarme real (n={len(sub2)})" if i == 0 else None)

    ax.set_ylabel(LABELS[sensor], fontsize=8)
    ax.legend(loc="upper left", fontsize=8, frameon=True)

axes[0].set_title("EXP22 -- 4 sensores: alarme real (verde) x FP (preto)")
axes[-1].set_xlabel("tempo")
axes[0].set_xlim(df_raw.index.min(), df_raw.index.max())
fig.tight_layout()

out_path = os.path.join(OUT, "serie_4sensores_exp22.png")
fig.savefig(out_path, dpi=170)
print("\nfigura salva em", out_path)
