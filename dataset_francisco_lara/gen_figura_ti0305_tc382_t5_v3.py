"""Versao 3 do grafico de series (TI_0305, TC382_03_A, T5_AVG_A):

Mudancas pedidas pelo usuario em relacao a v2:
  - pontos "laranja" (explicados por outro alarme do catalogo) sao
    eliminados visualmente -- ja sabemos que sao sinal real de outro
    alarme, nao precisam mais poluir o grafico nem as linhas vermelhas
    que os explicavam.
  - pontos remanescentes (residuo isolado, sem explicacao) viram
    PRETOS em vez de amarelos -- e o FP genuino final.
  - acerto de TRIP continua VERDE.
  - fundo BRANCO no periodo "ligado", cinza CLARO (alpha baixo) no
    periodo "desligado" -- mais legivel que o cinza mais forte da v2.
  - novo "trecho" (painel inferior, tabela) com os 8 TRIPs: causa,
    horario, primeira deteccao e antecedencia (quanto tempo antes do
    TRIP o modelo levantou o primeiro ponto verde daquele episodio).

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/dataset_francisco_lara/gen_figura_ti0305_tc382_t5_v3.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from clearml import Dataset

OUT = os.path.dirname(os.path.abspath(__file__))
POINT_CSV_MANCAL = "/home/dvar/.clearml/cache/storage_manager/global/c27dfd135eb394c5511ce610bd890c1e.point_anomalies_all.csv"
POINT_CSV_OLEO = "/home/dvar/.clearml/cache/storage_manager/global/0598e1e0e3b369da5d7b7bf2bc89d191.point_anomalies_all.csv"
SENSORS = ["954005_624_TI_0305", "TC382_03_A", "T5_AVG_A"]
LABELS = {
    "954005_624_TI_0305": "TI_0305 (°C)",
    "TC382_03_A": "TC382_03_A (°C)",
    "T5_AVG_A": "T5_AVG_A",
}
DATA_END = pd.Timestamp("2026-04-20 23:59:59")

CAUSA_POR_TRIP = {
    pd.Timestamp("2024-01-16 09:10:00"): "mancal",
    pd.Timestamp("2025-02-27 08:38:00"): "selagem",
    pd.Timestamp("2025-03-17 18:16:00"): "mancal",
    pd.Timestamp("2025-04-07 21:18:00"): "mancal",
    pd.Timestamp("2025-04-11 17:02:00"): "mancal",
    pd.Timestamp("2025-04-29 03:04:00"): "mancal",
    pd.Timestamp("2025-11-04 06:22:00"): "óleo",
    pd.Timestamp("2025-12-09 08:36:00"): "mancal",
    pd.Timestamp("2026-02-26 15:34:00"): "óleo",
}

print("lendo point_anomalies_all.csv (EXP28 mancal + EXP29 óleo)...", flush=True)
df_point = pd.read_csv(POINT_CSV_MANCAL, index_col=0, parse_dates=True)
df_point = df_point.loc[df_point.index <= DATA_END]
anom_times = df_point.index[df_point["is_anom_point"] == 1]
print(f"anomalias detectadas (mancal, EXP28 -- usada pro gráfico de série): {len(anom_times)}", flush=True)

# EXP29 (óleo) so entra na TABELA de antecedencia (mesmos sensores de
# série -- TI_0305/TC382/T5_AVG -- pertencem ao grupo mancal; o modelo
# de óleo detecta com outros sensores, mas o evento fisico e o mesmo
# TRIP curado, entao a deteccao dele conta pra "quando o conjunto da
# pipeline percebeu o evento primeiro").
df_point_oleo = pd.read_csv(POINT_CSV_OLEO, index_col=0, parse_dates=True)
df_point_oleo = df_point_oleo.loc[df_point_oleo.index <= DATA_END]
anom_times_oleo = df_point_oleo.index[df_point_oleo["is_anom_point"] == 1]
print(f"anomalias detectadas (óleo, EXP29 -- só entra na tabela de antecedência): {len(anom_times_oleo)}", flush=True)

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
raw_path = os.path.join(root, "sensores_full_2024_2026_30s.csv")
print("lendo series brutas...", flush=True)
df_raw = pd.read_csv(raw_path, usecols=["data_datetime"] + SENSORS)
df_raw["data_datetime"] = pd.to_datetime(df_raw["data_datetime"], errors="coerce")
for c in SENSORS:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
df_raw = df_raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
df_raw = df_raw.loc[(df_raw.index >= "2024-07-01") & (df_raw.index <= DATA_END)]

falhas = pd.read_csv(os.path.join(OUT, "alarmes_francisco_falhas.csv"))
falhas["Data da Ocorrência"] = pd.to_datetime(falhas["Data da Ocorrência"])
trip_times = falhas.loc[falhas["Tag Alarme"] == "FALHA_CURADA", "Data da Ocorrência"].sort_values()
trip_times_all = trip_times.copy()
trip_times = trip_times.loc[(trip_times >= df_raw.index.min()) & (trip_times <= df_raw.index.max())]
print(f"ocorrencias de TRIP curadas no periodo carregado: {len(trip_times)} (de {len(trip_times_all)} totais)", flush=True)

alarm = pd.read_csv(os.path.join(root, "alarmes_selecionados_turbina_a.csv"))
alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrência"], errors="coerce")
alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].copy()
alarm = alarm.dropna(subset=["Data da Ocorrencia"]).sort_values("Data da Ocorrencia")
outros_times = alarm["Data da Ocorrencia"]
outros_times = outros_times.loc[(outros_times >= df_raw.index.min()) & (outros_times <= df_raw.index.max())]

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


def nearest_ref(t_arr, ref_sorted, win):
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


near_trip = near_any(anom_times.values, trip_arr, win)
nearest_outro = nearest_ref(anom_times.values, outros_arr, win)
near_outro_only = ~np.isnat(nearest_outro) & ~near_trip

acerto_times = anom_times[near_trip]
laranja_times = anom_times[near_outro_only]          # eliminados visualmente
resto_times = anom_times[~near_trip & ~near_outro_only]  # ficam PRETOS

print(f"acerto (verde): {len(acerto_times)}  explicado por outro alarme (removido da vista): {len(laranja_times)}  "
      f"remanescente isolado (preto): {len(resto_times)}", flush=True)

# --- tabela de antecedencia por TRIP (combina EXP28 mancal + EXP29 óleo,
# ja que o resultado oficial de 8/8 e dos dois modelos juntos, nao so do
# de mancal usado na serie) ---
todas_deteccoes = pd.DatetimeIndex(acerto_times).union(pd.DatetimeIndex(anom_times_oleo))
nearest_trip_for_det = nearest_ref(todas_deteccoes.values, trip_arr, win)
rows = []
for t in trip_times:
    mask = nearest_trip_for_det == np.datetime64(t)
    detections = todas_deteccoes[mask]
    causa = CAUSA_POR_TRIP.get(pd.Timestamp(t), "?")
    pre = detections[detections <= t]
    if len(pre) == 0:
        # sem deteccao ANTES do TRIP dentro da janela de +-24h -- ou nao
        # detectado, ou so detectado depois (nao conta como antecedencia)
        pos = detections[detections > t]
        status = "detectado só depois" if len(pos) else "não detectado"
        rows.append({"trip": t, "causa": causa, "primeira_deteccao": pd.NaT,
                      "antecedencia_h": np.nan, "status": status})
        continue
    first_det = pre.min()
    antecedencia_h = (t - first_det).total_seconds() / 3600.0
    rows.append({"trip": t, "causa": causa, "primeira_deteccao": first_det,
                  "antecedencia_h": antecedencia_h, "status": "ok"})

tabela_antecedencia = pd.DataFrame(rows)
print("\n--- antecedência de detecção por TRIP ---")
print(tabela_antecedencia.to_string(index=False))
tabela_antecedencia.to_csv(os.path.join(OUT, "antecedencia_deteccao_8trips.csv"), index=False)

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

# --- figura: 3 paineis de sensores + 1 painel de tabela ---
fig = plt.figure(figsize=(16, 12.5))
gs = fig.add_gridspec(4, 1, height_ratios=[3, 3, 3, 2], hspace=0.12)
axes = [fig.add_subplot(gs[i]) for i in range(3)]
ax_table = fig.add_subplot(gs[3])
ax_table.axis("off")

fig.patch.set_facecolor("white")

for ax, sensor in zip(axes, SENSORS):
    ax.set_facecolor("white")
    ax.plot(df_raw.index, df_raw[sensor], color="#184F95", linewidth=0.3, zorder=1)

    for s, e in zip(starts, ends):
        ax.axvspan(s, e, color="#B0B0B0", alpha=0.18, zorder=0, linewidth=0)

    for i, t in enumerate(trip_times):
        ax.axvline(t, color="#7A1FA2", linewidth=1.5, linestyle="--", alpha=0.9, zorder=3,
                   label=f"ocorrência de TRIP (n={len(trip_times)})" if i == 0 else None)

    vals_resto = df_raw[sensor].reindex(resto_times.intersection(df_raw.index))
    ax.scatter(vals_resto.index, vals_resto.values, color="black", s=14, zorder=4,
               label=f"residual isolado, sem explicação (n={len(resto_times)})")

    vals_ok = df_raw[sensor].reindex(acerto_times.intersection(df_raw.index))
    ax.scatter(vals_ok.index, vals_ok.values, color="#0CA30C", s=18, zorder=5, edgecolor="black", linewidth=0.3,
               label=f"acerto TRIP (n={len(acerto_times)})")

    ax.set_ylabel(LABELS[sensor], fontsize=9)
    ax.legend(loc="upper left", fontsize=7, frameon=True)

axes[0].set_title(
    "EXP28 -- TI_0305 / TC382_03_A / T5_AVG_A\n"
    "cinza claro = desligado | branco = ligado | roxo = TRIP curado | "
    "verde = acerto | preto = residual sem explicação "
    "(pontos explicados por outro alarme do catálogo foram removidos da vista)",
    fontsize=10,
)
axes[-1].set_xlabel("tempo")
for ax in axes:
    ax.set_xlim(df_raw.index.min(), df_raw.index.max())
for ax in axes[:-1]:
    ax.tick_params(labelbottom=False)

# --- painel de tabela: antecedência de detecção por TRIP ---
table_rows = []
for _, r in tabela_antecedencia.iterrows():
    trip_str = r["trip"].strftime("%d/%m/%Y %H:%M")
    if pd.isna(r["antecedencia_h"]):
        det_str, ant_str = r["status"], "—"
    else:
        det_str = r["primeira_deteccao"].strftime("%d/%m/%Y %H:%M")
        ant_str = f"{r['antecedencia_h']:.1f} h"
    table_rows.append([trip_str, r["causa"], det_str, ant_str])

col_labels = ["TRIP (curado)", "Causa", "Primeira detecção (verde ou óleo)", "Antecedência"]
tbl = ax_table.table(cellText=table_rows, colLabels=col_labels, loc="center", cellLoc="center",
                      bbox=[0.0, 0.0, 1.0, 0.80])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#184F95")
        cell.set_text_props(color="white", weight="bold")
    else:
        cell.set_facecolor("#F2F2F2" if row % 2 == 0 else "white")
ant_media = tabela_antecedencia.loc[tabela_antecedencia["status"] == "ok", "antecedencia_h"].mean()
n_ok = (tabela_antecedencia["status"] == "ok").sum()
ax_table.text(0.5, 0.95,
              f"Antecedência de detecção dos 8 TRIPs (1ª detecção da pipeline, mancal ou óleo, antes do TRIP)\n"
              f"média = {ant_media:.1f} h em {n_ok}/8 eventos com precursor antes do TRIP",
              ha="center", va="top", fontsize=9.5, transform=ax_table.transAxes)

fig.subplots_adjust(top=0.93, bottom=0.03, left=0.05, right=0.99, hspace=0.15)
out_path = os.path.join(OUT, "serie_ti0305_tc382_t5_exp28_v3.png")
fig.savefig(out_path, dpi=170, facecolor="white")
print("\nfigura salva em", out_path)
