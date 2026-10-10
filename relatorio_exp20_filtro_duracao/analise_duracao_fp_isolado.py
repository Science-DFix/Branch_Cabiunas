"""Duracao dos episodios ISOLADOS (FP disperso, sem nenhum alarme real
por perto) que sobrevivem ao filtro de 4,5min do EXP20 -- ja sao, por
construcao, episodios >=4,5min (o filtro ja removeu tudo mais curto).
Este script mede especificamente QUANTO acima de 4,5min esses residuos
persistem, pra ver se ha margem pra um corte maior mirado neles.

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/relatorio_exp20_filtro_duracao/analise_duracao_fp_isolado.py
"""
import pandas as pd
import numpy as np
import os
from clearml import Dataset

OUT = "/home/dvar/REPO_CABIUNAS/cabiunas-models/relatorio_exp20_filtro_duracao"
POINT_CSV = "/home/dvar/.clearml/cache/storage_manager/global/18605388c830fc41b0cfa3439a4d5249.point_anomalies_all.csv"
WIN_MIN = 1440

print("lendo point_anomalies_all.csv (EXP20, filtro 4.5min)...", flush=True)
df_point = pd.read_csv(POINT_CSV, index_col=0, parse_dates=True)

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
alarm_path = os.path.join(root, "alarmes_selecionados_turbina_a.csv")
alarm = pd.read_csv(alarm_path)
alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrência"], errors="coerce")
alarm["Tag"] = alarm["Tag Alarme"]
alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].copy()
sub_all = alarm[alarm["Tag"].isin(["TC382_03_A", "T5_AVG_A"])].sort_values("Data da Ocorrencia").reset_index(drop=True)
alarm_times_sorted = sub_all["Data da Ocorrencia"].sort_values().values.astype("datetime64[ns]")
print(f"alarmes (historico completo, 2 tags): {len(sub_all)}", flush=True)

flags = df_point["is_anom_point"].values.astype(bool)
index = df_point.index

dt_seconds = index.to_series().diff().dt.total_seconds().median()
if not np.isfinite(dt_seconds) or dt_seconds <= 0:
    dt_seconds = 30.0
print(f"cadencia: {dt_seconds}s", flush=True)


def runs_from_bool(mask):
    m = mask.astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts, ends))


runs = runs_from_bool(flags)
print(f"episodios continuos pos-filtro-4.5min: {len(runs)}", flush=True)

win = np.timedelta64(WIN_MIN, "m")
records = []
for s, e in runs:
    dur_min = (e - s) * dt_seconds / 60.0
    mid_time = index[s].to_numpy()
    pos = np.searchsorted(alarm_times_sorted, mid_time)
    cands = []
    if pos > 0:
        cands.append(alarm_times_sorted[pos - 1])
    if pos < len(alarm_times_sorted):
        cands.append(alarm_times_sorted[pos])
    # usa o INICIO e o FIM do episodio -- perto se qualquer ponto do episodio estiver a <=24h de um alarme
    near = False
    for t in (index[s].to_numpy(), index[e - 1].to_numpy()):
        p = np.searchsorted(alarm_times_sorted, t)
        c = []
        if p > 0:
            c.append(alarm_times_sorted[p - 1])
        if p < len(alarm_times_sorted):
            c.append(alarm_times_sorted[p])
        if any(abs(t - x) <= win for x in c):
            near = True
            break
    records.append({"start": index[s], "end": index[e - 1], "duration_min": dur_min, "near_alarm": near})

df_ep = pd.DataFrame(records)
df_ep.to_csv(os.path.join(OUT, "episodios_pos_filtro_com_classificacao.csv"), index=False)

iso = df_ep.loc[~df_ep["near_alarm"], "duration_min"]
near = df_ep.loc[df_ep["near_alarm"], "duration_min"]

print(f"\ntotal de episodios: {len(df_ep)}  |  isolados: {len(iso)}  |  perto de alarme: {len(near)}")


def describe(name, arr):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        print(f"{name}: vazio")
        return
    print(f"{name}: n={len(arr)}  media={arr.mean():.1f}min  mediana={np.median(arr):.1f}min  "
          f"p25={np.percentile(arr,25):.1f}  p75={np.percentile(arr,75):.1f}  p90={np.percentile(arr,90):.1f}  "
          f"max={arr.max():.1f}min ({arr.max()/60:.1f}h)")


describe("ISOLADOS (FP disperso, pos-filtro)", iso)
describe("PERTO DE ALARME (pos-filtro)", near)

print("\ndistribuicao dos isolados por faixa de duracao:")
bins = [4.5, 6, 8, 10, 15, 20, 30, 60, 120, np.inf]
labels = ["4.5-6", "6-8", "8-10", "10-15", "15-20", "20-30", "30-60", "60-120", "120+"]
cats = pd.cut(iso, bins=bins, labels=labels, right=False)
print(cats.value_counts().sort_index().to_string())

print("\n10 episodios isolados mais longos:")
print(df_ep.loc[~df_ep["near_alarm"]].sort_values("duration_min", ascending=False).head(10).to_string())
