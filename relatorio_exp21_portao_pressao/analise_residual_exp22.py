"""Repete a analise de cruzamento com o catalogo completo (47 tags),
agora em cima do residual do EXP22 (com filtro de duracao + portao de
pressao + corte de dados + portao de mudanca de nivel ja aplicados).

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/relatorio_exp21_portao_pressao/analise_residual_exp22.py
"""
import pandas as pd
import numpy as np
import os
from clearml import Dataset

OUT = os.path.dirname(os.path.abspath(__file__))
POINT_CSV = "/home/dvar/.clearml/cache/storage_manager/global/950100358f8e4a8dcba408279af95237.point_anomalies_all.csv"
WIN_2TAGS_MIN = 1440
WIN_PRESSAO_MIN = 480
WIN_CATALOGO_MIN = 1440
PRESSURE_TAGS = ["PI_6240319_AL", "PAL_6240315", "PDAL_6240302"]
OOS_START = pd.Timestamp("2025-07-01")
DATA_END = pd.Timestamp("2026-04-20 23:59:59")

print("lendo point_anomalies_all.csv (EXP22)...", flush=True)
df_point = pd.read_csv(POINT_CSV, index_col=0, parse_dates=True)
df_point = df_point.loc[df_point.index <= DATA_END]
print(f"pontos: {len(df_point)}", flush=True)

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
alarm_path = os.path.join(root, "alarmes_selecionados_turbina_a.csv")
alarm = pd.read_csv(alarm_path)
alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrência"], errors="coerce")
alarm["Tag"] = alarm["Tag Alarme"]
alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].copy()
alarm = alarm.dropna(subset=["Data da Ocorrencia"]).sort_values("Data da Ocorrencia").reset_index(drop=True)
print(f"catalogo completo: {len(alarm)} eventos, {alarm['Tag'].nunique()} tags", flush=True)

sub2 = alarm[alarm["Tag"].isin(["TC382_03_A", "T5_AVG_A"])]
sub_p = alarm[alarm["Tag"].isin(PRESSURE_TAGS)]


def build_exclusion(index, times, minutes):
    mask = pd.Series(False, index=index)
    delta = pd.Timedelta(minutes=minutes)
    for tt in times:
        t0, t1 = tt - delta, tt + delta
        mask.loc[(mask.index >= t0) & (mask.index <= t1)] = True
    return mask


near2 = build_exclusion(df_point.index, sub2["Data da Ocorrencia"], WIN_2TAGS_MIN)
nearp = build_exclusion(df_point.index, sub_p["Data da Ocorrencia"], WIN_PRESSAO_MIN)
near_official = (near2 | nearp).values

on_arr = (df_point["operational_state"] == "on").values
oos_arr = (df_point.index >= OOS_START)
normal_mask = on_arr & oos_arr & (~near_official)

flags = df_point["is_anom_point"].values.astype(bool)
fp_mask = normal_mask & flags
fp_times = df_point.index[fp_mask]
print(f"\npontos normais (denominador): {normal_mask.sum()}")
print(f"pontos FP oficiais (EXP22): {len(fp_times)}  (normal_alert_rate={fp_mask.sum()/normal_mask.sum()*100:.4f}%)")

# cruza cada ponto de FP oficial com o CATALOGO COMPLETO (47 tags), +-24h
alarm_times_sorted = alarm["Data da Ocorrencia"].sort_values().values.astype("datetime64[ns]")
alarm_sorted = alarm.sort_values("Data da Ocorrencia").reset_index(drop=True)
win_cat = np.timedelta64(WIN_CATALOGO_MIN, "m")

near_full = np.zeros(len(fp_times), dtype=bool)
nearest_tag = [None] * len(fp_times)
nearest_dist_h = np.zeros(len(fp_times))
for i, t in enumerate(fp_times.values):
    p = np.searchsorted(alarm_times_sorted, t)
    cands_idx = []
    if p > 0:
        cands_idx.append(p - 1)
    if p < len(alarm_times_sorted):
        cands_idx.append(p)
    best_d, best_tag = None, None
    for ci in cands_idx:
        d = abs(t - alarm_times_sorted[ci])
        if best_d is None or d < best_d:
            best_d, best_tag = d, alarm_sorted.loc[ci, "Tag"]
    nearest_dist_h[i] = best_d / np.timedelta64(1, "h") if best_d is not None else np.inf
    nearest_tag[i] = best_tag
    near_full[i] = (best_d is not None) and (best_d <= win_cat)

n_near_full = near_full.sum()
n_isolado = len(fp_times) - n_near_full
print(f"\ncoincidem com QUALQUER um dos 47 tags (+-24h): {n_near_full} ({n_near_full/max(1,len(fp_times))*100:.1f}%)")
print(f"genuinamente isolados: {n_isolado} ({n_isolado/max(1,len(fp_times))*100:.1f}%)")

df_out = pd.DataFrame({"time": fp_times, "near_47tags": near_full, "nearest_tag": nearest_tag,
                        "nearest_dist_h": nearest_dist_h})
df_out.to_csv(os.path.join(OUT, "fp_exp22_vs_catalogo_completo.csv"), index=False)

print("\ntags mais proximas dos que SOBRAM isolados:")
print(df_out.loc[~df_out["near_47tags"], "nearest_tag"].value_counts().head(15).to_string())

# agrupa os isolados em episodios
iso = df_out.loc[~df_out["near_47tags"]].sort_values("time").reset_index(drop=True)
if len(iso):
    gaps = iso["time"].diff().dt.total_seconds().fillna(1e9) / 60.0
    iso["ep_id"] = (gaps > 30).cumsum()
    eps = iso.groupby("ep_id")["time"].agg(["min", "max", "count"])
    eps.columns = ["start", "end", "n_points"]
    print(f"\nepisodios isolados distintos: {len(eps)}")
    print(eps.to_string())
else:
    print("\nnenhum ponto genuinamente isolado restante!")
