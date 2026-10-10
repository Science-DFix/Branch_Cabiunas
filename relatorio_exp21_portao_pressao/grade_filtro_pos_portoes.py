"""Testa um SEGUNDO filtro de duracao minima, aplicado DEPOIS de todos os
portoes (carga/volatilidade/mudanca de nivel/congelamento) do EXP22 -- ideia
do usuario: os fragmentos residuais que sobram sao, em sua maioria (13/20
episodios "isolados"), pontos unicos ou pares de pontos (0-1min) -- efemeros
demais para serem precursores reais (que tendem a persistir dezenas de
minutos). Reaproveita a MESMA funcao `apply_min_duration_filter` (RLE) ja
usada como filtro PRE-portoes (EXP20), so que agora rodando por cima do
is_anom_point final do EXP22 -- reconstrucao literal, nao aproximacao.

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/relatorio_exp21_portao_pressao/grade_filtro_pos_portoes.py
"""
import sys
import pandas as pd
import numpy as np
import os

sys.path.insert(0, "/home/dvar/REPO_CABIUNAS/cabiunas-models")
from clearml import Dataset
from src.cnn1d_ae.scoring import apply_min_duration_filter, eval_alarm_hit_rate, compute_normal_alert_rate

OUT = os.path.dirname(os.path.abspath(__file__))
POINT_CSV = "/home/dvar/.clearml/cache/storage_manager/global/950100358f8e4a8dcba408279af95237.point_anomalies_all.csv"
WIN_2TAGS_MIN = 1440
WIN_PRESSAO_MIN = 480
PRESSURE_TAGS = ["PI_6240319_AL", "PAL_6240315", "PDAL_6240302"]
OOS_START = pd.Timestamp("2025-07-01")
DATA_END = pd.Timestamp("2026-04-20 23:59:59")
GRID_MIN = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

print("lendo point_anomalies_all.csv (EXP22)...", flush=True)
df_point_base = pd.read_csv(POINT_CSV, index_col=0, parse_dates=True)
df_point_base = df_point_base.loc[df_point_base.index <= DATA_END]
print(f"pontos: {len(df_point_base)}  is_anom_point={int(df_point_base['is_anom_point'].sum())}", flush=True)

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
alarm_path = os.path.join(root, "alarmes_selecionados_turbina_a.csv")
alarm = pd.read_csv(alarm_path)
alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrência"], errors="coerce")
alarm["Tag"] = alarm["Tag Alarme"]
alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].copy()
alarm = alarm.dropna(subset=["Data da Ocorrencia"]).sort_values("Data da Ocorrencia").reset_index(drop=True)

df_alarm_eval = alarm.loc[alarm["Tag"].isin(["TC382_03_A", "T5_AVG_A"]) & (alarm["Data da Ocorrencia"] >= OOS_START)]
print(f"alarmes de avaliacao (2 tags, OOS): {len(df_alarm_eval)}", flush=True)


def build_exclusion(index, times, minutes):
    mask = pd.Series(False, index=index)
    delta = pd.Timedelta(minutes=minutes)
    for tt in times:
        t0, t1 = tt - delta, tt + delta
        mask.loc[(mask.index >= t0) & (mask.index <= t1)] = True
    return mask


sub2 = alarm[alarm["Tag"].isin(["TC382_03_A", "T5_AVG_A"])]
sub_p = alarm[alarm["Tag"].isin(PRESSURE_TAGS)]
near2 = build_exclusion(df_point_base.index, sub2["Data da Ocorrencia"], WIN_2TAGS_MIN)
nearp = build_exclusion(df_point_base.index, sub_p["Data da Ocorrencia"], WIN_PRESSAO_MIN)
near_alarm_mask = near2 | nearp

base_hit = eval_alarm_hit_rate(df_alarm_eval, df_point_base, 1440)
base_normal = compute_normal_alert_rate(df_point_base, near_alarm_mask)
print(f"\nEXP22 (sem 2o filtro): hit_rate={base_hit['hit_rate']*100:.1f}% "
      f"({base_hit['alarms_with_detected_anomaly_in_window']}/{base_hit['n_alarms']})  "
      f"normal_alert_rate={base_normal*100:.4f}%", flush=True)

print("\ngrade do 2o filtro (pos-portoes):", flush=True)
rows = []
for thr in GRID_MIN:
    dfp = apply_min_duration_filter(df_point_base.copy(), thr)
    hit = eval_alarm_hit_rate(df_alarm_eval, dfp, 1440)
    normal = compute_normal_alert_rate(dfp, near_alarm_mask)
    n_anom = int(dfp["is_anom_point"].sum())
    rows.append({"min_duration": thr, "hit_rate": hit["hit_rate"], "hits": hit["alarms_with_detected_anomaly_in_window"],
                 "n_alarms": hit["n_alarms"], "normal_alert_rate": normal, "n_anom_points": n_anom})
    print(f"  thr={thr:>4.1f}min  hit_rate={hit['hit_rate']*100:5.1f}% "
          f"({hit['alarms_with_detected_anomaly_in_window']}/{hit['n_alarms']})  "
          f"normal_alert_rate={normal*100:.4f}%  pontos_restantes={n_anom}", flush=True)

pd.DataFrame(rows).to_csv(os.path.join(OUT, "grade_filtro_pos_portoes_result.csv"), index=False)
print(f"\nsalvo em {os.path.join(OUT, 'grade_filtro_pos_portoes_result.csv')}", flush=True)
