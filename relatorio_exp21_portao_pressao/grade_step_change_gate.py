"""Grade LITERAL (reconstruindo a serie a cada corte, sem aproximacao)
do novo portao de MUDANCA DE NIVEL (step-change): complementa o portao
de rampa existente (que reage a TAXA de variacao suavizada) para pegar
degraus quase instantaneos de carga -- like os vistos em
2025-12-16 01:31, 2025-08-27 07:39 e 2025-08-27 01:51 (temperatura E
vibracao mudando de patamar juntas, rapido demais pra o portao de
rampa atual reagir).

Indice = |media_curta - media_longa| / (desvio_longo + eps) do proxy de
carga (T5_AVG_A) -- mesma matematica do localz ja usado como FEATURE em
_build_changepoint_features (preprocess.py), aqui aplicado como PORTAO
(suprime deteccao, nao alimenta o modelo).

Reusa o point_anomalies_all.csv ja calculado da task mais recente do
EXP21 (com filtro de duracao + portao de pressao + corte de
DATA_END_DATE ja aplicados) como base -- so adiciona o novo portao por
cima e reavalia de verdade.

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/relatorio_exp21_portao_pressao/grade_step_change_gate.py
"""
import pandas as pd
import numpy as np
import os
from clearml import Task, Dataset

OUT = os.path.dirname(os.path.abspath(__file__))
TASK_ID = "e271fa9258c24495b5d9ea2075e079b3"  # EXP21 + corte de dados
WIN_2TAGS_MIN = 1440
WIN_PRESSAO_MIN = 480
PRESSURE_TAGS = ["PI_6240319_AL", "PAL_6240315", "PDAL_6240302"]
DATA_END = pd.Timestamp("2026-04-20 23:59:59")
OOS_START = pd.Timestamp("2025-07-01")

SHORT_WINDOW_MIN = 5.0
LONG_WINDOW_MIN = 60.0
THRESHOLD_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]

print("baixando point_anomalies_all.csv do EXP21...", flush=True)
t = Task.get_task(task_id=TASK_ID)
p = t.artifacts["TC382_T5_vibracao_mancais_multiescala/csv/point_anomalies_all.csv"].get_local_copy()
df_point = pd.read_csv(p, index_col=0, parse_dates=True)
df_point = df_point.loc[df_point.index <= DATA_END]
print(f"pontos: {len(df_point)}", flush=True)

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
alarm_path = os.path.join(root, "alarmes_selecionados_turbina_a.csv")
alarm = pd.read_csv(alarm_path)
alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrência"], errors="coerce")
alarm["Tag"] = alarm["Tag Alarme"]
alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].copy()
alarm = alarm.dropna(subset=["Data da Ocorrencia"]).sort_values("Data da Ocorrencia").reset_index(drop=True)

sub2 = alarm[alarm["Tag"].isin(["TC382_03_A", "T5_AVG_A"])]
sub_p = alarm[alarm["Tag"].isin(PRESSURE_TAGS)]
df_alarm_eval = sub2.loc[sub2["Data da Ocorrencia"] >= OOS_START].reset_index(drop=True)
print(f"alarmes OOS avaliados: {len(df_alarm_eval)}", flush=True)


def build_exclusion(index, times, minutes):
    mask = pd.Series(False, index=index)
    delta = pd.Timedelta(minutes=minutes)
    for tt in times:
        t0, t1 = tt - delta, tt + delta
        mask.loc[(mask.index >= t0) & (mask.index <= t1)] = True
    return mask


near2 = build_exclusion(df_point.index, sub2["Data da Ocorrencia"], WIN_2TAGS_MIN)
nearp = build_exclusion(df_point.index, sub_p["Data da Ocorrencia"], WIN_PRESSAO_MIN)
near_alarm = (near2 | nearp).values

on_arr = (df_point["operational_state"] == "on").values
oos_arr = (df_point.index >= OOS_START)
normal_mask = on_arr & oos_arr & (~near_alarm)
print(f"pontos normais (denominador): {normal_mask.sum()}", flush=True)


def eval_flags(flags, label):
    win = pd.Timedelta(minutes=WIN_2TAGS_MIN)
    hits = 0
    for tt in df_alarm_eval["Data da Ocorrencia"]:
        t0, t1 = tt - win, tt + win
        m = (df_point.index >= t0) & (df_point.index <= t1)
        if flags[m].any():
            hits += 1
    fp = flags[normal_mask].mean() if normal_mask.any() else 0.0
    print(f"[{label}] hit_rate={hits}/{len(df_alarm_eval)} ({hits/len(df_alarm_eval)*100:.1f}%)  "
          f"normal_alert_rate={fp*100:.4f}%", flush=True)
    return hits, fp


flags0 = df_point["is_anom_point"].values.astype(bool)
hits0, fp0 = eval_flags(flags0, "SANITY CHECK (sem step-change gate, deveria bater com EXP21 atual)")

print("\nlendo T5_AVG_A bruto (proxy de carga)...", flush=True)
raw_path = os.path.join(root, "sensores_full_2024_2026_30s.csv")
df_load = pd.read_csv(raw_path, usecols=["data_datetime", "T5_AVG_A"])
df_load["data_datetime"] = pd.to_datetime(df_load["data_datetime"], errors="coerce")
df_load["T5_AVG_A"] = pd.to_numeric(df_load["T5_AVG_A"], errors="coerce")
df_load = df_load.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
df_load = df_load.loc[(df_load.index >= "2024-07-01") & (df_load.index <= DATA_END)]
# mesmo tratamento de buraco pequeno do pipeline (limite 3 amostras) + ffill/bfill pro resto
df_load["T5_AVG_A"] = df_load["T5_AVG_A"].interpolate(limit=3, limit_direction="both").ffill().bfill()
df_load = df_load.reindex(df_point.index, method="nearest", tolerance=pd.Timedelta(seconds=30))

dt_seconds = df_load.index.to_series().diff().dt.total_seconds().median()
sw = max(2, int(round(SHORT_WINDOW_MIN * 60 / dt_seconds)))
lw = max(sw + 1, int(round(LONG_WINDOW_MIN * 60 / dt_seconds)))
short_mean = df_load["T5_AVG_A"].rolling(sw, min_periods=1).mean()
long_mean = df_load["T5_AVG_A"].rolling(lw, min_periods=1).mean()
long_std = df_load["T5_AVG_A"].rolling(lw, min_periods=1).std().fillna(0.0)
step_index = ((short_mean - long_mean).abs() / (long_std + 1e-6)).fillna(0.0).values
print(f"indice de mudanca de nivel calculado (sw={sw} amostras/{SHORT_WINDOW_MIN}min, "
      f"lw={lw} amostras/{LONG_WINDOW_MIN}min)", flush=True)
print(pd.Series(step_index).describe().to_string(), flush=True)

print(f"\nvarrendo {len(THRESHOLD_GRID)} limiares...", flush=True)
results = []
for thr in THRESHOLD_GRID:
    blocked = step_index > thr
    flags = flags0 & (~blocked)
    hits, fp = eval_flags(flags, f"threshold={thr}")
    results.append({"threshold": thr, "hits": hits, "n_alarms": len(df_alarm_eval),
                     "normal_alert_rate": fp, "frac_blocked": blocked.mean()})

df_res = pd.DataFrame(results)
df_res.to_csv(os.path.join(OUT, "grade_step_change_gate_result.csv"), index=False)
print("\nresultado salvo em grade_step_change_gate_result.csv")

zero_cost = df_res[df_res["hits"] == hits0]
if len(zero_cost):
    best = zero_cost.loc[zero_cost["normal_alert_rate"].idxmin()]
    print(f"\nMELHOR limiar custo-zero (preserva os {hits0} hits atuais): threshold={best['threshold']}  "
          f"normal_alert_rate={best['normal_alert_rate']*100:.4f}%  "
          f"reducao vs sem o portao: {(1-best['normal_alert_rate']/fp0)*100:.1f}%")
else:
    print("\nnenhum limiar testado preserva todos os hits atuais.")
