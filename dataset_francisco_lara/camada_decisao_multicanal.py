"""Camada de decisao multi-canal (persistencia + CUSUM + votacao +
refratario) combinando os scores continuos de EXP33 (temperatura) e
EXP34 (vibracao) com um canal de alarme de processo (estatistica pura,
sem ML -- ideia do supervisor: alarmes normais, mesmo sem TRIP, sao
informacao de mudanca que pode preceder um TRIP).

Substitui o OR/AND ingenuo testado antes (8/8 com FP 26,99/mes; 6/8
com FP 14,15/mes) por uma votacao de verdade, no espirito do detector
de 4 sinais do Thallys (DOC_EQUIPE/ROTEIRO_APRESENTACAO_TC33003A.pdf):
cada canal so "vota" se o sinal for sustentado (persistencia OU CUSUM),
o voto exige >=2 canais concordando, e um refratario de 48h evita
reabrir o mesmo evento.

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/dataset_francisco_lara/camada_decisao_multicanal.py
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, "/home/dvar/REPO_CABIUNAS/cabiunas-models")
from src.cnn1d_ae.scoring import (
    compute_persistence_gate, compute_cusum_gate, combine_channels_vote,
    apply_refractory, group_alerts_into_episodes, classify_episodes_regua,
    compute_regua_metrics, compute_operational_period_days,
)
from clearml import Task, Dataset

OUT = os.path.dirname(os.path.abspath(__file__))

# tasks v2 (com sequence_scores_all.csv)
TID_TEMP = "eb46b8afc61c447bb773fe836a504773"
KEY_TEMP_SCORE = "mancal_temperatura_isolada/csv/sequence_scores_all.csv"
KEY_TEMP_POINT = "mancal_temperatura_isolada/csv/point_anomalies_all.csv"

TID_VIB = "806178140f38456cbb094c67320b7cec"
KEY_VIB_SCORE = "mancal_vibracao_isolada/csv/sequence_scores_all.csv"
KEY_VIB_POINT = "mancal_vibracao_isolada/csv/point_anomalies_all.csv"

OOS_SPLIT = pd.Timestamp("2025-07-01")
ALARM_CHANNEL_TAGS = ["PI_6240319_AL", "PAL_6240315", "PDAL_6240302", "TC382_05_A", "PAH_6240319"]


def alarm_channel_bool(index: pd.DatetimeIndex, alarm_times: pd.Series, window_hours: float) -> pd.Series:
    """Canal de alarme de processo: True se um alarme de `alarm_times`
    ocorreu nas ultimas `window_hours` -- estatistica pura (proximidade
    temporal), sem nenhum modelo."""
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


print("baixando scores e pontos do EXP33/34 (v2, com sequence_scores_all.csv)...", flush=True)
t_temp = Task.get_task(task_id=TID_TEMP)
p_temp_score = t_temp.artifacts[KEY_TEMP_SCORE].get_local_copy()
p_temp_point = t_temp.artifacts[KEY_TEMP_POINT].get_local_copy()

t_vib = Task.get_task(task_id=TID_VIB)
p_vib_score = t_vib.artifacts[KEY_VIB_SCORE].get_local_copy()
p_vib_point = t_vib.artifacts[KEY_VIB_POINT].get_local_copy()

df_temp_score = pd.read_csv(p_temp_score, index_col=0, parse_dates=True)
df_temp_point = pd.read_csv(p_temp_point, index_col=0, parse_dates=True, low_memory=False)
df_vib_score = pd.read_csv(p_vib_score, index_col=0, parse_dates=True)
df_vib_point = pd.read_csv(p_vib_point, index_col=0, parse_dates=True, low_memory=False)

idx = df_temp_score.index.intersection(df_vib_score.index)
print(f"indice comum: {len(idx)} amostras", flush=True)

score_temp = df_temp_score.loc[idx, "score"]
thr_temp = float(df_temp_score.loc[idx, "threshold"].iloc[0])
score_vib = df_vib_score.loc[idx, "score"]
thr_vib = float(df_vib_score.loc[idx, "threshold"].iloc[0])
op_state = df_temp_point.loc[idx, "operational_state"]

print(f"limiar temperatura (percentil vencedor)={thr_temp:.3f}  limiar vibracao={thr_vib:.3f}", flush=True)

# baseline (treino, antes do corte OOS, so operacao 'on') para o CUSUM
train_mask = (idx < OOS_SPLIT) & (op_state == "on")
base_mean_temp, base_std_temp = score_temp.loc[train_mask].mean(), score_temp.loc[train_mask].std()
base_mean_vib, base_std_vib = score_vib.loc[train_mask].mean(), score_vib.loc[train_mask].std()
print(f"baseline temperatura: mean={base_mean_temp:.3f} std={base_std_temp:.3f}", flush=True)
print(f"baseline vibracao:    mean={base_mean_vib:.3f} std={base_std_vib:.3f}", flush=True)

falhas = pd.read_csv(os.path.join(OUT, "alarmes_francisco_falhas.csv"))
falhas["Data da Ocorrência"] = pd.to_datetime(falhas["Data da Ocorrência"])
failure_times = falhas.loc[falhas["Tag Alarme"] == "FALHA_CURADA", "Data da Ocorrência"].sort_values()

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
alarm = pd.read_csv(os.path.join(root, "alarmes_selecionados_turbina_a.csv"))
alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrência"], errors="coerce")
alarm["Tag"] = alarm["Tag Alarme"]
alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].dropna(subset=["Data da Ocorrencia"])
alarm_leading_times = alarm.loc[alarm["Tag"].isin(ALARM_CHANNEL_TAGS), "Data da Ocorrencia"]

ft = failure_times.loc[(failure_times >= idx.min()) & (failure_times <= idx.max())]
print(f"falhas no periodo: {len(ft)}", flush=True)


def avalia(nome, is_anom_bool):
    df_comb = pd.DataFrame({"is_anom_point": is_anom_bool.astype(int), "operational_state": op_state})
    dias = compute_operational_period_days(df_comb)
    eps = group_alerts_into_episodes(df_comb["is_anom_point"], merge_gap_minutes=120)
    cls = classify_episodes_regua(eps, ft, df_comb["operational_state"], 48, 48, 2)
    m = compute_regua_metrics(cls, ft, dias)
    detectadas = sorted(pd.Timestamp(d).date() for d in cls.loc[cls["classe"] == "deteccao", "falha_associada"].dropna().unique())
    print(f"{nome}: falhas={m['falhas_detectadas']}/{m['n_falhas_catalogadas']}  "
          f"FP/mes={m['falso_positivo_por_mes']:.2f}  inconclusivo={m['n_episodios_inconclusivo']}  "
          f"detectadas={detectadas}", flush=True)
    return m


print("\n=== canais individuais (so persistencia, 30min) ===", flush=True)
persist_temp = compute_persistence_gate(score_temp, thr_temp, persistence_minutes=30.0)
persist_vib = compute_persistence_gate(score_vib, thr_vib, persistence_minutes=30.0)
avalia("temperatura (persistencia 30min)", persist_temp)
avalia("vibracao (persistencia 30min)", persist_vib)

print("\n=== canal de alarme de processo (estatistica pura) ===", flush=True)
for janela_h in [2.0, 6.0, 12.0, 24.0]:
    canal_alarme = alarm_channel_bool(idx, alarm_leading_times, window_hours=janela_h)
    avalia(f"alarme de processo (janela={janela_h}h)", canal_alarme)

print("\n=== votacao (persistencia + alarme, >=2 canais) ===", flush=True)
canal_alarme_6h = alarm_channel_bool(idx, alarm_leading_times, window_hours=6.0)
for min_votes in [2]:
    voto = combine_channels_vote(
        {"temperatura": persist_temp, "vibracao": persist_vib, "alarme": canal_alarme_6h},
        min_votes=min_votes, required_any=["temperatura", "vibracao"],
    )
    avalia(f"voto>={min_votes} (persist temp+vib+alarme6h, exige temp ou vib)", voto)
    for refrat_h in [24.0, 48.0]:
        voto_refrat = apply_refractory(voto, refractory_minutes=refrat_h * 60.0)
        avalia(f"  + refratario {refrat_h}h", voto_refrat)

print("\n=== canais com CUSUM adicionado (persistencia OU cusum) ===", flush=True)
for h_mult in [5.0, 10.0, 20.0]:
    cusum_temp = compute_cusum_gate(score_temp, base_mean_temp, slack=0.5 * base_std_temp, h=h_mult * base_std_temp)
    cusum_vib = compute_cusum_gate(score_vib, base_mean_vib, slack=0.5 * base_std_vib, h=h_mult * base_std_vib)
    canal_temp_full = persist_temp | cusum_temp
    canal_vib_full = persist_vib | cusum_vib
    voto_full = combine_channels_vote(
        {"temperatura": canal_temp_full, "vibracao": canal_vib_full, "alarme": canal_alarme_6h},
        min_votes=2, required_any=["temperatura", "vibracao"],
    )
    voto_full_refrat = apply_refractory(voto_full, refractory_minutes=48.0 * 60.0)
    avalia(f"h={h_mult}x std (persist+cusum, voto>=2, refratario 48h)", voto_full_refrat)
