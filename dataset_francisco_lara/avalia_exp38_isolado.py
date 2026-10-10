"""Avalia o EXP38 (pressao de oleo isolada) sozinho contra a regua
rigorosa (48h antes / 48h depois / parada real >=2h), mesmo checklist
usado para EXP33 (temperatura) e EXP34 (vibracao) antes de entrar na
votacao multicanal.

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/dataset_francisco_lara/avalia_exp38_isolado.py
"""
import os
import sys

import pandas as pd
from clearml import Task

sys.path.insert(0, "/home/dvar/REPO_CABIUNAS/cabiunas-models")
from src.cnn1d_ae.scoring import (
    group_alerts_into_episodes,
    classify_episodes_regua,
    compute_regua_metrics,
    compute_operational_period_days,
)

OUT = os.path.dirname(os.path.abspath(__file__))
TID_OLEO = "d1b70f12e55040c9a38fb3cce0703f17"
KEY = "oleo_pressao_isolada/csv/point_anomalies_all.csv"

t = Task.get_task(task_id=TID_OLEO)
p = t.artifacts[KEY].get_local_copy()
df = pd.read_csv(p, index_col=0, parse_dates=True, low_memory=False)

falhas = pd.read_csv(os.path.join(OUT, "alarmes_francisco_falhas.csv"))
falhas["Data da Ocorrência"] = pd.to_datetime(falhas["Data da Ocorrência"])
failure_times = falhas.loc[falhas["Tag Alarme"] == "FALHA_CURADA", "Data da Ocorrência"].sort_values()
ft = failure_times.loc[(failure_times >= df.index.min()) & (failure_times <= df.index.max())]

dias_vigiados = compute_operational_period_days(df)
print(f"EXP38 (óleo pressão isolada) -- falhas catalogadas no período: {len(ft)}  dias vigiados: {dias_vigiados:.1f}")

eps = group_alerts_into_episodes(df["is_anom_point"], merge_gap_minutes=120.0)
print(f"episódios (gap<2h fundido): {len(eps)}")

cls = classify_episodes_regua(eps, ft, df["operational_state"], 48.0, 48.0, 2.0)
print(cls["classe"].value_counts())

metrics = compute_regua_metrics(cls, ft, dias_vigiados)
print("\n=== EXP38 sozinho (régua rigorosa) ===")
for k, v in metrics.items():
    print(f"  {k}: {v}")

detectadas = sorted(pd.Timestamp(d) for d in cls.loc[cls["classe"] == "deteccao", "falha_associada"].dropna().unique())
faltantes = [pd.Timestamp(f) for f in ft if not any(abs((pd.Timestamp(f) - d).total_seconds()) < 60 for d in detectadas)]
print("\nfalhas detectadas:")
for d in detectadas:
    print(" ", d)
print("falhas não detectadas:", faltantes)

cls.to_csv(os.path.join(OUT, "regua_episodios_EXP38.csv"), index=False)
print(f"\nsalvo em {os.path.join(OUT, 'regua_episodios_EXP38.csv')}")
