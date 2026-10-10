"""Aplica a regua de avaliacao proposta pelo usuario (episodios com
fusao de gap<2h; deteccao=falha catalogada em ate 48h depois do inicio
do episodio; inconclusivo=parada real >=2h em ate 48h depois do fim do
episodio; senao falso_positivo) contra os dados JA CALCULADOS do EXP30
(mancal) e EXP31 (oleo) -- sem re-treinar nada, so reavalia o mesmo
is_anom_point com a nova regua.

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/dataset_francisco_lara/aplica_regua_episodios.py
"""
import pandas as pd
from clearml import Task
import os
import sys

sys.path.insert(0, "/home/dvar/REPO_CABIUNAS/cabiunas-models")
from src.cnn1d_ae.scoring import group_alerts_into_episodes, classify_episodes_regua, compute_regua_metrics

OUT = os.path.dirname(os.path.abspath(__file__))

t_mancal = Task.get_task(task_id="aa87ae49ae314f02bbc011d2b843de41")
p_mancal = t_mancal.artifacts["TC382_T5_vibracao_mancais_multiescala_v2/csv/point_anomalies_all.csv"].get_local_copy()

t_oleo = Task.get_task(task_id="5707fcb08773452386dceef0b61d64ea")
p_oleo = t_oleo.artifacts["trip_oleo_lub_pdit0305_v2/csv/point_anomalies_all.csv"].get_local_copy()

falhas = pd.read_csv(os.path.join(OUT, "alarmes_francisco_falhas.csv"))
falhas["Data da Ocorrência"] = pd.to_datetime(falhas["Data da Ocorrência"])
failure_times = falhas.loc[falhas["Tag Alarme"] == "FALHA_CURADA", "Data da Ocorrência"].sort_values()

for nome, path in [("EXP30 mancal", p_mancal), ("EXP31 óleo", p_oleo)]:
    print(f"\n{'=' * 60}\n{nome}\n{'=' * 60}")
    df = pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False)
    period_days = (df.index.max() - df.index.min()).total_seconds() / 86400.0

    ft = failure_times.loc[(failure_times >= df.index.min()) & (failure_times <= df.index.max())]
    print(f"falhas catalogadas no período: {len(ft)}  período: {period_days:.1f} dias")

    eps = group_alerts_into_episodes(df["is_anom_point"], merge_gap_minutes=120.0)
    print(f"episódios (raw, sem fusão adicional além do gap<2h): {len(eps)}")

    classified = classify_episodes_regua(
        eps, ft, df["operational_state"],
        pre_window_hours=48.0, post_window_hours=48.0, min_stoppage_hours=2.0,
    )
    print(classified["classe"].value_counts())

    metrics = compute_regua_metrics(classified, ft, period_days)
    print("\nmétricas da régua:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    out_csv = os.path.join(OUT, f"regua_episodios_{nome.split()[1]}.csv")
    classified.to_csv(out_csv, index=False)
    print(f"salvo em {out_csv}")
