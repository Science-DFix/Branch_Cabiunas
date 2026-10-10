"""Agrupa os pontos de anomalia (is_anom_point=1) do EXP28 em EPISODIOS
(gap>30min separa, mesmo criterio ja usado no projeto) e mede a
DURACAO de cada um, separando verde (perto de um TRIP curado, +-24h)
de amarelo (FP, isolado) -- mesmo formato de tabela que o Francisco usa
no relatorio dele (inicio/fim/classe/duracao).

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/dataset_francisco_lara/duracao_episodios_verde_amarelo.py
"""
import pandas as pd
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))
POINT_CSV = "/home/dvar/.clearml/cache/storage_manager/global/c27dfd135eb394c5511ce610bd890c1e.point_anomalies_all.csv"
DATA_END = pd.Timestamp("2026-04-20 23:59:59")
GAP_MIN = 30

print("lendo point_anomalies_all.csv (EXP28)...", flush=True)
df = pd.read_csv(POINT_CSV, index_col=0, parse_dates=True)
df = df.loc[df.index <= DATA_END]
anom_times = df.index[df["is_anom_point"] == 1]

falhas = pd.read_csv(os.path.join(OUT, "alarmes_francisco_falhas.csv"))
falhas["Data da Ocorrência"] = pd.to_datetime(falhas["Data da Ocorrência"])
trip_times = falhas.loc[falhas["Tag Alarme"] == "FALHA_CURADA", "Data da Ocorrência"].sort_values()
trip_arr = trip_times.values.astype("datetime64[ns]")
win = np.timedelta64(24, "h")


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

# agrupa TODOS os pontos anomalos (independente da cor) em episodios
# contiguos (gap>30min), depois classifica o episodio pela MAIORIA dos
# seus pontos (verde se qualquer ponto do episodio esta perto de um
# TRIP, amarelo caso contrario) -- mesma logica de "episodio" que o
# Francisco usa (um alerta continuo e 1 evento, nao N pontos).
gaps = pd.Series(anom_times).diff().dt.total_seconds().fillna(1e9) / 60.0
ep_id = (gaps > GAP_MIN).cumsum()
eps = pd.DataFrame({"time": anom_times, "ep_id": ep_id.values, "near_trip": near_trip})
grouped = eps.groupby("ep_id").agg(inicio=("time", "min"), fim=("time", "max"),
                                    n_pontos=("time", "count"), n_perto=("near_trip", "sum"))
grouped["classe"] = np.where(grouped["n_perto"] > 0, "verde", "amarelo")
grouped["duracao_h"] = (grouped["fim"] - grouped["inicio"]).dt.total_seconds() / 3600.0

print(f"\ntotal de episodios: {len(grouped)}", flush=True)
print(grouped[["inicio", "fim", "classe", "duracao_h"]].to_string())

print("\n=== resumo ===")
for classe in ["verde", "amarelo"]:
    sub = grouped.loc[grouped["classe"] == classe, "duracao_h"]
    print(f"\n{classe} ({len(sub)} episodios):")
    print(f"  total: {sub.sum():.1f}h")
    print(f"  media: {sub.mean():.2f}h")
    print(f"  min-max: {sub.min():.2f}h - {sub.max():.2f}h")

n_meses = (df.index.max() - df.index.min()).total_seconds() / (30 * 86400)
amarelo_h_mes = grouped.loc[grouped["classe"] == "amarelo", "duracao_h"].sum() / n_meses
verde_h_mes = grouped.loc[grouped["classe"] == "verde", "duracao_h"].sum() / n_meses
print(f"\nperiodo total: ~{n_meses:.1f} meses")
print(f"amarelo (FP): {amarelo_h_mes:.2f} h/mes")
print(f"verde (acerto): {verde_h_mes:.2f} h/mes")

grouped.to_csv(os.path.join(OUT, "episodios_verde_amarelo_exp28.csv"))
print(f"\nsalvo em {os.path.join(OUT, 'episodios_verde_amarelo_exp28.csv')}")
