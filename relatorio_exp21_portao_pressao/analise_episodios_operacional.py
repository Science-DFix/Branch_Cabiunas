"""Reframe operacional: o que importa pro operador nao e "quantos PONTOS
de 30s sao FP", e sim "quantos ALERTAS/EPISODIOS distintos o sistema
dispara por mes, e quantos desses sao ruido". Agrupa is_anom_point final
do EXP22 (dominio on+OOS, igual a producao) em EPISODIOS (gap>30min
separa), classifica cada episodio contra: (a) os 2 tags oficiais +-24h,
(b) o catalogo completo de 47 tags +-24h, (c) sobreposicao com os 4
timestamps confirmados do padrao recorrente de vibracao no mancal 354.
O que sobrar sem nenhuma correspondencia e o "alerta de ruido" de
verdade, contado em EPISODIOS (nao em pontos) e normalizado por mes.

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/relatorio_exp21_portao_pressao/analise_episodios_operacional.py
"""
import pandas as pd
import numpy as np
import os
from clearml import Dataset

OUT = os.path.dirname(os.path.abspath(__file__))
POINT_CSV = "/home/dvar/.clearml/cache/storage_manager/global/950100358f8e4a8dcba408279af95237.point_anomalies_all.csv"
DATA_END = pd.Timestamp("2026-04-20 23:59:59")
OOS_START = pd.Timestamp("2025-07-01")
WIN_2TAGS_MIN = 1440
WIN_CATALOGO_MIN = 1440
GAP_MIN = 30  # mesmo criterio de agrupamento ja usado nas analises anteriores

print("lendo point_anomalies_all.csv (EXP22)...", flush=True)
df_point = pd.read_csv(POINT_CSV, index_col=0, parse_dates=True)
df_point = df_point.loc[df_point.index <= DATA_END]

on_arr = (df_point["operational_state"] == "on")
oos_arr = (df_point.index >= OOS_START)
dom = df_point.loc[on_arr & oos_arr]
n_months = (dom.index.max() - dom.index.min()).total_seconds() / (30 * 86400)
print(f"dominio operacional (on+OOS): {len(dom)} pontos, {dom.index.min()} a {dom.index.max()} (~{n_months:.1f} meses)", flush=True)

anom_times = dom.index[dom["is_anom_point"] == 1]
print(f"pontos anomalos no dominio: {len(anom_times)}", flush=True)

# agrupa em episodios (gap>30min)
gaps = pd.Series(anom_times).diff().dt.total_seconds().fillna(1e9) / 60.0
ep_id = (gaps > GAP_MIN).cumsum()
eps = pd.DataFrame({"time": anom_times, "ep_id": ep_id.values})
episodes = eps.groupby("ep_id")["time"].agg(["min", "max", "count"])
episodes.columns = ["start", "end", "n_points"]
print(f"\ntotal de EPISODIOS distintos (alertas que um operador veria): {len(episodes)}", flush=True)
print(f"taxa: {len(episodes)/n_months:.2f} alertas/mes", flush=True)

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
alarm_path = os.path.join(root, "alarmes_selecionados_turbina_a.csv")
alarm = pd.read_csv(alarm_path)
alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrência"], errors="coerce")
alarm["Tag"] = alarm["Tag Alarme"]
alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].copy()
alarm = alarm.dropna(subset=["Data da Ocorrencia"]).sort_values("Data da Ocorrencia").reset_index(drop=True)

sub2_times = alarm.loc[alarm["Tag"].isin(["TC382_03_A", "T5_AVG_A"]), "Data da Ocorrencia"].values.astype("datetime64[ns]")
cat_times = alarm["Data da Ocorrencia"].values.astype("datetime64[ns]")
win2 = np.timedelta64(WIN_2TAGS_MIN, "m")
winc = np.timedelta64(WIN_CATALOGO_MIN, "m")

MANCAL_354_EVENTS = pd.to_datetime([
    "2025-10-17 13:49:00", "2025-12-15 07:17:30", "2025-12-24 10:54:30", "2026-03-10 10:23:00",
]).values.astype("datetime64[ns]")
win354 = np.timedelta64(30, "m")


def near_any_center(center, times_sorted, win):
    if len(times_sorted) == 0:
        return False
    pos = np.searchsorted(times_sorted, center)
    cands = []
    if pos > 0:
        cands.append(times_sorted[pos - 1])
    if pos < len(times_sorted):
        cands.append(times_sorted[pos])
    return any(abs(center - c) <= win for c in cands)


cats = []
for _, row in episodes.iterrows():
    center = (row["start"] + (row["end"] - row["start"]) / 2).to_datetime64()
    if near_any_center(center, sub2_times, win2):
        cats.append("alarme_oficial")
    elif near_any_center(center, MANCAL_354_EVENTS, win354):
        cats.append("padrao_mancal354")
    elif near_any_center(center, cat_times, winc):
        cats.append("catalogo_47tags")
    else:
        cats.append("isolado")

episodes["categoria"] = cats
resumo = episodes["categoria"].value_counts()
print("\ncategorias dos episodios:")
print(resumo.to_string())
print(f"\ntaxa de episodios ISOLADOS (ruido de verdade): {resumo.get('isolado', 0)/n_months:.2f}/mes")

episodes.to_csv(os.path.join(OUT, "episodios_operacional_exp22.csv"))
print(f"\nsalvo em {os.path.join(OUT, 'episodios_operacional_exp22.csv')}")

print("\nepisodios isolados (detalhe):")
print(episodes.loc[episodes["categoria"] == "isolado"].to_string())
