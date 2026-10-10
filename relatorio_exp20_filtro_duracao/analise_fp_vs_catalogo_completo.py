"""Recruza as anomalias detectadas (EXP20, filtro 4,5min) com o
CATALOGO COMPLETO de alarmes (47 tags: temperatura, pressao, vibracao),
nao so os 2 sensores avaliados (TC382_03_A, T5_AVG_A). Pergunta: quantos
dos pontos/episodios antes classificados como "isolados" na verdade
coincidem com um alarme de OUTRO sensor (ex: pressao de oleo/gas),
sendo parte de um evento fisico real que so nao dispara os 2 tags
avaliados?

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/relatorio_exp20_filtro_duracao/analise_fp_vs_catalogo_completo.py
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
alarm = alarm.dropna(subset=["Data da Ocorrencia"]).sort_values("Data da Ocorrencia").reset_index(drop=True)
print(f"CATALOGO COMPLETO: {len(alarm)} eventos ACT, {alarm['Tag'].nunique()} tags distintos", flush=True)

# so os 2 tags avaliados, pra comparacao lado a lado
sub2 = alarm[alarm["Tag"].isin(["TC382_03_A", "T5_AVG_A"])]
print(f"so os 2 tags avaliados: {len(sub2)} eventos", flush=True)

alarm_times_full = alarm["Data da Ocorrencia"].sort_values().values.astype("datetime64[ns]")
alarm_tags_by_time = alarm.sort_values("Data da Ocorrencia")[["Data da Ocorrencia", "Tag"]].reset_index(drop=True)

win = np.timedelta64(WIN_MIN, "m")


def nearest_alarm_info(t):
    """Retorna (min_dist_horas, tag_mais_proximo) usando o catalogo completo."""
    p = np.searchsorted(alarm_times_full, t)
    cands_idx = []
    if p > 0:
        cands_idx.append(p - 1)
    if p < len(alarm_times_full):
        cands_idx.append(p)
    best = None
    for ci in cands_idx:
        d = abs(t - alarm_times_full[ci])
        if best is None or d < best[0]:
            best = (d, ci)
    return best[0] / np.timedelta64(1, "h"), alarm_tags_by_time.loc[best[1], "Tag"]


# --- nivel de PONTO (todas as 6406 anomalias) ---
flags = df_point["is_anom_point"].values.astype(bool)
anom_times = df_point.index[flags]
print(f"\nanomalias totais (pos-filtro 4.5min): {len(anom_times)}", flush=True)

near_full = np.zeros(len(anom_times), dtype=bool)
nearest_tag = [None] * len(anom_times)
nearest_dist_h = np.zeros(len(anom_times))
for i, t in enumerate(anom_times.values):
    d_h, tag = nearest_alarm_info(t)
    nearest_dist_h[i] = d_h
    nearest_tag[i] = tag
    near_full[i] = d_h <= WIN_MIN / 60.0

print(f"perto de QUALQUER alarme do catalogo completo (<=24h): {near_full.sum()} ({near_full.mean()*100:.1f}%)")
print(f"isolados mesmo contra o catalogo completo: {(~near_full).sum()} ({(~near_full).mean()*100:.1f}%)")

# comparacao: quantos que eram "isolados" so-vs-2-tags viram "perto" quando olhamos o catalogo completo
near_2tags = np.zeros(len(anom_times), dtype=bool)
alarm_times_2 = sub2["Data da Ocorrencia"].sort_values().values.astype("datetime64[ns]")
pos2 = np.searchsorted(alarm_times_2, anom_times.values)
for i in range(len(anom_times)):
    p = pos2[i]
    cands = []
    if p > 0:
        cands.append(alarm_times_2[p - 1])
    if p < len(alarm_times_2):
        cands.append(alarm_times_2[p])
    t = anom_times.values[i]
    near_2tags[i] = any(abs(t - c) <= win for c in cands) if cands else False

recuperados = (~near_2tags) & near_full
print(f"\nEram 'isolados' so contra os 2 tags mas tem alarme de OUTRO sensor por perto: "
      f"{recuperados.sum()} / {(~near_2tags).sum()} isolados originais "
      f"({recuperados.sum()/max(1,(~near_2tags).sum())*100:.1f}%)")

df_result = pd.DataFrame({
    "time": anom_times, "near_2tags": near_2tags, "near_full_catalog": near_full,
    "nearest_tag": nearest_tag, "nearest_dist_h": nearest_dist_h,
})
df_result.to_csv(os.path.join(OUT, "anomalias_vs_catalogo_completo.csv"), index=False)

print("\ntags mais frequentes entre os alarmes mais proximos dos pontos AINDA isolados (catalogo completo):")
ainda_isolados = df_result.loc[~df_result["near_full_catalog"]]
print(ainda_isolados["nearest_tag"].value_counts().head(15).to_string())

print("\ntags mais frequentes entre os 'recuperados' (perto so quando olha catalogo completo):")
rec = df_result.loc[recuperados]
print(rec["nearest_tag"].value_counts().head(15).to_string())

print(f"\n=== RESUMO FINAL ===")
print(f"Total de anomalias detectadas: {len(anom_times)}")
print(f"  Perto de alarme dos 2 sensores avaliados: {near_2tags.sum()} ({near_2tags.mean()*100:.1f}%)")
print(f"  Perto de alarme de OUTRO sensor (recuperado): {recuperados.sum()} ({recuperados.sum()/len(anom_times)*100:.1f}%)")
print(f"  Isolado mesmo contra os 47 tags do catalogo: {(~near_full).sum()} ({(~near_full).mean()*100:.1f}%)")
