"""Restringe a analise anterior EXATAMENTE aos pontos que compoem a
metrica oficial de normal_alert_rate (0,247%) do EXP20: periodo OOS,
estado "on", fora de +-24h dos 2 sensores avaliados, marcados como
anomalia. Desses, quantos coincidem com um alarme de OUTRO sensor do
catalogo completo (47 tags)?

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/relatorio_exp20_filtro_duracao/analise_fp_metrico_vs_catalogo.py
"""
import pandas as pd
import numpy as np
import os
from clearml import Dataset

OUT = "/home/dvar/REPO_CABIUNAS/cabiunas-models/relatorio_exp20_filtro_duracao"
POINT_CSV = "/home/dvar/.clearml/cache/storage_manager/global/18605388c830fc41b0cfa3439a4d5249.point_anomalies_all.csv"
WIN_MIN = 1440
OOS_START = pd.Timestamp("2025-07-01")

print("lendo point_anomalies_all.csv...", flush=True)
df_point = pd.read_csv(POINT_CSV, index_col=0, parse_dates=True)

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
alarm_path = os.path.join(root, "alarmes_selecionados_turbina_a.csv")
alarm = pd.read_csv(alarm_path)
alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrência"], errors="coerce")
alarm["Tag"] = alarm["Tag Alarme"]
alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].copy()
alarm = alarm.dropna(subset=["Data da Ocorrencia"]).sort_values("Data da Ocorrencia").reset_index(drop=True)

sub2 = alarm[alarm["Tag"].isin(["TC382_03_A", "T5_AVG_A"])].sort_values("Data da Ocorrencia")
alarm_times_2 = sub2["Data da Ocorrencia"].values.astype("datetime64[ns]")
alarm_times_full = alarm["Data da Ocorrencia"].values.astype("datetime64[ns]")
alarm_full_sorted = alarm.sort_values("Data da Ocorrencia").reset_index(drop=True)

win = pd.Timedelta(minutes=WIN_MIN)
win_np = np.timedelta64(WIN_MIN, "m")

# --- exatamente a definicao de normal_alert_rate: on, OOS, fora de +-24h dos 2 tags ---
on_mask = (df_point["operational_state"] == "on").values
oos_mask = (df_point.index >= OOS_START)

near2 = pd.Series(False, index=df_point.index)
for t in sub2["Data da Ocorrencia"]:
    t0, t1 = t - win, t + win
    near2.loc[(near2.index >= t0) & (near2.index <= t1)] = True
near2_arr = near2.values

normal_mask = on_mask & oos_mask & (~near2_arr)
flags = df_point["is_anom_point"].values.astype(bool)

fp_mask = normal_mask & flags
print(f"pontos normais (denominador da metrica): {normal_mask.sum()}")
print(f"pontos marcados FP (numerador, is_anom_point=1 dentro do normal_mask): {fp_mask.sum()}")
print(f"normal_alert_rate recomputado: {fp_mask.sum()/normal_mask.sum()*100:.4f}%  (esperado: 0,2468%)")

fp_times = df_point.index[fp_mask]
print(f"\ncruzando os {len(fp_times)} pontos de FP oficial com o catalogo completo ({alarm['Tag'].nunique()} tags)...")

near_full = np.zeros(len(fp_times), dtype=bool)
nearest_tag = [None] * len(fp_times)
nearest_dist_h = np.zeros(len(fp_times))
for i, t in enumerate(fp_times.values):
    p = np.searchsorted(alarm_times_full, t)
    cands_idx = []
    if p > 0:
        cands_idx.append(p - 1)
    if p < len(alarm_times_full):
        cands_idx.append(p)
    best_d, best_tag = None, None
    for ci in cands_idx:
        d = abs(t - alarm_times_full[ci])
        if best_d is None or d < best_d:
            best_d, best_tag = d, alarm_full_sorted.loc[ci, "Tag"]
    nearest_dist_h[i] = best_d / np.timedelta64(1, "h")
    nearest_tag[i] = best_tag
    near_full[i] = best_d <= win_np

n_total = len(fp_times)
n_near_full = near_full.sum()
n_isolado = n_total - n_near_full
print(f"\n=== RESULTADO (pontos que compoem a metrica de 0,247%) ===")
print(f"Total de pontos contados como FP: {n_total}")
print(f"  Coincidem com alarme de OUTRO sensor (<=24h): {n_near_full} ({n_near_full/n_total*100:.1f}%)")
print(f"  Genuinamente isolados (nenhum alarme de nenhum dos 47 tags por perto): {n_isolado} ({n_isolado/n_total*100:.1f}%)")

fp_rate_original = fp_mask.sum() / normal_mask.sum()
fp_rate_genuino = n_isolado / normal_mask.sum()
print(f"\nnormal_alert_rate oficial (como reportado hoje): {fp_rate_original*100:.4f}%")
print(f"normal_alert_rate 'genuino' (excluindo coincidencia com outro sensor): {fp_rate_genuino*100:.4f}%")
print(f"reducao se considerarmos so o genuino: {(1-fp_rate_genuino/fp_rate_original)*100:.1f}%")

df_out = pd.DataFrame({"time": fp_times, "near_other_sensor": near_full,
                        "nearest_tag": nearest_tag, "nearest_dist_h": nearest_dist_h})
df_out.to_csv(os.path.join(OUT, "fp_oficial_vs_catalogo_completo.csv"), index=False)

print("\ntags mais frequentes entre os FP 'recuperados' (perto de outro sensor):")
rec = df_out.loc[df_out["near_other_sensor"]]
print(rec["nearest_tag"].value_counts().head(15).to_string())
