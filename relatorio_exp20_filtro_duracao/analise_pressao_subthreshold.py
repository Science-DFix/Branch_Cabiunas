"""Checa se os 455 pontos genuinamente isolados (FP sem nenhum alarme
por perto, nem dos 47 tags do catalogo) mostram sinal de PRESSAO se
movendo abaixo do limiar de alarme -- os 3 tags de pressao que mais
explicaram os FP "recuperados" foram PI_6240319_AL, PAL_6240315,
PDAL_6240302, que correspondem as colunas brutas:
  954005_624_PI_0319, 954005_624_PI_0315, 954005_624_PDI_0302

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/relatorio_exp20_filtro_duracao/analise_pressao_subthreshold.py
"""
import pandas as pd
import numpy as np
import os
from clearml import Dataset

OUT = "/home/dvar/REPO_CABIUNAS/cabiunas-models/relatorio_exp20_filtro_duracao"

df_fp = pd.read_csv(os.path.join(OUT, "fp_oficial_vs_catalogo_completo.csv"), parse_dates=["time"])
isolados = df_fp.loc[~df_fp["near_other_sensor"]].sort_values("time").reset_index(drop=True)
print(f"pontos genuinamente isolados: {len(isolados)}", flush=True)

# agrupa em episodios (gap <= 30min = mesmo episodio)
gaps = isolados["time"].diff().dt.total_seconds().fillna(1e9) / 60.0
isolados["ep_id"] = (gaps > 30).cumsum()
episodios = isolados.groupby("ep_id")["time"].agg(["min", "max", "count"]).reset_index(drop=True)
episodios.columns = ["start", "end", "n_points"]
print(f"episodios distintos (gap>30min): {len(episodios)}", flush=True)

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
raw_path = os.path.join(root, "sensores_full_2024_2026_30s.csv")
cols = ["data_datetime", "954005_624_PI_0319", "954005_624_PI_0315", "954005_624_PDI_0302"]
print("lendo colunas de pressao...", flush=True)
df_p = pd.read_csv(raw_path, usecols=cols)
df_p["data_datetime"] = pd.to_datetime(df_p["data_datetime"], errors="coerce")
for c in cols[1:]:
    df_p[c] = pd.to_numeric(df_p[c], errors="coerce")
df_p = df_p.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
pressure_cols = cols[1:]

# baseline: media/desvio moveis causais de 24h, terminando 2h antes da janela de checagem
# (evita vazamento -- baseline "recente" sem incluir o proprio episodio)
window_samples = int(24 * 3600 / 30)  # 24h a 30s
baseline_mean = df_p[pressure_cols].rolling(window_samples, min_periods=window_samples // 4).mean().shift(240)  # shift 2h
baseline_std = df_p[pressure_cols].rolling(window_samples, min_periods=window_samples // 4).std().shift(240)

results = []
for _, ep in episodios.iterrows():
    t0, t1 = ep["start"] - pd.Timedelta(hours=2), ep["end"] + pd.Timedelta(hours=2)
    window = df_p.loc[(df_p.index >= t0) & (df_p.index <= t1)]
    if window.empty:
        continue
    row = {"start": ep["start"], "end": ep["end"], "n_points": ep["n_points"]}
    max_abs_z = 0.0
    best_col = None
    for c in pressure_cols:
        bm = baseline_mean.loc[window.index, c]
        bs = baseline_std.loc[window.index, c].replace(0, np.nan)
        z = (window[c] - bm) / bs
        z_abs_max = z.abs().max()
        row[f"maxabsz_{c}"] = z_abs_max
        if pd.notna(z_abs_max) and z_abs_max > max_abs_z:
            max_abs_z = z_abs_max
            best_col = c
    row["max_abs_z_overall"] = max_abs_z
    row["best_col"] = best_col
    results.append(row)

df_res = pd.DataFrame(results)
df_res.to_csv(os.path.join(OUT, "episodios_isolados_vs_pressao_subthreshold.csv"), index=False)

print(f"\nepisodios avaliados: {len(df_res)}", flush=True)
for thr in [1.5, 2.0, 2.5, 3.0]:
    n = (df_res["max_abs_z_overall"] >= thr).sum()
    print(f"episodios com |z| >= {thr} em pelo menos 1 canal de pressao: {n}/{len(df_res)} ({n/len(df_res)*100:.1f}%)")

print("\ndistribuicao do max |z| entre pressao nos episodios isolados:")
print(df_res["max_abs_z_overall"].describe().to_string())

print("\n10 episodios com maior |z| de pressao (mais suspeitos de serem sub-threshold real):")
print(df_res.sort_values("max_abs_z_overall", ascending=False).head(10).to_string())
