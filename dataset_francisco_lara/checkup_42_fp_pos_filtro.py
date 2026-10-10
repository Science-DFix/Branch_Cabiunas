"""Checkup individual dos 42 episodios de falso positivo remanescentes
depois do filtro de duracao minima de 45min (pipeline_unificada_final,
votacao >=2 + filtro 45min + refratario 48h) -- mesmo metodo do
checkup dos 24 isolados do EXP30 (Apendice A do relatorio): z-score
fresco do episodio inteiro contra um baseline de 2h antes, numa unica
passada chunked pelo CSV bruto.

Diferenca em relacao ao checkup anterior: os episodios aqui tem
duracao real (48min a 9,65h), nao sao pontuais -- e o conjunto de
sensores inclui tambem pressao de oleo (PI_0308, PDIT_0305), ja que a
pipeline atual usa esse canal.

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/dataset_francisco_lara/checkup_42_fp_pos_filtro.py
"""
import os

import numpy as np
import pandas as pd
from clearml import Dataset

OUT = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(OUT, "..", "runs_pipeline_unificada_final")

VIB = ["TV_351X_A", "TV_351Y_A", "TV_352X_A", "TV_352Y_A", "TV_353X_A", "TV_353Y_A",
       "TV_354X_A", "TV_354Y_A", "TV_355X_A", "TV_355Y_A"]
OLEO = ["954005_624_PI_0308", "954005_624_PDIT_0305"]
COLS = ["TC382_03_A", "T5_AVG_A", "954005_624_TI_0305"] + VIB + OLEO

cls = pd.read_csv(os.path.join(RUN_DIR, "episodios_classificados.csv"), parse_dates=["start", "end", "falha_associada"])
fp = cls[cls["classe"] == "falso_positivo"].sort_values("start").reset_index(drop=True)
print(f"{len(fp)} episodios de falso positivo a auditar", flush=True)

windows = {}
for i, row in fp.iterrows():
    t0, t1 = row["start"], row["end"]
    name = t0.strftime("%Y-%m-%d_%Hh%M")
    windows[name] = dict(baseline=(t0 - pd.Timedelta(hours=2), t0 - pd.Timedelta(minutes=3)),
                          episode=(t0, t1))

global_min = min(w["baseline"][0] for w in windows.values())
global_max = max(w["episode"][1] for w in windows.values())
print(f"varrendo CSV de {global_min} ate {global_max} (uma passada)...", flush=True)

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
raw_path = os.path.join(root, "sensores_full_2024_2026_30s.csv")
cols_read = ["data_datetime"] + COLS

collected = []
for chunk in pd.read_csv(raw_path, usecols=cols_read, chunksize=200_000, dtype=str):
    chunk["data_datetime"] = pd.to_datetime(chunk["data_datetime"], errors="coerce")
    chunk = chunk.dropna(subset=["data_datetime"])
    m = (chunk["data_datetime"] >= global_min) & (chunk["data_datetime"] <= global_max)
    if m.any():
        collected.append(chunk.loc[m])

df = pd.concat(collected, ignore_index=True)
for c in COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.set_index("data_datetime").sort_index()
print(f"linhas coletadas: {len(df)}", flush=True)

resumo = []
for name, w in windows.items():
    b0, b1 = w["baseline"]
    e0, e1 = w["episode"]
    baseline = df.loc[b0:b1]
    episode = df.loc[e0:e1]
    if baseline.empty or episode.empty:
        resumo.append({"episodio": name, "abs_z_max": np.nan, "sensor_max": "SEM_DADOS", "dur_min": (e1 - e0).total_seconds() / 60.0})
        continue
    bmean, bstd = baseline.mean(), baseline.std()
    # pico pontual (max |z| dentro do episodio), nao media do episodio --
    # episodios longos (fusao de varios minutos/horas) diluiriam um pico
    # breve real se usassemos a media inteira do episodio.
    z_series = (episode - bmean) / bstd
    z_peak = z_series.abs().max()
    z_sorted = z_peak.sort_values(ascending=False)
    n_sensores_acima_3 = (z_sorted > 3).sum()
    resumo.append({
        "episodio": name, "dur_min": (e1 - e0).total_seconds() / 60.0,
        "abs_z_max": float(z_sorted.iloc[0]),
        "sensor_max": z_sorted.index[0], "n_sensores_|z|>3": int(n_sensores_acima_3),
    })

resumo_df = pd.DataFrame(resumo)


def classifica(row):
    if pd.isna(row["abs_z_max"]):
        return "sem dados"
    if row["abs_z_max"] >= 10 and row["n_sensores_|z|>3"] >= 2:
        return "CONFIRMADO real (forte)"
    if row["abs_z_max"] >= 3 and row["n_sensores_|z|>3"] >= 2:
        return "confirmado real (moderado)"
    if row["abs_z_max"] >= 3:
        return "borderline (1 sensor só)"
    return "FRACO / sem explicação"


resumo_df["classificacao"] = resumo_df.apply(classifica, axis=1)
resumo_df = resumo_df.sort_values("abs_z_max", ascending=False)
print("\n=== CHECKUP DOS 42 FP PÓS-FILTRO (z-score fresco) ===")
print(resumo_df.to_string(index=False))

print("\n=== RESUMO POR CATEGORIA ===")
print(resumo_df["classificacao"].value_counts())
print(f"\n% confirmado real (forte+moderado): {100*resumo_df['classificacao'].isin(['CONFIRMADO real (forte)','confirmado real (moderado)']).mean():.1f}%")

resumo_df.to_csv(os.path.join(OUT, "checkup_42_fp_pos_filtro.csv"), index=False)
print(f"\nsalvo em {os.path.join(OUT, 'checkup_42_fp_pos_filtro.csv')}")
