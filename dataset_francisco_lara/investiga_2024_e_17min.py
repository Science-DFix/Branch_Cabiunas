"""Investiga os 2 episodios isolados de 2024 (nunca checados antes) e
reconfirma o de 17min (2025-08-27) do residual do EXP28 -- mesma
tecnica de z-score contra baseline de 2h, leitura chunked numa unica
passada pra evitar OOM.

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/dataset_francisco_lara/investiga_2024_e_17min.py
"""
import pandas as pd
import numpy as np
import os
from clearml import Dataset

OUT = os.path.dirname(os.path.abspath(__file__))

VIB = ["TV_351X_A", "TV_351Y_A", "TV_352X_A", "TV_352Y_A", "TV_353X_A", "TV_353Y_A",
       "TV_354X_A", "TV_354Y_A", "TV_355X_A", "TV_355Y_A"]
COLS = ["TC382_03_A", "T5_AVG_A", "954005_624_TI_0305"] + VIB

EPISODIOS = [
    ("2024-09-30_18h04", "2024-09-30 18:04:30", "2024-09-30 18:05:00"),
    ("2024-09-30_19h08", "2024-09-30 19:08:30", "2024-09-30 19:08:30"),
    ("2024-12-12_12h09", "2024-12-12 12:09:00", "2024-12-12 12:10:00"),
    ("2024-12-12_14h34", "2024-12-12 14:34:30", "2024-12-12 14:34:30"),
    ("2025-08-27_00h37_17min", "2025-08-27 00:37:30", "2025-08-27 00:54:30"),
]

windows = {}
for name, t0, t1 in EPISODIOS:
    t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)
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
    if len(chunk) and chunk["data_datetime"].max() > global_max:
        break

df = pd.concat(collected, ignore_index=True)
for c in COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.set_index("data_datetime").sort_index()
print(f"linhas coletadas: {len(df)}", flush=True)

for name, w in windows.items():
    b0, b1 = w["baseline"]
    e0, e1 = w["episode"]
    baseline = df.loc[b0:b1]
    episode = df.loc[e0:e1]
    if baseline.empty or episode.empty:
        print(f"\n{name}: SEM DADOS suficientes (baseline={len(baseline)}, episodio={len(episode)})")
        continue
    bmean, bstd = baseline.mean(), baseline.std()
    z = (episode.mean() - bmean) / bstd
    z_sorted = z.sort_values(key=lambda s: s.abs(), ascending=False)
    print(f"\n--- {name} ({e0} a {e1}) ---")
    print(z_sorted.to_string())
