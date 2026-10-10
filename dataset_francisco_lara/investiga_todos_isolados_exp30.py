"""Investigacao completa (honesta) dos 24 episodios isolados do EXP30:
z-score contra baseline de 2h, mesma metodologia ja usada, cobrindo
TODOS os episodios -- inclusive os que nunca foram checados
individualmente antes (a lista anterior de "24 fechados" cobria so um
subconjunto). Leitura chunked numa unica passada pra evitar OOM.

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/dataset_francisco_lara/investiga_todos_isolados_exp30.py
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
    ("2025-08-26_22h18", "2025-08-26 22:18:30", "2025-08-26 22:20:00"),
    ("2025-08-27_01h51", "2025-08-27 01:51:00", "2025-08-27 01:51:00"),
    ("2025-10-22_18h18", "2025-10-22 18:18:30", "2025-10-22 18:22:30"),
    ("2025-12-16_05h22", "2025-12-16 05:22:00", "2025-12-16 05:22:30"),
    ("2026-01-21_11h59", "2026-01-21 11:59:30", "2026-01-21 11:59:30"),
    ("2026-03-11_09h32", "2026-03-11 09:32:30", "2026-03-11 09:32:30"),
    ("2026-03-17_18h50", "2026-03-17 18:50:30", "2026-03-17 18:51:00"),
    ("2026-03-18_08h15", "2026-03-18 08:15:00", "2026-03-18 08:15:00"),
    ("2026-03-20_14h08", "2026-03-20 14:08:30", "2026-03-20 14:09:30"),
    ("2026-03-23_21h39", "2026-03-23 21:39:00", "2026-03-23 21:39:00"),
    ("2026-03-27_21h23", "2026-03-27 21:23:00", "2026-03-27 21:23:00"),
    ("2026-04-10_23h03", "2026-04-10 23:03:00", "2026-04-10 23:03:30"),
    ("2026-04-11_08h57", "2026-04-11 08:57:00", "2026-04-11 08:57:00"),
    ("2026-04-11_10h20", "2026-04-11 10:20:30", "2026-04-11 10:21:00"),
    ("2026-04-12_17h33", "2026-04-12 17:33:30", "2026-04-12 17:33:30"),
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
        print(f"\n{name}: SEM DADOS suficientes (baseline={len(baseline)}, episodio={len(episode)})")
        continue
    bmean, bstd = baseline.mean(), baseline.std()
    z = (episode.mean() - bmean) / bstd
    z_sorted = z.sort_values(key=lambda s: s.abs(), ascending=False)
    max_abs_z = z_sorted.abs().max()
    print(f"\n--- {name} ({e0} a {e1}) --- |z|max={max_abs_z:.2f}")
    print(z_sorted.head(4).to_string())
    resumo.append({"episodio": name, "abs_z_max": max_abs_z, "sensor_max": z_sorted.index[0]})

print("\n\n=== RESUMO ORDENADO POR |z|max ===")
resumo_df = pd.DataFrame(resumo).sort_values("abs_z_max", ascending=False)
print(resumo_df.to_string(index=False))
