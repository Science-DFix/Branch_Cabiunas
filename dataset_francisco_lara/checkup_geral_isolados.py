"""Checkup geral e definitivo dos 24 episodios isolados do EXP30 --
z-score fresco (nao reaproveitado de memoria) contra baseline de 2h,
todos os 24 numa unica passada chunked.

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/dataset_francisco_lara/checkup_geral_isolados.py
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
    ("2025-08-26_22h18", "2025-08-26 22:18:30", "2025-08-26 22:20:00"),
    ("2025-08-27_00h37_17min", "2025-08-27 00:37:30", "2025-08-27 00:54:30"),
    ("2025-08-27_01h51", "2025-08-27 01:51:00", "2025-08-27 01:51:00"),
    ("2025-10-17_13h49", "2025-10-17 13:49:00", "2025-10-17 13:54:30"),
    ("2025-10-22_18h18", "2025-10-22 18:18:30", "2025-10-22 18:22:30"),
    ("2025-12-15_07h17", "2025-12-15 07:17:30", "2025-12-15 07:21:30"),
    ("2025-12-16_05h22", "2025-12-16 05:22:00", "2025-12-16 05:22:30"),
    ("2025-12-24_10h54", "2025-12-24 10:54:30", "2025-12-24 10:59:00"),
    ("2026-01-21_11h59", "2026-01-21 11:59:30", "2026-01-21 11:59:30"),
    ("2026-03-10_10h23", "2026-03-10 10:23:00", "2026-03-10 10:42:00"),
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
        resumo.append({"episodio": name, "abs_z_max": np.nan, "sensor_max": "SEM_DADOS"})
        continue
    bmean, bstd = baseline.mean(), baseline.std()
    z = (episode.mean() - bmean) / bstd
    z_sorted = z.sort_values(key=lambda s: s.abs(), ascending=False)
    n_sensores_acima_3 = (z_sorted.abs() > 3).sum()
    resumo.append({
        "episodio": name, "abs_z_max": float(z_sorted.abs().iloc[0]),
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
print("\n=== 3) CHECKUP COMPLETO DOS 24 EPISODIOS ISOLADOS (z-score fresco) ===")
print(resumo_df.to_string(index=False))

print("\n=== RESUMO POR CATEGORIA ===")
print(resumo_df["classificacao"].value_counts())

resumo_df.to_csv(os.path.join(OUT, "checkup_geral_isolados.csv"), index=False)
print(f"\nsalvo em {os.path.join(OUT, 'checkup_geral_isolados.csv')}")
