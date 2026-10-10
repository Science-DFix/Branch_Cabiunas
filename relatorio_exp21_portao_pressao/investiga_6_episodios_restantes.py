"""Fecha a investigacao dos 79 pontos do EXP22 ainda nao explicados
(6 episodios >1min, fora do padrao de "borda de gate" e fora do evento de
vibracao ja confirmado em 2026-03-10). Mesma tecnica que resolveu aquele
caso: z-score de TC382_03_A/T5_AVG_A/10 canais de vibracao contra uma
baseline de 2h antes de cada episodio. Leitura em UMA UNICA passada
chunked (chunksize=200_000, dtype=str) pelo CSV bruto, coletando so as
janelas necessarias -- evita OOM (ja tivemos 2 kills nesta maquina lendo
o arquivo inteiro sem filtrar).

Tambem verifica as colunas `*_blocked` do point_anomalies_all.csv perto
de cada episodio, pra checar se algum portao quase disparou.

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/relatorio_exp21_portao_pressao/investiga_6_episodios_restantes.py
"""
import pandas as pd
import numpy as np
import os
from clearml import Dataset

OUT = os.path.dirname(os.path.abspath(__file__))
POINT_CSV = "/home/dvar/.clearml/cache/storage_manager/global/950100358f8e4a8dcba408279af95237.point_anomalies_all.csv"

EPISODES = [
    ("2025-08-26_22h18", "2025-08-26 22:18:30", "2025-08-26 22:20:00"),
    ("2025-08-27_00h37", "2025-08-27 00:37:30", "2025-08-27 00:54:30"),
    ("2025-10-17_13h49", "2025-10-17 13:49:00", "2025-10-17 13:54:30"),
    ("2025-10-22_18h18", "2025-10-22 18:18:30", "2025-10-22 18:22:30"),
    ("2025-12-15_07h17", "2025-12-15 07:17:30", "2025-12-15 07:21:30"),
    ("2025-12-24_10h54", "2025-12-24 10:54:30", "2025-12-24 10:59:00"),
]

VIB = ["TV_351X_A", "TV_351Y_A", "TV_352X_A", "TV_352Y_A", "TV_353X_A", "TV_353Y_A",
       "TV_354X_A", "TV_354Y_A", "TV_355X_A", "TV_355Y_A"]
COLS = ["TC382_03_A", "T5_AVG_A"] + VIB

print("=== 1) checando colunas *_blocked perto de cada episodio (+-30min) ===", flush=True)
df_point = pd.read_csv(POINT_CSV, index_col=0, parse_dates=True)
blocked_cols = ["duration_filter_blocked", "load_gate_blocked", "volatility_gate_blocked", "step_change_gate_blocked"]
for name, t0, t1 in EPISODES:
    t0p, t1p = pd.Timestamp(t0) - pd.Timedelta(minutes=30), pd.Timestamp(t1) + pd.Timedelta(minutes=30)
    win = df_point.loc[(df_point.index >= t0p) & (df_point.index <= t1p)]
    any_blocked = {c: int(win[c].sum()) for c in blocked_cols}
    print(f"{name}: {any_blocked}", flush=True)

print("\n=== 2) coletando janelas brutas (baseline 2h antes + episodio) ===", flush=True)
windows = {}
for name, t0, t1 in EPISODES:
    t0 = pd.Timestamp(t0)
    t1 = pd.Timestamp(t1)
    baseline_start = t0 - pd.Timedelta(hours=2)
    baseline_end = t0 - pd.Timedelta(minutes=3)
    episode_start = t0
    episode_end = t1
    windows[name] = dict(baseline=(baseline_start, baseline_end), episode=(episode_start, episode_end))

global_min = min(w["baseline"][0] for w in windows.values())
global_max = max(w["episode"][1] for w in windows.values())
print(f"varrendo CSV de {global_min} ate {global_max} (uma passada)...", flush=True)

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
raw_path = os.path.join(root, "sensores_full_2024_2026_30s.csv")

collected = []
cols_read = ["data_datetime"] + COLS
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

print("\n=== 3) z-score por episodio (baseline 2h antes vs janela do episodio) ===", flush=True)
summary_rows = []
for name, w in windows.items():
    b0, b1 = w["baseline"]
    e0, e1 = w["episode"]
    baseline = df.loc[b0:b1]
    episode = df.loc[e0:e1]
    if baseline.empty or episode.empty:
        print(f"\n{name}: SEM DADOS (baseline={len(baseline)}, episodio={len(episode)})", flush=True)
        continue
    bmean, bstd = baseline.mean(), baseline.std()
    z = (episode.mean() - bmean) / bstd
    z_sorted = z.sort_values(key=lambda s: s.abs(), ascending=False)
    top = z_sorted.head(3)
    print(f"\n--- {name} ({e0} a {e1}) ---")
    print(z_sorted.to_string())
    summary_rows.append({
        "episodio": name, "top1_canal": top.index[0], "top1_z": top.iloc[0],
        "top2_canal": top.index[1], "top2_z": top.iloc[1],
    })

print("\n=== RESUMO ===")
df_summary = pd.DataFrame(summary_rows)
print(df_summary.to_string(index=False))
df_summary.to_csv(os.path.join(OUT, "resumo_6_episodios_restantes.csv"), index=False)
print(f"\nsalvo em {os.path.join(OUT, 'resumo_6_episodios_restantes.csv')}")
