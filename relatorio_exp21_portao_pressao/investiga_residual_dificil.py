"""Plota o contexto bruto (sensores do grupo + pressao + RUNNING_A) ao
redor dos episodios isolados com MENOR z-score de pressao (os que nem a
pressao sub-limiar explica) -- exclui os que ja identificamos como
buraco de dados (>=2026-04-21).

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/relatorio_exp21_portao_pressao/investiga_residual_dificil.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from clearml import Dataset

OUT = os.path.dirname(os.path.abspath(__file__))
EP_CSV = "/home/dvar/REPO_CABIUNAS/cabiunas-models/relatorio_exp20_filtro_duracao/episodios_isolados_vs_pressao_subthreshold.csv"

df_ep = pd.read_csv(EP_CSV, parse_dates=["start", "end"])
df_ep = df_ep[df_ep["start"] < "2026-04-21"]  # exclui o buraco de dados ja explicado
df_ep = df_ep.sort_values("max_abs_z_overall")
print(f"episodios remanescentes (fora do buraco de dados): {len(df_ep)}", flush=True)
print(df_ep.head(8)[["start", "end", "n_points", "max_abs_z_overall", "best_col"]].to_string())

top5 = df_ep.head(5)

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
raw_path = os.path.join(root, "sensores_full_2024_2026_30s.csv")
cols = ["data_datetime", "TC382_03_A", "T5_AVG_A",
        "TV_351X_A", "TV_351Y_A", "TV_352X_A", "TV_352Y_A", "TV_353X_A",
        "TV_353Y_A", "TV_354X_A", "TV_354Y_A", "TV_355X_A", "TV_355Y_A",
        "954005_624_PI_0319", "954005_624_PI_0315", "954005_624_PDI_0302",
        "RUNNING_A"]
windows = [(ep["start"] - pd.Timedelta(hours=3), ep["end"] + pd.Timedelta(hours=3)) for _, ep in top5.iterrows()]

print("\nlendo raw EM PEDACOS (so as janelas necessarias, pra nao estourar memoria)...", flush=True)
chunks = []
for chunk in pd.read_csv(raw_path, usecols=cols, chunksize=200_000, dtype=str):
    chunk["data_datetime"] = pd.to_datetime(chunk["data_datetime"], errors="coerce")
    chunk = chunk.dropna(subset=["data_datetime"])
    keep = np.zeros(len(chunk), dtype=bool)
    for t0, t1 in windows:
        keep |= ((chunk["data_datetime"] >= t0) & (chunk["data_datetime"] <= t1)).values
    if keep.any():
        chunks.append(chunk.loc[keep])

df_raw = pd.concat(chunks, ignore_index=True)
for c in cols[1:]:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
df_raw = df_raw.set_index("data_datetime").sort_index()
print(f"linhas relevantes carregadas: {len(df_raw)}", flush=True)

vib_cols = ["TV_351X_A", "TV_351Y_A", "TV_352X_A", "TV_352Y_A", "TV_353X_A",
            "TV_353Y_A", "TV_354X_A", "TV_354Y_A", "TV_355X_A", "TV_355Y_A"]
pressure_cols = ["954005_624_PI_0319", "954005_624_PI_0315", "954005_624_PDI_0302"]

fig, axes = plt.subplots(len(top5), 3, figsize=(15, 3.2 * len(top5)))
for i, (_, ep) in enumerate(top5.iterrows()):
    t0, t1 = ep["start"] - pd.Timedelta(hours=3), ep["end"] + pd.Timedelta(hours=3)
    w = df_raw.loc[t0:t1]

    ax = axes[i, 0]
    ax.plot(w.index, w["TC382_03_A"], color="#184F95", label="TC382_03_A")
    ax2 = ax.twinx()
    ax2.plot(w.index, w["T5_AVG_A"], color="#EB6834", alpha=0.6, label="T5_AVG_A")
    ax.axvspan(ep["start"], ep["end"], color="red", alpha=0.15)
    ax.set_title(f"{ep['start']:%Y-%m-%d %H:%M} (z={ep['max_abs_z_overall']:.2f}) -- temp", fontsize=9)
    ax.tick_params(labelsize=7)

    ax = axes[i, 1]
    for c in vib_cols:
        ax.plot(w.index, w[c], linewidth=0.7, alpha=0.7)
    ax.axvspan(ep["start"], ep["end"], color="red", alpha=0.15)
    ax.set_title("vibração (10 canais)", fontsize=9)
    ax.tick_params(labelsize=7)

    ax = axes[i, 2]
    for c in pressure_cols:
        ax.plot(w.index, w[c], linewidth=1, label=c.replace("954005_624_", ""))
    ax.axvspan(ep["start"], ep["end"], color="red", alpha=0.15)
    ax.legend(fontsize=6, loc="upper right")
    ax.set_title("pressão (3 tags do portão)", fontsize=9)
    ax.tick_params(labelsize=7)

fig.tight_layout()
out_path = os.path.join(OUT, "residuais_dificeis_contexto.png")
fig.savefig(out_path, dpi=150)
print("\nfigura salva em", out_path)
