import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

OUT = "/home/dvar/REPO_CABIUNAS/cabiunas-models/relatorio_exp20_filtro_duracao"
df = pd.read_csv(f"{OUT}/episodios_pos_filtro_com_classificacao.csv")

iso = df.loc[~df["near_alarm"], "duration_min"]
near = df.loc[df["near_alarm"], "duration_min"]

fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.arange(4.5, 60, 2.5)
ax.hist(iso.clip(upper=59), bins=bins, color="#D03B3B", alpha=0.75, label=f"isolado (FP disperso, n={len(iso)})")
ax.hist(near.clip(upper=59), bins=bins, color="#0CA30C", alpha=0.55, label=f"perto de alarme real (n={len(near)})")
ax.axvline(4.5, color="black", linestyle="--", linewidth=1, label="corte do filtro (4,5min)")
ax.set_xlabel("duração do episódio pós-filtro (min) -- truncado em 59min p/ visualização")
ax.set_ylabel("nº de episódios")
ax.set_title("Duração dos episódios residuais após o filtro de 4,5min")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(f"{OUT}/hist_duracao_fp_isolado.png", dpi=180)
print("salvo")
