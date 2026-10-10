"""Serie completa com classificacao de 3 cores, refletindo EXATAMENTE o
que o portao de pressao do EXP21 reconhece:
  verde   = perto (+-24h) de um dos 2 sensores avaliados (TC382_03_A/T5_AVG_A)
  laranja = nao perto dos 2 avaliados, mas perto (+-8h) de um dos 3 tags
            de pressao do portao (PI_6240319_AL/PAL_6240315/PDAL_6240302)
            -- "explicado pelo portao de pressao do EXP21"
  vermelho = isolado mesmo depois do portao -- FP genuino residual

is_anom_point e IDENTICO entre EXP20 e EXP21 (o portao de pressao so
muda a avaliacao/normal_alert_rate, nao o que e detectado) -- reusa o
point_anomalies_all.csv ja baixado do EXP20.

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/relatorio_exp21_portao_pressao/gen_figura_3cores.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from clearml import Dataset

OUT = os.path.dirname(os.path.abspath(__file__))
POINT_CSV = "/home/dvar/.clearml/cache/storage_manager/global/18605388c830fc41b0cfa3439a4d5249.point_anomalies_all.csv"
WIN_2TAGS_MIN = 1440
WIN_PRESSAO_MIN = 480  # 8h, igual ao EXTRA_NEAR_ALARM_WINDOW_MINUTES do EXP21
PRESSURE_TAGS = ["PI_6240319_AL", "PAL_6240315", "PDAL_6240302"]

print("lendo point_anomalies_all.csv...", flush=True)
df_point = pd.read_csv(POINT_CSV, index_col=0, parse_dates=True)

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
raw_path = os.path.join(root, "sensores_full_2024_2026_30s.csv")
print("lendo serie bruta do sensor alvo...", flush=True)
df_raw = pd.read_csv(raw_path, usecols=["data_datetime", "TC382_03_A"])
df_raw["data_datetime"] = pd.to_datetime(df_raw["data_datetime"], errors="coerce")
df_raw["TC382_03_A"] = pd.to_numeric(df_raw["TC382_03_A"], errors="coerce")
df_raw = df_raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
df_raw = df_raw.loc[(df_raw.index >= "2024-07-01") & (df_raw.index <= "2026-04-30 23:59:59")]

alarm_path = os.path.join(root, "alarmes_selecionados_turbina_a.csv")
alarm = pd.read_csv(alarm_path)
alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrência"], errors="coerce")
alarm["Tag"] = alarm["Tag Alarme"]
alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].copy()
alarm = alarm.dropna(subset=["Data da Ocorrencia"]).sort_values("Data da Ocorrencia").reset_index(drop=True)

sub2 = alarm[alarm["Tag"].isin(["TC382_03_A", "T5_AVG_A"])].sort_values("Data da Ocorrencia")
sub_pressao = alarm[alarm["Tag"].isin(PRESSURE_TAGS)].sort_values("Data da Ocorrencia")
print(f"alarmes dos 2 sensores avaliados: {len(sub2)}  |  alarmes de pressao (3 tags): {len(sub_pressao)}", flush=True)

anom_times = df_point.index[df_point["is_anom_point"] == 1]
print(f"anomalias totais: {len(anom_times)}", flush=True)

alarm_times_2 = sub2["Data da Ocorrencia"].values.astype("datetime64[ns]")
alarm_times_p = sub_pressao["Data da Ocorrencia"].values.astype("datetime64[ns]")
win2 = np.timedelta64(WIN_2TAGS_MIN, "m")
winp = np.timedelta64(WIN_PRESSAO_MIN, "m")


def near_any(t_arr, alarm_times_sorted, win):
    if len(alarm_times_sorted) == 0:
        return np.zeros(len(t_arr), dtype=bool)
    pos = np.searchsorted(alarm_times_sorted, t_arr)
    out = np.zeros(len(t_arr), dtype=bool)
    for i in range(len(t_arr)):
        p = pos[i]
        cands = []
        if p > 0:
            cands.append(alarm_times_sorted[p - 1])
        if p < len(alarm_times_sorted):
            cands.append(alarm_times_sorted[p])
        out[i] = any(abs(t_arr[i] - c) <= win for c in cands)
    return out


near2_flags = near_any(anom_times.values, alarm_times_2, win2)
nearp_flags = near_any(anom_times.values, alarm_times_p, winp)

cat = np.where(near2_flags, "verde", np.where(nearp_flags, "laranja", "vermelho"))
n_verde = (cat == "verde").sum()
n_laranja = (cat == "laranja").sum()
n_vermelho = (cat == "vermelho").sum()
print(f"\nverde (perto dos 2 avaliados): {n_verde} ({n_verde/len(cat)*100:.1f}%)")
print(f"laranja (explicado pelo portao de pressao, +-8h): {n_laranja} ({n_laranja/len(cat)*100:.1f}%)")
print(f"vermelho (isolado mesmo com o portao): {n_vermelho} ({n_vermelho/len(cat)*100:.1f}%)")
print(f"\n(para comparacao: antes do portao de pressao, verde+isolado era {n_verde}+{n_laranja+n_vermelho} "
      f"-- isolado caiu de {n_laranja+n_vermelho} para {n_vermelho})")

fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax = axes[0]
ax.plot(df_raw.index, df_raw["TC382_03_A"], color="#184F95", linewidth=0.35, zorder=1, label="TC382_03_A (bruto)")

for color, label, mask in [
    ("#D03B3B", f"isolado (FP genuíno, n={n_vermelho})", cat == "vermelho"),
    ("#EB9C34", f"explicado por pressão ±8h (n={n_laranja})", cat == "laranja"),
    ("#0CA30C", f"perto de alarme real, ±24h (n={n_verde})", cat == "verde"),
]:
    idx = anom_times[mask]
    vals = df_raw["TC382_03_A"].reindex(idx.intersection(df_raw.index))
    ax.scatter(vals.index, vals.values, color=color, s=9, zorder=4, label=label)

data_start, data_end = df_raw.index.min(), df_raw.index.max()
ax.set_xlim(data_start, data_end)
ax.set_ylabel("Temperatura (°C)")
ax.set_title("TC382_03_A -- classificação com o portão de pressão do EXP21 (±8h, 3 tags)")
ax.legend(loc="upper left", fontsize=8, frameon=True)

ax2 = axes[1]
ax2.hist(anom_times[cat == "vermelho"], bins=100, color="#D03B3B", label="isolado (residual)")
ax2.set_ylabel("nº pontos\n(isolado)")
ax2.set_xlabel("tempo")
ax2.set_xlim(data_start, data_end)
ax2.legend(loc="upper left", fontsize=8, frameon=False)

fig.tight_layout()
out_path = os.path.join(OUT, "serie_completa_3cores_exp21.png")
fig.savefig(out_path, dpi=180)
print("\nfigura salva em", out_path)
