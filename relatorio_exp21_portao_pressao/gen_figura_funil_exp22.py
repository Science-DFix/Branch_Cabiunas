"""Serie completa com 3 categorias, refletindo o funil INTEIRO ate o EXP22:
  verde  = alarme real detectado -- is_anom_point final=1 E perto de um
           alarme reconhecido (+-24h dos 2 tags avaliados OU +-8h dos 3
           tags de pressao do portao do EXP21)
  cinza  = FP eliminado -- estava marcado pelo modelo bruto (antes de
           qualquer portao) mas foi suprimido por algum portao (filtro de
           duracao/carga/volatilidade/mudanca de nivel)
  vermelho = FP restante -- continua marcado hoje (is_anom_point final=1)
           e nao esta perto de nenhum alarme reconhecido (os 222 residuais)

raw_flag (marcado pelo modelo bruto) = is_anom_point FINAL OR qualquer
coluna `*_blocked` == True -- essas colunas registram exatamente o que
cada portao suprimiu, entao a uniao reconstroi o que o OCSVM+limiar
marcou ANTES de qualquer portao.

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/relatorio_exp21_portao_pressao/gen_figura_funil_exp22.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from clearml import Dataset

OUT = os.path.dirname(os.path.abspath(__file__))
POINT_CSV = "/home/dvar/.clearml/cache/storage_manager/global/950100358f8e4a8dcba408279af95237.point_anomalies_all.csv"
WIN_2TAGS_MIN = 1440
WIN_PRESSAO_MIN = 480
PRESSURE_TAGS = ["PI_6240319_AL", "PAL_6240315", "PDAL_6240302"]
DATA_END = pd.Timestamp("2026-04-20 23:59:59")
BLOCKED_COLS = ["duration_filter_blocked", "load_gate_blocked", "volatility_gate_blocked", "step_change_gate_blocked"]

print("lendo point_anomalies_all.csv (EXP22)...", flush=True)
df_point = pd.read_csv(POINT_CSV, index_col=0, parse_dates=True)
df_point = df_point.loc[df_point.index <= DATA_END]

raw_flag = df_point["is_anom_point"].astype(bool).values.copy()
for c in BLOCKED_COLS:
    raw_flag |= df_point[c].astype(bool).values
final_flag = df_point["is_anom_point"].astype(bool).values
eliminated = raw_flag & ~final_flag
print(f"pontos: {len(df_point)}  raw_flag={raw_flag.sum()}  final(is_anom_point)={final_flag.sum()}  "
      f"eliminados_por_portoes={eliminated.sum()}", flush=True)

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
raw_path = os.path.join(root, "sensores_full_2024_2026_30s.csv")
print("lendo serie bruta do sensor alvo...", flush=True)
df_raw = pd.read_csv(raw_path, usecols=["data_datetime", "TC382_03_A"])
df_raw["data_datetime"] = pd.to_datetime(df_raw["data_datetime"], errors="coerce")
df_raw["TC382_03_A"] = pd.to_numeric(df_raw["TC382_03_A"], errors="coerce")
df_raw = df_raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
df_raw = df_raw.loc[(df_raw.index >= "2024-07-01") & (df_raw.index <= "2026-04-20 23:59:59")]

alarm_path = os.path.join(root, "alarmes_selecionados_turbina_a.csv")
alarm = pd.read_csv(alarm_path)
alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrência"], errors="coerce")
alarm["Tag"] = alarm["Tag Alarme"]
alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].copy()
alarm = alarm.dropna(subset=["Data da Ocorrencia"]).sort_values("Data da Ocorrencia").reset_index(drop=True)

sub2 = alarm[alarm["Tag"].isin(["TC382_03_A", "T5_AVG_A"])].sort_values("Data da Ocorrencia")
sub_pressao = alarm[alarm["Tag"].isin(PRESSURE_TAGS)].sort_values("Data da Ocorrencia")

alarm_times_2 = sub2["Data da Ocorrencia"].values.astype("datetime64[ns]")
alarm_times_p = sub_pressao["Data da Ocorrencia"].values.astype("datetime64[ns]")
win2 = np.timedelta64(WIN_2TAGS_MIN, "m")
winp = np.timedelta64(WIN_PRESSAO_MIN, "m")


def near_any(t_arr, alarm_times_sorted, win):
    if len(alarm_times_sorted) == 0 or len(t_arr) == 0:
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


OOS_START = pd.Timestamp("2025-07-01")
on_arr_final = (df_point.loc[final_flag, "operational_state"] == "on").values
oos_arr_final = (df_point.index[final_flag] >= OOS_START)

final_times = df_point.index[final_flag]
near2_final = near_any(final_times.values, alarm_times_2, win2)
nearp_final = near_any(final_times.values, alarm_times_p, winp)
near_official_final = near2_final | nearp_final

# vermelho (FP restante) so conta no dominio OFICIAL de avaliacao
# (operational_state=='on' E periodo OOS >= 2025-07-01), igual a
# compute_normal_alert_rate/df_point_eval_idx na pipeline real -- bate
# com o normal_alert_rate=0,043% / 222 pontos ja reportados. verde
# (contexto de alarme real) fica com o historico inteiro, so ilustrativo.
vermelho_mask = (~near_official_final) & on_arr_final & oos_arr_final
verde_mask = near_official_final
n_verde = verde_mask.sum()
n_vermelho = vermelho_mask.sum()
n_cinza = eliminated.sum()

print(f"\nverde (alarme real detectado): {n_verde}")
print(f"vermelho (FP restante): {n_vermelho}")
print(f"cinza (FP eliminado pelos portoes): {n_cinza}")

eliminated_times = df_point.index[eliminated]

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_raw.index, df_raw["TC382_03_A"], color="#184F95", linewidth=0.35, zorder=1, label="TC382_03_A (bruto)")

vals_cinza = df_raw["TC382_03_A"].reindex(eliminated_times.intersection(df_raw.index))
ax.scatter(vals_cinza.index, vals_cinza.values, color="#9B9B9B", s=7, zorder=2,
           label=f"FP eliminado pelos portões (n={n_cinza})")

vals_vermelho = df_raw["TC382_03_A"].reindex(final_times[vermelho_mask].intersection(df_raw.index))
ax.scatter(vals_vermelho.index, vals_vermelho.values, color="#D03B3B", s=10, zorder=4,
           label=f"FP restante (n={n_vermelho})")

vals_verde = df_raw["TC382_03_A"].reindex(final_times[verde_mask].intersection(df_raw.index))
ax.scatter(vals_verde.index, vals_verde.values, color="#0CA30C", s=10, zorder=5,
           label=f"alarme real detectado (n={n_verde})")

ax.set_xlim(df_raw.index.min(), df_raw.index.max())
ax.set_ylabel("Temperatura (°C)")
ax.set_title("TC382_03_A -- funil completo do EXP22: FP eliminado x FP restante x alarme real")
ax.legend(loc="upper left", fontsize=9, frameon=True)
fig.tight_layout()

out_path = os.path.join(OUT, "serie_completa_funil_exp22.png")
fig.savefig(out_path, dpi=180)
print("\nfigura salva em", out_path)

fig2, ax2 = plt.subplots(figsize=(6, 5))
cats = ["FP eliminado\n(pelos portões)", "FP restante\n(residual)", "Alarme real\ndetectado"]
counts = [n_cinza, n_vermelho, n_verde]
colors = ["#9B9B9B", "#D03B3B", "#0CA30C"]
bars = ax2.bar(cats, counts, color=colors)
for b, c in zip(bars, counts):
    ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + max(counts) * 0.01, str(c),
              ha="center", va="bottom", fontsize=10, fontweight="bold")
ax2.set_ylabel("nº de pontos (30s)")
ax2.set_title("EXP22 -- distribuição dos pontos marcados pelo modelo bruto")
fig2.tight_layout()
out_path2 = os.path.join(OUT, "barras_funil_exp22.png")
fig2.savefig(out_path2, dpi=180)
print("figura salva em", out_path2)
