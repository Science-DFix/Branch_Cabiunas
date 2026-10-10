"""Valida OFFLINE (sem re-treinar nada) a ideia da supressao cirurgica
baseada em mecanismo: um episodio residual curto e "suspeito" se, logo
em seguida (dentro de LOOKAHEAD_MINUTES), algum portao (*_blocked)
liga e fica ligado por pelo menos SUSTAIN_MINUTES -- assinatura de
borda-de-entrada de portao causal, nao ruido nem precursor real.

Critico: verifica que NENHUM dos 12 episodios "verde" (acerto de TRIP
curado) seria afetado -- se afetar, a ideia e descartada antes de
tocar em codigo de producao.

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/dataset_francisco_lara/valida_supressao_cirurgica.py
"""
import pandas as pd
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))
POINT_CSV = "/home/dvar/.clearml/cache/storage_manager/global/c27dfd135eb394c5511ce610bd890c1e.point_anomalies_all.csv"
BLOCKED_COLS = ["load_gate_blocked", "volatility_gate_blocked", "step_change_gate_blocked"]
LOOKAHEAD_MINUTES = 10.0
SUSTAIN_MINUTES = 3.0

print("lendo point_anomalies_all.csv (EXP28)...", flush=True)
df = pd.read_csv(POINT_CSV, index_col=0, parse_dates=True)
df = df.sort_index()

dt_seconds = df.index.to_series().diff().dt.total_seconds().median()
lookahead_samples = max(1, int(round((LOOKAHEAD_MINUTES * 60.0) / dt_seconds)))
sustain_samples = max(1, int(round((SUSTAIN_MINUTES * 60.0) / dt_seconds)))
print(f"grid={dt_seconds}s lookahead={lookahead_samples} amostras sustain={sustain_samples} amostras", flush=True)

flags = df["is_anom_point"].values.astype(bool)
m = flags.astype(np.int8)
d = np.diff(np.concatenate(([0], m, [0])))
starts = np.where(d == 1)[0]
ends = np.where(d == -1)[0]  # exclusive
print(f"episodios residuais (pos-todos-os-portoes): {len(starts)}", flush=True)

any_blocked = np.zeros(len(df), dtype=bool)
for c in BLOCKED_COLS:
    any_blocked |= df[c].fillna(False).values.astype(bool)

n = len(df)
flagged = np.zeros(len(starts), dtype=bool)
for i, (s, e) in enumerate(zip(starts, ends)):
    win_end = min(n, e + lookahead_samples)
    seg = any_blocked[e:win_end]
    if len(seg) < sustain_samples:
        continue
    # procura primeira transicao False->True dentro da janela que fique
    # sustentada por >= sustain_samples consecutivos
    for j in range(len(seg) - sustain_samples + 1):
        if seg[j:j + sustain_samples].all():
            flagged[i] = True
            break

episode_start_times = df.index[starts]
episode_end_times = df.index[ends - 1]
episode_len_min = (ends - starts) * dt_seconds / 60.0

res = pd.DataFrame({
    "inicio": episode_start_times, "fim": episode_end_times,
    "duracao_min": episode_len_min, "n_pontos": ends - starts,
    "flagged_supressao": flagged,
})
print(f"\nepisodios flagados pela regra (candidatos a supressao): {flagged.sum()} de {len(starts)}", flush=True)
print(f"duracao dos flagados: media={res.loc[flagged,'duracao_min'].mean():.2f}min "
      f"max={res.loc[flagged,'duracao_min'].max():.2f}min total={res.loc[flagged,'duracao_min'].sum()/60:.2f}h", flush=True)

# cruza com verde/amarelo ja classificados
epi = pd.read_csv(os.path.join(OUT, "episodios_verde_amarelo_exp28.csv"), parse_dates=["inicio", "fim"])
print("\ncolunas do episodios_verde_amarelo_exp28.csv:", epi.columns.tolist())

# junta por proximidade de inicio (tolerancia = 1 amostra)
res = res.sort_values("inicio").reset_index(drop=True)
epi = epi.sort_values("inicio").reset_index(drop=True)
merged = pd.merge_asof(epi, res[["inicio", "flagged_supressao", "duracao_min"]].rename(columns={"duracao_min": "duracao_min_chk"}),
                        on="inicio", direction="nearest", tolerance=pd.Timedelta(seconds=dt_seconds * 2))

print("\n--- cruzamento com classificacao verde/amarelo ---")
print(merged.groupby(["classe", "flagged_supressao"]).size() if "classe" in merged.columns else merged.columns.tolist())

out_path = os.path.join(OUT, "validacao_supressao_cirurgica.csv")
res.to_csv(out_path, index=False)
merged.to_csv(os.path.join(OUT, "validacao_supressao_cirurgica_merged.csv"), index=False)
print(f"\nsalvo em {out_path}")
