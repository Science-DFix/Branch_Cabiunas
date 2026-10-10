"""Checa quantos dos 42 FP pos-filtro tem um alarme de processo
DIFERENTE dos 5 ja usados no canal 4 (para nao ser circular) ocorrendo
perto (+-2h) -- com CONTROLE NEGATIVO obrigatorio (mesma disciplina
ja aplicada nesta investigacao para o enriquecimento por catalogo):
compara contra a taxa em instantes aleatorios do periodo monitorado.

Uso:
    PYTHONPATH=. python3 dataset_francisco_lara/checa_alarmes_proximos_42fp.py
"""
import os

import pandas as pd
from clearml import Dataset

from src.cnn1d_ae.scoring import annotate_alert_catalog_context, compute_catalog_enrichment_control

OUT = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(OUT, "..", "runs_pipeline_unificada_final")
WINDOW_HOURS = 2.0
CANAL4_TAGS = ["PI_6240319_AL", "PAL_6240315", "PDAL_6240302", "TC382_05_A", "PAH_6240319"]

cls = pd.read_csv(os.path.join(RUN_DIR, "episodios_classificados.csv"), parse_dates=["start", "end", "falha_associada"])
fp = cls[cls["classe"] == "falso_positivo"].sort_values("start").reset_index(drop=True)
print(f"{len(fp)} episodios de falso positivo", flush=True)

df_final = pd.read_csv(os.path.join(RUN_DIR, "point_anomalies_final.csv"), index_col=0, parse_dates=True)
op_state = df_final["operational_state"]

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
alarm = pd.read_csv(os.path.join(root, "alarmes_selecionados_turbina_a.csv"))
alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrência"], errors="coerce")
alarm["Tag"] = alarm["Tag Alarme"]
alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].dropna(subset=["Data da Ocorrencia"])
print(f"catálogo total ACT: {len(alarm)} linhas, {alarm['Tag'].nunique()} tags", flush=True)

outros = alarm[~alarm["Tag"].isin(CANAL4_TAGS)].copy()
print(f"catálogo 'outros alarmes' (exclui os 5 do canal 4): {len(outros)} linhas, {outros['Tag'].nunique()} tags", flush=True)

# df_point sintetico: is_anom_point=1 exatamente no INICIO de cada um dos 42 FP
idx_completo = df_final.index
is_anom = pd.Series(0, index=idx_completo)
starts_validos = [df_final.index.asof(t) for t in fp["start"]]
is_anom.loc[starts_validos] = 1

df_point = pd.DataFrame({"is_anom_point": is_anom, "operational_state": op_state})
df_point = annotate_alert_catalog_context(df_point, outros, window_hours=WINDOW_HOURS)

# tabela por episodio
fp_anotado = fp.copy()
fp_anotado["inicio_valido"] = starts_validos
fp_anotado["explicado_por_outro_alarme"] = df_point.loc[starts_validos, "alert_confidence"].values == "explicado_catalogo"
fp_anotado["tag_alarme_proximo"] = df_point.loc[starts_validos, "alert_catalog_tag"].values
fp_anotado["distancia_h"] = df_point.loc[starts_validos, "alert_catalog_distance_h"].values

print("\n=== 42 FP -- explicados por outro alarme (não os 5 do canal 4), janela ±2h ===")
print(fp_anotado[["start", "explicado_por_outro_alarme", "tag_alarme_proximo", "distancia_h"]].to_string(index=False))

taxa_fp = fp_anotado["explicado_por_outro_alarme"].mean()
print(f"\ntaxa entre os 42 FP: {100*taxa_fp:.1f}%")

# controle negativo obrigatorio
control = compute_catalog_enrichment_control(df_point, outros, window_hours=WINDOW_HOURS, n_samples=5000, random_seed=42)
print("\n=== CONTROLE NEGATIVO (baseline aleatório, mesmo catálogo 'outros', mesma janela) ===")
for k, v in control.items():
    print(f"  {k}: {v}")

fp_anotado.to_csv(os.path.join(OUT, "alarmes_proximos_42fp.csv"), index=False)
print(f"\nsalvo em {os.path.join(OUT, 'alarmes_proximos_42fp.csv')}")
