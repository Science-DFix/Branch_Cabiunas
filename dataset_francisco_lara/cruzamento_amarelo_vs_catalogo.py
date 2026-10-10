"""Cruza os 202 episodios "amarelo" (isolados dos 9 TRIPs curados do
Francisco) contra o CATALOGO COMPLETO de 47 tags de alarme -- pra
separar FP de verdade de eventos reais que so nao estao na lista
curada dele (mesma logica que ja usamos pro EXP22 residual, agora
aplicada aos episodios do EXP28).

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/dataset_francisco_lara/cruzamento_amarelo_vs_catalogo.py
"""
import pandas as pd
import numpy as np
import os
from clearml import Dataset

OUT = os.path.dirname(os.path.abspath(__file__))
WIN_CATALOGO_H = 24

print("lendo episodios_verde_amarelo_exp28.csv...", flush=True)
eps = pd.read_csv(os.path.join(OUT, "episodios_verde_amarelo_exp28.csv"))
eps["inicio"] = pd.to_datetime(eps["inicio"])
eps["fim"] = pd.to_datetime(eps["fim"])
eps["centro"] = eps["inicio"] + (eps["fim"] - eps["inicio"]) / 2
amarelos = eps.loc[eps["classe"] == "amarelo"].copy()
print(f"episodios amarelos: {len(amarelos)}", flush=True)

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
alarm = pd.read_csv(os.path.join(root, "alarmes_selecionados_turbina_a.csv"))
alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrência"], errors="coerce")
alarm["Tag"] = alarm["Tag Alarme"]
alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].copy()
alarm = alarm.dropna(subset=["Data da Ocorrencia"]).sort_values("Data da Ocorrencia").reset_index(drop=True)
print(f"catalogo completo: {len(alarm)} eventos, {alarm['Tag'].nunique()} tags", flush=True)

alarm_times_sorted = alarm["Data da Ocorrencia"].values.astype("datetime64[ns]")
win = np.timedelta64(WIN_CATALOGO_H, "h")

explicado = []
tag_mais_proxima = []
dist_h = []
for _, row in amarelos.iterrows():
    # usa o INICIO do episodio (mais correto pra "o que aconteceu perto
    # do disparo"), nao o centro -- evita diluir episodios longos
    t = np.datetime64(row["inicio"])
    p = np.searchsorted(alarm_times_sorted, t)
    cands_idx = []
    if p > 0:
        cands_idx.append(p - 1)
    if p < len(alarm_times_sorted):
        cands_idx.append(p)
    best_d, best_tag = None, None
    for ci in cands_idx:
        d = abs(t - alarm_times_sorted[ci])
        if best_d is None or d < best_d:
            best_d, best_tag = d, alarm.loc[ci, "Tag"]
    dh = best_d / np.timedelta64(1, "h") if best_d is not None else np.inf
    dist_h.append(dh)
    tag_mais_proxima.append(best_tag)
    explicado.append(dh <= WIN_CATALOGO_H)

amarelos["explicado_catalogo"] = explicado
amarelos["tag_mais_proxima"] = tag_mais_proxima
amarelos["dist_h"] = dist_h

n_explicado = sum(explicado)
n_isolado = len(amarelos) - n_explicado
print(f"\nexplicados pelo catalogo completo (+-{WIN_CATALOGO_H}h): {n_explicado} ({n_explicado/len(amarelos)*100:.1f}%)")
print(f"genuinamente isolados: {n_isolado} ({n_isolado/len(amarelos)*100:.1f}%)")

h_explicado = amarelos.loc[amarelos["explicado_catalogo"], "duracao_h"].sum()
h_isolado = amarelos.loc[~amarelos["explicado_catalogo"], "duracao_h"].sum()
print(f"\nhoras explicadas: {h_explicado:.1f}h")
print(f"horas genuinamente isoladas: {h_isolado:.1f}h")
n_meses = 21.9
print(f"FP genuino: {h_isolado/n_meses:.2f} h/mes (era 1,86 h/mes antes do cruzamento)")

print("\ntags mais proximas dos que ficam isolados:")
print(amarelos.loc[~amarelos["explicado_catalogo"], "tag_mais_proxima"].value_counts().head(15).to_string())

print("\ntags que mais explicam os amarelos:")
print(amarelos.loc[amarelos["explicado_catalogo"], "tag_mais_proxima"].value_counts().head(15).to_string())

amarelos.to_csv(os.path.join(OUT, "amarelos_vs_catalogo_completo.csv"), index=False)
print(f"\nsalvo em {os.path.join(OUT, 'amarelos_vs_catalogo_completo.csv')}")

print("\nepisodios genuinamente isolados (detalhe):")
iso = amarelos.loc[~amarelos["explicado_catalogo"], ["inicio", "fim", "n_pontos", "duracao_h"]]
print(iso.sort_values("duracao_h", ascending=False).to_string())
