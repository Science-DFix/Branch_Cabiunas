#!/usr/bin/env python3
"""Sonda da J14: roda os passos do `tc33003a_exemplo.py` (lote) num processo novo e registra a memória por etapa.

    python j14_sonda_ram.py <csv> <pasta scripts do pacote> <pasta modelos> <registro_trips.csv> <saida.json>

Não importa o clearml. Mesmos passos e mesma ordem do exemplo (e17bf27): carregar_dados, preparar_grade uma vez,
carregar_modelos, diagnostico_entrada, monitor_drift, preprocessar, prever.
"""
import json
import resource
import sys
from pathlib import Path

csv, scripts, modelos, trips_csv, saida = sys.argv[1:6]
sys.path.insert(0, scripts)
import pandas as pd                     # noqa: E402
import cabiunas_inference as ci         # noqa: E402

R = {"etapas": []}


def marca(nome):
    rss = None
    for ln in open("/proc/self/status"):
        if ln.startswith("VmRSS:"):
            rss = int(ln.split()[1]) / 1024
    R["etapas"].append(dict(etapa=nome, rss_mb=round(rss, 1),
                            pico_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)))


def quadro(df):
    obj = [c for c in df.columns if df[c].dtype == object]
    exemplos = {}
    for c in obj[:5]:
        v = df[c].dropna()
        v = v[pd.to_numeric(v, errors="coerce").isna()]
        exemplos[c] = v.astype(str).value_counts().head(3).to_dict()
    return dict(linhas=len(df), colunas=df.shape[1],
                dtypes={str(k): int(v) for k, v in df.dtypes.value_counts().items()},
                mem_profunda_mb=round(df.memory_usage(deep=True).sum() / 1e6, 1),
                colunas_object=obj, texto_nas_object=exemplos)


marca("partida")
df = ci.carregar_dados(csv); marca("carregar_dados"); R["apos_carregar"] = quadro(df)
df = ci.preparar_grade(df); marca("preparar_grade"); R["apos_grade"] = quadro(df)
mods = ci.carregar_modelos(Path(modelos)); marca("carregar_modelos")
trips = ci.carregar_trips(Path(trips_csv))
diag = ci.diagnostico_entrada(mods, df); marca("diagnostico_entrada")
drift = ci.monitor_drift(mods, df); marca("monitor_drift")
proc = ci.preprocessar(mods, df, trips=trips); marca("preprocessar")
res = ci.prever(mods, proc); marca("prever")
Path(saida).write_text(json.dumps(R, ensure_ascii=False, default=str))
