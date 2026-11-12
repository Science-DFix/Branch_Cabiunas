#!/usr/bin/env python3
"""Comando de UM passo do motor com estado, para o medidor de telemetria do Thallys (J11).

Faz o que a execução de produção do motor faria: lê o CSV de 3 d do export de 30 s, monta a grade (`preparar_grade`),
carrega os bundles, relê o estado (tabela de 45 d + buffer do vb) e roda um passo. Mesmo papel do
`tc33003a_exemplo.py` no lote.

    python j11_motor_cmd.py <pasta com bruto3.csv, tab.parquet, buf.parquet> <pasta scripts do pacote> <pasta modelos>
"""
import sys
import types
from pathlib import Path

# o motor vive no j34, que importa o clearml para enviar tasks; aqui ele não é usado e não deve pesar na medida
_stub = types.ModuleType("clearml"); _stub.Task = None; _stub.Dataset = None
sys.modules["clearml"] = _stub

pasta, scripts, modelos = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
sys.path[:0] = [str(Path(__file__).resolve().parent), scripts]
import pandas as pd                     # noqa: E402
import cabiunas_inference as ci         # noqa: E402
import j34_estado_robustez as J         # noqa: E402

df = ci.preparar_grade(ci.carregar_dados(pasta / "bruto3.csv"))
J.G.update(ci=ci, modelos=ci.carregar_modelos(modelos))
J.G["det"] = J.G["modelos"][-1]["detector"]; J.G["orig_z"] = ci._z_vibracao; ci._z_vibracao = J._z_com_buffer
trips = sorted(ci.carregar_trips(Path(modelos).parent / "registro_trips.csv"))
m = J.Motor()
m.tab = pd.read_parquet(pasta / "tab.parquet").iloc[:-1]
m.buf = pd.read_parquet(pasta / "buf.parquet")
m.ultimo = m.tab.index[-1]
out = m.passo(df, df.index[-1], trips)
print(f"motor: {0 if out is None else len(out)} linha(s) nova(s)")
