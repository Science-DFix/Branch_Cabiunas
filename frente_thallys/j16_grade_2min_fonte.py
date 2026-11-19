#!/usr/bin/env python3
"""J16 -- a grade de 2 min que alimenta a pipeline do Thallys é a mediana do export de 30 s?

PRÉ-REGISTRO (18/11/2026, commitado antes de enviar)
Origem: a [PERGUNTA] do documento à engenharia (o histórico que o worker lê é de 30 s ou de 2 min já montada pela
  mediana?). No Cica, a pipeline do Thallys (worker_verifica_harness, worker_exp63/64 da d77de7d) lê o
  "CSV de produção 2 min" (Dataset bb85d853, data_2025-01-01_2026-04-30_raw.csv, 134,7 MB) mais o complemento de
  30 s de 2024 (c9afd177). O documento diz "não usar uma grade de 2 min pronta de outra fonte"; os nossos J1-J15
  montaram a grade do export de 30 s (a97ba56b) com `preparar_grade`. Se a grade do Thallys for essa mesma mediana,
  a resposta é "2 min montada pela mediana" e os números do documento valem como estão.
Desenho: pacote e17bf27 (insumos f5e1b16b), pandas 2.3.3. A = preparar_grade(carregar_dados(bb85d853));
  B = preparar_grade(carregar_dados(export de 30 s desde 2025-01-01, montado como no J1)). Compara no índice comum
  as 38 colunas: célula igual se |Δ| <= 1e-9 ou NaN nos dois. Para diagnóstico, também contra a leitura de 30 s
  mais próxima do início da janela (asfreq) e contra a média da janela.
EXPECTATIVA: A = B em >= 99,9% das células (~70%; o sinal reproduzido do 30 s bateu t, p e sp da ref_sinais).
CRITÉRIO: >= 99,9% iguais -> a grade da pipeline é a mediana do 30 s; a [PERGUNTA] fecha como "linha de 2 min
  montada pela mediana" (números do documento valem). Se não, identificar a construção (asfreq/média) e reabrir a
  frase "não usar grade de 2 min pronta" com o Thallys antes de fechar o documento.

RESULTADO (task 003a69f8, Cica, pandas 2.3.3). 349.200 linhas de 2 min × 38 colunas, mesmo índice nas duas:
    mediana ........ 16,4% das células iguais a 1e-9; |Δ| máx 3,05e-5; NaN idênticos (0 só de um lado).
    asfreq (início)  13,1%; |Δ| máx 1.071; 13.010 NaN só na grade do Thallys.
    média .......... 14,1%; |Δ| máx 759; 13.010 NaN só na grade do Thallys.
PLACAR: pelo critério pré-registrado (|Δ| <= 1e-9 em >= 99,9%), NÃO. A POSTERIORI: 3,05e-5 = 2^-15 = 1 passo de
  float32 na faixa 256-512 (o `preparar_grade` devolve float32; o CSV do Thallys foi gravado em texto com menos
  dígitos), e o padrão de NaN coincide; as construções alternativas erram em centenas. A grade da pipeline é a
  mediana do 30 s até o arredondamento do texto. A jusante isto já estava medido: o controle refeito do 30 s (25/10)
  deu os mesmos 7/8 e 9 FP publicados a partir deste CSV.
LEITURA: a [PERGUNTA] fecha como "linha de 2 min montada pela mediana"; os números do documento valem. A frase "não
  usar grade de 2 min pronta de outra fonte" continua certa e não atinge o bb85d853. Ressalva: é a pipeline do Cica,
  tomada como a do worker por decisão do usuário (18/11).
  _(registrado em 18/11/2026 23:55)_

Uso:
  python frente_thallys/j16_grade_2min_fonte.py --enqueue
"""
from clearml import Task  # noqa: F401  (antes do argparse: o worker injeta argumentos por patch)
import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import j1_janela_entrada as j1      # noqa: E402

NOME = "frente_thallys | J16 grade de 2 min da pipeline = mediana do 30 s?"
DS_2MIN = "bb85d853f92545beb771d303354302ab"
SAIDA = Path(tempfile.gettempdir()) / "j16_saida"


def compara(a, b):
    import numpy as np
    idx = a.index.intersection(b.index); cols = [c for c in a.columns if c in b.columns]
    x = a.loc[idx, cols].astype("float64").to_numpy(); y = b.loc[idx, cols].astype("float64").to_numpy()
    nx, ny = np.isnan(x), np.isnan(y)
    igual = (nx & ny) | (~nx & ~ny & (np.abs(x - y) <= 1e-9))
    por_col = {c: round(float(igual[:, j].mean()), 6) for j, c in enumerate(cols)}
    d = np.where(~nx & ~ny, np.abs(x - y), np.nan)
    return dict(linhas_comuns=len(idx), so_em_a=int(len(a.index.difference(b.index))),
                so_em_b=int(len(b.index.difference(a.index))), colunas=len(cols),
                so_cols_a=[c for c in a.columns if c not in b.columns], so_cols_b=[c for c in b.columns if c not in a.columns],
                frac_celulas_iguais=round(float(igual.mean()), 6), nan_so_a=int((nx & ~ny).sum()),
                nan_so_b=int((~nx & ny).sum()), max_abs_dif=float(np.nanmax(d)) if np.isfinite(d).any() else 0.0,
                pior_colunas=sorted(por_col.items(), key=lambda kv: kv[1])[:8])


def executa(local: bool, insumos: str | None):
    import pandas as pd
    SAIDA.mkdir(parents=True, exist_ok=True)
    pk, _, _ = j1.monta(local, insumos)
    import cabiunas_inference as ci
    if local:
        csv2 = Path(os.environ["FT_2MIN"])
    else:
        from clearml import Dataset
        csv2 = Path(Dataset.get(dataset_id=DS_2MIN).get_local_copy()) / "data_2025-01-01_2026-04-30_raw.csv"
    csv30 = next((pk / "dados" / "2025_2026").glob("data_*_raw.csv"))
    res = dict(ambiente=j1.ambiente(), pandas=pd.__version__)
    with contextlib.redirect_stdout(io.StringIO()):
        r2 = ci.carregar_dados(csv2); r30 = ci.carregar_dados(csv30)
        A = ci.preparar_grade(r2); B = ci.preparar_grade(r30)
    passo2 = r2.index.to_series().diff().median()
    res["entrada_2min"] = dict(linhas=len(r2), colunas=len(r2.columns), passo=str(passo2),
                               ini=str(r2.index[0]), fim=str(r2.index[-1]))
    res["entrada_30s"] = dict(linhas=len(r30), colunas=len(r30.columns), ini=str(r30.index[0]), fim=str(r30.index[-1]))
    res["mediana"] = compara(A, B)
    # diagnóstico: outras construções da grade a partir do 30 s
    X = r30.apply(pd.to_numeric, errors="coerce")
    res["asfreq_inicio"] = compara(A, X.resample(ci.GRID).first())
    res["media"] = compara(A, X.resample(ci.GRID).mean())
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str), flush=True)
    (SAIDA / "j16_resultado.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    from clearml import Task as Tk
    tk = Tk.current_task()
    if tk is not None:
        tk.upload_artifact("j16_resultado", artifact_object=str(SAIDA / "j16_resultado.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enqueue", action="store_true"); ap.add_argument("--local", action="store_true")
    ap.add_argument("--insumos", default=j1.DS_INSUMOS)
    a = ap.parse_args()
    if a.enqueue:
        t = Task.init(project_name="TesteMLCab", task_name=NOME)
        t.set_base_docker("python:3.11-slim", docker_setup_bash_script=["apt-get update -qq", "apt-get install -y -qq libgomp1"])
        t.set_packages(j1.PACOTES); t.set_parameters({"General/insumos": a.insumos})
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=AQUI, text=True).strip()
        t.set_script(repository="git@github.com:Science-DFix/Branch_Cabiunas.git", branch="exp/ocsvm-validacao", commit=commit)
        t.execute_remotely(queue_name="default")
    if os.environ.get("CLEARML_TASK_ID"):
        tk = Task.init(project_name="TesteMLCab", task_name=NOME)
        a.insumos = tk.get_parameter("General/insumos") or a.insumos
    executa(a.local, a.insumos)


if __name__ == "__main__":
    main()
