#!/usr/bin/env python3
"""Retreino disparado pelo religamento apos parada longa: o baseline para de envelhecer?

De onde vem. limiares_robustos.py fechou que a instabilidade do detector e do
SINAL -- cada baseline mensal gera uma estrutura de alarme diferente. Uma fonte
concreta disso apareceu no nivel_c_canal_unico.py: depois da manutencao de 11
dias que se seguiu ao trip de 04/11/2025, o p ficou a 6-8x o limiar por OITO
DIAS, desde a primeira amostra apos o religamento, contra um PCA ajustado antes
do trip. O baseline mensal nao sabe que a maquina mudou de regime.

A regra. Alem do retreino mensal, depois de cada parada de pelo menos S horas,
assim que houver H horas de operacao ESTAVEL pos-religamento, o baseline e
refeito so com esses dados pos-religamento -- e o unico jeito de ele aprender o
regime novo. Ate la vale o modelo anterior; no corte mensal seguinte volta a
regra normal (os FIT pontos estaveis mais recentes, que ja incluem o pos).
O spread do mancal e refeito junto. vb nao muda (referencia rolante).

Protocolo -- o mesmo do limiares_robustos.py, para nao se enganar:
  SELECAO  cortes mensais nos dias 1, 8, 15, 22 -- (S, H) escolhido pelo pior.
  TESTE    cortes nos dias 4, 11, 18, 25 -- nunca vistos na escolha.

Risco conhecido: o baseline pos-religamento tem so H*30 pontos (720 a 2.160,
contra 20.000), entao o PCA e o p99 de normalizacao ficam mais ruidosos.

Uso (de dentro de scripts/pdm_fisico, com os dados):
    python retreino_pos_parada.py
    python retreino_pos_parada.py --clearml --remote
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
os.chdir(AQUI)
sys.path.insert(0, AQUI)

if {"--clearml", "--remote"} & set(sys.argv) or os.environ.get("CLEARML_TASK_ID"):
    import clearml  # noqa: F401  -- antes do argparse: o worker injeta os args por patch

import numpy as np
import pandas as pd

DIAS_SELECAO = [1, 8, 15, 22]
DIAS_TESTE = [4, 11, 18, 25]
S_H = [24, 72]            # parada minima que dispara o retreino
H_H = [24, 48, 72]        # horas estaveis pos-religamento antes de reajustar
FIT = 20_000
POR_H = 30
IMAGEM = "tensorflow/tensorflow:2.16.1-gpu"


def garante_dados(dataset_id):
    precisa = ["grade2min.parquet", "falhas.csv", "piso_fisico_cache.npz"]
    if all(os.path.exists(f) for f in precisa):
        return
    from clearml import Dataset
    D = Dataset.get(dataset_id=dataset_id).get_local_copy()
    for f in precisa:
        if not os.path.exists(f):
            os.symlink(os.path.join(D, f), f)


def religamentos(op: pd.Series, stable: pd.Series, S: float, H: float):
    """[(t0, r)]: t0 = primeira amostra operando apos parada >= S h; r = instante
    em que se acumulam H horas estaveis desde t0 SEM nova parada. Sem r, fica de fora."""
    idx = op.index
    o = op.to_numpy(); st = stable.to_numpy()
    grupo = np.cumsum(np.concatenate(([True], o[1:] != o[:-1])))
    out = []
    for gid in np.unique(grupo):
        pos = np.flatnonzero(grupo == gid)
        if o[pos[0]]:
            continue
        dur_h = (idx[pos[-1]] - idx[pos[0]]).total_seconds() / 3600 + 2 / 60
        if dur_h < S or pos[-1] + 1 >= len(o):
            continue
        i0 = pos[-1] + 1                                  # religamento
        prox = np.flatnonzero(grupo == gid + 1)           # a corrida operando seguinte
        cs = np.cumsum(st[prox])
        k = np.searchsorted(cs, int(H * POR_H))
        if k < len(prox):
            out.append((idx[i0], idx[prox[k]]))
    return out


def walkforward_evento(df, stable, op, dia, S, H):
    from cabiunas_pdm import config as C, detector as DET
    from robustez_retreino import scorer_max
    SM = scorer_max()
    idx = df.index
    base = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC")
    mensais = [c for c in (m + pd.Timedelta(days=dia - 1) for m in base) if idx[0] < c < idx[-1]]
    eventos = religamentos(op, stable, S, H) if S is not None else []
    marcos = sorted([(c, None) for c in mensais] + [(r, t0) for t0, r in eventos])
    n = len(idx)
    t = np.full(n, np.nan); p = np.full(n, np.nan); sp = np.full(n, np.nan)
    b_all = DET._spread_mancal(df).to_numpy().astype("float64")
    usados = 0
    for i, (c0, t0) in enumerate(marcos):
        c1 = marcos[i + 1][0] if i + 1 < len(marcos) else idx[-1] + pd.Timedelta("2min")
        s = (idx >= c0) & (idx < c1)
        if not s.any():
            continue
        if t0 is None:
            fit = df.loc[stable & (idx < c0), C.SENSOR_TAGS].dropna().tail(FIT)
            if len(fit) < FIT // 4:
                continue
        else:
            fit = df.loc[stable & (idx >= t0) & (idx < c0), C.SENSOR_TAGS].dropna()
            if len(fit) < int(0.8 * H * POR_H):
                continue
            usados += 1
        w = df.loc[s]
        t[s] = SM().fit(fit[C.TEMPERATURE_TAGS]).score(w[C.TEMPERATURE_TAGS])["pca_recon"].to_numpy()
        p[s] = SM().fit(fit[C.PRESSURE_TAGS]).score(w[C.PRESSURE_TAGS])["pca_recon"].to_numpy()
        b = DET._spread_mancal(fit)
        med = float(b.median()); mad = float((b - med).abs().median() * 1.4826)
        sp[s] = np.abs((b_all[s] - med) / mad)
    return {"t": t, "p": p, "sp": sp}, usados


def mede(args):
    """Uma configuracao numa realizacao. Roda em processo filho."""
    dia, S, H = args
    import publica_clearml as PC
    from cabiunas_pdm import config as C
    g = pd.read_parquet("grade2min.parquet")
    op = (g["RUNNING_A"] > 0.5).fillna(False)
    stable = (op & (g["T5_AVG_A"] > 300)).fillna(False)
    sin, usados = walkforward_evento(g[C.SENSOR_TAGS], stable, op, dia, S, H)
    al, mask, alvo, _, _, sel = PC.reproduz(v2=True, sinais=sin)
    res, _, _ = PC.metricas(al, mask, alvo, sel, permutacao=False)
    n = lambda s: int(s.split("/")[0])
    return dict(dia=dia, S=S if S is not None else "nenhum", H=H if H is not None else "-",
                retreinos_evento=usados, det=n(res["recall"]), inicio=n(res["recall_regua_inicio"]),
                banda=n(res["recall_banda_acionavel"]), fp_mes=res["fp_por_mes_regra_c"],
                h_fp=res["horas_fp_por_mes_regra_c"], carga=res["carga_h_por_mes"],
                lead_ini=res["lead_medio_inicio_h"])


def resumo(T):
    return (T.groupby(["S", "H"], sort=False)
             .agg(banda_pior=("banda", "min"), inicio_pior=("inicio", "min"), det_pior=("det", "min"),
                  fp_mes_pior=("fp_mes", "max"), carga_pior=("carga", "max"),
                  banda_med=("banda", "median"), inicio_med=("inicio", "median"),
                  fp_mes_med=("fp_mes", "median"), carga_med=("carga", "median"),
                  retreinos=("retreinos_evento", "median"))
             .reset_index())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-id", default="8b06a98f8b264820a9ecf2075a188395")
    ap.add_argument("--proc", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--clearml", action="store_true")
    ap.add_argument("--remote", action="store_true")
    ap.add_argument("--fila", default="default")
    ap.add_argument("--nome", default="geladeira::retreino_pos_parada")
    a = ap.parse_args()

    task = None
    if a.clearml or a.remote:
        from clearml import Task
        task = Task.init(project_name="TesteMLCab", task_name=a.nome, task_type=Task.TaskTypes.testing,
                         reuse_last_task_id=False, auto_connect_frameworks=False)
        task.set_base_docker(docker_image=IMAGEM)
        task.set_packages(["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "tzdata",
                           "matplotlib", "clearml"])
        task.add_tags(["geladeira", "retreino", "pos-parada"])
        if a.remote:
            task.execute_remotely(queue_name=a.fila, exit_process=True)

    garante_dados(a.dataset_id)
    c = mede((1, None, None))
    print("CONTROLE:", c, flush=True)
    assert (c["det"], c["inicio"], c["banda"], c["fp_mes"]) == (8, 6, 5, 0.344), \
        "controle nao reproduz o ponto publicado -- nada abaixo vale"

    from concurrent.futures import ProcessPoolExecutor
    confs = [(None, None)] + list(itertools.product(S_H, H_H))
    tarefas = [(d, S, H) for S, H in confs for d in DIAS_SELECAO + DIAS_TESTE]
    print(f"{len(tarefas)} execucoes em {a.proc} processos", flush=True)
    with ProcessPoolExecutor(max_workers=a.proc) as ex:
        T = pd.DataFrame(list(ex.map(mede, tarefas)))
    T.to_csv("retreino_pos_parada.csv", index=False)

    sel = resumo(T[T.dia.isin(DIAS_SELECAO)]).sort_values(
        ["banda_pior", "inicio_pior", "det_pior", "fp_mes_pior", "carga_pior"],
        ascending=[False, False, False, True, True])
    esc = sel[sel.S != "nenhum"].iloc[0]
    te = resumo(T[T.dia.isin(DIAS_TESTE)])
    te = te[(te.S == "nenhum") | ((te.S == esc.S) & (te.H == esc.H))]

    pd.set_option("display.width", 220)
    print("\n=== SELECAO (dias 1/8/15/22), ordenada pelo pior caso ===\n" + sel.to_string(index=False))
    print(f"\nESCOLHIDO: S={esc.S} h, H={esc.H} h")
    print("\n=== TESTE (dias 4/11/18/25, nunca vistos) ===\n" + te.to_string(index=False))
    print("\n=== por dia ===\n" + T.to_string(index=False))

    if task is not None:
        lg = task.get_logger()
        lg.report_table("selecao (pior caso)", "", iteration=0, table_plot=sel)
        lg.report_table("teste (nunca vistos)", "", iteration=0, table_plot=te)
        lg.report_table("por dia", "", iteration=0, table_plot=T)
        task.upload_artifact("retreino_pos_parada", artifact_object="retreino_pos_parada.csv")
        task.flush(wait_for_uploads=True)


if __name__ == "__main__":
    main()
