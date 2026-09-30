#!/usr/bin/env python3
"""Limiares escolhidos pelo PIOR retreino, nao pelo dia 1 -- e validados em retreinos nunca vistos.

De onde vem. robustez_retreino.py mostrou que o dia 1 e o unico corte IN-SAMPLE:
todos os limiares do v2 foram escolhidos sobre os sinais do retreino no dia 1, e
em qualquer outro corte -- outra realizacao do mesmo sinal, o que producao vai
encontrar -- o detector cai para banda 3-5/8 e 0,69-0,95 FP/mes. A mediana dos
quatro cortes (0,82 FP/mes, carga 117 h/mes) e mais honesta que o publicado.

O remedio direto: escolher o ponto de operacao pelo PIOR caso entre varias
realizacoes, em vez de pelo melhor numero numa so. E minimax, como o
ponto_de_deploy.py -- mas entre realizacoes do sinal, nao entre vizinhos de
parametro.

Protocolo, para nao trocar um sobreajuste por outro:
  SELECAO  cortes nos dias 1, 8, 15, 22 -- o ponto e escolhido pelo pior deles.
  TESTE    cortes nos dias 4, 11, 18, 25 -- realizacoes que nao participam da
           escolha. O publicado e o robusto sao medidos nelas.

Grade (fatores sobre os limiares do v2, preservando a estrutura de dois niveis):
  f_tp  -- canais t e p nos dois niveis (os que dependem do retreino)
  f_lo  -- nivel A inteiro (sensivel, >= 3 de 4)
  f_hi  -- nivel B inteiro (especifico, >= 2 de 4 + portao)
Publicado = (1, 1, 1). Criterio lexicografico no pior caso: banda, inicio,
deteccao, depois menor FP/mes e menor carga.

Uso (de dentro de scripts/pdm_fisico, com os dados):
    python limiares_robustos.py
    python limiares_robustos.py --clearml --remote
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
F_TP = [0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6]
F_LO = [0.85, 1.0, 1.15]
F_HI = [0.85, 1.0, 1.15]
IMAGEM = "tensorflow/tensorflow:2.16.1-gpu"
CACHE = "sinais_retreino_dia{:02d}.npz"      # regeneravel; *.npz esta no .gitignore


def garante_dados(dataset_id):
    precisa = ["grade2min.parquet", "falhas.csv", "piso_fisico_cache.npz"]
    if all(os.path.exists(f) for f in precisa):
        return
    from clearml import Dataset
    D = Dataset.get(dataset_id=dataset_id).get_local_copy()
    for f in precisa:
        if not os.path.exists(f):
            os.symlink(os.path.join(D, f), f)


def sinais_do_dia(dia):
    """t, p e sp com retreino no `dia` de cada mes -- o walk-forward de
    robustez_retreino.py, variante `unico` (o do publicado). Cacheado."""
    f = CACHE.format(dia)
    if not os.path.exists(f):
        from cabiunas_pdm import config as C
        from robustez_retreino import VARIANTES, walkforward
        g = pd.read_parquet("grade2min.parquet")
        stable = ((g["RUNNING_A"] > 0.5) & (g["T5_AVG_A"] > 300)).fillna(False)
        np.savez_compressed(f, **walkforward(g[C.SENSOR_TAGS], stable, dia, VARIANTES["unico"]))
    z = np.load(f)
    return {c: z[c] for c in ("t", "p", "sp")}


def limiares(PC, f_tp, f_lo, f_hi):
    K = {c: PC.K[c] * f_hi * (f_tp if c in ("t", "p") else 1.0) for c in PC.SIN}
    K_LO = {c: PC.K_LO[c] * f_lo * (f_tp if c in ("t", "p") else 1.0) for c in PC.SIN}
    return K, K_LO


def mede(args):
    """Um ponto da grade numa realizacao. Roda em processo filho."""
    dia, f_tp, f_lo, f_hi = args
    import publica_clearml as PC
    K, K_LO = limiares(PC, f_tp, f_lo, f_hi)
    al, mask, alvo, _, _, sel = PC.reproduz(v2=True, sinais=sinais_do_dia(dia), K_=K, K_LO_=K_LO)
    res, _, _ = PC.metricas(al, mask, alvo, sel, permutacao=False)
    n = lambda s: int(s.split("/")[0])
    return dict(dia=dia, f_tp=f_tp, f_lo=f_lo, f_hi=f_hi, det=n(res["recall"]),
                inicio=n(res["recall_regua_inicio"]), banda=n(res["recall_banda_acionavel"]),
                fp_mes=res["fp_por_mes_regra_c"], h_fp=res["horas_fp_por_mes_regra_c"],
                carga=res["carga_h_por_mes"], lead_ini=res["lead_medio_inicio_h"])


def pior_caso(T):
    """Por ponto da grade: o pior valor de cada metrica entre as realizacoes."""
    return (T.groupby(["f_tp", "f_lo", "f_hi"])
             .agg(banda=("banda", "min"), inicio=("inicio", "min"), det=("det", "min"),
                  fp_mes=("fp_mes", "max"), carga=("carga", "max"),
                  banda_med=("banda", "median"), fp_mes_med=("fp_mes", "median"),
                  carga_med=("carga", "median"))
             .reset_index()
             .sort_values(["banda", "inicio", "det", "fp_mes", "carga"],
                          ascending=[False, False, False, True, True]))


def roda(tarefas, n_proc):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_proc) as ex:
        return pd.DataFrame(list(ex.map(mede, tarefas, chunksize=4)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-id", default="8b06a98f8b264820a9ecf2075a188395")
    ap.add_argument("--proc", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--clearml", action="store_true")
    ap.add_argument("--remote", action="store_true")
    ap.add_argument("--fila", default="default")
    ap.add_argument("--nome", default="geladeira::limiares_robustos_ao_retreino")
    a = ap.parse_args()

    task = None
    if a.clearml or a.remote:
        from clearml import Task
        task = Task.init(project_name="TesteMLCab", task_name=a.nome, task_type=Task.TaskTypes.testing,
                         reuse_last_task_id=False, auto_connect_frameworks=False)
        task.set_base_docker(docker_image=IMAGEM)
        task.set_packages(["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "tzdata",
                           "matplotlib", "clearml"])
        task.add_tags(["geladeira", "retreino", "minimax", "limiares"])
        if a.remote:
            task.execute_remotely(queue_name=a.fila, exit_process=True)

    garante_dados(a.dataset_id)
    for d in DIAS_SELECAO + DIAS_TESTE:           # sinais antes do pool: sem corrida no cache
        sinais_do_dia(d)
        print(f"sinais do dia {d:2d} prontos", flush=True)

    # controle: publicado no dia 1 a partir do walk-forward recalculado
    c = mede((1, 1.0, 1.0, 1.0))
    print("CONTROLE:", c, flush=True)
    assert (c["det"], c["inicio"], c["banda"], c["fp_mes"]) == (8, 6, 5, 0.344), \
        "controle nao reproduz o ponto publicado -- nada abaixo vale"

    grade = list(itertools.product(F_TP, F_LO, F_HI))
    print(f"SELECAO: {len(grade)} pontos x {len(DIAS_SELECAO)} realizacoes, {a.proc} processos", flush=True)
    S = roda([(d, *g) for g in grade for d in DIAS_SELECAO], a.proc)
    S.to_csv("limiares_robustos_selecao.csv", index=False)
    P = pior_caso(S)
    esc = P.iloc[0]
    pub = P[(P.f_tp == 1.0) & (P.f_lo == 1.0) & (P.f_hi == 1.0)].iloc[0]
    esc_f = (float(esc.f_tp), float(esc.f_lo), float(esc.f_hi))

    print("TESTE nas realizacoes nunca vistas", flush=True)
    Tt = roda([(d, *f) for f in [(1.0, 1.0, 1.0), esc_f] for d in DIAS_TESTE], a.proc)
    Tt["ponto"] = np.where((Tt.f_tp == 1.0) & (Tt.f_lo == 1.0) & (Tt.f_hi == 1.0), "publicado", "robusto")
    if esc_f == (1.0, 1.0, 1.0):
        Tt["ponto"] = "publicado = robusto"
    Tt.to_csv("limiares_robustos_teste.csv", index=False)
    RT = (Tt.groupby("ponto")
            .agg(banda_pior=("banda", "min"), inicio_pior=("inicio", "min"), det_pior=("det", "min"),
                 fp_mes_pior=("fp_mes", "max"), carga_pior=("carga", "max"),
                 banda_med=("banda", "median"), inicio_med=("inicio", "median"),
                 fp_mes_med=("fp_mes", "median"), carga_med=("carga", "median"))
            .reset_index())
    dia1 = mede((1, *esc_f))

    pd.set_option("display.width", 220)
    print("\n=== SELECAO: 10 melhores pelo pior caso (dias 1/8/15/22) ===")
    print(P.head(10).to_string(index=False))
    print("\npublicado na mesma regua:\n" + pub.to_frame().T.to_string(index=False))
    print(f"\nESCOLHIDO: f_tp={esc_f[0]} f_lo={esc_f[1]} f_hi={esc_f[2]}")
    print("\n=== TESTE: realizacoes nunca vistas (dias 4/11/18/25) ===")
    print(RT.to_string(index=False))
    print("\npor dia:\n" + Tt.to_string(index=False))
    print("\no escolhido no dia 1 (o numero que se publicaria):", dia1)

    if task is not None:
        lg = task.get_logger()
        lg.report_table("selecao (pior caso)", "top 20", iteration=0, table_plot=P.head(20))
        lg.report_table("teste (nunca vistas)", "resumo", iteration=0, table_plot=RT)
        lg.report_table("teste (nunca vistas)", "por dia", iteration=0, table_plot=Tt)
        lg.report_table("escolhido no dia 1", "", iteration=0, table_plot=pd.DataFrame([dia1]))
        for f in ("limiares_robustos_selecao.csv", "limiares_robustos_teste.csv"):
            task.upload_artifact(f[:-4], artifact_object=f)
        task.flush(wait_for_uploads=True)


if __name__ == "__main__":
    main()
