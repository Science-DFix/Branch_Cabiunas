#!/usr/bin/env python3
"""O D do CVA no lugar do PCA nos canais t e p -- o detector inteiro, nao o canal.

De onde vem. A avaliacao sem rotulo (skill avalia-normalidade, tasks
avalia_normalidade::* no TesteMLCab) mostrou que o PCA mensal passa do proprio
limiar em 42% (temperatura) e 19% (pressao) do mes seguinte -- deriva -- e que a
estatistica D do CVA (dissimilaridade passado->futuro) fica em 4,1% e 3,2%, com a
melhor separacao dos trips (AUC 0,79 e 0,77). Canal melhor nao e detector melhor:
o DETECTOR_V2 §7 registra cinco canais melhores isoladamente que nao melhoraram
nada. Este script faz a pergunta que decide.

O que muda e o que nao muda. So os sinais t e p. Mascara, EWMA, degrau|CUSUM,
voto de dois niveis, escalada, refratario, duracao minima e a regua saem do
publica_clearml.py -- o mesmo codigo, nao uma copia (reproduz(sinais=...) e
metricas()).

A varredura e JUSTA. O D e muito mais "frio" que o PCA (4% contra 42% de
excedencia), entao os mesmos multiplicadores nao servem. Cada braco -- o PCA
incluido -- recebe a MESMA grade de fatores f_t, f_p, que multiplicam os
limiares de t e p nos dois niveis (a razao nivel A / nivel B fica a publicada).
O PCA em f = (1, 1) e o ponto publicado: e o controle, e tem de sair 8/8 ·
inicio 6/8 · banda 5/8 · 0,344 FP/mes.

Selecao pelos criterios do projeto, nao pelo otimo nos 8 eventos:
  - MINIMAX da vizinhanca (ponto_de_deploy.py): cada ponto vale o pior dos seus
    vizinhos a +-1 passo de grade; ordena por banda, inicio, deteccao e FP/mes.
  - HOLDOUT TEMPORAL (validacao_temporal.py): escolhe f so com o que aconteceu
    antes de 01/07/2025 e mede depois.

Uso (de dentro de scripts/pdm_fisico, com grade2min.parquet, falhas.csv e
piso_fisico_cache.npz no diretorio -- ou --dataset-id para baixar):
    python experimento_cva_detector.py
    python experimento_cva_detector.py --clearml --remote --dataset-id 8b06a98f...
"""
from __future__ import annotations

import argparse
import json
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
os.chdir(AQUI)
sys.path.insert(0, AQUI)

if {"--clearml", "--remote"} & set(sys.argv) or os.environ.get("CLEARML_TASK_ID"):
    import clearml  # noqa: F401  -- antes do argparse: o worker injeta os args por patch

import numpy as np
import pandas as pd

FIT_POINTS = 20_000                     # o mesmo do PCA walk-forward
CORTE = pd.Timestamp("2025-07-01", tz="UTC")
FATORES = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0]
CACHE = "cva_d_cache.npz"               # regeneravel; nao versionar
IMAGEM = "tensorflow/tensorflow:2.16.1-gpu"


# ────────────────────────────────────────────────────────── dados
def garante_dados(dataset_id: str | None):
    precisa = ["grade2min.parquet", "falhas.csv", "piso_fisico_cache.npz"]
    if all(os.path.exists(f) for f in precisa):
        return
    if not dataset_id:
        raise SystemExit(f"faltam {precisa} em {AQUI}; passe --dataset-id")
    from clearml import Dataset
    D = Dataset.get(dataset_id=dataset_id).get_local_copy()
    for f in precisa:
        if not os.path.exists(f):
            os.symlink(os.path.join(D, f), f)


# ────────────────────────────────────────────────────────── sinal
def sinais_cva(col: str = "D") -> dict:
    """t e p como a estatistica `col` do CVA, walk-forward mensal.

    Mesma regra de ajuste do PCA em piso_fisico.py::pre(): para o mes M, os
    FIT_POINTS pontos estaveis mais recentes ANTES de M; mes sem 1/4 disso fica
    NaN. O mes e pontuado inteiro (inclusive fora da mascara) porque o EWMA e o
    CUSUM correm sobre a serie -- a mascara filtra a decisao, nao o sinal."""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        if str(z["col"]) == col:
            print(f"cache {CACHE} reaproveitado", flush=True)
            return {"t": z["t"], "p": z["p"]}
    from cabiunas_pdm import config as C
    from dinamico import ScorerCVA
    g = pd.read_parquet("grade2min.parquet")
    idx = g.index
    stable = ((g["RUNNING_A"] > 0.5) & (g["T5_AVG_A"] > 300)).fillna(False)
    meses = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC")
    out = {c: np.full(len(idx), np.nan) for c in ("t", "p")}
    for i, m0 in enumerate(meses):
        m1 = meses[i + 1] if i + 1 < len(meses) else idx[-1] + pd.Timedelta("2min")
        sel = (idx >= m0) & (idx < m1)
        for c, tags in (("t", C.TEMPERATURE_TAGS), ("p", C.PRESSURE_TAGS)):
            fit = g.loc[stable & (idx < m0), tags].dropna().tail(FIT_POINTS)
            if len(fit) < FIT_POINTS // 4:
                continue
            out[c][sel] = ScorerCVA().fit(fit).score(g.loc[sel, tags])[col].to_numpy()
        print(f"  {m0:%Y-%m}", flush=True)
    np.savez_compressed(CACHE, col=col, **out)
    return out


# ────────────────────────────────────────────────────────── medida
def mede(PC, sinais, ft, fp, alvo_sub=None, periodo=None):
    """Um ponto da grade: reproduz o detector com limiares de t e p escalados."""
    K = dict(PC.K); K_LO = dict(PC.K_LO)
    K["t"] *= ft; K_LO["t"] *= ft
    K["p"] *= fp; K_LO["p"] *= fp
    al, mask, alvo, _, idx, sel = PC.reproduz(v2=True, sinais=sinais, K_=K, K_LO_=K_LO)
    if periodo is not None:
        a, b = periodo
        janela = pd.Series((idx >= a) & (idx < b), index=idx)
        al, sel = al & janela, sel & janela
        alvo = pd.Series([t for t in alvo if a <= t < b])
    res, _, _ = PC.metricas(al, mask, alvo, sel, permutacao=False)
    n = lambda s: int(s.split("/")[0])
    return dict(ft=ft, fp=fp, det=n(res["recall"]), ini=n(res["recall_regua_inicio"]),
                banda=n(res["recall_banda_acionavel"]), n_ev=len(alvo),
                fp_mes=res["fp_por_mes_regra_c"], h_fp=res["horas_fp_por_mes_regra_c"],
                carga=res["carga_h_por_mes"], lead_ini=res["lead_medio_inicio_h"],
                eps=res["episodios"])


def chave(r):
    """Ordem de preferencia do projeto: banda, inicio, deteccao; depois custo."""
    return (r["banda"], r["ini"], r["det"], -r["fp_mes"], -r["carga"])


def minimax(T: pd.DataFrame) -> pd.DataFrame:
    """Cada ponto vale o PIOR vizinho a +-1 passo (inclusive ele mesmo)."""
    pos = {f: i for i, f in enumerate(FATORES)}
    linhas = []
    for _, r in T.iterrows():
        viz = T[(T.ft.map(pos) - pos[r.ft]).abs().le(1) & (T.fp.map(pos) - pos[r.fp]).abs().le(1)]
        pior = min(viz.to_dict("records"), key=chave)
        linhas.append({**r.to_dict(), **{f"pior_{k}": pior[k] for k in
                                         ("banda", "ini", "det", "fp_mes", "carga")}})
    M = pd.DataFrame(linhas)
    return M.sort_values(["pior_banda", "pior_ini", "pior_det", "pior_fp_mes", "pior_carga"],
                         ascending=[False, False, False, True, True])


def braco(PC, nome, sinais):
    T = pd.DataFrame([mede(PC, sinais, ft, fp) for ft in FATORES for fp in FATORES])
    T.insert(0, "braco", nome)
    print(f"[{nome}] {len(T)} pontos", flush=True)
    return T


def holdout(PC, nome, sinais, ini_serie):
    """Escolhe (ft, fp) so no passado; mede no futuro."""
    tr = pd.DataFrame([mede(PC, sinais, ft, fp, periodo=(ini_serie, CORTE))
                       for ft in FATORES for fp in FATORES])
    melhor = max(tr.to_dict("records"), key=chave)
    fim = pd.Timestamp("2100-01-01", tz="UTC")
    te = mede(PC, sinais, melhor["ft"], melhor["fp"], periodo=(CORTE, fim))
    return dict(braco=nome, ft=melhor["ft"], fp=melhor["fp"],
                treino=f'{melhor["banda"]}/{melhor["n_ev"]} banda · {melhor["fp_mes"]} FP/mes',
                teste_banda=f'{te["banda"]}/{te["n_ev"]}', teste_ini=f'{te["ini"]}/{te["n_ev"]}',
                teste_det=f'{te["det"]}/{te["n_ev"]}', teste_fp_mes=te["fp_mes"],
                teste_carga=te["carga"])


# ────────────────────────────────────────────────────────── main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-id", default="8b06a98f8b264820a9ecf2075a188395")
    ap.add_argument("--col", default="D", help="estatistica do CVA para t e p")
    ap.add_argument("--clearml", action="store_true")
    ap.add_argument("--remote", action="store_true")
    ap.add_argument("--fila", default="default")
    ap.add_argument("--nome", default="experimento::cva_D_no_detector")
    a = ap.parse_args()

    task = None
    if a.clearml or a.remote:
        from clearml import Task
        task = Task.init(project_name="TesteMLCab", task_name=a.nome, task_type=Task.TaskTypes.testing,
                         reuse_last_task_id=False, auto_connect_frameworks=False)
        task.set_base_docker(docker_image=IMAGEM)
        task.set_packages(["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "tzdata",
                           "matplotlib", "clearml"])
        task.add_tags(["cva", "sfa-cva", "detector"])
        if a.remote:
            task.execute_remotely(queue_name=a.fila, exit_process=True)

    garante_dados(a.dataset_id)
    import publica_clearml as PC

    # controle: o ponto publicado tem de sair intacto
    c = mede(PC, None, 1.0, 1.0)
    print("CONTROLE (PCA, f=1):", json.dumps(c), flush=True)
    assert (c["det"], c["ini"], c["banda"], c["fp_mes"]) == (8, 6, 5, 0.344), \
        "o controle nao reproduz o ponto publicado -- nada abaixo vale"

    cva = sinais_cva(a.col)
    bracos = {"PCA (controle)": None,
              f"CVA {a.col} em t e p": cva,
              f"CVA {a.col} so em t": {"t": cva["t"]},
              f"CVA {a.col} so em p": {"p": cva["p"]}}

    T = pd.concat([braco(PC, n, s) for n, s in bracos.items()], ignore_index=True)
    T.to_csv("experimento_cva_grade.csv", index=False)
    M = pd.concat([minimax(g) for _, g in T.groupby("braco", sort=False)], ignore_index=True)
    topo = M.groupby("braco", sort=False).head(1)
    pontual = T.loc[T.groupby("braco", sort=False).apply(
        lambda g: max(g.index, key=lambda i: chave(T.loc[i])), include_groups=False)]

    ini_serie = pd.read_parquet("grade2min.parquet", columns=["RUNNING_A"]).index[0]
    H = pd.DataFrame([holdout(PC, n, s, ini_serie) for n, s in bracos.items()])

    pd.set_option("display.width", 200)
    cols = ["braco", "ft", "fp", "banda", "ini", "det", "fp_mes", "h_fp", "carga", "lead_ini", "eps"]
    print("\n=== MELHOR PONTO (otimo pontual -- otimista por construcao) ===")
    print(pontual[cols].to_string(index=False))
    print("\n=== PONTO DE DEPLOY (minimax da vizinhanca) ===")
    print(topo[cols + ["pior_banda", "pior_ini", "pior_det", "pior_fp_mes", "pior_carga"]].to_string(index=False))
    print(f"\n=== HOLDOUT TEMPORAL (escolhe antes de {CORTE:%d/%m/%Y}, mede depois) ===")
    print(H.to_string(index=False))

    if task is not None:
        lg = task.get_logger()
        lg.report_table("controle", "PCA f=1", iteration=0, table_plot=pd.DataFrame([c]))
        lg.report_table("otimo pontual", "por braco", iteration=0, table_plot=pontual[cols])
        lg.report_table("minimax (deploy)", "por braco", iteration=0, table_plot=topo)
        lg.report_table("holdout temporal", "por braco", iteration=0, table_plot=H)
        lg.report_table("grade completa", "todos", iteration=0, table_plot=T)
        task.upload_artifact("grade", artifact_object="experimento_cva_grade.csv")
        task.flush(wait_for_uploads=True)


if __name__ == "__main__":
    main()
