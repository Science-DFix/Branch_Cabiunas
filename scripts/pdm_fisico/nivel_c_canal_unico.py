#!/usr/bin/env python3
"""Nivel C -- UM canal forte e sustentado: pega o evento de canal unico? A que custo?

Da geladeira. README ("Aberto tambem") e DETECTOR_V2 §6: a parada de 24/11/2025
(43 h, alarme de pressao baixa no header de oleo, PAL_6240339 -- o unico evento
de SAUDE MECANICA entre 14 paradas longas fora do alvo) tem o canal p a 6,14x o
limiar por 48 h seguidas, e o detector nao dispara. portao_cego_pressao.py
mostrou que NAO e o portao sp|vb: mesmo sem portao, o voto >= 2 exige dois
canais e os outros tres estao mudos. Um caminho para evento de um canal so nunca
foi testado.

O mecanismo. Terceiro gatilho em OU com os niveis A e B do v2: um canal, sozinho,
acima de `fator` x o limiar do nivel B, sustentado por `horas`. Todo o resto --
escalada, refratario, duracao minima, regua -- e o v2 publicado, pelo mesmo codigo
(publica_clearml.reproduz(nivel_c=...)).

O que pesa contra, ja medido. score_continuo.py: "um canal em 50x engole tres
canais moderados" -- canal solitario extremo e comum, e e por isso que o voto
discretiza. Tres dos 4 FP do v2 nascem so pelo nivel B com forca ate 170,6
(pericia_fp_v2.py). O nivel C tende a comprar FP; a pergunta e quanto.

Leitura das metricas. O 24/11 NAO esta no alvo oficial (a regra conta so o
segundo estagio de protecao), entao um alarme antes dele entra na Regra C como
NEUTRO, nao como acerto. Ele e reportado a parte: nasceu em [t-48h, t]? E em
[t-48h, t-4h] (banda)? Os 8 oficiais seguem nas tres reguas.

RESULTADO (30/09/2026, task geladeira::nivel_c_canal_unico, c1eace0e) -- REFUTADO.
Zero de 20 configuracoes (fator 2-6x, 6-36 h) antecipam o 24/11 na regua de
inicio. O nivel C DISPARA -- episodio de 19/11 23:28 a 23/11 10:12 no 6x/6h --
mas nasce 82 h antes da parada, fora da janela.

POR QUE, e e estrutural. A maquina parou no trip oficial de 04/11 e so voltou em
15/11 20:32. Da PRIMEIRA amostra apos o religamento ate a parada de 24/11, o p
fica constante em 6-8x o limiar do nivel B, oito dias seguidos. Nao e precursor
que se desenvolve nas 48 h finais: e um DEGRAU DE NIVEL na volta da manutencao,
contra um baseline mensal ajustado antes do trip. Nenhum gatilho pode fazer
nascer dentro de [t-48h, t] uma condicao que comecou 8 dias antes -- e o mesmo
limite do 17/03 (precursor lento). Nem a regua "de pe" credita: o episodio C
termina 23 h antes da parada.

Custo nos 8 oficiais: com fator >= 4 as reguas e o FP ficam intactos (5/8 · 6/8
· 8/8 · 0,344), mas a carga sobe de 48,9 para 53-68 h/mes; abaixo de 4 a banda
cai para 3-4/8. Nada a ganhar.

Hipotese que sai daqui, NAO testada: o baseline mensal fica velho depois de
parada longa/manutencao (aqui, 11 dias). Retreino disparado pelo religamento
apos parada longa mudaria o p em 16-24/11 -- mas com UM caso nao da para dizer
se isso esconderia um precursor real (pressao de oleo antes de alarme de
pressao baixa de oleo) ou um artefato de regime.

Uso (de dentro de scripts/pdm_fisico, com os dados):
    python nivel_c_canal_unico.py
    python nivel_c_canal_unico.py --clearml --remote
"""
from __future__ import annotations

import argparse
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
os.chdir(AQUI)
sys.path.insert(0, AQUI)

if {"--clearml", "--remote"} & set(sys.argv) or os.environ.get("CLEARML_TASK_ID"):
    import clearml  # noqa: F401  -- antes do argparse: o worker injeta os args por patch

import pandas as pd

EVENTO_2411 = pd.Timestamp("2025-11-24 09:22", tz="UTC")   # inicio da parada de 43 h
FATORES = [2.0, 3.0, 4.0, 5.0, 6.0]
HORAS = [6, 12, 24, 36]
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


def mede(PC, AV, nivel_c):
    al, mask, alvo, _, idx, sel = PC.reproduz(v2=True, nivel_c=nivel_c)
    res, _, _ = PC.metricas(al, mask, alvo, sel, permutacao=False)
    eps = AV.episodios(al)
    J, TMIN = pd.Timedelta(hours=48), pd.Timedelta(hours=4)
    nasc = [a for a, _ in eps if EVENTO_2411 - J <= a <= EVENTO_2411]
    n = lambda s: int(s.split("/")[0])
    return dict(fator=None if nivel_c is None else nivel_c[0],
                horas=None if nivel_c is None else nivel_c[1],
                det=n(res["recall"]), inicio=n(res["recall_regua_inicio"]),
                banda=n(res["recall_banda_acionavel"]), fp=res["fp"],
                fp_mes=res["fp_por_mes_regra_c"], h_fp_mes=res["horas_fp_por_mes_regra_c"],
                carga=res["carga_h_por_mes"], eps=res["episodios"], lead_ini=res["lead_medio_inicio_h"],
                ev2411_inicio=bool(nasc),
                ev2411_banda=any(a <= EVENTO_2411 - TMIN for a in nasc),
                ev2411_lead_h=round((EVENTO_2411 - min(nasc)).total_seconds() / 3600, 1) if nasc else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-id", default="8b06a98f8b264820a9ecf2075a188395")
    ap.add_argument("--clearml", action="store_true")
    ap.add_argument("--remote", action="store_true")
    ap.add_argument("--fila", default="default")
    ap.add_argument("--nome", default="geladeira::nivel_c_canal_unico")
    a = ap.parse_args()

    task = None
    if a.clearml or a.remote:
        from clearml import Task
        task = Task.init(project_name="TesteMLCab", task_name=a.nome, task_type=Task.TaskTypes.testing,
                         reuse_last_task_id=False, auto_connect_frameworks=False)
        task.set_base_docker(docker_image=IMAGEM)
        task.set_packages(["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "tzdata",
                           "matplotlib", "clearml"])
        task.add_tags(["geladeira", "nivel-c", "24-11-2025"])
        if a.remote:
            task.execute_remotely(queue_name=a.fila, exit_process=True)

    garante_dados(a.dataset_id)
    import avalia as AV
    import publica_clearml as PC

    base = mede(PC, AV, None)
    print("CONTROLE (v2 publicado):", base, flush=True)
    assert (base["det"], base["inicio"], base["banda"], base["fp_mes"]) == (8, 6, 5, 0.344), \
        "controle nao reproduz o ponto publicado -- nada abaixo vale"
    linhas = [dict(base, config="v2 publicado")]
    for f in FATORES:
        for h in HORAS:
            r = mede(PC, AV, (f, h))
            linhas.append(dict(r, config=f"C: {f:g}x por {h} h"))
            print(linhas[-1], flush=True)
    T = pd.DataFrame(linhas)
    cols = ["config", "banda", "inicio", "det", "fp", "fp_mes", "h_fp_mes", "carga", "eps",
            "ev2411_inicio", "ev2411_banda", "ev2411_lead_h"]
    pd.set_option("display.width", 220)
    print("\n" + T[cols].to_string(index=False))
    pega = T[T.ev2411_inicio]
    print(f"\nconfiguracoes que antecipam o 24/11: {len(pega)} de {len(T) - 1}")
    if len(pega):
        ok = pega[(pega.banda >= 5) & (pega.inicio >= 6) & (pega.det >= 8)]
        print("... e mantem 5/8 · 6/8 · 8/8 nos oficiais:", len(ok))
        if len(ok):
            print(ok[cols].sort_values(["fp_mes", "carga"]).to_string(index=False))
    if task is not None:
        lg = task.get_logger()
        lg.report_table("grade nivel C", "todas", iteration=0, table_plot=T[cols])
        task.flush(wait_for_uploads=True)


if __name__ == "__main__":
    main()
