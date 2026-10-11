#!/usr/bin/env python3
"""A4 -- o 8/8 e os 2,88 FP/mês sobrevivem a retreinar os 3 OCSVM com outra semente ou outro split?

PRÉ-REGISTRADO EM 10/10/2026, ANTES DE ENVIAR AS TASKS. A0-A3 e B1-B2 usaram UM treino só (tasks
b494d457 / 9c344688 / fc4123fb, semente 42, split 01/07/2025). O OCSVM ajusta numa subamostra de 50 mil
pontos sorteada por RANDOM_SEED, e o AutoML de cada canal reescolhe percentil de limiar {95..99,9} e debounce
{1, 4, 12} a cada treino. No histórico, toda troca de modelo/grade perdeu 27/02 ou 17/03 (seção 3), e a
pipeline da Lara tinha ~±27 pp de ruído de semente. Sem saber esse ruído, nenhum ganho de C1-C3 é
interpretável: este é o piso.

RÉPLICAS (cada uma = 3 tasks no Cica, clones das de produção, mesmo código -- src/ idêntico a 6842ac7 --,
mudando SÓ o campo indicado; configs em frente_ocsvm/a4_configs/):
  s43, s44, s45, s46, s47   RANDOM_SEED = 43..47, split 01/07/2025          (PRIMÁRIO: ruído de semente)
  sp0501, sp0901            RANDOM_SEED = 42, split 01/05/2025 e 01/09/2025  (SECUNDÁRIO: ruído de corte)
Cada réplica passa pela MESMA camada de decisão de referência (canal 4 com as 5 tags e 24 h, voto >= 2,
45 min, 48 h) e pela régua de sempre. Controle do caminho: a "réplica" prod (as 3 tasks de produção
baixadas pelo mesmo código) tem de dar 8/8 e 2,8847 FP/mês.

O QUE SE MEDE, por réplica: trips detectados (de 8; fora da amostra de 3, com o split de 01/07/2025), banda
de 4 h, FP/mês, duty de cada canal OCSVM em operação, e o percentil/debounce que o AutoML escolheu.

LEITURA PRÉ-REGISTRADA (primário, 5 sementes):
  ROBUSTO  as 5 sementes dão >= 7/8 E FP/mês dentro de ±20% de 2,88 (2,30 a 3,46).
  FRÁGIL   alguma semente dá <= 6/8 OU FP/mês fora de ±20%.
Consequência: se FRÁGIL, o 8/8 é um sorteio favorável e C1-C3 só podem ser julgados com várias sementes
por braço (o efeito mínimo detectável passa a ser a dispersão daqui). Se ROBUSTO, C1-C3 podem ser julgados
com uma semente.

EXPECTATIVA REGISTRADA. FRÁGIL: mediana 7/8, ao menos uma semente <= 6/8, FP/mês 2,4-3,4; os trips que caem
são 27/02/25 e/ou 17/03/25; o AutoML troca de percentil em pelo menos um canal entre sementes. Splits:
pelo menos um dos dois <= 6/8.

Uso:
  python frente_ocsvm/a4_variabilidade.py gera                  # escreve as configs (commitar e dar push)
  python frente_ocsvm/a4_variabilidade.py envia [--so s43]      # clona as tasks de produção e enfileira
  python frente_ocsvm/a4_variabilidade.py avalia                # baixa os artefatos e aplica a régua
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import comum as C

RAIZ = C.AQUI.parent
CFG_DIR = C.AQUI / "a4_configs"
TASKS_JSON = C.AQUI / "a4_tasks.json"
REPO = "git@github.com:Science-DFix/Branch_Cabiunas.git"
BRANCH = "exp/ocsvm-validacao"
CANAIS = {  # canal: (config de produção, task de produção, artefato)
    "temp": ("test_grupo_exp33_mancal_temperatura_isolada", "b494d457161347c293e0d9ccae7ee15a",
             "mancal_temperatura_isolada"),
    "vib": ("test_grupo_exp34_mancal_vibracao_isolada", "9c344688f7914fbda89963d7af693729",
            "mancal_vibracao_isolada"),
    "oleo": ("test_grupo_exp38_oleo_pressao_isolada", "fc4123fb935c4907931c69c065b6bc5d",
             "oleo_pressao_isolada"),
}
REPLICAS = {f"s{s}": dict(RANDOM_SEED=s) for s in range(43, 48)}
REPLICAS.update(sp0501=dict(AUTOML_OOS_SPLIT_DATE="2025-05-01"), sp0901=dict(AUTOML_OOS_SPLIT_DATE="2025-09-01"))


def gera():
    CFG_DIR.mkdir(exist_ok=True)
    for rep, mud in REPLICAS.items():
        for canal, (cfg, _, _) in CANAIS.items():
            d = json.loads((RAIZ / "configs/calibracao_v4_eq" / f"{cfg}.json").read_text(encoding="utf-8"))
            assert all(k in d for k in mud), mud
            d.update(mud)
            (CFG_DIR / f"{rep}_{canal}.json").write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n",
                                                         encoding="utf-8")
    print(f"{len(REPLICAS) * len(CANAIS)} configs em {CFG_DIR}")


def envia(so: str | None):
    import subprocess
    from clearml import Task
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RAIZ, text=True).strip()
    remoto = subprocess.check_output(["git", "ls-remote", "cabiunas", f"refs/heads/{BRANCH}"], cwd=RAIZ, text=True)
    assert remoto.startswith(commit), f"faça push de {commit[:7]} para cabiunas/{BRANCH} antes"
    reg = json.loads(TASKS_JSON.read_text()) if TASKS_JSON.exists() else {}
    for rep in REPLICAS:
        for canal, (cfg, tid, _) in CANAIS.items():
            nome = f"{rep}_{canal}"
            if (so and so not in (rep, nome)) or nome in reg:
                continue
            t = Task.clone(source_task=tid, name=f"ocsvm-a4::{nome}::{cfg}")
            t.set_script(repository=REPO, branch=BRANCH, commit=commit)
            t.set_parameter("Args/config", f"frente_ocsvm/a4_configs/{nome}.json")
            for k, v in REPLICAS[rep].items():  # o painel herda os valores da produção; alinha com a config
                t.set_parameter(f"pipeline_config/{k}", v)
            Task.enqueue(t, queue_name="default")
            reg[nome] = t.id
            print(nome, t.id)
            TASKS_JSON.write_text(json.dumps(reg, indent=2) + "\n")


def canais_da_replica(ids: dict) -> pd.DataFrame:
    from clearml import Task
    dfs = {}
    for canal, (_, _, art) in CANAIS.items():
        p = Task.get_task(task_id=ids[canal]).artifacts[f"{art}/csv/point_anomalies_all.csv"].get_local_copy()
        dfs[canal] = pd.read_csv(p, index_col=0, parse_dates=True, low_memory=False)
    idx = dfs["temp"].index.intersection(dfs["vib"].index).intersection(dfs["oleo"].index)
    out = pd.DataFrame({c: dfs[k].loc[idx, "is_anom_point"].astype(bool)
                        for c, k in zip(C.CANAIS[:3], CANAIS)})
    out["operational_state"] = dfs["temp"].loc[idx, "operational_state"]
    return out


def hiper(tid: str, art: str) -> str:
    from clearml import Task
    a = Task.get_task(task_id=tid).artifacts
    k = next((k for k in a if k.startswith(art) and k.endswith("best_hyperparameters.json")), None)
    if k is None:
        return "?"
    h = json.loads(Path(a[k].get_local_copy()).read_text())
    return f"p{h.get('threshold_percentile')}/d{h.get('debounce')}"


def avalia():
    reg = json.loads(TASKS_JSON.read_text()) if TASKS_JSON.exists() else {}
    reps = {"prod": {c: v[1] for c, v in CANAIS.items()}}
    reps.update({r: {c: reg[f"{r}_{c}"] for c in CANAIS} for r in REPLICAS if all(f"{r}_{c}" in reg for c in CANAIS)})
    cat = C.catalogo()
    L = []
    for rep, ids in reps.items():
        try:
            df = canais_da_replica(ids)
        except Exception as e:  # task não terminou ou falhou
            print(f"{rep}: indisponível ({type(e).__name__}: {e})")
            continue
        ft = C.trips(df.index)
        c = {k: df[k] for k in C.CANAIS[:3]}
        c["c4"] = C.canal_alarme(df.index, cat.loc[cat.tag.isin(C.TAGS_C4), "t"], C.REF["janela_c4_h"])
        cls, m = C.avalia(C.decide(c, 2, C.REF["dur_min"], C.REF["refrat_h"]), df["operational_state"], ft)
        P = C.por_trip(cls, ft)
        on = df["operational_state"] == "on"
        L.append(dict(rep=rep, det=int(P.deteccao.notna().sum()),
                      fora=int(P.deteccao.notna()[P.amostra == "fora"].sum()), banda=int(P.banda_4h.sum()),
                      fp=round(m["falso_positivo_por_mes"], 4),
                      duty=" ".join(f"{df[k][on].mean():.3f}" for k in C.CANAIS[:3]),
                      automl=" ".join(hiper(ids[k], CANAIS[k][2]) for k in CANAIS),
                      perde=", ".join(t.strftime("%d/%m/%y") for t, d in zip(P.trip, P.deteccao) if pd.isna(d)) or "-"))
    T = pd.DataFrame(L)
    print(T.to_string(index=False))
    S = T[T.rep.str.match(r"s\d")]
    if len(S) == 5:
        rob = bool((S.det >= 7).all() and S.fp.between(2.30, 3.46).all())
        print(f"\nPRIMÁRIO (5 sementes): det min {S.det.min()} / mediana {S.det.median():.0f}; "
              f"FP/mês {S.fp.min():.3f} a {S.fp.max():.3f} -> {'ROBUSTO' if rob else 'FRÁGIL'}")
    else:
        print(f"\nPRIMÁRIO: só {len(S)} de 5 sementes disponíveis")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=["gera", "envia", "avalia"])
    ap.add_argument("--so")
    a = ap.parse_args()
    {"gera": gera, "envia": lambda: envia(a.so), "avalia": avalia}[a.acao]()
