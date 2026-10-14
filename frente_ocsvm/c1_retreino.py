#!/usr/bin/env python3
"""C1 -- retreinar os 3 OCSVM todo mês (walk-forward, janela expansiva) corta FP fora da amostra sem perder trip?

PRÉ-REGISTRADO EM 14/10/2026, ANTES DE ENVIAR AS TASKS. No EXP10c (modelo único), o walk-forward mensal cortou
~19% do FP pontual com o mesmo hit_rate (docs/analise_automl_exp10.md). A pergunta é se isso se repete nos
canais separados, depois do voto >= 2 com o canal 4, do filtro de 45 min e do refratário de 48 h.

DESENHO (decidido com o usuário em 14/10). O src/ proíbe ENABLE_WALKFORWARD_RETRAIN junto com o filtro de duração
por canal (automl_pipeline.py:290), e os 3 configs de produção usam esse filtro (4,5 min). Por isso há um controle:
  c0  treino único (corte 01/07/2025), ENABLE_MIN_DURATION_FILTER = false
  c1  igual ao c0 + ENABLE_WALKFORWARD_RETRAIN = true, WALKFORWARD_RETRAIN_FREQ = "MS" (retreina do zero no início
      de cada mês de 07/2025 a 04/2026, com todo o normal anterior; normalização e limiar recalculados a cada mês)
Nos dois braços, percentil e debounce ficam FIXOS nos de produção (temp p97,5/d1, vib p95/d4, óleo p99,9/d12),
para isolar a cadência de retreino da reescolha do AutoML. O A4 mostrou que essa reescolha pode derrubar a
vibração (sp0501: 4/8). Semente 42, mesmo src/, cada braço = 3 tasks clonadas das de produção.
  c1 x c0    efeito do retreino mensal (a pergunta)
  c0 x prod  efeito de tirar o filtro de 4,5 min (descritivo; na seção 3 do diário ele foi redundante)

O QUE SE MEDE. No c1, os pontos antes de 01/07/2025 saem com flag 0 (o walk-forward só pontua o período OOS), então
a comparação é FORA DA AMOSTRA (>= 01/07/2025): 3 trips (04/11/25, 09/12/25, 26/02/26), banda de 4 h, episódios
de FP e FP/mês fora, além do duty de cada canal OCSVM em operação. Mesma camada de decisão de referência (canal 4
com 5 tags/24 h, voto >= 2, 45 min, 48 h) e mesma régua. Referência da produção fora (A1): 3/3, banda 2/3,
23 FP, 2,877 FP/mês. Também se reporta o c0 no período inteiro (8 trips).

LEITURA PRÉ-REGISTRADA (c1 x c0, fora da amostra). O A4 deu ruído de treino de ~±1 episódio:
  APROVADO   c1 mantém 3/3 fora, banda >= a do c0 E tem pelo menos 3 episódios de FP a menos que o c0.
  REPROVADO  c1 perde algum trip fora, OU a redução é menor que 3 episódios (incluindo piora).
Se APROVADO, o próximo passo é retreino mensal com o AutoML livre (o caso de produção real) e o período inteiro.

EXPECTATIVA REGISTRADA. REPROVADO por efeito nulo: 3/3 mantido, FP fora do c1 a até ±2 episódios do c0. Os ~19%
pontuais do EXP10c se diluem no voto com o canal 4 e no refratário de 48 h. O c0 fica a até ±2 episódios da produção.

Uso:
  python frente_ocsvm/c1_retreino.py gera                # escreve as configs (commitar e dar push)
  python frente_ocsvm/c1_retreino.py envia --so c0       # clona as tasks de produção e enfileira um braço
  python frente_ocsvm/c1_retreino.py avalia              # baixa os artefatos e aplica a régua
"""
from __future__ import annotations
import argparse
import json

import pandas as pd
import comum as C
from a4_variabilidade import RAIZ, REPO, BRANCH, CANAIS, canais_da_replica

CFG_DIR = C.AQUI / "c1_configs"
TASKS_JSON = C.AQUI / "c1_tasks.json"
FIXOS = {"temp": (97.5, 1), "vib": (95.0, 4), "oleo": (99.9, 12)}  # percentil/debounce escolhidos na produção
BRACOS = {
    "c0": dict(ENABLE_MIN_DURATION_FILTER=False),
    "c1": dict(ENABLE_MIN_DURATION_FILTER=False, ENABLE_WALKFORWARD_RETRAIN=True, WALKFORWARD_RETRAIN_FREQ="MS"),
}


def mudancas(braco: str, canal: str) -> dict:
    pct, deb = FIXOS[canal]
    return dict(BRACOS[braco], AUTOML_THRESHOLD_PERCENTILES=[pct], AUTOML_DEBOUNCE_GRID=[deb])


def gera():
    CFG_DIR.mkdir(exist_ok=True)
    for braco in BRACOS:
        for canal, (cfg, _, _) in CANAIS.items():
            d = json.loads((RAIZ / "configs/calibracao_v4_eq" / f"{cfg}.json").read_text(encoding="utf-8"))
            d.update(mudancas(braco, canal))
            (CFG_DIR / f"{braco}_{canal}.json").write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n",
                                                           encoding="utf-8")
    print(f"{len(BRACOS) * len(CANAIS)} configs em {CFG_DIR}")


def envia(so: str | None):
    import subprocess
    from clearml import Task
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RAIZ, text=True).strip()
    remoto = subprocess.check_output(["git", "ls-remote", "cabiunas", f"refs/heads/{BRANCH}"], cwd=RAIZ, text=True)
    assert remoto.startswith(commit), f"faça push de {commit[:7]} para cabiunas/{BRANCH} antes"
    reg = json.loads(TASKS_JSON.read_text()) if TASKS_JSON.exists() else {}
    for braco in BRACOS:
        for canal, (cfg, tid, _) in CANAIS.items():
            nome = f"{braco}_{canal}"
            if (so and so not in (braco, nome)) or nome in reg:
                continue
            t = Task.clone(source_task=tid, name=f"ocsvm-c1::{nome}::{cfg}")
            t.set_script(repository=REPO, branch=BRANCH, commit=commit)
            t.set_parameter("Args/config", f"frente_ocsvm/c1_configs/{nome}.json")
            for k, v in mudancas(braco, canal).items():  # só o painel; o cfg vem do JSON de Args/config
                t.set_parameter(f"pipeline_config/{k}", str(v) if isinstance(v, list) else v)
            Task.enqueue(t, queue_name="default")
            reg[nome] = t.id
            print(nome, t.id)
            TASKS_JSON.write_text(json.dumps(reg, indent=2) + "\n")


def avalia():
    reg = json.loads(TASKS_JSON.read_text()) if TASKS_JSON.exists() else {}
    reps = {"prod": {c: v[1] for c, v in CANAIS.items()}}
    reps.update({b: {c: reg[f"{b}_{c}"] for c in CANAIS} for b in BRACOS if all(f"{b}_{c}" in reg for c in CANAIS)})
    cat = C.catalogo()
    L = []
    for rep, ids in reps.items():
        try:
            df = canais_da_replica(ids)
        except Exception as e:  # task não terminou ou falhou
            print(f"{rep}: indisponível ({type(e).__name__}: {e})")
            continue
        idx, op = df.index, df["operational_state"]
        ft = C.trips(idx)
        c = {k: df[k] for k in C.CANAIS[:3]}
        c["c4"] = C.canal_alarme(idx, cat.loc[cat.tag.isin(C.TAGS_C4), "t"], C.REF["janela_c4_h"])
        cls, m = C.avalia(C.decide(c, 2, C.REF["dur_min"], C.REF["refrat_h"]), op, ft)
        P = C.por_trip(cls, ft)
        fora = idx >= C.SPLIT
        dias = C.compute_operational_period_days(pd.DataFrame({"operational_state": op[fora]}))
        fp = int(((cls.start >= C.SPLIT) & (cls.classe == "falso_positivo")).sum())
        Pf = P[P.amostra == "fora"]
        on = fora & (op == "on").values
        L.append(dict(rep=rep, det_fora=int(Pf.deteccao.notna().sum()), banda_fora=int(Pf.banda_4h.sum()),
                      lead_fora=" ".join(f"{x:.1f}" for x in Pf.lead_h), FP_fora=fp,
                      FP_mes_fora=round(fp / (dias / 30.4375), 3),
                      duty_fora=" ".join(f"{df.loc[on, k].mean():.3f}" for k in C.CANAIS[:3]),
                      det_total=int(P.deteccao.notna().sum()), FP_mes_total=round(m["falso_positivo_por_mes"], 3)))
    T = pd.DataFrame(L)
    pd.set_option("display.width", 200)
    print(T.to_string(index=False))
    if {"c0", "c1"} <= set(T.rep):
        a, b = T.set_index("rep").loc["c0"], T.set_index("rep").loc["c1"]
        ok = b.det_fora == 3 and b.banda_fora >= a.banda_fora and a.FP_fora - b.FP_fora >= 3
        print(f"\nc1 x c0 fora: det {b.det_fora}/3, banda {b.banda_fora} x {a.banda_fora}, "
              f"FP {b.FP_fora} x {a.FP_fora} ({b.FP_fora - a.FP_fora:+d}) -> {'APROVADO' if ok else 'REPROVADO'}")
        print("(det_total e FP_mes_total do c1 não valem: antes de 01/07/2025 o walk-forward sai com flag 0)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=["gera", "envia", "avalia"])
    ap.add_argument("--so")
    a = ap.parse_args()
    {"gera": gera, "envia": lambda: envia(a.so), "avalia": avalia}[a.acao]()
