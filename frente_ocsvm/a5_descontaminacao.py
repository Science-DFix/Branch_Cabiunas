#!/usr/bin/env python3
"""A5 -- tirar do treino as 72 h antes de cada trip faz os trips de dentro da amostra virarem coincidência de 2 modelos?

PRÉ-REGISTRADO EM 14/10/2026, ANTES DE ENVIAR AS TASKS. O B3 achou (a posteriori) que a prioridade R1 (>= 2 dos 3
OCSVM ativos juntos) põe os 3 trips de FORA da amostra em ALTA e os 5 de DENTRO em BAIXA. Hipótese: o trecho antes
dos trips de dentro entrou no treino como normal, e os modelos aprenderam o precursor. O treino só exclui ±24 h em
volta dos alarmes dos sensores do grupo, nunca em volta dos trips.

OLHADO ANTES (conferência local da máscara de treino, sem treinar): fração das horas "on" em [T-72 h, T) que
entra no normal de treino, por grupo (temp · vib · óleo):
  27/02/25 100·100·100%   17/03/25 98·98·100%   07/04/25 100·100·100%   11/04/25 29·29·100%   29/04/25 100·100·100%
  os 3 de fora: 0% (estão depois do corte de 01/07/2025).

O TRATAMENTO. Os 3 OCSVM de produção retreinados com TRAIN_EXCLUDE_WINDOWS = [T-72 h, T+24 h] para os 8 trips
(só os 5 de dentro mudam algo no treino). Janela decidida com o usuário: as 72 h cobrem o precursor (a régua aceita
detecção até 48 h antes); as 24 h depois seguem a ideia dele de tratar ±24 h como "possível falha" (quase não
mudam nada, porque depois do trip a máquina costuma estar parada e o trecho parado já sai do treino). Todo o resto igual à produção: semente 42, corte em 01/07/2025, e percentil/
debounce FIXOS nos de produção (temp p97,5/d1, vib p95/d4, óleo p99,9/d12), porque o A4 mostrou que a reescolha
do AutoML derruba a vibração. Controle: a própria produção. Com a grade de um ponto, mesma semente e mesmo dado, o
treino é o da produção (o A4 reproduziu a produção nas sementes, e o c0 do C1 deu os mesmos episódios).

O QUE SE MEDE. (1) R1 na pipeline retreinada: trips em ALTA dentro (de 5) e fora (de 3), FP rebaixados.
(2) A pipeline de referência inteira (canal 4, voto >= 2, 45 min, 48 h): detecções, banda de 4 h, FP/mês, duty
dos canais. (3) Por trip, a antecedência.

LEITURA PRÉ-REGISTRADA (hipótese do mecanismo, pelo R1 dentro da amostra):
  CONFIRMA      >= 3 dos 5 trips de dentro em ALTA (hoje 0/5)  E  os 3 de fora continuam em ALTA.
  REFUTA        <= 1 dos 5 em ALTA.
  INCONCLUSIVO  2 dos 5, ou dentro confirma mas algum de fora cai.
Se CONFIRMA: a pipeline em produção deve ser treinada com essa exclusão, e o R1 vira a prioridade candidata para a
operação (ainda dependente de validação em dados novos). Isto testa o MECANISMO, não a generalização.

EXPECTATIVA REGISTRADA. CONFIRMA parcial: 3/5 dentro em ALTA (11/04 fica de fora, porque só 29% da janela estava no
treino de temp/vib e o efeito ali deve ser pequeno), fora 3/3. Na pipeline de referência, 8/8 mantido e FP/mês a
até ±10% de 2,88: os modelos ficam um pouco mais sensíveis, mas só perto dos trips.

Uso:
  python frente_ocsvm/a5_descontaminacao.py gera      # escreve as configs (commitar e dar push)
  python frente_ocsvm/a5_descontaminacao.py envia     # clona as 3 tasks de produção e enfileira
  python frente_ocsvm/a5_descontaminacao.py avalia    # baixa os artefatos e aplica a régua e o R1
"""
from __future__ import annotations
import argparse
import json

import numpy as np
import pandas as pd
import comum as C
from a4_variabilidade import RAIZ, REPO, BRANCH, CANAIS, canais_da_replica
from c1_retreino import FIXOS

CFG_DIR = C.AQUI / "a5_configs"
TASKS_JSON = C.AQUI / "a5_tasks.json"
HORAS_ANTES, HORAS_DEPOIS = 72, 24


def janelas() -> list[list[str]]:
    ft = C.trips(pd.DatetimeIndex(["2024-07-01", "2026-04-20"]))
    return [[str(t - pd.Timedelta(hours=HORAS_ANTES)), str(t + pd.Timedelta(hours=HORAS_DEPOIS))] for t in ft]


def mudancas(canal: str) -> dict:
    pct, deb = FIXOS[canal]
    return dict(TRAIN_EXCLUDE_WINDOWS=janelas(), AUTOML_THRESHOLD_PERCENTILES=[pct], AUTOML_DEBOUNCE_GRID=[deb])


def gera():
    CFG_DIR.mkdir(exist_ok=True)
    assert len(janelas()) == 8, janelas()
    for canal, (cfg, _, _) in CANAIS.items():
        d = json.loads((RAIZ / "configs/calibracao_v4_eq" / f"{cfg}.json").read_text(encoding="utf-8"))
        d.update(mudancas(canal))
        (CFG_DIR / f"a5_{canal}.json").write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(CANAIS)} configs em {CFG_DIR}; janelas: {janelas()}")


def envia():
    import subprocess
    from clearml import Task
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RAIZ, text=True).strip()
    remoto = subprocess.check_output(["git", "ls-remote", "cabiunas", f"refs/heads/{BRANCH}"], cwd=RAIZ, text=True)
    assert remoto.startswith(commit), f"faça push de {commit[:7]} para cabiunas/{BRANCH} antes"
    reg = json.loads(TASKS_JSON.read_text()) if TASKS_JSON.exists() else {}
    for canal, (cfg, tid, _) in CANAIS.items():
        nome = f"a5_{canal}"
        if nome in reg:
            continue
        t = Task.clone(source_task=tid, name=f"ocsvm-a5::{nome}::{cfg}")
        t.set_script(repository=REPO, branch=BRANCH, commit=commit)
        t.set_parameter("Args/config", f"frente_ocsvm/a5_configs/{nome}.json")
        for k, v in mudancas(canal).items():  # só o painel; o cfg vem do JSON de Args/config
            t.set_parameter(f"pipeline_config/{k}", str(v))
        Task.enqueue(t, queue_name="default")
        reg[nome] = t.id
        print(nome, t.id)
        TASKS_JSON.write_text(json.dumps(reg, indent=2) + "\n")


def avalia():
    reg = json.loads(TASKS_JSON.read_text()) if TASKS_JSON.exists() else {}
    reps = {"prod": {c: v[1] for c, v in CANAIS.items()}}
    if all(f"a5_{c}" in reg for c in CANAIS):
        reps["a5"] = {c: reg[f"a5_{c}"] for c in CANAIS}
    cat = C.catalogo()
    L, T = [], {}
    for rep, ids in reps.items():
        try:
            df = canais_da_replica(ids)
        except Exception as e:  # task não terminou ou falhou
            print(f"{rep}: indisponível ({type(e).__name__}: {e})")
            continue
        idx, op = df.index, df["operational_state"]
        ft = C.trips(idx)
        can = {k: df[k] for k in C.CANAIS[:3]}
        can["c4"] = C.canal_alarme(idx, cat.loc[cat.tag.isin(C.TAGS_C4), "t"], C.REF["janela_c4_h"])
        cls, m = C.avalia(C.decide(can, 2, C.REF["dur_min"], C.REF["refrat_h"]), op, ft)
        P = C.por_trip(cls, ft)
        n_ocsvm = sum(df[k].astype(int) for k in C.CANAIS[:3])
        cls["alta"] = [bool((n_ocsvm[(idx >= s) & (idx <= e)] >= 2).any()) for s, e in zip(cls.start, cls.end)]
        det = cls[(cls.classe == "deteccao") & cls.alta]
        alta = {pd.Timestamp(t) for t in det.falha_associada}
        fp = cls[cls.classe == "falso_positivo"]
        on = (op == "on").values
        T[rep] = dict(dentro=sum(t in alta for t in ft if t < C.SPLIT), fora=sum(t in alta for t in ft if t >= C.SPLIT))
        L.append(dict(rep=rep, det=int(P.deteccao.notna().sum()), fora=int(P.deteccao.notna()[P.amostra == "fora"].sum()),
                      banda=int(P.banda_4h.sum()), fp_mes=round(m["falso_positivo_por_mes"], 3),
                      duty=" ".join(f"{df[k][on].mean():.3f}" for k in C.CANAIS[:3]),
                      lead=" ".join("-" if np.isnan(x) else f"{x:.1f}" for x in P.lead_h),
                      R1_dentro=f"{T[rep]['dentro']}/5", R1_fora=f"{T[rep]['fora']}/3",
                      R1_FP_baixa=f"{int((~fp.alta).sum())}/{len(fp)}"))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(L).to_string(index=False))
    if "a5" in T:
        d, f = T["a5"]["dentro"], T["a5"]["fora"]
        v = "CONFIRMA" if d >= 3 and f == 3 else "REFUTA" if d <= 1 else "INCONCLUSIVO"
        print(f"\nR1 na pipeline descontaminada: dentro {d}/5, fora {f}/3 -> {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=["gera", "envia", "avalia"])
    a = ap.parse_args()
    {"gera": gera, "envia": envia, "avalia": avalia}[a.acao]()
