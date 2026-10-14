#!/usr/bin/env python3
"""B5 -- os episódios de FP antecedem alarmes reais do catálogo mais do que o acaso? (uma 4ª classe na régua)

PRÉ-REGISTRADO EM 14/10/2026, ANTES DE TREINAR E ANTES DE OLHAR OS ALARMES DEPOIS DOS FP. B1, B2, C1 e B3
mostraram que cortar FP em tempo real perde trip. A ideia do usuário: aceitar os FP e rever a CONTAGEM. A régua
tem 3 classes (scoring.py:198): detecção (antecede trip em 48 h), inconclusivo (antecede parada >= 2 h em 48 h
depois do fim) e FP (o resto). Se um FP antecede um alarme real que ainda não estava ativo, ele não é falso: o
modelo antecipou um alarme. Isso não muda o detector, só a leitura.

PROCESSO INTEIRO DO ZERO (pedido do usuário): os 3 OCSVM são retreinados como tasks novas no Cica, com os configs
de produção SEM mudança (configs/calibracao_v4_eq, AutoML livre, semente 42, corte 01/07/2025) e o src/ atual.
O B5 é medido nos artefatos NOVOS; a produção entra só como referência. Expectativa da reprodução: mesmos
hiperparâmetros (temp p97,5/d1, vib p95/d4, óleo p99,9/d12) e os mesmos episódios (8/8, 42 FP), com folga de
±1 episódio (ruído visto no A4).

O TESTE. Para cada episódio de FP (início s), "alarme novo" = alguma tag do recorte vai a ACT em (s, s + 48 h] e
essa MESMA tag não foi a ACT em [s - 24 h, s] (tira o chattering e os alarmes que o canal 4 já viu). Três recortes,
fixados antes:
  todas    todas as 47 tags do catálogo
  critico  só linhas com Prioridade HIGH ou LOLO
  sem_c4   todas menos as 5 tags do canal 4 (circularidade)
Acaso: a mesma regra em instantes de 5 em 5 min que poderiam ser FP -- máquina "on", sem trip em [t, t + 48 h] e
sem início de parada >= 2 h em [t, t + 48 h]. Enriquecimento = taxa nos FP / taxa no acaso; p = binomial
unilateral (n = nº de FP, p0 = taxa no acaso).

CRITÉRIO (por recorte):
  VÁLIDO     enriquecimento >= 1,5  E  p < 0,05/3 (Bonferroni nos 3 recortes).
  NULO       qualquer outro caso.
Se algum recorte for VÁLIDO, os FP que o cumprem viram "alarme antecipado" e se relata o FP/mês restante (se mais
de um for válido, vale o mais específico: critico > sem_c4 > todas). Os FP que sobram viram lista para a equipe.
Se nenhum for válido, os FP não antecipam alarme além do acaso, e a contagem de 2,9 FP/mês fica como está.

EXPECTATIVA REGISTRADA. NULO nos 3. O catálogo tem ~2,9 ativações por dia no período, então "todas" e "sem_c4"
saturam (acaso >= 60%) e não há espaço para 1,5x; "critico" pode chegar perto de 1,5x, mas com 42 FP não passa
no Bonferroni.

Uso:
  python frente_ocsvm/b5_alarme_antecipado.py envia --so temp   # clona a task de produção do canal e enfileira
  python frente_ocsvm/b5_alarme_antecipado.py avalia            # baixa os artefatos novos e aplica o teste
"""
from __future__ import annotations
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest
import comum as C
from a4_variabilidade import RAIZ, REPO, BRANCH, CANAIS, canais_da_replica, hiper

TASKS_JSON = C.AQUI / "b5_tasks.json"
ANTES_H, DEPOIS_H = 24, 48
H = np.timedelta64(3600, "s")


def envia(so: str | None):
    import subprocess
    from clearml import Task
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RAIZ, text=True).strip()
    remoto = subprocess.check_output(["git", "ls-remote", "cabiunas", f"refs/heads/{BRANCH}"], cwd=RAIZ, text=True)
    assert remoto.startswith(commit), f"faça push de {commit[:7]} para cabiunas/{BRANCH} antes"
    reg = json.loads(TASKS_JSON.read_text()) if TASKS_JSON.exists() else {}
    for canal, (cfg, tid, _) in CANAIS.items():
        nome = f"b5_{canal}"
        if (so and so != canal) or nome in reg:
            continue
        t = Task.clone(source_task=tid, name=f"ocsvm-b5::{nome}::{cfg}")
        t.set_script(repository=REPO, branch=BRANCH, commit=commit)
        t.set_parameter("Args/config", f"configs/calibracao_v4_eq/{cfg}.json")  # config de produção, sem mudança
        Task.enqueue(t, queue_name="default")
        reg[nome] = t.id
        print(nome, t.id)
        TASKS_JSON.write_text(json.dumps(reg, indent=2) + "\n")


def alarmes() -> pd.DataFrame:
    a = pd.read_csv(C.DADOS / "alarmes_selecionados_turbina_a.csv")
    a["t"] = pd.to_datetime(a["Data da Ocorrência"], errors="coerce")
    a = a[a["Status"].astype(str).str.startswith("ACT")].dropna(subset=["t"])
    return a.rename(columns={"Tag Alarme": "tag"})[["t", "tag", "Prioridade"]]


def alarme_novo(x: np.ndarray, a: pd.DataFrame) -> np.ndarray:
    """Para cada instante: alguma tag vai a ACT em (x, x + 48 h] sem ter ido a ACT em [x - 24 h, x]."""
    out = np.zeros(len(x), bool)
    for _, g in a.groupby("tag"):
        t = np.sort(g.t.values.astype("datetime64[ns]"))
        pos = np.searchsorted(t, x, side="right")  # 1º alarme > x
        prox = np.zeros(len(x), bool)
        ok = pos < len(t)
        prox[ok] = (t[pos[ok]] - x[ok]) <= DEPOIS_H * H
        ant = np.zeros(len(x), bool)
        ok = pos > 0
        ant[ok] = (x[ok] - t[pos[ok] - 1]) <= ANTES_H * H
        out |= prox & ~ant
    return out


def instantes_acaso(op: pd.Series, ft: pd.Series) -> np.ndarray:
    """Instantes de 5 em 5 min que poderiam ser FP: on, sem trip nem início de parada >= 2 h nas 48 h seguintes."""
    off = (op != "on").astype(int)
    d = off.diff().fillna(off.iloc[0])
    ini, fim = op.index[d == 1], op.index[d == -1]
    if off.iloc[0]:
        ini = ini.insert(0, op.index[0])
    if len(ini) > len(fim):
        fim = fim.append(pd.DatetimeIndex([op.index[-1]]))
    paradas = np.sort(np.array([s for s, e in zip(ini, fim) if e - s >= pd.Timedelta(hours=2)], "datetime64[ns]"))
    ev = np.sort(np.concatenate([paradas, ft.values.astype("datetime64[ns]")]))
    s = op[op == "on"].resample("5min").first().dropna().index.values.astype("datetime64[ns]")
    pos = np.searchsorted(ev, s, side="left")
    livre = np.ones(len(s), bool)
    ok = pos < len(ev)
    livre[ok] = (ev[pos[ok]] - s[ok]) > DEPOIS_H * H
    return s[livre]


def avalia():
    reg = json.loads(TASKS_JSON.read_text()) if TASKS_JSON.exists() else {}
    reps = {"prod": {c: v[1] for c, v in CANAIS.items()}}
    if all(f"b5_{c}" in reg for c in CANAIS):
        reps["b5"] = {c: reg[f"b5_{c}"] for c in CANAIS}
    else:
        print("b5: tasks incompletas no registro; só a produção")
    a = alarmes()
    rec = {"todas": a, "critico": a[a.Prioridade.isin(["HIGH", "LOLO"])], "sem_c4": a[~a.tag.isin(C.TAGS_C4)]}
    cat = C.catalogo()
    pd.set_option("display.width", 200)
    for rep, ids in reps.items():
        df = canais_da_replica(ids)
        idx, op = df.index, df["operational_state"]
        ft = C.trips(idx)
        can = {k: df[k] for k in C.CANAIS[:3]}
        can["c4"] = C.canal_alarme(idx, cat.loc[cat.tag.isin(C.TAGS_C4), "t"], C.REF["janela_c4_h"])
        cls, m = C.avalia(C.decide(can, 2, C.REF["dur_min"], C.REF["refrat_h"]), op, ft)
        P = C.por_trip(cls, ft)
        hp = " ".join(hiper(ids[c], CANAIS[c][2]) for c in CANAIS) if rep == "b5" else "produção"
        print(f"\n=== {rep} ({hp}): {int(P.deteccao.notna().sum())}/8, banda {int(P.banda_4h.sum())}/8, "
              f"{m['n_episodios_falso_positivo']} FP ({m['falso_positivo_por_mes']:.3f}/mês), "
              f"leads {' '.join(f'{x:.1f}' for x in P.lead_h)}")
        fp = cls[cls.classe == "falso_positivo"]
        xs = fp.start.values.astype("datetime64[ns]")
        acaso = instantes_acaso(op, ft)
        meses = m["n_episodios_falso_positivo"] / m["falso_positivo_por_mes"]
        L, marca = [], {}
        for nome, r in rec.items():
            k_fp = alarme_novo(xs, r)
            p0 = alarme_novo(acaso, r).mean()
            k = int(k_fp.sum())
            enr = (k / len(fp)) / p0 if p0 > 0 else np.nan
            p = binomtest(k, len(fp), p0, alternative="greater").pvalue
            ok = enr >= 1.5 and p < 0.05 / 3
            marca[nome] = k_fp
            L.append(dict(recorte=nome, FP_com_alarme=f"{k}/{len(fp)}", taxa_FP=round(k / len(fp), 3),
                          taxa_acaso=round(p0, 3), enriq=round(enr, 2), p=round(p, 4),
                          veredito="VÁLIDO" if ok else "NULO", FP_mes_restante=round((len(fp) - k) / meses, 3)))
        T = pd.DataFrame(L)
        print(f"acaso: {len(acaso)} instantes elegíveis\n{T.to_string(index=False)}")
        validos = [n for n in ("critico", "sem_c4", "todas") if T.set_index("recorte").loc[n, "veredito"] == "VÁLIDO"]
        if validos:
            print(f"-> recorte vigente: {validos[0]}; FP restantes:")
            print(fp.loc[~marca[validos[0]], ["start", "end"]].to_string(index=False))
        else:
            print("-> NULO nos 3 recortes: os FP não antecipam alarme além do acaso")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=["envia", "avalia"])
    ap.add_argument("--so", choices=list(CANAIS))
    a = ap.parse_args()
    {"envia": lambda: envia(a.so), "avalia": avalia}[a.acao]()
