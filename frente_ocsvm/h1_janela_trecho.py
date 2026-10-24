#!/usr/bin/env python3
"""H1 -- retreinar os 3 OCSVM com janelas que não atravessam a parada (DERIVED_RESET_AT_STATE_CHANGE). Cica.

PRÉ-REGISTRO (24/10/2026, antes de enviar qualquer task)
Pergunta: o G1 mostrou que as janelas móveis das features (4 h e 24 h) contam linhas e atravessam a parada. A 2 h
da partida, a janela de 24 h está 92% parada nos FP de partida (27% nas partidas sem alerta), e o |z| dessas escalas
vai a 5-33x o do regime. Calcular as janelas só dentro de cada trecho contínuo de estado operacional corta FP sem
perder trip? É uma correção da feature, não uma supressão: nada é escondido depois da partida.

TRATAMENTO (único braço). As configs de produção (configs/calibracao_v4_eq/test_grupo_exp33/34/38) mais:
    DERIVED_RESET_AT_STATE_CHANGE = true
Tudo o mais fica igual: AutoML livre (mesma grade de percentil x debounce), semente 42, split 01/07/2025, 50 mil
amostras, portões. Dentro de cada trecho, cada feature é a mesma conta da produção, feita como se o trecho fosse o
começo da série. Mediana e desvio usam o que houver; a tendência de w linhas fica 0 até w linhas; a textura fica 0
até w/4 linhas. 1 task por vez (temperatura, vibração, óleo), só com a fila sem tasks de outros.

AVALIAÇÃO. A camada de decisão da pipeline final com E5 (`f1_congelado.decide`: voto >= 2 de 4, canal 4 = 5 tags
em 24 h, 45 min, 48 h, falha de comunicação) e a régua de sempre. A produção avaliada do mesmo jeito dá 8/8,
41 FP e 2,816 FP/mês; fora da amostra, 22 FP e 2,752.
CRITÉRIO (o da frente, o mesmo do E6): PASSA = 8/8 E 3/3 fora E FP/mês fora <= 0,75 x 2,877 = 2,158.
Se passar, confirmar com as sementes 43 e 44 (3 tasks cada, uma por vez); vale se as duas também passarem.
Secundários: FP/mês no período inteiro; FP a <= 2 h da partida (produção: 15); antecedência de cada trip, com
atenção a T4, T6 e T7, que começam na partida; percentil e debounce escolhidos pelo AutoML em cada canal (o A4 mostrou
que a reescolha pode derrubar a vibração).
EXPECTATIVA REGISTRADA: os FP de partida caem para metade ou menos (<= 8 de 15). O FP/mês fora cai 15-30%, então o
veredito fica perto do limite; prevejo NÃO PASSA por pouco (fora entre 2,16 e 2,45). Mantém 8/8; se perder um trip,
será o T7 (09/12/2025, detectado ~0 h depois da partida).

Uso:
  python frente_ocsvm/h1_janela_trecho.py gera              # escreve as configs (commit + push antes de enviar)
  python frente_ocsvm/h1_janela_trecho.py envia --so temp   # clona a task de produção e enfileira UM canal
  python frente_ocsvm/h1_janela_trecho.py avalia            # baixa os artefatos e aplica a régua com E5
"""
from __future__ import annotations
import argparse
import json
import subprocess

import pandas as pd

import comum as C
import f1_congelado as F
from a4_variabilidade import RAIZ, REPO, BRANCH, CANAIS
from e4_adia_partida import horas_desde_partida
from src.cnn1d_ae.config import PipelineConfig, update_cfg_from_dict

CFG_DIR = C.AQUI / "h1_configs"
TASKS_JSON = C.AQUI / "h1_tasks.json"
SEMENTES = {"s42": 42, "s43": 43, "s44": 44}
NOMES_F = {"temp": "temperatura", "vib": "vibracao", "oleo": "oleo_pressao"}  # canal aqui -> canal em F.decide
LIM_FORA = 0.75 * 2.877


def mudancas(rep: str) -> dict:
    return dict(DERIVED_RESET_AT_STATE_CHANGE=True, RANDOM_SEED=SEMENTES[rep])


def gera():
    CFG_DIR.mkdir(exist_ok=True)
    for rep in SEMENTES:
        for canal, (cfg, _, _) in CANAIS.items():
            d = json.loads((RAIZ / "configs/calibracao_v4_eq" / f"{cfg}.json").read_text(encoding="utf-8"))
            assert "RANDOM_SEED" in d
            d.update(mudancas(rep))
            # o loader ignora chave desconhecida em silêncio: confere que a chave nova chega ao PipelineConfig
            assert update_cfg_from_dict(PipelineConfig(), d).DERIVED_RESET_AT_STATE_CHANGE is True
            (CFG_DIR / f"{rep}_{canal}.json").write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n",
                                                         encoding="utf-8")
    print(f"{len(SEMENTES) * len(CANAIS)} configs em {CFG_DIR}")


def envia(so: str):
    from clearml import Task
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RAIZ, text=True).strip()
    remoto = subprocess.check_output(["git", "ls-remote", "cabiunas", f"refs/heads/{BRANCH}"], cwd=RAIZ, text=True)
    assert remoto.startswith(commit), f"faça push de {commit[:7]} para cabiunas/{BRANCH} antes"
    reg = json.loads(TASKS_JSON.read_text()) if TASKS_JSON.exists() else {}
    nome = so if "_" in so else f"s42_{so}"
    rep, canal = nome.split("_")
    assert nome not in reg, f"{nome} já enviado: {reg[nome]}"
    cfg, tid, _ = CANAIS[canal]
    t = Task.clone(source_task=tid, name=f"ocsvm-h1::{nome}::{cfg}")
    t.set_script(repository=REPO, branch=BRANCH, commit=commit)
    t.set_parameter("Args/config", f"frente_ocsvm/h1_configs/{nome}.json")
    for k, v in mudancas(rep).items():  # só o painel; o cfg vem do JSON de Args/config
        t.set_parameter(f"pipeline_config/{k}", v)
    Task.enqueue(t, queue_name="default")
    reg[nome] = t.id
    TASKS_JSON.write_text(json.dumps(reg, indent=2) + "\n")
    print(nome, t.id)


def pontos_da_replica(ids: dict) -> dict:
    from clearml import Task
    out = {}
    for canal, (_, _, art) in CANAIS.items():
        p = Task.get_task(task_id=ids[canal]).artifacts[f"{art}/csv/point_anomalies_all.csv"].get_local_copy()
        out[NOMES_F[canal]] = pd.read_csv(p, index_col=0, parse_dates=True, low_memory=False)
    return out


def avalia():
    from clearml import Dataset
    from a4_variabilidade import hiper
    from pathlib import Path
    reg = json.loads(TASKS_JSON.read_text()) if TASKS_JSON.exists() else {}
    reps = {"prod": {c: v[1] for c, v in CANAIS.items()}}
    reps.update({r: {c: reg[f"{r}_{c}"] for c in CANAIS} for r in SEMENTES if all(f"{r}_{c}" in reg for c in CANAIS)})
    root = Path(Dataset.get(dataset_id=F.P.ALARM_CATALOG_DATASET_ID).get_local_copy())
    cat = F.le_catalogo(root / F.P.ALARM_CATALOG_FILE)
    txt = F.le_texto(str(root / F.P.SENSORS_FILE))
    ft = F.le_trips(C.DADOS / "alarmes_francisco_falhas.csv")
    L = []
    for rep, ids in reps.items():
        try:
            pontos = pontos_da_replica(ids)
        except Exception as e:  # task não terminou ou falhou
            print(f"{rep}: indisponível ({type(e).__name__}: {e})")
            continue
        _, final, op, cls, met = F.decide(pontos, cat, txt, ft)
        ftp = ft[(ft >= op.index.min()) & (ft <= op.index.max())].reset_index(drop=True)
        P = C.por_trip(cls, ftp)
        meses_fora = C.compute_operational_period_days(pd.DataFrame({"operational_state": op[op.index >= C.SPLIT]})) / 30.4375
        fp = cls[cls.classe == "falso_positivo"]
        hp = pd.Series(horas_desde_partida(op), index=op.index)
        fp_fora = int((fp.start >= C.SPLIT).sum())
        L.append(dict(rep=rep, det=met["falhas_detectadas"], fora=int((P.deteccao.notna() & (P.amostra == "fora")).sum()),
                      FP=len(fp), fp_mes=round(met["falso_positivo_por_mes"], 3), FP_fora=fp_fora,
                      fp_mes_fora=round(fp_fora / meses_fora, 3), FP_partida=int((hp.reindex(fp.start).values <= 2).sum()),
                      inc=met["n_episodios_inconclusivo"], lead=" ".join(f"{x:.1f}" for x in P.lead_h),
                      hiper=" ".join(hiper(ids[c], CANAIS[c][2]) for c in CANAIS)))
    T = pd.DataFrame(L)
    pd.set_option("display.width", 220)
    print(T.to_string(index=False))
    for r in T.itertuples():
        if r.rep != "prod":
            ok = r.det == 8 and r.fora == 3 and r.fp_mes_fora <= LIM_FORA
            print(f"{r.rep}: {r.det}/8, fora {r.fora}/3, FP/mês fora {r.fp_mes_fora} (limite {LIM_FORA:.3f}) -> "
                  f"{'PASSA' if ok else 'NÃO'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=["gera", "envia", "avalia"])
    ap.add_argument("--so")
    a = ap.parse_args()
    {"gera": gera, "envia": lambda: envia(a.so), "avalia": avalia}[a.acao]()
