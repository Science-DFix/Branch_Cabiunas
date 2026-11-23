#!/usr/bin/env python3
"""J20 -- a regra de PLENA que o Thallys adotou (135039e): R60_24 nas 30 instalações do J34/J18/J19.

PRÉ-REGISTRO (23/11/2026, commitado antes de enviar)
Origem: retorno do Thallys ao J19 (23/11). Ele pôs no `estado_saida` (commit 135039e da feat/pacote-respostas-engenharia):
  PLENA = base do vb cheia há 24 h E 60 d de dado desde o início (no lote, da entrada; no motor, da carga). Deduziu do
  J19 que, com 60 d, nenhuma das 30 fica em PLENA divergindo (com R45_24 só sobra 01/01/2026, carga de 30 d, divergindo
  até 23/01) e pediu a medida.
Limite de acesso: o 135039e não está nos remotos que temos; a regra é implementada aqui pela descrição dele, sobre as
  posições do pacote, como as do J19:
      R60_24  posições no bloco de 24 h antes da linha >= 12.720 E linha >= início do dado + 60 d;
      R60     idem sem o atraso de 24 h (para separar o efeito de cada parte).
  "Início do dado" = início da carga (T0 - H d), ou T0 sem carga: o mesmo `inicio` do R45 do J19.
Pacote e dados: os do J18/J19 (scripts e733279 do Dataset bed2e62f; modelos, registro e CSV do f5e1b16b).
Desenho: `j19.partes_12` (refaz as 30 instalações e as regras do J19) e, sobre os mesmos parquet, R60 e R60_24.
  V1: o resumo do J18 sai igual (como no J19). V2: as regras do J19 saem iguais ao J19 (R24: 1 instalação, só alarme;
  R45_24: 1, só alarme, PLENA 45 / 38 / 15 / 0 / 0 d).
  Para R60 e R60_24: PLENA em (mediana, máx) por carga; instalações com divergência em PLENA (sinais ou alarme).
EXPECTATIVAS: R60_24 com 0 de 30 divergindo (~85%); PLENA sem carga ~60–61 d (mediana), carga 7 d ~53 d, 30 d ~30 d,
  45 d ~15 d, 60 d desde o início ou até +1 d (~85%). R60 também 0 de 30 (~60%; sem as 24 h podem voltar as ~12 h do vb
  quando a base enche depois dos 60 d).
CRITÉRIO: se R60_24 dá 0 de 30, a dedução do Thallys vira medida e a regra fica como está. Se não, relatar quais
  instalações e quanto, e a decisão volta para ele.
LIMITES: os do J18 (8 trips, 6 datas, Cica); 100 d por instalação (a PLENA de R60 cabe na janela).

RESULTADO (task 802694ac, Cica, 9,9 min, commit 73da7be). V1: igual ao J18 (PLENA 19,12/12,25/0/0/0 d; divergências
  3/4/2/0/0). V2: igual ao J19 (R24 e R45_24: 1 instalação, só alarme; R45_24 PLENA 45/38/15/0/0 d).
      regra     instalações com divergência em PLENA   PLENA em, mediana (máx), por carga 0 / 7 / 30 / 45 / 60 d
      R60_24    0 de 30 (sinais 0, alarme 0)          60 (60) / 53 (53) / 30 (30) / 15 (15) / 0 (0)
      R60       0 de 30 (sinais 0, alarme 0)          60 (60) / 53 (53) / 30 (30) / 15 (15) / 0 (0)
  Nenhuma instalação fica sem PLENA nos 100 d. As 24 h não mudam nada nas 30: aos 60 d a base já está cheia há bem
  mais de 24 h.
PLACAR: R60_24 0 de 30 SIM; dias SIM (60 / 53 / 30 / 15 / 0, exatos); R60 também 0 de 30 SIM.
LEITURA: a dedução do Thallys vira medida: com R60_24, nenhuma das 30 instalações fica em PLENA divergindo da
  referência, e com carga de 60 d a PLENA vale desde o 1º instante. Nas 30, quem manda é o critério dos 60 d; as 24 h
  ficam como salvaguarda (sem efeito medido aqui). A regra fica como ele pôs.
  _(registrado em 23/11/2026 11:30)_

Uso:
  python frente_thallys/j20_plena_r60_24.py --enqueue
"""
from clearml import Task  # noqa: F401  (antes do argparse: o worker injeta argumentos por patch)
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import j1_janela_entrada as j1          # noqa: E402
import j18_e733279_ram_plena as j18     # noqa: E402
import j19_plena_threads as j19         # noqa: E402

NOME = "frente_thallys | J20 PLENA: regra R60_24 do Thallys nas 30 instalações"
SAIDA = Path(tempfile.gettempdir()) / "j20_saida"
G = j1.G


def analisa_60(J):
    import numpy as np
    import pandas as pd
    ci = G["ci"]
    ref = pd.read_parquet(J.SAIDA / "ref_limpo.parquet")
    passo = pd.Timedelta(hours=ci.VB_PASSO_H)
    L = []
    for t0_s in J.INSTALA:
        for H in J.CARGAS:
            r = pd.read_parquet(J.SAIDA / f"inst_{t0_s}_H{H}.parquet")
            T0 = J.T(t0_s); fim = T0 + J.D("100D"); inicio = T0 - J.D(f"{H}D")
            banco = G["df"].loc[inicio:] if H else G["df"].loc[T0:]
            est = ci.instantes_estaveis(banco.loc[:fim])
            Rr = ref.reindex(r.index)

            def cheia(atraso_h):
                q = r.index - pd.Timedelta(hours=atraso_h)
                return est.searchsorted(ci.EPOCA + ((q - ci.EPOCA) // passo) * passo, side="left") >= ci.VB_POSICOES_PLENA
            mem60 = r.index >= inicio + J.D("60D")
            pl = dict(R60=cheia(0) & mem60, R60_24=cheia(24) & mem60)
            dsin = np.zeros(len(r), bool)
            for s in ("t", "p", "sp", "vb"):
                a, b = r[s].to_numpy(float), Rr[s].to_numpy(float)
                ok = ~np.isnan(a) & ~np.isnan(b)
                dsin |= (np.isnan(a) != np.isnan(b)) | (ok & (np.abs(np.where(ok, a - b, 0)) > 1e-9))
            dal = r["is_anomaly"].astype(bool).to_numpy() != Rr["is_anomaly"].fillna(False).astype(bool).to_numpy()
            linha = dict(T0=t0_s, H=H)
            for k, m in pl.items():
                linha[f"{k}_plena_d"] = None if not m.any() else round((r.index[np.flatnonzero(m)[0]] - T0).total_seconds() / 86400, 2)
                linha[f"{k}_div_sinais"] = int((dsin & m).sum()); linha[f"{k}_div_alarme"] = int((dal & m).sum())
            L.append(linha)
    T = pd.DataFrame(L); T.to_csv(SAIDA / "j20_regras.csv", index=False)
    resumo = {}
    for k in ("R60", "R60_24"):
        div = (T[f"{k}_div_sinais"] > 0) | (T[f"{k}_div_alarme"] > 0)
        resumo[k] = dict(
            inst_div_qualquer=int(div.sum()), inst_div_sinais=int((T[f"{k}_div_sinais"] > 0).sum()),
            inst_div_alarme=int((T[f"{k}_div_alarme"] > 0).sum()),
            quais=T.loc[div, ["T0", "H", f"{k}_div_sinais", f"{k}_div_alarme"]].to_dict("records"),
            sem_plena=int(T[f"{k}_plena_d"].isna().sum()),
            plena_d={str(H): dict(med=round(float(x[f"{k}_plena_d"].median()), 2), max=round(float(x[f"{k}_plena_d"].max()), 2))
                     for H, x in T.groupby("H")})
    return resumo


def executa(local: bool, insumos: str | None, scripts: str, teste: bool, procs: int):
    SAIDA.mkdir(parents=True, exist_ok=True); t00 = time.time()
    pk, a97, _ = j1.monta(local, insumos)
    scripts_e17 = (pk / "scripts").resolve()
    if local:
        novo = Path(os.environ["FT_SCRIPTS_J17"])
    else:
        from clearml import Dataset
        novo = Path(Dataset.get(dataset_id=scripts).get_local_copy()) / "scripts"
    (pk / "scripts").unlink(); shutil.copytree(novo, pk / "scripts")
    i = next(k for k, c in enumerate(sys.path) if c and Path(c).resolve() == scripts_e17)
    sys.path[i] = str(pk / "scripts")
    import cabiunas_inference as ci
    assert Path(ci.__file__).resolve().parent == (pk / "scripts").resolve() and hasattr(ci, "estado_saida"), ci.__file__
    res = dict(ambiente=j1.ambiente()); print("ambiente:", res["ambiente"], flush=True)
    j19.SAIDA = SAIDA
    P = j19.partes_12(pk, a97, teste, procs)
    import j34_estado_robustez as J
    v2 = {k: P["regras"][k] for k in ("R24", "R45_24")}
    res.update(v1_igual_j18=P["v1_igual_j18"], v1=P["v1"], v2_regras_j19=v2, regras_60=analisa_60(J))
    print(json.dumps({k: v for k, v in res.items() if k != "ambiente"}, indent=1, ensure_ascii=False, default=str), flush=True)
    res["duracao_min"] = round((time.time() - t00) / 60, 1)
    (SAIDA / "j20_resultado.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    from clearml import Task as Tk
    tk = Tk.current_task()
    if tk is not None:
        for f in ("j20_resultado.json", "j20_regras.csv", "j19_regras.csv"):
            tk.upload_artifact(Path(f).stem, artifact_object=str(SAIDA / f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enqueue", action="store_true"); ap.add_argument("--local", action="store_true")
    ap.add_argument("--teste", action="store_true"); ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--insumos", default=j1.DS_INSUMOS); ap.add_argument("--scripts", default=j18.DS_SCRIPTS)
    a = ap.parse_args()
    if a.enqueue:
        t = Task.init(project_name="TesteMLCab", task_name=NOME)
        t.set_base_docker("python:3.11-slim", docker_setup_bash_script=["apt-get update -qq", "apt-get install -y -qq libgomp1"])
        t.set_packages(j1.PACOTES)
        t.set_parameters({"General/insumos": a.insumos, "General/scripts": a.scripts, "General/procs": str(a.procs)})
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=AQUI, text=True).strip()
        t.set_script(repository="git@github.com:Science-DFix/Branch_Cabiunas.git", branch="exp/ocsvm-validacao", commit=commit)
        t.execute_remotely(queue_name="default")
    if os.environ.get("CLEARML_TASK_ID"):
        tk = Task.init(project_name="TesteMLCab", task_name=NOME)
        a.insumos = tk.get_parameter("General/insumos") or a.insumos
        a.scripts = tk.get_parameter("General/scripts") or a.scripts
        a.procs = int(tk.get_parameter("General/procs") or a.procs)
    executa(a.local, a.insumos, a.scripts, a.teste, a.procs)


if __name__ == "__main__":
    main()
