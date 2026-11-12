#!/usr/bin/env python3
"""J12 -- qualidade da detecção com a base do vb parcial (o estado AQUECENDO pode emitir alarme?).

PRÉ-REGISTRO (12/11/2026, commitado antes de enviar)
Origem: a resposta à engenharia propõe o estado AQUECENDO (base do vb entre 3.000 e 12.720 posições estáveis) e o
  INDISPONÍVEL abaixo de 3.000 (sem vb: 2/8, J11). O pacote já deixa a sonda opinar com 25% da base
  (VB_FRACAO_BASE_MIN = 0,25 de 12.000 = 3.000 amostras), mas ninguém mediu a detecção com a base de fato curta: o
  comentário do pacote ("de 0,25 a 1,00 o alarme é idêntico") foi medido sobre o histórico inteiro, onde a base está
  sempre cheia. Se a detecção se mantiver com base parcial, o AQUECENDO pode emitir e a instalação nova fica útil em
  ~100 h estáveis em vez de ~64 d.
Pergunta: com a base do vb limitada a N amostras estáveis em TODO o período (pior caso do AQUECENDO: nunca enche),
  quantos trips o detector pega, com quantos FP, e em quantos instantes o alarme difere do controle?
Desenho: controle publicado (`sinais_causais` + `prever`, régua do Thallys = j1.regua, como J11 parte B), com
  ci.VB_HORAS_BASE = N / 30 e o limiar de opinião mantido em 3.000 amostras absolutas
  (ci.VB_FRACAO_BASE_MIN = 3.000 / N). N em {3.000, 4.500, 6.000, 9.000, 12.000}; N = 12.000 é o controle (400 h).
  Nada mais muda (guarda 24 h, passo 6 h, exclusão dos trips, bundles).
EXPECTATIVAS: N = 12.000 reproduz 7/8 FP 9 (~95%). N = 3.000 fica entre 5/8 e 7/8 com FP entre 9 e 16 (~60%).
  Detecção não melhora ao encurtar a base (~70%).
CRITÉRIO DE DECISÃO (para o documento): o menor N com 7/8 e FP <= 11 vira o limiar a partir do qual o AQUECENDO
  emite alarme. Se for 3.000, o AQUECENDO emite sempre; se nenhum N < 12.000 passar, o AQUECENDO não emite.
LIMITES: pior caso (base parcial o período todo); numa instalação real a base cresce. Um equipamento, 8 trips.

Uso:
  python frente_thallys/j12_base_vb_parcial.py --enqueue         # uma task na fila default
"""
from clearml import Task  # noqa: F401  (antes do argparse: o worker injeta argumentos por patch)
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import j1_janela_entrada as j1      # noqa: E402

NOME = "frente_thallys | J12 base do vb parcial (AQUECENDO emite?)"
SAIDA = Path(tempfile.gettempdir()) / "j12_saida"
G = j1.G
NS = [12000, 9000, 6000, 4500, 3000]
OPINA = 3000                         # limiar absoluto de produção: int(12.000 * 0,25)


def mede(teste: bool):
    import numpy as np
    import pandas as pd
    import contextlib
    import io
    from nv.controle_sinais import sinais_causais
    ci = G["ci"]
    horas0, frac0 = ci.VB_HORAS_BASE, ci.VB_FRACAO_BASE_MIN
    assert int(horas0 * ci.POR_HORA) == 12000 and int(12000 * frac0) == OPINA, (horas0, frac0)
    out, alarme = {}, {}
    for n in (NS[:1] + NS[-1:] if teste else NS):
        ci.VB_HORAS_BASE, ci.VB_FRACAO_BASE_MIN = n / ci.POR_HORA, OPINA / n
        t0 = time.time()
        S = sinais_causais(pd.Timestamp("2025-01-08 07:34", tz="UTC"), cache=None, extra=G["extra"])
        S["mask"] = S["mask"].astype(bool); S["reset"] = S["reset"].astype(bool)
        with contextlib.redirect_stdout(io.StringIO()):
            r = ci.prever({"detector": G["det"]}, S)
        al = r["is_anomaly"].astype(bool)
        out[n] = j1.regua(al, r["vigiado"].astype(bool))
        out[n]["vb_nan_pct"] = round(100 * float(S["vb"].isna().mean()), 2)
        alarme[n] = al
        print(f"   N {n:6d} ({n / ci.POR_HORA:.0f} h): {out[n]['det']}/8 FP {out[n]['fp']} ({out[n]['fp_mes']}/mês), "
              f"início {out[n]['inicio']}, banda {out[n]['banda']}, vb vazio {out[n]['vb_nan_pct']}% "
              f"({time.time() - t0:.0f} s)", flush=True)
    ci.VB_HORAS_BASE, ci.VB_FRACAO_BASE_MIN = horas0, frac0
    ref = alarme[12000]
    for n in alarme:
        a = alarme[n].reindex(ref.index, fill_value=False)
        out[n]["alarme_dif_pct"] = round(100 * float((a != ref).mean()), 3)
    passa = [n for n in sorted(out) if out[n]["det"] == 7 and out[n]["fp"] <= 11]
    out["menor_N_que_passa"] = min(passa) if passa else None
    print("   alarme diferente do controle (%):", {n: out[n]["alarme_dif_pct"] for n in alarme}, flush=True)
    print("   menor N com 7/8 e FP <= 11:", out["menor_N_que_passa"], flush=True)
    return out


def executa(local: bool, insumos: str | None, teste: bool):
    SAIDA.mkdir(parents=True, exist_ok=True); t00 = time.time()
    pk, a97, _ = j1.monta(local, insumos)
    res = dict(ambiente=j1.ambiente()); print("ambiente:", res["ambiente"], flush=True)
    j1.prepara(pk, a97, teste)
    res["N"] = mede(teste)
    res["duracao_min"] = round((time.time() - t00) / 60, 1)
    (SAIDA / "j12_resultado.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    from clearml import Task as Tk
    tk = Tk.current_task()
    if tk is not None:
        tk.upload_artifact("j12_resultado", artifact_object=str(SAIDA / "j12_resultado.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enqueue", action="store_true")
    ap.add_argument("--local", action="store_true"); ap.add_argument("--teste", action="store_true")
    ap.add_argument("--insumos", default=j1.DS_INSUMOS)
    a = ap.parse_args()
    if a.enqueue:
        t = Task.init(project_name="TesteMLCab", task_name=NOME)
        t.set_base_docker("python:3.11-slim", docker_setup_bash_script=["apt-get update -qq", "apt-get install -y -qq libgomp1"])
        t.set_packages(j1.PACOTES); t.set_parameters({"General/insumos": a.insumos})
        # o padrão do ClearML pega o remoto `origin` (SIMPred), onde este commit não existe: fixa o Branch_Cabiunas
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=AQUI, text=True).strip()
        t.set_script(repository="git@github.com:Science-DFix/Branch_Cabiunas.git", branch="exp/ocsvm-validacao", commit=commit)
        t.execute_remotely(queue_name="default")
    if os.environ.get("CLEARML_TASK_ID"):
        tk = Task.init(project_name="TesteMLCab", task_name=NOME)
        a.insumos = tk.get_parameter("General/insumos") or a.insumos
    executa(a.local, a.insumos, a.teste)


if __name__ == "__main__":
    main()
