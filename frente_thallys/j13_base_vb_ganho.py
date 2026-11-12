#!/usr/bin/env python3
"""J13 -- o 8/8 com base do vb curta (J12) é ganho real ou efeito de mais horas de alarme?

PRÉ-REGISTRO (12/11/2026, commitado antes de enviar)
Origem: no J12, a base do vb limitada a 3.000-9.000 amostras deu 8/8 (ganha o trip de 04/11/2025) com FP 15-19 e
  32-61% mais horas de alarme (controle 7/8 FP 9, 1.457 h). Três motivos para desconfiar: (1) mais horas acesas cobrem
  trips por acaso (EXP 50 do Thallys: vb com Ridge foi o único 8/8, com o dobro das horas); (2) o 04/11 é o trip que a
  cadeia do refratário custa (J34: um buraco de 7 d também o pega); (3) 5 bases contra os mesmos 8 trips.
Perguntas:
  A. O 04/11 vem do vb local antes do trip ou de uma mudança na cadeia de episódios?
     Controle com o vb trocado pelo da base de 3.000 SÓ em [trip - D, trip + 2 d], D em {2, 10, 30} d; e o inverso
     (base de 3.000 com o vb do controle na mesma janela). Mede se o 04/11 é pego.
  B. Com as MESMAS horas de alarme do controle, a base curta ainda dá 8/8?
     Para N em {3.000, 6.000}: multiplica k_nivel_a e k_nivel_b do vb por f (grade 1,00 a 2,00, passo 0,05) e escolhe,
     SEM olhar os trips, o f cujas horas de alarme ficam mais perto das do controle. Régua nesse f. Curva inteira
     (f, det, FP, horas) registrada.
Desenho: controle publicado (`sinais_causais` + `prever`, régua j1.regua), como J11/J12; base truncada como J12
  (ci.VB_HORAS_BASE = N / 30, limiar de opinião fixo em 3.000).
EXPECTATIVAS: A. troca de 2 d não pega o 04/11 e a de 30 d pega (~55%: o efeito é de cadeia). B. com horas iguais,
  N 3.000 e 6.000 voltam a 7/8 (~65%).
CRITÉRIO: ganho real = em B, 8/8 com FP <= 9 em pelo menos um N. Sem isso, o 8/8 do J12 é efeito de mais horas acesas,
  e a base de 400 h fica.
LIMITES: um equipamento, 8 trips; o f é escolhido pelas horas no mesmo período em que se mede a régua (sem rótulos,
  mas não fora da amostra).

Uso:
  python frente_thallys/j13_base_vb_ganho.py --enqueue         # uma task na fila default
"""
from clearml import Task  # noqa: F401  (antes do argparse: o worker injeta argumentos por patch)
import argparse
import copy
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

NOME = "frente_thallys | J13 base do vb curta: ganho real ou horas?"
SAIDA = Path(tempfile.gettempdir()) / "j13_saida"
G = j1.G
OPINA = 3000
TRIP = "2025-11-04 06:22"


def sinais(n):
    import pandas as pd
    from nv.controle_sinais import sinais_causais
    ci = G["ci"]
    h0, f0 = ci.VB_HORAS_BASE, ci.VB_FRACAO_BASE_MIN
    ci.VB_HORAS_BASE, ci.VB_FRACAO_BASE_MIN = n / ci.POR_HORA, OPINA / n
    try:
        S = sinais_causais(pd.Timestamp("2025-01-08 07:34", tz="UTC"), cache=None, extra=G["extra"])
    finally:
        ci.VB_HORAS_BASE, ci.VB_FRACAO_BASE_MIN = h0, f0
    S["mask"] = S["mask"].astype(bool); S["reset"] = S["reset"].astype(bool)
    return S


def regua(S, det=None):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        r = G["ci"].prever({"detector": det or G["det"]}, S)
    return j1.regua(r["is_anomaly"].astype(bool), r["vigiado"].astype(bool))


def curto(x):
    return dict(det=x["det"], fp=x["fp"], fp_mes=x["fp_mes"], horas=x["horas"], pega_0411=x["lead"].get(TRIP) is not None,
                lead_0411=x["lead"].get(TRIP))


def executa(local: bool, insumos: str | None, teste: bool):
    import numpy as np
    import pandas as pd
    SAIDA.mkdir(parents=True, exist_ok=True); t00 = time.time()
    pk, a97, _ = j1.monta(local, insumos)
    res = dict(ambiente=j1.ambiente()); print("ambiente:", res["ambiente"], flush=True)
    j1.prepara(pk, a97, teste)
    assert int(G["ci"].VB_HORAS_BASE * G["ci"].POR_HORA) == 12000
    S = {n: sinais(n) for n in (12000, 6000, 3000)}
    ctrl = regua(S[12000]); res["controle"] = curto(ctrl)
    print(f"controle: {res['controle']}", flush=True)
    assert ctrl["det"] == 7 and ctrl["fp"] == 9, "controle não reproduz o J12"

    print("PARTE A (de onde vem o 04/11)", flush=True)
    t = pd.Timestamp(TRIP, tz="UTC"); A = {}
    for d in (2, 10, 30):
        w = (S[12000].index >= t - pd.Timedelta(days=d)) & (S[12000].index <= t + pd.Timedelta(days=2))
        x = S[12000].copy(); x.loc[w, "vb"] = S[3000]["vb"].reindex(x.index)[w]
        y = S[3000].copy(); y.loc[w, "vb"] = S[12000]["vb"].reindex(y.index)[w]
        A[f"controle_com_vb3000_{d}d"] = curto(regua(x))
        A[f"base3000_com_vbctrl_{d}d"] = curto(regua(y))
        print(f"   D {d:2d} d: controle + vb curto -> {A[f'controle_com_vb3000_{d}d']}", flush=True)
        print(f"          base curta + vb ctrl -> {A[f'base3000_com_vbctrl_{d}d']}", flush=True)
    res["A"] = A

    print("PARTE B (mesmas horas de alarme)", flush=True)
    B = {}
    for n in (6000, 3000):
        curva = []
        for f in np.round(np.arange(1.0, 2.0001, 0.05 if not teste else 0.5), 2):
            det = copy.deepcopy(G["det"])
            det["k_nivel_a"]["vb"] *= float(f); det["k_nivel_b"]["vb"] *= float(f)
            curva.append(dict(f=float(f), **curto(regua(S[n], det))))
        esc = min(curva, key=lambda c: abs(c["horas"] - ctrl["horas"]))      # sem olhar os trips
        B[n] = dict(escolhido=esc, curva=curva)
        print(f"   N {n}: f {esc['f']} -> {esc['det']}/8 FP {esc['fp']} horas {esc['horas']} (controle {ctrl['horas']}); "
              f"pega 04/11 {esc['pega_0411']}", flush=True)
        for c in curva:
            print(f"      f {c['f']:.2f}: {c['det']}/8 FP {c['fp']:2d} horas {c['horas']:5d} 04/11 {c['pega_0411']}", flush=True)
    res["B"] = B
    res["ganho_real"] = any(B[n]["escolhido"]["det"] == 8 and B[n]["escolhido"]["fp"] <= 9 for n in B)
    print("ganho real (8/8 e FP <= 9 com horas iguais):", res["ganho_real"], flush=True)
    res["duracao_min"] = round((time.time() - t00) / 60, 1)
    (SAIDA / "j13_resultado.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    from clearml import Task as Tk
    tk = Tk.current_task()
    if tk is not None:
        tk.upload_artifact("j13_resultado", artifact_object=str(SAIDA / "j13_resultado.json"))


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
