#!/usr/bin/env python3
"""J6 -- bundle mensal que chega atrasado: o motor com estado precisa refazer o mês, e basta isso?

PRÉ-REGISTRO (06/11/2026, commitado antes de enviar)
Origem: o documento da engenharia pede, no replay, "trocas de bundle". O J34 já atravessa 16 viradas de mês com o bundle
  no prazo (bit a bit). Falta o caso de o retreino do dia 1 falhar ou atrasar: até o bundle novo chegar, `vigencia` cai
  no bundle do mês anterior, e o motor grava na tabela t/p/sp pontuados com ele. Quando o bundle chega, essas linhas
  ficam erradas, e o CUSUM do `prever` (que roda sobre a tabela de 45 d) as carrega para frente.
Pergunta: com o bundle do mês M disponível só em M-01 + A dias (A = 1, 3, 7; todos os meses de 2025-02 a 2026-04),
  (a) ignorar (P0) muda o resultado depois da chegada? (b) refazer desde M-01 quando o bundle chega (P1, o mesmo
  `passo(desde=...)` do J34) volta a ser bit a bit? (c) e o procedimento atual (janela de 60 d recalculada a cada corte)?

INSUMOS e motor: os do J34 (`j34_estado_robustez`, sem alteração: cortes de 6 h desde 2025-01-08 07:34, bruto 3 d,
  buffer do vb, tabela 45 d, carga inicial de 60 d, trips conhecidos sem atraso). A única mudança é a lista de bundles
  passada ao `preprocessar` em cada corte s: os de mes_servido M com M-01 + A <= s (o de 2025-01 sempre disponível).
CORRIDAS (paralelas): ref_limpo (60 d, bundles no prazo) = REFERÊNCIA; motor_limpo (V1); e para cada A: ref_A (60 d com
  bundles atrasados = como hoje), motor_A_p0, motor_A_p1.
MEDIDAS contra ref_limpo, desde 2025-02-01: % de instantes diferentes e máx |Δ| por sinal e no alarme (critério do J34,
  |Δ| > 1e-9), (i) em tudo, (ii) DEPOIS DA CHEGADA = fora dos trechos [M-01, M-01 + A) -- as linhas emitidas antes da
  chegada saem com o bundle velho em qualquer procedimento e não são revistas; régua (det, início, banda, FP, lead).
  BIT A BIT DEPOIS DA CHEGADA = 0% em t, p, sp, vb e alarme no recorte (ii).
V1: motor_limpo bit a bit contra ref_limpo (repete o J34); se falhar, o resto não vale.
EXPECTATIVAS: V1 passa (~97%). ref_A bit a bit depois da chegada nos 3 atrasos (~90%). motor P1 bit a bit depois da
  chegada nos 3 (~90%). motor P0: sinais t/p/sp/vb iguais depois da chegada (~70%, o bruto de 3 d é repontuado), mas
  alarme diferente depois da chegada com A = 7 (~65%) e régua diferente da referência com A = 7 (~40%). Antes da
  chegada t/p diferem em todas as variantes (~95%).
LIMITES: atraso igual em todos os meses (pior caso sistemático, não um mês isolado); bundles do git; 8 trips. Bundle
  VENCIDO (validade 62 d) não é exercitado: com A <= 7 o anterior ainda vale.

Uso:
  python frente_thallys/j6_bundle_atrasado.py --local --teste   # encanamento local, recorte até 15/04/2025
  python frente_thallys/j6_bundle_atrasado.py --enqueue         # uma task na fila default
"""
from clearml import Task  # noqa: F401  (antes do argparse: o worker injeta argumentos por patch)
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from multiprocessing import get_context
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import j1_janela_entrada as j1      # noqa: E402
import j34_estado_robustez as J     # noqa: E402

NOME = "frente_thallys | J6 bundle atrasado"
ATRASOS_D = [1, 3, 7]
SAIDA = Path(tempfile.gettempdir()) / "j6_saida"
G = j1.G
T, D = J.T, J.D


def _inicio(m):
    return T(m["modelo"]["mes_servido"] + "-01")


def chegada(m, A):
    """Instante em que o bundle fica disponível; o primeiro (2025-01) já existe no início do replay."""
    ini = _inicio(m)
    return ini if ini <= T("2025-01-01") else ini + D(f"{A}D")


def disponiveis(s, A):
    ms = [m for m in G["todos"] if chegada(m, A) <= s]
    return ms or G["todos"][:1]


def corre_ref(A):
    import pandas as pd
    pedacos, anterior, t0 = [], None, time.time()
    for s in J.cortes_cen("limpo"):
        G["modelos"] = disponiveis(s, A)
        w = G["df"].loc[s - D("60D"):s]
        proc = J._preproc(w, J.conhecidos("limpo", s), None)
        res = J._prever(proc)
        novo = proc[proc.index > anterior] if anterior is not None else proc
        pedacos.append(novo[J.COLS].join(res[["is_anomaly", "vigiado"]], how="left")); anterior = s
    r = pd.concat(pedacos); r = r[~r.index.duplicated(keep="first")]
    nome = "ref_limpo" if A == 0 else f"ref_{A}"
    r.to_parquet(SAIDA / f"{nome}.parquet"); print(f"   {nome}: {len(r)} linhas, {time.time() - t0:.0f} s", flush=True)


def corre_motor(A, modo):
    import pandas as pd
    m, pedacos, t0 = J.Motor(), [], time.time()
    cs = J.cortes_cen("limpo")
    pend = [(_inicio(b), chegada(b, A)) for b in G["todos"] if chegada(b, A) > cs[0]] if modo == "p1" else []
    G["modelos"] = disponiveis(cs[0], A)
    m.carga(G["df"].loc[cs[0] - D("60D"):cs[0]], J.conhecidos("limpo", cs[0]), cs[0])
    for s in cs[1:]:
        G["modelos"] = disponiveis(s, A)
        desde = None
        for x, ch in list(pend):
            if s >= ch:
                desde = x if desde is None else min(desde, x); pend.remove((x, ch))
        out = m.passo(G["df"], s, J.conhecidos("limpo", s), desde=desde)
        if out is not None:
            pedacos.append(out)
    r = pd.concat(pedacos); r = r[~r.index.duplicated(keep="first")]   # linha emitida não é revista
    nome = "motor_limpo" if A == 0 else f"motor_{A}_{modo}"
    r.to_parquet(SAIDA / f"{nome}.parquet"); print(f"   {nome}: {len(r)} linhas, {time.time() - t0:.0f} s", flush=True)


def tarefa(x):
    kind, A, modo = x
    corre_ref(A) if kind == "ref" else corre_motor(A, modo)
    return x


def antes_da_chegada(idx, A):
    import numpy as np
    fora = np.zeros(len(idx), bool)
    for b in G["todos"]:
        ini, ch = _inicio(b), chegada(b, A)
        if ch > ini:
            fora |= (idx >= ini) & (idx < ch)
    return fora


def analisa(As):
    import pandas as pd
    R0 = pd.read_parquet(SAIDA / "ref_limpo.parquet")
    rr = J.regua(R0)
    res = dict(regua_ref=rr)
    chave = lambda r: (r["det"], r["inicio"], r["banda"], r["fp"], r["lead"])   # noqa: E731
    nomes = ["motor_limpo"] + [f"{k}_{A}" for A in As for k in ("ref",)] + [f"motor_{A}_{p}" for A in As for p in ("p0", "p1")]
    for nome in nomes:
        X = pd.read_parquet(SAIDA / f"{nome}.parquet")
        A = 0 if nome == "motor_limpo" else int(nome.split("_")[1])
        dt = J.diverge(X, R0)
        Xd = X.loc["2025-02-01":]; Xd = Xd[~antes_da_chegada(Xd.index, A)]
        dd = J.diverge(Xd, R0)
        ra = J.regua(X)
        res[nome] = dict(tudo=dt, depois_da_chegada=dd, bit_a_bit_depois=J.igual(dd), bit_a_bit_tudo=J.igual(dt),
                         regua=ra, regua_igual=chave(ra) == chave(rr))
        print(f"   {nome:14s} depois bit a bit {res[nome]['bit_a_bit_depois']} | régua igual {res[nome]['regua_igual']} "
              f"({ra['det']}/8 FP {ra['fp']}) | depois: " + " ".join(f"{s} {dd[s + '_pct']}%" for s in ("t", "p", "sp", "vb", "alarme"))
              + " | tudo: " + " ".join(f"{s} {dt[s + '_pct']}%" for s in ("t", "p", "alarme")), flush=True)
    return res


def executa(local: bool, insumos: str | None, teste: bool, procs: int):
    SAIDA.mkdir(parents=True, exist_ok=True); t00 = time.time()
    J.SAIDA = SAIDA
    pk, a97, _ = j1.monta(local, insumos)
    j1.prepara(pk, a97, teste)
    ci = G["ci"]; G["orig_z"] = ci._z_vibracao; ci._z_vibracao = J._z_com_buffer
    J.prepara_cenarios()
    G["todos"] = sorted(G["modelos"], key=_inicio)
    assert all(m["modelo"].get("mes_servido") for m in G["todos"])
    res = dict(ambiente=j1.ambiente(), bundles=[m["modelo"]["mes_servido"] for m in G["todos"]])
    print("ambiente:", res["ambiente"], flush=True)
    As = ATRASOS_D if not teste else [7]
    trabalhos = [("ref", 0, None), ("motor", 0, "normal")]
    trabalhos += [("ref", A, None) for A in As] + [("motor", A, p) for A in As for p in ("p0", "p1")]
    print(f"{len(trabalhos)} corridas em paralelo", flush=True)
    with get_context("fork").Pool(min(procs, len(trabalhos))) as pool:
        for _ in pool.imap_unordered(tarefa, trabalhos):
            pass
    G["modelos"] = G["todos"]
    res["analise"] = analisa(As)
    res["duracao_min"] = round((time.time() - t00) / 60, 1)
    (SAIDA / "j6_resultado.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    from clearml import Task as Tk
    tk = Tk.current_task()
    if tk is not None:
        for f in sorted(SAIDA.glob("*")):
            tk.upload_artifact(f.stem, artifact_object=str(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enqueue", action="store_true")
    ap.add_argument("--local", action="store_true"); ap.add_argument("--teste", action="store_true")
    ap.add_argument("--insumos", default=j1.DS_INSUMOS); ap.add_argument("--procs", type=int, default=11)
    a = ap.parse_args()
    if a.enqueue:
        t = Task.init(project_name="TesteMLCab", task_name=NOME)
        t.set_base_docker("python:3.11-slim", docker_setup_bash_script=["apt-get update -qq", "apt-get install -y -qq libgomp1"])
        t.set_packages(j1.PACOTES); t.set_parameters({"General/insumos": a.insumos, "General/procs": str(a.procs)})
        # o padrão do ClearML pega o remoto `origin` (SIMPred), onde este commit não existe: fixa o Branch_Cabiunas
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=AQUI, text=True).strip()
        t.set_script(repository="git@github.com:Science-DFix/Branch_Cabiunas.git", branch="exp/ocsvm-validacao", commit=commit)
        t.execute_remotely(queue_name="default")
    if os.environ.get("CLEARML_TASK_ID"):
        tk = Task.init(project_name="TesteMLCab", task_name=NOME)
        a.insumos = tk.get_parameter("General/insumos") or a.insumos
        a.procs = int(tk.get_parameter("General/procs") or a.procs)
    executa(a.local, a.insumos, a.teste, a.procs)


if __name__ == "__main__":
    main()
