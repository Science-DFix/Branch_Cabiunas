#!/usr/bin/env python3
"""J9 -- carga inicial de instalação nova em blocos, dentro do limite de 1 milhão de células.

PRÉ-REGISTRO (08/11/2026, commitado antes de enviar)
Origem: o J34 recomendou instalar com carga inicial de >= 45 d de bruto (saída igual à referência desde o 1º
  instante). Mas 45 d × 720 × 37 ≈ 1,2 M células (60 d ≈ 1,6 M) numa carga só, acima do FEATURE_HISTORY_MAX_CELLS
  padrão do worker (1 M). O J8 mostrou que refazer em blocos a partir de uma foto é exato, mas na instalação nova não
  há foto anterior.
Pergunta: montar o estado inicial em blocos -- `carga` dos primeiros B dias (sem buffer) e depois
  `passo(dado, e_k, desde=e_{k-1})` bloco a bloco até T0 -- dá as mesmas emissões que o estado contínuo, e quanto
  custa?
Por que deve ser exato: a `carga` de um bloco é a mesma conta da carga inteira restrita aos primeiros B dias (tudo
  causal); cada bloco seguinte relê 3 d antes do seu início e leva o buffer do vb (até 26.000 estáveis), que é o que o
  passo normal faz a cada 6 h e o J34 mostrou bit a bit.

INSUMOS e motor: os do J34/J7/J8 (`j34_estado_robustez`, sem alteração; dado limpo, bundles do git). Trips: NENHUM,
  como no J7/J8.
PROCEDIMENTO. Um replay do motor contínuo (cortes de 6 h desde 2025-01-08 07:34) guarda as linhas emitidas em cada
  corte (referência). Instalações em 20 instantes T0 sorteados (semente 20261105) entre os cortes de 2025-04-01 até o
  fim − 14 d. Em cada T0, para H = 45, 60 d:
  - INTEIRA: `carga` de [T0 − H, T0] numa vez (controle; o J34 mediu igual desde T0 com H >= 45);
  - BLOCOS B = 14, 30 d: e_0 = T0 − H, e_k = min(e_0 + k·B, T0); `carga` de [e_0, e_1] e `passo(dado, e_k, [],
    desde=e_{k-1})` até T0.
  Depois, cada motor roda os cortes de (T0, T0 + 14 d] e emite linhas.
  EXATO = em todos esses cortes, linhas emitidas iguais às do motor contínuo no mesmo corte (t, p, sp, vb, mask, reset
  com |Δ| <= 1e-9 e NaN igual; alarme emitido igual).
  Mede: tempo da montagem (perf_counter), pico de memória Python (tracemalloc), nº de cargas, células por carga na
  conta da engenharia (linhas de 2 min do trecho lido × 37) e a maior delas.
EXPECTATIVAS: INTEIRA exata em 40/40 (~85%, como no J34); BLOCOS exatos em 80/80 (~75%); maior carga em blocos:
  B 30 <= 33 × 720 × 37 = 879 mil, B 14 <= 453 mil (100%, por construção); INTEIRA 45 d ≈ 1,2 M e 60 d ≈ 1,6 M
  (conta); montagem em blocos <= 2 s com B 30 (~70%); pico < 60 MB (~70%).
LIMITES: tracemalloc mede Python/numpy, não o RSS; horizonte de 14 d depois de T0 (o J34 usou 100 d com a carga
  inteira); o Cica não é o servidor do SIMPred; tempo de consulta ao banco não entra (é da engenharia).

Uso:
  python frente_thallys/j9_carga_blocos.py --local --teste   # encanamento local, recorte até 15/04/2025
  python frente_thallys/j9_carga_blocos.py --enqueue         # uma task na fila default
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
import j34_estado_robustez as J     # noqa: E402
import j7_custo_refazer as j7       # noqa: E402

NOME = "frente_thallys | J9 carga inicial em blocos"
CARGAS, BLOCOS = [45, 60], [14, 30]
N_INST, N_COLS, HORIZ_D = 20, 37, 14
SAIDA = Path(tempfile.gettempdir()) / "j9_saida"
G = j1.G
T, D = J.T, J.D


def celulas(a, b):
    return int((b - a) / D("2min")) * N_COLS


def compara(a, b):
    if a is None or b is None:
        return (a is None and b is None), "" if (a is None and b is None) else "um vazio"
    ok, porque = j7.igual_tab(a[J.COLS], b[J.COLS])
    if ok and not (a["is_anomaly"].fillna(False).astype(bool).to_numpy()
                   == b["is_anomaly"].fillna(False).astype(bool).to_numpy()).all():
        ok, porque = False, "alarme emitido"
    return ok, porque


def executa(local: bool, insumos: str | None, teste: bool):
    import numpy as np
    import pandas as pd
    SAIDA.mkdir(parents=True, exist_ok=True); t00 = time.time()
    pk, a97, _ = j1.monta(local, insumos)
    j1.prepara(pk, a97, teste)
    ci = G["ci"]; G["orig_z"] = ci._z_vibracao; ci._z_vibracao = J._z_com_buffer
    J.prepara_cenarios()
    res = dict(ambiente=j1.ambiente()); print("ambiente:", res["ambiente"], flush=True)
    df, cs = G["df"], J.cortes_cen("limpo")
    horiz = D(f"{HORIZ_D if not teste else 4}D")
    cargas = CARGAS if not teste else [45]
    blocos = BLOCOS if not teste else [14]

    ini = T("2025-04-01") if not teste else T("2025-04-05")
    cand = [c for c in cs if c >= ini and c + horiz <= cs[-1]]
    rng = np.random.default_rng(20261105)
    n = N_INST if not teste else 2
    alvo = sorted(cand[i] for i in rng.choice(len(cand), size=min(n, len(cand)), replace=False))

    m, emit = J.Motor(), {}
    m.carga(df.loc[cs[0] - D("60D"):cs[0]], [], cs[0])
    for s in cs[1:]:
        emit[s] = m.passo(df, s, [])
    print(f"replay: {len(cs)} cortes, {len(alvo)} instalações, {time.time() - t00:.0f} s", flush=True)

    regs = []
    for T0 in alvo:
        seg = [s for s in cs if T0 < s <= T0 + horiz]
        for H in cargas:
            e0 = T0 - D(f"{H}D")
            variantes = [("inteira", None)] + [(f"blocos_{B}", B) for B in blocos]
            for nome, B in variantes:
                x = J.Motor(); cel = []

                def monta():
                    if B is None:
                        cel.append(celulas(e0, T0)); x.carga(df.loc[e0:T0], [], T0); return
                    fins, e = [], e0
                    while e < T0:
                        e = min(e + D(f"{B}D"), T0); fins.append(e)
                    cel.append(celulas(e0, fins[0])); x.carga(df.loc[e0:fins[0]], [], fins[0])
                    a = fins[0]
                    for e in fins[1:]:
                        cel.append(celulas(a - D(f"{J.BRUTO_D}D"), e)); x.passo(df, e, [], desde=a); a = e
                _, dt, pico = j7.medido(monta)
                ok, porque, n_lin = True, "", 0
                for s in seg:
                    a_s = x.passo(df, s, [])
                    o, p = compara(a_s, emit[s])
                    n_lin += 0 if a_s is None else len(a_s)
                    if not o:
                        ok, porque = False, f"{s}: {p}"; break
                regs.append(dict(T0=str(T0), H=H, modo=nome, n_cargas=len(cel), celulas_max=max(cel),
                                 celulas_tot=sum(cel), s=dt, pico_mb=pico, linhas_comparadas=n_lin,
                                 exato=ok, porque=porque))
        print(f"   {T0}: " + " ".join(f"{r['H']}/{r['modo'].replace('blocos_', 'B')}:{'ok' if r['exato'] else 'X'}({r['s']}s)"
                                       for r in regs if r["T0"] == str(T0)), flush=True)

    R = pd.DataFrame(regs)
    resumo = {}
    for (H, modo), g in R.groupby(["H", "modo"], sort=False):
        d = dict(n=len(g), exatos=int(g.exato.sum()), s_med=float(g.s.median()), s_p95=float(g.s.quantile(0.95)),
                 s_max=float(g.s.max()), pico_med_mb=float(g.pico_mb.median()), pico_max_mb=float(g.pico_mb.max()),
                 cargas_med=float(g.n_cargas.median()), celulas_max=int(g.celulas_max.max()),
                 celulas_tot_med=float(g.celulas_tot.median()), linhas_med=float(g.linhas_comparadas.median()),
                 falhas=g.loc[~g.exato, "porque"].tolist())
        resumo[f"H{H}_{modo}"] = d
        print(f"   H={H} d {modo:10s} exatos {d['exatos']}/{d['n']} | {d['s_med']:.3f} s (p95 {d['s_p95']:.3f}) | "
              f"pico {d['pico_med_mb']:.0f} MB | cargas {d['cargas_med']:.0f} | maior carga {d['celulas_max']:,} cél | "
              f"linhas comparadas {d['linhas_med']:,.0f}", flush=True)
    res.update(resumo=resumo, registros=regs, duracao_min=round((time.time() - t00) / 60, 1))
    (SAIDA / "j9_resultado.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    from clearml import Task as Tk
    tk = Tk.current_task()
    if tk is not None:
        tk.upload_artifact("j9_resultado", artifact_object=str(SAIDA / "j9_resultado.json"))


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
