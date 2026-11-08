#!/usr/bin/env python3
"""J8 -- recuperação de correção antiga (> 17 d) dentro do limite de 1 milhão de células: foto diária + refazer em blocos.

PRÉ-REGISTRO (07/11/2026, commitado antes de enviar)
Origem: o J7 mostrou que refazer pelo estado é exato só até ~17 d atrás (o refazer corta o buffer do vb em `desde`).
  Acima disso recomendamos reconstruir de 60 d de bruto, que lê 60 × 720 × 37 ≈ 1,6 M células e ESTOURA o
  FEATURE_HISTORY_MAX_CELLS padrão do worker (1 M). A ideia de ler só as sondas TV_ no trecho antigo não resolve na
  conta: a tabela de 45 d de sinais (t, p, sp) usa as 37 colunas (14 temp + 12 pressão + mancal + 10 TV_ + 2 da
  máscara), e 45 × 720 × 37 + 15 × 720 × 12 ≈ 1,33 M. Por isso não é testada.
Pergunta: guardar uma foto diária do estado (3,3 MB) e, numa correção em `desde`, voltar à última foto f <= desde e
  refazer até o agora em blocos de B dias (cada bloco lê B + 3 d de bruto das 37 colunas) é exato, e quanto custa?
Por que deve ser exato: a foto em f tem o buffer do vb inteiro antes de f (até 26.000 estáveis); o bloco
  `passo(dado, e_k, desde=e_{k-1})` corta o estado em e_{k-1} (não perde nada do buffer anterior a e_{k-1} − 3 d) e
  relê [e_{k-1} − 3 d, e_k]. É o refazer do J7 com o `desde` sempre no fim do estado, onde ele é exato.

INSUMOS e motor: os do J34/J7 (`j34_estado_robustez`, sem alteração; dado limpo, bundles do git). Trips: NENHUM,
  como no J7 (o trip tardio já foi medido no J34; aqui se mede o mecanismo).
PROCEDIMENTO. Um replay do motor (cortes de 6 h desde 2025-01-08 07:34). Guarda: o estado nos 20 instantes c
  sorteados (semente 20261105) entre 2025-06-01 e o fim (o estado contínuo = referência), e a foto diária (1º corte de
  cada dia) que for a última <= c − A para cada idade A. Para cada c, A = 21, 30, 45, 60 d e B = 14, 30 d:
  cópia da foto f, blocos e_0 = f, e_k = min(f + k·B, c), `passo(dado, e_k, [], desde=e_{k-1})` até e_k = c.
  EXATO = tabela de sinais (t, p, sp, vb, mask, reset) igual à do estado contínuo em c (|Δ| <= 1e-9, NaN igual) e
  alarme do `prever` igual em toda a tabela; e as linhas emitidas no corte seguinte (c + 6 h) iguais (sinais e alarme).
  Mede: tempo total (perf_counter), pico de memória Python (tracemalloc), nº de blocos, células por bloco na conta
  da engenharia (linhas de 2 min do trecho lido × 37) e a maior delas.
  Previsão pela regra do J7, bloco a bloco: posições estáveis antes de e_{k-1} >= 12.900.
EXPECTATIVAS: exato em 160/160 (~85%); regra prevê exato em 160/160 e acerta (~80%); maior bloco <= 33 × 720 × 37 =
  879 mil células (100%, por construção); tempo com B = 30 d: A = 30 d ~1 s, A = 60 d <= 2,5 s (~65%); B = 14 d custa
  mais que B = 30 d na mesma idade (mais blocos, cada um relê 3 d; ~75%); pico < 50 MB (~70%).
LIMITES: tracemalloc mede Python/numpy, não o RSS; sem correção real no dado (o J34 já mostrou que o refazer com dado
  corrigido dá o da referência); foto diária em disco = 60 × 3,3 MB ≈ 200 MB por equipamento (conta, não medida); o
  Cica não é o servidor do SIMPred; tempo de consulta ao banco não entra (é da engenharia).

Uso:
  python frente_thallys/j8_foto_blocos.py --local --teste   # encanamento local, recorte até 15/04/2025
  python frente_thallys/j8_foto_blocos.py --enqueue         # uma task na fila default
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
import j34_estado_robustez as J     # noqa: E402
import j7_custo_refazer as j7       # noqa: E402

NOME = "frente_thallys | J8 foto diaria + refazer em blocos"
IDADES, BLOCOS = [21, 30, 45, 60], [14, 30]
N_INST, N_COLS, LIN_DIA = 20, 37, 720
SAIDA = Path(tempfile.gettempdir()) / "j8_saida"
G = j1.G
T, D = J.T, J.D


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
    idades = IDADES if not teste else [21, 30]
    blocos = BLOCOS if not teste else [14]

    ini = T("2025-06-01") if not teste else T("2025-04-05")
    cand = [c for c in cs[:-1] if c >= ini]
    rng = np.random.default_rng(20261105)
    n = N_INST if not teste else 3
    alvo = sorted(cand[i] for i in rng.choice(len(cand), size=min(n, len(cand)), replace=False))
    proximo = {c: cs[cs.index(c) + 1] for c in alvo}
    diarios, vistos = [], set()                       # 1º corte de cada dia = instante da foto diária
    for s in cs:
        if s.normalize() not in vistos:
            vistos.add(s.normalize()); diarios.append(s)
    foto_de = {(c, A): max(f for f in diarios if f <= c - D(f"{A}D")) for c in alvo for A in idades}
    precisa = set(foto_de.values())

    m, ref, fotos = J.Motor(), {}, {}
    m.carga(df.loc[cs[0] - D("60D"):cs[0]], [], cs[0])
    for s in cs[1:]:
        m.passo(df, s, [])
        if s in precisa:
            fotos[s] = copy.deepcopy(m)
        if s in alvo:
            ref[s] = copy.deepcopy(m)
    print(f"replay: {len(cs)} cortes, {len(ref)} instantes, {len(fotos)} fotos, {time.time() - t00:.0f} s", flush=True)

    _, est_all, _, _ = ci._mascara(ci._grade(df))
    est_all = est_all.astype(bool)

    def posicoes(x, desde):
        ws = desde - D(f"{J.BRUTO_D}D")
        return min(J.N_BUF, int((x.buf.index < ws).sum())) + int(est_all.loc[ws:desde - D("1ns")].sum())

    regs = []
    for c in alvo:
        base = ref[c]
        cont = copy.deepcopy(base); s2 = proximo[c]
        b_seg = cont.passo(df, s2, [])
        for A in idades:
            f = foto_de[(c, A)]
            for B in blocos:
                x = copy.deepcopy(fotos[f])
                fins = []
                e = f
                while e < c:
                    e = min(e + D(f"{B}D"), c); fins.append(e)
                celulas, prev = [], []

                def roda():
                    a = f
                    for e in fins:
                        prev.append(posicoes(x, a) >= J.N_CHEIA + 180)
                        celulas.append(int((e - (a - D(f"{J.BRUTO_D}D"))) / D("2min")) * N_COLS)
                        x.passo(df, e, [], desde=a); a = e
                _, dt, pico = j7.medido(roda)
                ok, porque = j7.igual_tab(x.tab, base.tab)
                if ok:
                    a_seg = x.passo(df, s2, [])
                    if a_seg is None or b_seg is None:
                        ok, porque = (a_seg is None and b_seg is None), "corte seguinte vazio"
                    else:
                        a2, b2 = a_seg.loc[a_seg.index > c], b_seg.loc[b_seg.index > c]
                        ok, porque = j7.igual_tab(a2[J.COLS], b2[J.COLS])
                        if ok and not (a2["is_anomaly"].fillna(False).astype(bool).to_numpy()
                                       == b2["is_anomaly"].fillna(False).astype(bool).to_numpy()).all():
                            ok, porque = False, "alarme emitido"
                        porque = porque or ""
                        if not ok:
                            porque = "seguinte: " + porque
                regs.append(dict(c=str(c), idade=A, bloco=B, foto=str(f), n_blocos=len(fins), s=dt, pico_mb=pico,
                                 celulas_max=max(celulas), celulas_tot=sum(celulas),
                                 previsto_exato=bool(all(prev)), exato=ok, porque=porque))
        print(f"   {c}: " + " ".join(f"{r['idade']}/{r['bloco']}:{'ok' if r['exato'] else 'X'}({r['s']}s)"
                                      for r in regs if r["c"] == str(c)), flush=True)

    R = pd.DataFrame(regs)
    resumo = {}
    for (A, B), g in R.groupby(["idade", "bloco"]):
        d = dict(n=len(g), exatos=int(g.exato.sum()), regra_acerta=int((g.previsto_exato == g.exato).sum()),
                 previstos_exatos=int(g.previsto_exato.sum()),
                 s_med=float(g.s.median()), s_p95=float(g.s.quantile(0.95)), s_max=float(g.s.max()),
                 pico_med_mb=float(g.pico_mb.median()), pico_max_mb=float(g.pico_mb.max()),
                 blocos_med=float(g.n_blocos.median()), celulas_max=int(g.celulas_max.max()),
                 celulas_tot_med=float(g.celulas_tot.median()),
                 falhas=g.loc[~g.exato, "porque"].value_counts().to_dict())
        resumo[f"A{A}_B{B}"] = d
        print(f"   A={A:2d} d B={B:2d} d  exatos {d['exatos']}/{d['n']} | {d['s_med']:.3f} s (p95 {d['s_p95']:.3f}) | "
              f"pico {d['pico_med_mb']:.0f} MB | blocos {d['blocos_med']:.0f} | maior bloco {d['celulas_max']:,} cél | "
              f"total {d['celulas_tot_med']:,.0f} cél | regra acerta {d['regra_acerta']}/{d['n']}", flush=True)
    res.update(resumo=resumo, registros=regs, duracao_min=round((time.time() - t00) / 60, 1))
    (SAIDA / "j8_resultado.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    from clearml import Task as Tk
    tk = Tk.current_task()
    if tk is not None:
        tk.upload_artifact("j8_resultado", artifact_object=str(SAIDA / "j8_resultado.json"))


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
