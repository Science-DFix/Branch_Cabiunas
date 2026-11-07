#!/usr/bin/env python3
"""J7 -- custo e limite do "refazer" do motor com estado (recuperação após correção, trip tardio, bundle atrasado).

PRÉ-REGISTRO (07/11/2026, commitado antes de enviar)
Origem: o documento da engenharia pede medir o "tempo de recuperação após correção". O J34 e o J6 fixaram a regra
  (correção de dado, trip conhecido tarde e bundle novo = refazer desde o início do trecho afetado), mas só mediram o
  reinício (ler o estado, 5 ms). Os casos reais pedem refazer de 1 d (correção) a 14 d (trip conhecido 7 d depois:
  desde trip − 7 d) ou até o mês inteiro (bundle). E há um limite: o buffer do vb guarda 26.000 linhas estáveis e usa
  13.000; refazer desde `desde` corta o buffer em `desde` e relê o bruto desde `desde` − 3 d. A base do vb é posicional:
  12.000 estáveis + guarda 720 antes do início do bloco de 6 h. Então só é exato se há >= 12.720 (+180, um bloco)
  posições estáveis antes de `desde` = min(13.000, buffer antes de `desde` − 3 d) + estáveis do bruto em [desde − 3 d,
  desde) -- ~17 d de operação contínua com o buffer de 26.000.
Pergunta: quanto custa refazer P dias, até onde é exato, e quanto custa a alternativa (reconstruir do bruto)?

INSUMOS e motor: os do J34 (`j34_estado_robustez`, sem alteração; dado limpo, bundles do git). Trips: NENHUM, no replay e no refazer. Mede-se só o mecanismo do buffer. O teste local
  mostrou por quê (07/04 e 17/03/2025): o pacote só aplica um trip quando o dado passa do instante dele, então as linhas
  gravadas antes disso não têm a exclusão [t − 7 d, t + 2 d] do vb e as refeitas depois têm -- o refazer diverge do
  estado gravado, de propósito (é a regra do trip tardio, já medida no J34). Nem a lista fixa evita isso; sem trips, a
  base do vb depende só das posições. Mudado antes do envio.
PROCEDIMENTO. Um replay do motor (cortes de 6 h desde 2025-01-08 07:34) guarda cópias do estado em 20 instantes
  sorteados (semente 20261105) entre 2025-04-01 e o fim. Em cada instante c, serial:
  - REFAZER P, P = 1, 7, 14, 17, 21, 30 d: cópia do estado, `passo(dado, c, trips, desde=c − P)` sobre o MESMO dado
    (sem correção, para medir só o mecanismo). Mede tempo (perf_counter), pico de memória Python (tracemalloc), linhas
    estáveis em [c − P, c] e linhas do buffer que sobram antes de c − P − 3 d (início do bruto relido). EXATO = tabela de sinais (t, p, sp, vb, mask,
    reset) igual à do estado sem refazer (|Δ| <= 1e-9, NaN igual) e alarme do `prever` igual em toda a tabela.
  - RECONSTRUIR: motor novo, `carga` de 60 d de bruto terminando em c. Mede tempo e pico; EXATO = as linhas emitidas no
    corte seguinte (c + 6 h) iguais às do estado contínuo no mesmo corte.
  Previsão pela contagem: exato <=> posições estáveis antes de c − P >= 12.900 (conta acima). Confere-se a regra.
  (A 1ª versão da conta, 'sobram >= 13.000 no buffer', errou 17 d no teste local e foi trocada antes do envio.)
EXPECTATIVAS: P <= 14 exato em 20/20 (~90%); 17 d exato em >= 18/20 (~60%); 30 d inexato na maioria dos instantes (~80%); a regra da contagem acerta
  em 120/120 (~75%); refazer 7 d com mediana <= 0,5 s (~70%) e 17 d <= 1 s (~70%); reconstruir ~0,3 s (como no J34) e
  exato em 20/20 (~85%).
LIMITES: tracemalloc mede Python/numpy, não o RSS; sem correção real no dado (o J34 já mostrou que o resultado de
  refazer com dado corrigido é o da referência); o Cica não é o servidor do SIMPred.

RESULTADO (07/11/2026 12:11; task e9ce6fc5, commit e65b697, 6 min no Cica; replay de 1.880 cortes em 255 s).
  Exatos e custo (mediana, p95):
    refazer  1 d  20/20  0,49 s (0,87)  11 MB
    refazer  7 d  20/20  0,55 s (0,89)  13 MB
    refazer 14 d  20/20  0,62 s (0,91)  17 MB
    refazer 17 d  20/20  0,65 s (0,95)  19 MB
    refazer 21 d  13/20  0,65 s (0,98)  22 MB
    refazer 30 d   6/20  0,66 s (1,00)  27 MB
    reconstruir de 60 d de bruto  20/20 (corte seguinte idêntico)  0,53 s (0,65)  38 MB
  Regra das posições: 119/120. A exceção é 2026-04-01, 21 d: 13.013 posições e vb |Δ| 6e-6; o limite não é afiado
    na margem.
PLACAR: ACERTEI <= 14 d 20/20; 17 d >= 18/20 (20/20); 30 d inexato na maioria (14/20); 17 d <= 1 s; reconstruir
  exato 20/20. ERREI regra 120/120 (119); refazer 7 d <= 0,5 s (0,55); reconstruir ~0,3 s (0,53).
LEITURA: correção, trip ou bundle até 17 d atrás = refazer pelo estado (lê P + 3 d de bruto, ~0,6 s). Mais antigo =
  reconstruir de 60 d de bruto (0,53 s, exato). O custo de CPU é o mesmo; muda o volume lido do banco. Para a regra,
  usar o limite em dias (17 d) ou dar margem à conta das posições.

Uso:
  python frente_thallys/j7_custo_refazer.py --local --teste   # encanamento local, recorte até 15/04/2025
  python frente_thallys/j7_custo_refazer.py --enqueue         # uma task na fila default
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
import tracemalloc
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import j1_janela_entrada as j1      # noqa: E402
import j34_estado_robustez as J     # noqa: E402

NOME = "frente_thallys | J7 custo do refazer"
PROFS = [1, 7, 14, 17, 21, 30]
N_INST = 20
SAIDA = Path(tempfile.gettempdir()) / "j7_saida"
G = j1.G
T, D = J.T, J.D


def igual_tab(A, B):
    import numpy as np
    if not A.index.equals(B.index):
        return False, "índices diferentes"
    for s in ("t", "p", "sp", "vb"):
        a, b = A[s].to_numpy(float), B[s].to_numpy(float)
        if (np.isnan(a) != np.isnan(b)).any():
            return False, f"{s} NaN"
        ok = ~np.isnan(a)
        if ok.any() and np.abs(a[ok] - b[ok]).max() > 1e-9:
            return False, f"{s} |Δ| {np.abs(a[ok] - b[ok]).max():.3g}"
    for s in ("mask", "reset"):
        if not (A[s].astype(bool).to_numpy() == B[s].astype(bool).to_numpy()).all():
            return False, s
    ra, rb = J._prever(A), J._prever(B)
    if not (ra["is_anomaly"].astype(bool).to_numpy() == rb["is_anomaly"].astype(bool).to_numpy()).all():
        return False, "alarme"
    return True, ""


def medido(f):
    tracemalloc.start(); t0 = time.perf_counter()
    r = f()
    dt = time.perf_counter() - t0; pico = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    return r, round(dt, 3), round(pico / 1e6, 1)


def instantes(cs, teste):
    import numpy as np
    ini = T("2025-04-01") if not teste else T("2025-03-15")
    cand = [c for c in cs[:-1] if c >= ini]
    rng = np.random.default_rng(20261105)
    n = N_INST if not teste else 3
    return sorted(rng.choice(len(cand), size=min(n, len(cand)), replace=False).tolist()), cand


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
    idx, cand = instantes(cs, teste); alvo = {cand[i] for i in idx}
    proximo = {c: cs[cs.index(c) + 1] for c in alvo}

    # replay único do motor, guardando cópias do estado nos instantes sorteados
    m, fotos = J.Motor(), {}
    m.carga(df.loc[cs[0] - D("60D"):cs[0]], [], cs[0])
    for s in cs[1:]:
        m.passo(df, s, [])
        if s in alvo:
            fotos[s] = copy.deepcopy(m)
    print(f"replay: {len(cs)} cortes, {len(fotos)} fotos, {time.time() - t00:.0f} s", flush=True)

    tags = G["det"]["tags_vibracao"]
    _, est_all, _, _ = ci._mascara(ci._grade(df))
    est_all = est_all.astype(bool)
    regs = []
    for c in sorted(fotos):
        base = fotos[c]; tr = []
        for P in PROFS:
            desde = c - D(f"{P}D")
            x = copy.deepcopy(base)
            n_est = int(est_all.loc[desde:c].sum())
            ws = desde - D(f"{J.BRUTO_D}D")   # o passo relê o bruto desde aqui e pega até N_BUF linhas do buffer antes
            sobra = int((x.buf.index < ws).sum())
            pos = min(J.N_BUF, sobra) + int(est_all.loc[ws:desde - D("1ns")].sum())   # posições estáveis antes de `desde`
            _, dt, pico = medido(lambda: x.passo(df, c, tr, desde=desde))
            ok, porque = igual_tab(x.tab, base.tab)
            regs.append(dict(c=str(c), modo=f"refazer_{P}", s=dt, pico_mb=pico, estaveis_trecho=n_est, sobra_buf=sobra,
                             posicoes_antes=pos, previsto_exato=bool(pos >= J.N_CHEIA + 180), exato=ok, porque=porque))
        # reconstruir a partir de 60 d de bruto, e comparar o corte seguinte
        y = J.Motor()
        _, dt, pico = medido(lambda: y.carga(df.loc[c - D("60D"):c], tr, c))
        cont = copy.deepcopy(base)
        s2 = proximo[c]; tr2 = []
        a, b = y.passo(df, s2, tr2), cont.passo(df, s2, tr2)
        if a is None or b is None:
            ok, porque = (a is None and b is None), "corte seguinte vazio"
        else:
            a = a.loc[a.index > c]; b = b.loc[b.index > c]
            ok, porque = igual_tab(a[J.COLS], b[J.COLS])
            if ok and not (a["is_anomaly"].fillna(False).astype(bool).to_numpy() == b["is_anomaly"].fillna(False).astype(bool).to_numpy()).all():
                ok, porque = False, "alarme emitido"
        regs.append(dict(c=str(c), modo="reconstruir_60d", s=dt, pico_mb=pico, exato=ok, porque=porque))
        print(f"   {c}: " + " ".join(f"{r['modo'].split('_')[1]}:{'ok' if r['exato'] else 'X'}({r['s']}s)"
                                      for r in regs if r["c"] == str(c)), flush=True)

    R = pd.DataFrame(regs)
    resumo = {}
    for modo, g in R.groupby("modo", sort=False):
        d = dict(n=len(g), exatos=int(g.exato.sum()), s_med=float(g.s.median()), s_p95=float(g.s.quantile(0.95)),
                 s_max=float(g.s.max()), pico_med_mb=float(g.pico_mb.median()), pico_max_mb=float(g.pico_mb.max()))
        if modo.startswith("refazer"):
            d.update(regra_acerta=int((g.previsto_exato == g.exato).sum()),
                     estaveis_med=float(g.estaveis_trecho.median()), estaveis_max=int(g.estaveis_trecho.max()),
                     falhas=g.loc[~g.exato, "porque"].value_counts().to_dict())
        resumo[modo] = d
        print(f"   {modo:16s} exatos {d['exatos']}/{d['n']} | {d['s_med']:.3f} s (p95 {d['s_p95']:.3f}) | "
              f"pico {d['pico_med_mb']:.0f} MB" + (f" | regra acerta {d['regra_acerta']}/{d['n']}" if 'regra_acerta' in d else ""), flush=True)
    res.update(resumo=resumo, registros=regs, duracao_min=round((time.time() - t00) / 60, 1))
    (SAIDA / "j7_resultado.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    from clearml import Task as Tk
    tk = Tk.current_task()
    if tk is not None:
        tk.upload_artifact("j7_resultado", artifact_object=str(SAIDA / "j7_resultado.json"))


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
