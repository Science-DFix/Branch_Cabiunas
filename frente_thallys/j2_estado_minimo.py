#!/usr/bin/env python3
"""J2 -- decompor a memória do detector v2: janela curta + estado pequeno no lugar dos 60-90 d de bruto. Uma task.

PRÉ-REGISTRO (05/11/2026, commitado antes de enviar)
Origem: perguntas 1 e 4 da engenharia (`tc33003a-janela-historica-e-gargalos-de-inferencia.md`): que componente pede
  histórico, e que estado persistir para não reconstruir tudo a cada execução; e o limite de 1 milhão de células do
  `FEATURE_HISTORY_MAX_CELLS`, que 60 e 90 d estouram. O J1 (05/11) mostrou que t, p e sp nunca divergem nem com 30 d e
  que o único componente que pede histórico longo é a referência de 400 h estáveis do vb.
  O que guarda memória no código (`cabiunas_inference.py`), conferido:
    vb ......... base = 12.000 amostras ESTÁVEIS (400 h) terminando 720 amostras estáveis (guarda 24 h) antes do bloco
                 de 6 h -> o estado é um buffer das últimas ~13.000 linhas estáveis das 10 sondas TV_ (130 mil células);
    EWMA ....... meia-vida 1 h (t, p) e 30 min (sp, vb): esquece 2^-24 em 1 dia;
    blackout ... 6 h; CUSUM ... multiplicado por 0,25 a cada instante fora da máscara;
    decisão .... refratário 72 h, escalada por idade 96 h, junção de episódios a 2 h, duração mínima.
Pergunta: com o vb alimentado por esse buffer, qual a menor janela de bruto que dá o mesmo resultado que a janela de
  60 d (= histórico completo bit a bit no J1)? E se, além disso, se persistir a saída do `preprocessar` (6 colunas por
  linha), quantos dias dela o `prever` precisa?

INSUMOS: os do J1 (Dataset f5e1b16b + bruto a97ba56b), mesma grade, bundles e trips; reaproveita monta/prepara/regua
  do `j1_janela_entrada.py`.
REPLAY: cortes a cada 6 h desde 2025-01-08 07:34 (+ subcortes nos trips, cada trecho só conhece os trips anteriores);
  em cada corte guardam-se as linhas novas desde o corte anterior. 6 h e não 1 dia: com janela de 1 d e corte diário a
  linha guardada mais nova teria 0 h de aquecimento do EWMA; com 6 h tem >= 18 h. Variantes:
    ref60 ..... `preprocessar` + `prever` sobre 60 d de bruto = REFERÊNCIA (o mesmo corte de 6 h);
    a1/a3/a7/a14  bruto de N d; o vb usa o buffer de 13.000 linhas estáveis anteriores à janela (estado A);
    a7_sem .... bruto de 7 d sem o buffer (controle: o que o estado do vb compra);
    b7/b14/b30/b45/b60  bruto de 3 d com o buffer do vb; as linhas novas do `preprocessar` (t, p, sp, vb, mask,
                reset) são anexadas a uma tabela persistida, e o `prever` roda sobre os últimos M d dela (estado B).
                b60 deve reproduzir ref60 por construção (mesma série de sinais, mesma janela de decisão).
  O vb do estado é calculado pela PRÓPRIA `_z_vibracao` do pacote sobre [buffer + janela] (monkeypatch), não por
  reimplementação.
MEDIDAS (desde 2025-02-01, contra ref60): % de instantes com sinal diferente (|Δ| > 1e-9 ou NaN diferente) e máx |Δ|
  por sinal; % de instantes com alarme diferente; régua (det, início, banda, FP, FP/mês, antecedência por trip).
  EQUIVALENTE NOS SINAIS / NA OPERAÇÃO = como no J1. CUSTO (serial, depois dos replays): ref60, a7, b14, b45, b60 em 20
  instantes sorteados (semente 20261105) desde 2025-04-01: tempo de preprocessar + prever (perf_counter, com a
  montagem do buffer), pico tracemalloc e células de entrada (bruto + buffer + tabela persistida).
VERIFICAÇÃO: V1 o vb das variantes a* com estado é igual ao ref60 em 100% dos instantes (se não, o buffer está errado e
  a leitura das variantes com estado para). V2 (relato) régua do ref60 a 6 h contra o J1 (7/8, FP 9 com corte diário).
TESTE LOCAL DE ENCANAMENTO (05/11, até 2025-04-15, ANTES deste commit; mudou o desenho, registrado aqui): V1 passou
  (vb do estado = ref60; a1 difere só na 9ª casa, EWMA); t, p, sp bit a bit desde a3; mas o ALARME diverge 3% até com
  a14 e b14 (FP 1 -> 2 e 3), tudo em 27/03-07/04/2025, depois de um episódio de 196 h (09-17/03). Causa: o refratário é
  armado só pelos episódios EMITIDOS, então a cadeia de bloqueios depende de onde a janela de decisão começa -- a memória
  da decisão não é 72 h, é a cadeia. Por isso entraram b30, b45 e b60 (a tabela de 6 colunas e 60 d tem 259 mil células).
EXPECTATIVAS: V1 passa (~90%). t, p, sp: a1 difere de ref60 mas com máx |Δ| < 1e-4 (~60%); a3 em diante bit a bit
  (~80%). Alarme: a1 e a3 divergem por perderem o refratário e o começo dos episódios (~70%); a7 equivalente na
  operação (~50%); a14 bit a bit no alarme (~50%). a7_sem: vb NaN na maior parte, régua pior que ref60 (~85%). b7 e b14
  NÃO equivalentes (~80%, visto no teste local); b30 equivalente na operação (~40%); b45 equivalente na operação (~60%);
  b60 bit a bit no alarme (~90%). Custo: b60 mediana <= metade do ref60 (~60%); b60 abaixo de 1 milhão
  de células (~95%; ~470 mil = 84 mil de bruto + 130 mil de buffer + 259 mil de tabela).
LIMITES: 8 trips; o buffer é montado em memória a partir da grade completa (em produção seria uma tabela); a montagem
  do buffer recalcula blocos do vb que ficam fora da janela (custo é teto, não piso); o Cica não é o servidor do SIMPred.

Uso:
  python frente_thallys/j2_estado_minimo.py --local --teste     # encanamento local, até 2025-04-15
  python frente_thallys/j2_estado_minimo.py --enqueue           # uma task na fila default
"""
from clearml import Task  # noqa: F401  (antes do argparse: o worker injeta argumentos por patch)
import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import time
import tracemalloc
from multiprocessing import get_context
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import j1_janela_entrada as j1   # noqa: E402

NOME = "frente_thallys | J2 estado mínimo do v2"
PASSO_DIAS = 0.25
N_BUF = 13_000          # 12.000 de base + 720 de guarda + 180 de um bloco de 6 h + folga
VARIANTES = {
    "ref60": dict(n=60, estado=False, m=None),
    "a1": dict(n=1, estado=True, m=None), "a3": dict(n=3, estado=True, m=None),
    "a7": dict(n=7, estado=True, m=None), "a14": dict(n=14, estado=True, m=None),
    "a7_sem": dict(n=7, estado=False, m=None),
    "b7": dict(n=3, estado=True, m=7), "b14": dict(n=3, estado=True, m=14),
    "b30": dict(n=3, estado=True, m=30), "b45": dict(n=3, estado=True, m=45), "b60": dict(n=3, estado=True, m=60),
}
CUSTO = ["ref60", "a7", "b14", "b45", "b60"]
SAIDA = Path(tempfile.gettempdir()) / "j2_saida"
G = j1.G
COLS = ["t", "p", "sp", "vb", "mask", "reset"]


# ───────────────────────────── estado do vb ─────────────────────────────
def prepara_estado():
    """Linhas estáveis da grade completa (só as sondas) e o patch da `_z_vibracao`."""
    ci = G["ci"]
    g = ci._grade(G["df"]); _, est, _, _ = ci._mascara(g)
    tags = G["det"]["tags_vibracao"]
    G["vib_est"] = g.loc[est.to_numpy(), tags]
    G["orig_z"] = ci._z_vibracao
    G["usa_estado"] = False
    ci._z_vibracao = _z_com_estado


def _z_com_estado(g, estavel, tags, trips, excl):
    import numpy as np
    import pandas as pd
    orig = G["orig_z"]
    if not G["usa_estado"]:
        return orig(g, estavel, tags, trips, excl)
    V = G["vib_est"]
    pos = V.index.searchsorted(g.index[0])
    buf = V.iloc[max(0, pos - N_BUF):pos]
    gb = pd.concat([buf[tags], g[tags]])
    eb = pd.concat([pd.Series(True, index=buf.index), estavel.astype(bool)])
    z = orig(gb, eb, tags, trips, excl)
    return np.asarray(z)[len(buf):]


# ───────────────────────────── replay ─────────────────────────────
def _proc(fim, n_dias, estado, conhecidos):
    import pandas as pd
    ci, df = G["ci"], G["df"]
    G["usa_estado"] = estado
    w = df.loc[max(df.index[0], fim - pd.Timedelta(days=n_dias)):fim]
    with contextlib.redirect_stdout(io.StringIO()):
        return w, ci.preprocessar(G["modelos"], w, trips=conhecidos)


def _prever(proc):
    with contextlib.redirect_stdout(io.StringIO()):
        return G["ci"].prever({"detector": G["det"]}, proc)


def replay(chave):
    import pandas as pd
    v, trips = VARIANTES[chave], G["trips"]
    pedacos, tabela, anterior, t0 = [], [], None, time.time()
    for c in j1.cortes(PASSO_DIAS):
        sub = [f for f in trips if anterior is not None and anterior < f < c] + [c]
        for s in sub:
            conhecidos = trips if s == c else [t for t in trips if t < s]
            _, proc = _proc(s, v["n"], v["estado"], conhecidos)
            novo = proc[proc.index > anterior] if anterior is not None else proc
            if v["m"] is None:
                res = _prever(proc)
            else:   # estado B: tabela persistida da saída do preprocessar, o prever lê os últimos M dias
                tabela.append(novo[COLS])
                ini = s - pd.Timedelta(days=v["m"])
                while len(tabela) > 1 and tabela[0].index[-1] < ini:
                    tabela.pop(0)
                hist = pd.concat(tabela).loc[ini:]
                res = _prever(hist)
            out = novo[COLS].join(res[["is_anomaly", "vigiado"]], how="left")
            pedacos.append(out); anterior = s
    r = pd.concat(pedacos); r = r[~r.index.duplicated(keep="first")]
    r.to_parquet(SAIDA / f"replay_{chave}.parquet")
    print(f"   replay {chave}: {len(r)} linhas, {time.time() - t0:.0f} s", flush=True)
    return chave


def compara(A, R):
    import numpy as np
    A = A.loc["2025-02-01":]; R = R.reindex(A.index)
    out = {}
    for s in ("t", "p", "sp", "vb"):
        a, b = A[s].to_numpy(float), R[s].to_numpy(float)
        nan_dif = np.isnan(a) != np.isnan(b); ok = ~np.isnan(a) & ~np.isnan(b)
        d = np.zeros_like(a); d[ok] = np.abs(a[ok] - b[ok])
        out[f"{s}_pct"] = round(100 * float(((d > 1e-9) | nan_dif).mean()), 3)
        out[f"{s}_max"] = float(d.max()) if ok.any() else 0.0
        out[f"{s}_nan_pct"] = round(100 * float(nan_dif.mean()), 3)
    out["alarme_pct"] = round(100 * float((A["is_anomaly"].astype(bool) != R["is_anomaly"].astype(bool)).mean()), 3)
    out["mask_pct"] = round(100 * float((A["mask"].astype(bool) != R["mask"].astype(bool)).mean()), 3)
    return out


# ───────────────────────────── custo ─────────────────────────────
def custo(teste: bool):
    import numpy as np
    import pandas as pd
    df = G["df"]
    ref = pd.read_parquet(SAIDA / "replay_ref60.parquet")
    rng = np.random.default_rng(j1.SEMENTE)
    cand = df.loc["2025-04-01":].index if not teste else df.index[-200:]
    instantes = [pd.Timestamp(x) for x in sorted(rng.choice(cand, size=3 if teste else 20, replace=False))]
    L = []
    for chave in CUSTO:
        v = VARIANTES[chave]
        for fim in instantes:
            conhecidos = [t for t in G["trips"] if t < fim]
            tracemalloc.start(); t0 = time.perf_counter()
            w, proc = _proc(fim, v["n"], v["estado"], conhecidos)
            t1 = time.perf_counter()
            if v["m"] is None:
                _prever(proc); persist = 0
            else:
                hist = ref.loc[fim - pd.Timedelta(days=v["m"]):fim, COLS]
                hist = pd.concat([hist.iloc[:-1], proc[COLS].iloc[-1:]])
                _prever(hist); persist = int(hist.size)
            t2 = time.perf_counter()
            pico = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop()
            buf = N_BUF * len(G["det"]["tags_vibracao"]) if v["estado"] else 0
            L.append(dict(variante=chave, fim=str(fim), preproc_s=t1 - t0, prever_s=t2 - t1, total_s=t2 - t0,
                          pico_mb=pico, celulas_bruto=int(w.size), celulas_buffer=buf, celulas_tabela=persist,
                          celulas_total=int(w.size) + buf + persist))
    T = pd.DataFrame(L); T.to_csv(SAIDA / "custo.csv", index=False)
    R = {}
    for k, x in T.groupby("variante", sort=False):
        med = float(x.total_s.median())
        R[k] = dict(total_med_s=round(med, 3), total_p95_s=round(float(x.total_s.quantile(.95)), 3),
                    preproc_med_s=round(float(x.preproc_s.median()), 3), prever_med_s=round(float(x.prever_s.median()), 3),
                    pico_med_mb=round(float(x.pico_mb.median()), 1), celulas_total=int(x.celulas_total.median()),
                    celulas_bruto=int(x.celulas_bruto.median()), celulas_buffer=int(x.celulas_buffer.median()),
                    celulas_tabela=int(x.celulas_tabela.median()),
                    cpu_s_dia_cada_2min=round(720 * med, 1), cpu_s_dia_cada_15min=round(96 * med, 1))
    return R


# ───────────────────────────── principal ─────────────────────────────
def executa(local: bool, insumos: str | None, teste: bool, procs: int):
    import pandas as pd
    SAIDA.mkdir(parents=True, exist_ok=True)
    t00 = time.time()
    pk, a97, _ = j1.monta(local, insumos)
    j1.prepara(pk, a97, teste)
    prepara_estado()
    res = dict(ambiente=j1.ambiente()); print("ambiente:", res["ambiente"], flush=True)

    print("REPLAYS (em paralelo)", flush=True)
    with get_context("fork").Pool(min(procs, len(VARIANTES))) as pool:
        list(pool.imap_unordered(replay, list(VARIANTES)))
    ref = pd.read_parquet(SAIDA / "replay_ref60.parquet")
    res["regua"], res["divergencia"] = {}, {}
    for k in VARIANTES:
        x = pd.read_parquet(SAIDA / f"replay_{k}.parquet")
        res["regua"][k] = j1.regua(x["is_anomaly"].astype(bool), x["vigiado"].astype(bool))
        if k != "ref60":
            res["divergencia"][k] = compara(x, ref)
    res["V1_vb_estado_igual"] = {k: res["divergencia"][k]["vb_pct"] == 0 for k in VARIANTES if VARIANTES[k]["estado"]}
    rr = res["regua"]["ref60"]
    chave_op = lambda r: (r["det"], r["inicio"], r["banda"], r["fp"], tuple(sorted(r["lead"].items())))
    res["equivalente_operacao"] = {k: chave_op(res["regua"][k]) == chave_op(rr) for k in VARIANTES if k != "ref60"}
    res["equivalente_sinais"] = {k: all(v == 0 for kk, v in res["divergencia"][k].items() if kk.endswith("_pct"))
                                 for k in VARIANTES if k != "ref60"}
    for k in VARIANTES:
        r = res["regua"][k]; d = res["divergencia"].get(k, {})
        print(f"   {k:>6s}: {r['det']}/8 início {r['inicio']} banda {r['banda']} FP {r['fp']} | "
              + " ".join(f"{s} {d.get(s + '_pct', 0)}%" for s in ("t", "p", "sp", "vb", "alarme")), flush=True)
    print("V1 (vb do estado = ref60):", res["V1_vb_estado_igual"], flush=True)

    print("CUSTO (serial)", flush=True)
    res["custo"] = custo(teste); print(json.dumps(res["custo"], indent=1), flush=True)
    res["duracao_min"] = round((time.time() - t00) / 60, 1)
    salva(res)


def salva(res):
    (SAIDA / "j2_resultado.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    from clearml import Task as T
    tk = T.current_task()
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
        import subprocess
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
