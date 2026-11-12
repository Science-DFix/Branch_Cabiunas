#!/usr/bin/env python3
"""J10 -- custo do detector v2 SEM tracemalloc: tempo e RSS do lote (30/45/60/90 d) e do motor, com e sem a grade.

PRÉ-REGISTRO (11/11/2026, commitado antes de enviar)
Origem: a avaliação do Thallys (11/11) aponta que os números de custo da resposta à engenharia não batem com a
  telemetria da release/v2.1 (memória 213/319 MB contra 167 MB com 90 d; "14 s para montar 90 d de grade" contra 5,4 s
  da execução inteira). Conferido no código: o J1 mediu tempo e memória com o tracemalloc LIGADO
  (`j1_janela_entrada.py`, parte B), o que deixa o Python mais lento e mede alocação do Python, não o RSS; a tabela da
  seção 6 misturava isso com o RSS do motor (J34). O J34 já mediu lote 60 d (0,33 s, RSS 201 MB) e motor (0,15 s,
  RSS 173 MB) do jeito certo, mas não as outras janelas nem a grade.
Pergunta: com a mesma medida para tudo (processo limpo, perf_counter, pico de RSS = VmHWM, sem tracemalloc), quanto
  custam o lote de 30/45/60/90 d e o motor, sem e com a montagem da grade a partir do export de 30 s?

INSUMOS: os do J1/J34 (Dataset f5e1b16b = pacote e17bf27 + nv d77de7d; bruto 30 s a97ba56b). Motor = `J34.Motor`
  sem alteração, estado montado com 60 d como no J34.
PROCEDIMENTO. 20 instantes sorteados (semente 20261105, como o custo do J34) desde 2025-04-01. Em cada um, cada modo
  roda num subprocesso novo (os mesmos imports do J34), que mede o RSS antes do cálculo e o pico ao fim:
  - imports: só os imports;
  - lote_N (N = 30, 45, 60, 90): `preprocessar` + `prever` sobre a grade de 2 min de N d (grade já pronta);
  - lote_bruto_N (N = 60, 90): `preparar_grade` sobre o bruto de 30 s de N d + `preprocessar` + `prever`;
  - motor: um passo do motor (3 d de grade, estado de disco);
  - motor_bruto: `preparar_grade` sobre 3 d de bruto de 30 s (só janelas fechadas) + um passo do motor.
  Tempo por fase (grade, cálculo) e total, sem a leitura dos arquivos. Mediana e p95 nos 20 instantes.
  Conferência: o lote_60 tem de dar o mesmo da medida do J34 (0,33 s; RSS 201 MB) dentro de ±20%.
EXPECTATIVAS: lote_60 0,25–0,45 s e RSS 180–230 MB (~80%, = J34); lote_90 <= 0,7 s (~75%); RSS de lote_90 <= 250 MB
  (~70%); grade de 90 d a partir de 30 s <= 2 s (~75%; o docstring do pacote diz ~1,1 s) -- ou seja, os 14 s do J1
  eram efeito do tracemalloc; lote_bruto_90 <= 3 s (~70%); motor_bruto <= 0,3 s (~70%).
LIMITES: Cica, não o servidor do SIMPred; leitura do banco fora; RSS inclui os imports (pandas, numpy, scikit-learn,
  clearml), por isso também se reporta o acréscimo sobre o RSS dos imports.

Uso:
  python frente_thallys/j10_custo_sem_tracemalloc.py --local --teste   # encanamento local
  python frente_thallys/j10_custo_sem_tracemalloc.py --enqueue         # uma task na fila default
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

NOME = "frente_thallys | J10 custo sem tracemalloc"
LOTES, LOTES_BRUTO = [30, 45, 60, 90], [60, 90]
SAIDA = Path(tempfile.gettempdir()) / "j10_saida"
G = j1.G
D = J.D

FILHO = r"""
import sys, time, json
def hwm():
    return next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmHWM")) / 1024
modo, pasta = sys.argv[1], sys.argv[2]
sys.path[:0] = sys.argv[3].split(":")
import pandas as pd, numpy as np
import j34_estado_robustez as J
import cabiunas_inference as ci
J.G.update(ci=ci, modelos=ci.carregar_modelos(sys.argv[4]))
J.G["det"] = J.G["modelos"][-1]["detector"]; J.G["orig_z"] = ci._z_vibracao; ci._z_vibracao = J._z_com_buffer
trips = [pd.Timestamp(t) for t in json.loads(open(pasta + "/trips.json").read())]
r0 = hwm(); tg = 0.0; tc = 0.0
if modo.startswith("lote_bruto_"):
    w = pd.read_parquet(pasta + f"/bruto{modo.split('_')[-1]}.parquet")
    t0 = time.perf_counter(); g = ci.preparar_grade(w); t1 = time.perf_counter()
    J._prever(J._preproc(g, trips, None)); t2 = time.perf_counter(); tg, tc = t1 - t0, t2 - t1
elif modo.startswith("lote_"):
    g = pd.read_parquet(pasta + f"/grade{modo.split('_')[-1]}.parquet")
    t1 = time.perf_counter(); J._prever(J._preproc(g, trips, None)); tc = time.perf_counter() - t1
elif modo.startswith("motor"):
    m = J.Motor(); m.tab = pd.read_parquet(pasta + "/tab.parquet").iloc[:-1]; m.buf = pd.read_parquet(pasta + "/buf.parquet")
    m.ultimo = m.tab.index[-1]
    if modo == "motor_bruto":
        w = pd.read_parquet(pasta + "/bruto3.parquet")
        t0 = time.perf_counter(); g = ci.preparar_grade(w); tg = time.perf_counter() - t0
    else:
        g = pd.read_parquet(pasta + "/grade3.parquet")
    t1 = time.perf_counter(); m.passo(g, g.index[-1], trips); tc = time.perf_counter() - t1
print(json.dumps(dict(rss_mb=hwm(), rss_antes_mb=r0, grade_s=tg, calculo_s=tc, total_s=tg + tc)))
"""


def executa(local: bool, insumos: str | None, teste: bool):
    import numpy as np
    import pandas as pd
    SAIDA.mkdir(parents=True, exist_ok=True); t00 = time.time()
    pk, a97, _ = j1.monta(local, insumos)
    j1.prepara(pk, a97, teste)
    ci = G["ci"]; G["orig_z"] = ci._z_vibracao; ci._z_vibracao = J._z_com_buffer
    J.prepara_cenarios()
    res = dict(ambiente=j1.ambiente()); print("ambiente:", res["ambiente"], flush=True)
    df, raw = G["df"], G["raw"]
    # o bruto traz texto do PI ("No Data"...) em colunas object: o parquet pede tipo único -> texto (o
    # `preparar_grade` do filho converte, como faria com o export real)
    obj = [c for c in raw.columns if raw[c].dtype == object]
    if obj:
        raw = raw.astype({c: str for c in obj})
    lotes = LOTES if not teste else [30, 60]
    brutos = LOTES_BRUTO if not teste else [60]
    rng = np.random.default_rng(j1.SEMENTE)
    cand = df.loc[pd.Timestamp("2025-04-01", tz="UTC"):].index if not teste else df.index[-200:]
    inst = sorted(cand[i] for i in rng.choice(len(cand), size=3 if teste else 20, replace=False))
    filho = SAIDA / "filho.py"; filho.write_text(FILHO)
    caminho = ":".join([str(AQUI), str(pk / "scripts")])
    modos = ["imports"] + [f"lote_{n}" for n in lotes] + [f"lote_bruto_{n}" for n in brutos] + ["motor", "motor_bruto"]
    L = []
    for fim in inst:
        trips = J.conhecidos("limpo", fim)
        pasta = Path(tempfile.mkdtemp(prefix="j10_"))
        m = J.Motor(); m.carga(df.loc[fim - D("60D"):fim], trips, fim)
        m.tab.to_parquet(pasta / "tab.parquet"); m.buf.to_parquet(pasta / "buf.parquet")
        df.loc[fim - D("3D"):fim].to_parquet(pasta / "grade3.parquet")
        # bruto só até o fim da última janela de 2 min (janelas fechadas): [fim - N, fim + 2 min)
        fecha = fim + D("2min") - D("1ns")
        raw.loc[fim - D("3D"):fecha].to_parquet(pasta / "bruto3.parquet")
        for n in lotes:
            df.loc[fim - D(f"{n}D"):fim].to_parquet(pasta / f"grade{n}.parquet")
        for n in brutos:
            raw.loc[fim - D(f"{n}D"):fecha].to_parquet(pasta / f"bruto{n}.parquet")
        (pasta / "trips.json").write_text(json.dumps([str(t) for t in trips]))
        for modo in modos:
            # "nada": o filho só faz os imports e carrega os bundles, sem cálculo
            o = subprocess.run([sys.executable, str(filho), modo if modo != "imports" else "nada", str(pasta), caminho,
                                str(pk / "modelos")], capture_output=True, text=True)
            if o.returncode != 0:
                L.append(dict(fim=str(fim), modo=modo, erro=o.stderr[-400:])); continue
            r = json.loads(o.stdout.strip().splitlines()[-1])
            L.append(dict(fim=str(fim), modo=modo, celulas=None, **r))
        print(f"   {fim}: " + " ".join(f"{x['modo']}={x.get('total_s', float('nan')):.2f}s/{x.get('rss_mb', float('nan')):.0f}MB"
                                       for x in L if x["fim"] == str(fim)), flush=True)
    C = pd.DataFrame(L); C.to_csv(SAIDA / "j10_custo.csv", index=False)
    imp = C.loc[C.modo == "imports", "rss_mb"].median()
    resumo = {}
    for modo, g in C.groupby("modo", sort=False):
        if "erro" in g and g["erro"].notna().any():
            resumo[modo] = dict(erros=g["erro"].dropna().unique().tolist()[:2])
            continue
        resumo[modo] = dict(n=len(g), total_s_med=round(float(g.total_s.median()), 3),
                            total_s_p95=round(float(g.total_s.quantile(0.95)), 3),
                            grade_s_med=round(float(g.grade_s.median()), 3), calculo_s_med=round(float(g.calculo_s.median()), 3),
                            rss_mb_med=round(float(g.rss_mb.median()), 1), rss_mb_max=round(float(g.rss_mb.max()), 1),
                            acima_imports_mb=round(float(g.rss_mb.median() - imp), 1))
        r = resumo[modo]
        print(f"   {modo:14s} total {r['total_s_med']:.3f} s (p95 {r['total_s_p95']:.3f}) | grade {r['grade_s_med']:.3f} | "
              f"cálculo {r['calculo_s_med']:.3f} | RSS {r['rss_mb_med']:.0f} MB (máx {r['rss_mb_max']:.0f}; "
              f"+{r['acima_imports_mb']:.0f} sobre imports)", flush=True)
    res.update(resumo=resumo, duracao_min=round((time.time() - t00) / 60, 1))
    (SAIDA / "j10_resultado.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    from clearml import Task as Tk
    tk = Tk.current_task()
    if tk is not None:
        tk.upload_artifact("j10_resultado", artifact_object=str(SAIDA / "j10_resultado.json"))
        tk.upload_artifact("j10_custo", artifact_object=str(SAIDA / "j10_custo.csv"))


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
