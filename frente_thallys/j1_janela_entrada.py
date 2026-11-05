#!/usr/bin/env python3
"""J1 -- janela de entrada do detector v2 do Thallys: qualidade (quanto histórico) e custo no Cica. Uma task.

PRÉ-REGISTRO (05/11/2026, commitado antes de enviar)
Origem: o documento da engenharia (`tc33003a-janela-historica-e-gargalos-de-inferencia.md`, recebido em 05/11) pede
  à equipe de modelos: que componente exige o maior histórico, o menor histórico validado para o último ponto, a
  tolerância e o orçamento de RAM/latência, medidos por replay cronológico com bundles mensais. O Thallys já mediu
  parte disso (`INTEGRACAO_DASHBOARD.md` e cabeçalho do `cabiunas_inference.py`, 30/09–08/10): 45 d igual ao histórico
  completo em 61 execuções semanais, 30 d diverge em até 17,1%; 60 d nunca fica abaixo das 400 h estáveis; 90 d
  recomendado; job a cada 15 min, 0,74 s e 20 MB com 90 d no PC dele.
  O que falta: (i) o pacote atual (`preprocessar` mudou em 07/10); (ii) todos os instantes, não 61 execuções;
  (iii) se a divergência muda os ALARMES; (iv) o instante em que o alerta pode ser emitido ao vivo, já que o `prever`
  preenche o episódio para trás; (v) custo medido no cluster, não num PC.
Pergunta: qual é a menor janela de entrada que dá o mesmo resultado que o histórico completo, nos sinais e nos alarmes,
  e quanto custa cada janela no Cica?

INSUMOS: pacote de produção da `feat/pacote-consolidado` (scripts, 16 bundles do git, registro de trips), o harness
  `src/nv` da `exp/modos-de-falha`, a série do controle publicado (`controle_v2/atual`, Dataset 86e3990f) -> Dataset de
  insumos do J1; bruto de 30 s = Dataset a97ba56b (CSV de produção = cabeçalho + linhas desde 2025-01-01, md5 igual ao
  local; complemento de set–dez/2024 como no `reproduz.sinais`).

PARTE A, qualidade. Replay com corte DIÁRIO desde 2025-01-08 07:34 (+ subcortes nos trips, como `sinais_causais`: cada
  trecho só conhece os trips anteriores). Em cada corte, `preprocessar` e `prever` rodam SÓ sobre a janela [corte − N,
  corte], como em produção, e guardam-se as linhas novas desde o corte anterior. N = 30, 35, 45, 60, 90 d e "inf"
  (todo o histórico disponível desde 2024-09-01) = REFERÊNCIA.
  Medidas, desde 2025-02-01, contra a referência: % de instantes com sinal diferente (|Δ| > 1e-9 ou NaN diferente) e
  máx |Δ|, por sinal (t, p, sp, vb); % de instantes com alarme diferente; régua do Thallys (det, início, banda, FP, FP/mês,
  antecedência por trip). Horas estáveis (máscara `estavel` do pacote) nos N dias anteriores a cada instante: mínimo e
  % do tempo abaixo de 400 h e de 100 h.
  EQUIVALENTE NOS SINAIS = 0 instantes diferentes nos 4 sinais e na série de alarmes.
  EQUIVALENTE NA OPERAÇÃO = régua idêntica (det, início, banda, FP e antecedência de cada trip ao minuto).
  Reporta-se a menor N de cada tipo.
PARTE B, custo no Cica (serial, antes das outras). Para N = 30, 45, 60, 90: 20 instantes sorteados (semente
  20261105) entre 2025-04-01 e o fim. Em cada um, tempo de `preparar_grade` sobre o bruto de 30 s da janela, de
  `preprocessar` e de `prever` (perf_counter), pico de memória Python (tracemalloc), linhas e células da janela. Saem
  mediana e p95, e o custo por dia por equipamento: 720 execuções (a cada linha de 2 min, como a integração atual)
  contra 96 (a cada 15 min, como recomenda o Thallys). Registra CPU, núcleos, RAM e workers ocupados.
PARTE C, emissão ao vivo (N = 60). Para cada episódio da série N = 60 desde 2025-02-01: execuções a cada 15 min, de
  4 h antes do início em lote até o fim do episódio (no máx. 12 h depois do início), com a janela de 60 d terminando em
  cada instante (a busca começa depois do fim do episódio anterior). Execução "acesa" = alarme em alguma linha dos
  últimos 15 min. Emissão = 1ª passagem de apagada para acesa; se fica acesa sem interrupção desde o episódio anterior,
  o episódio é "emendado" (ao vivo é o mesmo alerta). Mede o atraso emissão − início em lote, os emendados e, para os
  trips, a antecedência ao vivo e se fica na banda de 4 h.
VERIFICAÇÕES: V1 o controle publicado refeito aqui (sinais com janela de 75 d semanal + `prever` na série inteira,
  `reproduz.controle`) dá 7/8, FP 9 -- se falhar, para antes da A e da C. V2 (relato) concordância instante a instante
  da série de alarme do V1 com o `controle_v2/atual` do 86e3990f. V3 (relato) régua da referência "inf" diária contra
  o V1: o corte diário com decisão dentro da janela não é o mesmo procedimento do controle publicado.
EXPECTATIVAS: equivalente nos sinais: 90 d (~90%), 60 d (~80%), 45 d (~55%), 35 d (~15%), 30 d (~10%). Equivalente na
  operação: 45 d (~70%), 35 d (~40%), 30 d (~25%). Menor N equivalente nos sinais = 45 d (~45%). O componente que
  quebra primeiro é o vb (~75%); o CUSUM aparece em 35–45 d (~30%). Parte C: atraso mediano de emissão entre 1,5 e
  2,5 h (~60%); pelo menos 1 trip sai da banda de 4 h ao vivo (~50%). Parte B: 90 d com mediana total <= 1,5 s (~60%)
  e pico <= 200 MB (~70%).
LIMITES: 8 trips; os bundles são os do git (o último decide, como no controle); o Cica não é o servidor do SIMPred e a
  carga dos outros usuários mexe no tempo; tracemalloc mede memória Python/numpy, não o RSS do processo.

Uso:
  python frente_thallys/j1_janela_entrada.py --cria-insumos         # Dataset de insumos (1 vez)
  python frente_thallys/j1_janela_entrada.py --local --teste        # encanamento local, poucos cortes
  python frente_thallys/j1_janela_entrada.py --enqueue              # uma task na fila default
"""
from clearml import Task  # noqa: F401  (antes do argparse: o worker injeta argumentos por patch)
import argparse
import contextlib
import io
import json
import os
import platform
import sys
import tempfile
import time
import tracemalloc
from multiprocessing import get_context
from pathlib import Path

AQUI = Path(__file__).resolve().parent
NOME = "frente_thallys | J1 janela de entrada do v2"
DS_A97 = "a97ba56ba14840fbb1125c2a82f883c9"
DS_INSUMOS = "f5e1b16b56324d6ab242e49bce224d07"   # criado por --cria-insumos em 05/11/2026
LOCAL_PKG = AQUI / "dados" / "nvdados" / "estrutura_pra_prod" / "Cabiunas"
LOCAL_NV = Path.home() / "FRENTE_THALLYS" / "repro_thallys_regra_carga_9fp_4fp" / "src" / "nv"
LOCAL_ATUAL = Path.home() / "FRENTE_THALLYS/clearml_fixos/ds_86e3990fde354c039ce38aab508f344d/controle_v2/atual/serie.csv.gz"
LOCAL_A97 = Path.home() / "FRENTE_THALLYS/clearml_fixos" / f"ds_{DS_A97}" / "sensores_full_2024_2026_30s.csv"
PACOTES = ["numpy==2.3.5", "pandas==2.3.3", "pyarrow==23.0.1", "scikit-learn==1.8.0", "scipy==1.16.3", "tzdata",
           "clearml"]
JANELAS = {"inf": None, "90": 90, "60": 60, "45": 45, "35": 35, "30": 30}
N_CUSTO = [30, 45, 60, 90]
SEMENTE = 20261105
SAIDA = Path(tempfile.gettempdir()) / "j1_saida"

G = {}   # estado global do processo (herdado pelos filhos via fork)


# ───────────────────────────── insumos e montagem ─────────────────────────────
def cria_insumos():
    from clearml import Dataset
    ds = Dataset.create(dataset_name="frente_thallys J1 insumos (pacote-consolidado + nv + controle atual)",
                        dataset_project="TesteMLCab")
    for nome in ("scripts", "modelos"):
        ds.add_files(str(LOCAL_PKG / nome), dataset_path=f"Cabiunas/{nome}", wildcard="*", recursive=True)
    ds.remove_files("Cabiunas/scripts/__pycache__/*"); ds.remove_files("Cabiunas/scripts/_pdf/*")
    ds.add_files(str(LOCAL_PKG / "registro_trips.csv"), dataset_path="Cabiunas")
    ds.add_files(str(LOCAL_NV), dataset_path="nv", recursive=True)
    ds.remove_files("nv/__pycache__/*"); ds.remove_files("nv/*/__pycache__/*")
    ds.add_files(str(LOCAL_ATUAL), dataset_path="controle_atual")
    ds.upload(); ds.finalize(); print("insumos J1:", ds.id, flush=True)


def monta(local: bool, insumos: str | None):
    """Árvore NV_DADOS (pacote + CSV de produção) e caminhos do nv; devolve (pkg, nv_pai, a97_csv, atual_csv)."""
    if local:
        pkg_src, nv_src, a97, atual = LOCAL_PKG, LOCAL_NV, LOCAL_A97, LOCAL_ATUAL
    else:
        from clearml import Dataset
        i = Path(Dataset.get(dataset_id=insumos).get_local_copy())
        pkg_src, nv_src, atual = i / "Cabiunas", i / "nv", i / "controle_atual" / "serie.csv.gz"
        a97 = Path(Dataset.get(dataset_id=DS_A97).get_local_copy()) / "sensores_full_2024_2026_30s.csv"
    r = Path(tempfile.mkdtemp(prefix="j1_nvdados_")); pk = r / "estrutura_pra_prod" / "Cabiunas"; pk.mkdir(parents=True)
    for nome in ("scripts", "modelos", "registro_trips.csv"):
        (pk / nome).symlink_to(pkg_src / nome)
    d = pk / "dados" / "2025_2026"; d.mkdir(parents=True)
    with open(a97) as f, open(d / "data_2025-01-01_2026-04-30_raw.csv", "w") as g:   # = CSV de produção local (md5)
        g.write(f.readline())
        for ln in f:
            if ln.startswith("2025-01-01 00:00:00"):
                g.write(ln); break
        for ln in f:
            g.write(ln)
    os.environ["NV_DADOS"] = str(r)
    sys.path[:0] = [str(nv_src.parent), str(pkg_src / "scripts")]
    return pk, a97, atual


def prepara(pk: Path, a97: Path, teste: bool):
    import pandas as pd
    import cabiunas_inference as ci
    x = pd.read_csv(a97, index_col=0, parse_dates=[0], dtype=str, nrows=1_100_000)
    x = x.apply(pd.to_numeric, errors="coerce"); x.index = x.index.tz_localize("UTC")
    extra = x.loc["2024-09-01":"2024-12-31 23:59:59"]
    with contextlib.redirect_stdout(io.StringIO()):
        raw = ci.carregar_dados(next((pk / "dados" / "2025_2026").glob("data_*_raw.csv")))
        raw = pd.concat([extra.loc[extra.index < raw.index[0], raw.columns], raw]).sort_index()
        df = ci.preparar_grade(raw)
        modelos = ci.carregar_modelos(pk / "modelos")
    trips = sorted(ci.carregar_trips(pk / "registro_trips.csv"))
    real = df["RUNNING_A"].notna() & df[[c for c in df.columns if c.startswith("TV_")]].notna().any(axis=1)
    df = df.loc[:df.index[real][-1]]
    if teste:
        df = df.loc[:"2025-04-15"]
    G.update(ci=ci, raw=raw, df=df, modelos=modelos, det=modelos[-1]["detector"], trips=trips, extra=extra)


# ───────────────────────────── replay ─────────────────────────────
def _roda(fim, n_dias, trips_conhecidos):
    ci, df = G["ci"], G["df"]
    ini = df.index[0] if n_dias is None else max(df.index[0], fim - __import__("pandas").Timedelta(days=n_dias))
    w = df.loc[ini:fim]
    with contextlib.redirect_stdout(io.StringIO()):
        proc = ci.preprocessar(G["modelos"], w, trips=trips_conhecidos)
        res = ci.prever({"detector": G["det"]}, proc)
    return proc, res


def cortes(passo_dias: float):
    import pandas as pd
    inicio = pd.Timestamp("2025-01-08 07:34", tz="UTC"); fim = G["df"].index[-1]
    c = [inicio]
    while c[-1] + pd.Timedelta(days=passo_dias) < fim:
        c.append(c[-1] + pd.Timedelta(days=passo_dias))
    return c + [fim]


def replay(chave: str):
    """Corte diário, `preprocessar` + `prever` dentro da janela, guarda as linhas novas (como `sinais_causais`)."""
    import pandas as pd
    n_dias, trips = JANELAS[chave], G["trips"]
    pedacos, anterior, t0 = [], None, time.time()
    for c in cortes(1.0):
        sub = [f for f in trips if anterior is not None and anterior < f < c] + [c]
        for s in sub:
            conhecidos = trips if s == c else [t for t in trips if t < s]
            proc, res = _roda(s, n_dias, conhecidos)
            out = proc[["t", "p", "sp", "vb", "mask", "reset"]].join(res[["is_anomaly", "vigiado"]])
            pedacos.append(out[out.index > anterior] if anterior is not None else out); anterior = s
    r = pd.concat(pedacos); r = r[~r.index.duplicated(keep="first")]
    r.to_parquet(SAIDA / f"replay_{chave}.parquet")
    print(f"   replay {chave}: {len(r)} linhas, {time.time() - t0:.0f} s", flush=True)
    return chave


def controle_publicado():
    """V1: `sinais_causais` (75 d, semanal) + `prever` na série inteira = `reproduz.controle`."""
    import pandas as pd
    from nv.controle_sinais import sinais_causais
    S = sinais_causais(pd.Timestamp("2025-01-08 07:34", tz="UTC"), cache=None, extra=G["extra"])
    S["mask"] = S["mask"].astype(bool); S["reset"] = S["reset"].astype(bool)
    with contextlib.redirect_stdout(io.StringIO()):
        res = G["ci"].prever({"detector": G["det"]}, S)
    return res["is_anomaly"].astype(bool), res["vigiado"].astype(bool)


# ───────────────────────────── medidas ─────────────────────────────
def regua(al, vig):
    import pandas as pd
    from nv.avalia import avalia
    from nv.causal import carrega_registro
    T = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]
    r = avalia(al, vig, T, inicio_periodo=pd.Timestamp("2025-02-01", tz="UTC"))
    return dict(det=r.det, inicio=r.inicio, banda=r.banda, fp=r.fp, fp_mes=round(r.fp_mes, 3),
                horas=round(r.horas_alarme), lead={k[:16]: (None if v["lead_h"] is None else round(v["lead_h"], 3))
                                                   for k, v in r.por_evento.items()})


def compara(chave, ref):
    import numpy as np
    import pandas as pd
    A = pd.read_parquet(SAIDA / f"replay_{chave}.parquet").loc["2025-02-01":]
    R = ref.reindex(A.index)
    out = {}
    for s in ("t", "p", "sp", "vb"):
        a, b = A[s].to_numpy(float), R[s].to_numpy(float)
        nan_dif = np.isnan(a) != np.isnan(b); ok = ~np.isnan(a) & ~np.isnan(b)
        d = np.zeros_like(a); d[ok] = np.abs(a[ok] - b[ok])
        out[f"{s}_pct"] = round(100 * float(((d > 1e-9) | nan_dif).mean()), 3)
        out[f"{s}_max"] = float(d.max()) if ok.any() else 0.0
    out["alarme_pct"] = round(100 * float((A["is_anomaly"].astype(bool) != R["is_anomaly"].astype(bool)).mean()), 3)
    out["mask_pct"] = round(100 * float((A["mask"].astype(bool) != R["mask"].astype(bool)).mean()), 3)
    return out


def horas_estaveis():
    import pandas as pd
    ci = G["ci"]
    _, est, _, _ = ci._mascara(ci._grade(G["df"]))
    h = est.astype(float) * (2 / 60)
    out = {}
    for k, n in JANELAS.items():
        if n is None:
            continue
        s = h.rolling(f"{n}D").sum().loc["2025-02-01":]
        out[k] = dict(min_h=round(float(s.min()), 1), quando_min=str(s.idxmin()), pct_abaixo_400=round(100 * float((s < 400).mean()), 2),
                      pct_abaixo_100=round(100 * float((s < 100).mean()), 2))
    return out


def ambiente():
    info = dict(python=platform.python_version(), plataforma=platform.platform(), nucleos=os.cpu_count())
    with contextlib.suppress(Exception):
        info["cpu"] = next(ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name"))
        info["ram_gb"] = round(int(next(ln.split()[1] for ln in open("/proc/meminfo") if ln.startswith("MemTotal"))) / 1e6, 1)
    with contextlib.suppress(Exception):
        from clearml.backend_api.session.client import APIClient
        info["workers_ocupados"] = sum(1 for w in APIClient().workers.get_all() if w.task)
    return info


def custo(teste: bool):
    import numpy as np
    import pandas as pd
    ci, df, raw = G["ci"], G["df"], G["raw"]
    rng = np.random.default_rng(SEMENTE)
    cand = df.loc["2025-04-01":].index if not teste else df.index[-200:]
    L = []
    for n in N_CUSTO:
        for fim in sorted(rng.choice(cand, size=3 if teste else 20, replace=False)):
            fim = pd.Timestamp(fim); ini = fim - pd.Timedelta(days=n)
            conhecidos = [t for t in G["trips"] if t < fim]
            tracemalloc.start()
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                g = ci.preparar_grade(raw.loc[ini:fim])
            t1 = time.perf_counter()
            w = df.loc[ini:fim]
            with contextlib.redirect_stdout(io.StringIO()):
                proc = ci.preprocessar(G["modelos"], w, trips=conhecidos)
                t2 = time.perf_counter()
                ci.prever({"detector": G["det"]}, proc)
            t3 = time.perf_counter()
            pico = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop()
            L.append(dict(n=n, fim=str(fim), grade_s=t1 - t0, preproc_s=t2 - t1, prever_s=t3 - t2, total_s=t3 - t1,
                          pico_mb=pico, linhas=len(w), celulas=int(w.size), linhas_grade=len(g)))
    T = pd.DataFrame(L); T.to_csv(SAIDA / "custo.csv", index=False)
    R = {}
    for n, x in T.groupby("n"):
        med = float(x.total_s.median())
        R[str(n)] = dict(total_med_s=round(med, 3), total_p95_s=round(float(x.total_s.quantile(.95)), 3),
                         preproc_med_s=round(float(x.preproc_s.median()), 3), prever_med_s=round(float(x.prever_s.median()), 3),
                         grade30s_med_s=round(float(x.grade_s.median()), 3), pico_med_mb=round(float(x.pico_mb.median()), 1),
                         pico_max_mb=round(float(x.pico_mb.max()), 1), linhas=int(x.linhas.median()), celulas=int(x.celulas.median()),
                         cpu_s_dia_cada_2min=round(720 * med, 1), cpu_s_dia_cada_15min=round(96 * med, 1))
    return R


def emissao_ao_vivo(teste: bool):
    import pandas as pd
    from nv.avalia import episodios_de_serie
    A = pd.read_parquet(SAIDA / "replay_60.parquet")
    al = A["is_anomaly"].astype(bool).loc["2025-02-01":]
    eps = episodios_de_serie(al)[: 2 if teste else None]
    T = [t for t in G["trips"] if t >= pd.Timestamp("2025-02-27", tz="UTC")]
    L, b_ant = [], None
    for a, b in eps:
        # a busca começa depois do episódio anterior: o alarme dele, ainda aceso, não é emissão deste
        piso = a - pd.Timedelta(hours=4) if b_ant is None else max(a - pd.Timedelta(hours=4), b_ant + pd.Timedelta(minutes=2))
        # emissão = 1ª passagem de apagado para aceso entre execuções; aceso sem interrupção desde o anterior = emendado
        fim_busca = min(b, a + pd.Timedelta(hours=12)); c = piso; emit = None; viu_apagado = False
        while c <= fim_busca:
            proc, res = _roda(c, 60, [t for t in G["trips"] if t < c])
            aceso = bool(res["is_anomaly"].astype(bool).loc[c - pd.Timedelta(minutes=15) + pd.Timedelta(seconds=1):c].any())
            if aceso and viu_apagado:
                emit = c; break
            viu_apagado |= not aceso
            c += pd.Timedelta(minutes=15)
        emendado = emit is None and not viu_apagado
        b_ant = b
        trip = next((t for t in T if a <= t + pd.Timedelta(hours=48) and b >= t - pd.Timedelta(hours=48) and a <= t), None)
        L.append(dict(inicio_lote=str(a), fim_lote=str(b), emendado=emendado, emissao=str(emit) if emit is not None else None,
                      atraso_h=None if emit is None else round((emit - a).total_seconds() / 3600, 2),
                      trip=str(trip) if trip is not None else None,
                      antecedencia_lote_h=None if trip is None else round((trip - a).total_seconds() / 3600, 2),
                      antecedencia_vivo_h=None if (trip is None or emit is None) else round((trip - emit).total_seconds() / 3600, 2)))
    pd.DataFrame(L).to_csv(SAIDA / "emissao_ao_vivo.csv", index=False)
    return L


# ───────────────────────────── principal ─────────────────────────────
def executa(local: bool, insumos: str | None, teste: bool, procs: int):
    import pandas as pd
    SAIDA.mkdir(parents=True, exist_ok=True)
    t00 = time.time()
    pk, a97, atual = monta(local, insumos)
    prepara(pk, a97, teste)
    amb = ambiente(); print("ambiente:", amb, flush=True)
    res = dict(ambiente=amb)

    v1 = dict(det=None, fp=None)
    if not teste:   # o teste local só confere o encanamento; o V1 refaz o controle inteiro
        al0, vig0 = controle_publicado()
        v1 = regua(al0, vig0); res["V1_controle_publicado"] = v1
        print(f"V1 controle publicado: {v1['det']}/8 FP {v1['fp']} ({v1['fp_mes']}/mês)", flush=True)
        s = pd.read_csv(atual); s["ts"] = pd.to_datetime(s["ts"], utc=True); s = s.set_index("ts")["alarme"].astype(bool)
        comum = al0.index.intersection(s.index)
        res["V2_concordancia_atual_pct"] = round(100 * float((al0.reindex(comum) == s.reindex(comum)).mean()), 4)
        print(f"V2 concordância com controle_v2/atual: {res['V2_concordancia_atual_pct']}% de {len(comum)} instantes", flush=True)
        if (v1["det"], v1["fp"]) != (7, 9):
            res["parada"] = "V1 falhou"; salva(res); print("V1 falhou -- parando"); return

    print("PARTE B (custo, serial)", flush=True)
    res["B_custo"] = custo(teste); print(json.dumps(res["B_custo"], indent=1), flush=True)

    print("PARTE A (replays em paralelo)", flush=True)
    with get_context("fork").Pool(min(procs, len(JANELAS))) as pool:
        list(pool.imap_unordered(replay, list(JANELAS)))
    ref = pd.read_parquet(SAIDA / "replay_inf.parquet")
    res["A_horas_estaveis"] = horas_estaveis()
    res["A_regua"] = {}; res["A_divergencia"] = {}
    for k in JANELAS:
        x = pd.read_parquet(SAIDA / f"replay_{k}.parquet")
        res["A_regua"][k] = regua(x["is_anomaly"].astype(bool), x["vigiado"].astype(bool))
        if k != "inf":
            res["A_divergencia"][k] = compara(k, ref)
    rr = res["A_regua"]["inf"]; chave_op = lambda r: (r["det"], r["inicio"], r["banda"], r["fp"], tuple(sorted(r["lead"].items())))
    eq_sin = [k for k in JANELAS if k != "inf" and all(v == 0 for kk, v in res["A_divergencia"][k].items() if kk.endswith("_pct"))]
    eq_op = [k for k in JANELAS if k != "inf" and chave_op(res["A_regua"][k]) == chave_op(rr)]
    res["A_menor_equivalente_sinais"] = min(eq_sin, key=int) if eq_sin else None
    res["A_menor_equivalente_operacao"] = min(eq_op, key=int) if eq_op else None
    res["V3_inf_vs_publicado"] = dict(inf=(rr["det"], rr["fp"]), publicado=(v1["det"], v1["fp"]))
    for k in JANELAS:
        r = res["A_regua"][k]; d = res["A_divergencia"].get(k, {})
        print(f"   {k:>3s}: {r['det']}/8 início {r['inicio']} banda {r['banda']} FP {r['fp']} | "
              + " ".join(f"{s} {d.get(s + '_pct', 0)}%" for s in ("t", "p", "sp", "vb", "alarme")), flush=True)
    print(f"menor janela equivalente: sinais {res['A_menor_equivalente_sinais']} | operação {res['A_menor_equivalente_operacao']}", flush=True)

    print("PARTE C (emissão ao vivo, N = 60)", flush=True)
    res["C_emissao"] = emissao_ao_vivo(teste)
    for e in res["C_emissao"]:
        print("  ", e, flush=True)
    res["duracao_min"] = round((time.time() - t00) / 60, 1)
    salva(res)


def salva(res):
    (SAIDA / "j1_resultado.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    from clearml import Task as T
    tk = T.current_task()
    if tk is not None:
        for f in sorted(SAIDA.glob("*")):
            tk.upload_artifact(f.stem, artifact_object=str(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cria-insumos", action="store_true"); ap.add_argument("--enqueue", action="store_true")
    ap.add_argument("--local", action="store_true"); ap.add_argument("--teste", action="store_true")
    ap.add_argument("--insumos", default=DS_INSUMOS); ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    if a.cria_insumos:
        return cria_insumos()
    if a.enqueue:
        assert a.insumos, "rode --cria-insumos e preencha DS_INSUMOS"
        t = Task.init(project_name="TesteMLCab", task_name=NOME)
        t.set_base_docker("python:3.11-slim", docker_setup_bash_script=["apt-get update -qq", "apt-get install -y -qq libgomp1"])
        t.set_packages(PACOTES); t.set_parameters({"General/insumos": a.insumos, "General/procs": str(a.procs)})
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
