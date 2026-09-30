#!/usr/bin/env python3
"""Avaliacao de MODELO DE NORMALIDADE sem loss de gradiente (PCA, SFA, CVA, ...).

Modelos estatisticos nao tem curva de loss. O que responde "o modelo esta
aprendendo o normal?" e o equivalente de treino x validacao, curva de
aprendizado e teste de sensibilidade -- tudo medido em dado NORMAL fora do
ajuste, sem depender dos 8 rotulos de trip:

  L1 generalizacao   score no ajuste (treino) x no mes seguinte (validacao),
                     walk-forward mensal. Taxa de excedencia do limiar 1,0
                     (p99 do baseline) na validacao: nominal 1%.
  L2 curva           o mesmo, variando o tamanho do baseline (FIT_POINTS).
                     Se a validacao ainda melhora no maior tamanho, falta dado.
  L3 estabilidade    angulos principais entre subespacos de meses seguidos.
                     Subespaco que gira muito = modelo aprendendo ruido/regime.
  L4 sensibilidade   falhas SINTETICAS injetadas em trechos normais fora do
                     ajuste (degrau, rampa, ruido, sensor congelado), em
                     multiplos do MAD do sensor. Taxa de deteccao e atraso.
  L5 separacao       (opcional, com --falhas) score nas 48 h antes de cada trip
                     x janelas aleatorias de 48 h: enriquecimento, AUC e p de
                     permutacao. E o unico nivel que usa rotulo.

O scorer e qualquer classe com fit(DataFrame) -> self e score(DataFrame) ->
DataFrame, cuja coluna de score e normalizada (>1 = fora do normal), como o
`ScorerMax` (copia fiel em scorers.py, ao lado deste arquivo).

Uso:
  python avalia_normalidade.py --demo                       # dado sintetico
  python avalia_normalidade.py --dataset-id 8b06a98f8b264820a9ecf2075a188395 \\
      --familia temperatura \\
      --scorer-path .claude/skills/avalia-normalidade/scripts:scripts/pdm_fisico \\
      --scorer scorers:ScorerMax --col pca_recon \\
      --saida avaliacao_pca_t.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

POR_H = 30          # amostras de 2 min por hora
SUSTAIN = 15        # 30 min acima do limiar = acendeu (mesma regra do detector)
TEMPERATURA = ["954005_624_TI_0325", "954005_624_TI_0315", "954005_624_TI_0317",
               "954005_624_TI_0305", "954005_624_TI_0307", "954005_624_TI_0303",
               "954005_624_TI_0301", "TC382_01_A", "TC382_02_A", "TC382_03_A",
               "TC382_04_A", "TC382_05_A", "TC382_06_A", "T5_AVG_A"]
PRESSAO = ["954005_624_PI_0315", "954005_624_PI_0319", "954005_624_PI_0340",
           "954005_624_PI_0339", "954005_624_PDI_0317", "954005_624_PDI_0302",
           "954005_624_PDIT_0305", "954005_624_PI_0307", "954005_624_PI_0308",
           "PI_5134001", "954005_624_PDI_0338", "954005_624_PDI_0301"]


# ──────────────────────────────────────────────────────────── utilidades
def carrega_scorer(spec: str, kwargs: dict):
    mod, cls = spec.split(":")
    C = getattr(importlib.import_module(mod), cls)
    return lambda: C(**kwargs)


def pontua(sc, df: pd.DataFrame, col: str | None) -> np.ndarray:
    out = sc.score(df)
    s = out[col] if col else out.iloc[:, 0]
    return np.asarray(s, dtype="float64")


def sustentado(x: np.ndarray, thr: float = 1.0) -> np.ndarray:
    acima = np.nan_to_num(x, nan=0.0) > thr
    run = pd.Series(acima.astype(int)).rolling(SUSTAIN, min_periods=SUSTAIN).sum()
    return (run >= SUSTAIN).to_numpy()


def meses_validos(idx, stable, n_fit):
    meses = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz=idx.tz)
    cum = np.cumsum(stable.to_numpy())
    out = []
    for i, m0 in enumerate(meses):
        m1 = meses[i + 1] if i + 1 < len(meses) else idx[-1] + pd.Timedelta("2min")
        antes = cum[max(idx.searchsorted(m0) - 1, 0)] if idx.searchsorted(m0) > 0 else 0
        if antes >= n_fit // 4 and stable[(idx >= m0) & (idx < m1)].sum() > 0:
            out.append((m0, m1))
    return out


def ajusta(fac, df, stable, tags, m0, n_fit):
    fit = df.loc[stable & (df.index < m0), tags].dropna().tail(n_fit)
    return fac().fit(fit), fit


def base_subespaco(sc):
    """Base ortonormal (d x k) do subespaco aprendido, se o scorer expuser uma."""
    if hasattr(sc, "subespaco"):
        return np.asarray(sc.subespaco())
    pca = getattr(sc, "pca", None)
    if pca is not None and hasattr(pca, "components_"):
        return pca.components_.T
    return None


# ──────────────────────────────────────────────────────────── L1 + L3
def generalizacao(df, stable, tags, fac, col, n_fit, meses):
    from scipy.linalg import subspace_angles
    linhas, serie, base_ant = [], pd.Series(np.nan, index=df.index), None
    for m0, m1 in meses:
        sc, fit = ajusta(fac, df, stable, tags, m0, n_fit)
        s_tr = pontua(sc, fit, col)
        sel = (df.index >= m0) & (df.index < m1)
        s_mes = pontua(sc, df.loc[sel, tags], col)
        s_mes = np.where(stable[sel].to_numpy(), s_mes, np.nan)
        serie[sel] = s_mes
        v = s_mes[np.isfinite(s_mes)]
        B = base_subespaco(sc)
        ang = None
        if B is not None and base_ant is not None and B.shape[0] == base_ant.shape[0]:
            ang = float(np.degrees(subspace_angles(base_ant, B)).max())
        base_ant = B
        linhas.append(dict(
            mes=f"{m0:%Y-%m}", n_fit=len(fit), n_val=int(len(v)),
            med_treino=float(np.nanmedian(s_tr)), med_val=float(np.median(v)) if len(v) else np.nan,
            exc_treino=float(np.nanmean(s_tr > 1)), exc_val=float(np.mean(v > 1)) if len(v) else np.nan,
            duty_sust_val=float(np.nanmean(sustentado(s_mes)[stable[sel].to_numpy()])) if len(v) else np.nan,
            k=None if B is None else int(B.shape[1]), angulo_max_graus=ang))
    T = pd.DataFrame(linhas)
    T["razao_med"] = T["med_val"] / T["med_treino"]
    return T, serie


# ──────────────────────────────────────────────────────────── L2
def curva(df, stable, tags, fac, col, tamanhos, meses):
    linhas = []
    for n in tamanhos:
        ev, med = [], []
        for m0, m1 in meses:
            sc, fit = ajusta(fac, df, stable, tags, m0, n)
            if len(fit) < n * 0.9:
                continue
            sel = (df.index >= m0) & (df.index < m1) & stable.to_numpy()
            s = pontua(sc, df.loc[sel, tags], col)
            s = s[np.isfinite(s)]
            if len(s):
                ev.append(np.mean(s > 1)); med.append(np.median(s))
        linhas.append(dict(n_fit=n, meses=len(ev),
                           exc_val=float(np.mean(ev)) if ev else np.nan,
                           med_val=float(np.mean(med)) if med else np.nan))
    return pd.DataFrame(linhas)


# ──────────────────────────────────────────────────────────── L4
def trechos_contiguos(mask: np.ndarray, n: int) -> list[int]:
    """Inicios de trechos com n amostras seguidas em mask."""
    run = pd.Series(mask.astype(int)).rolling(n, min_periods=n).sum().to_numpy()
    fins = np.flatnonzero(run >= n)
    return [int(f - n + 1) for f in fins[::n // 2]]


def injeta(X: np.ndarray, j: int, i0: int, tipo: str, m: float, mad: float, rng):
    Y = X.copy()
    n = len(Y) - i0
    if tipo == "degrau":
        Y[i0:, j] += m * mad
    elif tipo == "rampa":
        Y[i0:, j] += np.minimum(np.arange(n) / (24 * POR_H), 1.0) * m * mad
    elif tipo == "ruido":
        Y[i0:, j] += rng.normal(0, m * mad, n)
    elif tipo == "congelado":
        Y[i0:, j] = Y[i0, j]
    return Y


def sensibilidade(df, stable, tags, fac, col, n_fit, meses, magnitudes, tentativas, seed):
    rng = np.random.default_rng(seed)
    L_CTX, L_POS = 24 * POR_H, 24 * POR_H          # 24 h de contexto + 24 h apos a injecao
    L = L_CTX + L_POS
    res = []
    for m0, m1 in meses:
        sc, fit = ajusta(fac, df, stable, tags, m0, n_fit)
        mad = ((fit - fit.median()).abs().median() * 1.4826).replace(0, np.nan).fillna(1e-6).to_numpy()
        sel = (df.index >= m0) & (df.index < m1)
        W = df.loc[sel, tags]
        ok = (stable[sel] & W.notna().all(axis=1)).to_numpy()
        inis = trechos_contiguos(ok, L)
        if not inis:
            continue
        for _ in range(tentativas):
            a = int(rng.choice(inis))
            seg = W.iloc[a:a + L]
            s0 = pontua(sc, seg, col)
            if sustentado(s0)[L_CTX:].any():     # trecho ja acende sozinho: nao serve de controle
                continue
            j = int(rng.integers(len(tags)))
            X = seg.to_numpy(dtype="float64")
            for tipo in ("degrau", "rampa", "ruido", "congelado"):
                for m in (magnitudes if tipo != "congelado" else [1.0]):
                    Y = injeta(X, j, L_CTX, tipo, m, mad[j], rng)
                    s1 = pontua(sc, pd.DataFrame(Y, index=seg.index, columns=tags), col)
                    hit = np.flatnonzero(sustentado(s1)[L_CTX:])
                    res.append(dict(tipo=tipo, mag=m, sensor=tags[j], det=bool(len(hit)),
                                    atraso_h=(hit[0] / POR_H) if len(hit) else np.nan))
    R = pd.DataFrame(res)
    if R.empty:
        return R
    return (R.groupby(["tipo", "mag"])
             .agg(n=("det", "size"), taxa_det=("det", "mean"), atraso_med_h=("atraso_h", "median"))
             .reset_index())


# ──────────────────────────────────────────────────────────── L5
def separacao(serie: pd.Series, stable, falhas, n_nulo, seed):
    rng = np.random.default_rng(seed)
    s = serie.where(stable)
    jan = pd.Timedelta(hours=48)

    def estat(t):
        w = s.loc[t - jan:t - pd.Timedelta("2min")]
        return float(np.nanpercentile(w, 90)) if w.notna().sum() > 60 else np.nan

    alvo = np.array([estat(pd.Timestamp(t)) for t in falhas])
    cand = s.dropna().index
    longe = np.ones(len(cand), bool)
    for t in falhas:
        t = pd.Timestamp(t)
        longe &= ~((cand > t - pd.Timedelta(days=7)) & (cand < t + pd.Timedelta(days=2)))
    cand = cand[longe]
    nulo = np.array([estat(cand[i]) for i in rng.integers(len(cand), size=n_nulo)])
    nulo = nulo[np.isfinite(nulo)]
    a = alvo[np.isfinite(alvo)]
    if not len(a) or not len(nulo):
        return {}
    auc = float(np.mean([(x > nulo).mean() + 0.5 * (x == nulo).mean() for x in a]))
    obs = np.mean(a)
    perm = [np.mean(rng.choice(nulo, len(a), replace=False)) for _ in range(5000)]
    return dict(eventos=int(len(a)), nulo=int(len(nulo)),
                p90_alvo_mediana=float(np.median(a)), p90_nulo_mediana=float(np.median(nulo)),
                enriquecimento=float(np.median(a) / np.median(nulo)),
                auc=auc, p_perm=float((np.sum(np.array(perm) >= obs) + 1) / (len(perm) + 1)),
                por_evento=[None if not np.isfinite(x) else round(float(x), 3) for x in alvo])


# ──────────────────────────────────────────────────────────── veredito
def veredito(G, C, S, sep):
    v = []
    ev = G["exc_val"].median()
    if ev <= 0.03:
        v.append(f"calibracao OK: excedencia mediana na validacao {ev:.2%} (nominal 1%)")
    else:
        v.append(f"DERIVA: excedencia mediana na validacao {ev:.2%} contra 1% nominal -- o "
                 f"baseline nao representa o mes seguinte")
    rz = G["razao_med"].median()
    v.append(f"gap treino->validacao: score mediano {rz:.2f}x" + ("  (ALTO)" if rz > 1.5 else ""))
    if G["angulo_max_graus"].notna().any():
        a = G["angulo_max_graus"].median()
        v.append(f"estabilidade: angulo principal maximo mediano entre meses {a:.1f} graus"
                 + ("  (subespaco INSTAVEL)" if a > 45 else ""))
    if len(C) >= 2 and C["exc_val"].notna().sum() >= 2:
        c = C.dropna(subset=["exc_val"])
        e1, e0 = c["exc_val"].iloc[-1], c["exc_val"].iloc[-2]
        v.append("curva: " + ("ainda melhora no maior baseline -- mais dado ajuda"
                              if e1 < 0.8 * e0 else "saturada -- mais dado nao ajuda"))
    if not S.empty:
        d = S[(S.tipo == "degrau") & (S.taxa_det >= 0.8)]
        v.append("sensibilidade: degrau detectado >=80% a partir de "
                 + (f"{d.mag.min():g} MAD" if len(d) else "NENHUMA magnitude testada"))
    if sep:
        v.append(f"separacao (rotulos): enriquecimento {sep['enriquecimento']:.2f}x, "
                 f"AUC {sep['auc']:.2f}, p={sep['p_perm']:.3f} com {sep['eventos']} eventos")
    return v


# ──────────────────────────────────────────────────────────── demo
def demo_dados(seed=0, dias=240, d=8):
    """Processo sintetico: 3 fatores latentes AR(1) lentos, degraus de carga,
    paradas. Normal por construcao -- serve para validar a propria avaliacao."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=dias * 24 * POR_H, freq="2min", tz="UTC")
    n = len(idx)
    F = np.zeros((n, 3))
    for i in range(1, n):
        F[i] = 0.999 * F[i - 1] + rng.normal(0, 0.05, 3)
    carga = np.repeat(rng.choice([0.0, 0.5, 1.0], n // (3 * 24 * POR_H) + 1), 3 * 24 * POR_H)[:n]
    A = rng.normal(0, 1, (3, d))
    X = F @ A + np.outer(carga, rng.normal(0, 1, d)) + rng.normal(0, 0.1, (n, d))
    df = pd.DataFrame(X, index=idx, columns=[f"S{j}" for j in range(d)])
    run = np.ones(n, bool)
    for a in rng.integers(0, n - POR_H * 48, 12):
        run[a:a + int(rng.integers(6, 48)) * POR_H] = False
    df["RUNNING_A"] = run.astype(float)
    df["T5_AVG_A"] = np.where(run, 450.0, 50.0)
    return df, list(df.columns[:d])


class DemoPCA:
    """PCA de referencia (media do erro / p99 do baseline) para o --demo."""

    def __init__(self, var=0.95):
        self.var = var

    def fit(self, X):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import RobustScaler
        self.cols = list(X.columns)
        self.sc = RobustScaler().fit(X)
        self.pca = PCA(self.var, svd_solver="full").fit(self.sc.transform(X))
        self.p99 = float(np.percentile(self._raw(X), 99))
        return self

    def _raw(self, X):
        Z = self.sc.transform(X[self.cols])
        return ((Z - self.pca.inverse_transform(self.pca.transform(Z))) ** 2).mean(axis=1)

    def score(self, X):
        ok = X[self.cols].notna().all(axis=1).to_numpy()
        r = np.full(len(X), np.nan)
        if ok.any():
            r[ok] = self._raw(X[ok]) / self.p99
        return pd.DataFrame({"score": r}, index=X.index)


# ──────────────────────────────────────────────────────────── clearml
IMAGEM = "tensorflow/tensorflow:2.16.1-gpu"   # a mesma do resto do repositorio


def inicia_clearml(a):
    """Task no ClearML. Com --remote, enfileira e ENCERRA este processo.

    O worker clona o repositorio no commit registrado (mais o diff nao
    commitado) e roda este mesmo comando.

    Dependencias: a lista inteira e SUBSTITUIDA por set_packages, sem pino de
    versao. O ClearML, sozinho, congela as versoes desta maquina (Python 3.13,
    numpy 2.3) -- a imagem do worker e Python 3.11 com o numpy < 2 do
    TensorFlow, e a primeira task morreu no pip. Mesma licao do
    roda_clearml.py. Os scorers entram por importlib e nao seriam detectados de
    qualquer forma."""
    from clearml import Task
    scorer = "DemoPCA" if a.demo else (a.scorer or "?").split(":")[-1]
    fam = "demo" if a.demo else (a.familia or "tags")
    nome = a.nome or f"avalia_normalidade::{fam}::{scorer}::{a.col or 'score'}"
    task = Task.init(project_name=a.projeto, task_name=nome,
                     task_type=Task.TaskTypes.qc, reuse_last_task_id=False,
                     auto_connect_frameworks=False)
    task.set_base_docker(docker_image=IMAGEM)
    # tzdata: a imagem do TensorFlow nao traz banco de fusos, e o pandas 3 com
    # pyarrow converte o indice UTC do parquet via zoneinfo -- sem ele, falha
    # na leitura (o roda_clearml.py trazia pytz pelo mesmo motivo, no pandas 2)
    task.set_packages(["numpy", "pandas", "scipy", "scikit-learn", "pyarrow",
                       "tzdata", "clearml"])
    task.add_tags(["avalia-normalidade", fam, scorer, a.col or "score"])
    if a.remote:
        if not (a.demo or a.dataset_id):
            raise SystemExit("--remote precisa de --dataset-id: o worker nao tem o seu disco")
        task.execute_remotely(queue_name=a.fila, exit_process=True)
    return task


def publica_clearml(task, G, C, S, sep, ver, saida):
    """Tabelas L1-L5 no painel, metricas-resumo como valores unicos (para
    comparar tasks lado a lado na UI) e o JSON completo como artefato."""
    lg = task.get_logger()
    lg.report_table("L1+L3 generalizacao", "por mes", iteration=0, table_plot=G.round(4))
    lg.report_table("L2 curva", "por n_fit", iteration=0, table_plot=C.round(4))
    if not S.empty:
        lg.report_table("L4 sensibilidade", "injecao sintetica", iteration=0, table_plot=S.round(3))
    for _, r in G.iterrows():                       # a "curva" de validacao no tempo
        i = int(pd.Timestamp(r["mes"]).strftime("%Y%m"))
        lg.report_scalar("excedencia", "treino", iteration=i, value=float(r["exc_treino"]))
        lg.report_scalar("excedencia", "validacao", iteration=i, value=float(r["exc_val"]))
    for _, r in C.dropna(subset=["exc_val"]).iterrows():
        lg.report_scalar("curva de aprendizado", "excedencia validacao",
                         iteration=int(r["n_fit"]), value=float(r["exc_val"]))
    resumo = {"exc_val_mediana": G["exc_val"].median(),
              "razao_med_mediana": G["razao_med"].median(),
              "angulo_max_mediano": G["angulo_max_graus"].median()}
    if not S.empty:
        d = S[(S.tipo == "degrau") & (S.taxa_det >= 0.8)]
        resumo["degrau_80pct_mad"] = float(d.mag.min()) if len(d) else float("nan")
        r3 = S[(S.tipo == "rampa") & (S.mag == 3)]
        if len(r3):
            resumo["rampa3_taxa_det"] = float(r3.taxa_det.iloc[0])
    if sep:
        resumo.update(auc=sep["auc"], p_perm=sep["p_perm"], enriquecimento=sep["enriquecimento"])
    for k, v in resumo.items():
        if v is not None and np.isfinite(v):
            lg.report_single_value(k, float(v))
    task.upload_artifact("avaliacao", artifact_object=saida)
    task.set_comment("\n".join(ver))
    task.flush(wait_for_uploads=True)


# ──────────────────────────────────────────────────────────── main
def main():
    # O ClearML injeta os argumentos salvos na task PATCHEANDO o argparse -- e so
    # consegue se ja estiver importado antes do parse_args. No worker o script roda
    # sem argumentos na linha de comando (tudo vem do servidor), e o agente marca o
    # ambiente com CLEARML_TASK_ID.
    import os
    if {"--clearml", "--remote"} & set(sys.argv) or os.environ.get("CLEARML_TASK_ID"):
        import clearml  # noqa: F401
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--parquet")
    ap.add_argument("--dataset-id", help="Dataset ClearML com grade2min.parquet e falhas.csv "
                    "(o do detector: 8b06a98f8b264820a9ecf2075a188395)")
    ap.add_argument("--familia", choices=["temperatura", "pressao"])
    ap.add_argument("--tags", help="lista separada por virgula (alternativa a --familia)")
    ap.add_argument("--scorer", help="modulo:Classe")
    ap.add_argument("--scorer-path", default=".", help="diretorios separados por ':'")
    ap.add_argument("--scorer-kwargs", default="{}")
    ap.add_argument("--col", help="coluna do score em score() (padrao: a primeira)")
    ap.add_argument("--n-fit", type=int, default=20_000)
    ap.add_argument("--tamanhos", default="2500,5000,10000,20000,40000")
    ap.add_argument("--magnitudes", default="1,2,3,5,8")
    ap.add_argument("--meses-amostra", type=int, default=6, help="meses usados em L2 e L4")
    ap.add_argument("--tentativas", type=int, default=15, help="trechos por mes em L4")
    ap.add_argument("--falhas", help="CSV com coluna `evento` (UTC) -- ativa L5")
    ap.add_argument("--desde", help="avaliar so meses a partir desta data (ex. 2025-01-01)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--saida", default="avaliacao_normalidade.json")
    ap.add_argument("--clearml", action="store_true",
                    help="registra como task no ClearML (projeto --projeto)")
    ap.add_argument("--remote", action="store_true",
                    help="com --clearml: enfileira em --fila e sai; o worker roda")
    ap.add_argument("--projeto", default="TesteMLCab")
    ap.add_argument("--fila", default="default")
    ap.add_argument("--nome", help="nome da task (padrao: familia::scorer::col)")
    a = ap.parse_args()
    task = inicia_clearml(a) if (a.clearml or a.remote) else None

    if a.demo:
        df, tags = demo_dados(a.seed)
        fac, col = (lambda: DemoPCA()), "score"
        a.n_fit, a.tamanhos = 10_000, "1250,2500,5000,10000,20000"
        if a.scorer:                     # --demo com outro scorer: compara no mesmo sintetico
            for d in reversed(a.scorer_path.split(":")):
                sys.path.insert(0, str(Path(d).resolve()))
            fac, col = carrega_scorer(a.scorer, json.loads(a.scorer_kwargs)), a.col
    else:
        if a.dataset_id:
            from clearml import Dataset
            D = Path(Dataset.get(dataset_id=a.dataset_id).get_local_copy())
            a.parquet = a.parquet or str(D / "grade2min.parquet")
            if a.falhas is None and (D / "falhas.csv").exists():
                a.falhas = str(D / "falhas.csv")
        if not (a.parquet and a.scorer and (a.familia or a.tags)):
            ap.error("sem --demo, informe --dataset-id ou --parquet, --scorer e --familia/--tags")
        for d in reversed(a.scorer_path.split(":")):
            sys.path.insert(0, str(Path(d).resolve()))
        df = pd.read_parquet(a.parquet)
        tags = a.tags.split(",") if a.tags else (TEMPERATURA if a.familia == "temperatura" else PRESSAO)
        fac, col = carrega_scorer(a.scorer, json.loads(a.scorer_kwargs)), a.col
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    stable = ((df["RUNNING_A"] > 0.5) & (df["T5_AVG_A"] > 300)).fillna(False)

    meses = meses_validos(df.index, stable, a.n_fit)
    if a.desde:
        meses = [m for m in meses if m[0] >= pd.Timestamp(a.desde, tz="UTC")]
    if not meses:
        raise SystemExit("nenhum mes com baseline suficiente")
    amostra = [meses[i] for i in np.unique(np.linspace(0, len(meses) - 1,
                                                       min(a.meses_amostra, len(meses))).astype(int))]
    pd.set_option("display.width", 160)

    print(f"L1+L3 generalizacao e estabilidade -- {len(meses)} meses, n_fit={a.n_fit}", flush=True)
    G, serie = generalizacao(df, stable, tags, fac, col, a.n_fit, meses)
    print(G.round(4).to_string(index=False))

    print(f"\nL2 curva de aprendizado -- {len(amostra)} meses", flush=True)
    C = curva(df, stable, tags, fac, col, [int(x) for x in a.tamanhos.split(",")], amostra)
    print(C.round(4).to_string(index=False))

    print("\nL4 sensibilidade por injecao sintetica", flush=True)
    S = sensibilidade(df, stable, tags, fac, col, a.n_fit, amostra,
                      [float(x) for x in a.magnitudes.split(",")], a.tentativas, a.seed)
    print(S.round(3).to_string(index=False) if not S.empty else "  sem trechos contiguos de 48 h")

    sep = {}
    if a.falhas:
        f = pd.to_datetime(pd.read_csv(a.falhas)["evento"], utc=True)
        print("\nL5 separacao contra os trips", flush=True)
        sep = separacao(serie, stable, list(f), 2000, a.seed)
        print(json.dumps(sep, indent=1))

    print("\nVEREDITO")
    ver = veredito(G, C, S, sep)
    for linha in ver:
        print("  - " + linha)

    Path(a.saida).write_text(json.dumps(dict(
        generalizacao=G.to_dict("records"), curva=C.to_dict("records"),
        sensibilidade=S.to_dict("records"), separacao=sep, veredito=ver,
        args=vars(a)), indent=1, default=str), encoding="utf-8")
    print(f"\nJSON: {a.saida}")
    if task is not None:
        publica_clearml(task, G, C, S, sep, ver, a.saida)


if __name__ == "__main__":
    main()
