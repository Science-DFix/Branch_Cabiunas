#!/usr/bin/env python3
"""Técnicas atuais de detecção de drift, na mesma régua, nos nossos dados.

AS TÉCNICAS (todas comparando a semana com o baseline do bundle em vigor):
  mediana       |mediana da semana - mediana do baseline| / sigma, máximo entre as
                tags. É o monitor atual (`monitor_drift`).
  PSI           Population Stability Index, o padrão do mercado financeiro: 10 faixas
                pelos decis do baseline, soma de (p_sem - p_base) * ln(p_sem/p_base).
                Máximo entre as tags. Regra de bolso: > 0,25 = drift grande.
  Wasserstein   distância de transporte W1 entre as distribuições, em sigmas do
                baseline. Parente do KS, mas NÃO satura: 3 e 300 sigma dão valores
                diferentes. Máximo entre as tags.
  classificador "classificador de domínio": um modelo (gradient boosting) tenta
                distinguir amostras do baseline e da semana, usando as 26 tags JUNTAS.
                AUC ~0,5 = indistinguíveis; ~1 = drift. É multivariado: pega mudança de
                CORRELAÇÃO que nenhuma medida por tag vê. A divisão treino/teste é por
                BLOCO DE TEMPO -- dividir ao acaso vazaria vizinhos autocorrelacionados
                e daria AUC ~1 sempre.
  T² e Q        o clássico do controle estatístico multivariado, sobre o PCA do próprio
                bundle: T² (Hotelling) mede desvio DENTRO do subespaço do PCA -- mudança
                que o modelo reconstrói e o nosso canal não vê --, Q (SPE) mede o
                resíduo -- é a família do nosso canal. Métrica: fração da semana acima do
                p99 do baseline (esperado ~1%). Máximo entre temperatura e pressão.

A RÉGUA. "Semana com drift conhecido" = termina até 14 dias depois de uma
manutenção (HSX >= 24 h): nelas ~14 tags mudam de nível juntas, contra 4 nas demais
(`drift_eventos.py`). Cada técnica recebe o limiar que dispara em 10% das semanas
NORMAIS; ganha quem pega mais semanas pós-manutenção com esse mesmo custo. E o AUC
de cada métrica entre os dois grupos, sem limiar nenhum. Imperfeita -- há drift sem
manutenção -- mas a mesma para todas. São 14 semanas positivas: diferença de 1-2 é
ruído.

RESULTADO (30/09/2026). Nenhuma técnica separa bem as semanas pós-manutenção:

    técnica         AUC   pega pós-manut.   valor típico numa semana NORMAL
    mediana         0,53       3/14           3,9 sigma
    PSI             0,56       0/14           7,8   (regra de bolso: > 0,25 = drift grande)
    Wasserstein     0,52       0/14          10,8 sigma, p90 de 362 sigma
    classificador   0,72       6/14           AUC 0,95
    T² (Hotelling)  0,45       0/14           2% acima do p99
    Q (SPE)         0,63       2/14           8% acima do p99

O achado é a última coluna: CONTRA O BASELINE, TODA SEMANA JÁ ESTÁ EM DRIFT. Um
classificador distingue a semana corrente das ~5 semanas de referência com AUC 0,95 em
semana comum; PSI e Wasserstein saturam ou explodem pelo mesmo motivo. O salto da
manutenção existe de uma semana para a seguinte (`drift_eventos.py`), mas contra o
baseline se perde no drift que acontece o tempo todo. Consequências:
  · monitorar: detectores genéricos dispariam sempre aqui; o monitor certo mira casos
    patológicos (degrau persistente, sensor travado), não "existe drift";
  · retreinar: drift contínuo pede adaptação contínua -- ver `retreino_semanal.py`.
Curiosidade: o T² só acorda em nov/2025 (31-47% acima do p99), o caso do instrumento.

Uso:  PYTHONPATH=. python drift_tecnicas.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import drift_nos_dados as DN
import drift_eventos as DE
import regua_fp as R

C, DF, IX, STABLE, FIT, VIG, DC = DN.C, DN.DF, DN.IX, DN.STABLE, DN.FIT, DN.VIG, DN.DC
TAGS = C.TEMPERATURE_TAGS + C.PRESSURE_TAGS
FAM = {"temperatura": C.TEMPERATURE_TAGS, "pressao": C.PRESSURE_TAGS}


def psi(b, w, n=10):
    q = np.unique(np.quantile(b, np.linspace(0, 1, n + 1)))
    if len(q) < 3:
        return np.nan
    pb = np.histogram(b, q)[0] / len(b); pw = np.histogram(np.clip(w, q[0], q[-1]), q)[0] / len(w)
    pb, pw = np.clip(pb, 1e-4, None), np.clip(pw, 1e-4, None)
    return float(np.sum((pw - pb) * np.log(pw / pb)))


def auc_dominio(B, W, rng):
    n = min(len(B), len(W))
    Bs = B.iloc[np.sort(rng.choice(len(B), n, replace=False))].to_numpy()
    Ws = W.iloc[:n].to_numpy() if len(W) >= n else W.to_numpy()
    h = n // 2
    Xtr = np.vstack([Bs[:h], Ws[:h]]); ytr = np.r_[np.zeros(h), np.ones(h)]
    Xte = np.vstack([Bs[h:], Ws[h:]]); yte = np.r_[np.zeros(len(Bs) - h), np.ones(len(Ws) - h)]
    clf = HistGradientBoostingClassifier(max_iter=100, max_depth=3, learning_rate=0.1, random_state=0)
    clf.fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


def t2_q(sc, B, W):
    """fração da semana acima do p99 do baseline, para T² e Q, com o PCA do bundle."""
    def calc(X):
        Xs = sc.scaler.transform(X); Z = sc.pca.transform(Xs)
        t2 = np.sum(Z ** 2 / sc.pca.explained_variance_, axis=1)
        q = np.sum((Xs - sc.pca.inverse_transform(Z)) ** 2, axis=1)
        return t2, q
    t2b, qb = calc(B); t2w, qw = calc(W)
    return float((t2w > np.quantile(t2b, .99)).mean()), float((qw > np.quantile(qb, .99)).mean())


def medidas() -> pd.DataFrame:
    man = DE.manutencoes()
    cortes = list(pd.date_range(pd.Timestamp("2025-01-01", tz="UTC"), IX[-1], freq="MS"))
    rng = np.random.default_rng(0)
    L = []
    for i, c0 in enumerate(cortes):
        c1 = cortes[i + 1] if i + 1 < len(cortes) else IX[-1]
        base = DF.loc[STABLE & (IX < c0), C.SENSOR_TAGS].dropna().tail(FIT)
        scs = {f: DC.ScorerMax().fit(base[cols]) for f, cols in FAM.items()}
        B = base[TAGS]
        sig = (B.quantile(.75) - B.quantile(.25)) / 1.349
        for w0 in pd.date_range(c0, c1, freq="7D"):
            w1 = min(w0 + pd.Timedelta(days=7), c1)
            sel = (IX >= w0) & (IX < w1) & VIG
            if sel.sum() < 2 * 720:
                continue
            W = DF.loc[sel, C.SENSOR_TAGS].dropna()[TAGS]
            med = ((W.median() - B.median()) / sig.replace(0, np.nan)).abs().max()
            ps = np.nanmax([psi(B[c].to_numpy(), W[c].to_numpy()) for c in TAGS])
            ws = np.nanmax([wasserstein_distance(B[c], W[c]) / sig[c] for c in TAGS if sig[c] > 0])
            auc = auc_dominio(B, W, rng)
            t2q = [t2_q(scs[f], base[cols], W[cols]) for f, cols in FAM.items()]
            pos = any(w0 - pd.Timedelta(days=14) <= b <= w1 for _, b in man)
            L.append(dict(semana=w0, pos_manut=pos, mediana=float(med), PSI=float(ps), Wasserstein=float(ws),
                          classificador=auc, T2=max(x[0] for x in t2q), Q=max(x[1] for x in t2q)))
    return pd.DataFrame(L)


def main():
    M = medidas()
    pd.set_option("display.width", 180)
    pos, neg = M[M.pos_manut], M[~M.pos_manut]
    print(f"{len(M)} semanas: {len(pos)} pós-manutenção (drift conhecido), {len(neg)} normais\n")
    print(f"{'técnica':14s} {'AUC semanas':>12s} {'limiar (p90 normais)':>21s} {'pega pós-manut.':>16s} "
          f"{'mediana normal':>15s} {'mediana pós':>12s}  nov/2025 (15-22/11)")
    for c in ("mediana", "PSI", "Wasserstein", "classificador", "T2", "Q"):
        auc = roc_auc_score(M.pos_manut, M[c].fillna(0))
        thr = float(neg[c].quantile(0.90))
        rec = int((pos[c] >= thr).sum())
        nov = M[(M.semana >= pd.Timestamp("2025-11-15", tz="UTC")) & (M.semana < pd.Timestamp("2025-11-29", tz="UTC"))][c]
        print(f"{c:14s} {auc:12.2f} {thr:21.3f} {rec:9d}/{len(pos)} {neg[c].median():15.3f} {pos[c].median():12.3f}"
              f"  {np.round(nov.to_numpy(), 3).tolist()}")
    print("\ncorrelação de Spearman entre as técnicas (semana a semana):")
    print(M[["mediana", "PSI", "Wasserstein", "classificador", "T2", "Q"]].corr("spearman").round(2).to_string())
    M.to_csv(R.CACHE / "drift_tecnicas.csv", index=False)


if __name__ == "__main__":
    main()
