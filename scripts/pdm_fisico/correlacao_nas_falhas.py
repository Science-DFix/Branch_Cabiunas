#!/usr/bin/env python3
"""A estrutura de correlação entre as tags se mantém nos momentos que antecedem as falhas?

DIAGNÓSTICO (03/10/2026). Não muda detector nem veredito. Duas perguntas do usuário:
  (1) a correlação ENTRE OS SENSORES DE CADA GRUPO (dentro de cada subsistema);
  (2) essa estrutura se mantém "até nos momentos das falhas"?

O QUE SE COMPARA. As janelas de 48 h antes de cada um dos 8 trips-alvo (só os instantes em regime: RUNNING_A
> 0,5 e T5 > 300) contra a operação NORMAL (regime, fora de [T-7 d, T+2 d] de qualquer trip; a mesma
exclusão do `vb`). Spearman, recalculado DENTRO de cada amostra.

O CONTROLE NEGATIVO, que é o que torna o resultado interpretável. Uma correlação calculada em ~1.000
pontos autocorrelacionados de 48 h oscila só por acaso. Por isso o mesmo cálculo é repetido em 1.000 sorteios
de janelas de 48 h de operação normal, com o MESMO número de pontos em regime que cada janela pré-falha
(sub-amostrados ao tamanho dela). A distância observada só vale contra essa distribuição.

AS MEDIDAS, por amostra (sempre contra a correlação da operação normal):
  D_dentro  média de |rho_amostra - rho_normal| nos pares de tags do MESMO subsistema;
  D_entre   idem nos pares de subsistemas DIFERENTES.
  Versão BRUTA e versão CONDICIONADA À CARGA (mediana de T5 por faixa, medianas tiradas da operação normal,
  para a janela não se condicionar a si mesma; T5_AVG_A fica fora desta).
Por evento e agrupando os 8. p = fração dos sorteios normais com distância >= a observada.

Uso:  PYTHONPATH=. python correlacao_nas_falhas.py [horas]   # horas = 48 (padrão), 12 ou 6
"""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from cabiunas_pdm import config as C
import correlacao_entre_grupos as CG

AQUI = Path(__file__).resolve().parent
import sys
HORAS = int(sys.argv[1]) if len(sys.argv) > 1 else 48          # janela antes do trip: 48 (padrão), 12 ou 6 h
JAN = pd.Timedelta(hours=HORAS)
N_SORT = 1000
RNG = np.random.default_rng(0)
G = CG.G
curto = CG.curto


def trips() -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    f = pd.read_csv(AQUI / "falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    return list(f), [t for t in f if t >= pd.Timestamp("2025-01-01", tz="UTC")]


def rho(S: np.ndarray) -> np.ndarray:
    """Spearman dentro da amostra: postos por coluna, depois Pearson."""
    R = np.argsort(np.argsort(S, axis=0), axis=0).astype(float)
    return np.corrcoef(R.T)


def condiciona(X: pd.DataFrame, Xn: pd.DataFrame, nb: int = 30) -> pd.DataFrame:
    """Subtrai a mediana por faixa de T5, com faixas e medianas da operação NORMAL."""
    edges = np.unique(Xn["T5_AVG_A"].quantile(np.linspace(0, 1, nb + 1)).to_numpy())
    cn = pd.cut(Xn["T5_AVG_A"], edges, include_lowest=True)
    med = Xn.groupby(cn, observed=True).median()
    c = pd.cut(X["T5_AVG_A"].clip(edges[0], edges[-1]), edges, include_lowest=True)
    return (X - med.reindex(c).to_numpy()).drop(columns="T5_AVG_A")


def indices(cols: list[str]):
    grp = {t: i for i, (g, ts) in enumerate(G.items()) for t in ts}
    gi = np.array([grp[c] for c in cols])
    mesmo = gi[:, None] == gi[None, :]
    iu = np.triu(np.ones((len(cols),) * 2, bool), 1)
    return gi, mesmo & iu, (~mesmo) & iu


def dist(M: np.ndarray, Mn: np.ndarray, dentro, entre, gi):
    d = np.abs(M - Mn)
    por_grupo = {g: float(d[np.ix_(gi == g, gi == g)][np.triu(np.ones(((gi == g).sum(),) * 2, bool), 1)].mean())
                 for g in range(len(G)) if (gi == g).sum() > 1}
    return float(d[dentro].mean()), float(d[entre].mean()), por_grupo


def main():
    pd.set_option("display.width", 220)
    todas, alvo = trips()
    X = CG.dados().dropna()
    exclui = np.zeros(len(X), bool)
    for t in todas:
        exclui |= (X.index >= t - pd.Timedelta(days=7)) & (X.index <= t + pd.Timedelta(days=2))
    Xn = X[~exclui]
    jan = {t: X.loc[(X.index >= t - JAN) & (X.index < t)] for t in alvo}
    print(f"operação normal: {len(Xn)} instantes em regime ({len(Xn) * 2 / 60:.0f} h) | janelas pré-falha:")
    for t, w in jan.items():
        print(f"   {t:%Y-%m-%d}: {len(w):4d} instantes em regime ({len(w) * 2 / 60:4.1f} h de 48 h)")
    versoes = {"BRUTA": (Xn, {t: w for t, w in jan.items()}), "CONDICIONADA À CARGA": (condiciona(Xn, Xn), {t: condiciona(w, Xn) for t, w in jan.items()})}
    tn = Xn.index.to_numpy()
    saidas = {}
    for nome, (N, J) in versoes.items():
        cols = list(N.columns); gi, dentro, entre = indices(cols)
        Mn = rho(N.to_numpy())
        obs_ev = {t: dist(rho(w.to_numpy()), Mn, dentro, entre, gi) for t, w in J.items()}
        pool = pd.concat(list(J.values()))
        obs_pool = dist(rho(pool.to_numpy()), Mn, dentro, entre, gi)
        Mpre = rho(pool.to_numpy())
        nul_pool, nul_ev = [], {t: [] for t in J}
        z_par = []
        for _ in range(N_SORT):
            amostras = []
            for t, w in J.items():
                ne = len(w)
                for _t in range(200):
                    s = tn[RNG.integers(len(tn))]
                    sub = N.loc[(N.index >= s) & (N.index < s + JAN.to_timedelta64())]
                    if len(sub) >= ne:
                        break
                sub = sub.iloc[np.sort(RNG.choice(len(sub), ne, replace=False))]
                nul_ev[t].append(dist(rho(sub.to_numpy()), Mn, dentro, entre, gi)[:2]); amostras.append(sub)
            P = rho(pd.concat(amostras).to_numpy())
            nul_pool.append(dist(P, Mn, dentro, entre, gi)); z_par.append(P)
        Z = np.array(z_par); mu, sd = Z.mean(axis=0), Z.std(axis=0) + 1e-9
        saidas[nome] = dict(cols=cols, gi=gi, Mn=Mn, Mpre=Mpre, z=(Mpre - mu) / sd)
        print(f"\n{'=' * 100}\n{nome}: janelas pré-falha agrupadas (n = {len(pool)}) contra {N_SORT} sorteios de operação normal\n{'=' * 100}")
        for k, lab in ((0, "D_dentro (mesmo subsistema)"), (1, "D_entre (subsistemas diferentes)")):
            nl = np.array([x[k] for x in nul_pool])
            print(f"  {lab:34s}: observado {obs_pool[k]:.3f} | normal: média {nl.mean():.3f}, p95 {np.quantile(nl, .95):.3f} | p = {(nl >= obs_pool[k]).mean():.3f}")
        print("  por subsistema (D_dentro): observado | média do normal | p")
        nomes = list(G)
        for g, v in obs_pool[2].items():
            nl = np.array([x[2][g] for x in nul_pool])
            print(f"     {nomes[g].split(' (')[0]:34s} {v:.3f} | {nl.mean():.3f} | {(nl >= v).mean():.3f}")
        print("  POR EVENTO (D_dentro | p ;  D_entre | p)")
        for t in J:
            nd, ne_ = np.array(nul_ev[t])[:, 0], np.array(nul_ev[t])[:, 1]
            print(f"     {t:%Y-%m-%d}  n={len(J[t]):4d}   {obs_ev[t][0]:.3f} | {(nd >= obs_ev[t][0]).mean():.3f}     {obs_ev[t][1]:.3f} | {(ne_ >= obs_ev[t][1]).mean():.3f}")
    # pares com maior desvio (bruta), contra o nulo
    S = saidas["BRUTA"]; cols = S["cols"]; iu = np.triu(np.ones(S["z"].shape, bool), 1)
    pares = sorted([(abs(S["z"][i, j]), cols[i], cols[j], S["Mn"][i, j], S["Mpre"][i, j]) for i, j in zip(*np.nonzero(iu))], reverse=True)[:10]
    print(f"\nMAIORES DESVIOS POR PAR (bruta), em desvios-padrão do sorteio normal (630 pares: |z| > 3,5 é o que se espera achar)")
    for z, a, b, rn, rp in pares:
        print(f"   {curto(a):12s} x {curto(b):12s}  normal {rn:+.2f} -> pré-falha {rp:+.2f}   |z| {z:.1f}")
    if HORAS == 48:
        desenha_dentro(S); desenha_delta(S)
        print("\n-> correlacao_dentro_dos_grupos.png, correlacao_pre_falha.png")


def _paineis(titulo, matrizes_fn, saida, vmin, vmax, anota_z=None, sub=None):
    cm = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    grupos = [(n, ts) for n, ts in G.items() if len(ts) > 1]
    fig, axs = plt.subplots(2, 3, figsize=(17, 11.5), facecolor="#fcfcfb")
    for ax, (nome, ts) in zip(axs.flat, grupos):
        M = matrizes_fn(ts)
        ax.imshow(M, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(len(ts))); ax.set_yticks(range(len(ts)))
        ax.set_xticklabels([curto(t) for t in ts], rotation=90, fontsize=8, color="#52514e"); ax.set_yticklabels([curto(t) for t in ts], fontsize=8, color="#52514e")
        fs = 8 if len(ts) <= 5 else (6.5 if len(ts) <= 9 else 6)
        for i in range(len(ts)):
            for j in range(len(ts)):
                if i != j:
                    forte = anota_z is not None and abs(anota_z(ts)[i, j]) > 3.5
                    ax.text(j, i, f"{M[i, j]:+.2f}".replace("+0.", "+.").replace("-0.", "-."), ha="center", va="center", fontsize=fs,
                            color="#0b0b0b", fontweight="bold" if forte else "normal")
        ax.set_title(nome.split(" (")[0] + (f"  ({len(ts)} tags)"), fontsize=11, color="#0b0b0b", loc="left")
        ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
    for ax in axs.flat[len(grupos):]:
        ax.axis("off")
    fig.suptitle(titulo, fontsize=13, color="#0b0b0b", x=0.02, ha="left")
    if sub:
        fig.text(0.02, 0.945, sub, fontsize=9.5, color="#52514e")
    fig.subplots_adjust(top=0.90, hspace=0.45, wspace=0.28)
    fig.savefig(saida, dpi=125, bbox_inches="tight"); plt.close(fig)


def desenha_dentro(S):
    idx = {c: i for i, c in enumerate(S["cols"])}
    _paineis("Correlação entre os sensores de cada grupo (operação normal, em regime, Spearman)",
             lambda ts: S["Mn"][np.ix_([idx[t] for t in ts], [idx[t] for t in ts])], AQUI / "correlacao_dentro_dos_grupos.png", -1, 1,
             sub="Cada painel é um subsistema; o número é o rho entre duas tags do grupo.")


def desenha_delta(S):
    idx = {c: i for i, c in enumerate(S["cols"])}
    sel = lambda M, ts: M[np.ix_([idx[t] for t in ts], [idx[t] for t in ts])]
    _paineis("Mudança da correlação nas 48 h antes das falhas (pré-falha menos normal)", lambda ts: sel(S["Mpre"], ts) - sel(S["Mn"], ts),
             AQUI / "correlacao_pre_falha.png", -1, 1, anota_z=lambda ts: sel(S["z"], ts),
             sub="Janelas dos 8 trips agrupadas. Azul: a correlação cai; vermelho: sobe. Em negrito, |z| > 3,5 contra janelas normais sorteadas.")


if __name__ == "__main__":
    main()
