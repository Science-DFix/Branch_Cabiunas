#!/usr/bin/env python3
"""E3 -- curva de compromisso trips detectados x FP/mês (descritiva; não escolhe regra).

Pergunta (do usuário: "vamos para item 3"): já que nada separa trip de FP, quanto FP se economiza aceitando perder
trips? A operação escolhe o ponto. Dois "botões", OCSVM de produção, período inteiro e fora da amostra (>= 01/07/2025):
  (a) SENSIBILIDADE: o canal de cada modelo só conta se r = score/limiar >= k (k >= 1; k = 1 é a produção).
      Aproximação: flag_k = flag_produção E r >= k (debounce/portões ficam como na produção; só tira pontos).
      Também o voto >= 3 como ponto à parte.
  (b) CAMADA DE DECISÃO: a grade do A3 (3.968 configurações de filtro, janela do canal 4, refratário e tags; pkl
      salvo pelo a3_loeo_decisao.py). Envelope: para cada nº de trips, a configuração de menor FP/mês. É OTIMISTA:
      escolhida olhando os 8 trips (o A3 mostrou LOEO 5/8); vale como teto, não como promessa.
Saída: tabela no terminal e relatorio_ocsvm_cnn1d_mensal/figs/curva_compromisso.png.

RESULTADO (19/10/2026), produção = 8/8, 2,885 FP/mês (fora 3/3, 2,877):
  (a) k = 3: 7/8 (perde T2, de dentro), fora 3/3; FP/mês 2,335 (-19%), fora 2,627 (-9%).
      k = 4-12: 6/8 (T2, T4), fora 3/3; 2,06-2,13 (-26 a -29%), fora 2,502 (-13%).
      k = 16: 5/8, fora 3/3; 1,854 (-36%), fora 2,502. k >= 24 perde 6 trips. Voto >= 3: 0/8.
      Os 3 trips de fora resistem até k = 16 (score bem acima do limiar), mas os FP de fora também: o corte fora
      para em -13%.
  (b) envelope A3: 8/8 com só PI_6240319_AL + PAL_6240315 dá 2,404 (-17%), fora 2,251 (-22%); 7/8 dá 1,923 (-33%).
      Escolhido vendo os trips (LOEO do A3 = 5/8): teto otimista.
Leitura: não há ponto "de graça". Cada trip a menos compra ~0,2-0,3 FP/mês no período inteiro e bem menos fora.

Uso: python frente_ocsvm/e3_curva_compromisso.py
"""
from __future__ import annotations
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import comum as C
from d3_score_continuo import razoes

FIG = C.AQUI.parent / "relatorio_ocsvm_cnn1d_mensal" / "figs" / "curva_compromisso.png"
KS = [1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]


def placar(canais, op, ft, dur=45.0, jan=None, refr=48.0, votos=2, meses=None):
    cls, m = C.avalia(C.decide(canais, votos, dur, refr), op, ft)
    P = C.por_trip(cls, ft)
    fp_fora = int(((cls.classe == "falso_positivo") & (cls.start >= C.SPLIT)).sum())
    perdidos = [f"T{i}" for i, d in enumerate(P.deteccao, 1) if pd.isna(d)]
    return dict(det=int(P.deteccao.notna().sum()), det_fora=int(P[P.amostra == "fora"].deteccao.notna().sum()),
                fp_mes=round(m["falso_positivo_por_mes"], 3), fp_mes_fora=round(fp_fora / meses, 3),
                perdidos=",".join(perdidos) or "-")


def main():
    df = C.carrega_canais()
    idx, op = df.index, df["operational_state"]
    ft = C.trips(idx)
    cat = C.catalogo()
    meses = C.compute_operational_period_days(pd.DataFrame({"operational_state": op[idx >= C.SPLIT]})) / 30.4375
    c4 = C.canal_alarme(idx, cat.loc[cat.tag.isin(C.TAGS_C4), "t"], C.REF["janela_c4_h"])
    R = razoes(idx)
    pd.set_option("display.width", 200)

    L = []
    for k in KS:
        ch = {c: df[c].astype(bool) & (R[c] >= k).fillna(False) for c in C.CANAIS[:3]}
        ch["c4"] = c4
        L.append(dict(botao="sensibilidade", k=k, **placar(ch, op, ft, meses=meses)))
    ch = {c: df[c].astype(bool) for c in C.CANAIS[:3]}; ch["c4"] = c4
    L.append(dict(botao="voto>=3", k=1, **placar(ch, op, ft, votos=3, meses=meses)))
    A = pd.DataFrame(L)
    print("(a) SENSIBILIDADE -- produção: k = 1")
    print(A.to_string(index=False))

    G = pd.read_pickle(C.DADOS / "a3_grade.pkl")
    G["n"] = G.det.map(lambda v: int(np.sum(v)))
    B = []
    for n in range(8, 3, -1):
        g = G[G.n == n]
        if not len(g):
            continue
        dur, jan, refr, tags = g.loc[g.fp.idxmin(), "cfg"]
        chB = {c: df[c].astype(bool) for c in C.CANAIS[:3]}
        chB["c4"] = C.canal_alarme(idx, cat.loc[cat.tag.isin(tags), "t"], jan)
        B.append(dict(n=n, filtro_min=dur, janela_c4_h=jan, refrat_h=refr, tags=len(tags),
                      **placar(chB, op, ft, dur=dur, refr=refr, meses=meses)))
    B = pd.DataFrame(B)
    print("\n(b) ENVELOPE DA GRADE DO A3 (otimista: escolhido vendo os trips)")
    print(B.to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for a, col, det, tit in ((ax[0], "fp_mes", "det", "período inteiro (8 trips)"),
                             (ax[1], "fp_mes_fora", "det_fora", "fora da amostra, desde 07/2025 (3 trips)")):
        s = A[A.botao == "sensibilidade"]
        a.plot(s[col], s[det], "o-", color="#d73027", label="sensibilidade (r ≥ k)")
        for _, r in s.iterrows():
            if col == "fp_mes" and r.k in (1, 3, 4, 16, 64):
                a.annotate(f"k={r.k:g}", (r[col], r[det]), textcoords="offset points", xytext=(4, 4), fontsize=7)
        v = A[A.botao == "voto>=3"].iloc[0]
        a.plot(v[col], v[det], "s", color="k", label="voto ≥ 3")
        a.plot(B[col], B[det], "^--", color="#4575b4", label="envelope da grade A3 (otimista)")
        a.plot(A.iloc[0][col], A.iloc[0][det], "*", ms=14, color="#1a9850", label="produção")
        a.set_title(tit); a.set_xlabel("FP/mês"); a.grid(alpha=0.3)
    ax[0].set_ylabel("trips detectados"); ax[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Curva de compromisso da OCSVM de produção: trips detectados × FP/mês", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG); print(f"\nfigura: {FIG}")


if __name__ == "__main__":
    main()
