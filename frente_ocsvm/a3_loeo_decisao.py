#!/usr/bin/env python3
"""A3 -- a camada de decisão escolhida SEM cada trip ainda o pega? (LOEO aninhado)

PRÉ-REGISTRADO EM 10/10/2026, ANTES DE RODAR. O filtro de 45 min, a janela de 24 h do canal 4, o
refratário de 48 h e as 5 tags do canal 4 foram escolhidos olhando os mesmos 8 trips. O A2 mostrou que o
canal 4 só funciona ALINHADO com os trips -- e o alinhamento pode ter sido escolhido.

A GRADE (3.968 configurações; os 3 canais OCSVM e o voto >= 2 ficam fixos):
  filtro de duração   {0, 10, 20, 30, 40, 45, 50, 60} min
  janela do canal 4   {6, 12, 24, 48} h
  refratário          {24, 36, 48, 60} h
  tags do canal 4     os 31 subconjuntos não vazios das 5 tags de produção
Cada configuração: quais dos 8 trips detecta (régua de sempre), FP/mês e banda de 4 h.
LIMITE, dito antes: as 5 tags já vieram de um catálogo de 47 escolhido olhando os trips; a grade só
reescolhe DENTRO delas. Isto SUBESTIMA o otimismo de seleção, não o superestima.

O LOEO. Para cada trip e: entre todas as configurações, escolhe a que detecta mais dos OUTROS 7; empate,
menor FP/mês; empate, a mais próxima da referência (distância: passos de grade em cada eixo + número de
tags diferentes). Avalia se ela detecta e (e se na banda). Mesmo critério de setembro (8/8 primeiro,
depois FP), só que sem ver e.
  PRIMÁRIO: detecções LOEO (de 8).
  SECUNDÁRIOS: (i) o mesmo LOEO com as 5 tags fixas (só filtro, janela e refratário); (ii) FP/mês
  mediano das 8 configurações escolhidas; (iii) quantas configurações dão 8/8 dentro da grade e onde a
  referência está entre elas.

EXPECTATIVA REGISTRADA. LOEO 6/8 a 7/8 (os frágeis: 26/02/2026, nascido a 3,8 h, e 11/04/2025, que cai
com filtro de 14-15 min); com as 5 tags fixas, 7/8. FP/mês das escolhidas perto de 2,9. A referência
está entre as de 8/8 com menor FP, mas não é a única.
Leitura prevista: se o LOEO der >= 7/8, a escolha da decisão não é o que sustenta o 8/8; se der <= 5/8, o
8/8 é em boa parte seleção.

Uso:  python frente_ocsvm/a3_loeo_decisao.py
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import comum as C

DUR = [0, 10, 20, 30, 40, 45, 50, 60]
JAN = [6, 12, 24, 48]
REFR = [24, 36, 48, 60]
SUBS = [tuple(s) for k in range(1, 6) for s in itertools.combinations(C.TAGS_C4, k)]
REF = (45, 24, 48, tuple(C.TAGS_C4))
_G: dict = {}


def _init():
    df = C.carrega_canais()
    cat = C.catalogo()
    _G.update(df=df, cat=cat, ft=C.trips(df.index), ocsvm={c: df[c].astype(bool) for c in C.CANAIS[:3]})


def _tarefa(cfg):
    dur, jan, ref, tags = cfg
    df, cat, ft = _G["df"], _G["cat"], _G["ft"]
    c = dict(_G["ocsvm"])
    c["c4"] = C.canal_alarme(df.index, cat.loc[cat.tag.isin(tags), "t"], jan)
    fin = C.decide(c, 2, dur, ref)
    cls, m = C.avalia(fin, df["operational_state"], ft)
    P = C.por_trip(cls, ft)
    return cfg, P.deteccao.notna().to_numpy(), P.banda_4h.to_numpy(), m["falso_positivo_por_mes"]


def dist(cfg) -> float:
    d, j, r, t = cfg
    return (abs(DUR.index(d) - DUR.index(REF[0])) + abs(JAN.index(j) - JAN.index(REF[1]))
            + abs(REFR.index(r) - REFR.index(REF[2])) + len(set(t) ^ set(REF[3])))


def loeo(T: pd.DataFrame, mascara=None) -> pd.DataFrame:
    X = T if mascara is None else T[mascara]
    D = np.stack(X.det.values); B = np.stack(X.banda.values)
    L = []
    for e in range(D.shape[1]):
        outros = np.delete(D, e, axis=1).sum(1)
        cand = X[outros == outros.max()]
        esc = cand.assign(d=cand.cfg.map(dist)).sort_values(["fp", "d"]).iloc[0]
        i = X.index.get_loc(esc.name)
        L.append(dict(evento=e, cfg=esc.cfg, outros=int(outros.max()), detecta=bool(D[i, e]),
                      banda=bool(B[i, e]), fp=round(esc.fp, 3)))
    return pd.DataFrame(L)


def main():
    import multiprocessing as mp
    cfgs = list(itertools.product(DUR, JAN, REFR, SUBS))
    with mp.get_context("fork").Pool(6, initializer=_init) as pool:
        R = pool.map(_tarefa, cfgs, chunksize=16)
    T = pd.DataFrame([dict(cfg=c, det=d, banda=b, fp=f) for c, d, b, f in R])
    T.to_pickle(C.DADOS / "a3_grade.pkl")
    _init(); ft = _G["ft"]
    nomes = [t.strftime("%d/%m/%y") for t in ft]
    T["n"] = T.det.map(lambda v: int(np.sum(v)))
    ref = T[T.cfg == REF].iloc[0]
    print(f"grade: {len(T)} configurações | referência: {ref.n}/8, {ref.fp:.3f} FP/mês")
    oito = T[T.n == 8].sort_values("fp")
    print(f"configurações com 8/8: {len(oito)}; a referência é a {list(oito.cfg).index(REF) + 1}ª mais barata entre elas"
          f" (mínimo {oito.fp.min():.3f} em {oito.iloc[0].cfg[:3]} com {len(oito.iloc[0].cfg[3])} tags)")
    for nome, msk in (("PRIMÁRIO (grade inteira)", None), ("SECUNDÁRIO (5 tags fixas)", T.cfg.map(lambda c: c[3] == REF[3]))):
        Lr = loeo(T, msk)
        Lr["evento"] = Lr.evento.map(lambda e: nomes[e])
        print(f"\n{nome}: LOEO {int(Lr.detecta.sum())}/8 detectados, {int(Lr.banda.sum())}/8 na banda | "
              f"FP/mês das escolhidas: mediana {Lr.fp.median():.3f}")
        print(Lr.assign(cfg=Lr.cfg.map(lambda c: f"{c[0]}min {c[1]}h {c[2]}h {len(c[3])}tags")).to_string(index=False))


if __name__ == "__main__":
    main()
