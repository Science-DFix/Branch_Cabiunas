#!/usr/bin/env python3
"""Fixar o dia do retreino é problema? E o dia 1 é melhor por estrutura ou por ajuste?

A PERGUNTA. Produção retreina sempre no dia 1 (`constroi_bundle.py --mes`). Nas 8
composições da régua o dia 1 é o melhor em tudo. Se for sorte pura, o dia 1 não
seria o melhor de forma consistente ao longo do tempo. O teste: dividir o período
avaliado em duas metades e ver a posição do dia 1 em cada uma.

RESULTADO (29/09/2026):

                     1ª metade (jan-jul/25)     2ª metade (ago/25-abr/26)
    trips                    5                          3
    detecção dia 1          5/5 (outros 2-4)           3/3 (outros 2-3)
    FP/mês dia 1        0,000 -- 1º de 8           0,575 -- 1º de 8
    carga dia 1          41,2 -- 3º de 8            54,0 -- 1º de 8

O dia 1 é o melhor nas DUAS metades. Por acaso puro, ser o 1º em FP nas duas teria
~1,6% de chance. Então não é sorte pura -- mas o teste NÃO separa as duas
explicações que sobram:
  · ajuste: limiares, FIT_POINTS e vizinhança foram escolhidos sobre os sinais do
    dia 1 no período INTEIRO; as duas metades estão dentro da amostra do ajuste;
  · estrutura: algo na operação alinhado ao calendário faria o baseline que termina
    no dia 1 representar melhor o mês seguinte.
Separar exigiria reajustar o ponto de operação em cada dia de retreino (~560
configurações x 8 dias) e ver se o dia X, ajustado para si, fica tão bom quanto o
dia 1. Ou esperar os meses de produção.

CONSEQUÊNCIA. Fixar o dia é necessário (bundle determinístico, validade,
reprodução) e o dia 1 deve ficar: é o melhor nas duas metades, e trocar quebraria a
cadeia validada. O que não se deve esperar é que o dia 1 de um mês FUTURO repita o
histórico: é uma composição que o ajuste nunca viu.

Uso:  PYTHONPATH=. python dia_fixo.py
"""
from __future__ import annotations
import pandas as pd
import regua_fp as R

CORTE = pd.Timestamp("2025-08-01", tz="UTC")


def main():
    metades = {"1a metade (jan-jul/25)": R.idx < CORTE, "2a metade (ago/25-abr/26)": R.idx >= CORTE}
    L = []
    for dia in R.DIAS:
        fin = R.detector(*R.sinais(dia))["fin"]
        cls = R.DB.classifica_regra_c(R.AV.episodios(fin), R.PARADAS)
        for nome, sm in metades.items():
            s = pd.Series(sm, index=R.idx)
            meses = float((R.mask & s).sum()) * 2 / 60 / 730
            fp = [(a, b) for a, b, k, _ in cls if k == "FP" and s.loc[a]]
            ne = [(a, b) for a, b, k, _ in cls if k == "NEUTRO" and s.loc[a]]
            h = lambda E: sum((b - a).total_seconds() / 3600 for a, b in E)
            trips = [t for t in R.alvo if s.loc[:t].iloc[-1]]
            det = sum(1 for t in trips if fin.loc[t - R.JAN:t - pd.Timedelta(minutes=2)].any())
            L.append(dict(dia=dia, metade=nome, trips=len(trips), det=det,
                          fp_mes=len(fp) / meses, carga=(h(fp) + h(ne)) / meses))
    T = pd.DataFrame(L)
    for nome in metades:
        S = T[T.metade == nome].set_index("dia")
        S["pos_fp"] = S.fp_mes.rank(method="min").astype(int)
        S["pos_carga"] = S.carga.rank(method="min").astype(int)
        print(f"\n{nome} ({S.trips.iloc[0]} trips)")
        print(S[["det", "fp_mes", "carga", "pos_fp", "pos_carga"]].round(3).to_string())
    T.to_csv(R.CACHE / "dia_fixo.csv", index=False)


if __name__ == "__main__":
    main()
