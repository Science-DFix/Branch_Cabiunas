#!/usr/bin/env python3
"""O D do CVA como VETO de episodio (nao como canal): separa FP de acerto no nascimento?

De onde vem. experimento_cva_detector.py refutou o D como SUBSTITUTO de t e p
(mesma cobertura a 4-5x o custo). A motivacao original do SFA/CVA era outra: os
falsos positivos nascidos na borda do blackout, que seriam transiente de partida
-- ponto de operacao se acomodando, nao dinamica anormal. Se for isso, o D (que
mede se o futuro segue o que o passado previa) deveria estar BAIXO no nascimento
desses FP e ALTO nos acertos, e poderia vetar episodio sem gerar alarme proprio.

Medida. Os 21 episodios do ponto publicado (v2), classificados pela Regra C
(TP / NEUTRO / FP). Para cada um, o D de t e de p nas 2 h seguintes ao
nascimento -- causal em relacao ao alarme, que so e emitido depois da duracao
minima de 120 min. Separacao FP x TP por AUC de postos (0,5 = nenhuma).

RESULTADO (30/09/2026) -- REFUTADO, e a premissa estava desatualizada.

1. A premissa era do v1. "8 dos 12 FP nascem na borda de 6,47 h" e contagem do
   ponto antigo. No v2 publicado restam 4 FP, e so 2 na borda. Dos 11 episodios
   que nascem na borda, 3 sao acertos e 6 sao NEUTRO. O que pesa na carga de
   48,9 h/mes nao e FP: sao os NEUTRO longos (153, 135, 112, 52 h).
2. O D nao separa, e na direcao errada. Os FP tem D MAIOR, nao menor, que os
   acertos (mediana do D de p nas 2 h: FP 1,03 x TP 0,09). E ha valores
   absurdos nos dois grupos (D de p ate 272.340x o p99 num FP, 7.920x num TP):
   o D explode com evento de instrumentacao de pressao -- coerente com a
   sensibilidade a sensor congelado medida na avaliacao (57%). O maximo do D de
   p da AUC 0,82 FP x TP -- separacao real, mas na direcao OPOSTA a hipotese e
   inutil como veto: o corte em 2,0 que remove 3 dos 4 FP remove junto 3 TP
   (27/02, 29/01 e 26/02, com D de p em 7.920, 51 e 86). Temperatura: AUC 0,43
   a 0,55, nada.
3. Com 4 FP, o teto do filtro seria 0,344 -> 0 FP/mes; nenhum limiar chega perto
   sem custar deteccao.

Nao repetir sem informacao nova. O que sobra de acionavel: a carga e dominada
por episodios NEUTRO longos (alarme de pe antes de parada real): 491,7 h de
NEUTRO contra 76,1 h de FP no periodo -- e o problema
de fusao de episodio ja documentado no CONTEXTO_05-06_SET_2026.md, nao de sinal.

Uso (de dentro de scripts/pdm_fisico, com os dados e o cva_d_cache.npz gerado
por experimento_cva_detector.py):  python filtro_cva_d.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import avalia as AV
import publica_clearml as PC
from plota_estilo_francisco import classifica_regra_c, paradas_reais_2h

JANELA_H = 2.0          # horas apos o nascimento -- a duracao minima e 120 min


def auc(a, b):
    """P(valor de a > valor de b), empate = 1/2. a = FP, b = TP."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan
    return float(np.mean([(x > b).mean() + 0.5 * (x == b).mean() for x in a]))


def main():
    al, mask, alvo, _, idx, _ = PC.reproduz(v2=True)
    cls = classifica_regra_c(AV.episodios(al), paradas_reais_2h())
    z = np.load("cva_d_cache.npz")
    D = pd.DataFrame({"t": z["t"], "p": z["p"]}, index=idx).where(mask)

    g = pd.read_parquet("grade2min.parquet", columns=["RUNNING_A"])
    op = (g["RUNNING_A"] > 0.5).fillna(False)
    partida = op & ~op.shift(fill_value=False)
    ultima = pd.Series(np.where(partida, idx, pd.NaT), index=idx).ffill()

    linhas = []
    for a, b, k, *_ in cls:
        w = D.loc[a:a + pd.Timedelta(hours=JANELA_H)]
        linhas.append(dict(
            inicio=a.strftime("%Y-%m-%d %H:%M"), classe=k,
            dur_h=round((b - a).total_seconds() / 3600, 1),
            dist_partida_h=round((a - ultima.loc[a]).total_seconds() / 3600, 2),
            Dt_max=w["t"].max(), Dp_max=w["p"].max(),
            Dt_med=w["t"].median(), Dp_med=w["p"].median()))
    T = pd.DataFrame(linhas)
    T["borda_blackout"] = T["dist_partida_h"].between(6.0, 7.0)

    pd.set_option("display.width", 200)
    print(T.round(2).to_string(index=False))
    print("\nepisodios por classe:", T["classe"].value_counts().to_dict())
    print("na borda do blackout:", T[T.borda_blackout]["classe"].value_counts().to_dict())
    print("\nseparacao FP x TP (AUC; 0,5 = nenhuma, <0,5 = FP MENOR que TP):")
    fp, tp = T[T.classe == "FP"], T[T.classe == "TP"]
    for c in ("Dt_max", "Dp_max", "Dt_med", "Dp_med"):
        print(f"  {c:7s}  AUC = {auc(fp[c], tp[c]):.2f}   "
              f"mediana FP {fp[c].median():.2f}  TP {tp[c].median():.2f}")
    horas = T.groupby("classe")["dur_h"].sum()
    print("\nhoras de alarme por classe:", horas.round(1).to_dict())


if __name__ == "__main__":
    main()
