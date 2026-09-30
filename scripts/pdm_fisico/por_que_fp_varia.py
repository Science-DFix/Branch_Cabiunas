#!/usr/bin/env python3
"""Por que o FP vai de 0,344 a 1,033 com o mesmo detector e o mesmo código?

A pergunta é justa: nas 8 composições da régua o detector, os limiares e os dados
são idênticos. Só muda o DIA do retreino mensal -- e com ele, QUAIS 20.000 pontos
estáveis entram no baseline de cada mês.

O MECANISMO QUE SE MEDE AQUI, elo por elo:
  1. o baseline muda a ESCALA dos canais t e p: eles são erro de reconstrução
     dividido pelo p99 do próprio baseline. Um p99 menor infla o canal inteiro;
  2. escala maior -> o canal passa mais tempo acima do limiar (ciclo);
  3. como os canais são estados perto do limiar, uma mudança pequena de escala
     muda o ciclo por dias, e o voto (3 de 4) multiplica isso;
  4. mais tempo de voto -> mais episódios -> mais FP.

A escala efetiva é medida como a razão, instante a instante, entre o canal na
composição e o canal no dia 1, na mediana de cada mês -- é o quanto aquele mês
ficou "mais alto" ou "mais baixo" só por ter outro baseline.

RESULTADO (29/09/2026) -- a explicação esperada caiu; a verdadeira é o MOMENTO.

  · o elo 1 se confirma, e forte: trocar o baseline muda a escala dos canais mês a
    mês -- o `p` de um mês fica de 25x menor a 10x maior que no dia 1;
  · os elos 2-4 caem: o tempo de voto ligado é parecido em todas as composições
    (40-51% da operação normal) e NÃO se correlaciona com o FP (Spearman -0,29,
    p 0,49). O dia 1 nem tem o menor tempo de voto.

O funil do pós-processamento mostra o mecanismo:

    cenário   trechos de voto   episódios   em janela de trip   NEUTRO   FP
    dia 1           50             21          10 (48%)            7      4
    dia 4           40             25           7 (28%)            6     12
    dia 8           43             23           6 (26%)            6     11
    dia 25          45             24          10 (42%)            7      7

O detector gera quase a mesma quantidade de alarme em todos os cenários. FP é, por
definição, o episódio que NÃO caiu perto de um trip nem de uma parada. No dia 1,
metade dos alarmes calhou de cair dentro de uma janela de 48 h antes de trip; nos
outros cenários, um quarto a dois quintos. O que sobra vira FP.

E o momento é sensível por três razões que se somam: o baseline desloca QUANDO cada
canal cruza o limiar; o refratário de 72 h faz o primeiro trecho de um grupo
bloquear os seguintes (mudou o primeiro, muda quem sobrevive -- o dia 1 barra 24
dos 50 trechos, o que mais barra); e a classificação é por SOBREPOSIÇÃO, então um
alarme longo que encosta numa janela de trip conta como acerto e, deslocado umas
horas, vira FP.

Uso:  PYTHONPATH=. python por_que_fp_varia.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import regua_fp as R
import coativacao as CO


def main():
    normal, _ = CO.janelas()
    t1, p1, _, _ = R.sinais(1)
    m = R.mask.to_numpy()
    meses = pd.Series(R.idx.to_period("M").astype(str), index=R.idx)
    L = []
    for dia in R.DIAS:
        t, p, ms, ds = R.sinais(dia)
        d = R.detector(t, p, ms, ds)
        met = R.mede(d["fin"])
        esc = {}
        for nome, x, x1 in (("t", t, t1), ("p", p, p1)):
            ok = m & np.isfinite(x) & np.isfinite(x1) & (x1 > 0)
            r = pd.Series(x[ok] / x1[ok], index=R.idx[ok])
            por_mes = r.groupby(meses[ok]).median()
            esc[nome] = por_mes
        L.append(dict(
            dia=dia,
            escala_t_med=float(esc["t"].median()), escala_t_faixa=f"{esc['t'].min():.2f}-{esc['t'].max():.2f}",
            escala_p_med=float(esc["p"].median()), escala_p_faixa=f"{esc['p'].min():.2f}-{esc['p'].max():.2f}",
            ciclo_t=100 * d["A"]["t"].to_numpy()[normal].mean(),
            ciclo_p=100 * d["A"]["p"].to_numpy()[normal].mean(),
            voto_A=100 * d["vA"].to_numpy()[normal].mean(),
            voto_total=100 * d["voto"].to_numpy()[normal].mean(),
            episodios=met["episodios"], n_fp=met["n_fp"], fp_mes=met["fp_mes"]))
    T = pd.DataFrame(L).set_index("dia")
    pd.set_option("display.width", 190)
    print("ESCALA = canal nesta composição / canal no dia 1 (mediana por mês; 1,00 = igual)")
    print("CICLO e VOTO = % do tempo em operação normal\n")
    print(T.round(3).to_string())
    for c in ("escala_p_med", "ciclo_p", "voto_total", "episodios"):
        r, pv = spearmanr(T[c], T["fp_mes"])
        print(f"  Spearman {c:13s} x FP/mês: {r:+.2f} (p {pv:.3f})")
    T.to_csv(R.CACHE / "por_que_fp_varia.csv")


if __name__ == "__main__":
    main()
