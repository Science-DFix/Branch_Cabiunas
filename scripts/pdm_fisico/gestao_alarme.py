#!/usr/bin/env python3
"""Gestão de alarme na apresentação (ISA-18.2): notifica no nascimento, depois
"condição conhecida".

POR QUÊ AQUI. Todas as alavancas de detector testadas nesta frente trocam custo
por detecção (`cusum_memoria.py`, `ewma_vigiado.py`, `regimes.py`,
`transicao_observada.py`). A carga do operador, porém, é dominada por episódios
longos: os 7 NEUTRO somam 492 h no dia 1, e há episódios de 112 a 154 h. Pela
ISA-18.2, alarme ativo por dias é "alarme obsoleto" -- um defeito de projeto, não
informação.

A REGRA. O detector não muda. Na APRESENTAÇÃO, cada episódio fica ATIVO nas
primeiras N horas; depois vira "condição conhecida" e sai da carga ativa. O
reanúncio já existe no v2: a escalada por idade corta o episódio e cria um
nascimento novo quando a força cruza 20x depois de 96 h -- e nascimento novo é
notificação nova.

O QUE NÃO PODE MUDAR, POR CONSTRUÇÃO. Nenhum nascimento muda, então a régua de
início e a banda acionável ficam idênticas, e as notificações por mês também.

O RISCO, que é o que se mede. Um episódio longo que já virou condição conhecida
quando o trip chega: o operador foi avisado no nascimento, mas nas 48 h anteriores
ao trip não há alarme ATIVO. Conta-se, por composição, quantos trips "de pé" na
referência perdem o alarme ativo.

N em 12, 24 e 48 h. É parâmetro de PROJETO, não de ajuste: não há critério de
aceite, há uma tabela de troca para a engenharia escolher.

RESULTADO (29/09/2026). Funciona como projeto de apresentação -- não reduz FP.

    (mediana das 8)   notif/mês   início   carga ATIVA    trips sem alarme ativo nas 48 h
    atual               2,07       6/8     130,5 h/mês       --
    ativo 12 h          2,07       6/8      13,8  (-89%)     1  (0 a 3)
    ativo 24 h          2,07       6/8      24,5  (-81%)     1  (0 a 2)
    ativo 48 h          2,07       6/8      43,0  (-67%)     1  (0 a 2)

O custo é quase todo um trip: 17/03/2025, cujo alarme nasceu 195-201 h (8 dias)
antes -- o operador seria avisado em 09/03 e o alarme viraria condição conhecida
muito antes do trip. Em uma composição cada, 09/12 e 07/04 também perdem.

O limite, dito claramente: as notificações falsas continuam ~0,86 por mês. O que
cai são as horas de alarme aceso sem nada novo a dizer. É a melhor troca desta
frente, e é decisão de projeto da integração, não resultado do detector.

Uso:  PYTHONPATH=. python gestao_alarme.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import regua_fp as R

NS = (12.0, 24.0, 48.0)


def apresenta(fin: pd.Series, N: float) -> pd.Series:
    """ativo = as primeiras N horas de cada episódio."""
    ativo = pd.Series(False, index=fin.index)
    for a, b in R.AV.episodios(fin):
        ativo.loc[a:min(b, a + pd.Timedelta(hours=N))] = True
    return ativo & fin


def mede(fin: pd.Series, N: float | None) -> dict:
    ativo = fin if N is None else apresenta(fin, N)
    eps = R.AV.episodios(fin)
    cls = R.DB.classifica_regra_c(eps, R.PARADAS)
    mes = float(R.mask.sum()) * 2 / 60 / 730
    h_ativa = lambda k: sum(float(ativo.loc[a:b].sum()) * 2 / 60 for a, b, kk, _ in cls if kk == k)
    de_pe = sum(1 for t in R.alvo if ativo.loc[t - R.JAN:t - pd.Timedelta(minutes=2)].any())
    ini = sum(1 for t in R.alvo if any(t - R.JAN <= a <= t for a, _ in eps))
    return dict(notif_mes=len(eps) / mes, inicio=ini, de_pe_ativo=de_pe,
                carga_ativa=(h_ativa("FP") + h_ativa("NEUTRO")) / mes,
                h_fp_ativa=h_ativa("FP") / mes)


def main():
    L = []
    for dia in R.DIAS:
        fin = R.detector(*R.sinais(dia))["fin"]
        L.append(dict(dia=dia, N="atual", **mede(fin, None)))
        for N in NS:
            L.append(dict(dia=dia, N=f"{N:.0f} h", **mede(fin, N)))
    T = pd.DataFrame(L)
    pd.set_option("display.width", 170)
    print("POR COMPOSIÇÃO")
    print(T.pivot(index="dia", columns="N", values=["carga_ativa", "de_pe_ativo"]).round(1).to_string())
    print(f"\n{'':8s} {'notif/mês':>10s} {'início':>7s} {'de pé ativo':>12s} {'carga ativa':>12s} "
          f"{'h FP ativa':>11s}   (mediana das 8 composições)")
    for N in ["atual"] + [f"{n:.0f} h" for n in NS]:
        S = T[T.N == N]
        print(f"{N:8s} {S.notif_mes.median():10.2f} {S.inicio.median():5.0f}/8 {S.de_pe_ativo.median():10.1f}/8"
              f" {S.carga_ativa.median():12.1f} {S.h_fp_ativa.median():11.1f}")
    ref = T[T.N == "atual"].set_index("dia")
    for N in [f"{n:.0f} h" for n in NS]:
        S = T[T.N == N].set_index("dia")
        perde = (ref.de_pe_ativo - S.de_pe_ativo)
        print(f"  {N}: trips que perdem o alarme ATIVO nas 48 h: mediana {perde.median():.1f}, "
              f"faixa {perde.min():.0f} a {perde.max():.0f}  | carga ativa cai "
              f"{100 * (1 - S.carga_ativa.median() / ref.carga_ativa.median()):.0f}%")
    T.to_csv(R.CACHE / "gestao_alarme.csv", index=False)


if __name__ == "__main__":
    main()
