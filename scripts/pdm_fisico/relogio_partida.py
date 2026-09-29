#!/usr/bin/env python3
"""O comparador que faltava: um alarme que só sabe que a máquina acabou de partir.

A DESCOBERTA. 5 dos 8 trips acontecem entre 9 e 31 h depois de uma partida, e a
máquina passa só 16,7% do tempo de operação nessa janela:

    janela após a partida   tempo de pé nela   trips   esperado   enriq.   p binomial
    <= 24 h                     13,4%          4/8      1,08      3,7x     0,015
    <= 32 h                     16,7%          5/8      1,34      3,7x     0,005
    <= 48 h                     22,9%          5/8      1,83      2,7x     0,019

O risco de trip é ~3,7x maior no primeiro dia após uma partida. As janelas foram
escolhidas depois de ver os dados, mas o efeito se mantém de 24 a 48 h e segue
abaixo de 0,05 com Bonferroni pelas 4 testadas. Só 1 dos 5 é trip repetido
(11/04, logo após o 07/04); os outros vêm de paradas comuns.

O COMPARADOR. Se a partida sozinha já prevê trip, o detector tem de ser comparado
contra um alarme que não olha sinal nenhum: aceso da abertura da máscara até H
horas depois de cada partida. Nenhum nulo usado antes controlava isso -- o
deslocamento circular tira o alarme de cima das partidas junto com os trips.

RESULTADO (29/09/2026):

    regra                cobertura   det   início   banda   FP/mês
    relógio <= 24 h         9,3%     5/8    5/8      4/8     2,84
    detector (mediana)     ~34%      6,5    6/8      4/8     0,86

Separando os trips por regime (detector, mediana das 8 composições):

    pós-partida (<= 32 h)   5 trips   detector 3,5/5   relógio 5/5
    regime longo            3 trips   detector 3/3     relógio 0/3

Na régua operacional (banda), o relógio EMPATA com o detector. Nos trips pós-partida
o detector é PIOR que o relógio. O valor físico do detector -- o que nada mais
pega -- está nas 3 falhas de regime longo: 3/3 em 7 das 8 composições. Com 3
eventos isso é fraco (p ~ 0,04 com cobertura de 34%), mas consistente.

A consequência para a comunicação: o "8/8" é "3/3 no regime longo, onde só ele
pega" mais "3,5/5 pós-partida, onde um relógio pega 5/5".

Uso:  PYTHONPATH=. python relogio_partida.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import binom
import regua_fp as R
import pericia_fp_atual as PF
import nulo_circular as NC

POS_H = 32.0


def enriquecimento():
    op = PF.PP.op.to_numpy(); hrel = PF.horas_desde_religamento().to_numpy()
    aval = op & np.asarray(R.sel)
    tr = np.array([hrel[R.idx.searchsorted(t) - 1] for t in R.alvo])
    L = []
    for H in (12, 24, 32, 48):
        base = float((hrel[aval] <= H).mean()); k = int((tr <= H).sum())
        L.append(dict(janela_h=H, tempo_de_pe=base, trips=k, esperado=8 * base,
                      enriquecimento=k / (8 * base), p=binom.sf(k - 1, 8, base)))
    return pd.DataFrame(L), tr


def relogio():
    hrel = PF.horas_desde_religamento(); L = []
    for H in (24, 32, 48):
        fin = (hrel <= H) & R.mask
        m = R.mede(fin); obs, nulo = NC.det_por_deslocamento(fin)
        L.append(dict(regra=f"partida <= {H} h", cobertura=float(fin[R.mask].mean()),
                      det=m["det"], inicio=m["inicio"], banda=m["banda"],
                      fp_mes=m["fp_mes"], carga=m["carga_mes"],
                      acima_acaso=obs - nulo.mean(), p=float((nulo >= obs).mean())))
    return pd.DataFrame(L)


def por_regime():
    hrel = PF.horas_desde_religamento()
    pos = [t for t in R.alvo if hrel.iloc[R.idx.searchsorted(t) - 1] <= POS_H]
    lon = [t for t in R.alvo if t not in pos]
    L = []
    for dia in R.DIAS:
        fin = R.detector(*R.sinais(dia))["fin"]; eps = R.AV.episodios(fin)
        det = lambda ts: sum(1 for t in ts if fin.loc[t - R.JAN:t - pd.Timedelta(minutes=2)].any())
        ini = lambda ts: sum(1 for t in ts if any(t - R.JAN <= a <= t for a, _ in eps))
        L.append(dict(dia=dia, det_pos=det(pos), ini_pos=ini(pos),
                      det_longo=det(lon), ini_longo=ini(lon)))
    return pd.DataFrame(L).set_index("dia"), pos, lon


if __name__ == "__main__":
    pd.set_option("display.width", 170)
    E, tr = enriquecimento()
    print(f"horas desde a partida nos 8 trips: {np.round(np.sort(tr), 1)}\n")
    print(E.round(4).to_string(index=False))
    print("\nRELÓGIO DE PARTIDA\n" + relogio().round(3).to_string(index=False))
    T, pos, lon = por_regime()
    print(f"\nDETECTOR POR REGIME  (pós-partida: {[t.strftime('%m-%d') for t in pos]}; "
          f"regime longo: {[t.strftime('%Y-%m-%d') for t in lon]})")
    print(T.to_string())
