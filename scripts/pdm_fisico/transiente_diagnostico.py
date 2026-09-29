#!/usr/bin/env python3
"""Na borda do blackout, o transiente ainda está no sinal ou só na memória do EWMA?

CONTEXTO. 6 dos 7 NEUTRO e 2 dos 4 FP nascem em 6,4667 h do religamento, todos
acesos pelo DEGRAU (`anatomia_canais.py`). O EWMA dos canais é calculado sobre o
sinal cru inteiro -- inclusive durante o blackout e a parada -- e só depois
mascarado. Então, quando a máscara abre, o EWMA alto pode ter duas causas, e
cada uma pede um remédio diferente:

  · o transiente AINDA ESTÁ no sinal cru em 6,5 h  -> modelar a trajetória de
    partida e subtraí-la;
  · o transiente JÁ PASSOU e o EWMA só carrega a memória das primeiras horas
    -> reiniciar o EWMA quando a máscara abre. Sem modelo nenhum.

O QUE SE MEDE. Para cada religamento seguido de pelo menos 30 h de operação, o
sinal cru e o EWMA de cada canal em função do tempo desde o religamento, em
unidades do limiar do nível A (1,0 = no limiar). Mediana e p75 entre os
religamentos, em 6,5 h, 8 h, 12 h e 24 h, contra o regime (24-30 h). E à parte,
nos 11 episódios nascidos na borda.

RESULTADO (28/09/2026). 159 religamentos, 57 seguidos de >= 30 h de operação.
O transiente JÁ PASSOU no sinal cru em 6,5 h; o que está alto é o EWMA:

    canal   cru 6,5 h   regime   EWMA 6,5 h
    t         0,28       0,30      1,01
    p         0,14       0,17      5,12   (p75 29,3)
    sp        0,61       0,40      0,61
    vb        1,13       1,14      1,14

Nos 11 nascimentos na borda, p cru vai de 0,04 a 1,43 e o EWMA de 1,30 a 72,9.
Não é caso de modelar o transiente: o EWMA come os dados da partida que a
máscara declara inválidos. A correção está em `ewma_vigiado.py`.

Uso:  PYTHONPATH=. python transiente_diagnostico.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import regua_fp as R
import pericia_fp_atual as PF

TAUS = (6.5, 8.0, 12.0, 24.0)
HORIZ_H = 30.0


def religamentos(min_run_h=HORIZ_H):
    """índices dos religamentos seguidos de >= min_run_h horas de máquina de pé."""
    op = PF.PP.op.to_numpy(); part = PF.PP.part.to_numpy()
    ini = np.flatnonzero(part)
    n = int(min_run_h * 30)
    ok = [i for i in ini if i + n < len(op) and op[i:i + n].all()]
    return np.asarray(ok), ini


def trajetorias(sinal: np.ndarray, ini: np.ndarray, n: int) -> np.ndarray:
    return np.stack([sinal[i:i + n] for i in ini])


def main():
    todos_ok, todos = religamentos()
    print(f"religamentos no período: {len(todos)}  |  seguidos de >= {HORIZ_H:.0f} h "
          f"de operação: {len(todos_ok)}")
    t, p, ms, ds = R.sinais(1)
    d = R.detector(t, p, ms, ds)
    z = np.load("piso_fisico_cache.npz")
    spv = np.abs((z["b_all"] - ms) / ds)
    cru = {"t": t, "p": p, "sp": spv, "vb": R.DB.cru_pub["vb"].to_numpy()}
    n = int(HORIZ_H * 30)
    col = lambda h: int(h * 30)
    print(f"\nem unidades do limiar do nível A (1,0 = no limiar);  "
          f"regime = mediana em 24-30 h\n")
    print(f"  {'canal':5s} {'série':6s} {'regime':>7s}   " +
          "   ".join(f"{h:>4.1f} h med/p75" for h in TAUS))
    for c in R.SIN:
        thr = R.DB.BASE[c] * R.DB.K_LO[c]
        for nome, s in (("cru", cru[c] / thr), ("EWMA", d["EW"][c].to_numpy() / thr)):
            M = trajetorias(s, todos_ok, n)
            reg = np.nanmedian(M[:, col(24):col(30)])
            cel = []
            for h in TAUS:
                v = M[:, col(h)]
                cel.append(f"{np.nanmedian(v):6.2f}/{np.nanquantile(v, .75):5.2f}")
            print(f"  {c:5s} {nome:6s} {reg:7.2f}   " + "   ".join(cel))
        print()

    # nos episódios nascidos na borda
    T = pd.read_csv(R.CACHE / "pericia_fp_atual.csv")
    T["inicio"] = pd.to_datetime(T["inicio"]).dt.tz_localize("UTC")
    print("NOS 11 NASCIMENTOS NA BORDA — sinal cru e EWMA no nascimento (× limiar A),"
          " e o cru médio em 6,0-6,5 h")
    print(f"  {'classe':7s} {'início':17s}  " +
          "  ".join(f"{c + ' cru/EWMA/cru6h':>20s}" for c in R.SIN))
    for _, r in T[T.borda].iterrows():
        i = R.idx.get_loc(r.inicio)
        cel = []
        for c in R.SIN:
            thr = R.DB.BASE[c] * R.DB.K_LO[c]
            e = d["EW"][c].to_numpy()[i] / thr
            ult = cru[c][i - 15:i + 1] / thr
            cel.append(f"{np.nanmean(ult):6.2f}/{e:5.2f}/{np.nanmean(cru[c][i - 30:i - 15] / thr):5.2f}")
        print(f"  {r.classe:7s} {r.inicio:%Y-%m-%d %H:%M}  " + "  ".join(f"{x:>20s}" for x in cel))


if __name__ == "__main__":
    main()
