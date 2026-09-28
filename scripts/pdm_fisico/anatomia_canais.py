#!/usr/bin/env python3
"""Por que os canais ficam acesos 34-59% do tempo normal? Degrau ou CUSUM?

Cada canal é `degrau | CUSUM`: o degrau acende com o EWMA acima do limiar por
15 amostras seguidas; o CUSUM acende com o acumulador acima de H. `coativacao.py`
mostrou que, em operação normal, os canais se comportam como ESTADOS (ligados por
dias) e não como eventos -- e que isso põe um teto na informação que cada um
carrega: razão de verossimilhança máxima de 1/ciclo.

Aqui se separa quem produz esse ciclo:
  · ciclo só do degrau, só do CUSUM, e da união -- em operação normal;
  · fração do tempo aceso em que o EWMA já está ABAIXO do limiar (o canal só
    está de pé pela memória do acumulador);
  · duração típica de cada trecho aceso;
  · nos episódios nascidos na borda do blackout (6,4667 h), qual ramo acendeu
    cada canal essencial no nascimento.

CORREÇÃO QUE MOTIVA ESTE SCRIPT. Supus que o CUSUM atravessava a parada (o reset
multiplica por 0,25) e por isso os canais nasciam acesos na borda. Relendo o
código: o fator é aplicado a CADA instante mascarado, 180 vezes no blackout, e o
acumulador vai a zero. Não há travessia. A borda 6,4667 h = (180 + 14) × 2 min é
o 15º instante liberado -- o SUSTAIN do degrau. A medida abaixo confirma ou não.

Uso:  PYTHONPATH=. python anatomia_canais.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import regua_fp as R
import coativacao as CO
import pericia_fp_atual as PF

DB = R.DB


def ramos(E_all: dict, K: dict) -> dict:
    """degrau, CUSUM e o acumulador, por canal, exatamente como em R.detector."""
    m = R.mask
    out = {}
    for c in R.SIN:
        thr = DB.BASE[c] * K[c]
        E = E_all[c].where(m)
        deg = ((E > thr).astype(int).rolling(DB.SUSTAIN, min_periods=DB.SUSTAIN).sum()
               >= DB.SUSTAIN) & m
        S = DB.cusum(((E / thr).clip(upper=20) - DB.KAPPA).fillna(0.0).to_numpy(), DB.reset)
        cu = pd.Series(S > DB.H_CUSUM, index=R.idx) & m
        out[c] = dict(deg=deg, cu=cu, S=pd.Series(S, index=R.idx),
                      abaixo=(E <= thr).fillna(False))
    return out


def trechos(x: np.ndarray) -> np.ndarray:
    """duração (h) de cada trecho contíguo True."""
    if not x.any():
        return np.array([])
    d = np.diff(np.concatenate(([0], x.astype(int), [0])))
    ini, fim = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return (fim - ini) * 2 / 60


def main():
    t, p, ms, ds = R.sinais(1)
    d = R.detector(t, p, ms, ds)
    normal, pre = CO.janelas()
    for nivel, K in (("A", DB.K_LO), ("B", DB.KH)):
        rm = ramos(d["EW"], K)
        print(f"\nNÍVEL {nivel} — operação normal ({normal.sum()} instantes vigiados)")
        print(f"  {'canal':6s} {'só degrau':>10s} {'só CUSUM':>9s} {'união':>7s} "
              f"{'aceso c/ EWMA<limiar':>21s} {'trecho aceso: mediana':>22s} {'p90':>7s}")
        for c in R.SIN:
            dg, cu = rm[c]["deg"].to_numpy(), rm[c]["cu"].to_numpy()
            un = dg | cu
            ab = rm[c]["abaixo"].to_numpy()
            tr = trechos(un & normal)
            print(f"  {c:6s} {100 * dg[normal].mean():9.1f}% {100 * cu[normal].mean():8.1f}%"
                  f" {100 * un[normal].mean():6.1f}% {100 * (un & ab)[normal].sum() / max(un[normal].sum(), 1):20.1f}%"
                  f" {np.median(tr) if len(tr) else 0:20.1f} h {np.quantile(tr, .9) if len(tr) else 0:6.1f} h")

    # nascimentos na borda: que ramo acendeu os canais essenciais
    T = pd.read_csv(R.CACHE / "pericia_fp_atual.csv")
    T["inicio"] = pd.to_datetime(T["inicio"]).dt.tz_localize("UTC")
    rmA, rmB = ramos(d["EW"], DB.K_LO), ramos(d["EW"], DB.KH)
    print("\nNASCIMENTOS NA BORDA DO BLACKOUT — que ramo acendeu cada canal aceso")
    print(f"  {'classe':7s} {'início':17s} {'nível':6s} {'canal:ramo (A | B)'}")
    for _, r in T[T.borda].iterrows():
        a = r.inicio
        cel = []
        for c in R.SIN:
            ra = "deg" if rmA[c]["deg"].loc[a] else "cus" if rmA[c]["cu"].loc[a] else "-"
            rb = "deg" if rmB[c]["deg"].loc[a] else "cus" if rmB[c]["cu"].loc[a] else "-"
            if ra != "-" or rb != "-":
                cel.append(f"{c}:{ra}|{rb}")
        print(f"  {r.classe:7s} {a:%Y-%m-%d %H:%M}  {r.nivel:6s} {'  '.join(cel)}")


if __name__ == "__main__":
    main()
