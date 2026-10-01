#!/usr/bin/env python3
"""F3b (item 3.3 do plano do especialista): PCA dinâmico (DPCA) nos canais t e p.

PRÉ-REGISTRADO EM 01/10/2026, ANTES DE RODAR (commit deste arquivo sem resultado).

A IDEIA. O PCA estático só aprende correlação no MESMO instante. O DPCA aumenta a
matriz com cópias defasadas dos sensores, [X(t), X(t - L1), X(t - L2)], e passa a
aprender a dinâmica normal: uma rampa que o processo sempre faz deixa de ser resíduo;
uma dinâmica que nunca aconteceu vira resíduo. É o padrão em controle estatístico
multivariado de processos autocorrelacionados (Ku, Storer & Georgakis, 1995).

O DESENHO. Mesmo baseline (20.000 pontos estáveis antes do corte), mesmo ScorerMax:
o score é o máximo, sobre as colunas (sensor, defasagem), do erro normalizado pelo
p99 de cada coluna, dividido pelo p99 do score no baseline. Os limiares são múltiplos
desse p99 -- adimensionais --, então nada é reajustado. A grade de 2 min é regular
(passo único em 612.629 instantes): a defasagem é por posição. sp e vb não mudam.
  defasagens 0, 10 e 30 min (PRIMÁRIA, IC 95%)
  defasagens 0 e 60 min     (secundária, IC 97,5%)
Decisão: `bootstrap_regua.decide`, pareado nas 8 composições.

EXPECTATIVA REGISTRADA: baixa. As causas de FP medidas não são dinâmicas: a borda do
blackout (transiente que a máscara deixa entrar) e os estados longos (canais acesos
por memória do CUSUM). O DPCA pode reduzir resíduo de rampa, mas a rampa de partida
já fica fora da máscara.

Uso:  PYTHONPATH=. python dpca.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import regua_fp as R
import aprovacao_operador as AO
import bootstrap_regua as BR

C, DF, IX, STABLE, FIT, DC = AO.C, AO.DF, AO.IX, AO.STABLE, AO.FIT, AO.DC
VARIANTES = (("lags 0-10-30 min", (0, 5, 15), True), ("lags 0-60 min", (0, 30), False))


_AUM: dict = {}


def aumenta(cols, lags) -> pd.DataFrame:
    """[X(t), X(t-L1), ...] na grade inteira, com nomes 'col@L' (memoizado)."""
    k = (tuple(cols), tuple(lags))
    if k not in _AUM:
        X = DF[cols]
        _AUM[k] = pd.concat([X.shift(L).add_suffix(f"@{L}") for L in lags], axis=1)
    return _AUM[k]


def sinais(dia: int, lags, rotulo: str):
    f = R.CACHE / f"dpca_{'_'.join(map(str, lags))}_d{dia:02d}.npz"
    if f.exists():
        z = np.load(f); return z["t"], z["p"], z["ms"], z["ds"]
    t, p, ms, ds = (x.copy() for x in R.sinais(dia))
    AT, AP = aumenta(C.TEMPERATURE_TAGS, lags), aumenta(C.PRESSURE_TAGS, lags)
    cs = AO.cortes(dia)
    for i, c0 in enumerate(cs):
        c1 = cs[i + 1] if i + 1 < len(cs) else AO.FIM
        base = DF.loc[STABLE & (IX < c0), C.SENSOR_TAGS].dropna().tail(FIT)
        if len(base) < FIT // 4:
            continue
        s = (IX >= c0) & (IX < c1)
        if not s.any():
            continue
        for A, alvo in ((AT, "t"), (AP, "p")):
            fit = A.loc[base.index].dropna()
            sc = DC.ScorerMax().fit(fit)
            v = sc.score(A.loc[s])["pca_recon"].to_numpy()
            if alvo == "t":
                t[s] = v
            else:
                p[s] = v
    np.savez(f, t=t, p=p, ms=ms, ds=ds)
    return t, p, ms, ds


def main():
    ref = BR.braco("referência", lambda d: R.detector(*R.sinais(d))["fin"], R.DIAS)
    for rot, lags, prim in VARIANTES:
        var = BR.braco(rot, lambda d, l=lags, r=rot: R.detector(*sinais(d, l, r))["fin"], R.DIAS)
        nivel = 0.95 if prim else 1 - 0.05 / 2
        x = BR.decide(ref, var, pareado=True, nivel=nivel)
        r, v = ref.tabela, var.tabela
        print(f"{rot:17s} {'(primária)' if prim else '(secundária)'}  det {x['det']} início {x['inicio']} banda {x['banda']}"
              f" FP {v.fp_mes.median():.3f} | Δcarga {x['d_carga']:+.1f} [{x['ic_lo']:+.1f}; {x['ic_hi']:+.1f}]"
              f" IC {100 * nivel:.1f}% | ΔFP {x['d_fp']:+.3f} | carga cai em {int((v.carga_mes.values < r.carga_mes.values).sum())}/8"
              f" | {x['veredito']}", flush=True)
        print("     por composição: " + "  ".join(
            f"d{d}: {r.det[d]}→{v.det[d]}/{r.banda[d]}→{v.banda[d]}/{r.carga_mes[d]:.0f}→{v.carga_mes[d]:.0f}" for d in R.DIAS),
            flush=True)


if __name__ == "__main__":
    main()
