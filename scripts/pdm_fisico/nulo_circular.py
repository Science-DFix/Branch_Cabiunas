#!/usr/bin/env python3
"""Quantas detecções estão acima do acaso? O nulo que respeita o tempo.

POR QUÊ. `avalia.permuta` sorteia instantes UNIFORMES e pergunta se cairiam com
alarme nas 48 h anteriores. Isso ignora a estrutura temporal do alarme: os
alarmes deste detector duram dias (os canais são estados), e se os trips
tendem a vir depois de partidas -- que geram alarme de borda em 20% das vezes --
o sorteio uniforme subestima o acaso.

O TESTE. Desloca-se o alarme inteiro no tempo, circularmente, em relação aos
trips (deslocamento circular, o nulo padrão em séries temporais). Isso preserva
tudo no alarme -- duração, autocorrelação, ciclo -- e só quebra o alinhamento
com os trips. Deslocamentos de 30 a ~450 dias, de dia em dia; menos de 30 dias
fica de fora para não recolocar o alarme em cima do próprio evento.

Para cada composição e cada versão do detector: detecção observada, média e
desvio do nulo, e p = P(nulo >= observado).

Uso:  PYTHONPATH=. python nulo_circular.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import regua_fp as R

MIN_DIAS = 30


def det_por_deslocamento(fin: pd.Series) -> tuple[int, np.ndarray]:
    sel = R.sel.to_numpy() if hasattr(R.sel, "to_numpy") else np.asarray(R.sel)
    ix = R.idx[sel]
    a = fin.to_numpy()[sel].astype(np.int64)
    n = len(a)
    ini = ix.searchsorted([t - R.JAN for t in R.alvo])
    fim = ix.searchsorted(list(R.alvo))
    def conta(x):
        cs = np.concatenate(([0], np.cumsum(x)))
        return int(((cs[fim] - cs[ini]) > 0).sum())
    obs = conta(a)
    dia = 30 * 24          # amostras de 2 min por dia
    desl = np.arange(MIN_DIAS * dia, n - MIN_DIAS * dia, dia)
    nulo = np.array([conta(np.roll(a, s)) for s in desl])
    return obs, nulo


def main():
    L = []
    for dia in R.DIAS:
        t, p, ms, ds = R.sinais(dia)
        for nome, canais in (("referência", ()), ("EWMA vigiado", ("t", "p", "sp", "vb"))):
            fin = R.detector(t, p, ms, ds, ewma_vigiado=canais)["fin"]
            obs, nulo = det_por_deslocamento(fin)
            L.append(dict(dia=dia, variante=nome, det=obs, nulo_media=nulo.mean(),
                          nulo_dp=nulo.std(), acima=obs - nulo.mean(),
                          p=float((nulo >= obs).mean()), n_desl=len(nulo)))
    T = pd.DataFrame(L)
    pd.set_option("display.width", 160)
    print(f"NULO CIRCULAR — {T.n_desl.iloc[0]} deslocamentos por composição\n")
    print(T.pivot(index="dia", columns="variante",
                  values=["det", "nulo_media", "nulo_dp", "acima", "p"]).round(3).to_string())
    print()
    for v in ("referência", "EWMA vigiado"):
        S = T[T.variante == v]
        print(f"{v:13s} det {S.det.median():.1f}  acaso {S.nulo_media.median():.2f} ± "
              f"{S.nulo_dp.median():.2f}  ACIMA {S.acima.median():.2f} "
              f"[{S.acima.min():.2f} a {S.acima.max():.2f}]  p mediano {S.p.median():.3f}"
              f"  | p < 0,05 em {int((S.p < 0.05).sum())}/8")
    D = T.pivot(index="dia", columns="variante", values="acima")
    dd = D["EWMA vigiado"] - D["referência"]
    print(f"\nacima do acaso, pareado (vigiado − ref): mediana {dd.median():+.2f}, "
          f"faixa {dd.min():+.2f} a {dd.max():+.2f}, vigiado melhor em {int((dd > 0).sum())}/8")
    T.to_csv(R.CACHE / "nulo_circular.csv", index=False)


if __name__ == "__main__":
    main()
