#!/usr/bin/env python3
"""Precisa retreinar todo mês? A cadência medida no detector atual, em vários cenários.

POR QUÊ REFAZER. A tabela que justifica o retreino mensal (docstring de
`constroi_bundle.py`) tem três fraquezas:
  · foi medida no detector v1 (0,517 FP/mês), não no v2 que está em produção;
  · foi medida só com retreino no dia 1 -- o cenário que se revelou o melhor dos 8;
  · nela o TRIMESTRAL teve menos FP (0,431) e menos horas (4,48) que o mensal
    (0,517; 7,15), ao custo de 1 detecção -- que com 8 eventos está no ruído.
"Congelar é ruim" estava bem sustentado. "Tem de ser mensal", não.

O TESTE. Walk-forward com retreino a cada k meses (1, 2, 3, 6) e congelado (um
único ajuste antes de 2025). Para cada cadência, os cenários variam o DIA do corte
(1, 4, 8, ..., 25) e a FASE (em que mês o ciclo começa: com k = 3 há três fases
possíveis). O detector é o v2 do ponto de deploy, sem nada mudado. O mês é sempre
pontuado pelo último ajuste feito ANTES dele.

CRITÉRIO, escrito antes de rodar. Uma cadência mais espaçada que a mensal é
aceitável se, na mediana dos seus cenários, detecção, início e banda não ficam
abaixo da mensal e FP/mês e carga não ficam acima. Com 8 eventos, diferença de
meia detecção na mediana é ruído -- a leitura é de tendência, não de prova.

RESULTADO (30/09/2026) -- o mensal se confirma, por outro motivo. 104 cenários.

    cadência     cenários   det   início   banda   FP/mês   carga h/mês
    mensal           8      6,5     6       4,5     0,86      130,5
    bimestral       16      6,5     5       4,0     0,95      178,8
    trimestral      24      7,0     5       4,0     0,82      187,3
    semestral       48      7,0     5       4,0     0,82      177,9
    congelado        8      8,0     4       3,0     1,55      315,7

  · o FP/mês quase não muda até o semestral (0,82-0,95, dentro do ruído). O que
    piora é a CARGA: +36% a +44% de horas de alarme, porque com baseline velho os
    canais ficam acesos mais tempo;
  · o nascimento na janela piora: início 6 -> 5 e banda 4,5 -> 4 com qualquer
    cadência mais espaçada. Com 8 eventos é tendência, não prova -- a carga é a
    diferença robusta, por ser medida contínua;
  · a detecção "de pé" SOBE com baseline velho (7 e até 8/8 congelado) enquanto
    início e banda caem. Não é detecção melhor: é alarme aceso mais tempo cobrindo
    mais janelas. É por isso que a régua "de pé" engana;
  · congelado é o pior em tudo que importa: FP x1,8, carga x2,4, banda 3/8.

A tabela antiga (v1, só dia 1) mostrava o trimestral MAIS BARATO que o mensal
(4,48 contra 7,15 h/mês). No v2 e em vários cenários é o contrário: +44% de carga.
E o custo do mensal é 3,4 s por mês -- não há argumento de custo contra ele.

Uso:  PYTHONPATH=. python cadencia_retreino.py
"""
from __future__ import annotations
import io, contextlib
import numpy as np, pandas as pd
import regua_fp as R
with contextlib.redirect_stdout(io.StringIO()):
    import drift_composicao as DC

CADENCIAS = (1, 2, 3, 6, None)          # None = congelado
T_CONG = pd.Timestamp("2025-01-01", tz="UTC")


def walkforward(k: int | None, fase: int, dia: int):
    """Como `drift_composicao.walkforward_dia`, com retreino a cada k meses."""
    IX, DF, STABLE, C, DET, FIT = DC.IX, DC.DF, DC.STABLE, DC.C, DC.DET, DC.FIT
    meses = pd.date_range(IX[0].normalize().replace(day=1), IX[-1], freq="MS", tz="UTC")
    if k is None:
        cortes = [T_CONG + pd.Timedelta(days=dia - 1) - pd.DateOffset(months=1)]
    else:
        cortes = [m + pd.Timedelta(days=dia - 1) for i, m in enumerate(meses) if i % k == fase]
    cortes = [c for c in cortes if IX[0] < c < IX[-1]]
    n = len(IX)
    t = np.full(n, np.nan); p = np.full(n, np.nan)
    ms = np.full(n, np.nan); ds = np.full(n, np.nan)
    for i, c0 in enumerate(cortes):
        c1 = cortes[i + 1] if i + 1 < len(cortes) else IX[-1] + pd.Timedelta("2min")
        fit = DF.loc[STABLE & (IX < c0), C.SENSOR_TAGS].dropna().tail(FIT)
        if len(fit) < FIT // 4:
            continue
        s = (IX >= c0) & (IX < c1)
        if not s.any():
            continue
        w = DF.loc[s]
        t[s] = DC.ScorerMax().fit(fit[C.TEMPERATURE_TAGS]).score(w[C.TEMPERATURE_TAGS])["pca_recon"].to_numpy()
        p[s] = DC.ScorerMax().fit(fit[C.PRESSURE_TAGS]).score(w[C.PRESSURE_TAGS])["pca_recon"].to_numpy()
        b = DET._spread_mancal(fit)
        ms[s] = float(b.median()); ds[s] = float((b - b.median()).abs().median() * 1.4826)
    return t, p, ms, ds


def sinais(k, fase, dia):
    if k == 1 and fase == 0:
        return R.sinais(dia)                      # o mesmo cache da régua
    f = R.CACHE / f"cad_k{k or 0}_f{fase}_d{dia:02d}.npz"
    if f.exists():
        z = np.load(f); return z["t"], z["p"], z["ms"], z["ds"]
    t, p, ms, ds = walkforward(k, fase, dia)
    np.savez(f, t=t, p=p, ms=ms, ds=ds)
    return t, p, ms, ds


def main():
    L = []
    for k in CADENCIAS:
        fases = range(k) if k else [0]
        for fase in fases:
            for dia in R.DIAS:
                m = R.mede(R.detector(*sinais(k, fase, dia))["fin"]); m.pop("cls")
                L.append(dict(cadencia=("congelado" if k is None else f"{k} mes" + ("es" if k > 1 else "")),
                              k=k or 99, fase=fase, dia=dia, **m))
    T = pd.DataFrame(L)
    pd.set_option("display.width", 180)
    cols = ["det", "inicio", "banda", "fp_mes", "h_fp_mes", "carga_mes"]
    G = T.groupby(["k", "cadencia"])
    print(f"{'cadência':11s} {'cenários':>9s} " + " ".join(f"{c:>15s}" for c in cols))
    for (k, nome), S in G:
        cel = [f"{S[c].median():6.2f} [{S[c].min():.4g}-{S[c].max():.4g}]" if c.endswith("mes")
               else f"{S[c].median():4.1f} [{S[c].min():.0f}-{S[c].max():.0f}]" for c in cols]
        print(f"{nome:11s} {len(S):9d} " + " ".join(f"{x:>15s}" for x in cel))
    ref = T[T.k == 1]
    print("\ncontra o mensal (mediana):")
    for (k, nome), S in G:
        if k == 1:
            continue
        ok = (S.det.median() >= ref.det.median() and S.inicio.median() >= ref.inicio.median()
              and S.banda.median() >= ref.banda.median()
              and S.fp_mes.median() <= ref.fp_mes.median() and S.carga_mes.median() <= ref.carga_mes.median())
        print(f"  {nome:11s} det {S.det.median() - ref.det.median():+.1f}  início {S.inicio.median() - ref.inicio.median():+.1f}"
              f"  banda {S.banda.median() - ref.banda.median():+.1f}  FP/mês {S.fp_mes.median() - ref.fp_mes.median():+.3f}"
              f"  carga {S.carga_mes.median() - ref.carga_mes.median():+.1f}   -> {'aceitável' if ok else 'pior'}")
    T.to_csv(R.CACHE / "cadencia_retreino.csv", index=False)


if __name__ == "__main__":
    main()
