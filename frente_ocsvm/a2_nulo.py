#!/usr/bin/env python3
"""A2 -- o 8/8 está acima do acaso? Três nulos.

PRÉ-REGISTRADO EM 10/10/2026, ANTES DE RODAR. Os 8/8 dependem do canal 4 (sem ele: 0/8), que fica aceso
46,8% do tempo em operação (A0) e tem enriquecimento 1,23x (p = 0,19). Nunca se mediu a pipeline inteira
contra o acaso. Na régua, "detecção" = um episódio NASCE em [T-48 h, T].

N1  TRIPS EM INSTANTES SORTEADOS. Para cada instante t em operação ("on"), se algum episódio da decisão
    final nasce em [t-48 h, t]. A fração q é a chance de "detectar" um instante qualquer. Esperado por
    acaso: 8q; p-valor binomial de observar >= 8 de 8 (e >= 3 de 3 fora da amostra, com q do período fora).
N2  PARADAS REAIS COMO SE FOSSEM TRIPS. Pseudo-trips = início de cada parada real (operational_state !=
    "on" por >= 2 h) a mais de 48 h de qualquer trip curado. Taxa de "detecção" nelas contra 8/8 nos
    trips. Se for parecida, a pipeline antecipa PARADAS, não trips (a lição do "relógio de partida" da
    frente física: trips acontecem logo depois de partidas, e o alarme se concentra ali).
N3  CANAL 4 DESLOCADO. O canal 4 é deslocado circularmente por k = 7, 14, ..., 280 dias (40 deslocamentos;
    mesmo duty, mesma textura) e a decisão inteira é refeita (voto >= 2, 45 min, 48 h). Distribuição de
    detecções e FP/mês. Se a mediana ficar perto de 8/8, o alinhamento do canal 4 com os trips não
    importa: ele funciona como um portão que fica aberto metade do tempo.
    Descritivo junto: canal 4 SEMPRE LIGADO (o limite: voto >= 2 vira "qualquer canal OCSVM").

EXPECTATIVA REGISTRADA.
  N1: q ~ 0,2 a 0,3 (75 episódios em ~600 dias de operação); 8 de 8 com p muito pequeno -- a pipeline bate
      instantes aleatórios. Fora da amostra, 3/3 com p ~ 0,02 a 0,05.
  N2: taxa nas paradas reais de 40% a 70%, bem acima de q: boa parte do "acerto" vem de antecipar paradas.
  N3: mediana de 5 a 7 detecções com o canal 4 deslocado (o alinhamento importa pouco); canal 4 sempre
      ligado: 8/8 com FP/mês bem maior.
  Se N1 passar e N2 e N3 confirmarem a expectativa, a leitura é: a pipeline detecta "algo antes de
  parar", e o 8/8 específico de trips não está demonstrado.

Uso:  python frente_ocsvm/a2_nulo.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import binom
import comum as C

DESLOC_DIAS = list(range(7, 281, 7))            # 40 deslocamentos
JAN = pd.Timedelta(hours=48)


def taxa_em(instantes: np.ndarray, inicios: np.ndarray) -> np.ndarray:
    """Para cada instante t, se algum início de episódio cai em [t-48 h, t]."""
    pos = np.searchsorted(inicios, instantes, side="right") - 1
    ok = pos >= 0
    out = np.zeros(len(instantes), bool)
    out[ok] = (instantes[ok] - inicios[pos[ok]]) <= np.timedelta64(JAN)
    return out


def paradas(op: pd.Series, min_h=2.0) -> pd.DatetimeIndex:
    off = (op != "on").to_numpy().astype(np.int8)
    d = np.diff(np.concatenate(([0], off, [0])))
    ini, fim = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    idx = op.index
    return pd.DatetimeIndex([idx[a] for a, b in zip(ini, fim) if (idx[b] - idx[a]) >= pd.Timedelta(hours=min_h)])


def _n3(k_dias):
    import comum as CC
    df = CC.carrega_canais()
    passos = int(k_dias * 24 * 120)                # grade de 30 s
    canais = {c: df[c].astype(bool) for c in CC.CANAIS[:3]}
    canais["canal_alarme_processo"] = pd.Series(np.roll(df["canal_alarme_processo"].astype(bool).values, passos), index=df.index)
    fin = CC.decide(canais, CC.REF["min_votes"], CC.REF["dur_min"], CC.REF["refrat_h"])
    _, m = CC.avalia(fin, df["operational_state"], CC.trips(df.index))
    return k_dias, m["falhas_detectadas"], m["falso_positivo_por_mes"]


def main():
    df = C.carrega_canais()
    idx, op = df.index, df["operational_state"]
    ft = C.trips(idx)
    fin = C.decide({c: df[c].astype(bool) for c in C.CANAIS}, **{k: C.REF[k] for k in ("min_votes", "dur_min", "refrat_h")})
    cls, m = C.avalia(fin, op, ft)
    ini = np.sort(pd.DatetimeIndex(cls.start).values.astype("datetime64[ns]"))
    print(f"referência: {m['falhas_detectadas']}/8, {len(cls)} episódios, {m['falso_positivo_por_mes']:.3f} FP/mês\n")

    # N1
    on = idx[(op == "on").values].values.astype("datetime64[ns]")
    hit = taxa_em(on, ini)
    q = hit.mean()
    fora = on >= np.datetime64(C.SPLIT)
    qf = hit[fora].mean(); qd = hit[~fora].mean()
    print("N1 -- instantes de operação sorteados")
    print(f"  q (todo o período) = {q:.3f} -> esperado {8 * q:.2f} de 8; P(>= 8 de 8) = {binom.sf(7, 8, q):.2e}")
    print(f"  dentro da amostra q = {qd:.3f} -> P(>= 5 de 5) = {binom.sf(4, 5, qd):.2e}")
    print(f"  fora da amostra   q = {qf:.3f} -> P(>= 3 de 3) = {binom.sf(2, 3, qf):.3f}\n")

    # N2
    par = paradas(op)
    longe = np.array([np.min(np.abs((ft - p).dt.total_seconds())) > 48 * 3600 for p in par])
    pseudo = par[longe].values.astype("datetime64[ns]")
    h2 = taxa_em(pseudo, ini)
    pf = pseudo >= np.datetime64(C.SPLIT)
    print("N2 -- paradas reais (>= 2 h, a mais de 48 h de um trip) como pseudo-trips")
    print(f"  {len(pseudo)} paradas: 'detectadas' {h2.sum()} ({h2.mean():.1%}) | dentro {h2[~pf].mean():.1%} de {(~pf).sum()}"
          f" | fora {h2[pf].mean():.1%} de {pf.sum()}")
    print(f"  contra os trips: 8/8 (100%); contra instantes quaisquer: {q:.1%}\n")

    # N3
    import multiprocessing as mp
    with mp.get_context("fork").Pool(6) as pool:
        R = pool.map(_n3, DESLOC_DIAS)
    T = pd.DataFrame(R, columns=["desloc_dias", "det", "fp_mes"])
    canais = {c: df[c].astype(bool) for c in C.CANAIS[:3]}
    canais["canal_alarme_processo"] = pd.Series(True, index=idx)
    _, m1 = C.avalia(C.decide(canais, C.REF["min_votes"], C.REF["dur_min"], C.REF["refrat_h"]), op, ft)
    print("N3 -- canal 4 deslocado (40 deslocamentos de 7 a 280 dias)")
    print(f"  detecções: mediana {T.det.median():.1f}, faixa {T.det.min()}-{T.det.max()}, "
          f"8/8 em {(T.det == 8).sum()} de {len(T)}; >= 7 em {(T.det >= 7).sum()}")
    print(f"  FP/mês: mediana {T.fp_mes.median():.2f}, faixa {T.fp_mes.min():.2f}-{T.fp_mes.max():.2f}")
    print(f"  P(det >= 8 | canal 4 deslocado) = {(T.det >= 8).mean():.3f}")
    print(f"  canal 4 SEMPRE LIGADO: {m1['falhas_detectadas']}/8, {m1['falso_positivo_por_mes']:.2f} FP/mês")
    print("\n  distribuição das detecções:", T.det.value_counts().sort_index().to_dict())
    T.to_csv(C.DADOS / "a2_n3_deslocamentos.csv", index=False)


if __name__ == "__main__":
    main()
