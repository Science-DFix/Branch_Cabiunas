#!/usr/bin/env python3
"""B2 -- janela do canal 4 x filtro de duração (x refratário): existe ponto mais barato que sobrevive fora da amostra?

PRÉ-REGISTRADO EM 10/10/2026, ANTES DE RODAR -- MAS NÃO É CEGO: o A3 (secundário, 5 tags fixas) já mostrou
que (45 min, 6 h, 36 h) pega 8/8 a 2,61 FP/mês e que o LOEO com as 5 tags dá 8/8. Isso foi visto. Para o
B2 trazer informação nova, a escolha é feita SÓ com o período anterior ao split (01/07/2025: 5 trips e o
FP desses meses) e o teste é o período posterior (3 trips e o FP desses meses), que nenhuma escolha usa.
Ressalva honesta: a grade abaixo foi desenhada sabendo que 6 h e 36 h aparecem no A3; isso favorece o B2.

A GRADE (5 tags de produção fixas; 3 canais OCSVM e voto >= 2 fixos):
  filtro de duração  {30, 40, 45, 50, 60, 75, 90} min
  janela do canal 4  {3, 4, 6, 8, 12, 18, 24, 36} h
  refratário         {24, 30, 36, 42, 48} h            -> 280 configurações
(No histórico, "refratário != 48 h" só testou 60 h, que perde 2 trips; 24-42 h nunca foi testado.)

ESCOLHA (só período DENTRO, < 01/07/2025): a configuração que detecta mais dos 5 trips de dentro; empate,
menor FP/mês de dentro; empate, a mais próxima da referência (passos de grade em cada eixo).

PRIMÁRIO -- a escolhida avaliada no período FORA (>= 01/07/2025):
  detecções dos 3 trips de fora e banda de 4 h; FP/mês de fora contra a referência (45 min, 24 h, 48 h)
  no mesmo período, com IC 95% da diferença por bootstrap em blocos de mês-calendário (2.000
  reamostragens dos meses de fora; FP/mês = episódios FP / dias "on" x 30,4375, como a régua). Um
  episódio conta no período em que NASCE.
SECUNDÁRIOS: LOEO sobre os 8 trips na grade inteira (mesmo critério do A3); quantas configurações dão
8/8; a escolhida no período inteiro (8/8 e FP/mês total); onde ficam os ganhos/perdas por trip.

CRITÉRIO DE ACEITE: fora 3/3 E banda de fora >= a da referência (2/3) E FP/mês de fora <= 90% do da
referência com o IC 95% da diferença inteiro abaixo de 0 E LOEO 8/8.

EXPECTATIVA REGISTRADA. Dentro, a escolha cai numa janela curta (6-8 h) e refratário < 48 h, a ~2,3-2,6
FP/mês. Fora: 3/3 mantidos (antecedências de fora são curtas, 3,8-13,7 h, e janela curta não as atrapalha);
FP/mês de fora cai 5-15% contra a referência, mas com ~10 meses e ~25 episódios o IC cruza 0. LOEO 8/8.
Leitura prevista: REPROVADO pelo IC -- direção favorável, mas o período fora é curto demais para provar
custo; a ação seria manter a referência e reavaliar quando houver mais meses pós-split.

Uso:  python frente_ocsvm/b2_janela_filtro.py
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import comum as C

DUR = [30, 40, 45, 50, 60, 75, 90]
JAN = [3, 4, 6, 8, 12, 18, 24, 36]
REFR = [24, 30, 36, 42, 48]
REF = (45, 24, 48)
N_BOOT, SEED = 2000, 20261010
_G: dict = {}


def _init():
    df = C.carrega_canais()
    cat = C.catalogo()
    t4 = cat.loc[cat.tag.isin(C.TAGS_C4), "t"]
    _G.update(df=df, t4=t4, ft=C.trips(df.index), ocsvm={c: df[c].astype(bool) for c in C.CANAIS[:3]})


def blocos(cls: pd.DataFrame, op: pd.Series) -> pd.DataFrame:
    """Episódios FP (pelo mês em que nascem) e dias 'on' por mês-calendário."""
    dt = op.index.to_series().diff().median().total_seconds() / 86400.0
    dias = (op == "on").groupby(op.index.to_period("M")).sum() * dt
    n = pd.to_datetime(cls.loc[cls.classe == "falso_positivo", "start"]).dt.to_period("M").value_counts()
    return pd.DataFrame({"fp": n.reindex(dias.index, fill_value=0), "dias": dias})


def _tarefa(cfg):
    dur, jan, ref = cfg
    df, ft = _G["df"], _G["ft"]
    c = dict(_G["ocsvm"])
    c["c4"] = C.canal_alarme(df.index, _G["t4"], jan)
    cls, m = C.avalia(C.decide(c, 2, dur, ref), df["operational_state"], ft)
    P = C.por_trip(cls, ft)
    return dict(cfg=cfg, det=P.deteccao.notna().to_numpy(), banda=P.banda_4h.to_numpy(),
                fp=m["falso_positivo_por_mes"], meses=blocos(cls, df["operational_state"]))


def taxa(b: pd.DataFrame, i=None) -> float:
    f, d = b.fp.to_numpy(), b.dias.to_numpy()
    if i is not None:
        f, d = f[i], d[i]
    return f.sum() / d.sum() * 30.4375


def dist(cfg) -> int:
    return sum(abs(G.index(v) - G.index(r)) for G, v, r in zip((DUR, JAN, REFR), cfg, REF))


def main():
    import multiprocessing as mp
    with mp.get_context("fork").Pool(6, initializer=_init) as pool:
        R = pool.map(_tarefa, list(itertools.product(DUR, JAN, REFR)), chunksize=4)
    _init()
    ft = _G["ft"]
    nomes = [t.strftime("%d/%m/%y") for t in ft]
    fora = (ft >= C.SPLIT).to_numpy()
    split = pd.Period(C.SPLIT, "M")
    for r in R:
        r["fp_dentro"] = taxa(r["meses"][r["meses"].index < split])
        r["fp_fora"] = taxa(r["meses"][r["meses"].index >= split])
    pd.to_pickle(R, C.DADOS / "b2_grade.pkl")
    ref = next(r for r in R if r["cfg"] == REF)

    esc = min(R, key=lambda r: (-r["det"][~fora].sum(), r["fp_dentro"], dist(r["cfg"])))
    a = esc["meses"][esc["meses"].index >= split]
    b = ref["meses"][ref["meses"].index >= split]
    rng = np.random.default_rng(SEED)
    k = len(a)
    bs = [taxa(a, i) - taxa(b, i) for i in (rng.integers(0, k, k) for _ in range(N_BOOT))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"grade: {len(R)} configurações | meses fora: {k}")
    for nome, r in (("referência", ref), ("escolhida (só dentro)", esc)):
        c = r["cfg"]
        print(f"{nome:22s} {c[0]} min, {c[1]} h, {c[2]} h | dentro {r['det'][~fora].sum()}/5, "
              f"{r['fp_dentro']:.3f} FP/mês | fora {r['det'][fora].sum()}/3 (banda {r['banda'][fora].sum()}/3), "
              f"{r['fp_fora']:.3f} FP/mês | total {r['det'].sum()}/8, {r['fp']:.3f}")
    d = esc["fp_fora"] - ref["fp_fora"]
    print(f"PRIMÁRIO: FP/mês fora {d:+.3f} ({d / ref['fp_fora']:+.1%}), IC 95% {lo:+.3f} a {hi:+.3f}")
    perde = [n for n, x, y in zip(nomes, esc["det"], ref["det"]) if y and not x]
    ganha = [n for n, x, y in zip(nomes, esc["det"], ref["det"]) if x and not y]
    print(f"por trip: perde {perde or '-'}, ganha {ganha or '-'}")

    D = np.stack([r["det"] for r in R])
    print(f"\nSECUNDÁRIO: {int((D.sum(1) == 8).sum())} configurações dão 8/8")
    L = []
    for e in range(8):
        outros = np.delete(D, e, axis=1).sum(1)
        i = min(np.flatnonzero(outros == outros.max()), key=lambda j: (R[j]["fp"], dist(R[j]["cfg"])))
        L.append((nomes[e], R[i]["cfg"], bool(D[i, e]), round(R[i]["fp"], 3)))
    loeo = sum(x[2] for x in L)
    print(f"LOEO (grade inteira, 8 trips): {loeo}/8 | " + "; ".join(f"{n} {c} {'ok' if ok else 'PERDE'}" for n, c, ok, _ in L))

    aceita = (esc["det"][fora].sum() == 3 and esc["banda"][fora].sum() >= ref["banda"][fora].sum()
              and esc["fp_fora"] <= 0.9 * ref["fp_fora"] and hi < 0 and loeo == 8)
    print(f"\nDECISÃO: {'ACEITO' if aceita else 'REPROVADO'}")


if __name__ == "__main__":
    main()
