#!/usr/bin/env python3
"""T-a -- a carga na definição do Thallys (EXP 61/62) como regra de "atenção" na pipeline OCSVM final (com E5). Local.

PRÉ-REGISTRO (25/10/2026, commitado antes de rodar)
Pergunta (pedido do usuário: "levar a carga para a nossa pipeline"): no detector v2, rebaixar a "atenção" o alarme
em carga baixa corta FP em todas as bases testadas (T2 e T3, frente_thallys/). Na OCSVM isso se repete? O nosso D3
(18/10) já testou "carga no alerta" e não separou (AUC 0,40-0,47), mas com outra definição. A do Thallys, nunca
testada aqui: L = fração das amostras estáveis (RUNNING_A > 0,5 e T5 > 300) dos 30 d de referência com T5 abaixo da
mediana do T5 na janela de notificação.
BASE: a pipeline final com E5 (`f1_congelado.decide` sobre os canais de `point_anomalies_final.csv`): 8/8, 41 FP,
2,816 FP/mês; fora da amostra 3/3, 22 FP, 2,752. Episódios = os da régua (junta a < 120 min), todas as classes.
CAUSALIDADE: o alerta da nossa pipeline sai em n = início + 45 min (filtro de duração). Janela de notificação =
[n - 2 h, n]; referência = os 30 d antes da janela. É o espelho causal do Thallys, cuja notificação sai 2 h depois do
nascimento e cuja janela é [nascimento, nascimento + 2 h]. Janela sem amostra estável ou referência com < 30
amostras -> L indefinido -> o episódio fica (regra do Thallys).
BRAÇO C (principal): L < 0,5 -> atenção (o episódio sai da série de alarme; pós-processamento, o refratário já armado
fica como está, como no EXP 62).
BRAÇO C4 (segundo braço, mesmo critério): L < 0,5 OU canal 4 apagado em todo o [início, n] -> atenção (o T3 do v2).
VIZINHANÇA (um fator por vez, 8, nos dois braços): θ 0,4; 0,6; indicador = média dos TC382_01..06; referência 15 d;
60 d; janela 1 h; 4 h; a janela literal do Thallys [início, início + 2 h] (equivale a decidir 2 h depois).
PLACEBO (principal de cada braço): os rebaixamentos embaralhados entre os episódios (500 vezes, semente 20261026).
p_conj = fração com FP <= o da regra E detecção >= a da regra.
CRITÉRIO (o da frente, o mesmo do E4/E6/H1): PASSA = 8/8 E 3/3 fora E FP/mês fora <= 0,75 x 2,877 = 2,158.
  ROBUSTO = PASSA E >= 6 das 8 vizinhas com 8/8 e FP/mês fora <= 2,158 E p_conj < 0,10.
  FRÁGIL = PASSA sem o resto.  NÃO = não PASSA.
  Relato extra (não decide): o critério do Thallys aqui, ou seja, perda <= 1 trip e FP/mês fora <= 2,158.
SECUNDÁRIOS: por episódio, L e o canal 4 na notificação, com a classe e as horas desde a partida; quantos dos 16 FP de
partida e dos T4, T6, T7 (que começam na partida) têm L indefinido ou baixo.
EXPECTATIVAS:
  Os episódios de partida têm janela de notificação na rampa do T5: ou L indefinido (fica) ou L baixo (sai). Se saem,
  saem junto T4/T6/T7, como em todo filtro de partida (E4, H0). Prevejo: braço C com FP 41 -> 28-36 (~55%); perde >= 1
  de T4/T6/T7 (~55%); PASSA (~20%); NÃO (~70%), FRÁGIL (~20%), ROBUSTO (~10%).
  Braço C4: o canal 4 é um dos 4 votos e fica aceso em ~98% das 6 h pós-partida; tira 2-6 FP a mais que o C (~55%) e
  com mais risco de trip. NÃO (~75%).
LIMITES: 8 trips; o T5 é um indicador imperfeito de carga; a definição veio do v2, e não foi ajustada aqui.
VERIFICAÇÕES: V1 a base reproduz 8/8, 41 FP, 2,816/mês, fora 3/3, 22 FP; V2 sem rebaixar nada, a série é idêntica;
V3 o L daqui, com a janela literal e a referência do Thallys, reproduz `L_de` do EXP 62 em 50 instantes sorteados
(diferença 0).

Uso:  python frente_ocsvm/ta_carga_thallys.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(AQUI), str(AQUI.parent / "frente_thallys")]
import reproduz  # noqa: E402,F401  (fixa o NV_DADOS para o EXP 62 do Thallys, usado na V3)
import comum as C  # noqa: E402
import f1_congelado as F  # noqa: E402

GRADE = Path.home() / ".clearml/cache/storage_manager/datasets/ds_8b06a98f8b264820a9ecf2075a188395/grade2min.parquet"
SENSORES = Path.home() / ".clearml/cache/storage_manager/datasets/ds_a97ba56ba14840fbb1125c2a82f883c9/sensores_full_2024_2026_30s.csv"
FILTRO = pd.Timedelta(minutes=45)
LIM_FORA = 0.75 * 2.877
PRINC = dict(theta=0.5, ind="T5", ref_d=30, jan_h=2.0, literal=False)
VIZ = [dict(PRINC, theta=0.4), dict(PRINC, theta=0.6), dict(PRINC, ind="TC382"), dict(PRINC, ref_d=15),
       dict(PRINC, ref_d=60), dict(PRINC, jan_h=1.0), dict(PRINC, jan_h=4.0), dict(PRINC, literal=True)]
N_PERM, SEMENTE = 500, 20261026


def indicadores():
    g = pd.read_parquet(GRADE); g.index = pd.DatetimeIndex(g.index).tz_convert(None)   # UTC naive, como a pipeline
    est = ((g["RUNNING_A"] > 0.5) & (g["T5_AVG_A"] > 300)).fillna(False)
    return {"T5": g["T5_AVG_A"].where(est), "TC382": g[[f"TC382_0{i}_A" for i in range(1, 7)]].mean(axis=1).where(est)}


def L(serie, w0, w1, ref_d):
    """Fração da referência (ref_d dias antes de w0, até w0 - 2 min) abaixo da mediana da janela [w0, w1]."""
    w = serie.loc[w0:w1].dropna(); r = serie.loc[w0 - pd.Timedelta(days=ref_d):w0 - pd.Timedelta(minutes=2)].dropna()
    if len(w) == 0 or len(r) < 30:
        return np.nan
    return float((r < float(w.median())).mean())


def L_ep(IND, ini, cfg):
    if cfg["literal"]:
        return L(IND[cfg["ind"]], ini, ini + pd.Timedelta(hours=2), cfg["ref_d"])
    n = ini + FILTRO
    return L(IND[cfg["ind"]], n - pd.Timedelta(hours=cfg["jan_h"]), n, cfg["ref_d"])


def base():
    df = C.carrega_canais()
    pontos = {g: pd.DataFrame({"is_anom_point": df[f"canal_{g}"].astype(bool), "operational_state": df["operational_state"]})
              for g in ("temperatura", "vibracao", "oleo_pressao")}
    cat = F.le_catalogo(C.DADOS / "alarmes_selecionados_turbina_a.csv")
    txt = F.le_texto(str(SENSORES))
    ft = F.le_trips(C.DADOS / "alarmes_francisco_falhas.csv")
    canais, final, op, cls, met = F.decide(pontos, cat, txt, ft)
    ftp = ft[(ft >= op.index.min()) & (ft <= op.index.max())].reset_index(drop=True)
    return canais, final, op, cls, ftp


def mede(final, op, ftp, meses_fora):
    cls, met = F.P.evaluate(final, op, ftp)
    P = C.por_trip(cls, ftp)
    fp = cls[cls.classe == "falso_positivo"]
    fora = int((fp.start >= C.SPLIT).sum())
    return dict(det=int(P.deteccao.notna().sum()), fora_det=int((P.deteccao.notna() & (P.amostra == "fora")).sum()),
                fp=len(fp), fp_mes=met["falso_positivo_por_mes"], fp_fora=fora, fp_mes_fora=fora / meses_fora,
                perdidos=[f"T{i + 1}" for i in range(len(P)) if pd.isna(P.deteccao.iloc[i])], P=P, cls=cls)


def aplica(final, eps, rebaixa):
    f = final.copy()
    for (a, b), r in zip(eps, rebaixa):
        if r:
            f.loc[a:b] = False
    return f


def main():
    canais, final, op, cls0, ftp = base()
    meses_fora = C.compute_operational_period_days(pd.DataFrame({"operational_state": op[op.index >= C.SPLIT]})) / 30.4375
    m0 = mede(final, op, ftp, meses_fora)
    eps = list(zip(cls0.start, cls0.end))
    IND = indicadores()
    c4 = canais["alarme_processo"]
    ok = {"V1": (m0["det"], m0["fora_det"], m0["fp"], round(m0["fp_mes"], 3), m0["fp_fora"]) == (8, 3, 41, 2.816, 22),
          "V2": aplica(final, eps, [False] * len(eps)).equals(final)}
    E62 = __import__("base_n19000")._exp("exp62", "62_regra_de_carga.py")
    t5u = IND["T5"].copy(); t5u.index = t5u.index.tz_localize("UTC")
    rng = np.random.default_rng(0); amostra = rng.choice(len(IND["T5"].dropna()), 50, replace=False)
    ts = IND["T5"].dropna().index[amostra]
    ok["V3"] = all((np.isnan(x) and np.isnan(y)) or x == y for x, y in
                   ((L(IND["T5"], t, t + pd.Timedelta(hours=2), 30), E62.L_de(t5u, t.tz_localize("UTC"), 30, 2)[0]) for t in ts))
    print(f"base: {m0['det']}/8 fora {m0['fora_det']}/3 FP {m0['fp']} {m0['fp_mes']:.3f}/mês, fora {m0['fp_fora']} "
          f"{m0['fp_mes_fora']:.3f} | verificações {ok}", flush=True)
    if not all(ok.values()):
        print("verificação falhou -- parando"); return

    sem_c4 = [not bool(c4.loc[a:a + FILTRO].any()) for a, _ in eps]

    def braco(cfg, usa_c4):
        Ls = [L_ep(IND, a, cfg) for a, _ in eps]
        rb = [(np.isfinite(l) and l < cfg["theta"]) or (usa_c4 and s) for l, s in zip(Ls, sem_c4)]
        return mede(aplica(final, eps, rb), op, ftp, meses_fora), rb, Ls

    def linha(m):
        return (f"{m['det']}/8 fora {m['fora_det']}/3 FP {m['fp']:2d} {m['fp_mes']:.3f}/mês, fora {m['fp_fora']:2d} "
                f"{m['fp_mes_fora']:.3f} | perdidos {','.join(m['perdidos']) or '-'}")

    passa = lambda m: m["det"] == 8 and m["fora_det"] == 3 and m["fp_mes_fora"] <= LIM_FORA
    rng = np.random.default_rng(SEMENTE)
    for nome, usa in (("C (carga)", False), ("C4 (carga OU sem canal 4)", True)):
        m1, rb, Ls = braco(PRINC, usa)
        cont = 0; arr = np.array(rb)
        for _ in range(N_PERM):
            mp = mede(aplica(final, eps, list(rng.permutation(arr))), op, ftp, meses_fora)
            cont += (mp["fp"] <= m1["fp"]) and (mp["det"] >= m1["det"])
        p = cont / N_PERM
        print(f"\nBRAÇO {nome}: rebaixa {sum(rb)} de {len(eps)} episódios")
        print(f"   base     {linha(m0)}")
        print(f"   regra    {linha(m1)} | placebo p_conj {p:.3f}")
        viz_ok = 0
        for cfg in VIZ:
            m, _, _ = braco(cfg, usa); bom = m["det"] == 8 and m["fp_mes_fora"] <= LIM_FORA; viz_ok += bom
            print(f"   θ {cfg['theta']} | {cfg['ind']:5s} | ref {cfg['ref_d']:2d} d | jan {cfg['jan_h']:.0f} h"
                  f"{' literal' if cfg['literal'] else '        '} -> {linha(m)} -> {'ok' if bom else 'não'}")
        cl = "NÃO" if not passa(m1) else "ROBUSTO" if (viz_ok >= 6 and p < 0.10) else "FRÁGIL"
        extra = (m0["det"] - m1["det"] <= 1) and m1["fp_mes_fora"] <= LIM_FORA
        print(f"   CLASSIFICAÇÃO: PASSA {'sim' if passa(m1) else 'não'} | vizinhança {viz_ok}/8 | placebo {p:.3f} -> {cl}"
              f" | critério do Thallys (perda <= 1 e fora <= {LIM_FORA:.3f}): {'sim' if extra else 'não'}")
        if not usa:
            hp = pd.Series(__import__("e4_adia_partida").horas_desde_partida(op), index=op.index)
            print("\n   POR EPISÓDIO (início, classe, h desde a partida, L, canal 4 na notificação, rebaixado)")
            for (a, b), cl_, l, s, r in zip(eps, cls0.classe, Ls, sem_c4, rb):
                print(f"   {a:%d/%m/%y %H:%M} {cl_:15s} {hp.get(a, np.nan):6.1f} h  L {l:5.2f}  c4 {'não' if s else 'sim'}"
                      f"  {'-> atenção' if r else ''}")


if __name__ == "__main__":
    main()
