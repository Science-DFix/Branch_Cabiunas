#!/usr/bin/env python3
"""E4 -- adiar o alerta nas primeiras horas após a partida corta FP sem perder trip? Local, sem treino.

PRÉ-REGISTRO (19/10/2026, antes de rodar)
Pergunta (do usuário: "vamos de item 1", recomendação do relatorio_falsos_positivos): o E1 mostrou que metade dos FP
(16 de 36 da OCSVM desde 01/2025) começa até 2 h depois de a máquina voltar a "on", mas T4, T6 e T7 também (1,1 / 0,6
/ 0,0 h após a partida, antecedência 20 / 14 / 8 h). Descartar mataria os 3; ADIAR pode salvá-los se o voto continuar.
Tratamento: os votos não contam nas primeiras h horas depois de cada partida (transição de qualquer estado para "on"
em operational_state). Depois disso, a camada de decisão é a de produção: o voto >= 2 precisa durar 45 min, refratário
48 h. Ou seja, o alerta só sai se a anomalia seguir acesa por 45 min DEPOIS da janela.
  h em {0 (= produção), 0,5, 1, 2, 3, 4, 6} h.
Modelo e episódios: OCSVM de produção (point_anomalies_final.csv, A0), canal 4 de produção, período inteiro.
LOEO sobre h: sem o trip i, o h que detecta mais dos outros 7; empate, menor FP/mês; empate, menor h. Conta se o
trip i é detectado. h final = mesmo critério com os 8.
PASSA = LOEO 8/8 E com o h final 3/3 fora E FP/mês fora <= 0,75 x 2,877 = 2,158 (mesmo critério da D3/E2).
Secundários: FP/mês no período inteiro; antecedência dos 8 trips com o h final (quantos ficam na banda de 4 h);
quantos FP de partida somem e quantos só são adiados.
Expectativa: h = 1-2 mantém 8/8 (T7 é o mais frágil: começa na partida e tem 8,4 h de antecedência), corta 15-25% dos
FP no período inteiro e menos fora (fora tem 6 FP de partida em 23). Prevejo NÃO pelo critério de custo fora, por
pouco; LOEO 7/8 a 8/8.

Uso: python frente_ocsvm/e4_adia_partida.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import comum as C
from src.cnn1d_ae.scoring import apply_min_duration_filter, apply_refractory, combine_channels_vote

HS = [0, 0.5, 1, 2, 3, 4, 6]
LIM_FORA = 0.75 * 2.877


def horas_desde_partida(op: pd.Series) -> np.ndarray:
    on = op.eq("on").values
    t = op.index.values.astype("datetime64[s]").astype(np.int64)
    ini = np.where(on & ~np.r_[False, on[:-1]], t, 0)
    ini = np.maximum.accumulate(ini)
    return np.where(ini > 0, (t - ini) / 3600.0, np.inf)


def roda(voto, hp, h, op, ft, meses_fora):
    v = voto & ~(hp < h)
    dur = apply_min_duration_filter(pd.DataFrame({"is_anom_point": v.astype(int)}), C.REF["dur_min"])["is_anom_point"].astype(bool)
    cls, m = C.avalia(apply_refractory(dur, refractory_minutes=C.REF["refrat_h"] * 60.0), op, ft)
    P = C.por_trip(cls, ft)
    fp_fora = int(((cls.classe == "falso_positivo") & (cls.start >= C.SPLIT)).sum())
    return dict(h=h, det=set(P.trip[P.deteccao.notna()]), fp_mes=m["falso_positivo_por_mes"],
                fp_mes_fora=fp_fora / meses_fora, n_fp=int((cls.classe == "falso_positivo").sum()), fp_fora=fp_fora,
                P=P, cls=cls)


def main():
    df = C.carrega_canais()
    idx, op = df.index, df["operational_state"]
    cat = C.catalogo()
    ch = {k: df[k].astype(bool) for k in C.CANAIS[:3]}
    ch["c4"] = C.canal_alarme(idx, cat.loc[cat.tag.isin(C.TAGS_C4), "t"], C.REF["janela_c4_h"])
    ft = C.trips(idx)
    meses_fora = C.compute_operational_period_days(pd.DataFrame({"operational_state": op[idx >= C.SPLIT]})) / 30.4375
    voto = combine_channels_vote(ch, min_votes=2).astype(bool)
    hp = pd.Series(horas_desde_partida(op), index=idx)

    R = {h: roda(voto, hp, h, op, ft, meses_fora) for h in HS}
    pd.set_option("display.width", 200)
    print(pd.DataFrame([dict(h=h, det=len(r["det"]), det_fora=sum(t >= C.SPLIT for t in r["det"]), FP=r["n_fp"],
                             fp_mes=round(r["fp_mes"], 3), FP_fora=r["fp_fora"], fp_mes_fora=round(r["fp_mes_fora"], 3),
                             perdidos=",".join(f"T{i}" for i, t in enumerate(ft, 1) if t not in r["det"]) or "-")
                        for h, r in R.items()]).to_string(index=False))

    def escolhe(alvo):
        return min(HS, key=lambda h: (-len(R[h]["det"] & alvo), R[h]["fp_mes"], h))

    loeo = []
    for t in ft:
        h_i = escolhe(set(ft) - {t})
        loeo.append((t, h_i, t in R[h_i]["det"]))
    n_loeo = sum(ok for *_, ok in loeo)
    print("\nLOEO:", ", ".join(f"{t:%d/%m/%y} h={h:g} {'ok' if ok else 'PERDE'}" for t, h, ok in loeo))
    hf = escolhe(set(ft))
    r = R[hf]
    fora = sum(t >= C.SPLIT for t in r["det"])
    passa = n_loeo == 8 and fora == 3 and r["fp_mes_fora"] <= LIM_FORA
    print(f"\nh final = {hf:g} h: {len(r['det'])}/8, fora {fora}/3, FP/mês {r['fp_mes']:.3f} "
          f"({100 * (r['fp_mes'] / R[0]['fp_mes'] - 1):+.0f}%), fora {r['fp_mes_fora']:.3f} "
          f"({100 * (r['fp_mes_fora'] / R[0]['fp_mes_fora'] - 1):+.0f}%; limite {LIM_FORA:.3f}); LOEO {n_loeo}/8 "
          f"-> {'PASSA' if passa else 'NÃO'}")
    print("\nantecedência (h) produção x h final:")
    print(pd.DataFrame({"trip": R[0]["P"].trip.dt.strftime("%d/%m/%y"), "prod": R[0]["P"].lead_h.values,
                        f"h={hf:g}": r["P"].lead_h.values, "banda_4h": r["P"].banda_4h.values}).to_string(index=False))
    # FP de partida da produção: somem ou só são adiados?
    c0 = R[0]["cls"]
    fp0 = c0[(c0.classe == "falso_positivo") & (hp.reindex(c0.start).values <= 2)]
    cf = r["cls"]
    adiados = sum(((cf.start >= s) & (cf.start <= s + pd.Timedelta(hours=8)) & (cf.classe == "falso_positivo")).any()
                  for s in fp0.start)
    print(f"\nFP da produção a <= 2 h da partida: {len(fp0)}; com h final, {adiados} viram FP adiado (até 8 h depois), "
          f"{len(fp0) - adiados} somem")


if __name__ == "__main__":
    main()
