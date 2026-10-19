#!/usr/bin/env python3
"""E2 -- o alarme "novo" (canal 4 acendendo depois de silêncio) separa trip de FP? Local, sem treino.

PRÉ-REGISTRO (19/10/2026, antes de rodar)
Pergunta (do usuário: "vamos para item 2"): PI_6240319_AL e PAL_6240315 tocam centenas de vezes, como alarmes
crônicos (ISA-18.2: standing/chattering). Hipótese: antes de um trip o alarme aparece NOVO, depois de dias apagado; nos
FP ele já vinha tocando. O B1 testou outra coisa (debounce dentro do canal, renovação das 24 h); o D3 mediu a
persistência do score do modelo, não o histórico do alarme.
Modelo e episódios: OCSVM de produção (point_anomalies_final.csv, A0), camada de decisão de produção, período
inteiro (07/2024 a 04/2026): 8 detecções e 42 FP; inconclusivos ficam fora. Momento do alerta = t0 + 45 min; as
medidas usam só alarmes do catálogo (5 tags do canal 4) até esse instante.
  N1 idade do canal 4: horas que o canal 4 está aceso sem interrupção no momento do alerta (0 se apagado).
     "Novo" = pequeno.
  N2 recorrência: número de ativações das 5 tags nos 7 dias antes do momento do alerta.
Direção fixada: trip tem N1 MENOR e N2 MENOR.
ETAPA 1 (primária): Mann-Whitney unilateral, detecções x FP, separa = p < 0,05/2 (Bonferroni). Relatar AUC (P(trip do
  lado "trip")) e, descritivo, só fora da amostra (3 trips).
ETAPA 2 (só para N que separar): portão no início de cada trecho de voto, mesmo da D3 (decide_com_portao), corte por
  LOEO numa grade de quantis. PASSA = LOEO 8/8 e, com o corte escolhido nos 8, 3/3 fora com FP/mês fora <= 2,158.
Expectativa: NULO. O E1 já mostrou (descritivo, desde 01/2025) que as detecções têm MAIS ativações recentes que os FP
(mediana de 9,5 contra 6,5 alarmes da tag do canal 4 em 7 dias), e 4 dos 8 trips caem em alarme crônico. Espero AUC
de N2 <= 0,5 e N1 entre 0,4 e 0,6 -> Etapa 2 não roda -> NÃO.

RESULTADO (19/10/2026, depois do pré-registro 208f86d): NÃO SEPARA -> NÃO (Etapa 2 não roda). Referência = A0 (8/42/25).
  N   med det  med FP  AUC    p      AUC fora (3 trips, descritivo)
  N1  3,75 h   14,6 h  0,546  0,35   0,493
  N2  9,5      7,5     0,341  0,92   0,543
N1 vai na direção da hipótese (o canal 4 dos trips está aceso há menos tempo), mas fraco e sem fora; 2 trips (16/03 e
27/04/2025) e 1 de fora (03/11/2025) têm canal 4 aceso há 26-74 h. N2 vai contra: trips caem em semanas de MAIS alarme.
Expectativa certa (nulo).

Uso: python frente_ocsvm/e2_novidade_alarme.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import comum as C
from d3_score_continuo import VOTO, decide_com_portao, trechos
from src.cnn1d_ae.scoring import apply_min_duration_filter, apply_refractory, combine_channels_vote

DIRECAO = {"N1": "menor", "N2": "menor"}
ALFA = 0.05 / 2


def medidas(t0, c4, idade, tempos):
    ta = t0 + VOTO
    i = c4.index.searchsorted(ta, side="right") - 1
    n1 = float(idade[i]) if c4.iloc[i] else 0.0
    n2 = int(((tempos <= ta) & (tempos > ta - pd.Timedelta(days=7))).sum())
    return {"N1": n1, "N2": n2}


def idade_acesa(c4: pd.Series) -> np.ndarray:
    """Horas desde que o canal 4 acendeu, ponto a ponto (0 onde apagado)."""
    x = c4.values.astype(bool)
    t = c4.index.values.astype("datetime64[s]").astype(np.int64)
    ini = np.where(x & ~np.r_[False, x[:-1]], t, 0)
    ini = np.maximum.accumulate(ini)
    return np.where(x, (t - ini) / 3600.0, 0.0)


def main():
    df = C.carrega_canais()
    idx, op = df.index, df["operational_state"]
    cat = C.catalogo()
    tempos = cat.loc[cat.tag.isin(C.TAGS_C4), "t"].sort_values().reset_index(drop=True)
    canais = {k: df[k].astype(bool) for k in C.CANAIS[:3]}
    canais["c4"] = C.canal_alarme(idx, tempos, C.REF["janela_c4_h"])
    idade = idade_acesa(canais["c4"])
    ft = C.trips(idx)

    voto = combine_channels_vote(canais, min_votes=2)
    dur = apply_min_duration_filter(pd.DataFrame({"is_anom_point": voto.astype(int)}), C.REF["dur_min"])["is_anom_point"].astype(bool)
    cls, _ = C.avalia(apply_refractory(dur, refractory_minutes=C.REF["refrat_h"] * 60.0), op, ft)
    print(f"referência: classes {cls.classe.value_counts().to_dict()} (A0: 8 detecções, 42 FP, 25 inconclusivos)")

    # ETAPA 1
    E = pd.DataFrame([dict(start=e.start, classe=e.classe, **medidas(e.start, canais["c4"], idade, tempos))
                      for _, e in cls.iterrows()])
    E["fora"] = E.start >= C.SPLIT
    pd.set_option("display.width", 200)
    print("\nETAPA 1 -- detecções x FP (período inteiro; fora = depois de 01/07/2025, descritivo)")
    L, passam = [], []
    for f, lado in DIRECAO.items():
        alt = "greater" if lado == "maior" else "less"
        d, fp = E.loc[E.classe == "deteccao", f], E.loc[E.classe == "falso_positivo", f]
        u, p = mannwhitneyu(d, fp, alternative=alt)
        df_, ff = E.loc[(E.classe == "deteccao") & E.fora, f], E.loc[(E.classe == "falso_positivo") & E.fora, f]
        auc_f = mannwhitneyu(df_, ff, alternative=alt)[0] / (len(df_) * len(ff))
        # AUC = P(detecção do lado "trip" de um FP): com "menor", 1 - U/(n m)
        auc, auc_f = (u / (len(d) * len(fp)), auc_f) if lado == "maior" else (1 - u / (len(d) * len(fp)), 1 - auc_f)
        L.append(dict(N=f, lado_trip=lado, n_det=len(d), n_fp=len(fp), med_det=round(d.median(), 2),
                      med_fp=round(fp.median(), 2), AUC=round(auc, 3), p=round(p, 4), separa=p < ALFA,
                      AUC_fora=round(auc_f, 3)))
        if p < ALFA:
            passam.append(f)
    print(pd.DataFrame(L).to_string(index=False))
    print("\nPor detecção:")
    print(E[E.classe == "deteccao"].round(2).to_string(index=False))

    if not passam:
        print("\nNenhuma medida separa (Bonferroni) -> Etapa 2 não roda. Veredito: NÃO.")
        return

    # ETAPA 2 (igual à D3)
    T = pd.DataFrame([dict(pos=(a, b), **medidas(idx[a], canais["c4"], idade, tempos)) for a, b in trechos(dur)])
    meses_fora = C.compute_operational_period_days(pd.DataFrame({"operational_state": op[idx >= C.SPLIT]})) / 30.4375

    def roda(f, c):
        cl, _ = C.avalia(decide_com_portao(dur, T, f, c, DIRECAO[f]), op, ft)
        det = set(pd.to_datetime(cl.loc[cl.classe == "deteccao", "falha_associada"]))
        fp_fora = int(((cl.classe == "falso_positivo") & (cl.start >= C.SPLIT)).sum())
        return det, int((cl.classe == "falso_positivo").sum()), fp_fora

    for f in passam:
        grade = np.unique(np.nanquantile(T[f], np.linspace(0, 1, 21)))
        res = {c: roda(f, c) for c in grade}
        loeo = 0
        for t in ft:
            ok = [c for c in grade if (set(ft) - {t}) <= res[c][0]]
            c_i = min(ok, key=lambda c: res[c][1]) if ok else None
            loeo += c_i is not None and t in res[c_i][0]
        ok8 = [c for c in grade if set(ft) <= res[c][0]]
        c8 = min(ok8, key=lambda c: res[c][1]) if ok8 else None
        if c8 is None:
            print(f"\n{f}: nenhum corte mantém os 8 trips; LOEO {loeo}/8 -> NÃO")
            continue
        det, fp, fpf = res[c8]
        fora = sum(t >= C.SPLIT and t in det for t in ft)
        passa = loeo == 8 and fora == 3 and fpf / meses_fora <= 0.75 * 2.877
        print(f"\n{f}: LOEO {loeo}/8; corte com os 8 = {c8:.2f}: {len(det)}/8, FP {fp}, fora {fora}/3 com {fpf} FP "
              f"({fpf / meses_fora:.3f}/mês) -> {'PASSA' if passa else 'NÃO'}")


if __name__ == "__main__":
    main()
