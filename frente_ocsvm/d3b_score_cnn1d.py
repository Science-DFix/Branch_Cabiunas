#!/usr/bin/env python3
"""D3b -- o score contínuo do CNN-1D mensal separa trip de FP? Mesmo teste da D3 (d3_score_continuo.py), com CNN-1D.

PRÉ-REGISTRO (18/10/2026, antes de enviar as tasks)
Pergunta (do usuário: "faça para CNN1D"): a D3 deu nulo com a OCSVM (AUC 0,40-0,47). O CNN-1D olha janelas de 30 min
(forma temporal, não ponto isolado) e é retreinado todo mês; o score dele separa trip de FP?
Dados: braço cnn1d_mensal_jan_sc (d1_disputa.py) = config idêntica ao cnn1d_mensal_jan da rodada 1d, rodado de novo
no commit que passa a salvar, no walk-forward, r = score / limiar do próprio mês (sequence_scores_all.csv, limiar 1).
Canais binários e r vêm das mesmas 3 tasks novas (temp, vib, óleo); canal 4 e regras de produção.
Episódios: recorte de 01/01/2025 ao fim (o braço só emite flag desde 01/2025). Na 1d foram 8 detecções e 45 FP; a
nova execução deve repetir isso dentro do ruído do TF (se não repetir, relatar e seguir com o que der).
Medidas F1 (magnitude), F2 (persistência em 6 h) e F3 (mudança de carga em T5_AVG_A), Etapa 1 (Mann-Whitney
unilateral, Bonferroni 0,05/3, mesma direção da D3) e Etapa 2 (portão por trecho de voto com corte por LOEO) são
EXATAMENTE as da D3, importadas de d3_score_continuo.py.
PASSA = LOEO 8/8 e, com o corte escolhido nos 8, 3/3 fora com FP/mês fora <= 2,158 (0,75 x OCSVM de produção).
Expectativa: nulo na Etapa 1, como na OCSVM (AUC entre 0,4 e 0,6): a D2 mostrou que CNN-1D e OCSVM acendem nos
mesmos episódios, e a D3 que os FP têm score tão alto e persistente quanto os trips -> NÃO.

RESULTADO (19/10/2026, depois do pré-registro 77f09c4): NÃO SEPARA -> NÃO (Etapa 2 não roda).
Tasks: temp ffe9825c, óleo de4e9963 (em paralelo com a temp, a pedido do usuário), vib dc73ba24 (sozinha, por
último). Reprodução da 1d exata: 8 detecções, 45 FP, 23 inconclusivos desde 01/2025, mesmas antecedências.
  F   med det  med FP  AUC    p      AUC fora (3 trips, descritivo)
  F1  2,63     3,40    0,341  0,92   0,253
  F2  0,251    0,632   0,391  0,84   0,556
  F3  4,21     11,86   0,395  0,18   0,375
Nas três, os FP ficam do lado "trip" mais que as detecções (AUC < 0,5). Expectativa certa (nulo).

Uso: python frente_ocsvm/d3b_score_cnn1d.py
"""
from __future__ import annotations
import json

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import comum as C
from a4_variabilidade import canais_da_replica
from d1_disputa import CANAIS, TASKS_JSON
from d3_score_continuo import ALFA, CSV, DIRECAO, decide_com_portao, medidas, trechos
from src.cnn1d_ae.scoring import apply_min_duration_filter, apply_refractory, combine_channels_vote

INICIO = pd.Timestamp("2025-01-01")
BRACO = "cnn1d_mensal_jan_sc"


def razoes(ids, idx):
    from clearml import Task
    r = {}
    for g, k in zip(("temp", "vib", "oleo"), C.CANAIS[:3]):
        p = Task.get_task(task_id=ids[g]).artifacts[f"{CANAIS[g][2]}/csv/sequence_scores_all.csv"].get_local_copy()
        s = pd.read_csv(p, index_col=0, parse_dates=True)
        r[k] = (s["score"] / s["threshold"]).reindex(idx)
    return pd.DataFrame(r)


def main():
    reg = json.loads(TASKS_JSON.read_text())
    ids = {g: reg[f"{BRACO}_{g}"] for g in ("temp", "vib", "oleo")}
    df = canais_da_replica(ids)
    idx, op = df.index, df["operational_state"]
    on = op.eq("on")
    cat = C.catalogo()
    canais = {k: df[k] for k in C.CANAIS[:3]}
    canais["c4"] = C.canal_alarme(idx, cat.loc[cat.tag.isin(C.TAGS_C4), "t"], C.REF["janela_c4_h"])
    ft = C.trips(idx)
    R = razoes(ids, idx)
    carga = pd.to_numeric(pd.read_csv(CSV, usecols=["data_datetime", "T5_AVG_A"], parse_dates=["data_datetime"],
                                      low_memory=False).set_index("data_datetime")["T5_AVG_A"], errors="coerce").reindex(idx)

    voto = combine_channels_vote(canais, min_votes=2)
    dur = apply_min_duration_filter(pd.DataFrame({"is_anom_point": voto.astype(int)}), C.REF["dur_min"])["is_anom_point"].astype(bool)
    cls, _ = C.avalia(apply_refractory(dur, refractory_minutes=C.REF["refrat_h"] * 60.0), op, ft)
    cls = cls[cls.start >= INICIO]
    print(f"episódios desde 01/2025: {cls.classe.value_counts().to_dict()} (1d: 8 detecções, 45 FP)")

    # ETAPA 1
    E = pd.DataFrame([dict(start=e.start, classe=e.classe, **medidas(e.start, canais, R, carga, on)) for _, e in cls.iterrows()])
    E["fora"] = E.start >= C.SPLIT
    pd.set_option("display.width", 200)
    print("\nETAPA 1 -- detecções x FP (01/2025 ao fim; fora = depois de 01/07/2025, descritivo)")
    L, passam = [], []
    for f, lado in DIRECAO.items():
        alt = "greater" if lado == "maior" else "less"
        d, fp = E.loc[E.classe == "deteccao", f].dropna(), E.loc[E.classe == "falso_positivo", f].dropna()
        u, p = mannwhitneyu(d, fp, alternative=alt)
        df_ = E.loc[(E.classe == "deteccao") & E.fora, f].dropna()
        ff = E.loc[(E.classe == "falso_positivo") & E.fora, f].dropna()
        auc_f = mannwhitneyu(df_, ff, alternative=alt)[0] / (len(df_) * len(ff)) if len(df_) and len(ff) else np.nan
        L.append(dict(F=f, lado_trip=lado, n_det=len(d), n_fp=len(fp), med_det=round(d.median(), 3),
                      med_fp=round(fp.median(), 3), AUC=round(u / (len(d) * len(fp)), 3), p=round(p, 4),
                      separa=p < ALFA, AUC_fora=round(auc_f, 3)))
        if p < ALFA:
            passam.append(f)
    print(pd.DataFrame(L).to_string(index=False))
    print("\nPor detecção:")
    print(E[E.classe == "deteccao"].round(3).to_string(index=False))

    if not passam:
        print("\nNenhuma medida separa (Bonferroni) -> Etapa 2 não roda. Veredito: NÃO.")
        return

    # ETAPA 2 (igual à D3, com FP contados desde 01/2025)
    pos = trechos(dur)
    T = pd.DataFrame([dict(pos=(a, b), **medidas(idx[a], canais, R, carga, on)) for a, b in pos])
    meses_fora = C.compute_operational_period_days(pd.DataFrame({"operational_state": op[idx >= C.SPLIT]})) / 30.4375

    def roda(f, c):
        cl, _ = C.avalia(decide_com_portao(dur, T, f, c, DIRECAO[f]), op, ft)
        det = set(pd.to_datetime(cl.loc[cl.classe == "deteccao", "falha_associada"]))
        fp = int(((cl.classe == "falso_positivo") & (cl.start >= INICIO)).sum())
        fp_fora = int(((cl.classe == "falso_positivo") & (cl.start >= C.SPLIT)).sum())
        return det, fp, fp_fora

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
        print(f"\n{f}: LOEO {loeo}/8; corte com os 8 = {c8:.3f}: {len(det)}/8, FP {fp} desde 01/2025, fora {fora}/3 "
              f"com {fpf} FP ({fpf / meses_fora:.3f}/mês) -> {'PASSA' if passa else 'NÃO'}")


if __name__ == "__main__":
    main()
