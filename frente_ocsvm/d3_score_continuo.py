#!/usr/bin/env python3
"""D3 -- o score contínuo separa trip de FP? (magnitude, persistência e carga no momento do alerta). Local, sem treino.

PRÉ-REGISTRO (18/10/2026, antes de rodar)
Pergunta (do usuário, item 2 do relatório relatorio_ocsvm_cnn1d_mensal): alarmes e votos não separam trip de FP
(60 de 62 episódios nascem de PI_6240319_AL/PAL_6240315 + 1 modelo). Informação que NÃO é "qual alarme tocou" separa?
Modelo: OCSVM de produção (b494d457, 9c344688, fc4123fb), a única família com score contínuo salvo
(sequence_scores_all.csv). O CNN-1D mensal não salva score no walk-forward (automl_pipeline.py:785).

Episódios: os da camada de decisão de produção (canal 4 = 5 tags/24 h, voto >= 2, 45 min, refratário 48 h), período
inteiro (07/2024 a 04/2026): 8 detecções e 42 FP (A0); inconclusivos ficam fora da comparação.
Momento do alerta: o filtro de 45 min marca o trecho desde o início do voto, então o alerta real sai em t0 + 45 min
(t0 = início do episódio). Todas as medidas usam só dados até t0 + 45 min, com a máquina ligada.
r = score / limiar do grupo (r > 1 = acima do limiar). Canal "aceso" = modelo com flag em [t0, t0 + 45 min).
  F1 magnitude:     max sobre os canais acesos da mediana de r em [t0, t0 + 45 min).
  F2 persistência:  max sobre os canais acesos da fração do tempo com r > 1 em [t0 - 6 h, t0) (vem crescendo há
                    horas x pico isolado).
  F3 carga:         |mediana de T5_AVG_A em [t0, t0 + 45 min) - mediana em [t0 - 24 h, t0)| (mudança de carga; T5_AVG_A
                    é o sensor do portão de carga da produção).
ETAPA 1 (primária): Mann-Whitney unilateral, detecções x FP. Direção fixada: trip tem F1 maior, F2 maior e F3 MENOR.
  Separa = p < 0,05/3 (Bonferroni). Também relatar AUC e o resultado só fora da amostra (descritivo, 3 trips).
ETAPA 2 (só para F que separar): portão no início de cada trecho de voto (depois dos 45 min, antes do refratário):
  o trecho só alerta se F está do lado "trip" do corte c. c vem de uma grade (quantis de F nos trechos) por LOEO: sem o
  trip i, escolhe o c que mais corta FP mantendo os outros 7; conta se o trip i ainda é detectado.
  PASSA = LOEO 8/8 e, com c escolhido nos 8, 3/3 fora com FP/mês fora <= 0,75 x 2,877 = 2,158.
Expectativa: nenhuma separa na Etapa 1 (F1 e F3 nulos; F2 com AUC até ~0,7, sem passar Bonferroni), porque os 5 trips
de dentro estão no treino da OCSVM (o B3 mostrou que ali os modelos acendem menos). Etapa 2 não roda -> NÃO.

RESULTADO (18/10/2026, depois do pré-registro eef43f6): NÃO SEPARA -> NÃO (Etapa 2 não roda).
  F   med det  med FP  AUC    p      AUC fora (3 trips x 23 FP, descritivo)
  F1  26,1     43,9    0,413  0,78   0,561
  F2  0,209    0,406   0,396  0,83   0,608
  F3  5,09     11,44   0,469  0,40   0,397
Nas três, os FP ficam do lado "trip" tanto ou mais que as detecções. Expectativa certa (nulo); errei o F2 (previ AUC
~0,7; deu 0,40 no total). A posteriori: F2 é baixo nos 5 trips de dentro (0,03-0,31) e alto nos 3 de fora (0,5-1,0),
o mesmo padrão do B3 (trecho pré-trip de dentro entrou no treino como normal); fora, porém, os FP também têm F2 alto
(AUC 0,61). Correção depois da 1ª execução: a coluna AUC fora saía NaN (faltava dropna); não muda critério nem veredito.

Uso: python frente_ocsvm/d3_score_continuo.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import comum as C
from a4_variabilidade import canais_da_replica
from d1_disputa import CANAIS, PROD
from src.cnn1d_ae.scoring import apply_min_duration_filter, apply_refractory, combine_channels_vote

CSV = (C.DADOS / "alarmes_selecionados_turbina_a.csv").resolve().parent / "sensores_full_2024_2026_30s.csv"
VOTO, ANTES, CARGA = pd.Timedelta(minutes=45), pd.Timedelta(hours=6), pd.Timedelta(hours=24)
DIRECAO = {"F1": "maior", "F2": "maior", "F3": "menor"}  # lado "trip"
ALFA = 0.05 / 3


def razoes(idx):
    from clearml import Task
    r = {}
    for g, k in zip(("temp", "vib", "oleo"), C.CANAIS[:3]):
        _, tid, art = CANAIS[g]
        p = Task.get_task(task_id=tid).artifacts[f"{art}/csv/sequence_scores_all.csv"].get_local_copy()
        s = pd.read_csv(p, index_col=0, parse_dates=True)
        r[k] = (s["score"] / s["threshold"]).reindex(idx)
    return pd.DataFrame(r)


def medidas(t0, canais, R, carga, on):
    w = slice(t0, t0 + VOTO - pd.Timedelta(seconds=1))
    acesos = [k for k in C.CANAIS[:3] if canais[k].loc[w].any()]
    f1 = max((R[k].loc[w][on.loc[w]].median() for k in acesos), default=np.nan)
    a = slice(t0 - ANTES, t0 - pd.Timedelta(seconds=1))
    f2 = max(((R[k].loc[a][on.loc[a]] > 1).mean() for k in acesos), default=np.nan)
    b = slice(t0 - CARGA, t0 - pd.Timedelta(seconds=1))
    f3 = abs(carga.loc[w][on.loc[w]].median() - carga.loc[b][on.loc[b]].median())
    return {"F1": f1, "F2": f2, "F3": f3}


def trechos(dur: pd.Series):
    m = dur.values.astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def decide_com_portao(dur, ini_feat, f, c, lado):
    """Zera os trechos de voto (pós-45 min) cujo F no início está do lado FP do corte, depois aplica o refratário."""
    x = dur.values.copy()
    for (a, b), v in zip(ini_feat["pos"], ini_feat[f]):
        ok = (v >= c) if lado == "maior" else (v <= c)
        if not ok or not np.isfinite(v):
            x[a:b] = False
    return apply_refractory(pd.Series(x, index=dur.index), refractory_minutes=C.REF["refrat_h"] * 60.0)


def main():
    df = canais_da_replica(PROD)
    idx, op = df.index, df["operational_state"]
    on = op.eq("on")
    cat = C.catalogo()
    canais = {k: df[k] for k in C.CANAIS[:3]}
    canais["c4"] = C.canal_alarme(idx, cat.loc[cat.tag.isin(C.TAGS_C4), "t"], C.REF["janela_c4_h"])
    ft = C.trips(idx)
    R = razoes(idx)
    carga = pd.to_numeric(pd.read_csv(CSV, usecols=["data_datetime", "T5_AVG_A"], parse_dates=["data_datetime"])
                          .set_index("data_datetime")["T5_AVG_A"], errors="coerce").reindex(idx)

    voto = combine_channels_vote(canais, min_votes=2)
    dur = apply_min_duration_filter(pd.DataFrame({"is_anom_point": voto.astype(int)}), C.REF["dur_min"])["is_anom_point"].astype(bool)
    cls, met = C.avalia(apply_refractory(dur, refractory_minutes=C.REF["refrat_h"] * 60.0), op, ft)
    print(f"referência: {met.get('n_detected', '?')} detecções; classes {cls.classe.value_counts().to_dict()}")

    # ETAPA 1
    E = pd.DataFrame([dict(start=e.start, classe=e.classe, **medidas(e.start, canais, R, carga, on)) for _, e in cls.iterrows()])
    E["fora"] = E.start >= C.SPLIT
    pd.set_option("display.width", 200)
    print("\nETAPA 1 -- detecções x FP (período inteiro; fora = depois de 01/07/2025, descritivo)")
    L, passam = [], []
    for f, lado in DIRECAO.items():
        d, fp = E.loc[E.classe == "deteccao", f].dropna(), E.loc[E.classe == "falso_positivo", f].dropna()
        alt = "greater" if lado == "maior" else "less"
        u, p = mannwhitneyu(d, fp, alternative=alt)
        auc = u / (len(d) * len(fp))
        df_ = E.loc[(E.classe == "deteccao") & E.fora, f].dropna()
        ff = E.loc[(E.classe == "falso_positivo") & E.fora, f].dropna()
        auc_f = mannwhitneyu(df_, ff, alternative=alt)[0] / (len(df_) * len(ff)) if len(df_) and len(ff) else np.nan
        L.append(dict(F=f, lado_trip=lado, n_det=len(d), n_fp=len(fp), med_det=round(d.median(), 3),
                      med_fp=round(fp.median(), 3), AUC=round(auc, 3), p=round(p, 4), separa=p < ALFA,
                      AUC_fora=round(auc_f, 3)))
        if p < ALFA:
            passam.append(f)
    print(pd.DataFrame(L).to_string(index=False))
    print("\nPor detecção:")
    print(E[E.classe == "deteccao"].round(3).to_string(index=False))

    if not passam:
        print("\nNenhuma medida separa (Bonferroni) -> Etapa 2 não roda. Veredito: NÃO.")
        return

    # ETAPA 2
    pos = trechos(dur)
    T = pd.DataFrame([dict(pos=(a, b), **medidas(idx[a], canais, R, carga, on)) for a, b in pos])
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
            outros = set(ft) - {t}
            ok = [c for c in grade if outros <= res[c][0]]
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
        print(f"\n{f}: LOEO {loeo}/8; corte com os 8 = {c8:.3f}: {len(det)}/8, FP {fp}, fora {fora}/3 com {fpf} FP "
              f"({fpf / meses_fora:.3f}/mês) -> {'PASSA' if passa else 'NÃO'}")


if __name__ == "__main__":
    main()
