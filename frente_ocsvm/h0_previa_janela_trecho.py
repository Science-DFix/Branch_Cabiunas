#!/usr/bin/env python3
"""H0 -- prévia local do H1: os modelos de produção CONGELADOS com as features de janela por trecho. Sem retreino.

PRÉ-REGISTRO (24/10/2026, antes de rodar)
Pergunta (do usuário: "deixe guardado para o cica e vamos testar localmente"): antes de retreinar no Cica (H1), o
que acontece se as features de janela que não atravessa a parada (DERIVED_RESET_AT_STATE_CHANGE) entram nos 3 OCSVM
de produção como estão? Normalização, clipping (`f1_congelado.json`), limiar, debounce e portões ficam iguais, assim
como a camada de decisão com E5 e a régua. É o caminho do F1 (`f1_congelado.pontua_grupo`); só as janelas mudam.
Artefatos do cache local (Cica fora do ar), identificados como no G1. O estado operacional é o do F1. Os trechos são
os pedaços contínuos desse estado, e o script confere que o estado devolvido pelo F1 é o mesmo usado nos trechos.
Base: o mesmo caminho com o score de produção em cache (E6); tem de dar 8/8, 41 FP e 2,816 FP/mês, senão nada vale.

LIMITE DA PRÉVIA: o modelo foi treinado com as features antigas. As features novas só diferem nas primeiras 24 h de
cada trecho. Algumas ficam com valores que o treino quase nunca viu, como a tendência e a textura em 0 no começo do
trecho, e isso pode criar FP que o retreino não teria. Por isso a prévia não adota nada; ela indica para onde o H1
deve ir. O H1 no Cica (`h1_janela_trecho.py`) é o teste que decide.
LEITURA (critério da frente, só indicativo): PROMISSOR = 8/8 E 3/3 fora E FP/mês fora <= 2,158. Também se mede: FP a
<= 2 h da partida (base: 15); quais FP somem e quais aparecem; a antecedência de cada trip (T4, T6 e T7 começam na
partida); e o duty de cada canal OCSVM em operação.
EXPECTATIVA REGISTRADA: os FP de partida caem para 8 ou menos de 15. Aparecem FP novos nas primeiras horas, pela
tendência e textura em 0 fora do normal de treino, e por isso o total cai menos: 41 -> 33-38. O veredito indicativo é
NÃO PROMISSOR por FP, com 8/8 mantido.

Uso: python frente_ocsvm/h0_previa_janela_trecho.py   (~30-40 min, ~7 GB; score novo em dados/h0/)
"""
from __future__ import annotations
import json
from dataclasses import replace

import numpy as np
import pandas as pd

import comum as C
import f1_congelado as F
import g1_sinal_dominante as G
from e4_adia_partida import horas_desde_partida
from src.cnn1d_ae.automl_models import ocsvm_error
from src.cnn1d_ae.preprocess import build_group_dataframe, build_sensor_dataframe
from src.cnn1d_ae.scoring import build_operational_state

SAIDA = C.DADOS / "h0"
LIM_FORA = 0.75 * 2.877


def estado_f1(cfg, df_raw, cong) -> pd.Series:
    """O estado operacional que `f1_congelado.pontua_grupo` calcula (mesmos argumentos)."""
    s0 = cfg.SENSOR_GROUPS[0]["sensors"][0]
    idx = build_group_dataframe(replace(cfg, ENABLE_DERIVED_FEATURES=False), df_raw, df_raw, [s0])[0].index
    ref = build_sensor_dataframe(cfg, df_raw, df_raw, cfg.OPERATIONAL_REF_SENSOR)[0][cfg.OPERATIONAL_REF_SENSOR]
    assert cfg.OFF_TARGET_ABS_THRESHOLD is None
    return build_operational_state(
        index=idx, sensor_series=ref, off_value_quantile=cfg.OFF_VALUE_QUANTILE,
        off_abs_threshold=cfg.OFF_ABS_THRESHOLD, off_long_min_hours=cfg.OFF_LONG_MIN_HOURS,
        transient_padding_minutes=cfg.TRANSIENT_PADDING_MINUTES, transient_diff_quantile=cfg.TRANSIENT_DIFF_QUANTILE,
        transient_diff_abs_threshold=cong["diff_thr"])


def resumo(nome, pontos, cat, txt, ft):
    _, final, op, cls, met = F.decide(pontos, cat, txt, ft)
    ftp = ft[(ft >= op.index.min()) & (ft <= op.index.max())].reset_index(drop=True)
    P = C.por_trip(cls, ftp)
    meses_fora = C.compute_operational_period_days(pd.DataFrame({"operational_state": op[op.index >= C.SPLIT]})) / 30.4375
    fp = cls[cls.classe == "falso_positivo"].copy()
    hp = pd.Series(horas_desde_partida(op), index=op.index)
    fp["h_partida"] = hp.reindex(fp.start).values
    on = op.eq("on")
    lin = dict(braco=nome, det=met["falhas_detectadas"], fora=int((P.deteccao.notna() & (P.amostra == "fora")).sum()),
               FP=len(fp), fp_mes=round(met["falso_positivo_por_mes"], 3), FP_fora=int((fp.start >= C.SPLIT).sum()),
               fp_mes_fora=round(int((fp.start >= C.SPLIT).sum()) / meses_fora, 3),
               FP_partida=int((fp.h_partida <= 2).sum()), inc=met["n_episodios_inconclusivo"],
               duty=" ".join(f"{pontos[g].loc[on.index[on], 'is_anom_point'].mean():.3f}" for g in F.CANAIS))
    return lin, P, fp


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    cong = json.loads(F.CONGELADO.read_text())
    sens = str(G.SENS)
    cat = F.le_catalogo(G.CACHE_CLEARML / "datasets" / f"ds_{F.P.ALARM_CATALOG_DATASET_ID}" / F.P.ALARM_CATALOG_FILE)
    txt = F.le_texto(sens)
    ft = F.le_trips(C.DADOS / "alarmes_francisco_falhas.csv")
    base, nova = {}, {}
    for g in F.CANAIS:
        cfg = G.config(g)
        norm, hp, modelos = G.artefatos_cache(cfg, g)
        err0 = np.load(C.DADOS / "e6" / f"score_{g}.npy")
        cols = sorted(set(cfg.SENSOR_GROUPS[0]["sensors"]) | {cfg.OPERATIONAL_REF_SENSOR}
                      | {c for c in [cfg.LOAD_GATE_SENSOR, cfg.STEP_CHANGE_GATE_SENSOR] if c})
        df_raw = F.le_sensores(cfg, sens, cols, cfg.DATA_END_DATE)
        print(f"[{g}] base (score de produção em cache)...", flush=True)
        base[g], _ = F.pontua_grupo(cfg, df_raw, modelos[0], norm, hp, cong[g], err=err0)

        estado = estado_f1(cfg, df_raw, cong[g])
        assert estado.reindex(base[g].index).fillna("on").equals(base[g]["operational_state"]), f"[{g}] estado difere do F1"
        trechos = estado.ne(estado.shift()).cumsum()
        print(f"[{g}] {int(trechos.iloc[-1])} trechos; pontuando com janelas por trecho...", flush=True)
        orig = F.build_group_dataframe
        F.build_group_dataframe = lambda c, a, b, s: orig(c, a, b, s, segments=trechos)
        try:
            # o modelo certo entre os de mesma dimensão: o que reproduz o score de produção (como no G1)
            m = modelos[0] if len(modelos) == 1 else None
            arq = SAIDA / f"score_{g}.npy"
            err1 = np.load(arq) if arq.exists() else None
            if m is None:
                d0 = pd.concat([orig(cfg, df_raw, df_raw, [s])[0] for s in norm["feature_columns"]
                                if s in cfg.SENSOR_GROUPS[0]["sensors"]], axis=1)
                rng = np.random.default_rng(1)
                k = np.sort(rng.choice(len(d0), 3000, replace=False))
                X = d0[norm["feature_columns"]].iloc[k].to_numpy()
                lo, hi = (np.array([cong[g][q][c] for c in norm["feature_columns"]]) for q in ("clip_low", "clip_high"))
                mu = np.array([norm["normalize_center"][c] for c in norm["feature_columns"]])
                sd = np.array([norm["normalize_scale"][c] for c in norm["feature_columns"]])
                Z = ((np.clip(X, lo, hi).astype(np.float32).astype(np.float64) - mu) / sd).astype(np.float32)
                erros = [float(np.max(np.abs(ocsvm_error(mm, Z) - err0[k]))) for mm in modelos]
                m = modelos[int(np.argmin(erros))]
                assert min(erros) < 1e-3 * float(np.max(np.abs(err0[k]))), erros
                del d0
            nova[g], _ = F.pontua_grupo(cfg, df_raw, m, norm, hp, cong[g], err=err1)
        finally:
            F.build_group_dataframe = orig
        assert nova[g]["operational_state"].equals(base[g]["operational_state"]), g
        if err1 is None:
            np.save(arq, nova[g]["score"].values.astype(np.float32))
        mudou = (nova[g]["score"].values != err0)
        print(f"[{g}] instantes com score diferente: {mudou.mean():.3%}", flush=True)
        del df_raw

    L, Ps, FPs = [], {}, {}
    for nome, pts in (("base", base), ("janela_trecho", nova)):
        lin, Ps[nome], FPs[nome] = resumo(nome, pts, cat, txt, ft)
        L.append(lin)
    T = pd.DataFrame(L)
    pd.set_option("display.width", 220)
    print("\n" + T.to_string(index=False))
    b = T.set_index("braco").loc["base"]
    assert (b.det, b.FP) == (8, 41), "a base não reproduz a produção com E5"
    n = T.set_index("braco").loc["janela_trecho"]
    ok = n.det == 8 and n.fora == 3 and n.fp_mes_fora <= LIM_FORA
    print(f"\nindicativo: {n.det}/8, fora {n.fora}/3, FP/mês fora {n.fp_mes_fora} (limite {LIM_FORA:.3f}) -> "
          f"{'PROMISSOR' if ok else 'NÃO PROMISSOR'}")
    print("\nantecedência (h):")
    print(pd.DataFrame({"trip": Ps["base"].trip.dt.strftime("%d/%m/%y"), "base": Ps["base"].lead_h.values,
                        "janela_trecho": Ps["janela_trecho"].lead_h.values}).to_string(index=False))

    def casa(a, b):  # FP de `a` com algum FP de `b` começando a até 2 h
        return a.start.apply(lambda s: ((b.start - s).abs() <= pd.Timedelta(hours=2)).any())
    fb, fn = FPs["base"], FPs["janela_trecho"]
    fb["fica"], fn["ja_existia"] = casa(fb, fn).values, casa(fn, fb).values
    print(f"\nFP da base que somem: {int((~fb.fica).sum())} (de partida: {int((~fb.fica & (fb.h_partida <= 2)).sum())} de "
          f"{int((fb.h_partida <= 2).sum())}); FP novos: {int((~fn.ja_existia).sum())} "
          f"(a <= 2 h da partida: {int((~fn.ja_existia & (fn.h_partida <= 2)).sum())})")
    print("FP novos:\n" + fn.loc[~fn.ja_existia, ["start", "end", "h_partida"]].round(1).to_string(index=False))
    T.to_csv(SAIDA / "resumo.csv", index=False)
    fb.to_csv(SAIDA / "fp_base.csv", index=False)
    fn.to_csv(SAIDA / "fp_janela_trecho.csv", index=False)


if __name__ == "__main__":
    main()
