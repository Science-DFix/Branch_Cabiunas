#!/usr/bin/env python3
"""E6 -- limiar próprio para o regime pós-partida, nos modelos de produção congelados. Local, sem retreino.

PRÉ-REGISTRO (19/10/2026, antes de rodar)
Pergunta (do usuário, item 4 do plano: "regime pós-partida no modelo"): o E4 mostrou que os FP de partida são um
regime de horas, não um pico. A ideia era pôr esse regime no normal de treino, mas a medida feita antes deste
pré-registro mostra que ele JÁ ESTÁ lá: das horas 1-3 depois de a máquina voltar a "on", 67% entram no normal de
treino de temperatura/vibração e 100% no de óleo, a mesma proporção do resto da operação (só os 60 min do
transiente ficam fora). Mas é pouco: as primeiras 6 h são ~6% do treino, e a OCSVM (nu = 0,05) põe a parte mais
rara da operação fora da fronteira. Retreinar com esse trecho "incluído" não muda nada. O que muda é dar ao regime
o próprio limiar.
Também medido antes: PI_6240319_AL a <= 3 h de 52% das 83 partidas (base em operação: 9,6% de janelas de 6 h);
PAL_6240315 em 66% (base 6,2%). O canal 4 acende na partida, e falta um modelo para fechar o voto.

TRATAMENTO (por canal OCSVM, mesmo modelo, normalização, debounce e portões da produção, via f1_congelado.py):
  limiar(t) = L_pos  se t está nas primeiras H horas de "on" depois de uma partida (horas_desde_partida do E4,
              calculada no estado operacional do próprio canal),
            = limiar de produção  no resto.
  L_pos = o MESMO percentil da produção (threshold_percentile do best_hyperparameters) sobre o score dos pontos
  do normal de treino (antes de 01/07/2025) que estão nas primeiras H horas. É o limiar que o AutoML daria se o
  regime fosse um grupo à parte: a mesma taxa de alerta no treino que o resto da operação. Nada é escolhido
  olhando trips.
  H = 6 h (fixado: o E4 viu FP de partida reaparecerem até ~8 h depois; 6 h cobre a maior parte e é o maior h do
  E4). H = 3 e 12 h só como sensibilidade; não entram na decisão.
  Conferência da ferramenta: o mesmo percentil sobre todo o normal de treino tem de dar o limiar da produção.
Base: pipeline final com E5 (8/8, 41 FP, 2,816 FP/mês); período inteiro e fora da amostra (>= 01/07/2025).
CRITÉRIO (o da frente): PASSA = 8/8 E 3/3 fora E FP/mês fora <= 0,75 x 2,877 = 2,158. Sem parâmetro ajustado nos
trips, não há LOEO. Secundários: FP/mês no período inteiro; antecedência por trip; quantos FP de partida somem.
EXPECTATIVA REGISTRADA: L_pos > limiar de produção nos 3 canais (o regime é mais raro, o score é mais alto). Corta
20-35% dos FP no período inteiro. Arrisca T4, T6 e T7, que começam a 1,1 / 0,6 / 0,0 h da partida: prevejo perder
1 ou 2 deles -> NÃO, por perda de trip. Se não perder nenhum, pode passar.

Uso: python frente_ocsvm/e6_limiar_pos_partida.py   (~30 min na 1ª vez; o score fica em cache em dados/e6/)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from clearml import Dataset, Task

import comum as C
import f1_congelado as F
from e4_adia_partida import horas_desde_partida
from src.cnn1d_ae.preprocess import _long_gap_mask

H_PRINCIPAL, H_SENS = 6.0, [3.0, 12.0]
LIM_FORA = 0.75 * 2.877
CACHE = C.DADOS / "e6"


def perto(idx: pd.DatetimeIndex, tempos: pd.Series, minutos: float) -> np.ndarray:
    """build_exclusion_mask (|t - alarme| <= minutos), vetorizado."""
    t = np.sort(pd.DatetimeIndex(tempos.dropna()).values)
    x = idx.values
    if not len(t):
        return np.zeros(len(x), bool)
    p = np.searchsorted(t, x)
    d = np.minimum(np.abs(x - t[np.clip(p - 1, 0, len(t) - 1)]), np.abs(t[np.clip(p, 0, len(t) - 1)] - x))
    return d <= np.timedelta64(int(minutos * 60), "s")


def normal_de_treino(cfg, df_raw, state, catalogo) -> np.ndarray:
    """A máscara `~exclude` de run_automl_group, antes do corte do treino."""
    g = cfg.SENSOR_GROUPS[0]
    idx = state.index
    ev = catalogo.loc[catalogo["Tag"].isin(g.get("eval_sensors") or g["sensors"]), "Data da Ocorrencia"]
    excl = perto(idx, ev, cfg.EXCLUDE_MINUTES_AROUND_ALARM)
    if cfg.EXCLUDE_LONG_GAPS_FROM_TRAIN:
        for s in g["sensors"]:
            excl |= _long_gap_mask(pd.to_numeric(df_raw[s], errors="coerce"), cfg.INTERPOLATE_LIMIT).values
    excl |= (state != "on").values
    for a, b in cfg.TRAIN_EXCLUDE_WINDOWS or []:
        excl |= (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
    return ~excl & (idx < pd.Timestamp(cfg.AUTOML_OOS_SPLIT_DATE))


def main():
    root = Path(Dataset.get(dataset_id=F.P.ALARM_CATALOG_DATASET_ID).get_local_copy())
    sens = str(root / F.P.SENSORS_FILE)
    catalogo = F.le_catalogo(root / F.P.ALARM_CATALOG_FILE)
    cong = json.loads(F.CONGELADO.read_text())
    CACHE.mkdir(parents=True, exist_ok=True)
    Hs = [0.0, H_PRINCIPAL] + H_SENS
    pontos = {h: {} for h in Hs}
    limiares = []
    for g, (tid, grupo) in F.CANAIS.items():
        task = Task.get_task(task_id=tid)
        cfg = F.config_da_task(task)
        modelo, norm, hp = F.artefatos(task, grupo)
        cols = sorted(set(cfg.SENSOR_GROUPS[0]["sensors"]) | {cfg.OPERATIONAL_REF_SENSOR}
                      | {c for c in [cfg.LOAD_GATE_SENSOR, cfg.STEP_CHANGE_GATE_SENSOR] if c})
        df_raw = F.le_sensores(cfg, sens, cols, cfg.DATA_END_DATE)
        arq = CACHE / f"score_{g}.npy"
        err = np.load(arq) if arq.exists() else None
        print(f"[{g}] {'score em cache' if err is not None else 'pontuando a OCSVM (lento)'}...", flush=True)
        p0, _ = F.pontua_grupo(cfg, df_raw, modelo, norm, hp, cong, err=err)
        if err is None:
            err = p0["score"].values.astype(np.float32)
            np.save(arq, err)
        state = p0["operational_state"]
        normal = normal_de_treino(cfg, df_raw.set_index(cfg.TIME_COL), state, catalogo)
        pct = hp["threshold_percentile"]
        conf = float(np.percentile(err[normal], pct))
        print(f"[{g}] conferência: percentil {pct} do normal de treino = {conf:.6g}; produção = {hp['threshold']:.6g}")
        hpart = horas_desde_partida(state)
        pontos[0.0][g] = p0
        for h in Hs[1:]:
            reg = hpart < h
            L = float(np.percentile(err[normal & reg], pct))
            limiares.append(dict(canal=g, H=h, L_prod=hp["threshold"], L_pos=L, razao=L / hp["threshold"],
                                 pts_treino=int((normal & reg).sum()), frac_treino=(normal & reg).sum() / normal.sum()))
            pontos[h][g], _ = F.pontua_grupo(cfg, df_raw, modelo, norm, hp, cong, err=err,
                                             limiar=np.where(reg, L, hp["threshold"]))
        del df_raw

    pd.set_option("display.width", 200)
    print("\nlimiares:\n" + pd.DataFrame(limiares).round(4).to_string(index=False))
    ft = F.le_trips(C.DADOS / "alarmes_francisco_falhas.csv")
    txt = F.le_texto(sens)
    R = {}
    for h in Hs:
        _, final, op, cls, met = F.decide(pontos[h], catalogo, txt, ft)
        ftp = ft[(ft >= op.index.min()) & (ft <= op.index.max())]
        meses_fora = C.compute_operational_period_days(pd.DataFrame({"operational_state": op[op.index >= C.SPLIT]})) / 30.4375
        Pt = C.por_trip(cls, ftp.reset_index(drop=True))
        fp_fora = int(((cls.classe == "falso_positivo") & (cls.start >= C.SPLIT)).sum())
        R[h] = dict(H=h, det=met["falhas_detectadas"], fora=int(sum(Pt.deteccao.notna() & (Pt.trip >= C.SPLIT))),
                    FP=met["n_episodios_falso_positivo"], fp_mes=met["falso_positivo_por_mes"], FP_fora=fp_fora,
                    fp_mes_fora=fp_fora / meses_fora, inc=met["n_episodios_inconclusivo"],
                    perdidos=",".join(f"T{i}" for i, ok in enumerate(Pt.deteccao.notna(), 1) if not ok) or "-",
                    P=Pt, cls=cls, op=op)
    print("\n" + pd.DataFrame([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                                 if k not in ("P", "cls", "op")} for r in R.values()]).to_string(index=False))
    r, b = R[H_PRINCIPAL], R[0.0]
    passa = r["det"] == 8 and r["fora"] == 3 and r["fp_mes_fora"] <= LIM_FORA
    print(f"\nH = {H_PRINCIPAL:g} h: {r['det']}/8, fora {r['fora']}/3, FP/mês {r['fp_mes']:.3f} "
          f"({100 * (r['fp_mes'] / b['fp_mes'] - 1):+.0f}%), fora {r['fp_mes_fora']:.3f} "
          f"({100 * (r['fp_mes_fora'] / b['fp_mes_fora'] - 1):+.0f}%; limite {LIM_FORA:.3f}) -> {'PASSA' if passa else 'NÃO'}")
    print("\nantecedência (h):")
    print(pd.DataFrame({"trip": b["P"].trip.dt.strftime("%d/%m/%y"),
                        **{f"H={h:g}": R[h]["P"].lead_h.values for h in Hs}}).to_string(index=False))
    hp0 = pd.Series(horas_desde_partida(b["op"]), index=b["op"].index)
    c0 = b["cls"]
    fp0 = c0[(c0.classe == "falso_positivo") & (hp0.reindex(c0.start).values <= 2)]
    cf = r["cls"]
    fica = sum(((cf.start >= s - pd.Timedelta(hours=1)) & (cf.start <= s + pd.Timedelta(hours=8))
                & (cf.classe == "falso_positivo")).any() for s in fp0.start)
    print(f"\nFP da base a <= 2 h da partida: {len(fp0)}; com H = {H_PRINCIPAL:g} h, {fica} continuam (até 8 h depois), "
          f"{len(fp0) - fica} somem")


if __name__ == "__main__":
    main()
