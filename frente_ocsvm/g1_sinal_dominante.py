#!/usr/bin/env python3
"""G1 -- qual sinal "manda" em cada grupo, nos FP de partida, nos trips e nas partidas normais. Local, sem retreino.

PRÉ-REGISTRO (24/10/2026, antes de rodar)
Pergunta (do usuário): "encontrar o grupo entre os sinais que podem ser mais sensíveis a estes FP perto do
religamento [...] cada sensor deve ter um sinal mais forte que pode ditar o grupo inteiro (minha hipótese) e a
partir dele calcular uma feature de acordo com sua origem". Este teste é só o diagnóstico: diz qual sensor e qual
família de feature acende cada canal, para depois desenhar (e pré-registrar) as features por origem.

BASE = a mesma do Cica: `sensores_full_2024_2026_30s.csv` do Dataset a97ba56b, configs de produção do commit
6842ac7, as mesmas features (`build_group_dataframe`, sensor a sensor como no F1), o clipping congelado do F1
(`f1_congelado.json`) e a normalização, o modelo e o limiar das tasks b494d457 / 9c344688 / fc4123fb. O Cica está
fora do ar (24/10): os artefatos vêm do cache local do ClearML, baixados em 19/10 pelo F1/E6, e são identificados
pelo limiar e pelas features. Conferência: o score recalculado tem de bater com `dados/e6/score_<canal>.npy` (a
produção, conferida no F1 com 0 divergências) em 3.000 instantes sorteados; senão o teste não vale.

CLASSES (pipeline final com E5: `runs_pipeline_unificada_final/`, 41 FP; horas desde a partida como no E4)
  TRIP  episódios de detecção (8 trips);
  FPP   FP que começam a <= 2 h da partida (15);
  FPO   os outros FP (26);
  PN    partidas normais: nenhum episódio começa nas 6 h depois da partida; janela [0, 6 h) em "on";
  NR    regime: "on", >= 24 h da partida, fora de episódio e longe de trip; 5.000 instantes sorteados (semente 0).
Por grupo, TRIP/FPP/FPO só contam os episódios em que esse canal acendeu, e as medidas de oclusão usam só os
instantes com o canal aceso (`canal_<grupo>` = 1).

MEDIDAS (por grupo)
  M1 oclusão por sensor. Nos instantes acesos, "sozinho" = todas as features dos OUTROS sensores no centro do
     treino (z = 0), e "sem" = as features DESTE sensor em z = 0. Mede a fração dos instantes que continuam acima
     do limiar de produção.
  M2 oclusão por família de feature (todos os sensores juntos), por escala (instantâneo = bruto + delta_1; 12, 120,
     480, 2880 linhas = 6 min, 1 h, 4 h, 24 h) e por tipo (nível = bruto + mediana; variabilidade = desvio;
     tendência = delta + trend; textura = curtose, assimetria e crista). "Sem" e "sozinho", como no M1.
  M3 feature a feature: mediana do z no episódio (todos os instantes em "on" da janela). AUC por episódio de TRIP
     contra FPP e de TRIP contra PN (direção livre, max(AUC, 1 - AUC), com o sinal reportado).
  M4 janelas que atravessam a parada. Nas partidas, a fração das últimas w linhas fora de "on" e a mediana de |z|
     de cada escala, por hora desde a partida (0,5 a 26 h), em PN e nas partidas com FPP.
CRITÉRIOS (descritivos, fixados aqui)
  - Sensor que "dita o grupo" numa classe: sozinho mantém >= 50% dos instantes acesos E, sem ele, <= 50% continuam.
    A hipótese do usuário tem apoio num grupo se isso acontece nos TRIP; e o sensor é o "culpado" do FP de partida
    se acontece nos FPP.
  - Feature candidata a "galinha dos ovos de ouro": AUC TRIP x FPP >= 0,75 E AUC TRIP x PN >= 0,75, na mesma
    direção. Com 8 trips, isso é hipótese a pré-registrar num teste com treino, nunca resultado.
  - Artefato das janelas longas: a 2 h da partida, > 50% da janela de 2880 (ou 480) fora de "on", |z| mediano
    dessa escala em PN >= 2x o do NR, e voltando ao nível do NR até 26 h.
  - Circularidade: as classes usam os trips e os FP; nada daqui é adotado sem teste novo (pré-registro + Cica).
EXPECTATIVA REGISTRADA
  - Vibração: nos FPP mandam as escalas longas (480 e 2880), pelo artefato da janela; nenhum sensor sozinho dita o
    grupo (os 5 mancais sobem juntos na partida).
  - Temperatura: T5_AVG_A (carga) dita os FPP; TC382_03_A dita os TRIP.
  - Óleo: PI_0308 dita o grupo nas duas classes.
  - Artefato das janelas longas confirmado ao menos na vibração.
  - Galinha dos ovos de ouro: no máximo 1 feature passa nos dois AUC, e só num grupo (o D3 não separou trip de FP
    com o score).

Uso: python frente_ocsvm/g1_sinal_dominante.py   (saída em frente_ocsvm/dados/g1/)
"""
from __future__ import annotations
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import comum as C
import f1_congelado as F
from e4_adia_partida import horas_desde_partida
from src.cnn1d_ae.config import PipelineConfig, update_cfg_from_dict
from src.cnn1d_ae.preprocess import build_group_dataframe
from src.cnn1d_ae.automl_models import ocsvm_error

RUN = F.RAIZ / "runs_pipeline_unificada_final"
CACHE_CLEARML = Path.home() / ".clearml" / "cache" / "storage_manager"
SENS = CACHE_CLEARML / "datasets" / f"ds_{F.P.ALARM_CATALOG_DATASET_ID}" / F.P.SENSORS_FILE
SAIDA = C.DADOS / "g1"
CONFIGS = {"temperatura": "test_grupo_exp33_mancal_temperatura_isolada.json",
           "vibracao": "test_grupo_exp34_mancal_vibracao_isolada.json",
           "oleo_pressao": "test_grupo_exp38_oleo_pressao_isolada.json"}
LIMIAR = {"temperatura": 10.504486083984375, "vibracao": 0.17573027312755585, "oleo_pressao": 29.674667358398438}
HORAS_M4 = [0.5, 1, 2, 3, 4, 6, 8, 12, 24, 25, 26]
ESCALAS = ["inst", "12", "120", "480", "2880"]
TIPOS = ["nivel", "variabilidade", "tendencia", "textura"]


def config(g: str) -> PipelineConfig:
    txt = subprocess.run(["git", "show", f"{F.COMMIT_PROD}:configs/calibracao_v4_eq/{CONFIGS[g]}"], cwd=F.RAIZ,
                         check=True, capture_output=True, text=True).stdout
    return update_cfg_from_dict(PipelineConfig(), json.loads(txt))


def artefatos_cache(cfg: PipelineConfig, g: str):
    """Normalização, hiperparâmetros e modelos candidatos do grupo, do cache local (Cica fora do ar)."""
    glob = CACHE_CLEARML / "global"
    sensores = set(cfg.SENSOR_GROUPS[0]["sensors"])
    norm = [n for n in (json.loads(f.read_text()) for f in glob.glob("*.normalization_stats.json"))
            if {c.split("__")[0] for c in n["feature_columns"]} == sensores]
    hp = [h for f in glob.glob("*.best_hyperparameters.json")
          if (h := json.loads(f.read_text())).get("threshold") == LIMIAR[g]]
    assert len({json.dumps(n, sort_keys=True) for n in norm}) == 1 and hp, (g, len(norm), len(hp))
    modelos = [m for f in glob.glob("*.model.pkl")
               if (m := pickle.loads(f.read_bytes())).n_features_in_ == len(norm[0]["feature_columns"])]
    return norm[0], hp[0], modelos


def familia(col: str) -> tuple[str, str]:
    if "__" not in col:
        return "inst", "nivel"
    k = col.split("__")[1]
    if k == "delta_1":
        return "inst", "tendencia"
    tipo, w = k.rsplit("_", 1)
    return w, {"roll_med": "nivel", "roll_std": "variabilidade", "trend": "tendencia"}.get(tipo, "textura")


def matriz_z(cfg, g, norm, cong, linhas: np.ndarray) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """z das features de produção (clipping congelado + normalização do treino) só nas `linhas` pedidas."""
    grupo = cfg.SENSOR_GROUPS[0]
    feats = norm["feature_columns"]
    cols = sorted(set(grupo["sensors"]) | {cfg.OPERATIONAL_REF_SENSOR})
    df_raw = F.le_sensores(cfg, str(SENS), cols, cfg.DATA_END_DATE)
    partes, idx = [], None
    for s in [s for s in grupo["sensors"] if s in feats]:
        d = build_group_dataframe(cfg, df_raw, df_raw, [s])[0]
        idx = d.index if idx is None else idx
        assert d.index.equals(idx)
        partes.append(d[[c for c in feats if c.split("__")[0] == s]].iloc[linhas].astype(np.float64))
        del d
    X = pd.concat(partes, axis=1)[feats]
    lo, hi = (np.array([cong[k][c] for c in feats]) for k in ("clip_low", "clip_high"))
    mu = np.array([norm["normalize_center"][c] for c in feats])
    sd = np.array([norm["normalize_scale"][c] for c in feats])
    xb = np.clip(X.to_numpy(), lo, hi).astype(np.float32).astype(np.float64)
    return pd.DataFrame(((xb - mu) / sd).astype(np.float32), index=X.index, columns=feats), idx


def acima(modelo, Z: np.ndarray, limiar: float, zera: np.ndarray | None = None) -> float:
    Z = Z.copy()
    if zera is not None:
        Z[:, zera] = 0.0
    return float((ocsvm_error(modelo, Z) > limiar).mean()) if len(Z) else np.nan


def classes(Pf: pd.DataFrame, e: pd.DataFrame, hp: np.ndarray, ft: pd.Series):
    """Janelas de cada classe: lista de (classe, rótulo, máscara de linhas de Pf)."""
    on = Pf.operational_state.eq("on").values
    t = Pf.index
    ev = []
    for r in e.itertuples():
        c = "TRIP" if r.classe == "deteccao" else "FPP" if r.classe == "falso_positivo" and r.h_desde_partida <= 2 \
            else "FPO" if r.classe == "falso_positivo" else None
        if c:
            rot = str(r.falha_associada)[:16] if c == "TRIP" else f"{r.start:%Y-%m-%d %H:%M}"
            ev.append((c, rot, (t >= r.start) & (t <= r.end) & on))
    inicio = np.flatnonzero(on & ~np.r_[False, on[:-1]])
    for i in inicio:
        s = t[i]
        if not ((e.start >= s) & (e.start < s + pd.Timedelta(hours=6))).any():
            ev.append(("PN", f"{s:%Y-%m-%d %H:%M}", (t >= s) & (t < s + pd.Timedelta(hours=6)) & on))
    longe = np.ones(len(t), bool)
    for r in e.itertuples():
        longe &= ~((t >= r.start - pd.Timedelta(hours=6)) & (t <= r.end + pd.Timedelta(hours=6)))
    for x in ft:
        longe &= ~((t >= x - pd.Timedelta(hours=72)) & (t <= x + pd.Timedelta(hours=24)))
    nr = np.flatnonzero(on & (hp >= 24) & longe)
    nr = np.sort(np.random.default_rng(0).choice(nr, 5000, replace=False))
    return ev, inicio, nr


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    Pf = pd.read_csv(RUN / "point_anomalies_final.csv", index_col=0, parse_dates=True, low_memory=False)
    e = pd.read_csv(RUN / "episodios_classificados.csv", parse_dates=["start", "end"])
    ft = C.trips(Pf.index)
    hp = horas_desde_partida(Pf.operational_state)
    ev, inicio, nr = classes(Pf, e, hp, ft)
    n_cls = pd.Series([c for c, _, _ in ev]).value_counts().to_dict()
    print(f"episódios por classe: {n_cls}; NR {len(nr)} instantes", flush=True)

    # M4: instantes a h horas de cada partida (PN e partidas com FPP)
    fpp_ini = set(e.start[(e.classe == "falso_positivo") & (e.h_desde_partida <= 2)])
    t_ini = Pf.index[inicio]
    pn_ini = {t_ini[k] for k, (i, s) in enumerate(zip(inicio, t_ini))
              if not ((e.start >= s) & (e.start < s + pd.Timedelta(hours=6))).any()}
    m4 = []
    for i, s in zip(inicio, t_ini):
        tipo = "PN" if s in pn_ini else "FPP" if any(s <= x <= s + pd.Timedelta(hours=2.5) for x in fpp_ini) else None
        if tipo:
            for h in HORAS_M4:
                j = i + int(round(h * 120))
                if j < len(Pf) and Pf.operational_state.iat[j] == "on" and abs(hp[j] - h) < 1e-6:
                    m4.append((tipo, h, j))
    m4 = pd.DataFrame(m4, columns=["tipo", "h", "linha"])
    fora_on = Pf.operational_state.ne("on").astype(float)
    janela_fora = pd.DataFrame({w: fora_on.rolling(int(w), min_periods=1).mean().values[m4.linha]
                                for w in ["480", "2880"]}).assign(tipo=m4.tipo.values, h=m4.h.values)

    rng = np.random.default_rng(1)
    conf_linhas = np.sort(rng.choice(len(Pf), 3000, replace=False))
    linhas_ev = np.unique(np.concatenate([np.flatnonzero(m) for _, _, m in ev]))
    linhas = np.unique(np.concatenate([linhas_ev, nr, m4.linha.values, conf_linhas]))
    cong = json.loads(F.CONGELADO.read_text())

    R_sens, R_fam, R_feat, R_m4 = [], [], [], []
    for g in F.CANAIS:
        cfg = config(g)
        norm, hpar, modelos = artefatos_cache(cfg, g)
        print(f"[{g}] montando features ({len(norm['feature_columns'])})...", flush=True)
        Z, idx = matriz_z(cfg, g, norm, cong[g], linhas)
        assert idx.equals(Pf.index), g
        # conferência contra o score de produção (E6), que também separa os modelos de mesma dimensão
        ref = np.load(C.DADOS / "e6" / f"score_{g}.npy")[conf_linhas]
        Zc = Z.loc[Pf.index[conf_linhas]].to_numpy()
        erros = [float(np.max(np.abs(ocsvm_error(m, Zc) - ref)) / np.max(np.abs(ref))) for m in modelos]
        k = int(np.argmin(erros))
        modelo = modelos[k]
        print(f"[{g}] conferência com o score do E6: erro relativo máximo {erros} -> modelo {k}", flush=True)
        assert erros[k] < 1e-4, f"[{g}] NÃO CONFERE"
        lim = hpar["threshold"]
        feats = norm["feature_columns"]
        sensores = [s for s in cfg.SENSOR_GROUPS[0]["sensors"] if s in feats]
        fam = pd.DataFrame([familia(c) for c in feats], columns=["escala", "tipo"], index=feats)
        col_s = {s: np.array([c.split("__")[0] == s for c in feats]) for s in sensores}
        canal = Pf[f"canal_{g}"].to_numpy().astype(bool)

        # M1 e M2 por classe (instantes acesos do canal nos episódios em que ele acendeu)
        for cls in ["TRIP", "FPP", "FPO", "PN"]:
            lin = np.unique(np.concatenate([np.flatnonzero(m & canal) for c, _, m in ev if c == cls] or [[]]).astype(int))
            Zl = Z.loc[Pf.index[lin]].to_numpy()
            n_ep = sum(1 for c, _, m in ev if c == cls and (m & canal).any())
            base = acima(modelo, Zl, lim)
            for s in sensores:
                R_sens.append(dict(grupo=g, classe=cls, sensor=s, n_ep=n_ep, n_pts=len(lin), base=base,
                                   sozinho=acima(modelo, Zl, lim, ~col_s[s]), sem=acima(modelo, Zl, lim, col_s[s])))
            for eixo, valores in (("escala", ESCALAS), ("tipo", TIPOS)):
                for v in valores:
                    sel = (fam[eixo] == v).to_numpy()
                    if sel.any():
                        R_fam.append(dict(grupo=g, classe=cls, eixo=eixo, familia=v, n_pts=len(lin), base=base,
                                          sozinho=acima(modelo, Zl, lim, ~sel), sem=acima(modelo, Zl, lim, sel)))
            print(f"[{g}] {cls}: {n_ep} episódios, {len(lin)} instantes acesos (acima do limiar: {base:.2f})", flush=True)

        # M3: mediana do z por episódio (todos os instantes em "on" da janela)
        med = {}
        for c, rot, m in ev:
            if c in ("TRIP", "FPP", "PN") and (c == "PN" or (m & canal).any()):
                med.setdefault(c, []).append(Z.loc[Pf.index[m]].median().to_numpy())
        nr_z = Z.loc[Pf.index[nr]]
        for j, f in enumerate(feats):
            linha = dict(grupo=g, feature=f, sensor=f.split("__")[0], escala=fam.escala[f], tipo=fam.tipo[f],
                         z_NR_p50=float(nr_z[f].median()), absz_NR_p95=float(nr_z[f].abs().quantile(0.95)))
            for c in med:
                linha[f"z_{c}"] = float(np.median([v[j] for v in med[c]]))
            for a in [a for a in ("FPP", "PN") if a in med and "TRIP" in med]:
                y = np.r_[np.ones(len(med["TRIP"])), np.zeros(len(med[a]))]
                x = np.r_[[v[j] for v in med["TRIP"]], [v[j] for v in med[a]]]
                auc = roc_auc_score(y, x)
                linha[f"auc_TRIP_{a}"] = max(auc, 1 - auc)
                linha[f"dir_TRIP_{a}"] = "+" if auc >= 0.5 else "-"
            R_feat.append(linha)

        # M4: |z| mediano por escala, por hora desde a partida
        az = Z.abs()
        nr_esc = {v: float(np.median(az.loc[Pf.index[nr], fam.index[fam.escala == v]].to_numpy())) for v in ESCALAS}
        for (tipo, h), d in m4.groupby(["tipo", "h"]):
            zz = az.loc[Pf.index[d.linha]]
            R_m4.append(dict(grupo=g, tipo=tipo, h=h, n=len(d),
                             **{f"absz_{v}": float(np.median(zz[fam.index[fam.escala == v]].to_numpy())) for v in ESCALAS},
                             **{f"NR_{v}": nr_esc[v] for v in ESCALAS}))
        del Z, az

    S, Fm, Ft, M4 = (pd.DataFrame(x) for x in (R_sens, R_fam, R_feat, R_m4))
    for nome, df in (("m1_sensor", S), ("m2_familia", Fm), ("m3_feature", Ft), ("m4_partida", M4)):
        df.to_csv(SAIDA / f"{nome}.csv", index=False)
    jf = janela_fora.groupby(["tipo", "h"]).median().reset_index()
    jf.to_csv(SAIDA / "m4_janela_fora_on.csv", index=False)

    pd.set_option("display.width", 220, "display.max_rows", 400)
    print("\nM1 -- fração dos instantes acesos que continua acima do limiar")
    S["dita"] = (S.sozinho >= 0.5) & (S.sem <= 0.5)
    print(S.round(2).to_string(index=False))
    print("\nM2 -- famílias")
    print(Fm.round(2).to_string(index=False))
    print("\nM3 -- features com AUC TRIP x FPP e TRIP x PN >= 0,75 na mesma direção")
    ouro = Ft[(Ft.auc_TRIP_FPP >= 0.75) & (Ft.auc_TRIP_PN >= 0.75) & (Ft.dir_TRIP_FPP == Ft.dir_TRIP_PN)]
    print(ouro.round(2).to_string(index=False) if len(ouro) else "nenhuma")
    print("\nM3 -- top 5 por grupo (média dos dois AUC)")
    Ft["auc_media"] = (Ft.auc_TRIP_FPP + Ft.auc_TRIP_PN) / 2
    print(Ft.sort_values("auc_media", ascending=False).groupby("grupo").head(5).round(2).to_string(index=False))
    print("\nM4 -- fração da janela fora de 'on' (mediana)")
    print(jf.round(2).to_string(index=False))
    print("\nM4 -- |z| mediano por escala")
    print(M4.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
