#!/usr/bin/env python3
"""F1 -- a pipeline de produção CONGELADA aplicada a dados novos (21/04/2026 em diante). Local, sem retreino.

PRÉ-REGISTRO (19/10/2026, antes de existir qualquer dado depois de 20/04/2026)
Tudo o que foi ajustado na frente OCSVM (limiar e debounce por canal, tags e janela do canal 4, voto, filtro de
45 min, refratário, regra de falha de comunicação) foi escolhido olhando os 8 trips de 07/2024 a 04/2026. O único
teste limpo é rodar a pipeline exatamente como está em eventos que ela nunca viu.

O QUE FICA CONGELADO
  - os 3 modelos OCSVM (`best_model/model.pkl` das tasks b494d457 / 9c344688 / fc4123fb), a normalização
    (`normalization_stats.json`) e o limiar/debounce escolhidos (`best_hyperparameters.json`);
  - os configs dessas tasks (commit 6842ac7: portões, máscara operacional, filtro de 4,5 min por canal);
  - as duas estatísticas que a produção tira da série inteira e que mudariam com dados novos: os quantis de
    clipping de cada feature (0,1% / 99,9%) e o limiar de |diff| do RUNNING_A que marca transiente (quantil 99%).
    Ficam fixos nos valores do período de referência (01/07/2024 a 20/04/2026), em `f1_congelado.json`;
  - a camada de decisão de `scripts/pipeline_unificada_final.py` (voto >= 2 de 4, canal 4 = 5 tags/24 h, filtro
    de 45 min, refratário de 48 h, regra de falha de comunicação de 6 h) e a régua (episódios, 48 h, parada >= 2 h).

CONFERÊNCIA DA FERRAMENTA (`confere`, antes de qualquer dado novo): no período de referência, o `is_anom_point` e o
`operational_state` reconstruídos têm de bater com o `point_anomalies_all.csv` de cada task, e a decisão final tem
de dar 8/8, 41 FP, 25 inconclusivos e 2,82 FP/mês (a produção com E5). Se não bater, a ferramenta não serve.
  RESULTADO DA CONFERÊNCIA (19/10/2026): CONFERE. 0 instantes divergentes em is_anom_point e em operational_state
  nos 3 canais (29.513 / 142.346 / 7.042 pontos anômalos, iguais aos da produção); decisão final 8/8, 41 FP,
  25 inconclusivos, 2,816 FP/mês. Roda em ~27 min e ~7 GB de pico (a pontuação da vibração domina).

HIPÓTESES PROSPECTIVAS (medidas só no período novo; o período de referência entra apenas como aquecimento das
janelas móveis). Trips = FALHA_CURADA da lista nova; se a planta mandar a causa das paradas, a régua usa a mesma.
  P1  detecção: a pipeline congelada detecta >= 75% dos trips novos (detecção = episódio que começa até 48 h antes).
      Com menos de 4 trips novos, P1 só é relatado (n pequeno demais para veredito).
  P2  FP/mês no período novo <= 4,3 (1,5 x os 2,877 fora da amostra). Relatar também o IC de Poisson de 95%.
  Se P1 e P2 passam, a pipeline se sustenta em dados novos. Se P2 falha, o FP de 2,8/mês era otimista.
  Regras candidatas (cada uma comparada à pipeline congelada no período novo, com o critério da frente: mantém
  todos os trips novos detectados e corta >= 25% do FP):
  S1  canal 4 sem PI_6240319_AL (4 tags). Motivo: o PI_6240319_AL parou de alarmar depois de 27/02/2026; se
      continuar parado, a regra não muda nada; se voltar, a pergunta é se ele ainda precede trip.
  S2  prioridade "dois modelos" (B3-R1): episódio ALTA se >= 2 canais OCSVM acenderam juntos. Não suprime; aprova
      se todos os trips novos detectados ficam em ALTA e >= 30% dos FP em BAIXA (critério do B3). Motivo: fora da
      amostra, 3/3 trips ficaram em ALTA e 52% dos FP em BAIXA (a posteriori, B3).
  S3  prioridade "alerta repetido": episódio ALTA se houve outro episódio nos 7 dias anteriores. Mesmo critério de
      S2. Motivo: 6/8 trips vs 22/41 FP na referência (só descritivo, contexto do alerta).
  Expectativa registrada: P1 passa se vierem trips parecidos com os de 2025 (canal 4 + vibração); P2 fica perto
  de 3/mês; S1 não muda nada se o PI_6240319_AL seguir parado; S2 e S3 reprovam, como reprovaram na referência.
  Fora do F1 (precisa de retreino no Cica): D2-B (exige o CNN-1D mensal re-treinado nos meses novos).

Uso:
  python frente_ocsvm/f1_congelado.py confere
      congela as estatísticas no período de referência, salva `f1_congelado.json` e confere contra a produção
  python frente_ocsvm/f1_congelado.py pontua SENSORES.csv CATALOGO.csv TRIPS.csv
      SENSORES.csv: o mesmo formato de `sensores_full_2024_2026_30s.csv`, cobrindo 07/2024 até a data nova;
      CATALOGO.csv: o catálogo de alarmes no formato de `alarmes_selecionados_turbina_a.csv`;
      TRIPS.csv: lista de falhas no formato de `alarmes_francisco_falhas.csv` (Tag Alarme == FALHA_CURADA).
      Saída em `dados/f1/`. As hipóteses acima são avaliadas por um script separado, escrito antes de abrir a saída.
"""
from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from clearml import Dataset, Task

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))
import pipeline_unificada_final as P  # noqa: E402
from src.cnn1d_ae.config import PipelineConfig, update_cfg_from_dict  # noqa: E402
from src.cnn1d_ae.io import _process_time_column  # noqa: E402
from src.cnn1d_ae.preprocess import build_group_dataframe, build_sensor_dataframe  # noqa: E402
from src.cnn1d_ae.automl_models import ocsvm_error  # noqa: E402
from src.cnn1d_ae.scoring import (  # noqa: E402
    build_operational_state, map_seq_to_point_anomalies, apply_min_duration_filter, apply_load_gate,
    compute_frozen_sensor_mask, apply_frozen_sensor_veto, compute_step_change_index, apply_step_change_gate,
    combine_channels_vote, apply_refractory, compute_comm_fail_mask,
)

COMMIT_PROD = "6842ac7"
CANAIS = {  # nome do canal na decisão -> (task, grupo)
    "temperatura": (P.TID_TEMP, "mancal_temperatura_isolada"),
    "vibracao": (P.TID_VIB, "mancal_vibracao_isolada"),
    "oleo_pressao": (P.TID_OLEO, "oleo_pressao_isolada"),
}
CONGELADO = AQUI / "f1_congelado.json"
SAIDA = AQUI / "dados" / "f1"


def config_da_task(task: Task) -> PipelineConfig:
    import subprocess
    caminho = task.get_parameters()["Args/config"]
    txt = subprocess.run(["git", "show", f"{COMMIT_PROD}:{caminho}"], cwd=RAIZ, check=True,
                         capture_output=True, text=True).stdout
    return update_cfg_from_dict(PipelineConfig(), json.loads(txt))


def artefatos(task: Task, grupo: str):
    a = {k: task.artifacts[f"{grupo}/best_model/{k}"].get_local_copy()
         for k in ["model.pkl", "normalization_stats.json", "best_hyperparameters.json"]}
    with open(a["model.pkl"], "rb") as f:
        modelo = pickle.load(f)
    return modelo, json.loads(Path(a["normalization_stats.json"]).read_text()), \
        json.loads(Path(a["best_hyperparameters.json"]).read_text())


def le_sensores(cfg: PipelineConfig, caminho: str, colunas: list[str], fim: str | None) -> pd.DataFrame:
    df = pd.read_csv(caminho, usecols=[cfg.TIME_COL] + colunas)
    df = _process_time_column(df, cfg.TIME_COL, cfg, "raw")
    df = df.loc[df[cfg.TIME_COL] >= pd.Timestamp(cfg.DATA_START_DATE)]
    if fim:
        df = df.loc[df[cfg.TIME_COL] <= pd.Timestamp(fim)]
    return df.reset_index(drop=True)


def pontua_grupo(cfg, df_raw, modelo, norm, hp, cong: dict | None, err=None, limiar=None):
    """O mesmo caminho de `run_automl_group` para o trial vencedor, sem treino. `cong` = estatísticas
    congeladas; None = tira da própria série (é como a produção fez) e devolve-as. `err` = score já calculado
    (pula a OCSVM); `limiar` = limiar por instante (padrão: o da produção, `hp["threshold"]`)."""
    grupo = cfg.SENSOR_GROUPS[0]
    feats = norm["feature_columns"]
    sensores = [s for s in grupo["sensors"] if s in feats]
    # features sensor a sensor (são independentes entre sensores): o mesmo resultado de build_group_dataframe
    # com todos juntos, sem estourar os 7 GB da máquina local com o float64 intermediário da vibração
    df_use = pd.concat([build_group_dataframe(cfg, df_raw, df_raw, [s])[0] for s in sensores], axis=1)
    brutos = df_use[sensores].copy()
    alvo = grupo.get("target_sensor") if grupo.get("target_sensor") in sensores else None

    ref, _ = build_sensor_dataframe(cfg, df_raw, df_raw, cfg.OPERATIONAL_REF_SENSOR)
    ref = ref[cfg.OPERATIONAL_REF_SENSOR]
    if cong is None:
        s = pd.to_numeric(ref.reindex(df_use.index), errors="coerce").ffill().bfill()
        dq = float(np.clip(cfg.TRANSIENT_DIFF_QUANTILE, 0.5, 0.9999))
        X = df_use[feats]
        cong = {"diff_thr": float(s.diff().abs().fillna(0.0).quantile(dq)),
                "clip_low": X.quantile(cfg.OUTLIER_Q_LOW).to_dict(), "clip_high": X.quantile(cfg.OUTLIER_Q_HIGH).to_dict()}
    state = build_operational_state(
        index=df_use.index, sensor_series=ref, off_value_quantile=cfg.OFF_VALUE_QUANTILE,
        off_abs_threshold=cfg.OFF_ABS_THRESHOLD, off_long_min_hours=cfg.OFF_LONG_MIN_HOURS,
        transient_padding_minutes=cfg.TRANSIENT_PADDING_MINUTES, transient_diff_quantile=cfg.TRANSIENT_DIFF_QUANTILE,
        secondary_series=df_use[alvo] if (cfg.OFF_TARGET_ABS_THRESHOLD is not None and alvo) else None,
        secondary_off_abs_threshold=cfg.OFF_TARGET_ABS_THRESHOLD, transient_diff_abs_threshold=cong["diff_thr"],
    )

    # clip e z-score em float64, depois float32, como clip_outliers/normalize_train_only; em blocos (memória)
    lo, hi = (np.array([cong[k][c] for c in feats]) for k in ("clip_low", "clip_high"))
    mu = np.array([norm["normalize_center"][c] for c in feats])
    sd = np.array([norm["normalize_scale"][c] for c in feats])
    X = df_use[feats].to_numpy()
    zidx = df_use.index
    keep = {c: df_use[c] for c in {alvo, cfg.LOAD_GATE_SENSOR, cfg.STEP_CHANGE_GATE_SENSOR} if c in sensores}
    del df_use
    if err is None:
        err = np.empty(len(X), dtype=np.float32)
        for i in range(0, len(X), 100_000):
            xb = np.clip(X[i:i + 100_000].astype(np.float64), lo, hi).astype(np.float32).astype(np.float64)
            err[i:i + 100_000] = ocsvm_error(modelo, ((xb - mu) / sd).astype(np.float32))
    del X
    flags = (err > (hp["threshold"] if limiar is None else limiar)).astype(int)

    d = int(hp["debounce"])
    df_point = map_seq_to_point_anomalies(flags, zidx, time_steps=1, point_rule="all_of_window",
                                          point_window=d, point_min_count=d)
    df_point["operational_state"] = state.reindex(df_point.index).fillna("on")
    df_point.loc[df_point["operational_state"] != "on", "is_anom_point"] = 0
    if cfg.ENABLE_MIN_DURATION_FILTER:
        df_point = apply_min_duration_filter(df_point, cfg.MIN_DURATION_FILTER_MINUTES)
    if cfg.ENABLE_LOAD_GATE:
        carga = keep[cfg.LOAD_GATE_SENSOR] if cfg.LOAD_GATE_SENSOR in sensores else \
            build_sensor_dataframe(cfg, df_raw, df_raw, cfg.LOAD_GATE_SENSOR)[0][cfg.LOAD_GATE_SENSOR]
        df_point = apply_load_gate(df_point, carga, ramp_max=cfg.LOAD_GATE_RAMP_MAX, level_min=cfg.LOAD_GATE_LEVEL_MIN,
                                   ramp_halflife_minutes=cfg.LOAD_GATE_RAMP_HALFLIFE_MINUTES,
                                   window_minutes=cfg.LOAD_GATE_WINDOW_MINUTES)
    assert not cfg.ENABLE_VOLATILITY_GATE and not cfg.ENABLE_SCORE_EWMA
    if cfg.ENABLE_FROZEN_SENSOR_VETO:
        df_point = apply_frozen_sensor_veto(df_point, compute_frozen_sensor_mask(brutos, cfg.FROZEN_SENSOR_VETO_WINDOW_MINUTES))
    if cfg.ENABLE_STEP_CHANGE_GATE:
        sensor = cfg.STEP_CHANGE_GATE_SENSOR or cfg.LOAD_GATE_SENSOR
        serie = keep[sensor] if sensor in sensores else build_sensor_dataframe(cfg, df_raw, df_raw, sensor)[0][sensor]
        idx = compute_step_change_index(serie, cfg.STEP_CHANGE_SHORT_WINDOW_MINUTES, cfg.STEP_CHANGE_LONG_WINDOW_MINUTES)
        df_point = apply_step_change_gate(df_point, idx, cfg.STEP_CHANGE_THRESHOLD)
    df_point["score"] = err
    return df_point, cong


def decide(pontos: dict, catalogo: pd.DataFrame, raw_txt: pd.DataFrame, ft: pd.Series, tags_c4=None):
    """A camada de decisão de `pipeline_unificada_final.main` (min_votes = 2), sobre os canais dados."""
    idx = pontos["temperatura"].index
    for p in pontos.values():
        idx = idx.intersection(p.index)
    op = pontos["temperatura"].loc[idx, "operational_state"]
    alarmes = catalogo.loc[catalogo["Tag"].isin(tags_c4 or P.ALARM_CHANNEL_TAGS), "Data da Ocorrencia"]
    canais = {}
    for g, p in pontos.items():
        cf = compute_comm_fail_mask(raw_txt[P.GROUP_SENSORS[g]], P.COMM_FAIL_WINDOW_HOURS).reindex(idx, fill_value=False)
        canais[g] = p.loc[idx, "is_anom_point"].astype(bool) & ~cf
    canais["alarme_processo"] = P.alarm_channel_bool(idx, alarmes, P.ALARM_WINDOW_HOURS)
    voto = combine_channels_vote(canais, min_votes=2)
    filtrado = apply_min_duration_filter(pd.DataFrame({"is_anom_point": voto.astype(int)}, index=idx),
                                         P.MIN_VOTE_DURATION_MINUTES)["is_anom_point"].astype(bool)
    final = apply_refractory(filtrado, refractory_minutes=P.REFRACTORY_HOURS * 60.0)
    cls, met = P.evaluate(final, op, ft[(ft >= idx.min()) & (ft <= idx.max())])
    return canais, final, op, cls, met


def le_catalogo(caminho) -> pd.DataFrame:
    a = pd.read_csv(caminho)
    a["Data da Ocorrencia"] = pd.to_datetime(a["Data da Ocorrência"], errors="coerce")
    a["Tag"] = a["Tag Alarme"]
    return a[a["Status"].astype(str).str.startswith("ACT")].dropna(subset=["Data da Ocorrencia"])


def le_trips(caminho) -> pd.Series:
    f = pd.read_csv(caminho)
    return pd.to_datetime(f.loc[f["Tag Alarme"] == "FALHA_CURADA", "Data da Ocorrência"]).sort_values()


def le_texto(caminho) -> pd.DataFrame:
    cols = [c for v in P.GROUP_SENSORS.values() for c in v]
    raw = pd.read_csv(caminho, usecols=["data_datetime"] + cols, dtype=str)
    raw.index = pd.to_datetime(raw.pop("data_datetime"))
    return raw.sort_index()


def roda(sensores_csv, catalogo_csv, trips_csv, fim, cong_todos: dict | None):
    pontos, novos = {}, {}
    for g, (tid, grupo) in CANAIS.items():
        print(f"[{g}] baixando modelo da task {tid[:8]} e pontuando...", flush=True)
        task = Task.get_task(task_id=tid)
        cfg = config_da_task(task)
        modelo, norm, hp = artefatos(task, grupo)
        cols = sorted(set(cfg.SENSOR_GROUPS[0]["sensors"]) | {cfg.OPERATIONAL_REF_SENSOR}
                      | {c for c in [cfg.LOAD_GATE_SENSOR, cfg.STEP_CHANGE_GATE_SENSOR] if c})
        cfg.DATA_END_DATE = fim
        df_raw = le_sensores(cfg, sensores_csv, cols, fim)
        pontos[g], novos[g] = pontua_grupo(cfg, df_raw, modelo, norm, hp, None if cong_todos is None else cong_todos[g])
    canais, final, op, cls, met = decide(pontos, le_catalogo(catalogo_csv), le_texto(sensores_csv), le_trips(trips_csv))
    return pontos, novos, canais, final, op, cls, met


def confere():
    root = Dataset.get(dataset_id=P.ALARM_CATALOG_DATASET_ID).get_local_copy()
    sens = str(Path(root) / P.SENSORS_FILE)
    trips = str(AQUI / "dados" / "alarmes_francisco_falhas.csv")
    pontos, cong, canais, final, op, cls, met = roda(sens, str(Path(root) / P.ALARM_CATALOG_FILE), trips,
                                                    "2026-04-20", None)
    CONGELADO.write_text(json.dumps(cong, indent=1))
    print(f"estatísticas congeladas em {CONGELADO.name}")

    ok = True
    for g, (tid, grupo) in CANAIS.items():
        prod = pd.read_csv(Task.get_task(task_id=tid).artifacts[f"{grupo}/csv/point_anomalies_all.csv"].get_local_copy(),
                           index_col=0, parse_dates=True, low_memory=False)
        p = pontos[g]
        mesmo_idx = p.index.equals(prod.index)
        dif = int((p["is_anom_point"].values != prod["is_anom_point"].values).sum()) if mesmo_idx else -1
        dif_op = int((p["operational_state"].values != prod["operational_state"].values).sum()) if mesmo_idx else -1
        print(f"[{g}] índice igual: {mesmo_idx}  pontos anômalos {int(p['is_anom_point'].sum())} vs "
              f"{int(prod['is_anom_point'].sum())}  divergências is_anom_point: {dif}  estado: {dif_op}")
        ok &= mesmo_idx and dif == 0 and dif_op == 0
    print(f"decisão final: {met['falhas_detectadas']}/{met['n_falhas_catalogadas']} trips, "
          f"{met['n_episodios_falso_positivo']} FP, {met['n_episodios_inconclusivo']} inconclusivos, "
          f"{met['falso_positivo_por_mes']:.3f} FP/mês")
    ok &= (met["falhas_detectadas"], met["n_episodios_falso_positivo"], met["n_episodios_inconclusivo"]) == (8, 41, 25)
    print("CONFERE" if ok else "NÃO CONFERE")


def pontua(sensores_csv, catalogo_csv, trips_csv):
    cong = json.loads(CONGELADO.read_text())
    pontos, _, canais, final, op, cls, met = roda(sensores_csv, catalogo_csv, trips_csv, None, cong)
    SAIDA.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({**{f"canal_{k}": v.astype(int) for k, v in canais.items()},
                       **{f"score_{k}": pontos[k]["score"].reindex(final.index) for k in pontos},
                       "is_anom_point": final.astype(int), "operational_state": op})
    df.to_csv(SAIDA / "point_anomalies_congelado.csv")
    cls.to_csv(SAIDA / "episodios_congelado.csv", index=False)
    (SAIDA / "metrics.json").write_text(json.dumps(met, indent=1, default=str))
    print(f"saída em {SAIDA} (período inteiro; as hipóteses olham só depois de 20/04/2026)")


if __name__ == "__main__":
    if sys.argv[1:2] == ["confere"]:
        confere()
    elif sys.argv[1:2] == ["pontua"] and len(sys.argv) == 5:
        pontua(*sys.argv[2:5])
    else:
        sys.exit(__doc__)
