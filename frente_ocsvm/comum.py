"""Peças comuns da frente OCSVM (DIARIO_DE_BORDO.md): dados, canal 4 e a camada de decisão.

A camada de decisão chama as MESMAS funções de `src/cnn1d_ae/scoring.py` que
`scripts/pipeline_unificada_final.py` usa; aqui só se troca de onde vêm os canais (os 4 já calculados
em `point_anomalies_final.csv`, sem ClearML) e se expõem os parâmetros para os testes.

Dados em `frente_ocsvm/dados/` (fora do git; links simbólicos):
  point_anomalies_final.csv           saída do pipeline final (tasks b494d457 / 9c344688 / fc4123fb)
  alarmes_francisco_falhas.csv        trips curados (Tag Alarme == "FALHA_CURADA")
  alarmes_selecionados_turbina_a.csv  catálogo de alarmes (Dataset a97ba56b)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
from src.cnn1d_ae.scoring import (  # noqa: E402
    combine_channels_vote, apply_refractory, apply_min_duration_filter,
    group_alerts_into_episodes, classify_episodes_regua, compute_regua_metrics,
    compute_operational_period_days,
)

DADOS = AQUI / "dados"
SPLIT = pd.Timestamp("2025-07-01")                 # AUTOML_OOS_SPLIT_DATE dos 3 configs
TAGS_C4 = ["PI_6240319_AL", "PAL_6240315", "PDAL_6240302", "TC382_05_A", "PAH_6240319"]
REF = dict(min_votes=2, janela_c4_h=24.0, dur_min=45.0, refrat_h=48.0)
CANAIS = ["canal_temperatura", "canal_vibracao", "canal_oleo_pressao", "canal_alarme_processo"]


def carrega_canais() -> pd.DataFrame:
    return pd.read_csv(DADOS / "point_anomalies_final.csv", index_col=0, parse_dates=True, low_memory=False)


def trips(idx: pd.DatetimeIndex) -> pd.Series:
    f = pd.read_csv(DADOS / "alarmes_francisco_falhas.csv")
    t = pd.to_datetime(f.loc[f["Tag Alarme"] == "FALHA_CURADA", "Data da Ocorrência"]).sort_values()
    return t[(t >= idx.min()) & (t <= idx.max())].reset_index(drop=True)


def catalogo() -> pd.DataFrame:
    a = pd.read_csv(DADOS / "alarmes_selecionados_turbina_a.csv")
    a["t"] = pd.to_datetime(a["Data da Ocorrência"], errors="coerce")
    a["tag"] = a["Tag Alarme"]
    return a[a["Status"].astype(str).str.startswith("ACT")].dropna(subset=["t"])[["t", "tag"]]


def canal_alarme(idx: pd.DatetimeIndex, tempos: pd.Series, janela_h: float) -> pd.Series:
    """Igual a `alarm_channel_bool` do pipeline final: houve alarme nas últimas `janela_h` horas."""
    t = np.sort(pd.DatetimeIndex(pd.Series(tempos).dropna()).values.astype("datetime64[ns]"))
    x = idx.values.astype("datetime64[ns]")
    out = np.zeros(len(x), bool)
    if len(t):
        pos = np.searchsorted(t, x, side="right") - 1
        ok = pos >= 0
        dt = (x[ok] - t[pos[ok]]).astype("timedelta64[s]").astype(np.float64) / 3600.0
        out[ok] = dt <= float(janela_h)
    return pd.Series(out, index=idx)


def decide(canais: dict, min_votes=2, dur_min=45.0, refrat_h=48.0) -> pd.Series:
    voto = combine_channels_vote(canais, min_votes=min_votes)
    v = apply_min_duration_filter(pd.DataFrame({"is_anom_point": voto.astype(int)}), dur_min)["is_anom_point"]
    return apply_refractory(v.astype(bool), refractory_minutes=refrat_h * 60.0)


def avalia(fin: pd.Series, op: pd.Series, ft: pd.Series) -> tuple[pd.DataFrame, dict]:
    df = pd.DataFrame({"is_anom_point": fin.astype(int), "operational_state": op})
    eps = group_alerts_into_episodes(df["is_anom_point"], merge_gap_minutes=120.0)
    cls = classify_episodes_regua(eps, ft, df["operational_state"], 48.0, 48.0, 2.0)
    return cls, compute_regua_metrics(cls, ft, compute_operational_period_days(df))


def por_trip(cls: pd.DataFrame, ft: pd.Series) -> pd.DataFrame:
    """1ª detecção e antecedência por trip; dentro/fora da amostra e banda de 4 h."""
    L = []
    for t in ft:
        d = cls[(cls.classe == "deteccao") & (pd.to_datetime(cls.falha_associada) == t)]
        ini = d.start.min() if len(d) else pd.NaT
        lead = (t - ini).total_seconds() / 3600 if len(d) else np.nan
        L.append(dict(trip=t, deteccao=ini, lead_h=round(lead, 1) if len(d) else np.nan,
                      amostra="dentro" if t < SPLIT else "fora", banda_4h=bool(len(d) and lead >= 4.0)))
    return pd.DataFrame(L)
