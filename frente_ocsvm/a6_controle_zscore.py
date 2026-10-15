#!/usr/bin/env python3
"""A6 -- os "61,9% dos FP com sinal físico real" (relatório operacional, seção 6.2) são específicos dos FP?

PRÉ-REGISTRADO EM 15/10/2026, ANTES DE RODAR. A seção 6.2 do relatório classificou os 42 FP pelo pico de |z| de 15
sensores (temperatura de mancal, T5, TI_0305, 10 vibrações, 2 de óleo) dentro do episódio, contra média e desvio das
2 h anteriores (terminando 3 min antes). Deu 26/42 (61,9%) "confirmado real" (|z| >= 3 em >= 2 sensores) e concluiu
que os FP são, na maioria, eventos físicos reais. O teste não teve controle: com 15 sensores a 30 s e um desvio de
2 h que pode ser pequeno em regime estável, |z| >= 3 em 2 sensores pode ser comum em QUALQUER janela.

MÉTODO. O mesmo do script original (dataset_francisco_lara/checkup_42_fp_pos_filtro.py): mesmos sensores, mesma
janela de baseline, mesmo pico de |z| e mesmas classes. Os 42 FP vêm da pipeline de referência (A0).
  Reprodução: os 42 FP devem dar 26 confirmados (11 fortes + 15 moderados), com folga de ±1.
  Controle: para cada FP, 20 janelas sorteadas de MESMA duração (semente 0), com início t tal que a máquina está
  "on" em t e em t + duração, e a janela [t - 2 h, t + duração + 2 h] não toca nenhum dos 75 episódios finais da
  pipeline. O baseline não tem restrição extra (como no original). 840 janelas.

LEITURA PRÉ-REGISTRADA (taxa de "confirmado" = forte + moderado):
  ESPECÍFICO      taxa nos FP - taxa no controle >= 20 p.p.  E  p < 0,05 (binomial unilateral, p0 = taxa controle).
  NÃO ESPECÍFICO  qualquer outro caso: os 61,9% não são evidência de que os FP sejam eventos físicos.
Se a reprodução falhar (fora de 25-27), o teste para e o motivo é investigado antes de qualquer leitura.

EXPECTATIVA REGISTRADA. NÃO ESPECÍFICO: o controle dá entre 40% e 55% de "confirmado" (diferença < 20 p.p.),
porque o critério mede variação de curto prazo de 15 sensores, e não algo próprio dos FP.

Uso:  python frente_ocsvm/a6_controle_zscore.py
"""
from __future__ import annotations
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest
import comum as C

VIB = ["TV_351X_A", "TV_351Y_A", "TV_352X_A", "TV_352Y_A", "TV_353X_A", "TV_353Y_A",
       "TV_354X_A", "TV_354Y_A", "TV_355X_A", "TV_355Y_A"]
OLEO = ["954005_624_PI_0308", "954005_624_PDIT_0305"]
COLS = ["TC382_03_A", "T5_AVG_A", "954005_624_TI_0305"] + VIB + OLEO
N_CTRL, SEMENTE = 20, 0
CONF = ("CONFIRMADO real (forte)", "confirmado real (moderado)")


def sensores() -> pd.DataFrame:
    raiz = os.path.expanduser("~/.clearml/cache/storage_manager/datasets/ds_a97ba56ba14840fbb1125c2a82f883c9")
    if not os.path.exists(os.path.join(raiz, "sensores_full_2024_2026_30s.csv")):  # sem cache: baixa do ClearML
        from clearml import Dataset
        raiz = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
    df = pd.read_csv(os.path.join(raiz, "sensores_full_2024_2026_30s.csv"), usecols=["data_datetime"] + COLS)
    df["data_datetime"] = pd.to_datetime(df["data_datetime"], errors="coerce")
    df = df.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def checkup(df: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> dict:
    """Igual ao script original: pico de |z| no episódio contra o baseline de 2 h (até 3 min antes)."""
    base = df.loc[t0 - pd.Timedelta(hours=2):t0 - pd.Timedelta(minutes=3)]
    ep = df.loc[t0:t1]
    if base.empty or ep.empty:
        return dict(abs_z_max=np.nan, n3=0)
    z = ((ep - base.mean()) / base.std()).abs().max().sort_values(ascending=False)
    return dict(abs_z_max=float(z.iloc[0]), n3=int((z > 3).sum()))


def classifica(r) -> str:
    if pd.isna(r["abs_z_max"]):
        return "sem dados"
    if r["abs_z_max"] >= 10 and r["n3"] >= 2:
        return CONF[0]
    if r["abs_z_max"] >= 3 and r["n3"] >= 2:
        return CONF[1]
    if r["abs_z_max"] >= 3:
        return "borderline (1 sensor só)"
    return "FRACO / sem explicação"


def main():
    can = C.carrega_canais()
    idx, op = can.index, can["operational_state"]
    ft = C.trips(idx)
    fin = C.decide({c: can[c].astype(bool) for c in C.CANAIS}, **{k: C.REF[k] for k in ("min_votes", "dur_min", "refrat_h")})
    cls, _ = C.avalia(fin, op, ft)
    fp = cls[cls.classe == "falso_positivo"].sort_values("start").reset_index(drop=True)
    print(f"{len(cls)} episódios, {len(fp)} FP")
    df = sensores()
    print(f"sensores: {len(df)} linhas")

    F = pd.DataFrame([checkup(df, r.start, r.end) for r in fp.itertuples()])
    F["classe"] = F.apply(classifica, axis=1)
    k = int(F.classe.isin(CONF).sum())
    print(f"\nREPRODUÇÃO: {F.classe.value_counts().to_dict()}  -> confirmados {k}/42 (esperado 26)")
    if not 25 <= k <= 27:
        print("reprodução fora de 25-27: parar e investigar")
        return

    on = (op == "on").values
    t_on = idx.values.astype("datetime64[ns]")
    ini = np.sort(cls.start.values.astype("datetime64[ns]")) - np.timedelta64(2, "h")
    fim = np.sort(cls.end.values.astype("datetime64[ns]")) + np.timedelta64(2, "h")
    cand = t_on[on]
    rng = np.random.default_rng(SEMENTE)
    L = []
    for r in fp.itertuples():
        dur = np.timedelta64(r.end - r.start)
        n = 0
        while n < N_CTRL:
            t = cand[rng.integers(len(cand))]
            j = np.searchsorted(t_on, t + dur)
            if j >= len(t_on) or not on[j]:
                continue
            a, b = t - np.timedelta64(2, "h"), t + dur + np.timedelta64(2, "h")
            if ((ini <= b) & (fim >= a)).any():  # toca algum episódio (com folga de 2 h)
                continue
            L.append(checkup(df, pd.Timestamp(t), pd.Timestamp(t + dur)))
            n += 1
    K = pd.DataFrame(L)
    K["classe"] = K.apply(classifica, axis=1)
    p0 = K.classe.isin(CONF).mean()
    p = binomtest(k, len(fp), p0, alternative="greater").pvalue
    dif = k / len(fp) - p0
    T = pd.DataFrame({"FP (42)": F.classe.value_counts(normalize=True), f"controle ({len(K)})": K.classe.value_counts(normalize=True)})
    print("\n" + (T.fillna(0) * 100).round(1).to_string())
    print(f"\nmediana do pico |z|: FP {F.abs_z_max.median():.1f} · controle {K.abs_z_max.median():.1f}")
    print(f"confirmado: FP {k / len(fp):.1%} · controle {p0:.1%} · diferença {dif * 100:+.1f} p.p. · p = {p:.4f}")
    print("->", "ESPECÍFICO" if dif >= 0.20 and p < 0.05 else "NÃO ESPECÍFICO")
    sem = fp.loc[F.classe == "FRACO / sem explicação", "start"].dt.strftime("%Y-%m-%d_%Hh%M").tolist()
    print("FP sem explicação física:", sem)


if __name__ == "__main__":
    main()
