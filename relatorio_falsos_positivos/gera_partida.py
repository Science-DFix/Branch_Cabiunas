#!/usr/bin/env python3
"""Página dos FP de partida (seção "Os FP de partida em detalhe" do relatório). Descritivo; nada é treinado.

Usa a saída da pipeline final com E5 (runs_pipeline_unificada_final/: point_anomalies_final.csv e
episodios_classificados.csv, 41 FP) e o score bruto dos 3 OCSVM de produção em cache do E6
(frente_ocsvm/dados/e6/score_*.npy, alinhado ponto a ponto com point_anomalies_final.csv).
Saída: figs/partida_regime.png, figs/partida_score.png e partida.json.
Uso: python relatorio_falsos_positivos/gera_partida.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "frente_ocsvm"))
from e4_adia_partida import horas_desde_partida  # noqa: E402

RUN = AQUI.parent / "runs_pipeline_unificada_final"
E6 = AQUI.parent / "frente_ocsvm" / "dados" / "e6"
FIGS = AQUI / "figs"
INICIO = pd.Timestamp("2025-01-01")
# limiar de produção e limiar pós-partida do E6 (H = 6 h), saída de e6_limiar_pos_partida.py
LIM = {"temperatura": (10.5045, 32.2728), "vibracao": (0.17573, 12.9501), "oleo_pressao": (29.6747, 31.2858)}
CANAIS = {"canal_temperatura": ("temperatura", "#d73027"), "canal_vibracao": ("vibração", "#4575b4"),
          "canal_oleo_pressao": ("óleo", "#1a9850"), "canal_alarme_processo": ("canal 4 (alarmes)", "#777777")}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 110, "savefig.dpi": 160,
                     "axes.spines.top": False, "axes.spines.right": False})


def carrega():
    P = pd.read_csv(RUN / "point_anomalies_final.csv", parse_dates=["data_datetime"], index_col=0)
    for g in LIM:
        s = np.load(E6 / f"score_{g}.npy")
        assert len(s) == len(P), g
        P["s_" + g] = s
    P["hp"] = horas_desde_partida(P.operational_state)
    P["voto"] = P[list(CANAIS)].sum(axis=1)
    e = pd.read_csv(RUN / "episodios_classificados.csv", parse_dates=["start", "end"])
    e["dur_h"] = (e.end - e.start).dt.total_seconds() / 3600
    return P, e


def destino_das_partidas(P, e):
    """O que acontece nas 6 h depois de cada partida desde 01/2025."""
    on = P.operational_state.eq("on")
    part = P.index[on & ~on.shift(fill_value=False)]
    L = []
    for s in part[part >= INICIO]:
        ep = e[(e.start >= s) & (e.start <= s + pd.Timedelta(hours=6))]
        refr = bool(((e.start < s) & (e.start >= s - pd.Timedelta(hours=48))).any())
        L.append("refratário" if ep.empty and refr else ep.classe.iloc[0] if len(ep) else "sem voto de 45 min")
    return pd.Series(L).value_counts().to_dict(), int((part >= INICIO).sum())


def fig_regime(P, e):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1, 1.1]})
    m = P.operational_state.eq("on") & (P.index >= INICIO)
    b = np.floor(P.hp[m & (P.hp < 24)] * 2) / 2
    base = P[m & (P.hp >= 24)]
    for c, (nome, cor) in CANAIS.items():
        y = P.loc[b.index, c].groupby(b).mean() * 100
        a1.plot(y.index + 0.25, y.values, color=cor, label=nome)
        a1.axhline(base[c].mean() * 100, color=cor, ls=":", lw=0.9)
    y = (P.loc[b.index, "voto"] >= 2).groupby(b).mean() * 100
    a1.plot(y.index + 0.25, y.values, color="k", lw=2, label="voto ≥ 2")
    a1.axhline((base.voto >= 2).mean() * 100, color="k", ls=":", lw=0.9)
    a1.set_xlabel("horas depois de entrar em \"on\""); a1.set_ylabel("% do tempo com o canal aceso")
    a1.set_title("(a) canais acesos após a partida (pontilhado = depois de 24 h)")
    a1.legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(1, 0.88)); a1.set_xlim(0, 20)

    ep = e[(e.h_desde_partida <= 2) & e.classe.isin(["falso_positivo", "deteccao"])].sort_values("h_desde_partida")
    for i, r in enumerate(ep.itertuples()):
        cor = "#d73027" if r.classe == "falso_positivo" else "#1a9850"
        a2.barh(i, r.dur_h, left=r.h_desde_partida, color=cor, height=0.7)
        a2.text(r.h_desde_partida + r.dur_h + 0.15, i, f"{r.start:%d/%m/%y}  " + r.modelos_acesos.replace("oleo_pressao", "óleo").replace("vibracao", "vibração"),
                va="center", fontsize=6.5)
    for x, rot in ((1, "E4 (h = 1)"), (6, "E6 (H = 6 h)")):
        a2.axvline(x, color="#555555", ls="--", lw=0.9); a2.text(x, len(ep) - 0.3, rot, fontsize=7, ha="center")
    a2.set_yticks([]); a2.set_xlim(0, 16); a2.invert_yaxis()
    a2.set_xlabel("horas depois de entrar em \"on\"")
    a2.set_title("(b) voto aceso: FP (vermelho) e trips (verde) a ≤ 2 h da partida")
    fig.tight_layout(); fig.savefig(FIGS / "partida_regime.png"); plt.close(fig)


def razoes(P, ep):
    """Percentil 90 do score no episódio / limiar de produção, por canal de modelo aceso."""
    L = []
    for r in ep.itertuples():
        w = P.loc[r.start:r.end]
        for g in r.modelos_acesos.split("+"):
            L.append(dict(start=r.start, classe=r.classe, canal=g, razao=w["s_" + g].quantile(0.9) / LIM[g][0],
                          acima_pos=(w["s_" + g] > LIM[g][1]).mean()))
    return pd.DataFrame(L)


def fig_score(Z, teto):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    gs = list(LIM)
    for j, g in enumerate(gs):
        for dx, c, cor, nome in ((-0.12, "falso_positivo", "#d73027", "FP"), (0.12, "deteccao", "#1a9850", "trip")):
            z = Z[(Z.canal == g) & (Z.classe == c)]
            ax.scatter(np.full(len(z), j + dx), z.razao.clip(lower=0.05), color=cor, s=22,
                       label=nome if j == 0 else None, zorder=3)
        ax.hlines(LIM[g][1] / LIM[g][0], j - 0.35, j + 0.35, color="k", ls="--")
        ax.hlines(teto[g], j - 0.35, j + 0.35, color="#7b3294", ls=":", lw=1.5)
    ax.hlines([], 0, 0, color="k", ls="--", label="limiar do E6 / limiar de produção")
    ax.hlines([], 0, 0, color="#7b3294", ls=":", lw=1.5, label="teto do score (máximo possível)")
    ax.axhline(1, color="#999999", lw=0.8)
    ax.set_yscale("log"); ax.set_xticks(range(len(gs))); ax.set_xticklabels(["temperatura", "vibração", "óleo"])
    ax.set_ylabel("score (p90 no episódio) / limiar de produção")
    ax.set_title("Episódios a ≤ 2 h da partida: trips e FP perto do teto do score")
    ax.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(FIGS / "partida_score.png"); plt.close(fig)


def main():
    P, e = carrega()
    fp = e[e.classe == "falso_positivo"]
    perto = fp.h_desde_partida <= 2
    ep = e[(e.h_desde_partida <= 2) & e.classe.isin(["falso_positivo", "deteccao"])]
    Z = razoes(P, ep)
    # a OCSVM (RBF) satura: longe de todo vetor de suporte o score é constante (= rho)
    teto = {g: float(P["s_" + g].max() / LIM[g][0]) for g in LIM}
    fig_regime(P, e); fig_score(Z, teto)
    destino, n_part = destino_das_partidas(P, e)
    m = P.operational_state.eq("on") & (P.index >= INICIO)
    um_modelo = fp.modelos_acesos.str.count(r"\+").eq(0)
    N = {"n_fp": len(fp), "fp_ate_2h": int(perto.sum()),
         "inicio_h": fp[perto].h_desde_partida.describe().round(2).to_dict(),
         "faixas_inicio": pd.cut(fp.h_desde_partida, [0, 1, 2, 6, 24, 1e9]).value_counts(sort=False)
         .rename(str).to_dict(),
         "dur_h_partida": fp[perto].dur_h.describe().round(2).to_dict(),
         "dur_h_outros": fp[~perto].dur_h.describe().round(2).to_dict(),
         "trips_ate_2h": ep[ep.classe == "deteccao"][["start", "h_desde_partida", "dur_h"]].round(2)
         .astype(str).to_dict("records"),
         "um_modelo": {"partida": float(um_modelo[perto].mean()), "outros": float(um_modelo[~perto].mean())},
         "tags_c4_partida": {t: int(fp[perto].tags_canal4.fillna("").str.contains(t).sum())
                             for t in ("PAL_6240315", "PI_6240319_AL")},
         "canal4_aceso_pct": {"0-6h": float(P.canal_alarme_processo[m & (P.hp < 6)].mean() * 100),
                              ">24h": float(P.canal_alarme_processo[m & (P.hp >= 24)].mean() * 100)},
         "voto2_pct": {"1-3h": float((P.voto[m & (P.hp >= 1) & (P.hp < 3)] >= 2).mean() * 100),
                       ">24h": float((P.voto[m & (P.hp >= 24)] >= 2).mean() * 100)},
         "partidas_desde_2025": n_part, "destino_6h": destino,
         "razao_p90_mediana": {f"{g}/{c}": v for (g, c), v in Z.groupby(["canal", "classe"]).razao.median().round(1).items()},
         "teto": {g: round(v, 2) for g, v in teto.items()},
         "razao_E6": {g: round(v[1] / v[0], 2) for g, v in LIM.items()},
         "fp_acima_pos_90pct": int((Z[Z.classe == "falso_positivo"].groupby("start").acima_pos.max() >= 0.9).sum())}
    N = json.loads(json.dumps(N, default=str))
    (AQUI / "partida.json").write_text(json.dumps(N, indent=1, ensure_ascii=False))
    print(json.dumps(N, indent=1, ensure_ascii=False))
    print(Z.round(2).to_string())


if __name__ == "__main__":
    main()
