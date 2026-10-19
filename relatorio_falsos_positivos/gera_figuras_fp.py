#!/usr/bin/env python3
"""Figuras e tabelas do relatório dos falsos positivos (relatorio_falsos_positivos.tex).

Usa a triagem do E1 (frente_ocsvm/dados/e1_triagem.csv, gerada por frente_ocsvm/e1_triagem_fp.py) e as medidas do E2
(recalculadas aqui, locais). Nada é treinado. Saída: figs/*.png, tabs/*.tex e numeros.json.
Uso: python relatorio_falsos_positivos/gera_figuras_fp.py
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
import matplotlib.dates as mdates

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "frente_ocsvm"))
import comum as C  # noqa: E402
from e2_novidade_alarme import idade_acesa, medidas  # noqa: E402
from src.cnn1d_ae.scoring import apply_min_duration_filter, apply_refractory, combine_channels_vote  # noqa: E402

FIGS, TABS = AQUI / "figs", AQUI / "tabs"
FIGS.mkdir(exist_ok=True); TABS.mkdir(exist_ok=True)
CAUSAS = ["falha_txt", "reta", "constante", "partida", "rajada", "pos_2702", "cronico"]
ROTULO = {"falha_txt": "Comm Fail / Out of Serv", "reta": "sinal interpolado", "constante": "sensor parado ≥ 10 min",
          "partida": "até 6 h após a partida", "rajada": "rajada de alarmes de temperatura",
          "pos_2702": "depois de 27/02/2026", "cronico": "alarme crônico (≥ 10 em 7 d)"}
SERIE_VIB = (pd.Timestamp("2026-01-31"), pd.Timestamp("2026-02-15"))
# categoria principal de cada FP, em ordem de prioridade (a 1ª que se aplica)
CATEG = [("falha de comunicação", "#7f3b08"), ("partida (≤ 2 h)", "#d73027"), ("rajada de temperatura", "#fdae61"),
         ("série de vibração 31/01–14/02/26", "#4575b4"), ("regime pós-27/02/26", "#8073ac"),
         ("alarme crônico", "#999999"), ("sem causa identificada", "#1a1a1a")]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 110, "savefig.dpi": 160,
                     "axes.spines.top": False, "axes.spines.right": False})


def categoria(r):
    if r.falha_txt:
        return CATEG[0][0]
    if r.h_partida <= 2:
        return CATEG[1][0]
    if r.rajada:
        return CATEG[2][0]
    if SERIE_VIB[0] <= r.start < SERIE_VIB[1]:
        return CATEG[3][0]
    if r.pos_2702:
        return CATEG[4][0]
    if r.cronico:
        return CATEG[5][0]
    return CATEG[6][0]


def fig_causas(T):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, m in zip(axs, ("OCSVM", "CNN-1D mensal")):
        t = T[T.modelo == m]
        n = t.classe.value_counts()
        y = np.arange(len(CAUSAS))
        for dy, c, cor, nome in ((-0.2, "deteccao", "#1a9850", "trips"), (0.2, "falso_positivo", "#d73027", "FP")):
            v = 100 * t[t.classe == c][CAUSAS].mean()
            ax.barh(y + dy, v.values, 0.4, color=cor, label=f"{nome} (n={n.get(c, 0)})")
        ax.set_yticks(y); ax.set_yticklabels([ROTULO[c] for c in CAUSAS]); ax.invert_yaxis()
        ax.set_xlabel("% dos episódios com a causa"); ax.set_title(m); ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Causas candidatas: frequência nos trips e nos FP (desde 01/2025)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGS / "causas.png"); plt.close(fig)


def fig_partida(T):
    t = T[T.modelo == "OCSVM"]
    bins = [-0.01, 0.5, 1, 2, 6, 24, 72, 1e5]
    rot = ["≤ 0,5 h", "0,5–1 h", "1–2 h", "2–6 h", "6–24 h", "1–3 d", "> 3 d"]
    fig, ax = plt.subplots(figsize=(8, 3.3))
    x = np.arange(len(rot))
    for dx, c, cor, nome in ((-0.27, "deteccao", "#1a9850", "trips"), (0, "falso_positivo", "#d73027", "FP"),
                             (0.27, "inconclusivo", "#f39c12", "inconclusivos")):
        h = pd.cut(t[t.classe == c].h_partida, bins, labels=rot).value_counts(sort=False)
        ax.bar(x + dx, h.values, 0.27, color=cor, label=f"{nome} (n={int(h.sum())})")
    ax.set_xticks(x); ax.set_xticklabels(rot); ax.set_xlabel("tempo entre a partida da máquina e o início do episódio")
    ax.set_ylabel("episódios"); ax.legend(fontsize=8)
    ax.set_title("OCSVM: quando os episódios começam em relação à última partida (desde 01/2025)")
    fig.tight_layout(); fig.savefig(FIGS / "partida.png"); plt.close(fig)


def fig_linha_tempo(T, ft):
    fig, axs = plt.subplots(2, 1, figsize=(11, 3.8), sharex=True)
    cores = dict(CATEG)
    for ax, m in zip(axs, ("OCSVM", "CNN-1D mensal")):
        t = T[(T.modelo == m) & (T.classe == "falso_positivo")]
        for cat, g in t.groupby("categoria"):
            ax.scatter(g.start, np.zeros(len(g)), s=60, marker="|", lw=3, color=cores[cat])
        d = T[(T.modelo == m) & (T.classe == "deteccao")]
        ax.scatter(d.start, np.full(len(d), 0.6), s=30, marker="v", color="#1a9850")
        for x in ft[ft >= pd.Timestamp("2025-01-01")]:
            ax.axvline(x, color="k", ls="--", lw=0.7, alpha=0.6)
        ax.set_yticks([]); ax.set_ylim(-0.6, 1.0); ax.set_ylabel(m, rotation=0, ha="right", va="center")
        ax.spines["left"].set_visible(False)
    hs = [plt.Line2D([], [], color=c, marker="|", ls="", ms=10, mew=3, label=n) for n, c in CATEG]
    hs += [plt.Line2D([], [], color="#1a9850", marker="v", ls="", label="início da detecção"),
           plt.Line2D([], [], color="k", ls="--", label="trip")]
    axs[0].legend(handles=hs, ncol=5, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
    fig.suptitle("FP por categoria principal ao longo do tempo", fontsize=10, y=1.0)
    fig.tight_layout(); fig.savefig(FIGS / "linha_tempo_fp.png", bbox_inches="tight"); plt.close(fig)


def novidade():
    """Medidas do E2 por episódio (OCSVM de produção, período inteiro)."""
    df = C.carrega_canais()
    idx, op = df.index, df["operational_state"]
    cat = C.catalogo()
    tempos = cat.loc[cat.tag.isin(C.TAGS_C4), "t"].sort_values().reset_index(drop=True)
    ch = {k: df[k].astype(bool) for k in C.CANAIS[:3]}
    ch["c4"] = C.canal_alarme(idx, tempos, C.REF["janela_c4_h"])
    idade = idade_acesa(ch["c4"])
    ft = C.trips(idx)
    voto = combine_channels_vote(ch, min_votes=2)
    dur = apply_min_duration_filter(pd.DataFrame({"is_anom_point": voto.astype(int)}), C.REF["dur_min"])["is_anom_point"].astype(bool)
    cls, _ = C.avalia(apply_refractory(dur, refractory_minutes=C.REF["refrat_h"] * 60.0), op, ft)
    return pd.DataFrame([dict(start=e.start, classe=e.classe, **medidas(e.start, ch["c4"], idade, tempos))
                         for _, e in cls.iterrows()])


def fig_novidade(E):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    rng = np.random.default_rng(0)
    for ax, f, tit in ((axs[0], "N1", "há quanto tempo o canal 4 está aceso (h)"),
                       (axs[1], "N2", "ativações das 5 tags nos 7 dias anteriores")):
        for i, (c, cor, nome) in enumerate((("deteccao", "#1a9850", "trips"), ("falso_positivo", "#d73027", "FP"))):
            v = E.loc[E.classe == c, f]
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=18, color=cor, alpha=0.8)
            ax.hlines(v.median(), i - 0.25, i + 0.25, color="k")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["trips (8)", "FP (42)"]); ax.set_title(tit)
    fig.suptitle("E2: a novidade do alarme não separa trip de FP (OCSVM, período inteiro; traço = mediana)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGS / "novidade.png"); plt.close(fig)


def tex_apendice(T):
    t = T[(T.modelo == "OCSVM") & (T.classe == "falso_positivo")].copy()
    L = [r"\begin{longtable}{l >{\raggedright\arraybackslash}p{1.4cm} >{\raggedright\arraybackslash}p{2.4cm} >{\raggedright\arraybackslash}p{2.9cm} r r >{\raggedright\arraybackslash}p{3.6cm}}", r"\toprule",
         r"início & modelo & gatilho do canal 4 & categoria & h pós partida & alarm. 7 d & outras causas \\",
         r"\midrule", r"\endhead"]
    for _, r in t.iterrows():
        outras = ", ".join(ROTULO[c] for c in CAUSAS if r[c] and c != "partida")
        L.append(" & ".join([r.start.strftime("%d/%m/%y %H:%M"), r.modelos, r.gatilho.replace("_", r"\_"),
                             r.categoria.replace("≤", r"$\leq$"), f"{r.h_partida:.1f}".replace(".", ","), str(int(r.n_alarmes_7d)),
                             outras.replace("≥", r"$\geq$")]) + r" \\")
    L += [r"\bottomrule", r"\end{longtable}"]
    (TABS / "fp_ocsvm.tex").write_text("\n".join(L) + "\n")


def main():
    T = pd.read_csv(C.DADOS / "e1_triagem.csv", parse_dates=["start"])
    T["categoria"] = [categoria(r) if r.classe == "falso_positivo" else "" for r in T.itertuples()]
    ft = C.trips(pd.DatetimeIndex([pd.Timestamp("2024-07-01"), pd.Timestamp("2026-04-20")]))
    fig_causas(T); fig_partida(T); fig_linha_tempo(T, ft); tex_apendice(T)
    E = novidade(); fig_novidade(E)
    N = {"categorias": {m: T[(T.modelo == m) & (T.classe == "falso_positivo")].categoria.value_counts().to_dict()
                        for m in ("OCSVM", "CNN-1D mensal")},
         "classes": {m: T[T.modelo == m].classe.value_counts().to_dict() for m in ("OCSVM", "CNN-1D mensal")},
         "causas": {m: T[T.modelo == m].groupby("classe")[CAUSAS].sum().astype(int).to_dict()
                    for m in ("OCSVM", "CNN-1D mensal")},
         "partida_1h_fp": {m: int(((T.modelo == m) & (T.classe == "falso_positivo") & (T.h_partida <= 1)).sum())
                           for m in ("OCSVM", "CNN-1D mensal")},
         "novidade_mediana": E.groupby("classe")[["N1", "N2"]].median().round(2).to_dict()}
    (AQUI / "numeros.json").write_text(json.dumps(N, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(N, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
