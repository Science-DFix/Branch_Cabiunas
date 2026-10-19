#!/usr/bin/env python3
"""Figuras e tabelas do relatório OCSVM x CNN-1D mensal (relatorio_ocsvm_cnn1d_mensal.tex).

Usa os artefatos já treinados no ClearML (produção OCSVM e rodada 1d do CNN-1D mensal), o catálogo de alarmes e o
CSV de sensores do Dataset a97ba56b. Nada é treinado aqui. Saída: figs/*.png, tabs/*.tex e numeros.json.
Uso: python relatorio_ocsvm_cnn1d_mensal/gera_figuras.py
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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "frente_ocsvm"))
import comum as C  # noqa: E402
from a4_variabilidade import canais_da_replica  # noqa: E402
from d1_disputa import PROD, TASKS_JSON  # noqa: E402

FIGS, TABS = AQUI / "figs", AQUI / "tabs"
FIGS.mkdir(exist_ok=True); TABS.mkdir(exist_ok=True)
CSV = Path(C.DADOS / "alarmes_selecionados_turbina_a.csv").resolve().parent / "sensores_full_2024_2026_30s.csv"
INICIO, FIM = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-04-20")
TV = [f"TV_35{i}{e}_A" for i in range(1, 6) for e in "XY"]
TEMP_TAGS = ["TC382_01_A", "TC382_02_A", "TC382_03_A", "TC382_04_A", "TC382_05_A", "TC382_06_A",
             "TI_6240315", "TI_6240317", "T5_AVG_A"]
RECORRENTES = ["PI_6240319_AL", "PAL_6240315"]
COR = {"deteccao": "#1a9850", "inconclusivo": "#f39c12", "falso_positivo": "#d73027"}
NOME = {"deteccao": "detecção", "inconclusivo": "inconclusivo", "falso_positivo": "FP"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 110, "savefig.dpi": 160,
                     "axes.spines.top": False, "axes.spines.right": False})


def carrega():
    reg = json.loads(TASKS_JSON.read_text())
    oc = canais_da_replica(PROD)
    cn = canais_da_replica({c: reg[f"cnn1d_mensal_jan_{c}"] for c in ("temp", "vib", "oleo")}).reindex(oc.index)
    for k in C.CANAIS[:3]:
        cn[k] = cn[k].fillna(False).astype(bool)
    cat = C.catalogo()
    c4 = C.canal_alarme(oc.index, cat.loc[cat.tag.isin(C.TAGS_C4), "t"], C.REF["janela_c4_h"])
    cols = ["data_datetime", "TC382_03_A", "T5_AVG_A", "954005_624_PI_0308", "954005_624_PDIT_0305"] + TV
    s = pd.read_csv(CSV, usecols=cols, parse_dates=["data_datetime"]).set_index("data_datetime").sort_index()
    s = s.loc[pd.Timestamp("2024-07-01"):FIM].apply(pd.to_numeric, errors="coerce")  # "Comm Fail"/"Out of Serv" -> NaN
    sinal = pd.DataFrame({"temp": s["TC382_03_A"], "vib": s[TV].max(axis=1), "oleo": s["954005_624_PI_0308"]})
    return oc, cn, c4, cat, sinal


def episodios(canais3: pd.DataFrame, c4: pd.Series, op: pd.Series, ft: pd.Series):
    ch = {k: canais3[k] for k in C.CANAIS[:3]}; ch["c4"] = c4
    fin = C.decide(ch, 2, C.REF["dur_min"], C.REF["refrat_h"])
    cls, _ = C.avalia(fin, op, ft)
    return cls, C.por_trip(cls, ft).set_index("trip")


def placar(cls, P, op, idx):
    out = {}
    for nome, ini in (("rec", INICIO), ("fora", C.SPLIT)):
        dias = C.compute_operational_period_days(pd.DataFrame({"operational_state": op[idx >= ini]}))
        e = cls[cls.start >= ini]
        fp = int((e.classe == "falso_positivo").sum())
        p = P[P.index >= ini]
        out[nome] = dict(det=int(p.deteccao.notna().sum()), n=len(p), FP=fp, inc=int((e.classe == "inconclusivo").sum()),
                         FPmes=round(fp / (dias / 30.4375), 3), meses=round(dias / 30.4375, 2))
    return out


def composicao(cls, canais3, c4, cat):
    """Por episódio desde 01/2025: canais acesos (>= 50% do episódio), tag que acendeu o canal 4, rajada de temperatura."""
    L = []
    for _, e in cls[cls.start >= INICIO].iterrows():
        w = slice(e.start, e.end)
        acesos = [g for g, k in zip(("temp", "vib", "óleo"), C.CANAIS[:3]) if canais3[k].loc[w].mean() >= 0.5]
        a4 = cat[(cat.t <= e.start) & (cat.t >= e.start - pd.Timedelta(hours=24)) & cat.tag.isin(C.TAGS_C4)]
        tags4 = set(a4.tag)
        viz = cat[(cat.t >= e.start - pd.Timedelta(hours=24)) & (cat.t <= e.end + pd.Timedelta(hours=24))]
        raj = rajadas(viz)
        L.append(dict(start=e.start, end=e.end, classe=e.classe, dur_h=(e.end - e.start).total_seconds() / 3600,
                      c4=bool(c4.loc[w].mean() >= 0.5), modelos="+".join(acesos) if acesos else "nenhum",
                      n_modelos=len(acesos),
                      gatilho=("PI e PAL" if tags4 >= set(RECORRENTES) else "só PI_6240319_AL" if "PI_6240319_AL" in tags4
                               else "só PAL_6240315" if "PAL_6240315" in tags4 else "outra tag" if tags4 else "sem alarme"),
                      tags4=",".join(sorted(tags4)), rajada_temp=len(raj) > 0))
    return pd.DataFrame(L)


def rajadas(cat_sub, janela_min=10, minimo=5):
    """Instantes em que >= `minimo` tags de temperatura distintas vão a ACT em `janela_min` minutos."""
    t = cat_sub[cat_sub.tag.isin(TEMP_TAGS)].sort_values("t")
    out = []
    for i, r in t.iterrows():
        j = t[(t.t >= r.t) & (t.t <= r.t + pd.Timedelta(minutes=janela_min))]
        if j.tag.nunique() >= minimo and (not out or r.t - out[-1] > pd.Timedelta(hours=1)):
            out.append(r.t)
    return out


def marca_trips(ax, ft, rotulo=False):
    for i, t in enumerate(ft, 1):
        ax.axvline(t, color="k", lw=0.8, ls="--", alpha=0.7)
        if rotulo:
            ax.text(t, 1.0, f"T{i}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7,
                    fontweight="bold")


def fig_periodos(ft):
    fig, ax = plt.subplots(figsize=(9, 3.0))
    barras = [
        ("OCSVM produção: normal de treino", "2024-07-01", "2025-07-01", "#4575b4"),
        ("OCSVM produção: pontuação", "2024-07-01", "2026-04-20", "#91bfdb"),
        ("OCSVM mensal (C1): retreinos", "2025-07-01", "2026-04-20", "#fdae61"),
        ("CNN-1D mensal 1c: retreinos", "2025-07-01", "2026-04-20", "#d9ef8b"),
        ("CNN-1D mensal 1d: retreinos", "2025-01-01", "2026-04-20", "#1a9850"),
        ("AutoML: escolha de limiar e debounce", "2025-07-01", "2026-04-20", "#bbbbbb"),
    ]
    for i, (n, a, b, c) in enumerate(barras[::-1]):
        ax.barh(i, pd.Timestamp(b) - pd.Timestamp(a), left=pd.Timestamp(a), color=c, edgecolor="k", lw=0.4)
        ax.text(pd.Timestamp(a), i, " " + n, va="center", fontsize=8)
    ax.set_yticks([])
    for t in ft:
        ax.axvline(t, color="k", lw=0.8, ls="--")
    ax.axvline(C.SPLIT, color="#d73027", lw=1.5)
    ax.text(C.SPLIT, -0.75, " split 01/07/2025", color="#d73027", fontsize=8)
    ax.axvline(INICIO, color="#1a9850", lw=1.2, ls=":")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
    ax.set_xlim(pd.Timestamp("2024-06-15"), pd.Timestamp("2026-05-01"))
    ax.set_title("Períodos de treino, pontuação e escolha do limiar (linhas tracejadas = 8 trips)")
    fig.tight_layout(); fig.savefig(FIGS / "periodos.png"); plt.close(fig)


def fig_visao(oc, cn, cls_o, cls_c, sinal, cat, ft, op):
    """Visão conjunta 01/2025-04/2026: sinal + faixas de pontos anômalos (OCSVM e CNN-1D) + episódios e alarmes."""
    rot = {"temp": ("Temperatura (grupo temp.)\nTC382_03_A", C.CANAIS[0]),
           "vib": ("Vibração de mancal\nmáx. dos 10 TV_*", C.CANAIS[1]),
           "oleo": ("Pressão de óleo\nPI_0308", C.CANAIS[2])}
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1.1]})
    s = sinal.loc[INICIO:FIM]
    on = op.reindex(s.index).eq("on")
    for ax, (g, (lab, k)) in zip(axs, rot.items()):
        x = s[g].where(on).resample("1h").median()
        ax.plot(x.index, x.values, color="#777", lw=0.4)
        lo, hi = np.nanpercentile(x.dropna(), [0.5, 99.5]); amp = hi - lo
        for flag, cor, y, lb in ((oc[k], "#d73027", lo - 0.15 * amp, "OCSVM"), (cn[k], "#2166ac", lo - 0.32 * amp, "CNN-1D mensal")):
            h = (flag.loc[INICIO:FIM] & on).resample("1h").max().fillna(False).astype(bool)
            ax.scatter(h.index[h.values], np.full(int(h.sum()), y), marker="|", s=18, lw=0.5, color=cor,
                       label=f"hora com ponto anômalo — {lb} ({int(h.sum())} h)")
        ax.set_ylim(lo - 0.42 * amp, hi + 0.08 * amp)
        ax.set_ylabel(lab, fontsize=8)
        ax.legend(loc="lower left", fontsize=6.5, frameon=False, ncol=2, bbox_to_anchor=(0, 1.06))
        marca_trips(ax, ft, rotulo=(ax is axs[0]))
        ax.axvline(C.SPLIT, color="#d73027", lw=1.0, alpha=0.6)
    ax = axs[3]
    for cls, y in ((cls_o, 3.2), (cls_c, 2.2)):
        e = cls[cls.start >= INICIO]
        for cl, dy, mk in (("deteccao", 0.3, "^"), ("inconclusivo", 0.0, "v"), ("falso_positivo", -0.3, "v")):
            z = e[e.classe == cl]
            ax.scatter(z.start, [y + dy] * len(z), marker=mk, s=30, color=COR[cl], edgecolor="k", lw=0.3, zorder=3)
    a = cat[(cat.t >= INICIO) & (cat.t <= FIM)]
    for y, tag in ((1.1, "PI_6240319_AL"), (0.5, "PAL_6240315")):
        z = a[a.tag == tag]
        ax.scatter(z.t, [y] * len(z), marker="|", s=25, color="#555", lw=0.6)
    ax.set_yticks([3.2, 2.2, 1.1, 0.5])
    ax.set_yticklabels(["episódios OCSVM", "episódios CNN-1D", "alarme PI_6240319_AL", "alarme PAL_6240315"], fontsize=7)
    ax.set_ylim(0.1, 3.8)
    ax.legend(handles=[Line2D([], [], marker="^" if k == "deteccao" else "v", ls="", color=COR[k], markeredgecolor="k",
                              label=NOME[k]) for k in COR], fontsize=7, ncol=3, loc="lower left", frameon=False,
              bbox_to_anchor=(0, 1.0))
    marca_trips(ax, ft)
    ax.axvline(C.SPLIT, color="#d73027", lw=1.0, alpha=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
    ax.set_xlim(INICIO, FIM)
    fig.suptitle("OCSVM (treino único) × CNN-1D mensal: sinais, pontos anômalos e episódios (01/2025 a 04/2026)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGS / "visao_conjunta.png"); plt.close(fig)


def fig_zoom(arq, titulo, t0, t1, oc, cn, cls_o, cls_c, c4, sinal, cat, ft):
    rot = [("temp", C.CANAIS[0], "TC382_03_A"), ("vib", C.CANAIS[1], "vib. máx. TV_*"),
           ("oleo", C.CANAIS[2], "PI_0308")]
    fig, axs = plt.subplots(5, 1, figsize=(10, 7.4), sharex=True,
                            gridspec_kw={"height_ratios": [1, 1, 1, 0.55, 0.85]})
    s = sinal.loc[t0:t1]
    for ax, (g, k, lab) in zip(axs, rot):
        x = s[g].resample("2min").median()
        ax.plot(x.index, x.values, color="#666", lw=0.6)
        antes = [t for t in ft if t0 <= t <= t1]
        ligada = oc["operational_state"].loc[t0:t1].eq("on").resample("2min").min().reindex(x.index).fillna(False).astype(bool)
        ref = x[ligada]  # escala só com a máquina ligada e antes do trip (paradas e a queda do trip achatam o resto)
        ref = ref.loc[:min(antes)] if antes else ref
        lo, hi = np.nanpercentile(ref.dropna(), [0.5, 99.5]) if ref.notna().any() else (0, 1)
        amp = hi - lo + 1e-9
        # duas faixas abaixo do sinal: pontos anômalos da OCSVM e do CNN-1D (no sinal eles se escondem um no outro)
        for flag, cor, y, lb in ((oc[k], "#d73027", lo - 0.18 * amp, "OCSVM"), (cn[k], "#2166ac", lo - 0.36 * amp, "CNN-1D mensal")):
            f = flag.loc[t0:t1].resample("2min").max().fillna(False).astype(bool)
            ax.scatter(f.index[f.values], np.full(int(f.sum()), y), marker="|", s=30, lw=0.8, color=cor,
                       label=f"ponto anômalo {lb}")
        ax.set_ylim(lo - 0.45 * amp, hi + 0.1 * amp)
        ax.set_ylabel(lab, fontsize=8)
    axs[0].legend(loc="upper left", fontsize=7, frameon=False, ncol=2)
    ax = axs[3]
    w = c4.loc[t0:t1]
    ax.fill_between(w.index, 0, w.astype(float).values, step="post", color="#999", alpha=0.5)
    a = cat[(cat.t >= t0) & (cat.t <= t1)]
    cores_tag = {"PI_6240319_AL": "#7b3294", "PAL_6240315": "#008837"}
    # temperatura primeiro e em faixa própria (a rajada costuma cair no mesmo minuto de PI_6240319_AL)
    for _, r in a.sort_values("tag", key=lambda c: c.isin(TEMP_TAGS), ascending=False).iterrows():
        if r.tag in TEMP_TAGS:
            ax.plot([r.t, r.t], [0.55, 1.0], color="#e66101", lw=0.9)
        else:
            ax.plot([r.t, r.t], [0, 0.48 if r.tag in C.TAGS_C4 else 0.25], color=cores_tag.get(r.tag, "#333"),
                    lw=1.0 if r.tag in C.TAGS_C4 else 0.7)
    ax.set_yticks([]); ax.set_ylabel("canal 4\n+ alarmes", fontsize=8)
    leg = [Patch(color="#999", alpha=0.5, label="canal 4 aceso"),
           Line2D([], [], color="#7b3294", label="PI_6240319_AL"), Line2D([], [], color="#008837", label="PAL_6240315"),
           Line2D([], [], color="#333", label="outra tag (mais alta = do canal 4)"),
           Line2D([], [], color="#e66101", label="tag de temperatura (metade de cima)")]
    ax.legend(handles=leg, fontsize=6.5, ncol=5, loc="lower left", frameon=False, bbox_to_anchor=(0, 1.0))
    ax = axs[4]
    for y, cls, lb in ((1, cls_o, "OCSVM"), (0, cls_c, "CNN-1D")):
        z = cls[(cls.end >= t0) & (cls.start <= t1)]
        for _, e in z.iterrows():
            ax.barh(y, max(e.end - e.start, pd.Timedelta(minutes=40)), left=e.start, height=0.6,
                    color=COR[e.classe], edgecolor="k", lw=0.3)
    ax.set_yticks([1, 0]); ax.set_yticklabels(["episódios OCSVM", "episódios CNN-1D"], fontsize=8)
    ax.set_ylim(-0.6, 1.6)
    ax.legend(handles=[Patch(color=COR[k], label=NOME[k]) for k in COR], fontsize=7, ncol=3, loc="lower left",
              frameon=False, bbox_to_anchor=(0, 1.0))
    for a_ in axs:
        for t in ft:
            if t0 <= t <= t1:
                a_.axvline(t, color="k", lw=1.2, ls="--")
    axs[0].text(0.995, 1.02, "linha tracejada = trip", transform=axs[0].transAxes, ha="right", fontsize=7)
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n%H:%M"))
    axs[-1].set_xlim(t0, t1)
    fig.suptitle(titulo, fontsize=10)
    fig.tight_layout(); fig.savefig(FIGS / arq); plt.close(fig)


def fig_antecedencia(Po, Pc):
    fig, ax = plt.subplots(figsize=(8, 3.0))
    x = np.arange(len(Po))
    ax.bar(x - 0.2, Po.lead_h.fillna(0), 0.4, color="#d73027", label="OCSVM (treino único)")
    ax.bar(x + 0.2, Pc.lead_h.fillna(0), 0.4, color="#2166ac", label="CNN-1D mensal (1d)")
    for i, (a, b) in enumerate(zip(Po.lead_h, Pc.lead_h)):
        ax.text(i - 0.2, (a if a == a else 0) + 0.8, f"{a:.1f}" if a == a else "—", ha="center", fontsize=7)
        ax.text(i + 0.2, (b if b == b else 0) + 0.8, f"{b:.1f}" if b == b else "—", ha="center", fontsize=7)
    ax.axhline(4, color="#555", lw=0.6, ls=":"); ax.text(len(Po) - 0.5, 4.5, "4 h", fontsize=7, ha="right")
    ax.axvline(4.5, color="#d73027", lw=1.0, alpha=0.6)
    ax.text(2, 47, "dentro da amostra da OCSVM", ha="center", fontsize=8)
    ax.text(6, 47, "fora da amostra", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([t.strftime("%d/%m/%y") for t in Po.index], fontsize=8)
    ax.set_ylabel("antecedência (h)"); ax.set_ylim(0, 52)
    ax.legend(fontsize=8, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.9))
    fig.tight_layout(); fig.savefig(FIGS / "antecedencia.png"); plt.close(fig)


def fig_fp_mensal(cls_o, cls_c):
    fig, ax = plt.subplots(figsize=(8, 2.8))
    meses = pd.period_range(INICIO, FIM, freq="M")
    def conta(cls):
        z = cls[(cls.classe == "falso_positivo") & (cls.start >= INICIO)]
        return z.start.dt.to_period("M").value_counts().reindex(meses, fill_value=0)
    a, b = conta(cls_o), conta(cls_c)
    x = np.arange(len(meses))
    ax.bar(x - 0.2, a.values, 0.4, color="#d73027", label=f"OCSVM ({a.sum()} FP)")
    ax.bar(x + 0.2, b.values, 0.4, color="#2166ac", label=f"CNN-1D mensal ({b.sum()} FP)")
    ax.set_xticks(x); ax.set_xticklabels([m.strftime("%m/%y") for m in meses], rotation=90, fontsize=7)
    ax.axvline(5.5, color="#d73027", lw=1.0, alpha=0.6)
    ax.set_ylabel("episódios de FP"); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(FIGS / "fp_mensal.png"); plt.close(fig)


def fig_composicao(comp):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    ordem = ["deteccao", "inconclusivo", "falso_positivo"]
    m = pd.crosstab(comp.classe, comp.modelos).reindex(ordem).fillna(0)
    m = m[m.sum().sort_values(ascending=False).index]
    m.index = [NOME[k] for k in m.index]
    m.plot(kind="barh", stacked=True, ax=axs[0], colormap="tab20", edgecolor="k", lw=0.3)
    axs[0].set_title("Modelos acesos (≥ 50% do episódio) junto do canal 4"); axs[0].set_xlabel("episódios")
    axs[0].legend(fontsize=7, frameon=False, title="modelo(s)", title_fontsize=7)
    g = pd.crosstab(comp.classe, comp.gatilho).reindex(ordem).fillna(0)
    g = g[[c for c in ["PI e PAL", "só PI_6240319_AL", "só PAL_6240315", "outra tag", "sem alarme"] if c in g]]
    g.index = [NOME[k] for k in g.index]
    g.plot(kind="barh", stacked=True, ax=axs[1], color=["#542788", "#8073ac", "#b2abd2", "#e08214", "#ccc"][:g.shape[1]],
           edgecolor="k", lw=0.3)
    axs[1].set_title("Tag do canal 4 nas 24 h antes do episódio"); axs[1].set_xlabel("episódios")
    axs[1].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(FIGS / "composicao_episodios.png"); plt.close(fig)


def fig_recorrencia(rec):
    r = rec.head(16)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(r.index, r.n, color="#bbb", edgecolor="k", lw=0.3, label="ativações desde 07/2024")
    ax.barh(r.index, r.n_trip, color="#d73027", label="das quais em [T−48 h, T+24 h] de algum trip")
    for i, (n, k) in enumerate(zip(r.n, r.n_trip)):
        ax.text(n + 4, i, f"{k}/{n} ({100 * k / n:.0f}%)", va="center", fontsize=7)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() in C.TAGS_C4:
            lbl.set_fontweight("bold")
    ax.set_xlabel("ativações (ACT)"); ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("Recorrência das tags do catálogo (negrito = 5 tags do canal 4)")
    fig.tight_layout(); fig.savefig(FIGS / "recorrencia_tags.png"); plt.close(fig)


def tex_tabela(df, arq, fmt=None, longa=False, colunas=None):
    """Tabela booktabs simples (o to_latex do pandas exige jinja2); `longa` = longtable com cabeçalho repetido."""
    fmt = fmt or (lambda v: f"{v:.1f}".replace(".", ","))
    esc = lambda x: str(x).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    cel = lambda v: fmt(v) if isinstance(v, float) else esc(v)
    amb = "longtable" if longa else "tabular"
    cab = [r"\toprule", " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    if longa:
        cab = cab + [r"\endhead"]
    L = [r"\begin{" + amb + "}{" + (colunas or "l" * df.shape[1]) + "}"] + cab + [" & ".join(cel(v) for v in r) + r" \\"
                                                                    for r in df.itertuples(index=False)]
    L += [r"\bottomrule", r"\end{" + amb + "}"]
    (TABS / arq).write_text("\n".join(L) + "\n")


def main():
    cache = AQUI / "cache.pkl"  # fora do git; apague para recarregar do ClearML e do CSV
    if cache.exists():
        oc, cn, c4, cat, sinal = pd.read_pickle(cache)
    else:
        oc, cn, c4, cat, sinal = carrega()
        pd.to_pickle((oc, cn, c4, cat, sinal), cache)
    sinal = sinal.apply(pd.to_numeric, errors="coerce")  # o CSV tem textos ocasionais nas colunas de sensor
    idx, op = oc.index, oc["operational_state"]
    ft = C.trips(idx)
    cls_o, Po = episodios(oc, c4, op, ft)
    cls_c, Pc = episodios(cn, c4, op, ft)
    N = {"ocsvm": placar(cls_o, Po, op, idx), "cnn": placar(cls_c, Pc, op, idx)}
    sel_on = (idx >= INICIO) & op.eq("on").values
    N["duty_on_rec"] = {"c4": round(float(c4[sel_on].mean()), 3),
                        **{f"ocsvm_{g}": round(float(oc[k][sel_on].mean()), 4) for g, k in zip(("temp", "vib", "oleo"), C.CANAIS[:3])},
                        **{f"cnn_{g}": round(float(cn[k][sel_on].mean()), 4) for g, k in zip(("temp", "vib", "oleo"), C.CANAIS[:3])}}
    # paradas >= 2 h desde 01/2025 (máquina sai de "on" por 2 h ou mais)
    o = op[idx >= INICIO].eq("on")
    blocos = (o != o.shift()).cumsum()
    off = o[~o].groupby(blocos[~o]).agg(lambda s: (s.index[-1] - s.index[0]).total_seconds() / 3600 + 1 / 120)
    N["paradas_2h_rec"] = int((off >= 2).sum())
    # recorrência das tags desde 07/2024
    c = cat[cat.t >= pd.Timestamp("2024-07-01")].copy()
    c["perto"] = c.t.map(lambda t: bool(((t >= ft - pd.Timedelta(hours=48)) & (t <= ft + pd.Timedelta(hours=24))).any()))
    rec = c.groupby("tag").agg(n=("t", "size"), n_trip=("perto", "sum")).sort_values("n", ascending=False)
    N["catalogo"] = {"ativacoes_total": int(len(cat)), "tags": int(cat.tag.nunique()),
                     "ativacoes_desde_0724": int(len(c)), "ini": str(cat.t.min()), "fim": str(cat.t.max())}
    N["recorrencia"] = {t: [int(rec.loc[t, "n"]), int(rec.loc[t, "n_trip"])] for t in rec.index[:12]}
    raj = rajadas(cat[cat.t >= pd.Timestamp("2024-07-01")])
    N["rajadas_temp"] = [str(t) for t in raj]
    comp_o = composicao(cls_o, oc, c4, cat)
    comp_c = composicao(cls_c, cn, c4, cat)
    N["comp_ocsvm"] = {"n": len(comp_o), "com_c4": int(comp_o.c4.sum()),
                       "classes": comp_o.classe.value_counts().to_dict(),
                       "gatilho_por_classe": pd.crosstab(comp_o.classe, comp_o.gatilho).to_dict(),
                       "modelos_por_classe": pd.crosstab(comp_o.classe, comp_o.modelos).to_dict(),
                       "rajada_por_classe": comp_o.groupby("classe").rajada_temp.sum().astype(int).to_dict()}
    N["leads"] = {"ocsvm": Po.lead_h.to_dict(), "cnn": Pc.lead_h.to_dict()}
    N["leads"] = {k: {str(t): (None if v != v else v) for t, v in d.items()} for k, d in N["leads"].items()}
    (AQUI / "numeros.json").write_text(json.dumps(N, indent=2, ensure_ascii=False, default=str))
    comp_o.to_csv(AQUI / "tabs" / "episodios_ocsvm.csv", index=False)

    # tabelas .tex
    t = comp_o.copy()
    t["início"] = t.start.dt.strftime("%d/%m/%y %H:%M"); t["dur. (h)"] = t.dur_h
    t["classe"] = t.classe.map(NOME); t["tags do canal 4 (24 h antes)"] = t.tags4.str.replace(",", ", ")
    t["rajada"] = t.rajada_temp.map({True: "sim", False: ""})
    tex_tabela(t[["início", "dur. (h)", "classe", "modelos", "tags do canal 4 (24 h antes)", "rajada"]],
               "episodios_ocsvm.tex", longa=True, colunas=r"llll >{\raggedright\arraybackslash}p{6.2cm} l")
    r = rec.head(16).reset_index()
    r["% perto de trip"] = (100 * r.n_trip / r.n).astype(float)
    r["canal 4"] = r.tag.isin(C.TAGS_C4).map({True: "sim", False: ""})
    r.columns = ["tag", "ativações", "perto de trip", "% perto de trip", "canal 4"]
    tex_tabela(r, "recorrencia.tex", fmt=lambda v: f"{v:.0f}")

    # figuras
    fig_periodos(ft)
    fig_visao(oc, cn, cls_o, cls_c, sinal, cat, ft, op)
    fig_antecedencia(Po, Pc)
    fig_fp_mensal(cls_o, cls_c)
    fig_composicao(comp_o)
    fig_recorrencia(rec)
    for i, T in enumerate(ft, 1):
        fig_zoom(f"zoom_trip_{i}.png", f"Trip {i}: {T.strftime('%d/%m/%Y %H:%M')} — janela de 60 h antes a 6 h depois",
                 T - pd.Timedelta(hours=60), T + pd.Timedelta(hours=6), oc, cn, cls_o, cls_c, c4, sinal, cat, ft)
    exemplos = [("zoom_fp_chattering.png", "FP em série: vibração + PI_6240319_AL recorrente (31/01 a 13/02/2026)",
                 "2026-01-31 00:00", "2026-02-13 12:00"),
                ("zoom_fp_rajada.png", "FP com rajada de alarmes de temperatura (24/07/2025)",
                 "2025-07-23 06:00", "2025-07-26 06:00"),
                ("zoom_fp_longo.png", "FP longo: vibração + PAL_6240315/PDAL_6240302 (15/11/2025)",
                 "2025-11-14 06:00", "2025-11-17 06:00"),
                ("zoom_inconclusivo.png", "Inconclusivo: temperatura + PI_6240319_AL, seguido de parada (17/08/2025)",
                 "2025-08-16 06:00", "2025-08-19 18:00")]
    for arq, tit, a, b in exemplos:
        fig_zoom(arq, tit, pd.Timestamp(a), pd.Timestamp(b), oc, cn, cls_o, cls_c, c4, sinal, cat, ft)
    print(json.dumps({k: N[k] for k in ("ocsvm", "cnn", "duty_on_rec", "paradas_2h_rec", "catalogo", "rajadas_temp")},
                     indent=1, ensure_ascii=False))
    print(json.dumps(N["comp_ocsvm"], indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
