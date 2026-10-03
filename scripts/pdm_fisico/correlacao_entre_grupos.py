#!/usr/bin/env python3
"""Correlação entre os grupos de entrada do detector (subsistemas da máquina e as 3 famílias do modelo).

DIAGNÓSTICO (03/10/2026). Não muda detector nem veredito.

AS PERGUNTAS. (1) Todos os dados de entrada são da máquina? Pelo catálogo (`metadata.csv`) as 36 tags são do
TC-33003A, de subsistemas diferentes, com uma exceção: `PI_5134001` é o header de ar de instrumentação da
U-5134, outra unidade, e está no canal de pressão. `RUNNING_A`, que decide quando o detector opina, está
"não catalogada". (2) Como os grupos se correlacionam entre si, e quanto disso é só o regime de carga?

O MÉTODO.
  · só instantes em regime (RUNNING_A > 0,5 e T5 > 300 °C), a mesma condição do treino;
  · correlação de Spearman (não supõe linearidade, robusta a outlier e a escala);
  · DUAS versões: BRUTA e CONDICIONADA À CARGA. Na condicionada, de cada tag se subtrai a mediana dentro de 30
    faixas de T5 (o proxy de carga já usado em `residuo_carga.py`), o que apaga o que sobe e desce junto só
    porque a máquina muda de carga. T5_AVG_A fica fora dela: é o próprio condicionante;
  · por bloco (par de grupos): média do |rho| entre as tags de um grupo e as do outro;
  · entre as 3 famílias: as correlações CANÔNICAS (CCA, por SVD dos blocos de covariância das variáveis
    padronizadas em posto), que resumem a dependência linear entre dois conjuntos inteiros.

Uso:  PYTHONPATH=. python correlacao_entre_grupos.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from cabiunas_pdm import config as C

AQUI = Path(__file__).resolve().parent
G = {   # subsistema -> tags (ordem = ordem no mapa)
    "Escape da turbina (T5 e ar de exaustão)": ["TC382_01_A", "TC382_02_A", "TC382_03_A", "TC382_04_A", "TC382_05_A", "TC382_06_A",
                                                "T5_AVG_A", "954005_624_TI_0315", "954005_624_TI_0317"],
    "Mancais do compressor (temperatura)": ["954005_624_TI_0301", "954005_624_TI_0303", "954005_624_TI_0305", "954005_624_TI_0307"],
    "Lubrificação (óleo)": ["954005_624_TI_0325", "954005_624_PI_0339", "954005_624_PI_0340", "954005_624_PI_0308", "954005_624_PDI_0338"],
    "Gás de selagem do compressor": ["954005_624_PDI_0302", "954005_624_PDIT_0305", "954005_624_PI_0307", "954005_624_PDI_0301"],
    "Gás combustível e de partida": ["954005_624_PI_0315", "954005_624_PDI_0317", "954005_624_PI_0319"],
    "Ar de instrumentos (outra unidade)": ["PI_5134001"],
    "Vibração (mancais 1 a 5 da turbina)": ["TV_351X_A", "TV_351Y_A", "TV_352X_A", "TV_352Y_A", "TV_353X_A", "TV_353Y_A",
                                             "TV_354X_A", "TV_354Y_A", "TV_355X_A", "TV_355Y_A"],
}
FAM = {"Temperatura": C.TEMPERATURE_TAGS, "Pressão": C.PRESSURE_TAGS, "Vibração": C.VIBRATION_TAGS}
curto = lambda c: c.replace("954005_624_", "")


def dados() -> pd.DataFrame:
    g = pd.read_parquet(AQUI / "grade2min.parquet")
    est = (g["RUNNING_A"] > 0.5) & (g["T5_AVG_A"] > 300)
    return g.loc[est, C.SENSOR_TAGS].astype("float64")


def condiciona(X: pd.DataFrame, nbins: int = 30) -> pd.DataFrame:
    """Subtrai, de cada tag, a mediana dentro da faixa de T5 (a carga). T5_AVG_A vira NaN."""
    faixa = pd.qcut(X["T5_AVG_A"], nbins, duplicates="drop")
    R = X - X.groupby(faixa, observed=True).transform("median")
    R["T5_AVG_A"] = np.nan
    return R


def blocos(M: pd.DataFrame, grupos: dict[str, list[str]]) -> pd.DataFrame:
    nomes = list(grupos)
    B = pd.DataFrame(index=nomes, columns=nomes, dtype=float)
    for a in nomes:
        for b in nomes:
            sub = M.loc[grupos[a], grupos[b]].abs().to_numpy()
            if a == b:
                sub = sub[~np.eye(len(sub), dtype=bool)] if len(sub) > 1 else np.array([np.nan])
            B.loc[a, b] = np.nanmean(sub)
    return B


def canonicas(X: pd.DataFrame, a: list[str], b: list[str], k: int = 3) -> np.ndarray:
    Z = X[a + b].dropna().rank().apply(lambda s: (s - s.mean()) / s.std())
    S = np.cov(Z.to_numpy().T); na = len(a)
    Sxx, Syy, Sxy = S[:na, :na] + 1e-6 * np.eye(na), S[na:, na:] + 1e-6 * np.eye(len(b)), S[:na, na:]
    inv = lambda M: (lambda w, v: (v / np.sqrt(w)) @ v.T)(*np.linalg.eigh(M))
    return np.linalg.svd(inv(Sxx) @ Sxy @ inv(Syy), compute_uv=False)[:k]


def desenha(raw: pd.DataFrame, cond: pd.DataFrame, ordem: list[str], limites: list[int], rotulos: list[str], saida: Path):
    cm = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])   # azul <-> cinza <-> vermelho
    fig, axs = plt.subplots(1, 2, figsize=(18.5, 8.4), facecolor="#fcfcfb")
    fig.subplots_adjust(left=0.22, right=0.93, top=0.90, bottom=0.14, wspace=0.28)
    for ax, M, tit in ((axs[0], raw, "Bruta: inclui o que sobe e desce junto com a carga"),
                       (axs[1], cond, "Condicionada à carga (T5): o que sobra")):
        im = ax.imshow(M.loc[ordem, ordem].to_numpy(), cmap=cm, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xticks(range(len(ordem))); ax.set_yticks(range(len(ordem)))
        ax.set_xticklabels([curto(c) for c in ordem], rotation=90, fontsize=7, color="#52514e")
        ax.set_yticklabels([curto(c) for c in ordem], fontsize=7, color="#52514e")
        for l in limites:
            ax.axhline(l - 0.5, color="#fcfcfb", lw=2.2); ax.axvline(l - 0.5, color="#fcfcfb", lw=2.2)
        ax.set_title(tit, fontsize=11, color="#0b0b0b", loc="left", pad=10)
        for s in ax.spines.values(): s.set_visible(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axs, shrink=0.62, pad=0.015); cb.set_label("correlação de Spearman", color="#52514e"); cb.outline.set_visible(False)
    # rótulos dos grupos no eixo esquerdo do primeiro painel
    ini = [0] + limites
    for i, nome in enumerate(rotulos):
        meio = (ini[i] + (limites + [len(ordem)])[i]) / 2 - 0.5
        axs[0].annotate(nome, xy=(-0.17, meio), xycoords=("axes fraction", "data"), ha="right", va="center", fontsize=9, color="#0b0b0b", annotation_clip=False)
        axs[0].plot([-0.155, -0.155], [ini[i] - 0.4, (limites + [len(ordem)])[i] - 0.6], transform=axs[0].get_yaxis_transform(), color="#52514e", lw=1.2, clip_on=False)
    axs[1].annotate("linha e coluna em branco: T5_AVG_A é o condicionante", xy=(0.0, -0.17), xycoords="axes fraction", fontsize=8, color="#52514e")
    fig.suptitle("TC-33003A: correlação entre as tags de entrada, por subsistema (instantes em regime)", fontsize=13, color="#0b0b0b", x=0.02, ha="left")
    fig.savefig(saida, dpi=130, bbox_inches="tight"); plt.close(fig)


def main():
    pd.set_option("display.width", 220)
    X = dados(); Xc = condiciona(X)
    print(f"{len(X)} instantes em regime, 36 tags (~{len(X) * 2 / 60:.0f} h)\n")
    raw = X.corr(method="spearman"); cond = Xc.corr(method="spearman")
    ordem = [c for g in G.values() for c in g]; limites = list(np.cumsum([len(g) for g in G.values()]))[:-1]
    assert sorted(ordem) == sorted(C.SENSOR_TAGS)

    print("POR SUBSISTEMA: média do |rho| entre as tags de cada par de grupos (bruta | condicionada à carga)")
    nomes = list(G); curtos = [n.split(" (")[0] for n in nomes]
    Br, Bc = blocos(raw, G), blocos(cond, G)
    for nome, B in (("BRUTA", Br), ("CONDICIONADA À CARGA", Bc)):
        D = B.copy(); D.index = curtos; D.columns = [c[:9] for c in curtos]
        print(f"\n{nome}\n" + D.round(2).to_string())
    print("\nPOR FAMÍLIA DO MODELO (T = 14 temperaturas, P = 12 pressões, V = 10 vibrações)")
    fb = {"bruta": blocos(raw, FAM), "condicionada": blocos(cond, FAM)}
    for k, B in fb.items():
        print(f"\n|rho| médio entre famílias, {k}:\n" + B.round(2).to_string())
    print("\nCORRELAÇÕES CANÔNICAS (as 3 primeiras): quão ligados são dois conjuntos inteiros")
    lin = []
    for (na, a), (nb, b) in [(("T", FAM["Temperatura"]), ("P", FAM["Pressão"])), (("T", FAM["Temperatura"]), ("V", FAM["Vibração"])),
                             (("P", FAM["Pressão"]), ("V", FAM["Vibração"]))]:
        lin.append(dict(par=f"{na} x {nb}", bruta=" / ".join(f"{x:.2f}" for x in canonicas(X, a, b)),
                        condicionada=" / ".join(f"{x:.2f}" for x in canonicas(Xc.drop(columns="T5_AVG_A"), [t for t in a if t != "T5_AVG_A"], b))))
    print(pd.DataFrame(lin).to_string(index=False))

    def pares(M, fam_a, fam_b, n=8):
        out = []
        for ta in fam_a:
            for tb in fam_b:
                if ta != tb and np.isfinite(M.loc[ta, tb]): out.append((ta, tb, M.loc[ta, tb]))
        return sorted(out, key=lambda x: -abs(x[2]))[:n]
    for nome, M in (("BRUTA", raw), ("CONDICIONADA", cond)):
        print(f"\nMAIORES |rho| ENTRE FAMÍLIAS DIFERENTES, {nome}")
        todas = []
        for (a, b) in (("Temperatura", "Pressão"), ("Temperatura", "Vibração"), ("Pressão", "Vibração")):
            todas += pares(M, FAM[a], FAM[b], 6)
        for ta, tb, r in sorted(todas, key=lambda x: -abs(x[2]))[:10]:
            print(f"   {curto(ta):12s} x {curto(tb):12s}  rho {r:+.2f}")
    desenha(raw, cond, ordem, limites, curtos, AQUI / "correlacao_entre_grupos.png")
    raw.to_csv(AQUI / "_cache_regua" / "correlacao_bruta.csv"); cond.to_csv(AQUI / "_cache_regua" / "correlacao_condicionada.csv")
    print("\n-> correlacao_entre_grupos.png")


if __name__ == "__main__":
    main()
