#!/usr/bin/env python3
"""Perícia dos episódios fora de janela de trip, no ponto de deploy atual.

POR QUE REFAZER. A perícia anterior (`pericia_fp_v2.py`) foi feita sobre outro
ponto de operação e tinha um erro de contagem do nível A, achado pela auditoria.
E as estatísticas que guiavam os experimentos -- "9 de 15 nascem na borda do
blackout", "o vb é essencial em 5 de 6" -- são de contagens antigas. No ponto
atual (v2, k_lo de deploy, retreino no dia 1) são 4 FP e 7 NEUTRO pela Regra C.

O QUE SE MEDE, por episódio (FP, NEUTRO e, para contraste, TP):

  · horas desde o último religamento no nascimento, e se cai na borda do
    blackout (6,4667 h cravado -- o primeiro instante que a máscara libera);
  · que nível disparou (A sensível, B específico) e quais canais estavam
    acesos no nascimento;
  · que sonda dirige o `vb` nas 2 h antes do nascimento;
  · que canais são ESSENCIAIS: o detector é rodado de novo com cada canal
    desligado, e o canal é essencial se o episódio deixa de existir.

A ordem das 10 sondas no cache do `vb` é CONFERIDA contra o dado bruto, porque o
script que gerou o cache não existe mais.

Uso:  PYTHONPATH=. python pericia_fp_atual.py
"""
from __future__ import annotations
import io, contextlib
import numpy as np, pandas as pd
import regua_fp as R
with contextlib.redirect_stdout(io.StringIO()):
    import pos_processamento as PP
    from publica_clearml import VIBRATION_TAGS

BORDA_H, TOL_H = 6.466667, 0.05
DIA = 1


def ordem_das_sondas() -> float:
    """max|dif| entre o cache e a grade bruta, coluna a coluna."""
    z = np.load("piso_fisico_cache.npz")
    hot, Xh = z["hot"], z["Xh"]
    G = PP.g[VIBRATION_TAGS].to_numpy()[hot]
    return float(np.nanmax(np.abs(G - Xh)))


def sonda_dominante() -> tuple[pd.DataFrame, np.ndarray]:
    """Z de cada sonda nos instantes quentes, como o vb o calcula."""
    z = np.load("piso_fisico_cache.npz")
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = np.abs((z["Xh"] - z["MED"]) / z["S"])
    Zf = np.full((len(R.idx), Z.shape[1]), np.nan)
    Zf[z["hot"]] = np.where(np.isfinite(Z), Z, np.nan)
    return pd.DataFrame(Zf, index=R.idx, columns=VIBRATION_TAGS), z["hot"]


def horas_desde_religamento() -> pd.Series:
    part = PP.part.to_numpy()
    ult = pd.Series(np.where(part, np.arange(len(part)), np.nan)).ffill().to_numpy()
    h = (np.arange(len(part)) - ult) * 2 / 60
    return pd.Series(h, index=R.idx)


def sobrevive(eps_ref, fin_var) -> list[bool]:
    return [bool(fin_var.loc[a:b].any()) for a, b in eps_ref]


def main():
    dif = ordem_das_sondas()
    print(f"ordem das sondas no cache do vb: max|dif| contra a grade bruta = {dif:.3e}"
          f"  -> {'CONFERE' if dif < 1e-6 else 'NÃO CONFERE'}\n")

    t, p, ms, ds = R.sinais(DIA)
    d = R.detector(t, p, ms, ds)
    fin = d["fin"]
    eps = R.AV.episodios(fin)
    cls = R.DB.classifica_regra_c(eps, R.PARADAS)
    ZS, _ = sonda_dominante()
    hrel = horas_desde_religamento()

    # canal essencial: roda sem ele e vê se o episódio sobrevive
    sem = {c: sobrevive(eps, R.detector(t, p, ms, ds, desliga=(c,))["fin"]) for c in R.SIN}

    linhas = []
    for k, ((a, b, kind, lead), ep) in enumerate(zip(cls, eps)):
        hr = float(hrel.loc[a])
        janela = ZS.loc[a - pd.Timedelta(hours=2):a]
        media = janela.mean()
        sonda = media.idxmax() if media.notna().any() else None
        linhas.append(dict(
            classe=kind, inicio=a.strftime("%Y-%m-%d %H:%M"),
            horas=round((b - a).total_seconds() / 3600 + 2 / 60, 1),
            h_religa=round(hr, 2), borda=abs(hr - BORDA_H) <= TOL_H,
            nivel=("A+B" if d["vA"].loc[a] and d["vB"].loc[a]
                   else "A" if d["vA"].loc[a] else "B" if d["vB"].loc[a] else "-"),
            acesos_A="".join(c[0] if c != "sp" else "s" for c in R.SIN if d["A"][c].loc[a]),
            acesos_B="".join(c[0] if c != "sp" else "s" for c in R.SIN if d["B"][c].loc[a]),
            sonda_vb=(sonda.replace("TV_", "").replace("_A", "") if sonda else "-"),
            forca_max=round(float(d["F"].loc[a:b].max()), 2),
            essenciais=",".join(c for c in R.SIN if not sem[c][k]) or "nenhum"))
    T = pd.DataFrame(linhas)
    pd.set_option("display.width", 180)
    for kind in ("FP", "NEUTRO", "TP"):
        S = T[T.classe == kind].drop(columns="classe")
        print(f"{kind}  ({len(S)} episódios, {S.horas.sum():.0f} h no total)")
        print(S.to_string(index=False))
        print()

    print("RESUMO")
    for kind in ("FP", "NEUTRO", "TP"):
        S = T[T.classe == kind]
        if not len(S):
            continue
        ess = {c: int(S.essenciais.str.contains(c).sum()) for c in R.SIN}
        print(f"  {kind:6s} n={len(S):2d}  na borda do blackout: {int(S.borda.sum())}"
              f"  |  nível no nascimento: {S.nivel.value_counts().to_dict()}"
              f"  |  canal essencial em: {ess}"
              f"  |  sondas: {S.sonda_vb.value_counts().to_dict()}")
    T.to_csv(R.CACHE / "pericia_fp_atual.csv", index=False)


if __name__ == "__main__":
    main()
