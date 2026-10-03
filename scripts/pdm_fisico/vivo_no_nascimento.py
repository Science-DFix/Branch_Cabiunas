#!/usr/bin/env python3
"""No nascimento de um episódio, a evidência VIVA separa FP de TP?

DIAGNÓSTICO A POSTERIORI (03/10/2026), com os 54 episódios distintos (25 FP, 15 NEUTRO, 14 TP) das 8
composições (`anatomia_fp_p.py`) -- a primeira vez que há massa para olhar o episódio, e não só o
dia 1 (6 FP e 8 TP). Não é candidato: decide se vale pré-registrar uma regra de confirmação.

A QUESTÃO. Um canal "aceso" pode estar aceso pela evidência de agora (EWMA acima do limiar A) ou só
pela memória do CUSUM (o EWMA já abaixo). Em 57% das horas de carga nenhum voto se forma com canais
vivos. Mas no NASCIMENTO? Por episódio, nas 2 h de confirmação: quantos canais estão acesos, quantos
vivos (EWMA > limiar A em >= metade das amostras) e quantos só por memória.

COMO SE JULGA. Os episódios das 8 composições se repetem (os mesmos dias), então se agrupa por início a
menos de 12 h (eventos distintos) e se compara TP contra FP por Mann-Whitney e pela AUC. Um canal
que só separa em um evento não é regra. Sem correção, 4 canais x 3 medidas = 12 testes: o limite de
Bonferroni é p < 0,004.

RESULTADO (03/10/2026) -- A EVIDÊNCIA VIVA NÃO SEPARA; A FORÇA SÓ SUGERE.
  54 episódios distintos: FP 25, NEUTRO 15, TP 14. No nascimento os canais acesos são VIVOS em todas as
  classes (canais só por memória: mediana 0 em FP, NEUTRO e TP): a memória segura a cauda, não o início.
    medida          TP    FP    AUC     p
    n_aceso         3     2    0,62   0,19
    n_vivo          2     2    0,61   0,24
    força           8,1   3,5  0,70   0,045
    razão do vb     2,7   1,2  0,72   0,025
  Nenhuma passa Bonferroni (12 testes, p < 0,004); a força e a razão do vb são nominais, com poder ~40%
  para AUC 0,7 neste n. É a MESMA direção do "piso de força" de 19/09 (FP com força baixa), agora em 25 FP
  distintos e não nos 4 de então -- mas os mesmos dias, não dados novos.
  CURVA DA FORÇA (episódios distintos; θ = piso aplicado à força nas 2 h de confirmação):
     θ    TP mantidos   FP removidos   NEUTRO removidos
    2,0      13/14          11/25            2/15
    3,0      11/14          12/25            4/15
    6,0      10/14          17/25            6/15
  Os FP se amontoam em força 1,0-1,4 (10 de 25) e há TP em 1,2, 2,3 e 2,5: a sobreposição é grande, e
  as maiores forças de FP (35 a 170) são os episódios longos, que um piso não toca.

Uso:  PYTHONPATH=. python vivo_no_nascimento.py
"""
from __future__ import annotations
import io, contextlib
import numpy as np, pandas as pd
from scipy import stats

with contextlib.redirect_stdout(io.StringIO()):
    import regua_fp as R

DB = R.DB
H2 = pd.Timedelta(hours=2)
LIM = {c: DB.BASE[c] * DB.K_LO[c] for c in R.SIN}


def main():
    pd.set_option("display.width", 220)
    L = []
    for d in R.DIAS:
        out = R.detector(*R.sinais(d)); cls = R.mede(out["fin"])["cls"]
        for a, b, k, _ in cls:
            w = slice(a, a + H2)
            r = dict(dia=d, a=a, classe=k, dur_h=(b - a).total_seconds() / 3600)
            n_aceso = n_vivo = 0
            for c in R.SIN:
                aceso = bool(out["A"][c].loc[w].mean() >= 0.5)
                vivo = bool((out["EW"][c].loc[w] > LIM[c]).mean() >= 0.5)
                r[f"aceso_{c}"], r[f"vivo_{c}"] = int(aceso), int(vivo)
                r[f"razao_{c}"] = float((out["EW"][c].loc[w] / LIM[c]).median())
                n_aceso += aceso; n_vivo += aceso and vivo
            r.update(n_aceso=n_aceso, n_vivo=n_vivo, n_memoria=n_aceso - n_vivo,
                     forca=float(out["F"].loc[w].max()))
            L.append(r)
        print(f"composição {d:2d} ok", flush=True)
    T = pd.DataFrame(L).sort_values("a"); grupo, ult = [], None
    for a in T.a:
        grupo.append(len(set(grupo)) if ult is None or (a - ult) > pd.Timedelta(hours=12) else grupo[-1]); ult = a
    T["evento"] = grupo
    num = [c for c in T.columns if c.startswith(("aceso_", "vivo_", "razao_")) or c in ("n_aceso", "n_vivo", "n_memoria", "forca", "dur_h")]
    U = T.groupby("evento").agg(classe=("classe", lambda s: s.mode().iloc[0]), **{c: (c, "median") for c in num})
    U.to_csv(R.CACHE / "vivo_no_nascimento.csv")
    print(f"\n{len(U)} episódios distintos: " + ", ".join(f"{k} {v}" for k, v in U.classe.value_counts().items()))
    print("\nMEDIANA POR CLASSE (episódios distintos)")
    print(U.groupby("classe")[["n_aceso", "n_vivo", "n_memoria", "forca", "dur_h"]].median().round(2).to_string())
    print("\nSEPARA TP DE FP? Mann-Whitney (TP contra FP) e AUC (0,5 = nada); Bonferroni p < 0,004")
    tp, fp = U[U.classe == "TP"], U[U.classe == "FP"]
    linhas = []
    for c in ["n_aceso", "n_vivo", "n_memoria", "forca"] + [f"vivo_{s}" for s in R.SIN] + [f"razao_{s}" for s in R.SIN]:
        u, p = stats.mannwhitneyu(tp[c], fp[c], alternative="two-sided")
        linhas.append(dict(medida=c, mediana_TP=tp[c].median(), mediana_FP=fp[c].median(), AUC=u / (len(tp) * len(fp)), p=p))
    print(pd.DataFrame(linhas).round(3).to_string(index=False))
    print(f"\nTP: {len(tp)} | FP: {len(fp)} | NEUTRO: {int((U.classe == 'NEUTRO').sum())}")


if __name__ == "__main__":
    main()
