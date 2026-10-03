#!/usr/bin/env python3
"""Como o canal p se comporta no nascimento de um FP e de um TP?

DIAGNÓSTICO A POSTERIORI (03/10/2026). Não é candidato e não muda veredito: serve para decidir se
há uma hipótese fundamentada para um canal p mais robusto, antes de escrever qualquer variante.

POR QUE O p. No ponto publicado o p participa de 5 dos 6 FP (t+p+vb, p+vb, p+sp), e as duas
variantes que cortaram FP nesta rodada mexiam nele (H2 -25%, C1 -20%) e perderam nascimentos.
`escala_degenerada.py` mostrou que o p é o máximo de desvios por tag ao quadrado, com salto de nível
de uma única tag (PDI_0302, PDI_0301, PDIT_0305) dando 100x a 2.000x o limiar.

A PERGUNTA. No nascimento (as 2 h que o episódio leva para ser confirmado), o p que acende o
episódio vem de UMA tag dominante ou de várias? CONCENTRAÇÃO = maior erro por tag / soma dos três
maiores (1,0 = uma tag só; 0,33 = três iguais). Em FP, TP e NEUTRO, nas 8 composições.

O CONTROLE: o que a concentração de uma tag diz sobre FP só vale se o TP não tiver a mesma. A
mesma medida no t, que não tem o problema de escala, serve de comparação.

RESULTADO (03/10/2026) -- A HIPÓTESE DO SALTO DE UMA TAG NÃO SE SUSTENTA.
  Episódios nas 8 composições: FP 74, NEUTRO 53, TP 65 (54 distintos: FP 25, NEUTRO 15, TP 14).
    classe   p participa   concentração (mediana)   > 0,8   erro do p no nascimento (x limiar A)
    FP           99%               0,51               19%               0,3
    NEUTRO      100%               0,54               15%               0,4
    TP           95%               0,55               27%               0,6
  · O p está aceso em quase todo episódio, de qualquer classe: não discrimina. FP e TP têm a mesma
    concentração (o TP, até um pouco mais), e o erro cru do p no nascimento é baixo nas duas (0,3x e
    0,6x): quem o acende é o EWMA de picos ocasionais, não uma tag em salto de nível. Sem apoio para
    um p estruturalmente mais robusto (agregador top-2, autoescala por regime, compressão do score).
  · ONDE ESTÁ A CARGA: dos 127 episódios FP+NEUTRO (12.390 h), os > 100 h são 32% dos episódios e
    carregam 81% das horas; os > 200 h, 10% e 51%. Por composição, 59 a 88%. Dos FP, 37 de 74 têm
    menos de 24 h. A carga em horas é um problema de episódio LONGO; a contagem, de episódio curto.

Uso:  PYTHONPATH=. python anatomia_fp_p.py
"""
from __future__ import annotations
import io, contextlib
import numpy as np, pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    import regua_fp as R
    import escala_degenerada as E

AO, DF, IX = E.AO, E.DF, E.IX
DB = R.DB
FAMS = {"t": E.FAM["t"], "p": E.FAM["p"]}
H2 = pd.Timedelta(hours=2)


def matriz(dia: int, fam: str) -> np.ndarray:
    cols = FAMS[fam]; Q = np.full((len(IX), len(cols)), np.nan)
    for c0, c1 in E.bundles(dia):
        sct, scp, _, _ = AO.bundle(c0)
        s = (IX >= c0) & (IX < c1)
        Q[s] = E.por_sensor(sct if fam == "t" else scp, DF.loc[s, cols])
    return Q


def concentracao(Q: np.ndarray, a: pd.Timestamp, fam: str):
    """(tag dominante, share do maior entre os 3 maiores, erro máximo em x o limiar A), mediana nas 2 h."""
    i0, i1 = IX.searchsorted(a), IX.searchsorted(a + H2)
    w = Q[i0:i1]
    if not len(w) or not np.isfinite(w).any():
        return None, np.nan, np.nan
    med = np.nanmedian(w, axis=0)
    o = np.argsort(med)[::-1]
    top3 = med[o[:3]].sum()
    return E.curto(FAMS[fam][o[0]]), float(med[o[0]] / top3) if top3 > 0 else np.nan, float(med[o[0]] / E.LIM[fam])


def main():
    pd.set_option("display.width", 230)
    linhas = []
    for d in R.DIAS:
        out = R.detector(*R.sinais(d)); fin = out["fin"]; cls = R.mede(fin)["cls"]
        Q = {f: matriz(d, f) for f in FAMS}
        for a, b, k, _ in cls:
            lit = "".join(c for c in R.SIN if bool(out["A"][c].loc[a:a + H2].any()) or bool(out["B"][c].loc[a:a + H2].any()))
            r = dict(dia=d, a=a, dur_h=(b - a).total_seconds() / 3600 + 2 / 60, classe=k, canais=lit)
            for f in FAMS:
                tag, share, mag = concentracao(Q[f], a, f)
                r.update({f"{f}_tag": tag, f"{f}_share": share, f"{f}_x_lim": mag})
            linhas.append(r)
        print(f"composição {d:2d} ok", flush=True)
    T = pd.DataFrame(linhas); T.to_csv(R.CACHE / "anatomia_fp_p.csv", index=False)
    T["p_aceso"] = T.canais.str.contains("p"); T["t_aceso"] = T.canais.str.contains("t")

    print("\nPOR CLASSE (todas as composições; episódios, não eventos distintos)")
    g = T.groupby("classe")
    print(pd.DataFrame({"n": g.size(), "dur_mediana_h": g.dur_h.median().round(1), "p_participa_%": (100 * g.p_aceso.mean()).round(0),
                        "t_participa_%": (100 * g.t_aceso.mean()).round(0)}).to_string())
    for f in ("p", "t"):
        x = T[T[f"{f}_aceso"]]
        gg = x.groupby("classe")
        print(f"\n{f.upper()}: nos episódios em que o {f} participa -- concentração (1,0 = uma tag só) e erro máximo em x o limiar A")
        print(pd.DataFrame({"n": gg.size(), "share_mediana": gg[f"{f}_share"].median().round(2), "share_>0,8_%": (100 * gg[f"{f}_share"].apply(lambda s: (s > 0.8).mean())).round(0),
                            "erro_x_lim_mediana": gg[f"{f}_x_lim"].median().round(1), "erro_>100x_%": (100 * gg[f"{f}_x_lim"].apply(lambda s: (s > 100).mean())).round(0)}).to_string())
        print("  tag dominante (% dos episódios com o canal):")
        for k, h in x.groupby("classe"):
            print(f"    {k:7s} " + "  ".join(f"{t} {100 * v:.0f}%" for t, v in h[f"{f}_tag"].value_counts(normalize=True).head(4).items()))

    # eventos distintos: junta episódios de composições diferentes que começam a menos de 12 h um do outro
    T = T.sort_values("a"); grupo, ult = [], None
    for a in T.a:
        if ult is None or (a - ult) > pd.Timedelta(hours=12): grupo.append(len(grupo))
        else: grupo.append(grupo[-1])
        ult = a
    T["evento"] = grupo
    U = T.groupby("evento").agg(a=("a", "min"), classe=("classe", lambda s: s.mode().iloc[0]), n_comp=("dia", "nunique"), dur_h=("dur_h", "median"),
                                canais=("canais", lambda s: s.mode().iloc[0]), p_tag=("p_tag", lambda s: s.mode().iloc[0]), p_share=("p_share", "median"),
                                p_x_lim=("p_x_lim", "median")).reset_index(drop=True)
    print(f"\nEPISÓDIOS DISTINTOS (agrupados por início a menos de 12 h): {len(U)}  | FP {int((U.classe == 'FP').sum())}, NEUTRO {int((U.classe == 'NEUTRO').sum())}, TP {int((U.classe == 'TP').sum())}")
    print(U.assign(a=U.a.dt.strftime("%Y-%m-%d %H:%M"), dur_h=U.dur_h.round(1), p_share=U.p_share.round(2), p_x_lim=U.p_x_lim.round(1)).to_string(index=False))


if __name__ == "__main__":
    main()
