#!/usr/bin/env python3
"""O voto conta evidências independentes?

A PERGUNTA. O nível A alarma com >= 3 dos 4 canais acesos. Um voto assim só filtra
ruído se os canais errarem de forma independente. Dois fatos sugerem que não:

  · os canais ficam acesos 24-52% do tempo vigiado (os do Diego, 0,5-11%). Com
    esse ciclo, três coincidirem por acaso não é raro;
  · `t` e `p` são ambos resíduo de PCA contra o MESMO ponto de operação. Uma
    mudança de regime sobe os dois juntos, e o voto conta a mesma evidência duas
    vezes.

O QUE SE MEDE, em operação normal (vigiada, fora das 48 h que antecedem os
trips) e nas janelas pré-trip, separadamente:

  1. ciclo de cada canal;
  2. coativação par a par: P(i e j) contra P(i)·P(j) -- o lift. Lift 1 é
     independência; lift alto é causa comum;
  3. número efetivo de canais independentes (razão de participação dos
     autovalores da matriz de correlação dos flags): 4 se forem independentes,
     1 se forem o mesmo canal;
  4. a pergunta prática: se `t` e `p` contassem como UM voto, que fração dos
     instantes de voto A sobreviveria em operação normal, e que fração nas
     janelas pré-trip? Se a fusão mata muito mais voto normal que voto pré-trip,
     é um experimento que vale rodar na régua. Se mata igual, não vale.

Tudo isto é estrutural e medido sobre centenas de milhares de instantes -- não
depende dos 4 falsos positivos. Roda nas 8 composições de baseline para ver se a
conclusão depende da sorte do dia 1.

Uso:  PYTHONPATH=. python coativacao.py
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import regua_fp as R

SIN = R.SIN
PARES = list(itertools.combinations(SIN, 2))


def janelas():
    """normal = vigiado fora das 48 h pré-trip; pre = vigiado dentro delas."""
    m = R.mask.to_numpy()
    pre = np.zeros(len(R.idx), dtype=bool)
    for t in R.alvo:
        pre |= ((R.idx >= t - R.JAN) & (R.idx < t))
    return m & ~pre, m & pre


def mede_nivel(F: dict, normal, pre) -> dict:
    X = {c: F[c].to_numpy() for c in SIN}
    out = {}
    for nome, sel in (("normal", normal), ("pre", pre)):
        M = np.column_stack([X[c][sel] for c in SIN]).astype(float)
        duty = M.mean(axis=0)
        lift, phi = {}, {}
        for i, j in PARES:
            a, b = X[i][sel], X[j][sel]
            pij, pi, pj = (a & b).mean(), a.mean(), b.mean()
            lift[f"{i}-{j}"] = pij / (pi * pj) if pi * pj > 0 else np.nan
            phi[f"{i}-{j}"] = np.corrcoef(a, b)[0, 1] if a.std() and b.std() else np.nan
        C = np.corrcoef(M.T)
        lam = np.clip(np.linalg.eigvalsh(np.nan_to_num(C)), 0, None)
        pr = lam.sum() ** 2 / (lam ** 2).sum()
        out[nome] = dict(duty=dict(zip(SIN, duty)), lift=lift, phi=phi,
                         n_efetivo=float(pr), n=int(sel.sum()))
    return out


def fusao_tp(A: dict, normal, pre) -> dict:
    """Voto A com t e p valendo UM voto: (t|p) + sp + vb >= 3."""
    X = {c: A[c].to_numpy() for c in SIN}
    cont = sum(X[c].astype(int) for c in SIN)
    vA = cont >= 3
    fund = (X["t"] | X["p"]).astype(int) + X["sp"].astype(int) + X["vb"].astype(int)
    vF = fund >= 3
    out = {}
    for nome, sel in (("normal", normal), ("pre", pre)):
        base = vA & sel
        out[nome] = dict(voto_A=float(base.mean() / max(sel.mean(), 1e-12)),
                         sobrevive=float((vF & base).sum() / max(base.sum(), 1)),
                         com_t_e_p=float((X["t"] & X["p"] & base).sum() / max(base.sum(), 1)))
    return out


def roda_dia(dia: int):
    t, p, ms, ds = R.sinais(dia)
    d = R.detector(t, p, ms, ds)
    normal, pre = janelas()
    return dict(A=mede_nivel(d["A"], normal, pre), B=mede_nivel(d["B"], normal, pre),
                fus=fusao_tp(d["A"], normal, pre))


def imprime(r: dict, dia: int):
    print(f"\n{'=' * 78}\nCOMPOSIÇÃO: retreino no dia {dia}"
          f"{'  (o publicado)' if dia == 1 else ''}\n{'=' * 78}")
    for nivel in ("A", "B"):
        x = r[nivel]
        print(f"\n  NÍVEL {nivel}   (normal: {x['normal']['n']} instantes vigiados | "
              f"pré-trip: {x['pre']['n']})")
        print(f"    {'':10s}" + "".join(f"{c:>9s}" for c in SIN))
        for nome in ("normal", "pre"):
            print(f"    ciclo {nome:4s}" + "".join(f"{100 * x[nome]['duty'][c]:8.1f}%"
                                              for c in SIN))
        print(f"\n    {'par':8s} {'lift normal':>12s} {'lift pré':>10s} {'phi normal':>11s} {'phi pré':>9s}")
        for par in x["normal"]["lift"]:
            print(f"    {par:8s} {x['normal']['lift'][par]:12.2f} {x['pre']['lift'][par]:10.2f}"
                  f" {x['normal']['phi'][par]:11.2f} {x['pre']['phi'][par]:9.2f}")
        print(f"\n    canais independentes efetivos: normal {x['normal']['n_efetivo']:.2f}"
              f" de 4 | pré-trip {x['pre']['n_efetivo']:.2f} de 4")
    f = r["fus"]
    print("\n  FUSÃO t+p NUM VOTO SÓ (nível A)")
    for nome in ("normal", "pre"):
        print(f"    {nome:6s}  voto A em {100 * f[nome]['voto_A']:5.2f}% do tempo | "
              f"com t e p juntos: {100 * f[nome]['com_t_e_p']:5.1f}% | "
              f"sobrevive à fusão: {100 * f[nome]['sobrevive']:5.1f}%")


if __name__ == "__main__":
    tabela = []
    for dia in R.DIAS:
        r = roda_dia(dia)
        if dia == 1:
            imprime(r, dia)
        f = r["fus"]; a = r["A"]
        tabela.append(dict(
            dia=dia,
            n_ef_A_normal=a["normal"]["n_efetivo"], n_ef_A_pre=a["pre"]["n_efetivo"],
            lift_tp_normal=a["normal"]["lift"]["t-p"], lift_tp_pre=a["pre"]["lift"]["t-p"],
            voto_normal_com_tp=f["normal"]["com_t_e_p"], voto_pre_com_tp=f["pre"]["com_t_e_p"],
            sobrevive_normal=f["normal"]["sobrevive"], sobrevive_pre=f["pre"]["sobrevive"]))
    T = pd.DataFrame(tabela).set_index("dia")
    print(f"\n{'=' * 78}\nAS 8 COMPOSIÇÕES\n{'=' * 78}")
    pd.set_option("display.width", 170)
    print(T.round(3).to_string())
    T.to_csv(R.CACHE / "coativacao.csv")


# ══════════════════════════════════════════════════ quanto cada canal discrimina
def informacao(dias=R.DIAS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Razão de verossimilhança de cada canal: ciclo pré-trip / ciclo normal.

    RV 1 = o canal está aceso na mesma proporção antes de um trip e em operação
    normal -- não carrega informação, é uma moeda que entra no voto. O normal tem
    ~245 mil instantes; o pré-trip, ~9,6 mil divididos em 8 eventos -- por isso a
    tabela por evento: um RV alto sustentado por um evento só não é sinal."""
    normal, pre = janelas()
    por_dia, por_evento = [], []
    for dia in dias:
        d = R.detector(*R.sinais(dia))
        for nivel in ("A", "B"):
            F = d[nivel]
            linha = dict(dia=dia, nivel=nivel)
            for c in SIN:
                x = F[c].to_numpy()
                linha[c] = x[pre].mean() / max(x[normal].mean(), 1e-12)
            por_dia.append(linha)
            if dia == 1:
                for t in R.alvo:
                    w = R.mask.to_numpy() & (R.idx >= t - R.JAN) & (R.idx < t)
                    e = dict(evento=t.strftime("%Y-%m-%d"), nivel=nivel,
                             horas_vigiadas=w.sum() / 30)
                    for c in SIN:
                        x = F[c].to_numpy()
                        e[c] = x[w].mean() / max(x[normal].mean(), 1e-12) if w.any() else np.nan
                    por_evento.append(e)
    return pd.DataFrame(por_dia), pd.DataFrame(por_evento)
