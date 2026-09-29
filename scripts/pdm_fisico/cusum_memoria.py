#!/usr/bin/env python3
"""Limitar a memória do CUSUM: o primeiro experimento na régua de 8 composições.

O MECANISMO (medido antes, `anatomia_canais.py`). Os canais são ESTADOS: ficam
acesos 34-59% do tempo em operação normal, e isso limita a informação de cada um
a 1/ciclo. Quase todo esse ciclo é do CUSUM -- `p` iria de 13,7% (só o degrau)
para 37,9% com ele, e em 63% do tempo aceso o EWMA já está abaixo do limiar. O
acumulador não tem teto: uma excursão forte o deixa acima de H por dias.

A HIPÓTESE. Limitar a memória derruba o ciclo normal, e com ele o teto de
informação sobe e os episódios encurtam -- a carga cai sem perder detecção.

JÁ FOI TENTADO? Sim, em `cusum_rezera.py`, mas sobre o detector v1 (um nível,
voto >= 2, refratário de 48 h) e só no dia 1 -- e nunca foi concluído por escrito.
O CUSUM com esquecimento (`cusum_vazante.py`) é outra coisa: decai SEMPRE e mata
a deriva lenta, que é o que o CUSUM entrega. Aqui a memória só é cortada DEPOIS
de o acumulador passar do limiar.

AS VARIANTES (todas aplicadas aos 4 canais, nos dois níveis):
  teto 1,25 · 1,5 · 2 · 3 · 4 · 8 × H     acumulador limitado a f·H
  head 0,50 · 0,75 × H                    ao sinalizar, volta a par·H
  zero                                    ao sinalizar, volta a 0

CRITÉRIO, ESCRITO ANTES DE RODAR:
  1. a regra da régua (`regua_fp.compara`): carga cai em >= 7 de 8 composições,
     nenhuma composição perde detecção, início e banda não caem na mediana;
  2. PLATÔ: um valor de `teto` só é aceito se um vizinho também passar. Testar
     nove variantes nas mesmas 8 composições é comparação múltipla -- um vencedor
     isolado é o que o acaso produz;
  3. todas as variantes são reportadas, as que passam e as que não.

RESULTADO (28/09/2026) -- REFUTADO, as nove variantes.

O mecanismo funciona como previsto. Com teto, o ciclo normal de `p` cai de 37,9%
para 16-26%, o voto A em operação normal de 33,5% para 15-21%, e o teto de
informação do `vb` sobe de 1,44 para 1,8-2,2. A carga cai em 7 de 8 composições
(mediana 130,5 -> 77-103 h/mês).

Mas a detecção cai: 5 a 7 das 8 composições perdem 1 ou 2 detecções, a régua de
início vai de 6/8 para 4,5/8 na mediana, a banda de 4/8 para 3,5/8. E o FP/mês
SOBE (0,861 -> 0,95-1,55): o episódio encurta e se parte em vários.

    variante    det      início  banda  FP/mês  carga   perde det
    atual       6 [5-8]  6       4      0,861   130,5   --
    teto 1,5H   6 [4-7]  4,5     3,5    1,292    76,9   6 de 8
    teto 2H     5 [4-8]  4,5     3,5    1,120    90,1   6 de 8
    teto 8H     6 [4-8]  5       3,5    0,947   102,8   6 de 8
    rezero      4 [3-7]  3,5     3      1,550    52,6   7 de 8

A LEITURA. Os canais disparam em momentos descoordenados (vb em ~p69, t em
p87). A memória longa do CUSUM é a COLA que faz evidências assíncronas
coincidirem no voto antes do trip. Cortá-la tira a cola. É a fronteira de
[[o-detector-esta-numa-fronteira]], agora com o mecanismo à vista: mesmo o teto
8H, que ainda deixa 1 a 3 dias de memória, perde detecção em 6 composições.

POR QUE A RÉGUA IMPORTA. No dia 1 o teto 2H PIORA a carga (+26 h/mês); nas
outras sete composições melhora (-22 a -91). O teste antigo, só no dia 1, teria
descartado a ideia pelo motivo errado.

CONSEQUÊNCIA. Reduzir a carga mexendo no detector custa nascimento. O corte tem de
ser na camada de apresentação -- notificar no nascimento e depois tratar o
episódio como condição conhecida (ISA-18.2) --, que por construção não muda
nenhum nascimento.

Uso:  PYTHONPATH=. python cusum_memoria.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import regua_fp as R
import coativacao as CO

VARIANTES = [("teto", f) for f in (1.25, 1.5, 2.0, 3.0, 4.0, 8.0)] + \
            [("head", 0.50), ("head", 0.75), ("zero", 0.0)]


def rotulo(modo, par):
    return {"teto": f"teto {par:g}H", "head": f"head {par:g}H", "zero": "rezero"}[modo]


def mecanismo(modo, par) -> dict:
    """Dia 1: ciclo normal dos canais (nível A) e razão de verossimilhança."""
    t, p, ms, ds = R.sinais(1)
    d = R.detector(t, p, ms, ds, cusum=(modo, par))
    normal, pre = CO.janelas()
    out = {}
    for c in R.SIN:
        x = d["A"][c].to_numpy()
        out[f"ciclo_{c}"] = x[normal].mean()
        out[f"rv_{c}"] = x[pre].mean() / max(x[normal].mean(), 1e-12)
    out["voto_A"] = d["vA"].to_numpy()[normal].mean()
    return out


def main():
    ref = R.distribuicao()
    print("REFERÊNCIA (detector atual) nas 8 composições")
    print(R.resumo(ref))
    mec0 = mecanismo("nunca", 0.0)

    linhas, detalhe = [], {}
    for modo, par in VARIANTES:
        nome = rotulo(modo, par)
        r = R.compara(lambda t, p, ms, ds, m=modo, q=par:
                      R.detector(t, p, ms, ds, cusum=(m, q))["fin"], nome, ref=ref)
        mec = mecanismo(modo, par)
        v = r["var"]
        linhas.append(dict(
            variante=nome, modo=modo, par=par,
            det=f"{v.det.median():.0f} [{v.det.min()}-{v.det.max()}]",
            inicio=v.inicio.median(), banda=v.banda.median(),
            fp_mes=v.fp_mes.median(), h_fp=v.h_fp_mes.median(),
            carga=v.carga_mes.median(),
            melhora=f"{r['melhora']}/8", perde_det=r["perde_det"],
            ciclo_p=100 * mec["ciclo_p"], ciclo_vb=100 * mec["ciclo_vb"],
            voto_A=100 * mec["voto_A"], rv_vb=mec["rv_vb"], passa=r["aceito"]))
        detalhe[nome] = r

    T = pd.DataFrame(linhas)
    # platô: para o teto, um vizinho imediato também tem de passar
    T["aceito"] = False
    tet = T[T.modo == "teto"].reset_index()
    for i, row in tet.iterrows():
        viz = [tet.passa.iloc[j] for j in (i - 1, i + 1) if 0 <= j < len(tet)]
        T.loc[row["index"], "aceito"] = bool(row.passa and any(viz))
    T.loc[T.modo != "teto", "aceito"] = T.loc[T.modo != "teto", "passa"]

    pd.set_option("display.width", 200)
    print(f"\n{'=' * 100}\nVARIANTES — mediana nas 8 composições (dia 1 para o mecanismo)\n{'=' * 100}")
    print(f"referência: det {ref.det.median():.0f} [{ref.det.min()}-{ref.det.max()}]  "
          f"início {ref.inicio.median():.0f}  banda {ref.banda.median():.0f}  "
          f"FP/mês {ref.fp_mes.median():.3f}  h FP {ref.h_fp_mes.median():.1f}  "
          f"carga {ref.carga_mes.median():.1f}  | ciclo p {100 * mec0['ciclo_p']:.1f}%  "
          f"vb {100 * mec0['ciclo_vb']:.1f}%  voto A {100 * mec0['voto_A']:.1f}%  "
          f"RV vb {mec0['rv_vb']:.2f}\n")
    cols = ["variante", "det", "inicio", "banda", "fp_mes", "h_fp", "carga", "melhora",
            "perde_det", "ciclo_p", "ciclo_vb", "voto_A", "rv_vb", "passa", "aceito"]
    print(T[cols].round(3).to_string(index=False))

    for nome, r in detalhe.items():
        if T.loc[T.variante == nome, "aceito"].iloc[0] or nome in ("teto 2H",):
            print(f"\n--- {nome}: diferença pareada por composição (variante − referência)")
            print(r["dif"][["det", "inicio", "banda", "fp_mes", "h_fp_mes",
                            "h_neutro_mes", "carga_mes"]].round(2).to_string())
    T.to_csv(R.CACHE / "cusum_memoria.csv", index=False)


if __name__ == "__main__":
    main()
