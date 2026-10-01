#!/usr/bin/env python3
"""R3: retreino semanal com janela de guarda, com G escolhido pelo platô.

PRÉ-REGISTRADO EM 01/10/2026, ANTES DE RODAR (commit deste arquivo sem resultado).

DE ONDE VEM. Em `retreino_semanal.py` duas guardas foram olhadas (11 e 21 dias): a de
11 corta a carga pela metade (-67 h/mês, IC exclui zero) e perde meia detecção; a de
21 melhora a detecção nas três réguas e não demonstra redução de carga. Escolher um G
agora, olhando essas duas, seria ajuste a posteriori. A regra abaixo foi escrita antes
de rodar a grade, e a grade não reaproveita os dois valores já vistos.

O DESENHO. Retreino a cada 7 dias, referência = os 20.000 pontos estáveis que terminam
G dias antes do corte (`retreino_semanal.walkforward_guarda`), 7 deslocamentos do
primeiro corte (0 a 6 dias). Grade REGULAR: G = 7, 10, 13, 16, 19, 22, 25, 28 dias.
Mesmos limiares do mensal; nada reajustado.

A ESCOLHA DE G (antes de qualquer teste):
  1. para cada G, as medianas entre os 7 cenários de det, início, banda e FP/mês, e
     a média da carga;
  2. cada métrica vira a MÉDIA DA VIZINHANÇA {G-3, G, G+3} (nas pontas, 2 pontos) --
     a regra do projeto contra escolher pico de ruído;
  3. elegíveis: G com carga da vizinhança <= carga média do mensal (133,4 h/mês) E FP
     da vizinhança <= FP mediano do mensal (0,861);
  4. entre os elegíveis, o de maior banda da vizinhança; empate -> maior início da
     vizinhança; empate -> menor carga da vizinhança;
  5. sem elegível: fica o mensal, e a curva entra no relatório.
O TESTE do G escolhido: `bootstrap_regua.decide(..., pareado=False, nivel=0.95)` --
desenho não pareado (os cenários semanais não são as composições mensais; os meses
são reamostrados juntos). GANHO = detecção estrita (medianas de det, início e banda
não abaixo do mensal) + IC da carga abaixo de zero + FP não sobe.
RESSALVA REGISTRADA: a escolha e o teste usam os mesmos 8 eventos. A vizinhança
reduz, não elimina, o otimismo. Um GANHO aqui é candidato a validação em dado novo,
não decisão de produção. E retreino semanal só faz sentido com o enquadramento do R2
(promoção automática, rotina auditada): 52 promoções por ano.

Uso:  PYTHONPATH=. python r3_guarda.py calcula G1,G2,...   # gera os caches
      PYTHONPATH=. python r3_guarda.py                     # escolhe e testa
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
import regua_fp as R
import retreino_semanal as RS
import bootstrap_regua as BR

GRADE = (7, 10, 13, 16, 19, 22, 25, 28)
DESLOC = range(7)
CARGA_MENSAL, FP_MENSAL = 133.4, 0.861


def sinais(g: int, d: int):
    f = R.CACHE / f"guarda{g}_d{d:02d}.npz"
    if f.exists():
        z = np.load(f); return z["t"], z["p"], z["ms"], z["ds"]
    sig = RS.walkforward_guarda(7, d, g)
    np.savez(f, t=sig[0], p=sig[1], ms=sig[2], ds=sig[3])
    return sig


def calcula(gs):
    for g in gs:
        for d in DESLOC:
            sinais(g, d)
        print(f"G={g} pronto", flush=True)


def main():
    pd.set_option("display.width", 200)
    ref = BR.braco("mensal", lambda d: R.detector(*R.sinais(d))["fin"], R.DIAS)
    bracos, L = {}, []
    for g in GRADE:
        b = BR.braco(f"G={g}", lambda d, g=g: R.detector(*sinais(g, d))["fin"], DESLOC)
        bracos[g] = b
        T = b.tabela
        L.append(dict(G=g, det=T.det.median(), inicio=T.inicio.median(), banda=T.banda.median(),
                      fp=T.fp_mes.median(), carga=T.carga_mes.mean(), fp_amp=T.fp_mes.max() - T.fp_mes.min()))
    M = pd.DataFrame(L).set_index("G")
    V = M.copy()
    for i, g in enumerate(GRADE):
        viz = [GRADE[j] for j in (i - 1, i, i + 1) if 0 <= j < len(GRADE)]
        V.loc[g] = M.loc[viz].mean()
    print("POR G (medianas entre os 7 cenários; carga = média):")
    print(M.round(3).to_string())
    print("\nMÉDIA DA VIZINHANÇA {G-3, G, G+3}:")
    print(V.round(3).to_string())
    eleg = V[(V.carga <= CARGA_MENSAL) & (V.fp <= FP_MENSAL)]
    print(f"\nelegíveis (carga <= {CARGA_MENSAL}, FP <= {FP_MENSAL} na vizinhança): {list(eleg.index)}")
    if eleg.empty:
        print("SEM ELEGÍVEL: fica o mensal.")
        M.to_csv(R.CACHE / "r3_guarda.csv"); return
    esc = eleg.sort_values(["banda", "inicio", "carga"], ascending=[False, False, True]).index[0]
    x = BR.decide(ref, bracos[esc], pareado=False, nivel=0.95)
    print(f"\nG ESCOLHIDO: {esc} dias  ->  det {x['det']} início {x['inicio']} banda {x['banda']}"
          f"  Δcarga {x['d_carga']:+.1f} [{x['ic_lo']:+.1f}; {x['ic_hi']:+.1f}]  ΔFP {x['d_fp']:+.3f}"
          f" [{x['fp_lo']:+.3f}; {x['fp_hi']:+.3f}]  ->  {x['veredito']}")
    print(f"  referência mensal: det {ref.tabela.det.median()} início {ref.tabela.inicio.median()}"
          f" banda {ref.tabela.banda.median()} FP {ref.tabela.fp_mes.median():.3f} carga {ref.tabela.carga_mes.mean():.1f}")
    M.assign(escolhido=lambda m: m.index == esc).to_csv(R.CACHE / "r3_guarda.csv")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "calcula":
        calcula([int(x) for x in sys.argv[2].split(",")])
    else:
        main()
