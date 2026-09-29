#!/usr/bin/env python3
"""Só vale nascimento que o detector VIU acontecer.

O MECANISMO (`regimes.py`). Metade dos episódios nasce no instante em que a
máscara abre, qualquer que seja o blackout: 11 de 21 com 6 h no dia 1, e 7 a 8
com 24, 32 ou 48 h. A borda anda junto com o blackout. Como vb e sp ficam acesos
a maior parte do tempo normal, o detector, ao começar a olhar, já encontra o voto
ligado -- o horário do nascimento é ditado pela máscara, não pela física.

A REGRA (`regua_fp._transicao`). Um trecho de voto só vale se foi precedido por
SUSTAIN amostras com o voto DESLIGADO, contadas depois de o detector já estar
armado. Não é parâmetro novo.

POR QUE NÃO É O QUE JÁ FALHOU. Quatro tentativas contra a borda (não-decaimento,
severidade, duração da parada, rampa de T5) tentavam SEPARAR as bordas boas das
ruins. Esta descarta TODA borda pelo mesmo motivo: nenhuma delas foi observada.
A triagem em duas filas (`triagem_partida.py`) fazia algo parecido na
apresentação, sobre o detector antigo.

É A ÚLTIMA DESTA FAMÍLIA. Rodada uma vez, e a família fecha qualquer que seja o
resultado -- são muitas tentativas sobre os mesmos 8 eventos.

CRITÉRIO, o mesmo de `regimes.py`, escrito antes: detecção do regime longo nunca
abaixo da referência em nenhuma composição e início do regime longo não cai na
mediana; carga cai em >= 7 de 8 e FP/mês cai na mediana. Sem parâmetro, sem platô.
Os trips pós-partida perdidos ficam com o aviso de procedimento.

RESULTADO (29/09/2026) -- REPROVADO. A família "borda" fecha aqui.

    (mediana das 8)   longo det   longo início   pós det   FP/mês   carga
    referência          3/3          2/3          3,5/5    0,861    130,5
    transição           2/3          1/3          2/5      0,431     39,4

A carga cai em 8/8 (-70%) e o FP pela metade, mas o regime longo perde 1 ou 2
detecções em TODAS as composições. Até as detecções de regime longo dependem de
votos que já estavam ligados quando a máscara abriu. O 26/02/2026 é o caso claro:
o alarme nasceu na partida de 29/01, 28 dias antes, e ficou 647 h ligado; em 86%
desse episódio nenhum canal passa de 1,5x o limiar (`cusum_vazante.py`). É um
estado travado desde a partida, não uma degradação vista crescendo.

E REVÊ O "ACIMA DO ACASO". O nulo circular (`nulo_circular.py`) tira o alarme de
cima das partidas -- e os trips se concentram depois delas. No mesmo teste o
relógio de partida, que não olha sinal nenhum, ganha 3,2 "acima do acaso". As 4,2
do detector não são todas informação do sinal: a parte que vem só do sinal é da
ordem de 1 evento além do relógio. Com 8 eventos, frágil nas duas direções.

Uso:  PYTHONPATH=. python transicao_observada.py
"""
from __future__ import annotations
import pandas as pd
import regua_fp as R
import regimes as RG


def main():
    pos, lon = RG.separa_trips()
    ref = pd.DataFrame([dict(dia=d, **RG.mede_regime(R.detector(*R.sinais(d))["fin"], pos, lon))
                        for d in R.DIAS]).set_index("dia")
    var = pd.DataFrame([dict(dia=d, **RG.mede_regime(
        R.detector(*R.sinais(d), exige_transicao=True)["fin"], pos, lon))
        for d in R.DIAS]).set_index("dia")
    cols = ["det_pos", "ini_pos", "det_longo", "ini_longo", "banda_longo",
            "n_fp", "fp_mes", "h_fp_mes", "n_neutro", "h_neutro_mes", "carga_mes", "episodios"]
    pd.set_option("display.width", 200)
    print("REFERÊNCIA"); print(ref[cols].round(3).to_string())
    print("\nTRANSIÇÃO OBSERVADA"); print(var[cols].round(3).to_string())
    c1 = bool((var.det_longo >= ref.det_longo).all()
              and var.ini_longo.median() >= ref.ini_longo.median())
    cai = int((var.carga_mes < ref.carga_mes - 1e-9).sum())
    c2 = bool(cai >= 7 and var.fp_mes.median() < ref.fp_mes.median())
    print(f"\n  mediana: regime longo det {var.det_longo.median():.1f}/3 (ref {ref.det_longo.median():.1f})"
          f"  início {var.ini_longo.median():.1f}/3 (ref {ref.ini_longo.median():.1f})"
          f"  | pós-partida det {var.det_pos.median():.1f}/5 (ref {ref.det_pos.median():.1f})")
    print(f"           FP/mês {var.fp_mes.median():.3f} (ref {ref.fp_mes.median():.3f})"
          f"  carga {var.carga_mes.median():.1f} (ref {ref.carga_mes.median():.1f})"
          f"  | carga cai em {cai}/8  | FP cai em {int((var.fp_mes < ref.fp_mes - 1e-9).sum())}/8")
    print(f"  critério 1 (regime longo intacto): {c1}   critério 2 (custo cai): {c2}"
          f"   -> {'ACEITO' if c1 and c2 else 'REPROVADO'}")
    print("\n  diferença pareada (variante − referência):")
    print((var - ref)[cols].round(2).to_string())
    pd.concat({"ref": ref, "transicao": var}).to_csv(R.CACHE / "transicao_observada.csv")


if __name__ == "__main__":
    main()
