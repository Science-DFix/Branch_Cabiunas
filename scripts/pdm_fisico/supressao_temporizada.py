#!/usr/bin/env python3
"""Supressao temporizada (timed shelving, ISA-18.2) no v2: a carga cai sem custar deteccao?

De onde vem. No v2 publicado a carga operacional -- o que a sala de controle ve
aceso sem preceder trip -- e 48,9 h/mes, e 87% dela sao episodios NEUTRO longos
(153, 135, 112, 52 h de alarme de pe antes de parada real; filtro_cva_d.py). Nao
e problema de sinal: SFA e CVA foram refutados como canal e como veto.

ATENCAO -- REPETE UMA CONCLUSAO JA REGISTRADA. corte_por_estabilizacao.py
(commit 44786de, 19/09) ja cortou episodio longo para baixar carga (48,9 -> 7,2
h/mes a 3x o FP) e a frente foi DESCARTADA porque falso positivo nao pode subir.
Este script chega ao mesmo lugar por outro mecanismo (teto de tempo em vez de
estabilizacao). Fica como segunda confirmacao, nao como achado novo.

O que ja foi medido e o que nao foi. O teto de permanencia (auto_reset.py,
teto_permanencia.py, teto_sob_regra_inicio.py) foi testado no V1 e com outro
objetivo -- GANHAR deteccao por renascimento. Caiu na regua "de pe" e empatou em
4/8 na de inicio. A metrica de carga so nasceu em 21/09, depois disso. Nunca se
mediu um teto no v2 com o objetivo de REDUZIR A CARGA.

A politica. Cada episodio final fica ANUNCIADO por no maximo T horas; depois e
suprimido (shelved). O estado do detector nao muda -- nascimentos, refratario e a
escalada por idade (que reanuncia quando a condicao piora) sao os do v2. E pura
politica de apresentacao, a mesma que a EEMUA 191 recomenda para alarme de pe.

O que isto NAO e. Nao muda nenhum nascimento, entao as reguas de inicio e de
banda ficam identicas por construcao. O que muda e a Regra C, que classifica pelo
intervalo ACESO: um episodio truncado termina longe do evento e pode virar FP --
um acerto "de pe" ha 8 dias deixa de estar aceso no trip, um NEUTRO de 153 h
cortado em 24 h deixa de estar colado na parada. Essa migracao e o custo honesto
da politica e e medida aqui, junto com a regua "de pe" (que o teto derruba).

RESULTADO (30/09/2026) -- funciona para a carga, com um custo que precisa de
decisao da operacao, nao nossa.

    teto   de pe  inicio  banda   FP  FP/mes   carga h/mes   aceso no trip
    sem     8/8    6/8     5/8     4   0,344      48,9          4/8
    48 h    7/8    6/8     5/8    10   0,861      38,8          3/8
    24 h    7/8    6/8     5/8    10   0,861      23,4          3/8
    12 h    7/8    6/8     5/8    11   0,947      13,6          2/8

- Inicio, banda e lead de inicio (15,66 h) NAO MUDAM em nenhum teto -- por
  construcao, e confirmado. A regua adotada e indiferente a politica.
- A carga cai 52% em 24 h e 72% em 12 h. Em 72 h ela SOBE (51,5): as horas dos
  acertos de pe migram para FP e 72 h ainda e longo.
- O preco e contabil e real ao mesmo tempo: FP/mes 0,344 -> 0,861 em 24 h. Seis
  episodios migram de classe -- 3 NEUTRO (153, 135, 112 h) e 3 TP de pe
  (140, 195, 648 h, nascidos dias antes do trip) viram FP, porque o alarme
  anunciado deixa de encostar no evento. Os 3 TP ja nao contavam na regua de
  inicio; o que se perde e o credito "de pe" (8/8 -> 7/8) e ter o alarme aceso
  no trip (4/8 -> 3/8).
- O ponto de inflexao e 24-48 h: de 48 para 24 h a carga cai 40% sem nenhum FP
  a mais. Abaixo de 24 h cada passo custa FP.

Leitura. Nao ha ganho gratis -- ha uma troca explicita entre DUAS metricas de
custo que ja publicamos lado a lado (regra_c_com_teto.py): acuracia (FP/mes)
piora, carga operacional melhora. Pela EEMUA 191 / ISA-18.2, alarme de pe e
defeito e a supressao temporizada e a pratica recomendada; mas se a operacao
valoriza o alarme continuo antes de parada, o numero certo e o sem teto. E
decisao de quem opera. NAO adotado no publica_clearml.py.

Uso (de dentro de scripts/pdm_fisico, com os dados):  python supressao_temporizada.py
"""
from __future__ import annotations

import pandas as pd

import avalia as AV
import publica_clearml as PC
from plota_estilo_francisco import classifica_regra_c, paradas_reais_2h

TETOS = [None, 72, 48, 24, 12, 6, 4]


def trunca(al: pd.Series, n_h) -> pd.Series:
    """Cada episodio anunciado por no maximo n_h horas a partir do inicio.

    Copia de auto_reset.py::trunca -- nao importada porque auto_reset.py (como
    ablacao.py) chama main() no nivel do modulo e roda o experimento inteiro."""
    if n_h is None:
        return al
    novo = pd.Series(False, index=al.index)
    for a, b in AV.episodios(al):
        janela = (al.index >= a) & (al.index <= min(b, a + pd.Timedelta(hours=n_h)))
        novo.loc[janela] = al.loc[janela]
    return novo


def mede(al, mask, alvo, sel, paradas):
    res, _, _ = PC.metricas(al, mask, alvo, sel, permutacao=False)
    eps = AV.episodios(al)
    cls = classifica_regra_c(eps, paradas)
    meses = (mask & sel).sum() * 2 / 60 / 730
    horas = lambda k: sum((b - a).total_seconds() / 3600 + 2 / 60 for a, b, c, _ in cls if c == k)
    aceso_no_trip = sum(bool(al.loc[t - pd.Timedelta(hours=2):t].any()) for t in alvo)
    n = lambda s: int(s.split("/")[0])
    return dict(de_pe=n(res["recall"]), inicio=n(res["recall_regua_inicio"]),
                banda=n(res["recall_banda_acionavel"]),
                fp=sum(c == "FP" for *_, c, _ in cls), neutro=sum(c == "NEUTRO" for *_, c, _ in cls),
                fp_mes=res["fp_por_mes_regra_c"],
                h_fp_mes=round(horas("FP") / meses, 1), h_neutro_mes=round(horas("NEUTRO") / meses, 1),
                carga=res["carga_h_por_mes"], h_tp_mes=round(horas("TP") / meses, 1),
                aceso_2h_antes_do_trip=aceso_no_trip, lead_ini=res["lead_medio_inicio_h"])


def main():
    al0, mask, alvo, _, idx, sel = PC.reproduz(v2=True)
    paradas = paradas_reais_2h()
    linhas = []
    for T in TETOS:
        r = mede(trunca(al0, T), mask, alvo, sel, paradas)
        linhas.append(dict(teto_h="sem" if T is None else T, **r))
    R = pd.DataFrame(linhas)
    pd.set_option("display.width", 220)
    print(R.to_string(index=False))

    # onde vai parar cada episodio longo, no teto de 24 h
    eps0 = classifica_regra_c(AV.episodios(al0), paradas)
    eps24 = {a: c for a, b, c, _ in classifica_regra_c(AV.episodios(trunca(al0, 24)), paradas)}
    print("\nmigracao de classe no teto de 24 h (so os que mudam):")
    for a, b, c, _ in eps0:
        if eps24.get(a) != c:
            print(f"  {a:%Y-%m-%d %H:%M}  {(b - a).total_seconds() / 3600:6.1f} h  {c} -> {eps24.get(a)}")


if __name__ == "__main__":
    main()
