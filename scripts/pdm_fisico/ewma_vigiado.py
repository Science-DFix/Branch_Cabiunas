#!/usr/bin/env python3
"""O EWMA só enxerga dado vigiado: a correção da borda do blackout.

O DIAGNÓSTICO (`transiente_diagnostico.py`). Nos 57 religamentos seguidos de
>= 30 h de operação, em unidades do limiar do nível A:

    canal   cru em 6,5 h   regime 24-30 h   EWMA em 6,5 h
    t           0,28            0,30             1,01
    p           0,14            0,17             5,12   (p75 29,3)
    sp          0,61            0,40             0,61
    vb          1,13            1,14             1,14

O transiente JÁ PASSOU quando a máscara abre. O que está alto é o EWMA, que é
calculado sobre o sinal cru inteiro -- parada e blackout inclusive -- e chega à
abertura da máscara carregando o resíduo enorme da partida, quando a máquina
ainda está longe do regime do baseline. Nos 11 nascimentos da borda, `p` cru vai
de 0,04 a 1,43 e o EWMA, de 1,30 a 72,9.

A CORREÇÃO. Alimentar o EWMA só com instantes vigiados (estável e fora do
blackout). Não é modelo de transiente, não tem parâmetro, e é o que o detector
deveria ter feito desde o início: a máscara existe para o detector não olhar a
partida, e o EWMA olhava.

POR QUE NÃO É O QUE JÁ FALHOU. As quatro tentativas anteriores contra a borda --
portão de não-decaimento, ordenação por severidade, duração da parada, rampa de
T5 -- tentavam SEPARAR, depois do fato, os episódios de borda bons dos ruins. E
encurtar o blackout mudava QUANTO se olha. Aqui não se separa nada: tira-se da
entrada um dado que a própria máscara declara inválido.

O RISCO, registrado antes de medir. O 04/11 nasce na borda com `p` essencial,
EWMA 5,04 e cru 0,18. Se ele cair, a leitura honesta é que ele era detectado pela
memória da partida -- o mesmo mecanismo dos NEUTRO --, não pela física.

CRITÉRIO (a régua, `regua_fp.compara`, com a guarda de FP/mês):
  principal     : os 4 canais          -- a correção como deve ser
  sensibilidade : só t e p             -- os dois que o diagnóstico acusa
Duas variantes, sem parâmetro contínuo: não há platô a exigir.

RESULTADO (28-29/09/2026) -- REPROVADO pela regra pré-registrada.

    (mediana das 8)      det      início  banda  FP/mês  carga   perde det
    atual              6 [5-8]    6       4      0,861   130,5   --
    EWMA vigiado       6 [5-6]    4       3      0,517   105,6   6 de 8
    só t e p           idêntico ao de 4 canais -- o efeito é todo de t e p

A carga cai em 8/8 e o FP/mês cai 40%, mas 6 composições perdem detecção.
O 04/11 caiu, como registrado antes de medir. No dia 1 os perdidos são dois, por
mecanismos DIFERENTES:

  04/11/2025  nasce na borda de uma partida COMUM: EWMA de p em 5,04 (mediana
              de partida normal: 5,12), vb em 1,14x o limiar (o nível normal
              dele). No detector atual, 20% das partidas comuns geram alarme na
              borda (o voto liga em 53%); com o EWMA vigiado, 7%. O 04/11 teve
              uma partida nas 48 h anteriores, e o alarme nasceu nela.
  11/04/2025  o sinal é IDÊNTICO nas duas versões. Um voto de 0,9 h em 10/04
              16:10, logo após o refratário anterior vencer, abre um novo
              bloqueio de 72 h -- e depois o filtro de duração o descarta. Ele
              nunca vira alarme, mas bloqueia o alarme real das 11/04 12:56. É a
              ordem refratário -> duração, cuja inversão já foi medida no dia 1
              e custou 7,5x as horas ([[margem-subsumida-pelo-v2]]).

LEITURA EXPLORATÓRIA, A POSTERIORI (`acima_do_acaso`). Detecção além do que a
própria cobertura acertaria por acaso (`avalia.permuta`):

    (mediana das 8)   det    acaso   ACIMA do acaso   cobertura
    atual             6,5    2,71        3,77           33,9%
    EWMA vigiado      5,5    2,12        3,50           26,6%

~60% da detecção perdida é acaso, mas não toda: acima do acaso cai 0,29 e o
vigiado só é melhor em 2 de 8. Não é correção grátis -- é mais um movimento na
fronteira, mais favorável que os outros (-40% de FP por -0,3 detecção real).

O NÚMERO QUE IMPORTA MAIS QUE O EXPERIMENTO. No detector atual, das ~6,5
detecções típicas, 2,7 são o que um detector sem informação acertaria com a
mesma cobertura. Só ~3,8 estão acima do acaso.

Uso:  PYTHONPATH=. python ewma_vigiado.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import regua_fp as R
import coativacao as CO

VARIANTES = {"4 canais": ("t", "p", "sp", "vb"), "só t e p": ("t", "p")}


def fn(canais):
    return lambda t, p, ms, ds: R.detector(t, p, ms, ds, ewma_vigiado=canais)["fin"]


def episodios_dia1(canais):
    t, p, ms, ds = R.sinais(1)
    d = R.detector(t, p, ms, ds, ewma_vigiado=canais)
    eps = R.AV.episodios(d["fin"])
    return R.DB.classifica_regra_c(eps, R.PARADAS), d


def main():
    ref = R.distribuicao()
    print("REFERÊNCIA — 8 composições\n" + R.resumo(ref))

    res = {}
    for nome, canais in VARIANTES.items():
        r = R.compara(fn(canais), nome, ref=ref)
        res[nome] = r
        v = r["var"]
        print(f"\n{'=' * 90}\nVARIANTE: EWMA vigiado em {nome}\n{'=' * 90}")
        print(R.resumo(v))
        print(f"\n  carga cai em {r['melhora']}/8  |  composições que perdem detecção: "
              f"{r['perde_det']}  |  ACEITO: {r['aceito']}")
        print("\n  diferença pareada (variante − referência):")
        print(r["dif"][["det", "inicio", "banda", "n_fp", "fp_mes", "h_fp_mes",
                        "n_neutro", "h_neutro_mes", "carga_mes"]].round(2).to_string())

    # mecanismo e episódios, dia 1, variante principal
    canais = VARIANTES["4 canais"]
    cls_ref, d_ref = episodios_dia1(())
    cls_var, d_var = episodios_dia1(canais)
    normal, pre = CO.janelas()
    print(f"\n{'=' * 90}\nMECANISMO, dia 1 — ciclo em operação normal (nível A) e RV\n{'=' * 90}")
    for c in R.SIN:
        a0, a1 = d_ref["A"][c].to_numpy(), d_var["A"][c].to_numpy()
        print(f"  {c:3s} ciclo {100 * a0[normal].mean():5.1f}% -> {100 * a1[normal].mean():5.1f}%"
              f"   RV {a0[pre].mean() / a0[normal].mean():4.2f} -> {a1[pre].mean() / max(a1[normal].mean(), 1e-12):4.2f}")
    print(f"  voto A em operação normal: {100 * d_ref['vA'].to_numpy()[normal].mean():.1f}% -> "
          f"{100 * d_var['vA'].to_numpy()[normal].mean():.1f}%")

    print(f"\n{'=' * 90}\nEPISÓDIOS, dia 1 — o que some, o que fica, o que nasce\n{'=' * 90}")
    fin_var = d_var["fin"]
    for a, b, k, _ in cls_ref:
        sob = fin_var.loc[a:b]
        if sob.any():
            a2 = sob.index[sob.to_numpy()][0]
            estado = "fica" if a2 == a else f"fica, nasce {(a2 - a).total_seconds() / 3600:+.1f} h"
        else:
            estado = "SOME"
        print(f"  {k:6s} {a:%Y-%m-%d %H:%M}  {(b - a).total_seconds() / 3600 + 2 / 60:6.1f} h   {estado}")
    novos = [(a, b, k) for a, b, k, _ in cls_var
             if not d_ref["fin"].loc[a:b].any()]
    for a, b, k in novos:
        print(f"  {k:6s} {a:%Y-%m-%d %H:%M}  {(b - a).total_seconds() / 3600 + 2 / 60:6.1f} h   NOVO")

    alvo_0411 = pd.Timestamp("2025-11-04 06:22", tz="UTC")
    for nome, d in (("referência", d_ref), ("EWMA vigiado", d_var)):
        w = d["fin"].loc[alvo_0411 - R.JAN:alvo_0411]
        print(f"\n  04/11 — {nome}: {'DETECTADO' if w.any() else 'perdido'}"
              + (f", alarme desde {w.index[w.to_numpy()][0]:%d/%m %H:%M}" if w.any() else ""))

    pd.DataFrame([dict(variante=n, melhora=r["melhora"], perde_det=r["perde_det"],
                       aceito=r["aceito"], **{f"{c}_med": r["var"][c].median()
                       for c in ("det", "inicio", "banda", "fp_mes", "h_fp_mes", "carga_mes")})
                  for n, r in res.items()]).to_csv(R.CACHE / "ewma_vigiado.csv", index=False)


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════ leitura exploratória, a posteriori
def acima_do_acaso(dias=R.DIAS) -> pd.DataFrame:
    """Detecção ACIMA do acaso, nas 8 composições -- referência e EWMA vigiado.

    POR QUÊ, E POR QUE É POSTERIOR. A regra reprovou o EWMA vigiado por perder
    detecção. Mas a perícia do 04/11 mostrou um nascimento numa partida comum
    (EWMA de p em 5,04, a mediana de uma partida normal é 5,12; vb no nível
    normal), e no detector atual 1 em cada 5 partidas comuns gera alarme na
    borda. Um detector que fica aceso mais tempo "acerta" mais trips por acaso.

    `avalia.permuta` -- a ferramenta que o projeto já usa, não uma inventada
    agora -- sorteia 8 instantes de operação e conta quantos cairiam com alarme
    nas 48 h anteriores. `acima = det - nulo` é o que o detector acerta além do
    que a sua própria cobertura acertaria sem informação nenhuma.

    ISTO NÃO MUDA O VEREDITO. A regra foi escrita antes e reprovou. Olhar pela
    lente do acaso DEPOIS de ver a reprovação é exatamente o tipo de flexibilidade
    que a régua existe para impedir. Serve para decidir se vale pré-registrar um
    experimento novo, não para aprovar este."""
    L = []
    for dia in dias:
        t, p, ms, ds = R.sinais(dia)
        for nome, canais in (("referência", ()), ("EWMA vigiado", VARIANTES["4 canais"])):
            fin = R.detector(t, p, ms, ds, ewma_vigiado=canais)["fin"]
            m = R.AV.avalia(fin, R.alvo, R.mask)
            pm = R.AV.permuta(fin, R.mask, m["det"], len(R.alvo))
            L.append(dict(dia=dia, variante=nome, det=m["det"], nulo=pm["nulo"],
                          acima=m["det"] - pm["nulo"], p=pm["p"],
                          cobertura=pm["cobertura"]))
    return pd.DataFrame(L)
