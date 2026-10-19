#!/usr/bin/env python3
"""E5 -- regra de falha de comunicação: ignorar o canal do modelo quando o sensor do grupo estava em falha.

PRÉ-REGISTRO (19/10/2026, antes de rodar)
Pergunta (do usuário: "vamos de item 2", relatorio_falsos_positivos): o E1 achou 2 FP (14 e 17/01/2026) com "Comm Fail"
/"Out of Serv" (texto no lugar da medida) nos sensores do grupo aceso, e nenhum trip com isso. É a única causa de dado
que nenhum trip tem.
Regra (física, sem parâmetro escolhido nos trips): o canal do modelo k não vota no instante t se algum sensor do grupo
k tem valor não numérico em (t - 6 h, t]. 6 h = a janela de antes do E1, fixada antes; não há varredura para escolher.
Grupos = sensores de treino dos configs de produção (EXP33: TC382_03_A, T5_AVG_A; EXP34: 10 TV_*; EXP38: PI_0308,
PDIT_0305). Canal 4 e camada de decisão de produção, OCSVM de produção, período inteiro.
Não é uma regra de custo (o ganho previsto é ~5%), é de HIGIENE. ADOTA se: 8/8, 3/3 fora, as 8 antecedências iguais
(± 0,1 h) e FP/mês não sobe. Descritivo: o mesmo com 1 h e 24 h (sensibilidade, não escolha).
Expectativa: some 1-2 FP (os de 14 e 17/01/2026; no período inteiro pode haver mais algum de 2024), trips intactos,
FP/mês 2,885 -> ~2,82 -> ADOTA.

RESULTADO (19/10/2026, depois do pré-registro d2cf017): ADOTA.
  6 h: 8/8, fora 3/3, antecedências iguais; FP 42 -> 41 (2,885 -> 2,816/mês; fora 2,877 -> 2,752). Somem os FP de
  14/01/26 18:57 e 17/01/26 01:00, mas aparece um em 15/01/26 12:56: o refratário do de 14/01 deixa de cobri-lo.
  Sensibilidade: 1 h -> 41 FP; 24 h -> 40 FP. Expectativa certa.
  Implementada em scripts/pipeline_unificada_final.py (compute_comm_fail_mask em scoring.py); a pipeline real
  reproduz 8/8 e 2,82.

Uso: python frente_ocsvm/e5_falha_comunicacao.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import comum as C
from e1_triagem_fp import GRUPO, sensores

JANELA_H = 6.0
CANAL_GRUPO = dict(zip(C.CANAIS[:3], ("temp", "vib", "óleo")))


def em_falha(txt: pd.DataFrame, janela_h: float) -> dict:
    """Por canal: houve valor não numérico em algum sensor do grupo nas últimas `janela_h` horas."""
    n = int(round(janela_h * 120))  # pontos de 30 s
    return {k: txt[GRUPO[g]].any(axis=1).astype(int).rolling(n, min_periods=1).max().astype(bool)
            for k, g in CANAL_GRUPO.items()}


def roda(df, c4, op, ft, falha, meses_fora):
    ch = {k: df[k].astype(bool) & ~falha[k] if falha else df[k].astype(bool) for k in C.CANAIS[:3]}
    ch["c4"] = c4
    cls, m = C.avalia(C.decide(ch, 2, C.REF["dur_min"], C.REF["refrat_h"]), op, ft)
    P = C.por_trip(cls, ft)
    fp = cls[cls.classe == "falso_positivo"]
    return dict(det=int(P.deteccao.notna().sum()), fora=int(P[P.amostra == "fora"].deteccao.notna().sum()),
                FP=len(fp), fp_mes=round(m["falso_positivo_por_mes"], 3),
                fp_mes_fora=round(int((fp.start >= C.SPLIT).sum()) / meses_fora, 3)), P, set(fp.start)


def main():
    df = C.carrega_canais()
    idx, op = df.index, df["operational_state"]
    cat = C.catalogo()
    c4 = C.canal_alarme(idx, cat.loc[cat.tag.isin(C.TAGS_C4), "t"], C.REF["janela_c4_h"])
    ft = C.trips(idx)
    meses_fora = C.compute_operational_period_days(pd.DataFrame({"operational_state": op[idx >= C.SPLIT]})) / 30.4375
    _, txt = sensores(inicio="2024-06-25")
    txt = txt.reindex(idx, fill_value=False)
    print("instantes com texto no lugar da medida, por grupo:",
          {g: int(txt[s].any(axis=1).sum()) for g, s in GRUPO.items()})

    ref, P0, fp0 = roda(df, c4, op, ft, None, meses_fora)
    print("produção:", ref)
    for h in (JANELA_H, 1.0, 24.0):
        r, P, fp = roda(df, c4, op, ft, em_falha(txt, h), meses_fora)
        dlead = np.nanmax(np.abs(P.lead_h.values - P0.lead_h.values))
        print(f"\njanela {h:g} h{' (PRÉ-REGISTRADA)' if h == JANELA_H else ' (sensibilidade)'}: {r}; "
              f"maior mudança de antecedência {dlead:.1f} h")
        print("  FP que somem:", sorted(t.strftime("%d/%m/%y %H:%M") for t in fp0 - fp) or "-",
              "| FP novos:", sorted(t.strftime("%d/%m/%y %H:%M") for t in fp - fp0) or "-")
        if h == JANELA_H:
            adota = (r["det"] == 8 and r["fora"] == 3 and dlead <= 0.1 and r["fp_mes"] <= ref["fp_mes"])
            print(f"  -> {'ADOTA' if adota else 'NÃO ADOTA'}")


if __name__ == "__main__":
    main()
