#!/usr/bin/env python3
"""D1 rodada 1d -- avaliação pré-registrada no docstring de d1_disputa.py (cnn1d_mensal_jan, walk-forward desde 01/2025).

Recorte de 01/01/2025 ao fim (o único em que o braço emite flag), OCSVM de produção no mesmo recorte. Reporta os 8 trips
com antecedência (h), FP e FP/mês no recorte e depois de 01/07/2025, e o veredito do critério.
Uso: python frente_ocsvm/d1_jan.py
"""
from __future__ import annotations
import json

import pandas as pd
import comum as C
from a4_variabilidade import canais_da_replica
from d1_disputa import combinacoes, TASKS_JSON, REF_FP, CORTE

INICIO = pd.Timestamp("2025-01-01")
REPS = ["prod", "todos_cnn1d_mensal_jan"] + [f"cnn1d_mensal_jan_so_{g}" for g in ("temp", "vib", "oleo")]


def main():
    reps = combinacoes(json.loads(TASKS_JSON.read_text()))
    cat = C.catalogo()
    leads, L = {}, []
    for rep in REPS:
        if rep not in reps:
            print(f"{rep}: sem as 3 tasks")
            continue
        try:
            df = canais_da_replica(reps[rep])
        except Exception as e:  # task não terminou ou falhou
            print(f"{rep}: indisponível ({type(e).__name__}: {e})")
            continue
        idx, op = df.index, df["operational_state"]
        c = {k: df[k] for k in C.CANAIS[:3]}
        c["c4"] = C.canal_alarme(idx, cat.loc[cat.tag.isin(C.TAGS_C4), "t"], C.REF["janela_c4_h"])
        ft = C.trips(idx)
        cls, _ = C.avalia(C.decide(c, 2, C.REF["dur_min"], C.REF["refrat_h"]), op, ft)
        P = C.por_trip(cls, ft).set_index("trip")
        leads[rep] = P.lead_h
        linha = dict(rep=rep)
        for nome, ini in (("rec", INICIO), ("fora", C.SPLIT)):
            dias = C.compute_operational_period_days(pd.DataFrame({"operational_state": op[idx >= ini]}))
            fp = int(((cls.start >= ini) & (cls.classe == "falso_positivo")).sum())
            p = P[P.index >= ini]
            linha.update({f"det_{nome}": f"{int(p.deteccao.notna().sum())}/{len(p)}", f"FP_{nome}": fp,
                          f"FPmes_{nome}": round(fp / (dias / 30.4375), 3)})
        L.append(linha)
    T = pd.DataFrame(L).set_index("rep")
    ref = T.loc["prod", "FPmes_rec"]
    T["veredito"] = ["—" if r == "prod" else
                     ("PASSA" if T.loc[r, "det_rec"] == "8/8" and T.loc[r, "FPmes_rec"] <= CORTE * ref
                      and T.loc[r, "det_fora"] == "3/3" and T.loc[r, "FPmes_fora"] <= CORTE * REF_FP["fora"]
                      else "NÃO") for r in T.index]
    pd.set_option("display.width", 200)
    print("Antecedência por trip (h; NaN = não detectado):")
    print(pd.DataFrame(leads).to_string())
    print(f"\nRecorte {INICIO.date()} ao fim (rec) e depois de {C.SPLIT.date()} (fora):")
    print(T.to_string())
    print(f"\nPASSA = 8/8 e FP/mês rec <= {CORTE} x OCSVM ({CORTE * ref:.3f}), e 3/3 fora com FP/mês <= "
          f"{CORTE * REF_FP['fora']:.3f}.")


if __name__ == "__main__":
    main()
