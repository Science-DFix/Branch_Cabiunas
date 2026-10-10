#!/usr/bin/env python3
"""A1 -- o 8/8 e o 2,88 FP/mês separados em dentro e fora da amostra.

PRÉ-REGISTRADO EM 10/10/2026, ANTES DE RODAR. O split dos 3 OCSVM é 01/07/2025
(`AUTOML_OOS_SPLIT_DATE`): limiar e debounce foram calibrados com o período anterior. Toda métrica da
pipeline de referência mistura os dois períodos.

O QUE SE MEDE, na pipeline de referência (A0), por período (dentro = antes do split; fora = depois):
  detecção (de N trips), banda acionável [T-48 h, T-4 h], antecedência mediana, episódios FP,
  inconclusivos, dias de operação vigiada e FP/mês. Um episódio é atribuído ao período em que nasce.
  Também por canal: duty de cada um dos 4 canais em operação, por período (deriva do OCSVM fora da amostra
  aparece como duty diferente).

EXPECTATIVA REGISTRADA. Detecção 5/5 dentro e 3/3 fora (já se sabe da tabela 7); banda 5/5 e 2/3. FP/mês
fora MAIOR que dentro (modelo e limiar envelhecem fora da amostra), da ordem de 3 a 4 contra 2 a 2,5.
Duty dos canais OCSVM maior fora que dentro.

Uso:  python frente_ocsvm/a1_dentro_fora.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import comum as C


def main():
    df = C.carrega_canais()
    idx, op = df.index, df["operational_state"]
    ft = C.trips(idx)
    fin = C.decide({c: df[c].astype(bool) for c in C.CANAIS}, **{k: C.REF[k] for k in ("min_votes", "dur_min", "refrat_h")})
    cls, m = C.avalia(fin, op, ft)
    P = C.por_trip(cls, ft)
    L = []
    for nome, sel in (("dentro", idx < C.SPLIT), ("fora", idx >= C.SPLIT), ("total", np.ones(len(idx), bool))):
        dias = C.compute_operational_period_days(pd.DataFrame({"operational_state": op[sel]}))
        e = cls[(cls.start < C.SPLIT) if nome == "dentro" else (cls.start >= C.SPLIT) if nome == "fora" else cls.start.notna()]
        p = P if nome == "total" else P[P.amostra == nome]
        on = sel & (op == "on").values
        L.append(dict(periodo=nome, trips=len(p), det=int(p.deteccao.notna().sum()), banda=int(p.banda_4h.sum()),
                      lead_med=float(p.lead_h.median()), FP=int((e.classe == "falso_positivo").sum()),
                      inconcl=int((e.classe == "inconclusivo").sum()), dias_op=round(dias, 1),
                      FP_mes=round((e.classe == "falso_positivo").sum() / (dias / 30.4375), 3),
                      **{f"duty_{c.split('_', 1)[1][:4]}": round(float(df.loc[on, c].astype(bool).mean()), 4) for c in C.CANAIS}))
    T = pd.DataFrame(L)
    pd.set_option("display.width", 200)
    print(T.to_string(index=False))
    print(f"\nconferência com o A0: total {m['falhas_detectadas']}/8, {m['n_episodios_falso_positivo']} FP, "
          f"{m['falso_positivo_por_mes']:.4f} FP/mês")


if __name__ == "__main__":
    main()
