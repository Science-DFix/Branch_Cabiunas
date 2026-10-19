#!/usr/bin/env python3
"""D2 -- CNN-1D de vibração (mensal desde 01/2025) como canal EXTRA da OCSVM, não como substituto. Local, sem treino.

PRÉ-REGISTRO (18/10/2026, antes de rodar qualquer braço)
Pergunta (do usuário): o CNN-1D mensal de vibração (`cnn1d_mensal_jan_vib`, task 892e93c1, rodada 1d) somado à
pipeline de produção ajuda? Na 1d ele acendeu em trips que a OCSVM pegava tarde (09/12/2025: 8,4 -> 21,1 h) e cobriu o
29/04/2025, mas como substituto subiu o FP.

Diagnóstico descritivo feito antes do pré-registro (produção, 01/2025 ao fim, 62 episódios: 8 detecções,
18 inconclusivos, 36 FP):
  - 61 de 62 episódios têm o canal 4 aceso; o disparo é quase sempre PI_6240319_AL e/ou PAL_6240315, as duas tags
    mais recorrentes do catálogo (409 e 326 ativações desde 07/2024; só 27 e 31 em [T-48 h, T+24 h] de algum trip).
    FP e detecção têm a MESMA composição: canal 4 + 1 modelo, quase sempre a vibração.
  - O CNN-1D de vibração acende junto com a OCSVM de vibração na maioria dos episódios (FP e detecções). Ele fica
    apagado ou quase apagado em alguns FP (31/01-12/02/2026, 25/03/2026) e também no trip de 26/02/2026 (2% do
    episódio).

Braços (camada de decisão da produção nos demais pontos: canal 4 = 5 tags/24 h, filtro 45 min, refratário 48 h):
  ref  produção: voto >= 2 de {temp, vib, óleo, c4}
  A    soma: CNN-1D vib como 5º canal, voto >= 2 de {temp, vib, óleo, c4, vib_cnn}
  B    confirmação: o voto da vibração exige as duas famílias, vib := vib_ocsvm AND vib_cnn; voto >= 2 de 4
Recorte: 01/01/2025 ao fim (o CNN-1D só emite flag desde 01/2025), e fora da amostra depois de 01/07/2025.
Critério (o da D1): PASSA = 8/8 no recorte e FP/mês <= 0,75 x produção no recorte, e 3/3 fora com FP/mês <= 0,75 x
produção fora. Também relatar a antecedência por trip.
Expectativa: A mantém 8/8 e SOBE o FP (o CNN acende quase onde a OCSVM acende; um canal a mais só soma votos) -> NÃO.
B corta FP (~20-30%), mas arrisca perder o trip de 26/02/2026 (CNN quase apagado) -> provável NÃO, por 7/8.
Ressalva: a votação 2-de-2 entre famílias da era EXP10c (modelo único) foi refutada; B é a versão por grupo, com o
canal 4 mantido.

RESULTADO (18/10/2026, depois do pré-registro 37e0ca1): NENHUM PASSA.
  braço  recorte 01/2025->         fora 07/2025->
  ref    8/8, 36 FP (3,051/mês)    3/3, 23 FP (2,877)
  A      8/8, 59 FP (5,001)        3/3, 39 FP (4,878)   +64% de FP; antecedência 43,2->32,8 h (07/04) e 36,7->12,4 h
                                                        (29/04), 8,4->21,2 h (09/12)
  B      8/8, 34 FP (2,882)        3/3, 21 FP (2,627)   -6% / -9%; antecedências idênticas à produção
Expectativa: acertei A. Em B errei o trip (o 26/02/2026 sobrevive, pelo óleo + canal 4) e superestimei o corte
(previ 20-30%, deu 6-9%). O CNN-1D quase sempre concorda com a OCSVM de vibração, então exigir os dois tira poucos FP.

Uso: python frente_ocsvm/d2_canal_extra.py
"""
from __future__ import annotations
import json

import pandas as pd
import comum as C
from a4_variabilidade import canais_da_replica
from d1_disputa import PROD, TASKS_JSON, CORTE

INICIO = pd.Timestamp("2025-01-01")


def main():
    reg = json.loads(TASKS_JSON.read_text())
    df = canais_da_replica(PROD)
    vcnn = canais_da_replica(dict(PROD, vib=reg["cnn1d_mensal_jan_vib"]))["canal_vibracao"]
    vcnn = vcnn.reindex(df.index, fill_value=False)
    idx, op = df.index, df["operational_state"]
    cat = C.catalogo()
    base = {k: df[k] for k in C.CANAIS[:3]}
    base["c4"] = C.canal_alarme(idx, cat.loc[cat.tag.isin(C.TAGS_C4), "t"], C.REF["janela_c4_h"])
    bracos = {"ref": base,
              "A_soma": dict(base, vib_cnn=vcnn),
              "B_confirmacao": dict(base, **{C.CANAIS[1]: base[C.CANAIS[1]] & vcnn})}
    ft = C.trips(idx)
    leads, L = {}, []
    for nome, canais in bracos.items():
        cls, _ = C.avalia(C.decide(canais, 2, C.REF["dur_min"], C.REF["refrat_h"]), op, ft)
        P = C.por_trip(cls, ft).set_index("trip")
        leads[nome] = P.lead_h
        linha = dict(braco=nome)
        for rec, ini in (("rec", INICIO), ("fora", C.SPLIT)):
            dias = C.compute_operational_period_days(pd.DataFrame({"operational_state": op[idx >= ini]}))
            fp = int(((cls.start >= ini) & (cls.classe == "falso_positivo")).sum())
            p = P[P.index >= ini]
            linha.update({f"det_{rec}": f"{int(p.deteccao.notna().sum())}/{len(p)}", f"FP_{rec}": fp,
                          f"FPmes_{rec}": round(fp / (dias / 30.4375), 3)})
        L.append(linha)
    T = pd.DataFrame(L).set_index("braco")
    ref_rec, ref_fora = T.loc["ref", "FPmes_rec"], T.loc["ref", "FPmes_fora"]
    T["veredito"] = ["—" if b == "ref" else
                     ("PASSA" if T.loc[b, "det_rec"] == "8/8" and T.loc[b, "FPmes_rec"] <= CORTE * ref_rec
                      and T.loc[b, "det_fora"] == "3/3" and T.loc[b, "FPmes_fora"] <= CORTE * ref_fora
                      else "NÃO") for b in T.index]
    pd.set_option("display.width", 200)
    print("Antecedência por trip (h; NaN = não detectado):")
    print(pd.DataFrame(leads).to_string())
    print(f"\nRecorte {INICIO.date()} ao fim (rec) e depois de {C.SPLIT.date()} (fora):")
    print(T.to_string())
    print(f"\nPASSA = 8/8 e FP/mês rec <= {CORTE * ref_rec:.3f}, e 3/3 fora com FP/mês <= {CORTE * ref_fora:.3f}.")


if __name__ == "__main__":
    main()
