#!/usr/bin/env python3
"""A0 -- a pipeline de referência se reproduz localmente, sem ClearML?

PRÉ-REGISTRADO EM 10/10/2026, ANTES DE RODAR. É conferência, não experimento: tudo o que vier depois
(A1, A2, A3, B1, B2, B3) parte daqui, então primeiro se prova que a reconstrução local é a pipeline.

TRÊS CONFERÊNCIAS, todas têm de passar:
  1. CANAL 4. Recalculado do catálogo em cache (Dataset a97ba56b, 5 tags, janela 24 h) tem de ser
     idêntico, instante a instante, à coluna `canal_alarme_processo` de `point_anomalies_final.csv`.
  2. DECISÃO. Voto >= 2 dos 4 canais da tabela -> filtro de 45 min -> refratário de 48 h tem de ser
     idêntico à coluna `is_anom_point` da mesma tabela.
  3. RÉGUA. As métricas têm de bater com `metrics.json` de 08/09: 8/8, 42 FP, 25 inconclusivos,
     2,8847 FP/mês; e a 1ª detecção por trip, com a tabela 7 do relatório operacional.

EXPECTATIVA. Passa nas três (o mesmo código, os mesmos artefatos).

Uso:  python frente_ocsvm/a0_reproducao.py
"""
from __future__ import annotations
import pandas as pd
import comum as C

REF_METRICAS = dict(falhas_detectadas=8, n_episodios_falso_positivo=42, n_episodios_inconclusivo=25,
                    falso_positivo_por_mes=2.8847252290052112)
REF_DETECCAO = ["2025-02-25 22:51", "2025-03-16 11:04", "2025-04-06 02:05", "2025-04-10 21:15",
                "2025-04-27 14:22", "2025-11-03 16:40", "2025-12-09 00:15", "2026-02-26 11:48"]


def main():
    df = C.carrega_canais()
    idx, op = df.index, df["operational_state"]
    ft = C.trips(idx)
    print(f"{len(idx)} instantes ({idx.min()} .. {idx.max()}), {len(ft)} trips na janela", flush=True)

    cat = C.catalogo()
    c4 = C.canal_alarme(idx, cat.loc[cat.tag.isin(C.TAGS_C4), "t"], C.REF["janela_c4_h"])
    d1 = int((c4.values != df["canal_alarme_processo"].astype(bool).values).sum())
    print(f"1. canal 4 recalculado x tabela: {d1} instantes diferentes  -> {'OK' if d1 == 0 else 'FALHA'}")
    print(f"   duty do canal 4: {c4.mean():.1%} do total, {c4[op == 'on'].mean():.1%} do tempo 'on'")

    canais = {c: df[c].astype(bool) for c in C.CANAIS}
    fin = C.decide(canais, C.REF["min_votes"], C.REF["dur_min"], C.REF["refrat_h"])
    d2 = int((fin.values != df["is_anom_point"].astype(bool).values).sum())
    print(f"2. decisão recalculada x tabela: {d2} instantes diferentes  -> {'OK' if d2 == 0 else 'FALHA'}")

    cls, m = C.avalia(fin, op, ft)
    ok3 = all(abs(m[k] - v) < 1e-9 for k, v in REF_METRICAS.items())
    print(f"3. régua: {m['falhas_detectadas']}/{m['n_falhas_catalogadas']} trips, {m['n_episodios_falso_positivo']} FP, "
          f"{m['n_episodios_inconclusivo']} inconclusivos, {m['falso_positivo_por_mes']:.4f} FP/mês  -> {'OK' if ok3 else 'FALHA'}")
    P = C.por_trip(cls, ft)
    det = P.deteccao.dt.strftime("%Y-%m-%d %H:%M").tolist()
    ok4 = det == REF_DETECCAO
    print(f"   1ª detecção por trip x tabela 7: {'OK' if ok4 else 'FALHA'}")
    print(P.to_string(index=False))
    print(f"\nA0: {'PASSOU' if d1 == 0 and d2 == 0 and ok3 and ok4 else 'NÃO PASSOU'}")


if __name__ == "__main__":
    main()
