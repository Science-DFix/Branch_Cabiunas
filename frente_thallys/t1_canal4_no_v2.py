#!/usr/bin/env python3
"""T1 -- o nosso canal 4 (alarme de processo nas últimas 24 h) como confirmação dos episódios do detector v2 do Thallys.

PRÉ-REGISTRO (25/10/2026, commitado antes de rodar; nada do canal 4 foi olhado sobre os episódios do v2)
Pergunta: na pipeline OCSVM, o canal 4 é o filtro de custo (A2: sozinho não detecta, mas leva 4,67 -> 2,88 FP/mês
sem perder trip, e só funciona alinhado com os trips). Ele separa também os episódios do v2? Episódio do v2 SEM
alarme de processo perto da notificação vira "atenção" (sai do alarme). O detector não muda; é pós-processamento da
série de alarme, como o EXP 60/62 do Thallys.

BASE: o controle "atual" reproduzido localmente com os bundles refeitos do zero (`reproduz.py`: 7/8, FP 9, 1.457 h).
Episódios = `ci._episodios` desde 01/02/2025 (18), régua `nv.avalia` (a do Thallys).
CANAL 4: `comum.canal_alarme` da frente OCSVM (igual a `alarm_channel_bool` da pipeline final), catálogo do Dataset
a97ba56b, Status ACT*. Fuso conferido antes (não olha resultado): o `PI_6240319_AL` (motor de partida) e o `PAL_6240315`
caem colados às 159 partidas com deslocamento 0 h (23% e 47% a <= 30 min), contra 4-30% com -3 h ou +3 h. O catálogo
fica no relógio dos sensores (UTC), como na nossa pipeline.

REGRA PRINCIPAL: tags = as 5 do canal 4 (PI_6240319_AL, PAL_6240315, PDAL_6240302, TC382_05_A, PAH_6240319), janela 24 h;
o episódio é confirmado se o canal 4 estiver aceso em algum instante de [nascimento, nascimento + 2 h] (a janela de
notificação: duracao_min = 120 min). Não confirmado -> atenção.
VIZINHANÇA (um fator por vez, 8): janela 12 h; 48 h; 6 h; só PI_6240319_AL + PAL_6240315 (envelope do A3); as 5 sem
PI_6240319_AL (S1 do F1); todas as tags do catálogo; avaliação só no instante do nascimento; avaliação em 4 h.
PLACEBOS (principal):
  P1 permutação: as confirmações embaralhadas entre os episódios (500 vezes, semente 20261026, a do EXP 62): suprime o
     MESMO número de episódios ao acaso. p_conj = fração com FP <= o da regra E detecção >= a da regra.
  P2 alinhamento: o catálogo deslocado em +-7, 14, 21, 28 e 35 d (10 deslocamentos), como o N3 do A2. Conta quantos
     igualam ou superam a regra (FP <= e detecção >=).
CRITÉRIO (o do Thallys, com a ressalva de que aqui há UMA base):
  PROMISSOR = principal com FP <= 6 e perda <= 1 detecção; E >= 6 das 8 vizinhas com perda <= 1 e FP <= 7; E P1 p_conj
              < 0,10; E P2 <= 1 de 10 deslocamentos igualando ou superando.
  FRÁGIL = principal ok, mas falha vizinhança, P1 ou P2.  NÃO = principal falha.
  PROMISSOR não é "adotar": exige as outras bases (n = 19.000; G168 e B800 quando houver o Cica ou o Thallys) e a sombra.
SECUNDÁRIOS (só relato): por episódio, o canal 4 e a carga do EXP 62; T1 combinado com o EXP 62 (atenção se sem canal 4
OU carga < 0,5; e atenção só se sem canal 4 E carga < 0,5); quais FP somem e quais trips caem.
EXPECTATIVAS:
  O canal 4 fica aceso em ~98% das primeiras 6 h após a partida (nossa medida) e 6 dos 9 FP do v2 nascem 6,5 h depois
  de ligar: esses devem continuar confirmados. Prevejo FP 9 -> 7-8 (~60%); detecção mantida 7/8 (~70%); principal com
  FP <= 6 (~20%); P1 p_conj < 0,10 (~40%); veredito NÃO (~75%), FRÁGIL (~15%), PROMISSOR (~10%).
  Com 12 h, os de borda ainda confirmam (6,5 h < 12 h). Com 6 h, alguns FP de borda deixam de confirmar.
LIMITES: 8 trips; as 5 tags do canal 4 foram escolhidas olhando os mesmos 8 trips na nossa pipeline (A3: LOEO 5/8 com
tags livres), então a base atual não é fora da amostra para elas; uma base só.
VERIFICAÇÕES: V1 sem a regra, 7/8, FP 9, 1.457 h; V2 o `canal_alarme` recalculado aqui reproduz a coluna
`canal_alarme_processo` de `point_anomalies_final.csv` (0 instantes diferentes); V3 a regra com tudo confirmado
devolve a série original.

Uso:  python frente_thallys/t1_canal4_no_v2.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "frente_ocsvm"))
import reproduz as R  # noqa: E402  (antes do nv: fixa o NV_DADOS)
import comum as C  # noqa: E402

PRINC = dict(tags="c4", janela_h=24.0, aval_h=2.0)
VIZ = [dict(PRINC, janela_h=12.0), dict(PRINC, janela_h=48.0), dict(PRINC, janela_h=6.0),
       dict(PRINC, tags="pi_pal"), dict(PRINC, tags="c4_sem_pi"), dict(PRINC, tags="todas"),
       dict(PRINC, aval_h=0.0), dict(PRINC, aval_h=4.0)]
TAGS = {"c4": C.TAGS_C4, "pi_pal": ["PI_6240319_AL", "PAL_6240315"],
        "c4_sem_pi": [t for t in C.TAGS_C4 if t != "PI_6240319_AL"], "todas": None}
N_PERM, SEMENTE = 500, 20261026
DESLOC_D = [-35, -28, -21, -14, -7, 7, 14, 21, 28, 35]


def canal4(idx_utc, cat, cfg, desloc_d=0):
    t = cat["t"] if TAGS[cfg["tags"]] is None else cat.loc[cat["tag"].isin(TAGS[cfg["tags"]]), "t"]
    return pd.Series(C.canal_alarme(idx_utc.tz_convert(None), t + pd.Timedelta(days=desloc_d), cfg["janela_h"]).values,
                     index=idx_utc)


def confirma(c4, eps, aval_h):
    return [bool(c4.loc[a:a + pd.Timedelta(hours=aval_h)].any()) for a, _ in eps]


def aplica(al, eps, conf):
    a = al.copy()
    for (ini, fim), ok in zip(eps, conf):
        if not ok:
            a.loc[ini:fim] = False
    return a


def carga(eps):
    """L do EXP 62 (principal: T5, referência 30 d, notificação 2 h), pelas funções do próprio script."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp62", R.REPO_T / "experiments" / "62_regra_de_carga.py")
    E62 = importlib.util.module_from_spec(spec); spec.loader.exec_module(E62)
    from nv import io as nvio
    g = nvio.carrega_grade(); t5 = g["T5_AVG_A"].where(nvio.mascaras(g)["estavel"])
    return [E62.L_de(t5, a, 30, 2)[0] for a, _ in eps]


def main():
    ci, _ = R.modelos_ci()
    al, vig = R.carrega_controle()
    eps = [(a, b) for a, b in ci._episodios(al.loc[R.INICIO8:])]
    cat = C.catalogo()
    ok = {}
    m0 = R.mede(al, vig)
    ok["V1"] = (m0["det"], m0["fp"], round(m0["horas"])) == (7, 9, 1457)
    pa = C.carrega_canais()
    c4_30s = C.canal_alarme(pa.index, cat.loc[cat["tag"].isin(C.TAGS_C4), "t"], 24.0)
    ok["V2"] = int((c4_30s != pa["canal_alarme_processo"].astype(bool)).sum()) == 0
    ok["V3"] = bool(aplica(al, eps, [True] * len(eps)).equals(al))
    print(f"V1 base: {m0['det']}/8 FP {m0['fp']} {m0['horas']:.0f} h | V2 canal 4 = pipeline final | V3 identidade -> {ok}")
    if not all(ok.values()):
        print("verificação falhou -- parando"); return

    def roda(cfg, desloc_d=0):
        conf = confirma(canal4(al.index, cat, cfg, desloc_d), eps, cfg["aval_h"])
        return R.mede(aplica(al, eps, conf), vig), conf

    m1, conf = roda(PRINC)
    perd = [e for e in m0["ev"] if m0["ev"][e] and not m1["ev"][e]]
    print(f"\nPRINCIPAL (5 tags | 24 h | notificação 2 h): {m0['det']}/8 FP {m0['fp']} {m0['horas']:.0f} h -> "
          f"{m1['det']}/8 FP {m1['fp']} {m1['horas']:.0f} h  FP/mês {m1['fp_mes']:.3f}  perdidos {perd}")
    rng = np.random.default_rng(SEMENTE); c = np.array(conf); cont = 0
    for _ in range(N_PERM):
        mp = R.mede(aplica(al, eps, rng.permutation(c)), vig); cont += (mp["fp"] <= m1["fp"]) and (mp["det"] >= m1["det"])
    p1 = cont / N_PERM
    p2 = []
    for d in DESLOC_D:
        md, _ = roda(PRINC, d); p2.append((d, md["det"], md["fp"], (md["fp"] <= m1["fp"]) and (md["det"] >= m1["det"])))
    n2 = sum(x[3] for x in p2)
    print(f"   P1 permutação ({N_PERM}): p_conj {p1:.3f}")
    print(f"   P2 deslocamentos: {n2}/10 igualam ou superam | " + " ".join(f"{d:+d}d:{dt}/8·{fp}" for d, dt, fp, _ in p2))

    print("\nVIZINHANÇA")
    viz_ok = 0
    for cfg in VIZ:
        m, _ = roda(cfg); bom = m0["det"] - m["det"] <= 1 and m["fp"] <= 7; viz_ok += bom
        print(f"   tags {cfg['tags']:10s} | janela {cfg['janela_h']:4.0f} h | aval {cfg['aval_h']:.0f} h -> "
              f"{m['det']}/8 FP {m['fp']:2d} {m['horas']:5.0f} h -> {'ok' if bom else 'não'}")
    a = m1["fp"] <= 6 and m0["det"] - m1["det"] <= 1
    cl = "NÃO" if not a else "PROMISSOR" if (viz_ok >= 6 and p1 < 0.10 and n2 <= 1) else "FRÁGIL"
    print(f"\nCLASSIFICAÇÃO: principal {'ok' if a else 'não'} | vizinhança {viz_ok}/8 | P1 {p1:.3f} | P2 {n2}/10 -> {cl}")

    L = carga(eps)
    print("\nSECUNDÁRIO -- por episódio (nascimento UTC, classe, canal 4 na notificação, carga do EXP 62)")
    tr = R.trips()  # classe só para leitura: o episódio cobre a janela de 48 h antes de um trip
    for (a_, b_), cf, l in zip(eps, conf, L):
        trip = any(pd.Timedelta(0) <= t - a_ <= pd.Timedelta(hours=48) or a_ <= t <= b_ for t in tr)
        print(f"   {a_:%d/%m/%y %H:%M}  {'trip' if trip else 'FP  '}  canal4 {'sim' if cf else 'não'}  carga {l:.2f}")
    for nome, f in (("sem c4 OU carga<0,5", lambda cf, l: cf and not (l < 0.5)), ("sem c4 E carga<0,5", lambda cf, l: cf or not (l < 0.5))):
        m = R.mede(aplica(al, eps, [f(cf, l) for cf, l in zip(conf, L)]), vig)
        print(f"   combinado ({nome} -> atenção): {m['det']}/8 FP {m['fp']} {m['horas']:.0f} h")


if __name__ == "__main__":
    main()
