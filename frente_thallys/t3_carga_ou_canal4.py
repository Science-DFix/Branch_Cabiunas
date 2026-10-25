#!/usr/bin/env python3
"""T3 -- regra "carga baixa OU sem alarme de processo -> atenção" sobre o detector v2, nos retreinos não publicados.

PRÉ-REGISTRO (25/10/2026, commitado antes de rodar)
Origem e o que já foi visto (declarado):
  - A carga (EXP 62 do Thallys) foi vista em todas as bases daqui: no publicado (9 -> 4) e nos 6 membros do T2.
  - A combinação foi vista pós-hoc no T1 (publicado: 7/8 FP 2) e, como secundário pré-registrado do T2, no comitê
    (7/8 FP 1) e na base n = 19.000 do PCA fiel (5/8 FP 3).
  - A combinação NÃO foi vista nos membros do T2 com a regra. Eles são o teste.
Pergunta: rebaixar a "atenção" o episódio em carga baixa OU sem alarme de processo na notificação corta FP além da
carga sozinha, sem perder detecção, em retreinos que ninguém olhou com essa regra?

BASES PRINCIPAIS: os membros do T2 dentro da faixa validada do pacote: (19.000, 0), (21.000, 0), (20.000, 7 d).
  O de 7 d dispara 1 alerta de faixa em 16 meses.
BASES DE ESTRESSE (só relato): (20.000, 14 d) e (20.000, 21 d). O pacote marca todos os meses delas como "fora da faixa
  histórica: revisar antes de publicar" (código 4); na operação não seriam publicadas.
BASES JÁ VISTAS (só relato, não contam): publicado, comitê do T2, n = 19.000 do PCA fiel.

REGRA PRINCIPAL: episódio (`ci._episodios` desde 01/02/2025) vira atenção se
  carga L < 0,5 (EXP 62: percentil do T5 nas 2 h de notificação contra os 30 d anteriores de operação estável)
  OU canal 4 apagado em [nascimento, nascimento + 2 h] (T1: as 5 tags do canal 4, janela de 24 h).
  O detector não muda.
VIZINHANÇA (um fator por vez, 8): limiar 0,4; 0,6; indicador = média dos TC382_01..06; referência de 15 d; canal 4
  com janela de 12 h; 48 h; só PI_6240319_AL + PAL_6240315; todas as tags do catálogo.
PLACEBO (principal, por base): os rebaixamentos embaralhados entre os episódios da base (500 vezes, semente 20261026):
  rebaixa o MESMO número de episódios ao acaso. p_conj = fração com FP <= o da regra E detecção >= a da regra.
CRITÉRIOS (fixados antes; perda e FP contra a mesma base sem regra):
  (a) principal: nas 3 bases principais, FP <= 6 e perda <= 1 detecção.
  (b) vizinhança: >= 6 das 8 com perda <= 1 em toda base principal e FP médio das 3 <= 7.
  (c) placebo: p_conj < 0,10 em >= 2 das 3 bases principais.
  (d) acréscimo sobre a carga: em >= 2 das 3 bases principais, a combinação tem FP menor que a carga sozinha com a mesma
      detecção. Sem (d), o canal 4 não acrescenta nada à regra do Thallys.
  ROBUSTO = (a) e (b) e (c) e (d). FRÁGIL = (a) sem algum dos outros. NÃO = (a) falha.
  ROBUSTO não é adotar: falta G168 e B800 (onde a carga perdeu o 17/03), o canal 4 tem tags escolhidas nos mesmos
  8 trips, e a validação de verdade é a sombra.
EXPECTATIVAS (a carga sozinha já é conhecida nas principais: 19 mil 4/8 FP 7 com perda de 1; 21 mil 6/8 FP 5;
  7 d 7/8 FP 6):
  A combinação tira mais 1-3 FP por base (~60%) e não perde detecção além da carga (~65%). (a) vale (~55%); (b) vale
  (~40%); (c) vale (~50%); (d) vale (~60%). Veredito ROBUSTO (~20%), FRÁGIL (~45%), NÃO (~35%).
  Estresse: 14 d e 21 d ficam acima de 6 FP (~70%).
LIMITES: 8 trips; as bases compartilham dados e trips, e os membros compartilham a maior parte do baseline; o T5 é um
  indicador imperfeito de carga.
VERIFICAÇÕES: V1 sem regra, as bases reproduzem o T2 (19 mil 5/8 FP 13; 21 mil 6/8 FP 12; 7 d 7/8 FP 16; 14 d 7/8 FP 17;
  21 d 7/8 FP 22); V2 com o canal 4 desligado na regra (só carga), reproduzem o "+ carga" do T2; V3 a regra sem rebaixar
  nada devolve a série original.

RESULTADO (25/10/2026; local, pré-registro 6886520). V1-V3 PASSAM. ROBUSTO pelo critério, mas RASPANDO: (b), (c) e
(d) passam exatamente no limite.
  PRINCIPAIS (sem regra -> só carga -> combinação):
    19 mil: 5/8 FP 13 -> 4/8 FP 7 -> 4/8 FP 4 (0,359/mês). Perde o 17/03, pela carga (como no G168/B800 do Thallys).
      p_conj 0,100, ou seja, NÃO < 0,10.
    21 mil: 6/8 FP 12 -> 6/8 FP 5 -> 6/8 FP 2 (0,179/mês). p_conj 0,004.
    7 d:    7/8 FP 16 -> 7/8 FP 6 -> 6/8 FP 4 (0,359/mês). Perde o 26/02/26, pelo canal 4 (o mesmo custo do T1).
      p_conj 0,008.
  ESTRESSE: 14 d 7/8 FP 17 -> 7/8 FP 8, p 0,010; 21 d 7/8 FP 22 -> 6/8 FP 12 (perde o 26/02), p 0,124.
  JÁ VISTAS: publicado 7/8 FP 2; comitê 7/8 FP 1; n = 19.000 fiel 5/8 FP 3.
  VIZINHANÇA 6/8, no limite. Falham θ 0,6 (3-5/8) e o canal 4 com janela de 12 h (o 19 mil cai para 2/8). Aguentam
    θ 0,4, TC382, referência de 15 d, janela de 48 h e as duas variações de tags (FP médio 3,3-4,3).
  (d) acréscimo sobre a carga 2/3: 19 mil 7 -> 4 e 21 mil 5 -> 2 com a mesma detecção; no de 7 d, 6 -> 4, mas custa o
    26/02.
  CUSTOS QUE SE REPETEM: o 17/03/25 cai pela carga (episódio de 09/03 nasce em carga 0,24, como achou o Thallys) e o
    26/02/26 cai pelo canal 4, quando o episódio travado de 29/01 não o cobre.
PLACAR: ACERTEI: a combinação tira 2-3 FP a mais por base (~60%); (a) vale (~55%); (c) vale (~50%); (d) vale (~60%);
  estresse acima de 6 FP (~70%). ERREI: sem perda além da carga (~65%: o 7 d perde o 26/02); (b) não valeria (dei
  ~40% para valer); veredito ROBUSTO (dei ~20%).
LEITURA: é a primeira regra que passa no critério completo do Thallys em bases que ele não viu, com FP de 2 a 4
  (0,18-0,36/mês). Mas o "robusto" é frágil na margem, tem dois custos conhecidos (17/03 e 26/02) e ainda falta passar
  em G168 e B800. Candidata para a sombra na forma "atenção", nunca supressão.

Uso:  python frente_thallys/t3_carga_ou_canal4.py
"""
from __future__ import annotations
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(AQUI), str(AQUI.parent / "frente_ocsvm")]
import reproduz as R  # noqa: E402  (antes do nv: fixa o NV_DADOS)
import base_n19000 as B  # noqa: E402
import t1_canal4_no_v2 as T1  # noqa: E402
import t2_comite_baselines as T2  # noqa: E402
import comum as C  # noqa: E402

import numpy as np  # noqa: E402

PRINCIPAIS = [(19000, 0), (21000, 0), (20000, 7)]
ESTRESSE = [(20000, 14), (20000, 21)]
REF_T2 = {(19000, 0): (5, 13), (21000, 0): (6, 12), (20000, 7): (7, 16), (20000, 14): (7, 17), (20000, 21): (7, 22)}
REF_CARGA = {(19000, 0): (4, 7), (21000, 0): (6, 5), (20000, 7): (7, 6), (20000, 14): (7, 10), (20000, 21): (7, 13)}
PRINC = dict(theta=0.5, ind="T5", ref_d=30, c4_janela=24.0, c4_tags="c4", usa_c4=True)
VIZ = [dict(PRINC, theta=0.4), dict(PRINC, theta=0.6), dict(PRINC, ind="TC382"), dict(PRINC, ref_d=15),
       dict(PRINC, c4_janela=12.0), dict(PRINC, c4_janela=48.0), dict(PRINC, c4_tags="pi_pal"), dict(PRINC, c4_tags="todas")]
N_PERM, SEMENTE = 500, 20261026


def nome_cfg(c):
    return f"θ {c['theta']} | {c['ind']:5s} | ref {c['ref_d']:2d} d | c4 {c['c4_tags']:6s} {c['c4_janela']:.0f} h"


def prepara():
    from nv import io as nvio
    E62 = B._exp("exp62", "62_regra_de_carga.py")
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"]
    IND = {"T5": g["T5_AVG_A"].where(est), "TC382": g[[f"TC382_0{i}_A" for i in range(1, 7)]].mean(axis=1).where(est)}
    return E62, IND, C.catalogo()


def rebaixa(al, eps, cfg, E62, IND, cat):
    L = [E62.L_de(IND[cfg["ind"]], a, cfg["ref_d"], 2)[0] for a, _ in eps]
    baixa = [np.isfinite(l) and l < cfg["theta"] for l in L]
    if cfg["usa_c4"]:
        conf = T1.confirma(T1.canal4(al.index, cat, dict(tags=cfg["c4_tags"], janela_h=cfg["c4_janela"])), eps, 2.0)
        baixa = [b or not c for b, c in zip(baixa, conf)]
    return baixa


def main():
    ci, det = R.modelos_ci()
    al0, vig = R.carrega_controle(); idx = al0.index
    E62, IND, cat = prepara()
    bases = {T2.nome(m): T2.alarme(T2.proc_de(m), det, idx) for m in PRINCIPAIS + ESTRESSE}
    vistas = {"publicado": al0, "comitê T2": T2.alarme(T2.comite({m: T2.proc_de(m) for m in T2.MEMBROS}, T2.PRINC), det, idx),
              "n=19.000 fiel": B.alarme_n(19000, det)}
    EPS = {k: [(a, b) for a, b in ci._episodios(al.loc[R.INICIO8:])] for k, al in {**bases, **vistas}.items()}
    todas = {**bases, **vistas}

    def mede_regra(k, cfg, flags=None):
        al = todas[k]; f = rebaixa(al, EPS[k], cfg, E62, IND, cat) if flags is None else flags
        return R.mede(T1.aplica(al, EPS[k], [not x for x in f]), vig), f

    ok = {}
    m0 = {k: R.mede(al, vig) for k, al in todas.items()}
    ok["V1"] = all((m0[T2.nome(m)]["det"], m0[T2.nome(m)]["fp"]) == REF_T2[m] for m in REF_T2)
    so_carga = dict(PRINC, usa_c4=False)
    mc = {k: mede_regra(k, so_carga)[0] for k in todas}
    ok["V2"] = all((mc[T2.nome(m)]["det"], mc[T2.nome(m)]["fp"]) == REF_CARGA[m] for m in REF_CARGA)
    ok["V3"] = all(T1.aplica(al, EPS[k], [True] * len(EPS[k])).equals(al) for k, al in todas.items())
    print(f"verificações: {ok}", flush=True)
    if not all(ok.values()):
        print("verificação falhou -- parando"); return

    print("\nPRINCIPAL (carga < 0,5 OU sem canal 4)   sem regra -> só carga -> combinação | placebo p_conj")
    R1, P1 = {}, {}
    rng = np.random.default_rng(SEMENTE)
    for k in todas:
        m1, f = mede_regra(k, PRINC); R1[k] = m1
        if k in bases:
            ff = np.array(f); cont = 0
            for _ in range(N_PERM):
                mp, _ = mede_regra(k, PRINC, list(rng.permutation(ff))); cont += (mp["fp"] <= m1["fp"]) and (mp["det"] >= m1["det"])
            P1[k] = cont / N_PERM
        perd = [e for e in m0[k]["ev"] if m0[k]["ev"][e] and not m1["ev"][e]]
        tipo = "principal" if k in [T2.nome(m) for m in PRINCIPAIS] else "estresse" if k in bases else "já vista"
        print(f"   {k:16s} [{tipo:9s}] {m0[k]['det']}/8·{m0[k]['fp']:2d} -> {mc[k]['det']}/8·{mc[k]['fp']:2d} -> "
              f"{m1['det']}/8·{m1['fp']:2d} ({m1['horas']:4.0f} h, {m1['fp_mes']:.3f}/mês) perdidos {perd}"
              + (f" | p_conj {P1[k]:.3f}" if k in P1 else ""), flush=True)

    pr = [T2.nome(m) for m in PRINCIPAIS]
    print("\nVIZINHANÇA (bases principais: det/8·FP)")
    viz_ok = 0
    for cfg in VIZ:
        r = {k: mede_regra(k, cfg)[0] for k in pr}
        bom = all(m0[k]["det"] - r[k]["det"] <= 1 for k in pr) and np.mean([r[k]["fp"] for k in pr]) <= 7; viz_ok += bom
        print(f"   {nome_cfg(cfg)} | " + " | ".join(f"{r[k]['det']}/8·{r[k]['fp']:2d}" for k in pr)
              + f" | FP médio {np.mean([r[k]['fp'] for k in pr]):.1f} -> {'ok' if bom else 'não'}", flush=True)
    a = all(R1[k]["fp"] <= 6 and m0[k]["det"] - R1[k]["det"] <= 1 for k in pr)
    b = viz_ok >= 6
    c = sum(P1[k] < 0.10 for k in pr) >= 2
    d = sum(R1[k]["fp"] < mc[k]["fp"] and R1[k]["det"] == mc[k]["det"] for k in pr) >= 2
    cl = "NÃO" if not a else "ROBUSTO" if (b and c and d) else "FRÁGIL"
    print(f"\nCLASSIFICAÇÃO: (a) {'ok' if a else 'não'} | (b) vizinhança {viz_ok}/8 | (c) placebo < 0,10 em "
          f"{sum(P1[k] < 0.10 for k in pr)}/3 | (d) acréscimo sobre a carga em "
          f"{sum(R1[k]['fp'] < mc[k]['fp'] and R1[k]['det'] == mc[k]['det'] for k in pr)}/3 -> {cl}")


if __name__ == "__main__":
    main()
