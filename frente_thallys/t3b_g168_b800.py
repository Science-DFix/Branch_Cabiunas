#!/usr/bin/env python3
"""T3b -- a regra do T3 ("carga baixa OU sem alarme de processo -> atenção") nas bases G168 e B800 do Thallys.

PRÉ-REGISTRO (05/11/2026, commitado antes de rodar)
Origem: o T3 (6886520, resultado bc9526c) saiu ROBUSTO raspando nas bases principais, mas declarou que faltavam G168 e
  B800, as bases onde a carga perdeu o 17/03 no EXP 61/62 do Thallys. Elas estão no Dataset 86e3990f
  (controle_v2/{G168,B800}/serie.csv.gz, ledgers das variantes da sombra), liberado com a volta do Cica em 05/11.
O que já foi visto (declarado): nessas bases, só os números publicados pelo Thallys: sem regra G168 8/8 FP 14 e
  B800 8/8 FP 12; carga sozinha (L < 0,5) 7/8 FP 6 nas duas, perdendo o 17/03. A combinação com o canal 4 NUNCA foi
  rodada nelas.
Pergunta: a combinação mantém a detecção da carga e corta FP além dela em bases que ninguém olhou com essa regra?

REGRA: exatamente a PRINC do T3 (θ 0,5, T5, referência de 30 d, canal 4 com as 5 tags e janela de 24 h, confirmação
  em [nascimento, nascimento + 2 h]). Episódios de `ci._episodios` desde 01/02/2025. Régua do Thallys (`R.mede`), com o
  `vigiado` de cada ledger.
PLACEBO: o do T3 (500 permutações dos rebaixamentos, semente 20261026), por base.
CRITÉRIOS (fixados antes; perda e FP contra a mesma base sem regra):
  (a) nas duas bases, FP <= 6 e perda <= 1 detecção.
  (c) placebo p_conj < 0,10 nas duas bases.
  (d) acréscimo sobre a carga: em >= 1 das 2 bases, FP menor que a carga sozinha com a mesma detecção.
  PASSA = (a) e (c) e (d). FRÁGIL = (a) sem (c) ou (d). NÃO = (a) falha.
  A vizinhança do T3 (8 variações) sai só como relato, não conta.
EXPECTATIVAS: a perda da carga (17/03) se mantém nas duas (~85%); o canal 4 não custa mais nenhum trip (~55%; o risco
  é o 26/02); a combinação tira mais 1-3 FP por base (~65%). (a) ~55%, (c) ~55%, (d) ~65%. Veredito PASSA ~30%,
  FRÁGIL ~30%, NÃO ~40%.
LIMITES: 8 trips; G168 e B800 são variantes do mesmo detector, sobre os mesmos dados e trips; as 5 tags do canal 4 foram
  escolhidas nos mesmos 8 trips.
VERIFICAÇÕES: V1 sem regra, G168 8/8 FP 14 e B800 8/8 FP 12; V2 só carga, 7/8 FP 6 nas duas (EXP 61/62); V3 o
  `atual` do mesmo Dataset reproduz o controle local (7/8 FP 9; carga 7/8 FP 4; combinação 7/8 FP 2, já vista no T1).
  Se V1 ou V2 falhar, para sem rodar a regra.

RESULTADO: (a preencher depois de rodar)

Uso:  python frente_thallys/t3b_g168_b800.py
"""
from __future__ import annotations
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(AQUI), str(AQUI.parent / "frente_ocsvm")]
import reproduz as R  # noqa: E402  (antes do nv: fixa o NV_DADOS)
import t3_carga_ou_canal4 as T3  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DS = Path.home() / ".clearml/cache/storage_manager/datasets/ds_86e3990fde354c039ce38aab508f344d/controle_v2"
REF0 = {"atual": (7, 9), "G168": (8, 14), "B800": (8, 12)}
REFC = {"atual": (7, 4), "G168": (7, 6), "B800": (7, 6)}
NOVAS = ["G168", "B800"]


def le(v):
    s = pd.read_csv(DS / v / "serie.csv.gz"); s["ts"] = pd.to_datetime(s["ts"], utc=True); s = s.set_index("ts").sort_index()
    s = s.reindex(pd.date_range(s.index[0], s.index[-1], freq="2min"))
    return s["alarme"].fillna(0).astype(bool), s["vigiado"].fillna(0).astype(bool)


def main():
    ci, _ = R.modelos_ci()
    E62, IND, cat = T3.prepara()
    B = {v: le(v) for v in REF0}
    EPS = {v: list(ci._episodios(al.loc[R.INICIO8:])) for v, (al, _) in B.items()}

    def mede(v, cfg, flags=None):
        al, vig = B[v]
        f = T3.rebaixa(al, EPS[v], cfg, E62, IND, cat) if flags is None else flags
        return R.mede(T3.T1.aplica(al, EPS[v], [not x for x in f]), vig), f

    m0 = {v: R.mede(*B[v]) for v in B}
    mc = {v: mede(v, dict(T3.PRINC, usa_c4=False))[0] for v in B}
    ok = {"V1": all((m0[v]["det"], m0[v]["fp"]) == REF0[v] for v in B),
          "V2": all((mc[v]["det"], mc[v]["fp"]) == REFC[v] for v in B)}
    for v in B:
        print(f"   {v:5s} sem regra {m0[v]['det']}/8·{m0[v]['fp']:2d} (ref {REF0[v]}) | só carga {mc[v]['det']}/8·{mc[v]['fp']:2d} "
              f"(ref {REFC[v]})", flush=True)
    print(f"verificações: {ok}", flush=True)
    if not all(ok.values()):
        print("verificação falhou -- parando"); return

    print("\nPRINCIPAL (carga < 0,5 OU sem canal 4)   sem regra -> só carga -> combinação | placebo p_conj")
    R1, P1 = {}, {}
    rng = np.random.default_rng(T3.SEMENTE)
    for v in B:
        m1, f = mede(v, T3.PRINC); R1[v] = m1
        if v in NOVAS:
            ff = np.array(f); cont = 0
            for _ in range(T3.N_PERM):
                mp, _ = mede(v, T3.PRINC, list(rng.permutation(ff))); cont += (mp["fp"] <= m1["fp"]) and (mp["det"] >= m1["det"])
            P1[v] = cont / T3.N_PERM
        perd = [e for e in m0[v]["ev"] if m0[v]["ev"][e] and not m1["ev"][e]]
        print(f"   {v:5s} [{'nova' if v in NOVAS else 'já vista (V3)'}] {m0[v]['det']}/8·{m0[v]['fp']:2d} -> "
              f"{mc[v]['det']}/8·{mc[v]['fp']:2d} -> {m1['det']}/8·{m1['fp']:2d} ({m1['horas']:4.0f} h, {m1['fp_mes']:.3f}/mês) "
              f"perdidos {perd}" + (f" | p_conj {P1[v]:.3f}" if v in P1 else ""), flush=True)
    print(f"V3 (atual = 7/8 FP 2): {(R1['atual']['det'], R1['atual']['fp']) == (7, 2)}")

    print("\nVIZINHANÇA (só relato: det/8·FP)")
    for cfg in T3.VIZ:
        r = {v: mede(v, cfg)[0] for v in NOVAS}
        print(f"   {T3.nome_cfg(cfg)} | " + " | ".join(f"{v} {r[v]['det']}/8·{r[v]['fp']:2d}" for v in NOVAS), flush=True)

    a = all(R1[v]["fp"] <= 6 and m0[v]["det"] - R1[v]["det"] <= 1 for v in NOVAS)
    c = all(P1[v] < 0.10 for v in NOVAS)
    nd = sum(R1[v]["fp"] < mc[v]["fp"] and R1[v]["det"] == mc[v]["det"] for v in NOVAS); d = nd >= 1
    cl = "NÃO" if not a else "PASSA" if (c and d) else "FRÁGIL"
    print(f"\nCLASSIFICAÇÃO: (a) {'ok' if a else 'não'} | (c) placebo < 0,10 em {sum(P1[v] < 0.10 for v in NOVAS)}/2 | "
          f"(d) acréscimo sobre a carga em {nd}/2 -> {cl}")


if __name__ == "__main__":
    main()
