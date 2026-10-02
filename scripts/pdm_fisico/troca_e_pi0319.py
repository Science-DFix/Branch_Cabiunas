#!/usr/bin/env python3
"""H1 (CUSUM zerado na troca de bundle) e H2 (PI_0319 fora do canal p).

PRÉ-REGISTRADO EM 01/10/2026, ANTES DE RODAR (commit deste arquivo sem resultado).

DE ONDE VÊM. As duas nasceram de olhar estes mesmos 8 eventos, e isso fica dito antes:
  H1  `memoria_nas_comparacoes.py`: no 09/12/2025 da composição 4, o bundle de
      novembro via o PDI_0301 a ~7x o limiar até a troca de 04/12; o bundle de
      dezembro vê o p normal, e o CUSUM levou o artefato 4 dias para dentro dele. O
      acumulador nunca reinicia na troca mensal -- nunca foi testado.
  H2  `f1_exclusao.py`: ~85% do corte de carga do F1 (-31 h/mês) veio de tirar o
      PI_0319 (gás do motor de partida) do p nos 53 dias em que o monitor o deu como
      travado. É sensor de estado: IQR semanal ~0,3 / ~0,007 / ~45 conforme a linha
      está pressurizada; o erro de reconstrução do p salta com o estado da linha, não
      com a saúde da máquina.

O DESENHO. As 8 composições de sempre, pareadas com a referência.
  H1  o detector de sempre, com o acumulador do CUSUM zerado em cada troca de bundle
      (30 instantes de reinício a partir do corte, 0,25^30 ~ 1e-18 -- a convenção de
      `recentragem.com_reset`). O degrau não muda. Nos 60 min do reinício o CUSUM
      não acumula; o degrau segue vigiando.
  H2  o canal p refeito em todos os bundles com 11 dos 12 sensores, sem o PI_0319.
      MESMAS LINHAS de ajuste da referência (dropna sobre os 36 sensores): mudar a
      composição do baseline é efeito conhecido e forte (F3, DPCA -- o dia 1 cai de 8
      para 4-5), e confundiria o teste. Em produção, `constroi_bundle` teria de manter
      a mesma seleção de linhas. t, sp e vb não mudam. `confere()` prova que o mesmo
      laço com os 12 sensores reproduz o p da referência
      (diferença relativa máxima 2e-4 no dia 1, a de sempre do recálculo).
DECISÃO: `bootstrap_regua.decide`, pareado, IC 97,5% em cada (Bonferroni sobre as
duas). GANHO = detecção estrita + carga cai em >= 7/8 + IC abaixo de zero + FP não
sobe. Se as duas derem GANHO, a combinação H1 + H2 roda e também tem de passar (IC
95%) para que entrem juntas; se só uma der, só ela segue.

EXPECTATIVA REGISTRADA.
  H1  a carga cai (a idade do bundle sobe o duty do p de 41% para 63% em um mês, e
      esse acumulado atravessa a troca). Risco alto em A: cada evento tem um corte
      nos 7 dias anteriores em ~2 das 8 composições, e um precursor que começou antes
      da troca perde a evidência acumulada.
  H2  a carga cai; a dúvida é se o PI_0319 carrega algum dos trips. Expectativa mais
      alta que a de H1, e por isso mesmo vale menos: o resultado do F1 já indicava a
      direção.
RESSALVA: hipóteses a posteriori nos mesmos 8 eventos. Um GANHO aqui é candidato a
validação em dado novo e a discussão com a engenharia, não decisão de produção.

Uso:  PYTHONPATH=. python troca_e_pi0319.py confere   # equivalência do laço de H2
      PYTHONPATH=. python troca_e_pi0319.py           # o teste
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
import regua_fp as R
import aprovacao_operador as AO
import bootstrap_regua as BR

C, DF, IX, STABLE, FIT, DC = AO.C, AO.DF, AO.IX, AO.STABLE, AO.FIT, AO.DC
N_RESET = 30
FORA = "954005_624_PI_0319"
P_SEM = [c for c in C.PRESSURE_TAGS if c != FORA]


def bundles(dia: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    cs = [c for c in AO.cortes(dia) if AO.bundle(c) is not None]
    return [(c, cs[k + 1] if k + 1 < len(cs) else AO.FIM) for k, c in enumerate(cs)]


def trocas(dia: int) -> np.ndarray:
    """reinício do acumulador nos N_RESET instantes a partir de cada troca."""
    r = np.zeros(len(IX), bool)
    for c, _ in bundles(dia):
        i = int(IX.searchsorted(c))
        r[i:i + N_RESET] = True
    return r


def canal_p(dia: int, cols) -> np.ndarray:
    p = R.sinais(dia)[1].copy()
    for c0, c1 in bundles(dia):
        F = DF.loc[STABLE & (IX < c0), C.SENSOR_TAGS].dropna().tail(FIT)
        s = (IX >= c0) & (IX < c1)
        if s.any():
            p[s] = DC.ScorerMax().fit(F[cols]).score(DF.loc[s, cols])["pca_recon"].to_numpy()
    return p


def sem_pi0319(dia: int):
    f = R.CACHE / f"sem_pi0319_d{dia:02d}.npz"
    t, p, ms, ds = R.sinais(dia)
    if f.exists():
        return t, np.load(f)["p"], ms, ds
    p2 = canal_p(dia, P_SEM)
    np.savez(f, p=p2)
    return t, p2, ms, ds


def confere(dia: int = 1) -> float:
    """max |dif| relativa entre o laço com os 12 sensores e o p da referência."""
    ref = R.sinais(dia)[1]
    p = canal_p(dia, list(C.PRESSURE_TAGS))
    ok = np.isfinite(ref) & np.isfinite(p)
    return float(np.max(np.abs(p[ok] - ref[ok]) / np.maximum(np.abs(ref[ok]), 1e-9)))


def linha(nome, x, ref, var, nivel):
    r, v = ref.tabela, var.tabela
    print(f"{nome:14s} det {x['det']} início {x['inicio']} banda {x['banda']} FP {v.fp_mes.median():.3f}"
          f" | Δcarga {x['d_carga']:+.1f} [{x['ic_lo']:+.1f}; {x['ic_hi']:+.1f}] IC {100 * nivel:.1f}%"
          f" | ΔFP {x['d_fp']:+.3f} | carga cai em {int((v.carga_mes.values < r.carga_mes.values).sum())}/8"
          f" | {x['veredito']}", flush=True)
    print("     por composição: " + "  ".join(
        f"d{d}: {r.det[d]}→{v.det[d]}/{r.banda[d]}→{v.banda[d]}/{r.carga_mes[d]:.0f}→{v.carga_mes[d]:.0f}"
        for d in R.DIAS), flush=True)


def main():
    ref = BR.braco("referência", lambda d: R.detector(*R.sinais(d))["fin"], R.DIAS)
    print(f"referência     det {ref.tabela.det.median()} início {ref.tabela.inicio.median()} banda {ref.tabela.banda.median()}"
          f" FP {ref.tabela.fp_mes.median():.3f} carga {ref.tabela.carga_mes.mean():.1f}", flush=True)
    h1 = BR.braco("H1", lambda d: R.detector(*R.sinais(d), reset_extra=trocas(d))["fin"], R.DIAS)
    h2 = BR.braco("H2", lambda d: R.detector(*sem_pi0319(d))["fin"], R.DIAS)
    x1, x2 = (BR.decide(ref, h, pareado=True, nivel=0.975) for h in (h1, h2))
    linha("H1 troca", x1, ref, h1, 0.975)
    linha("H2 sem PI_0319", x2, ref, h2, 0.975)
    if x1["veredito"] == "GANHO" and x2["veredito"] == "GANHO":
        h12 = BR.braco("H1+H2", lambda d: R.detector(*sem_pi0319(d), reset_extra=trocas(d))["fin"], R.DIAS)
        linha("H1 + H2", BR.decide(ref, h12, pareado=True, nivel=0.95), ref, h12, 0.95)
    pd.DataFrame([dict(h="H1", **x1), dict(h="H2", **x2)]).to_csv(R.CACHE / "troca_e_pi0319.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "confere":
        print("max |dif| relativa (dia 1):", confere(1))
    else:
        main()
