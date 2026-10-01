#!/usr/bin/env python3
"""F3 (item 1.6 do plano do especialista): PCA em dado limpo, p99 no baseline inteiro.

PRÉ-REGISTRADO EM 01/10/2026, ANTES DE RODAR (commit deste arquivo sem resultado).

POR QUE O 1.6 FALHOU (`baseline_limpo.py`, `baseline-precisa-de-anormalidade`).
Excluir +-7 dias em volta dos trips do baseline levou a carga de 6,6 para 68,0 h/mês
no dia 1. O mecanismo medido: o score é dividido pelo p99 do próprio baseline; sem as
janelas pré-falha, que são a cauda alta, o p99 desce, todo score sobe e o duty
explode. A idade do fit quase não mudou. Não era o PCA aprendendo a falha -- era a
escala.

A CORREÇÃO. Separar as duas coisas: o MODELO (RobustScaler + PCA) é ajustado nos
20.000 pontos estáveis mais recentes SEM as janelas em volta dos trips já ocorridos;
a CALIBRAÇÃO (p99 por sensor e p99 do score) é calculada com esse modelo sobre o
baseline INTEIRO de sempre (os 20.000 pontos estáveis mais recentes, com as janelas).
O par continua atômico: o p99 é sempre o daquele PCA, só que medido na amostra
completa. O spread (sp) e a vibração não mudam.

AS VARIANTES: exclusão de +-7 dias (PRIMÁRIA, IC 95%) e de +-48 h (secundária, IC
97,5%). Mesmos limiares, nada reajustado. Decisão: `bootstrap_regua.decide`, desenho
pareado nas 8 composições.

EXPECTATIVA REGISTRADA: efeito pequeno. As janelas pré-trip são no máximo ~7% (48 h)
a ~25% (7 dias) do ajuste, e um PCA de 95% da variância quase não muda com isso. Se o
PCA não aprende a falha, limpar o ajuste não muda nada; se aprende, a detecção sobe.

RESULTADO (01/10/2026) -- AS DUAS VARIANTES REPROVAM.

    variante            det   início banda  FP/mês  Δcarga [IC]             carga cai em
    referência          6,5    6,0   4,5   0,861   --
    +-7 d (primária)    5,0    4,5   3,5   0,732   +3,6 [-6,9; +17,5] 95%   5/8
    +-48 h (secundária) 5,5    4,5   3,0   0,732  +13,1 [+0,1; +30,3] 97,5%  3/8

  · A expectativa registrada (efeito pequeno) errou onde mais importa: na composição
    do dia 1 -- a publicada -- a detecção cai de 8 para 4 e a carga vai de 49 para
    142 (+-7 d) ou 135 (+-48 h). Nas outras sete o efeito é de +-1 evento e poucas
    horas, para os dois lados.
  · O mecanismo não é "o PCA aprendia a falha". Tirar as janelas e completar os
    20.000 pontos puxa o baseline para mais longe no passado: muda a COMPOSIÇÃO, e o
    dia 1 é justamente o ponto mais sensível a composição (o melhor de nove). A
    calibração separada preserva a escala, mas não protege da composição.
  · O FP cai na mediana (0,861 -> 0,732) e a carga não: é a fronteira de novo, com
    perda de detecção. Item 1.6 fechado nas duas formas.

Uso:  PYTHONPATH=. python f3_calibracao_separada.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import regua_fp as R
import aprovacao_operador as AO
import bootstrap_regua as BR

C, DF, IX, STABLE, FIT, DC = AO.C, AO.DF, AO.IX, AO.STABLE, AO.FIT, AO.DC
VARIANTES = (("+-7 d", pd.Timedelta(days=7), True), ("+-48 h", pd.Timedelta(hours=48), False))


class ScorerSeparado(DC.ScorerMax):
    """ScorerMax com modelo e calibração ajustados em amostras diferentes."""

    def fit_separado(self, limpo: pd.DataFrame, cheio: pd.DataFrame) -> "ScorerSeparado":
        self.fit(limpo)
        X = cheio.dropna()[self.cols]
        Xs = self.scaler.transform(X)
        e = (Xs - self.pca.inverse_transform(self.pca.transform(Xs))) ** 2
        p = np.nanpercentile(e, 99, axis=0)
        self.sens_p99_ = np.maximum(p, self.PHI * np.nanmedian(p))
        r, _ = self._raw_scores(X)
        self.recon_p99 = float(np.nanpercentile(r, 99))
        return self


def sinais(dia: int, janela: pd.Timedelta, rotulo: str):
    f = R.CACHE / f"f3_{rotulo.replace('+-', 'pm').replace(' ', '')}_d{dia:02d}.npz"
    if f.exists():
        z = np.load(f); return z["t"], z["p"], z["ms"], z["ds"]
    t, p, ms, ds = (x.copy() for x in R.sinais(dia))
    cs = AO.cortes(dia)
    for i, c0 in enumerate(cs):
        c1 = cs[i + 1] if i + 1 < len(cs) else AO.FIM
        cheio = DF.loc[STABLE & (IX < c0), C.SENSOR_TAGS].dropna().tail(FIT)
        if len(cheio) < FIT // 4:
            continue
        ok = STABLE & (IX < c0)
        for tt in R.alvo:
            if tt < c0:
                ok &= ~((IX >= tt - janela) & (IX <= tt + janela))
        limpo = DF.loc[ok, C.SENSOR_TAGS].dropna().tail(FIT)
        s = (IX >= c0) & (IX < c1)
        if not s.any() or len(limpo) < FIT // 4:
            continue
        w = DF.loc[s]
        t[s] = ScorerSeparado().fit_separado(limpo[C.TEMPERATURE_TAGS], cheio[C.TEMPERATURE_TAGS]) \
            .score(w[C.TEMPERATURE_TAGS])["pca_recon"].to_numpy()
        p[s] = ScorerSeparado().fit_separado(limpo[C.PRESSURE_TAGS], cheio[C.PRESSURE_TAGS]) \
            .score(w[C.PRESSURE_TAGS])["pca_recon"].to_numpy()
    np.savez(f, t=t, p=p, ms=ms, ds=ds)
    return t, p, ms, ds


def main():
    ref = BR.braco("referência", lambda d: R.detector(*R.sinais(d))["fin"], R.DIAS)
    for rot, jan, prim in VARIANTES:
        var = BR.braco(rot, lambda d, j=jan, r=rot: R.detector(*sinais(d, j, r))["fin"], R.DIAS)
        nivel = 0.95 if prim else 1 - 0.05 / 2
        x = BR.decide(ref, var, pareado=True, nivel=nivel)
        r, v = ref.tabela, var.tabela
        print(f"{rot:7s} {'(primária)' if prim else '(secundária)'}  det {x['det']} início {x['inicio']} banda {x['banda']}"
              f" FP {v.fp_mes.median():.3f} | Δcarga {x['d_carga']:+.1f} [{x['ic_lo']:+.1f}; {x['ic_hi']:+.1f}]"
              f" IC {100 * nivel:.1f}% | ΔFP {x['d_fp']:+.3f} | carga cai em {int((v.carga_mes.values < r.carga_mes.values).sum())}/8"
              f" | {x['veredito']}", flush=True)
        print("     por composição: " + "  ".join(
            f"d{d}: {r.det[d]}→{v.det[d]}/{r.banda[d]}→{v.banda[d]}/{r.carga_mes[d]:.0f}→{v.carga_mes[d]:.0f}" for d in R.DIAS),
            flush=True)


if __name__ == "__main__":
    main()
