#!/usr/bin/env python3
"""Um alerta de `recon_p99` (ou de span do baseline) no retreino prediz o mês ruim?

VALIDAÇÃO A POSTERIORI (02/10/2026), 16 bundles do pacote (2025-01 a 2026-04), composição do
dia 1. Existe porque a recomendação "alertar quando o recon_p99 sair da faixa histórica" foi
feita (`o-normalizador-e-o-ponto-fragil`) sem nunca ser medida contra o custo do mês.

MEDIDAS por mês servido: salto do recon_p99 de t e de p contra o bundle anterior, span do
baseline (dias de calendário), e três desfechos do detector de referência nesse mês: carga
(h de FP + NEUTRO por 730 h vigiadas), duty do canal t e duty do canal p (nível A).

RESULTADO (n = 15 meses com bundle anterior, Spearman):
  salto_t  x carga   rho +0,07 (p 0,81)     salto_p  x carga   rho +0,15 (p 0,59)
  span     x carga   rho -0,33 (p 0,23)     p99_t    x carga   rho +0,27 (p 0,33)
  p99_p    x carga   rho +0,02 (p 0,95)
  A única com p < 0,05 (salto_t x duty_p, rho -0,64, p 0,01) sai de 15 testes e não sobrevive
  a Bonferroni (0,003), e vai na direção oposta à do alerta. O salto normal do p99 mês a mês:
  quantis 10/50/90% = 0,79/1,03/1,29 (t) e 0,71/1,00/1,33 (p).
CONCLUSÃO: não há alerta de recon_p99, e span e folga do baseline marcam saída da faixa
validada, não bundle ruim. Poder baixo (n = 15; carga mensal dominada por poucos episódios
gigantes): não prova ausência de efeito, mas não sustenta um alerta.

Uso:  PYTHONPATH=. python valida_alerta_p99.py
"""
from __future__ import annotations
import io, contextlib, json, glob
import numpy as np, pandas as pd
from scipy import stats

with contextlib.redirect_stdout(io.StringIO()):
    import regua_fp as R

PKG = "../../estrutura_pra_prod/Cabiunas/modelos/"


def main():
    B = []
    for d in sorted(glob.glob(PKG + "model_*")):
        m = json.load(open(d + "/modelo.json")); n = json.load(open(d + "/normalizacao.json"))
        B.append(dict(mes=m["mes_servido"], p99_t=n["temperatura"]["recon_p99"], p99_p=n["pressao"]["recon_p99"],
                      span=(pd.Timestamp(m["baseline_fim"]) - pd.Timestamp(m["baseline_inicio"])).days))
    B = pd.DataFrame(B).set_index("mes")
    for c in ("t", "p"):
        B["salto_" + c] = B["p99_" + c] / B["p99_" + c].shift(1)
    out = R.detector(*R.sinais(1)); cls = R.mede(out["fin"])["cls"]
    vig = R.mask.to_numpy(); mes = pd.Series(R.idx.strftime("%Y-%m"), index=R.idx)
    carga = np.zeros(len(R.idx), bool)
    for a, b, k, _ in cls:
        if k in ("FP", "NEUTRO"):
            carga[(R.idx >= a) & (R.idx <= b)] = True
    rows = []
    for mm in B.index:
        s = (mes == mm).to_numpy() & np.asarray(R.sel)
        h_op = vig[s].sum() / 30
        rows.append(dict(mes=mm, carga=carga[s].sum() / 30 / max(h_op, 1) * 730,
                         duty_t=out["A"]["t"].to_numpy()[s].sum() / max(vig[s].sum(), 1),
                         duty_p=out["A"]["p"].to_numpy()[s].sum() / max(vig[s].sum(), 1)))
    D = B.join(pd.DataFrame(rows).set_index("mes")); v = D.dropna(subset=["salto_t"])
    print(D.round(2).to_string())
    for x in ("salto_t", "salto_p", "span", "p99_t", "p99_p"):
        for y in ("carga", "duty_t", "duty_p"):
            r, p = stats.spearmanr(v[x], v[y]); print(f"  {x:8s} x {y:6s}: rho {r:+.2f} (p {p:.2f})")


if __name__ == "__main__":
    main()
