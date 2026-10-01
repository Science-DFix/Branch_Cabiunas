#!/usr/bin/env python3
"""R1: biblioteca de referências -- o mês é pontuado contra os K bundles mais recentes.

PRÉ-REGISTRADO em `bootstrap_regua.py` (commit a6ddd6b), antes de rodar.

A HIPÓTESE (do especialista). Boa parte do drift é a máquina voltando a estados que
já foram normais (pós-manutenção, regimes, rezero). Com várias referências, esses
estados ficam cobertos; a degradação continua visível porque se afasta de todas.

A BIBLIOTECA do mês M de cada composição: o bundle do mês, INTACTO, mais os K - 1
bundles mensais anteriores refeitos com +-7 dias em volta dos trips já ocorridos
antes do corte de cada um excluídos do ajuste (a exclusão no bundle do mês derrubaria
o p99 -- o efeito 6,6 -> 68 h/mês -- e confundiria o teste).

AS VARIANTES
  mínimo   t e p = MÍNIMO, entre as K referências, do score normalizado de cada uma;
           sp com o centro/escala da referência que deu o mínimo em t.
  estado   a cada segunda-feira 00:00 UTC (e a cada corte), escolhe a referência cuja
           média de baseline está mais perto (Mahalanobis, covariância Ledoit-Wolf do
           próprio baseline) da média dos 7 dias vigiados anteriores, SÓ em T5_AVG_A,
           TI_0315, TI_0317, PI_0315 e PDI_0317. Com < 1 dia vigiado na janela, fica o
           bundle do mês. O CUSUM não reinicia na troca.
  Família de 4: mínimo K=2 (PRIMÁRIA, IC 95%); mínimo K=3, estado K=2, estado K=3
  (secundárias, IC 98,75%). Decisão pela regra de `bootstrap_regua.decide`: GANHO só
  com detecção estrita + carga em >= 7/8 + IC da carga abaixo de zero + FP não sobe.
  Andar ao longo da fronteira não conta.

EXPECTATIVA REGISTRADA: "estado" baixa (parente da normalização por regime); "mínimo"
deve cortar carga, a dúvida é a detecção.

RESULTADO (01/10/2026) -- AS QUATRO VARIANTES REPROVAM.

    variante              det  início banda  FP/mês  Δcarga [IC]              duty t  duty p
    referência            6,5   6,0   4,5   0,861   --                         33%    41%
    mínimo K=2 (primária) 5,0   4,5   3,0   0,990   -45,5 [-90,0; -6,0] 95%    20%    27%   A, C
    mínimo K=3            4,5   3,5   2,5   1,033   -50,8 [-133,5; +17,1] 98,75%  16%  20%  A, B2, C
    estado K=2            6,5   3,0   2,0   1,206    -2,1 [-52,1; +43,0]       38%    43%   A, B1, B2, C
    estado K=3            6,5   5,0   3,0   1,120    +0,0 [-83,9; +55,0]       46%    46%   A, B1, B2, C

  · O MECANISMO DA HIPÓTESE EXISTE: com o mínimo, o tempo aceso cai abaixo até do
    bundle novo (p 41% -> 27%) e a carga cai nas 8 composições, com IC abaixo de zero.
    Mas é a fronteira de novo: perde 1 a 3 detecções por composição, a banda cai de
    4,5 para 3,0 e o FP SOBE (episódios se partem), como no teto do CUSUM. O mínimo
    cobre os estados antigos e cobre também os precursores.
  · O "estado" não corta carga nenhuma e derruba o nascimento (início 6 -> 3 com
    K=2): trocar de referência toda semana muda a escala do score no meio de uma
    evolução, e o CUSUM soma contra referências diferentes. Expectativa registrada
    (baixa) confirmada.
  · Família fechada. A biblioteca não reduz a dependência de retreino sem cobrar
    detecção.

Uso:  PYTHONPATH=. python r1_biblioteca.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.covariance import LedoitWolf
import regua_fp as R
import aprovacao_operador as AO
import bootstrap_regua as BR
import idade_referencia as IR

C, DF, IX, STABLE, FIT, DC = AO.C, AO.DF, AO.IX, AO.STABLE, AO.FIT, AO.DC
SEL = ["T5_AVG_A", "954005_624_TI_0315", "954005_624_TI_0317", "954005_624_PI_0315", "954005_624_PDI_0317"]
EXCL = pd.Timedelta(days=7)
SEMANA = pd.Timedelta(days=7)
_LIB: dict = {}


def fit_rows(cut: pd.Timestamp, exclui: bool) -> pd.DataFrame:
    ok = STABLE & (IX < cut)
    if exclui:
        for t in R.alvo:
            if t < cut:
                ok &= ~((IX >= t - EXCL) & (IX <= t + EXCL))
    return DF.loc[ok, C.SENSOR_TAGS].dropna().tail(FIT)


def referencia(cut: pd.Timestamp, exclui: bool):
    """(scorer t, scorer p, med sp, mad sp, média e LW do seletor), memoizado."""
    k = (cut, exclui)
    if k not in _LIB:
        F = fit_rows(cut, exclui)
        if len(F) < FIT // 4:
            _LIB[k] = None
        else:
            b = DC.DET._spread_mancal(F)
            S = F[SEL].to_numpy()
            lw = LedoitWolf().fit(S)
            _LIB[k] = dict(t=DC.ScorerMax().fit(F[C.TEMPERATURE_TAGS]), p=DC.ScorerMax().fit(F[C.PRESSURE_TAGS]),
                           ms=float(b.median()), ds=float((b - b.median()).abs().median() * 1.4826),
                           mu=S.mean(axis=0), prec=lw.precision_)
    return _LIB[k]


def biblioteca(cs, i, K):
    """bundle do mês intacto + K-1 anteriores com exclusão."""
    lib = [referencia(cs[i], False)]
    for j in range(i - 1, max(i - K, -1), -1):
        r = referencia(cs[j], True)
        if r is not None:
            lib.append(r)
    return [x for x in lib if x is not None]


def sinais(dia: int, modo: str, K: int):
    f = R.CACHE / f"r1_{modo}_K{K}_d{dia:02d}.npz"
    if f.exists():
        z = np.load(f); return z["t"], z["p"], z["ms"], z["ds"]
    cs = AO.cortes(dia)
    n = len(IX)
    t, p, ms, ds = (np.full(n, np.nan) for _ in range(4))
    for i, c0 in enumerate(cs):
        c1 = cs[i + 1] if i + 1 < len(cs) else AO.FIM
        if referencia(c0, False) is None:
            continue
        lib = biblioteca(cs, i, K)
        s = (IX >= c0) & (IX < c1)
        if not s.any():
            continue
        w = DF.loc[s]
        if modo == "minimo":
            T = np.vstack([r["t"].score(w[C.TEMPERATURE_TAGS])["pca_recon"].to_numpy() for r in lib])
            P = np.vstack([r["p"].score(w[C.PRESSURE_TAGS])["pca_recon"].to_numpy() for r in lib])
            T0 = np.where(np.isnan(T), np.inf, T)
            arg = np.argmin(T0, axis=0)
            t[s] = np.nanmin(T, axis=0) if np.isfinite(T0).any() else np.nan
            p[s] = np.nanmin(P, axis=0)
            ms[s] = np.array([r["ms"] for r in lib])[arg]
            ds[s] = np.array([r["ds"] for r in lib])[arg]
        else:
            seg = pd.date_range(c0.normalize(), c1, freq="W-MON")
            bordas = sorted({c0, *[x for x in seg if c0 < x < c1], c1})
            for a, b in zip(bordas[:-1], bordas[1:]):
                ss = (IX >= a) & (IX < b)
                if not ss.any():
                    continue
                ja = (IX >= a - SEMANA) & (IX < a) & AO.MASK
                r = lib[0]
                if ja.sum() >= 720 and len(lib) > 1:
                    x = DF.loc[ja, SEL].dropna().to_numpy().mean(axis=0)
                    dist = [float((x - q["mu"]) @ q["prec"] @ (x - q["mu"])) for q in lib]
                    r = lib[int(np.argmin(dist))]
                ww = DF.loc[ss]
                t[ss] = r["t"].score(ww[C.TEMPERATURE_TAGS])["pca_recon"].to_numpy()
                p[ss] = r["p"].score(ww[C.PRESSURE_TAGS])["pca_recon"].to_numpy()
                ms[ss], ds[ss] = r["ms"], r["ds"]
    np.savez(f, t=t, p=p, ms=ms, ds=ds)
    return t, p, ms, ds


def duty(d) -> dict:
    nm = IR.normal()
    return {f"duty_{c}": float(d["A"][c].to_numpy()[nm].mean()) for c in ("t", "p")}


def main():
    pd.set_option("display.width", 200)
    ref = BR.braco("referência", lambda d: R.detector(*R.sinais(d))["fin"], R.DIAS)
    dref = pd.DataFrame([duty(R.detector(*R.sinais(d))) for d in R.DIAS]).median()
    print(f"referência: det {ref.tabela.det.median()} início {ref.tabela.inicio.median()} banda {ref.tabela.banda.median()}"
          f" FP {ref.tabela.fp_mes.median():.3f} carga média {ref.tabela.carga_mes.mean():.1f}"
          f"  duty t {100 * dref.duty_t:.0f}% p {100 * dref.duty_p:.0f}%\n", flush=True)
    L = []
    for modo, K, prim in (("minimo", 2, True), ("minimo", 3, False), ("estado", 2, False), ("estado", 3, False)):
        nome = f"{modo} K={K}"
        dets, dut = {}, []
        def g(d):
            dd = R.detector(*sinais(d, modo, K)); dut.append(duty(dd)); return dd["fin"]
        var = BR.braco(nome, g, R.DIAS)
        nivel = 0.95 if prim else 1 - 0.05 / 4
        x = BR.decide(ref, var, pareado=True, nivel=nivel)
        du = pd.DataFrame(dut).median()
        r, v = ref.tabela, var.tabela
        print(f"{nome:12s} {'(primária)' if prim else '(secundária)':13s} det {x['det']} início {x['inicio']} banda {x['banda']}"
              f" FP {v.fp_mes.median():.3f} | Δcarga {x['d_carga']:+.1f} [{x['ic_lo']:+.1f}; {x['ic_hi']:+.1f}] IC {100 * nivel:.2f}%"
              f" | ΔFP {x['d_fp']:+.3f} | duty t {100 * du.duty_t:.0f}% p {100 * du.duty_p:.0f}% | carga cai em "
              f"{int((v.carga_mes.values < r.carga_mes.values).sum())}/8 | {x['veredito']}", flush=True)
        print("      por composição: " + "  ".join(
            f"d{d}: {r.det[d]}→{v.det[d]}/{r.banda[d]}→{v.banda[d]}/{r.carga_mes[d]:.0f}→{v.carga_mes[d]:.0f}" for d in R.DIAS), flush=True)
        L.append(dict(variante=nome, primaria=prim, nivel=nivel, **x, duty_t=du.duty_t, duty_p=du.duty_p))
    pd.DataFrame(L).to_csv(R.CACHE / "r1_biblioteca.csv", index=False)


if __name__ == "__main__":
    main()
