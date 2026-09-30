#!/usr/bin/env python3
"""O teste de Kolmogorov-Smirnov acrescenta algo ao monitor de drift?

O MONITOR ATUAL (`cabiunas_inference.monitor_drift`) mede o DESLOCAMENTO DO CENTRO:
a mediana da semana em sigmas do baseline do bundle, com alerta em >= 10 sigma por
duas semanas seguidas. O KS mede a maior distância entre as distribuições
acumuladas (D, de 0 a 1) e pega mudança de nível, de espalhamento e de forma.

A LACUNA CONCRETA. Em nov/2025, depois da manutenção, o PDI_0301 não só mudou de
nível: o IQR dele caiu de 0,063 para 0,003 kgf/cm². Sensor que de repente fica quase
constante é sintoma de instrumento travado ou rezerado. A razão de IQR (semana /
baseline) mede isso de forma direta -- é a checagem que se pretende acrescentar.

AS TRÊS REGRAS, todas por tag, todas com persistência de duas semanas seguidas:
  mediana   |mediana da semana - centro| / (IQR/1,349) >= 10       (a atual)
  travado   IQR da semana / IQR do baseline <= R, R calibrado
  KS        D >= D*, D* calibrado para disparar na mesma fração de semanas que a
            regra da mediana dispara sem persistência (~11%)
Nenhuma usa p-valor. O p-valor do KS é relatado só para mostrar por que não serve:
com milhares de pontos autocorrelacionados, quase toda semana dá p ~ 0.

CRITÉRIO, ESCRITO ANTES DE RODAR. O KS entra no monitor só se gerar alertas
persistentes em semanas que NEM a mediana NEM o travado pegam, e se essas semanas
mostrarem uma mudança física interpretável (espalhamento, forma, bimodalidade).
Se só repetir as outras duas, fica de fora.

RESULTADO (30/09/2026) -- o KS FICA DE FORA; a regra de travado entra, reformulada.

  · p-valor: < 0,05 em 99,9% das 1.674 comparações -- inútil, como esperado;
  · o D SATURA: o limiar calibrado para disparar como a mediana cai em D* = 1,000,
    o máximo. Em 13% das semanas alguma tag já tem as distribuições totalmente
    separadas do baseline, e daí em diante 3 sigma e 300 sigma dão o mesmo D = 1;
  · só-KS persistente: 2 semanas. 29/09/2025 PI_0308, D = 1 com desvio de -3,6 sigma
    (saturação num sensor estreito -- artefato); 15/10/2025 spread do mancal, o
    início da excursão de outubro, uma semana antes da regra da mediana -- mas com
    a mediana já em 17 sigma, que o monitor dá como "atencao". Nada de novo.
  · CORREÇÃO: a persistência NÃO separa instrumento de máquina. A excursão do mancal
    de out/2025 durou duas semanas e dá degrau persistente pela mediana.
  · a regra "travado" com R fixo (IQR da semana <= 10% do IQR do baseline) dispara
    em 15 semanas, puxada pelo PI_0319 (gás do motor de partida, que varia pouco por
    natureza). Contar valores repetidos não funciona na grade interpolada. A versão
    que entra compara cada sensor com o PRÓPRIO típico nas semanas anteriores: IQR
    < 10% do típico por 3 semanas. Dispara em 6 de 59 semanas, só em PDI_0301 e
    PI_0319, só depois da manutenção de nov/2025.

Uso:  PYTHONPATH=. python ks_drift.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
import drift_nos_dados as DN
import regua_fp as R

C, DF, IX, STABLE, FIT, VIG = DN.C, DN.DF, DN.IX, DN.STABLE, DN.FIT, DN.VIG
TAGS = C.TEMPERATURE_TAGS + C.PRESSURE_TAGS
MANCAL = ["954005_624_TI_0305", "954005_624_TI_0301", "954005_624_TI_0303", "954005_624_TI_0307"]
CURTO = DN.CURTO


def spread(X):
    return X[MANCAL[0]] - X[MANCAL[1:]].mean(axis=1)


def medidas() -> pd.DataFrame:
    cortes = list(pd.date_range(pd.Timestamp("2025-01-01", tz="UTC"), IX[-1], freq="MS"))
    rng = np.random.default_rng(0)
    L = []
    for i, c0 in enumerate(cortes):
        c1 = cortes[i + 1] if i + 1 < len(cortes) else IX[-1]
        base = DF.loc[STABLE & (IX < c0), C.SENSOR_TAGS].dropna().tail(FIT)
        B = base[TAGS].copy(); B["spread_mancal"] = spread(base)
        for w0 in pd.date_range(c0, c1, freq="7D"):
            w1 = min(w0 + pd.Timedelta(days=7), c1)
            sel = (IX >= w0) & (IX < w1) & VIG
            if sel.sum() < 2 * 720:
                continue
            W = DF.loc[sel, C.SENSOR_TAGS].dropna()
            W = W[TAGS].assign(spread_mancal=spread(W))
            for c in B.columns:
                b, w = B[c].to_numpy(), W[c].to_numpy()
                iqr_b = np.subtract(*np.quantile(b, [.75, .25]))
                iqr_w = np.subtract(*np.quantile(w, [.75, .25]))
                sig = iqr_b / 1.349
                ks = ks_2samp(b, w)
                L.append(dict(semana=w0, tag=CURTO(c),
                              desloc=(np.median(w) - np.median(b)) / sig if sig > 0 else np.nan,
                              razao_iqr=iqr_w / iqr_b if iqr_b > 0 else np.nan,
                              D=ks.statistic, p=ks.pvalue,
                              valores_unicos=len(np.unique(np.round(b, 6))) / len(b)))
    return pd.DataFrame(L)


def persistente(M: pd.DataFrame, flag: pd.Series) -> pd.DataFrame:
    """semanas em que a MESMA tag dispara nesta e na semana anterior (7 dias antes)."""
    F = M.loc[flag, ["semana", "tag"]].copy()
    chave = set(zip(F.semana, F.tag))
    F["persist"] = [(s - pd.Timedelta(days=7), t) in chave for s, t in zip(F.semana, F.tag)]
    return F[F.persist]


def main():
    M = medidas()
    semanas = sorted(M.semana.unique())
    pd.set_option("display.width", 200)
    print(f"{len(semanas)} semanas x {M.tag.nunique()} variáveis = {len(M)} comparações\n")
    print(f"P-VALOR DO KS: p < 0,05 em {100 * (M.p < 0.05).mean():.1f}% das comparações; "
          f"p < 1e-10 em {100 * (M.p < 1e-10).mean():.1f}% -- inútil como gatilho, como esperado")
    quant = M.groupby("tag").valores_unicos.first().sort_values().head(4)
    print(f"variáveis mais quantizadas (fração de valores distintos no baseline): "
          + ", ".join(f"{t} {v:.4f}" for t, v in quant.items()))

    # calibração: fração de semanas em que "alguma variável" dispara, sem persistência
    semana_max = M.groupby("semana").agg(dmax=("D", "max"), shift=("desloc", lambda x: np.nanmax(np.abs(x))),
                                          rmin=("razao_iqr", "min"))
    alvo = float((semana_max["shift"] >= 10).mean())
    D_star = float(np.quantile(semana_max.dmax, 1 - alvo))
    R_low = 0.10
    print(f"\nCALIBRAÇÃO: a regra da mediana (>= 10 sigma) dispara em {100 * alvo:.0f}% das semanas;"
          f" D* = {D_star:.3f} dispara na mesma fração.")
    print(f"  razão de IQR mínima por semana, quantis 5/10/25/50%: "
          f"{np.round(semana_max.rmin.quantile([.05, .1, .25, .5]).to_numpy(), 3).tolist()}"
          f"  -> travado se razão <= {R_low}")

    regras = {"mediana": M.desloc.abs() >= 10, "travado": M.razao_iqr <= R_low, "KS": M.D >= D_star}
    P = {k: persistente(M, f) for k, f in regras.items()}
    sem = {k: set(v.semana) for k, v in P.items()}
    print("\nALERTAS PERSISTENTES (a mesma variável em duas semanas seguidas)")
    for k, v in P.items():
        print(f"  {k:8s} {len(sem[k]):2d} semanas: " + "; ".join(
            f"{s:%Y-%m-%d} {', '.join(sorted(v[v.semana == s].tag))}" for s in sorted(sem[k])))
    so_ks = sem["KS"] - sem["mediana"] - sem["travado"]
    print(f"\nSEMANAS QUE SÓ O KS PEGA: {len(so_ks)}")
    for s in sorted(so_ks):
        tg = P["KS"][P["KS"].semana == s].tag
        X = M[(M.semana == s) & (M.tag.isin(tg))][["tag", "D", "desloc", "razao_iqr"]]
        print(f"  {s:%Y-%m-%d}"); print(X.round(3).to_string(index=False))
    nov = pd.Timestamp("2025-11-15", tz="UTC")
    print("\nNOV/2025 — PDI_0301 semana a semana:")
    print(M[(M.tag == "PDI_0301") & (M.semana >= nov - pd.Timedelta(days=21)) &
            (M.semana <= nov + pd.Timedelta(days=21))][["semana", "desloc", "razao_iqr", "D"]]
          .assign(semana=lambda x: x.semana.dt.strftime("%Y-%m-%d")).round(3).to_string(index=False))
    M.to_csv(R.CACHE / "ks_drift.csv", index=False)


if __name__ == "__main__":
    main()
