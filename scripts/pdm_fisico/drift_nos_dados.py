#!/usr/bin/env python3
"""O drift nos DADOS, não no detector: o que deriva, como, e por quê.

POR QUÊ. Todo o trabalho de drift anterior atacou o detector -- tamanho do
baseline, ensemble de bundles, suavizar o p99, separar PCA e escala, tempo de
coerência, condicionar à carga, limpar o baseline. Tudo refutado ou inconclusivo
(`drift_*.py`, `residuo_carga.py`). Ninguém olhou o que, nos sensores, está
derivando. É daí que devem sair as ideias para o retreino.

O CATÁLOGO JÁ LEVANTA UMA SUSPEITA. A família "pressão" do modelo inclui tags que
derivam por motivos que não são saúde da máquina:
  PDI_0338   diferencial do filtro de óleo       -- entope e é trocado: dente de serra
  PDI_0301   diferencial do filtro de gás de selagem  -- idem
  PI_5134001 header de ar de instrumentação      -- utilidade
E há uma tag que o modelo não usa, HSX_6240001A, "Comando Manutenção".

AS QUATRO PERGUNTAS:
  A. quem comanda os canais t e p -- que sensor é o máximo quando o canal acende;
  B. o perfil de drift de cada tag, semana a semana, em operação estável;
  C. o que é o HSX_6240001A;
  D. quanto cada mês se desloca em relação ao baseline que o pontua.

RESULTADO (30/09/2026).

A. Cada canal é comandado por UM sensor, na prática.
   t   quando acende, o máximo é o TI_0305 (mancal radial LNA) em 68,5% do tempo --
       o MESMO sensor do canal sp. Explica a redundância t-sp (lift 1,57).
   p   quando acende: PDI_0301 (filtro de gás de selagem) 40%, PDIT_0305 (selagem
       primária LA) 22%, PDI_0317 (gás combustível) 15%, PDI_0338 (filtro de óleo)
       7%. Seis de cada dez horas do p aceso são selagem e filtro. Em nov/2025 o p
       ficou aceso 96% do mês, 100% pelo PDI_0301.
B. As tags que mais derivam: PDI_0302 (selagem, degraus de +-208 sigma), PDI_0301
   (razão 10), as vibrações TV_354 e TV_351 (razão 6-8, 30+ saltos de 3 sigma),
   PI_0319 (gás do motor de PARTIDA -- irrelevante com a máquina rodando -- razão
   6,8, 31 saltos) e o TI_0305 (razão 5,3, faixa de 46 sigma).
C. HSX_6240001A liga só com a máquina parada: é o modo de manutenção. 79
   acionamentos, 14 manutenções de >= 24 h, a maior de 47 dias (jul-ago/2024).
D. Todo mês se desloca contra o próprio baseline: a tag mediana anda ~1 sigma, a
   pior 1,2 a 18 sigma. Em jan/2025, 99% dos pontos do PDI_0338 ficaram fora da
   faixa que o baseline conhecia; em nov/2025, 93% dos do PDI_0301.

Uso:  PYTHONPATH=. python drift_nos_dados.py
"""
from __future__ import annotations
import io, contextlib
import numpy as np, pandas as pd
import regua_fp as R
with contextlib.redirect_stdout(io.StringIO()):
    import drift_composicao as DC
    import pos_processamento as PP

C, DET, DF, IX, STABLE, FIT = DC.C, DC.DET, DC.DF, DC.IX, DC.STABLE, DC.FIT
G = PP.g
VIG = (PP.estavel & ~PP.blk).to_numpy()
CURTO = lambda s: s.replace("954005_624_", "")


def quem_comanda():
    """A. Por mês (retreino no dia 1): que sensor é o máximo do canal quando ele passa
    do limiar do nível A, e em operação normal."""
    cortes = [c for c in pd.date_range(IX[0].normalize().replace(day=1), IX[-1], freq="MS", tz="UTC")
              if IX[0] < c < IX[-1]]
    thr = {"temperatura": R.DB.BASE["t"] * R.DB.K_LO["t"], "pressao": R.DB.BASE["p"] * R.DB.K_LO["p"]}
    tags = {"temperatura": C.TEMPERATURE_TAGS, "pressao": C.PRESSURE_TAGS}
    conta = {f: pd.Series(0.0, index=tags[f]) for f in tags}
    conta_acima = {f: pd.Series(0.0, index=tags[f]) for f in tags}
    por_mes = []
    for i, c0 in enumerate(cortes):
        c1 = cortes[i + 1] if i + 1 < len(cortes) else IX[-1] + pd.Timedelta("2min")
        if c0 < R.DB.alvo.min() - pd.Timedelta(days=60) and c0 < pd.Timestamp("2025-01-01", tz="UTC"):
            continue
        fit = DF.loc[STABLE & (IX < c0), C.SENSOR_TAGS].dropna().tail(FIT)
        if len(fit) < FIT // 4:
            continue
        s = (IX >= c0) & (IX < c1) & VIG
        linha = {"mes": c0.strftime("%Y-%m")}
        for fam in tags:
            sc = DC.ScorerMax().fit(fit[tags[fam]])
            X = DF.loc[s, tags[fam]].dropna()
            Xs = sc.scaler.transform(X)
            e = (Xs - sc.pca.inverse_transform(sc.pca.transform(Xs))) ** 2
            ne = e / sc.sens_p99_ / sc.recon_p99
            canal = ne.max(axis=1); arg = ne.argmax(axis=1)
            nomes = np.array(tags[fam])
            conta[fam] += pd.Series(nomes[arg]).value_counts().reindex(tags[fam], fill_value=0)
            ac = canal > thr[fam]
            conta_acima[fam] += pd.Series(nomes[arg[ac]]).value_counts().reindex(tags[fam], fill_value=0)
            if ac.any():
                top = pd.Series(nomes[arg[ac]]).value_counts()
                linha[f"{fam}_top"] = f"{CURTO(top.index[0])} ({100 * top.iloc[0] / ac.sum():.0f}%)"
                linha[f"{fam}_acima"] = 100 * ac.mean()
        por_mes.append(linha)
    return conta, conta_acima, pd.DataFrame(por_mes)


def perfil_drift():
    """B. Semana a semana, em operação estável."""
    tags = C.TEMPERATURE_TAGS + C.PRESSURE_TAGS + C.VIBRATION_TAGS
    X = G.loc[VIG, tags]
    sem = X.groupby(X.index.to_period("W"))
    n = sem.size()
    ok = n[n >= 2 * 720].index                      # semanas com >= 2 dias estáveis
    med = sem.median().loc[ok]
    dent = sem.agg(lambda v: 1.4826 * (v - v.median()).abs().median()).loc[ok]
    L = []
    for c in tags:
        m = med[c].dropna(); sd = float(np.nanmedian(dent[c]))
        if len(m) < 10 or not np.isfinite(sd) or sd <= 0:
            L.append(dict(tag=CURTO(c), semanas=len(m), razao_drift=np.nan)); continue
        dm = m.diff().dropna() / sd
        L.append(dict(tag=CURTO(c), semanas=len(m),
                      razao_drift=float(m.std() / sd),
                      faixa_sigmas=float((m.max() - m.min()) / sd),
                      maior_queda=float(dm.min()), maior_subida=float(dm.max()),
                      saltos_3sig=int((dm.abs() > 3).sum())))
    return pd.DataFrame(L).sort_values("razao_drift", ascending=False)


def hsx():
    """C. O 'Comando Manutenção'."""
    h = G["HSX_6240001A"]
    vc = h.value_counts(dropna=False).head(6)
    liga = (h.fillna(0) > 0.5) & ~(h.shift().fillna(0) > 0.5)
    ev = h.index[liga.to_numpy()]
    op = PP.op
    L = []
    for t0 in ev:
        seg = h.loc[t0:]
        fim = seg[~(seg.fillna(0) > 0.5)].index
        t1 = fim[0] if len(fim) else h.index[-1]
        L.append(dict(inicio=t0, horas=(t1 - t0).total_seconds() / 3600,
                      maquina_de_pe=bool(op.loc[t0])))
    return vc, pd.DataFrame(L)


def deslocamento_mensal():
    """D. Mês M contra o baseline que o pontua (20.000 pontos estáveis antes do dia 1)."""
    tags = C.TEMPERATURE_TAGS + C.PRESSURE_TAGS
    cortes = list(pd.date_range(pd.Timestamp("2025-01-01", tz="UTC"), IX[-1], freq="MS"))
    L = []
    for i, c0 in enumerate(cortes):
        c1 = cortes[i + 1] if i + 1 < len(cortes) else IX[-1] + pd.Timedelta("2min")
        base = DF.loc[STABLE & (IX < c0), tags].dropna().tail(FIT)
        mes = DF.loc[STABLE & (IX >= c0) & (IX < c1), tags].dropna()
        if len(mes) < 720:
            continue
        sd = 1.4826 * (base - base.median()).abs().median()
        d = ((mes.median() - base.median()) / sd.replace(0, np.nan)).abs()
        lo, hi = base.quantile(0.005), base.quantile(0.995)
        fora = ((mes < lo) | (mes > hi)).mean()
        L.append(dict(mes=c0.strftime("%Y-%m"), maior_desloc=CURTO(d.idxmax()), sigmas=float(d.max()),
                      desloc_mediano=float(d.median()),
                      pior_cobertura=CURTO(fora.idxmax()), fora_da_faixa=float(fora.max()),
                      **{CURTO(k): float(v) for k, v in d.items()}))
    return pd.DataFrame(L)


def main():
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
    conta, acima, pm = quem_comanda()
    for fam in ("temperatura", "pressao"):
        print(f"\nA. QUEM COMANDA O CANAL {'t' if fam == 'temperatura' else 'p'} "
              f"— sensor que é o máximo (jan/2025 a abr/2026, operação vigiada)")
        T = pd.DataFrame({"sempre %": 100 * conta[fam] / conta[fam].sum(),
                          "quando acende %": 100 * acima[fam] / max(acima[fam].sum(), 1)})
        T.index = [CURTO(i) for i in T.index]
        print(T.sort_values("quando acende %", ascending=False).round(1).to_string())
    print("\n   por mês — quem mais acende cada canal:")
    print(pm.round(1).to_string(index=False))
    B = perfil_drift()
    print("\nB. PERFIL DE DRIFT POR TAG — semanas estáveis; razão = desvio entre semanas / ruído dentro da semana")
    print(B.round(2).to_string(index=False))
    vc, ev = hsx()
    print("\nC. HSX_6240001A (Comando Manutenção) — valores:"); print(vc.to_string())
    print(f"   acionamentos: {len(ev)}")
    if len(ev):
        print(ev.assign(inicio=ev.inicio.dt.strftime("%Y-%m-%d %H:%M")).round(1).to_string(index=False))
    D = deslocamento_mensal()
    print("\nD. DESLOCAMENTO DO MÊS CONTRA O SEU BASELINE (|mediana do mês − mediana do baseline| / σ do baseline)")
    print(D[["mes", "maior_desloc", "sigmas", "desloc_mediano", "pior_cobertura", "fora_da_faixa"]].round(2).to_string(index=False))
    tops = D[[c for c in D.columns if c not in ("mes", "maior_desloc", "sigmas", "desloc_mediano",
                                                "pior_cobertura", "fora_da_faixa")]].median().sort_values(ascending=False)
    print("\n   deslocamento mediano por tag (σ), os maiores:"); print(tops.head(10).round(2).to_string())
    B.to_csv(R.CACHE / "drift_perfil.csv", index=False); D.to_csv(R.CACHE / "drift_mensal.csv", index=False)
    pm.to_csv(R.CACHE / "drift_quem_comanda.csv", index=False)


if __name__ == "__main__":
    main()
