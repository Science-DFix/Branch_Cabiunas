#!/usr/bin/env python3
"""A escala do RobustScaler colapsa em tags quase constantes -- e o canal vira outro.

DIAGNÓSTICO DE MECANISMO (02/10/2026), não candidato. Não mede detecção, início,
banda, FP nem carga: nenhum número aqui olha os 8 trips. Serve para decidir o que
pré-registrar.

O ACHADO QUE MOTIVA. Nos 16 bundles do pacote, o PCA de pressão retém de 1 a 8
componentes (k = 1 em 8 dos 16) e o de temperatura de 1 a 3. A causa é o
`RobustScaler`: ele divide cada tag pelo IQR do baseline, e nas pressões diferenciais
(PDIT_0305, PDI_0338, PDI_0301) esse IQR é 0,002-0,005; no PI_0319 (gás do motor de
partida, sensor de estado) vai de 0,0045 a 45,4 conforme a linha passou o mês
despressurizada ou não. Uma variação física pequena vira dezenas ou milhares de
unidades escaladas, uma tag concentra a variância, e o critério "95% da variância"
passa a significar "a tag mais quieta". Depois o erro é ELEVADO AO QUADRADO, passa
pela EWMA (média, não robusta) e só então o CUSUM corta em 20x o limiar -- tarde
demais para impedir que um valor gigante segure a EWMA acima do limiar por horas.

O QUE SE MEDE, nas 8 composições, por bundle e família (t, p):
  k              componentes retidos
  escala/IQR     a escala do bundle contra o IQR de longo prazo da tag (todas as
                 horas estáveis): < 0,1 = escala degenerada naquele mês
  cru > m x lim  fração dos instantes vigiados com o score CRU acima de m vezes o
                 limiar do nível A (m = 1, 20, 100, 1000)
  quem acende    das horas com a EWMA acima do limiar A, a tag com a maior EWMA do
                 próprio erro normalizado (o argmax da família)
  pós-brilho     das horas com a EWMA acima do limiar A, as que não têm NENHUMA
                 amostra crua acima do limiar nas 2 h anteriores
  massa saturada do incremento positivo do CUSUM, a parte vinda de instantes com a
                 EWMA em >= 20x o limiar (o teto do incremento)

RESULTADO (02/10/2026) -- EM 2/3 DOS MESES O CANAL p NÃO É MULTIVARIADO.

  · Componentes retidos, 219 bundles nas 8 composições: pressão k = 1 em 149 (68%),
    até 9; temperatura de 1 a 4 (k = 2 em 130). Nenhuma escala degenerada em t.
  · Por quê (composição 1, 27 bundles): depois do RobustScaler UMA tag concentra
    97,9% da variância escalada (mediana; mínimo 25%) -- o PI_0319 em 24 dos 27, com
    carga^2 = 1,000 no PC1. Ele é intermitente: o IQR vê a linha despressurizada e
    ignora as subidas a ~45, que viram milhares de IQR. Com StandardScaler (variância
    unitária, o padrão de PCA-MSPC) nenhuma tag passa de 8,3% e k fica em 8-10 todo
    mês; com o robusto, k salta de um mês para o outro (dez/25 1, jan/26 8, fev/26 1).
  · Então, em 2/3 dos meses, o "PCA de pressão" reconstrói só o PI_0319, e o p é o
    máximo de 11 desvios UNIVARIADOS ao quadrado, em IQR do mês. As pressões
    diferenciais têm IQR mensal de 0,002-0,006 kgf/cm2 e dois níveis de operação: o
    PDI_0302 (selagem primária, suprimento-balanço) vive em ~1,38 ou ~1,50 -- um salto
    de 0,12 vale 22 IQR, 480 ao quadrado, e o score cru chega a 2.000x o limiar.
  · O normalizador por tag é loteria: o sens_p99 do PDI_0302 é 0,039 num bundle e
    13.452 noutro (o baseline de setembro/25 continha o salto de agosto). O mesmo
    salto físico pontua 2.000x num mês e quase nada no seguinte.
  · O score cru do p é bimodal: passa de 100x o limiar em 0,40% do tempo vigiado e de
    1.000x em 0,37% (máximo 4.167x na mediana das composições; 12.590x do PI_5134001
    em jan/2026). São 104 h por composição com a EWMA do p >= 20x o limiar: o CUSUM
    recebe o incremento máximo (19,25 por amostra) e só drena 0,75. Esses instantes
    são 25% da massa positiva do CUSUM do p (17% do t). O teto de 20x existe, mas é
    aplicado DEPOIS da EWMA, e a EWMA de um trecho a 2.000x leva ~7 h para voltar a
    20x. Pós-brilho (aceso sem amostra crua acima do limiar nas 2 h anteriores): p 10%
    das horas acesas, t 1,7%.
  · Quem acende o p (1.357 h por composição): PDI_0301 30%, PDIT_0305 19%, PI_0339
    15%, PDI_0317 10%, PDI_0302 10%. Quem o satura: PDI_0302 58%, PDI_0301 18%,
    PDIT_0305 13%, PI_5134001 8%. O PI_5134001 é o header de ar de instrumentação da
    U-5134, OUTRA unidade (`metadata.csv`) -- está no canal de saúde do compressor.
  · Releitura do H2 (`troca_e_pi0319.py`): o PI_0319 nunca é a tag de maior erro (0 h
    acesas); ele é o PC1. Tirá-lo não tira um resíduo -- devolve ao PCA a estrutura
    das outras 11. É o mesmo defeito de escala, atacado por um lado só.
  · A EWMA congela sobre NaN (pandas devolve o último valor), mas no histórico isso
    soma 0,1 h vigiada: risco de produção com o historiador fora, não fonte de FP hoje.

Uso:  PYTHONPATH=. python escala_degenerada.py
"""
from __future__ import annotations
import io, contextlib
import numpy as np, pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    import regua_fp as R
    import aprovacao_operador as AO

C, DF, IX, STABLE = AO.C, AO.DF, AO.IX, AO.STABLE
FAM = {"t": list(C.TEMPERATURE_TAGS), "p": list(C.PRESSURE_TAGS)}
LIM = {c: R.DB.BASE[c] * R.DB.K_LO[c] for c in FAM}
MASK = R.mask.to_numpy().astype(bool)
HL = {c: pd.Timedelta(R.DB.HL[c]) for c in FAM}
JAN2H = 60                                   # 2 h em amostras de 2 min
_q = DF.loc[STABLE, C.SENSOR_TAGS]
IQR_G = (_q.quantile(0.75) - _q.quantile(0.25))
curto = lambda c: c.replace("954005_624_", "")


def bundles(dia: int):
    cs = [c for c in AO.cortes(dia) if AO.bundle(c) is not None]
    return [(c, cs[k + 1] if k + 1 < len(cs) else AO.FIM) for k, c in enumerate(cs)]


def por_sensor(sc, X: pd.DataFrame) -> np.ndarray:
    """e_j / sens_p99_j / recon_p99 por tag; o máximo na linha é o `pca_recon`."""
    out = np.full(X.shape, np.nan)
    ok = X.notna().all(axis=1).to_numpy()
    if ok.any():
        Xs = sc.scaler.transform(X[ok])
        e = (Xs - sc.pca.inverse_transform(sc.pca.transform(Xs))) ** 2
        out[ok] = e / sc.sens_p99_ / sc.recon_p99
    return out


def composicao(dia: int):
    linhas = []
    Q = {f: np.full((len(IX), len(cols)), np.nan) for f, cols in FAM.items()}
    for c0, c1 in bundles(dia):
        sct, scp, _, _ = AO.bundle(c0)
        s = (IX >= c0) & (IX < c1)
        for f, sc in (("t", sct), ("p", scp)):
            cols = FAM[f]
            Q[f][s] = por_sensor(sc, DF.loc[s, cols])
            r = sc.scaler.scale_ / IQR_G[cols].to_numpy()
            j = int(np.argmin(r))
            linhas.append(dict(dia=dia, corte=c0, fam=f, k=int(sc.pca.n_components_),
                               pior_tag=curto(cols[j]), escala_iqr=float(r[j]),
                               degeneradas=",".join(curto(cols[i]) for i in np.flatnonzero(r < 0.1)),
                               recon_p99=float(sc.recon_p99)))
    sinal = {}
    for f, cols in FAM.items():
        cru = np.nanmax(np.where(np.isfinite(Q[f]), Q[f], -np.inf), axis=1)
        cru[~np.isfinite(cru)] = np.nan
        ref = R.sinais(dia)[0 if f == "t" else 1]
        ok = np.isfinite(cru) & np.isfinite(ref)
        # o mesmo `pca_recon` da régua; ponto flutuante chega a 1,2e-3 relativo num instante
        assert np.nanmax(np.abs(cru[ok] - ref[ok]) / np.maximum(np.abs(ref[ok]), 1e-9)) < 1e-2
        lim = LIM[f]
        crus = pd.Series(cru, index=IX)
        E = crus.ewm(halflife=HL[f], times=IX).mean().to_numpy()
        Ej = pd.DataFrame(Q[f], index=IX).ewm(halflife=HL[f], times=IX).mean().to_numpy()
        vig = MASK & np.isfinite(E)
        aceso = vig & (E > lim)
        # quem acende: a tag de maior EWMA própria nas horas acesas
        arg = np.nanargmax(np.where(np.isfinite(Ej[aceso]), Ej[aceso], -np.inf), axis=1)
        quem = pd.Series(np.asarray(cols)[arg]).map(curto).value_counts(normalize=True)
        # pós-brilho: aceso sem amostra crua acima do limiar nas 2 h anteriores
        acima = pd.Series(np.nan_to_num(cru, nan=0.0) > lim, index=IX)
        recente = acima.astype(int).rolling(JAN2H, min_periods=1).max().to_numpy().astype(bool)
        # massa do incremento positivo do CUSUM e a parte saturada
        z = np.where(vig, E / lim, np.nan)
        inc = np.clip(np.nan_to_num(z, nan=0.0), None, 20) - R.DB.KAPPA
        pos = inc > 0
        sat = np.nan_to_num(z, nan=0.0) >= 20
        sinal[f] = dict(
            dia=dia, fam=f, h_vigiadas=vig.sum() / 30,
            **{f"cru>{m}x": float(np.mean(cru[vig & np.isfinite(cru)] > m * lim)) for m in (1, 20, 100, 1000)},
            cru_max_x=float(np.nanmax(cru[vig]) / lim),
            duty_ewma=float(aceso.sum() / vig.sum()),
            pos_brilho=float((aceso & ~recente).sum() / max(aceso.sum(), 1)),
            massa_saturada=float(inc[pos & sat].sum() / max(inc[pos].sum(), 1e-9)),
            h_ewma_sat=float((vig & sat).sum() / 30),
            quem=quem)
    return linhas, sinal


def main():
    pd.set_option("display.width", 220)
    B, S = [], []
    for dia in R.DIAS:
        b, s = composicao(dia)
        B += b; S += list(s.values())
        print(f"composição {dia:2d} ok", flush=True)
    B = pd.DataFrame(B)
    B.to_csv(R.CACHE / "escala_degenerada_bundles.csv", index=False)

    print("\n1) COMPONENTES RETIDOS E ESCALA DEGENERADA, por família (todas as composições)")
    for f in FAM:
        x = B[B.fam == f]
        print(f"  {f}: {len(x)} bundles | k: " + ", ".join(f"{k}={n}" for k, n in x.k.value_counts().sort_index().items())
              + f" | bundles com escala < 0,1 x IQR de longo prazo: {int((x.escala_iqr < 0.1).sum())}"
              + f" | recon_p99 {x.recon_p99.min():.2f}-{x.recon_p99.max():.2f}")
        deg = pd.Series([t for v in x.degeneradas if v for t in v.split(",")]).value_counts()
        print("     tags degeneradas (n.º de bundles): " + ", ".join(f"{t} {n}" for t, n in deg.items()))
        print(f"     k contra escala degenerada: k médio {x[x.escala_iqr < 0.1].k.mean():.2f} com, "
              f"{x[x.escala_iqr >= 0.1].k.mean():.2f} sem")

    print("\n2) O SCORE CRU NOS INSTANTES VIGIADOS (mediana das 8 composições)")
    T = pd.DataFrame([{k: v for k, v in s.items() if k != "quem"} for s in S])
    T.to_csv(R.CACHE / "escala_degenerada_sinais.csv", index=False)
    print(T.groupby("fam")[["cru>1x", "cru>20x", "cru>100x", "cru>1000x", "cru_max_x", "duty_ewma",
                            "pos_brilho", "massa_saturada", "h_ewma_sat"]].median().round(4).to_string())

    print("\n3) QUEM ACENDE: fração das horas com a EWMA acima do limiar A (média das 8)")
    for f in FAM:
        q = pd.concat([s["quem"] for s in S if s["fam"] == f], axis=1).fillna(0).mean(axis=1)
        print(f"  {f}: " + "  ".join(f"{t} {100 * v:.0f}%" for t, v in q.sort_values(ascending=False).head(6).items()))


if __name__ == "__main__":
    main()
