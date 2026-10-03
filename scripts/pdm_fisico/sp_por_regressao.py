#!/usr/bin/env python3
"""O spread do mancal (`sp`) melhora se o TI_0305 for PREVISTO pelos irmãos em vez de subtraído?

DIAGNÓSTICO DE VIABILIDADE (03/10/2026). Não muda o detector nem é candidato: mede se vale pré-registrar um.

DE ONDE VEM. `correlacao_entre_grupos.py`: dentro dos mancais do compressor, TI_0301, 0303 e 0307 se
correlacionam 0,94 a 0,99, mas o TI_0305 (o alvo do `sp`, o mancal radial LNA) só 0,63 com eles. O `sp` atual
é TI_0305 - mediana(irmãos) com a constante da própria referência: supõe que a diferença é estável. Com 0,63,
a diferença oscila sozinha, e essa oscilação é ruído para o `sp`. E o TI_0305 é a variável do trip de "temperatura
alta do mancal radial LNA". `ablacao_sp.py` e `piso_escala.py` estudaram o colapso do MAD do `sp`, mas não a
previsão por regressão (a de t e p, `residuo_carga.py`, usa a carga e foi refutada).

O MÉTODO (walk-forward mensal, retreino no dia 1, os 20.000 pontos estáveis anteriores, como o detector):
  M0  o atual: TI_0305 - mediana(TI_0301, TI_0303, TI_0307);
  M1  TI_0305 previsto por TI_0301, TI_0303, TI_0307 (mínimos quadrados, com intercepto);
  M2  M1 mais TI_0325 (temperatura do óleo do tanque), a variável que acopla os mancais ao óleo.
O score é o do detector: |resíduo - mediana| / MAD da referência, EWMA de meia-vida 30 min, só nos instantes
vigiados. Cada modelo é normalizado pelo PRÓPRIO MAD, então a comparação é justa em unidades de z.

O QUE SE MEDE.
  · R² e MAD (°C) na referência: quanto do TI_0305 os irmãos explicam e quanto ruído sobra;
  · na operação normal (fora de [T-7 d, T+2 d] de qualquer trip): a fração do tempo vigiado com o score acima
    do limiar A (3,0 x 0,9 = 2,7) e do B (3,0 x 1,7 = 5,1): o "falso alarme" do canal sozinho;
  · nos 8 trips: em que PERCENTIL dos máximos de 1.000 janelas normais de 48 h cai o máximo do score nas 48 h
    antes do trip (quanto mais perto de 100%, mais o evento se destaca). Controle negativo: as janelas normais.

RESULTADO (03/10/2026) -- NÃO MELHORA; NÃO SE PRÉ-REGISTRA.
    modelo                       MAD ref. (°C)   > limiar B   p95    nos 8 trips: percentil mediano / >= p95
    M0 atual (diferença)            1,11           18,3%     10,1        73 / 1
    M1 regressão nos irmãos         0,53 (R² 0,85) 20,7%     17,4        81 / 0
    M2 + temperatura do óleo        0,50 (R² 0,88) 23,5%     18,7        81 / 0
  · A regressão corta o MAD pela metade na referência, mas FORA DA AMOSTRA o score normalizado fica pior: mais
    tempo acima do limiar B e caudas mais pesadas. Dividir pelo MAD menor amplifica a deriva mensal da relação
    entre os mancais; o mínimo mensal do MAD cai de 0,31 para 0,21 °C (o denominador que colapsa, de
    `ablacao_sp.py`, agravado).
  · Nos 8 trips é misto: melhora 02/27 (24 -> 46), 04/07 (61 -> 84), 11/04 (0 -> 50) e 12/09 (46 -> 78); piora
    03/17 (98 -> 93), 04/29 e 02/26 (85 -> 78). A mediana sobe 73 -> 81, mas com 8 eventos isso está no ruído.
  · O 11/04/2025 fica no percentil 0 com o sp atual: o sp não o enxerga.

Uso:  PYTHONPATH=. python sp_por_regressao.py
"""
from __future__ import annotations
import io, contextlib
import numpy as np, pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    import regua_fp as R
    import aprovacao_operador as AO

C, DF, IX, STABLE, FIT, DB = AO.C, AO.DF, AO.IX, AO.STABLE, AO.FIT, R.DB
ALVO = "954005_624_TI_0305"
IRM = ["954005_624_TI_0301", "954005_624_TI_0303", "954005_624_TI_0307"]
MODELOS = {"M0 atual (diferença)": None, "M1 regressão nos irmãos": IRM, "M2 regressão nos irmãos + óleo": IRM + ["954005_624_TI_0325"]}
THR_A, THR_B = DB.BASE["sp"] * DB.K_LO["sp"], DB.BASE["sp"] * DB.KH["sp"]
HL = pd.Timedelta(DB.HL["sp"])
RNG = np.random.default_rng(0)


def scores() -> dict[str, pd.Series]:
    n = len(IX)
    brutos = {k: np.full(n, np.nan) for k in MODELOS}; info = {k: [] for k in MODELOS}
    cortes = [c for c in AO.cortes(1) if AO.bundle(c) is not None]
    for k, c0 in enumerate(cortes):
        c1 = cortes[k + 1] if k + 1 < len(cortes) else IX[-1] + pd.Timedelta("2min")
        F = DF.loc[STABLE & (IX < c0), C.SENSOR_TAGS].dropna().tail(FIT)
        s = (IX >= c0) & (IX < c1)
        W = DF.loc[s]
        for nome, cols in MODELOS.items():
            if cols is None:
                ref = F[ALVO] - F[IRM].median(axis=1); prev = W[ALVO] - W[IRM].median(axis=1)
                res_ref = ref
            else:
                A = np.column_stack([np.ones(len(F)), F[cols].to_numpy()])
                beta = np.linalg.lstsq(A, F[ALVO].to_numpy(), rcond=None)[0]
                res_ref = F[ALVO].to_numpy() - A @ beta
                prev = pd.Series(W[ALVO].to_numpy() - np.column_stack([np.ones(len(W)), W[cols].to_numpy()]) @ beta, index=W.index)
                res_ref = pd.Series(res_ref)
            med = float(np.median(res_ref)); mad = float(np.median(np.abs(res_ref - med)) * 1.4826)
            brutos[nome][s] = np.abs((prev.to_numpy() - med) / mad)
            r2 = 1 - float(np.var(res_ref)) / float(np.var(F[ALVO])) if cols is not None else np.nan
            info[nome].append((mad, r2))
    out = {k: pd.Series(v, index=IX).ewm(halflife=HL, times=IX).mean() for k, v in brutos.items()}
    for k, v in info.items():
        a = np.array(v)
        print(f"  {k:34s} MAD da referência (°C): mediana {np.median(a[:, 0]):.2f} (min {a[:, 0].min():.2f}, máx {a[:, 0].max():.2f})"
              + (f" | R² mediano {np.nanmedian(a[:, 1]):.2f}" if k != "M0 atual (diferença)" else ""))
    return out


def main():
    pd.set_option("display.width", 200)
    print("REFERÊNCIA (por mês, 20.000 pontos):")
    S = scores()
    vig = R.mask.to_numpy(); sel = np.asarray(R.sel)
    exclui = np.zeros(len(IX), bool)
    for t in R.alvo:
        exclui |= (IX >= t - pd.Timedelta(days=7)) & (IX <= t + pd.Timedelta(days=2))
    normal = vig & sel & ~exclui
    idx_norm = np.flatnonzero(normal)
    print(f"\nOPERAÇÃO NORMAL ({normal.sum() * 2 / 60:.0f} h vigiadas): fração do tempo acima do limiar, só o canal sp")
    print(f"  {'modelo':34s} {'> limiar A (2,7)':>16s} {'> limiar B (5,1)':>16s} {'p50':>6s} {'p95':>6s} {'p99':>6s}")
    for nome, z in S.items():
        v = z.to_numpy()[normal]
        print(f"  {nome:34s} {100 * np.mean(v > THR_A):15.1f}% {100 * np.mean(v > THR_B):15.1f}% {np.median(v):6.2f} {np.quantile(v, .95):6.2f} {np.quantile(v, .99):6.2f}")
    # janelas normais de 48 h
    W = int(48 * 30)
    print("\nOS 8 TRIPS: percentil do máximo do score nas 48 h antes do trip, entre os máximos de 1.000 janelas normais de 48 h")
    linhas = {n: [] for n in S}
    for nome, z in S.items():
        zz = z.to_numpy(); mx = []
        while len(mx) < 1000:
            i = int(RNG.choice(idx_norm)); j = zz[i:i + W][normal[i:i + W]]
            if len(j) >= 200: mx.append(np.nanmax(j))
        mx = np.array(mx)
        for T in R.alvo:
            m = (IX >= T - pd.Timedelta(hours=48)) & (IX < T) & vig
            linhas[nome].append(100 * float(np.mean(mx < np.nanmax(zz[m]))) if m.any() else np.nan)
    P = pd.DataFrame(linhas, index=[T.strftime("%Y-%m-%d") for T in R.alvo]).round(0)
    print(P.to_string()); print("\n  mediana dos 8:", P.median().round(0).to_dict())
    print("  eventos acima do percentil 90:", {k: int((P[k] >= 90).sum()) for k in P}, "| acima do 95:", {k: int((P[k] >= 95).sum()) for k in P})
    pd.DataFrame(linhas).to_csv(R.CACHE / "sp_por_regressao.csv")


if __name__ == "__main__":
    main()
