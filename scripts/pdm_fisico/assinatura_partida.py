#!/usr/bin/env python3
"""A arrancada que antecede um trip é diferente de uma arrancada normal?

POR QUÊ. 5 dos 8 trips vêm até 31 h após uma partida (risco ~3,7x,
`relogio_partida.py`), e o detector descarta as 6 primeiras horas de cada
partida. É o único subproblema com dado abundante: ~100 partidas no período
avaliado, contra 8 trips.

EXPLORATÓRIO, E POR QUE ISSO IMPORTA. São 5 positivos. Com tão poucos, a chance
de achar um separador que é coincidência é maior que a de achar um real. Por isso
as hipóteses vêm declaradas AQUI, antes de olhar os positivos, com direção fixa,
e a correção de Bonferroni é pelas quatro. Nenhuma operação ou manutenção foi
consultada para escolhê-las -- foram escolhidas pela física e pelo catálogo.

AS QUATRO HIPÓTESES (janela da arrancada: primeiras 6 h com a máquina de pé):
  F1  vibração: maior p95 de amplitude entre as 10 sondas TV_*        -> MAIOR é pior
      (desbalanceamento e desalinhamento aparecem ao passar pelas críticas)
  F2  mancal: maior temperatura entre os 4 mancais do compressor em 5,5-6 h
      (TI_0301/0303/0305/0307: escora ativo, escora inativo, radiais)  -> MAIOR é pior
  F3  spread do mancal radial LNA em 5,5-6 h: TI_0305 menos a média dos
      outros três                                                     -> MAIOR é pior
  F4  pressão mínima do óleo header de proteção (PI_0340) em 1-6 h   -> MENOR é pior

RESSALVA. F3 e F4 não são cegas: sei que o 09/12 foi por temperatura alta do
TI_0305 (TAHH_6240305) e que vários trips foram por pressão baixa de óleo
(PALL_6240340, que pelo número é o alarme do PI_0340). F1 e F2 são as limpas.

POSITIVOS: as 5 partidas seguidas de trip-alvo em até 32 h. NEGATIVOS: as demais
partidas do período avaliado com pelo menos 6 h de operação depois.

RESULTADO (29/09/2026) -- exploratório; 52 partidas, 5 positivas.

    hipótese               mediana normal   mediana pré-trip   p Bonferroni x4
    F2 mancal máx (limpa)       69,6 °C          79,6 °C            0,005
    F3 spread LNA (informada)   11,0             22,8               0,022
    F1 vibração p95 (limpa)     25,0             28,8               0,048
    F4 óleo mín (informada)      3,39             3,28              0,34

Os controles:
  · NÃO é só estação: dentro do mesmo mês, as positivas tendem a ser as partidas
    mais quentes (07/04 a mais quente de abril; 03/11 e 08/12 lideram os seus).
  · MAS é o período, não a partida: contra a própria máquina nos 7 dias estáveis
    anteriores, a diferença some (+4,4 °C contra +0,8 °C, p = 0,16). Os mancais já
    estavam quentes ANTES: linha de base 76,7 / 80,5 / 68,2 / 114,1 / 73,8 °C,
    contra 69,6 °C de mediana nas negativas.
  · partida quente após parada curta não explica: Spearman parada x F2 = +0,21
    (p = 0,13), e a parada mediana das negativas também é de ~1 h.

A F2 não é assinatura da arrancada: é "a máquina vinha rodando quente na semana
anterior". É sinal de NÍVEL absoluto de temperatura, ao qual os canais de PCA são
cegos por construção (o retreino mensal absorve o nível no baseline). Parente da
margem ao setpoint, que já foi absorvida pelo v2 ([[margem-subsumida-pelo-v2]]).
Hipótese para um teste futuro pré-registrado, não resultado.

Uso:  PYTHONPATH=. python assinatura_partida.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
import regua_fp as R
import pericia_fp_atual as PF
from publica_clearml import VIBRATION_TAGS

PP = PF.PP
MANCAIS = ["954005_624_TI_0301", "954005_624_TI_0303", "954005_624_TI_0305", "954005_624_TI_0307"]
OLEO = "954005_624_PI_0340"
JANELA = int(6 * 30)
POS_H = 32.0
HIP = {"F1 vibração p95": "maior", "F2 mancal máx": "maior",
       "F3 spread LNA": "maior", "F4 óleo mín": "menor"}


def partidas():
    op = PP.op.to_numpy(); part = PP.part.to_numpy() & np.asarray(R.sel)
    ini = [i for i in np.flatnonzero(part) if i + JANELA < len(op) and op[i:i + JANELA].all()]
    tr = [R.idx.searchsorted(t) for t in R.alvo]
    pos = {i for i in ini if any(0 < (j - i) * 2 / 60 <= POS_H for j in tr)}
    return ini, pos


def features(i: int) -> dict:
    g = PP.g.iloc[i:i + JANELA]
    fim = PP.g.iloc[i + JANELA - 15:i + JANELA]            # 5,5-6 h
    vib = g[VIBRATION_TAGS].quantile(0.95).max()
    man = fim[MANCAIS].mean()
    return {"F1 vibração p95": float(vib),
            "F2 mancal máx": float(man.max()),
            "F3 spread LNA": float(man[MANCAIS[2]] - man[[MANCAIS[0], MANCAIS[1], MANCAIS[3]]].mean()),
            "F4 óleo mín": float(PP.g[OLEO].iloc[i + 30:i + JANELA].min())}


def main():
    ini, pos = partidas()
    F = pd.DataFrame([dict(inicio=R.idx[i], positivo=i in pos, **features(i)) for i in ini])
    neg = F[~F.positivo]; P = F[F.positivo]
    print(f"partidas com >= 6 h de operação no período avaliado: {len(F)}  |  "
          f"positivas: {len(P)}  negativas: {len(neg)}\n")
    pd.set_option("display.width", 170)
    print("AS 5 POSITIVAS — valor e percentil entre as negativas (100 = a pior de todas)")
    for _, r in P.iterrows():
        cel = []
        for f, d in HIP.items():
            v = r[f]; x = neg[f].dropna()
            pc = 100 * ((x < v).mean() if d == "maior" else (x > v).mean())
            cel.append(f"{f.split()[0]} {v:8.2f} (p{pc:3.0f})")
        print(f"  {r.inicio:%Y-%m-%d %H:%M}   " + "   ".join(cel))
    print(f"\n{'hipótese':18s} {'mediana neg':>12s} {'mediana pos':>12s} {'p (U, 1 lado)':>14s} "
          f"{'p Bonferroni x4':>16s}")
    for f, d in HIP.items():
        x, y = neg[f].dropna(), P[f].dropna()
        alt = "greater" if d == "maior" else "less"
        p = mannwhitneyu(y, x, alternative=alt).pvalue
        print(f"{f:18s} {x.median():12.2f} {y.median():12.2f} {p:14.4f} {min(1.0, 4 * p):16.4f}")
    F.to_csv(R.CACHE / "assinatura_partida.csv", index=False)


if __name__ == "__main__":
    main()
