#!/usr/bin/env python3
"""Adaptação contínua: retreinar toda semana (janela deslizante) em vez de todo mês.

POR QUÊ (`drift_tecnicas.py`). Contra o baseline do bundle, TODA semana já está em
drift: um classificador distingue a semana corrente das ~5 semanas de referência com
AUC mediana de 0,95 mesmo em semana comum; o PSI de uma semana normal é 7,8 (a regra
de bolso chama > 0,25 de drift grande). O drift não é só de evento -- é contínuo e
onipresente. Para drift assim, a família de técnica indicada é a adaptação contínua:
janela deslizante (moving-window PCA, a prática em controle estatístico multivariado
de processos), aqui implementada como retreino a cada 7 ou 14 dias, com o mesmo
baseline de 20.000 pontos estáveis anteriores ao corte.

O RISCO CLÁSSICO. Modelo que se adapta rápido aprende degradação lenta como normal.
O detector depende de canais que ficam acesos por dias (a memória do CUSUM é a cola do
voto): se o baseline andar junto com a degradação, os canais apagam antes do trip.

JÁ SE SABE DO OUTRO LADO (`cadencia_retreino.py`): mensal é melhor que 2, 3 e 6 meses
e que congelado, pela carga (+36% a +44% nos mais espaçados) e pelo nascimento.
Semanal e quinzenal nunca foram testados.

CENÁRIOS. Semanal: 7 composições, deslocando o primeiro corte de 0 a 6 dias.
Quinzenal: 7, deslocando de 0 a 12 dias de 2 em 2. Referência: o mensal nas 8 da régua.

CRITÉRIO, ESCRITO ANTES DE RODAR. Os cenários semanais não são pareados com os do
mensal (não há "o mesmo dia" nos dois), então a comparação é entre DISTRIBUIÇÕES, como
em `cadencia_retreino.py`: aceito se, na mediana dos cenários, detecção, início e banda
não ficam abaixo do mensal, e FP/mês e carga ficam abaixo. Com 8 eventos, meia detecção
na mediana é ruído.

RESULTADO (30/09/2026).

JANELA DESLIZANTE SEM GUARDA -- REPROVADA, e com força:
    cadência      det   início   banda   FP/mês   carga
    mensal        6,5     6       4,5    0,861    130,5
    quinzenal     5,0     4       3,0    1,120    127,7
    semanal       4,0     2       1,0    0,947    113,2
O risco clássico, medido: o modelo aprende a degradação pré-trip como normal. Com o
teste de cadência (2-6 meses piora a carga), a cadência tem um fundo de vale no mensal.

JANELA COM INTERVALO DE GUARDA (`main_guarda`, decidido DEPOIS do resultado acima):
    desenho                idade média   det       início    banda     FP/mês          carga
    mensal                 15 d          6,5[5-8]  6[3-7]    4,5[2-5]  0,861[0,34-1,03] 130,5[49-185]
    semanal, guarda 11 d   14,5 d        6,0[6-7]  5[5-6]    4,0[3-4]  0,689[0,60-0,86]  62,9[50-87]
    semanal, guarda 21 d   24,5 d        7,0[7-8]  7[6-7]    5,0[5-5]  0,689[0,52-0,86] 157,2[75-189]
Pelo critério escrito antes, AS DUAS REPROVAM: ele exigia melhorar tudo junto. A de 11
dias corta o FP em 20% e a carga pela metade, perdendo meia detecção; a de 21 melhora a
detecção nas três réguas (banda 5/8 em TODOS os cenários) e corta o FP em 20%, com carga
+20%. As duas são muito mais estáveis entre cenários (amplitude do FP 0,26 e 0,34 contra
0,69), e sem reajustar nenhum limiar -- os limiares foram escolhidos para o mensal.

RESSALVAS. A guarda foi decidida depois de ver o semanal, e dois valores foram
olhados: escolher G agora seria ajustar sobre os mesmos 8 eventos. A estabilidade entre
cenários vem em parte de o semanal ter menos composições possíveis (7 contra ~30) --
o que é, em si, a vantagem: há menos jeitos de dar azar. A comparação justa é mediana
contra mediana; contra o dia 1 publicado (0,344 FP/mês), o semanal com guarda perde no
FP. É o candidato mais forte da frente de drift, e precisa de validação em dado novo.

Uso:  PYTHONPATH=. python retreino_semanal.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import regua_fp as R
import drift_nos_dados as DN

DC, C, DF, IX, STABLE, FIT = DN.DC, DN.C, DN.DF, DN.IX, DN.STABLE, DN.FIT
INICIO = pd.Timestamp("2024-02-05", tz="UTC")          # uma segunda-feira, depois do 1º mês com fit


def walkforward_passo(passo_dias: int, desloc: int):
    cortes = list(pd.date_range(INICIO + pd.Timedelta(days=desloc), IX[-1], freq=f"{passo_dias}D"))
    n = len(IX)
    t = np.full(n, np.nan); p = np.full(n, np.nan); ms = np.full(n, np.nan); ds = np.full(n, np.nan)
    for i, c0 in enumerate(cortes):
        c1 = cortes[i + 1] if i + 1 < len(cortes) else IX[-1] + pd.Timedelta("2min")
        fit = DF.loc[STABLE & (IX < c0), C.SENSOR_TAGS].dropna().tail(FIT)
        if len(fit) < FIT // 4:
            continue
        s = (IX >= c0) & (IX < c1)
        if not s.any():
            continue
        w = DF.loc[s]
        t[s] = DC.ScorerMax().fit(fit[C.TEMPERATURE_TAGS]).score(w[C.TEMPERATURE_TAGS])["pca_recon"].to_numpy()
        p[s] = DC.ScorerMax().fit(fit[C.PRESSURE_TAGS]).score(w[C.PRESSURE_TAGS])["pca_recon"].to_numpy()
        b = DC.DET._spread_mancal(fit)
        ms[s] = float(b.median()); ds[s] = float((b - b.median()).abs().median() * 1.4826)
    return t, p, ms, ds


def sinais(passo, desloc):
    f = R.CACHE / f"passo{passo}_d{desloc:02d}.npz"
    if f.exists():
        z = np.load(f); return z["t"], z["p"], z["ms"], z["ds"]
    t, p, ms, ds = walkforward_passo(passo, desloc)
    np.savez(f, t=t, p=p, ms=ms, ds=ds)
    return t, p, ms, ds


def main():
    ref = R.distribuicao()
    tabelas = {"mensal (régua)": ref}
    for passo, deslocs in ((7, range(0, 7)), (14, range(0, 14, 2))):
        L = []
        for d in deslocs:
            m = R.mede(R.detector(*sinais(passo, d))["fin"]); m.pop("cls")
            L.append(dict(desloc=d, **m))
        tabelas[f"a cada {passo} dias"] = pd.DataFrame(L).set_index("desloc")
    cols = ["det", "inicio", "banda", "fp_mes", "h_fp_mes", "h_neutro_mes", "carga_mes"]
    print(f"{'cadência':16s} {'cen.':>5s} " + " ".join(f"{c:>17s}" for c in cols))
    for nome, T in tabelas.items():
        cel = [(f"{T[c].median():5.1f} [{T[c].min():.0f}-{T[c].max():.0f}]" if c in ("det", "inicio", "banda")
                else f"{T[c].median():6.3f} [{T[c].min():.2f}-{T[c].max():.2f}]" if c == "fp_mes"
                else f"{T[c].median():6.1f} [{T[c].min():.0f}-{T[c].max():.0f}]") for c in cols]
        print(f"{nome:16s} {len(T):5d} " + " ".join(f"{x:>17s}" for x in cel))
    print("\ncontra o mensal (mediana):")
    for nome, T in tabelas.items():
        if nome.startswith("mensal"):
            continue
        ok = (T.det.median() >= ref.det.median() and T.inicio.median() >= ref.inicio.median()
              and T.banda.median() >= ref.banda.median()
              and T.fp_mes.median() < ref.fp_mes.median() and T.carga_mes.median() < ref.carga_mes.median())
        print(f"  {nome:16s} det {T.det.median() - ref.det.median():+.1f}  início {T.inicio.median() - ref.inicio.median():+.1f}"
              f"  banda {T.banda.median() - ref.banda.median():+.1f}  FP/mês {T.fp_mes.median() - ref.fp_mes.median():+.3f}"
              f"  carga {T.carga_mes.median() - ref.carga_mes.median():+.1f}   -> {'ACEITO' if ok else 'reprovado'}")
        T.to_csv(R.CACHE / f"retreino_{nome.split()[-2]}d.csv")


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════ janela com intervalo de guarda
def walkforward_guarda(passo_dias: int, desloc: int, guarda_dias: int):
    """Retreino a cada `passo_dias`, com a referência terminando `guarda_dias` antes do corte.

    POR QUÊ (resultado acima): semanal e quinzenal derrubam a banda de 4,5 para 1 e 3 --
    o modelo aprende a degradação pré-trip como normal. A variável que parece importar
    é a IDADE da referência, não a frequência: no mensal ela tem ~15 dias em média (0
    no dia 1, ~30 no fim); no semanal, ~3,5. A janela com intervalo de guarda separa as
    duas coisas: adapta toda semana, com a referência sempre mais velha que a
    degradação recente. O vb já faz isso (base termina 24 h antes do ponto).
    Decidido DEPOIS de ver o resultado do semanal; mesmo critério."""
    cortes = list(pd.date_range(INICIO + pd.Timedelta(days=desloc), IX[-1], freq=f"{passo_dias}D"))
    n = len(IX); G = pd.Timedelta(days=guarda_dias)
    t = np.full(n, np.nan); p = np.full(n, np.nan); ms = np.full(n, np.nan); ds = np.full(n, np.nan)
    for i, c0 in enumerate(cortes):
        c1 = cortes[i + 1] if i + 1 < len(cortes) else IX[-1] + pd.Timedelta("2min")
        fit = DF.loc[STABLE & (IX < c0 - G), C.SENSOR_TAGS].dropna().tail(FIT)
        if len(fit) < FIT // 4:
            continue
        s = (IX >= c0) & (IX < c1)
        if not s.any():
            continue
        w = DF.loc[s]
        t[s] = DC.ScorerMax().fit(fit[C.TEMPERATURE_TAGS]).score(w[C.TEMPERATURE_TAGS])["pca_recon"].to_numpy()
        p[s] = DC.ScorerMax().fit(fit[C.PRESSURE_TAGS]).score(w[C.PRESSURE_TAGS])["pca_recon"].to_numpy()
        b = DC.DET._spread_mancal(fit)
        ms[s] = float(b.median()); ds[s] = float((b - b.median()).abs().median() * 1.4826)
    return t, p, ms, ds


def main_guarda(guardas=(11, 21)):
    ref = R.distribuicao()
    print(f"{'desenho':28s} {'idade média':>11s} {'det':>9s} {'início':>9s} {'banda':>9s} {'FP/mês':>14s} {'carga':>14s}")
    def linha(nome, idade, T):
        print(f"{nome:28s} {idade:9.1f} d {T.det.median():5.1f} [{T.det.min():.0f}-{T.det.max():.0f}]"
              f" {T.inicio.median():5.1f} [{T.inicio.min():.0f}-{T.inicio.max():.0f}]"
              f" {T.banda.median():5.1f} [{T.banda.min():.0f}-{T.banda.max():.0f}]"
              f" {T.fp_mes.median():6.3f} [{T.fp_mes.min():.2f}-{T.fp_mes.max():.2f}]"
              f" {T.carga_mes.median():6.1f} [{T.carga_mes.min():.0f}-{T.carga_mes.max():.0f}]")
    linha("mensal (régua)", 15.0, ref)
    for g in guardas:
        L = []
        for d in range(0, 7):
            f = R.CACHE / f"guarda{g}_d{d:02d}.npz"
            if f.exists():
                z = np.load(f); sig = (z["t"], z["p"], z["ms"], z["ds"])
            else:
                sig = walkforward_guarda(7, d, g); np.savez(f, t=sig[0], p=sig[1], ms=sig[2], ds=sig[3])
            m = R.mede(R.detector(*sig)["fin"]); m.pop("cls"); L.append(dict(desloc=d, **m))
        T = pd.DataFrame(L).set_index("desloc")
        linha(f"semanal, guarda {g} d", g + 3.5, T)
        ok = (T.det.median() >= ref.det.median() and T.inicio.median() >= ref.inicio.median()
              and T.banda.median() >= ref.banda.median()
              and T.fp_mes.median() < ref.fp_mes.median() and T.carga_mes.median() < ref.carga_mes.median())
        print(f"{'':28s} -> {'ACEITO' if ok else 'reprovado'}  | amplitude FP/mês entre cenários "
              f"{T.fp_mes.max() - T.fp_mes.min():.3f} (mensal {ref.fp_mes.max() - ref.fp_mes.min():.3f})")
        T.to_csv(R.CACHE / f"retreino_semanal_guarda{g}.csv")
