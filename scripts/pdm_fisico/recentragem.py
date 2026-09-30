#!/usr/bin/env python3
"""Recentrar depois da manutenção: o retreino disparado por evento.

O QUE MOTIVA (`drift_eventos.py`). A máquina volta de cada manutenção com um
"normal" novo em muitos sensores de uma vez: nas semanas com manutenção, 14 tags
saltam > 3 sigma juntas (mediana), contra 4 nas demais (p 0,012). O retreino é só
no dia 1, e leva até um mês para absorver. Em nov/2025 o diferencial do filtro de
gás de selagem foi de +0,245 para -0,400 kgf/cm² e o canal p ficou aceso 96% do mês.

A CORREÇÃO. Um salto de nível é exatamente o que recentrar corrige, e recentrar
precisa de dias, não das ~5 semanas de um baseline inteiro. Depois de cada
manutenção, quando a máquina junta N dias de operação estável:
  · t e p: o centro do RobustScaler (a mediana de cada sensor) passa a ser a
    mediana desses N dias. PCA, escala (IQR), sens_p99 e recon_p99 ficam os do
    bundle do mês;
  · sp: a mediana do spread passa a ser a desses N dias; o MAD fica;
  · vb: não muda -- já tem referência móvel de 400 h.
Vale até o próximo corte mensal.

REGRAS, ESCRITAS ANTES DE RODAR:
  · manutenção = HSX_6240001A ligado >= 24 h (a mesma definição do diagnóstico);
  · N = 3 dias de operação estável (primário); N = 7 (sensibilidade);
  · só se aplica quando o baseline do bundle em vigor é MAJORITARIAMENTE de antes
    da manutenção (o fim dela é posterior ao ponto médio do baseline). Se a
    manutenção acabou cedo no baseline, o próprio retreino já absorveu; recentrar
    com 3 dias só trocaria uma mediana boa por uma ruidosa;
  · a janela de N dias é sempre posterior à manutenção e anterior ao uso: causal.

CRITÉRIO: a regra da régua (`regua_fp.compara`) -- carga cai em >= 7 de 8, nenhuma
composição perde detecção, início e banda não caem na mediana, FP/mês não sobe.
Aceita só se N = 3 E N = 7 passarem: com duas variantes, um vencedor isolado não
basta.

O RISCO CONHECIDO. `o-normalizador-e-o-ponto-fragil`: o par (PCA, p99) é atômico, e
suavizar ou desacoplar a escala piorou. Recentrar mexe no centro, não na escala,
mas desloca os resíduos que o p99 calibrou. Pode piorar.

RESULTADO (30/09/2026) -- REPROVADO nas duas formas. A recentragem por evento fecha.

  1º teste (pré-registrado, N = 3 e 7): carga cai em só 2/8, FP sobe na mediana
     (0,861 -> 0,947), uma composição perde detecção. E o mecanismo NÃO funcionou:
     o p em nov/2025 seguiu ~97% aceso.
  Diagnóstico: a recentragem FUNCIONA no sinal -- em 19-30/11 o p cru acima do
     limiar vai de 100% para 8,3% (PDI_0301 de 22,8 para 0,18; e as pressões do
     header de óleo PI_0339/PI_0340 de ~10 para 0,27, que também mudaram de nível
     na manutenção logo após o trip de óleo de 04/11). O que mantinha o canal aceso
     era o CUSUM: nos N dias até a recentragem ele somou ~20 por amostra contra a
     referência velha e leva semanas para descer.
  2º teste (a posteriori, `com_reset`: a mesma recentragem descartando o acumulado,
     mesmo critério): o mecanismo passa a funcionar -- p em nov/2025 cai para ~48%,
     o máximo possível com 18 dias antes da recentragem -- e as horas de NEUTRO caem
     13 h/mês em todas as composições. Mas o FP SOBE nas 8 (+0,09 a +0,26/mês).
  Por quê: os FP novos nascem em dezembro/2025, logo depois de o bundle de dezembro
     ser recentrado com a mediana de 15-19/11. Esse bundle já funcionava -- treinado
     com dado de antes e depois da manutenção, o PCA aprendeu o salto como uma de
     suas direções (o p dele ficava aceso 0,7%). Recentrá-lo com 3 dias logo após a
     partida o estragou.

A versão que sobraria -- recentrar só dentro do mês da manutenção -- seria desenhada
exatamente sobre o que se acabou de ver, para ganhar poucos episódios. Não foi
rodada. O achado útil vai para produção como MONITOR (`cabiunas_inference.monitor_drift`),
que avisa sem mexer no alarme.

Uso:  PYTHONPATH=. python recentragem.py
"""
from __future__ import annotations
import copy
import numpy as np, pandas as pd
import regua_fp as R
import drift_eventos as DE
import drift_nos_dados as DN
import coativacao as CO

DC, C, DF, IX, STABLE, FIT = DN.DC, DN.C, DN.DF, DN.IX, DN.STABLE, DN.FIT
VIG = DN.VIG
B_ALL = np.load("piso_fisico_cache.npz")["b_all"]
NS = (3, 7)


def ativacoes(N: int) -> list[dict]:
    """Para cada manutenção: quando os N dias estáveis se completam, e a janela."""
    out = []
    for a, b in DE.manutencoes():
        i0 = IX.searchsorted(b)
        cs = np.cumsum(VIG[i0:])
        k = int(np.searchsorted(cs, N * 720))
        if k >= len(cs):
            continue
        jan = np.zeros(len(IX), dtype=bool); jan[i0:i0 + k + 1] = VIG[i0:i0 + k + 1]
        out.append(dict(fim=b, ativa=IX[i0 + k], janela=jan))
    return out


def walkforward_recentrado(dia: int, N: int | None):
    """`drift_composicao.walkforward_dia` + recentragem pós-manutenção.
    N = None desliga a recentragem (deve reproduzir a régua)."""
    base = pd.date_range(IX[0].normalize().replace(day=1), IX[-1], freq="MS", tz="UTC")
    cortes = [c for c in (m + pd.Timedelta(days=dia - 1) for m in base) if IX[0] < c < IX[-1]]
    acts = ativacoes(N) if N else []
    n = len(IX)
    t = np.full(n, np.nan); p = np.full(n, np.nan); ms = np.full(n, np.nan); ds = np.full(n, np.nan)
    usadas = []
    for i, c0 in enumerate(cortes):
        c1 = cortes[i + 1] if i + 1 < len(cortes) else IX[-1] + pd.Timedelta("2min")
        fit = DF.loc[STABLE & (IX < c0), C.SENSOR_TAGS].dropna().tail(FIT)
        if len(fit) < FIT // 4:
            continue
        meio = fit.index[len(fit) // 2]
        scT = DC.ScorerMax().fit(fit[C.TEMPERATURE_TAGS])
        scP = DC.ScorerMax().fit(fit[C.PRESSURE_TAGS])
        b = DC.DET._spread_mancal(fit)
        med0 = float(b.median()); mad0 = float((b - b.median()).abs().median() * 1.4826)
        rel = [a for a in acts if a["fim"] > meio and a["ativa"] < c1]
        pontos = sorted({c0, *[max(a["ativa"], c0) for a in rel]})
        for j, s0 in enumerate(pontos):
            s1 = pontos[j + 1] if j + 1 < len(pontos) else c1
            seg = (IX >= s0) & (IX < s1)
            if not seg.any():
                continue
            ativos = [a for a in rel if a["ativa"] <= s0]
            sT, sP, med = scT, scP, med0
            if ativos:
                a = ativos[-1]
                sT = copy.deepcopy(scT); sP = copy.deepcopy(scP)
                sT.scaler.center_ = DF.loc[a["janela"], sT.cols].median().to_numpy()
                sP.scaler.center_ = DF.loc[a["janela"], sP.cols].median().to_numpy()
                med = float(np.nanmedian(B_ALL[a["janela"]]))
                usadas.append((c0, a["fim"], s0))
            w = DF.loc[seg]
            t[seg] = sT.score(w[C.TEMPERATURE_TAGS])["pca_recon"].to_numpy()
            p[seg] = sP.score(w[C.PRESSURE_TAGS])["pca_recon"].to_numpy()
            ms[seg] = med; ds[seg] = mad0
    return (t, p, ms, ds), usadas


def sinais(dia, N):
    f = R.CACHE / f"recentra_N{N}_d{dia:02d}.npz"
    if f.exists():
        z = np.load(f); return (z["t"], z["p"], z["ms"], z["ds"]), int(z["n_usadas"])
    (t, p, ms, ds), usadas = walkforward_recentrado(dia, N)
    np.savez(f, t=t, p=p, ms=ms, ds=ds, n_usadas=len(usadas))
    return (t, p, ms, ds), len(usadas)


def main():
    ref = R.distribuicao()
    print("REFERÊNCIA\n" + R.resumo(ref))
    normal, pre = CO.janelas()
    nov = ((R.idx >= pd.Timestamp("2025-11-01", tz="UTC")) & (R.idx < pd.Timestamp("2025-12-01", tz="UTC"))
           & R.mask.to_numpy())
    res = {}
    for N in NS:
        L = []; ciclo_nov = []
        for dia in R.DIAS:
            sig, nu = sinais(dia, N)
            d = R.detector(*sig)
            m = R.mede(d["fin"]); m.pop("cls")
            L.append(dict(dia=dia, recentragens=nu, **m))
            ciclo_nov.append(100 * d["A"]["p"].to_numpy()[nov].mean())
        var = pd.DataFrame(L).set_index("dia")
        dif = var[ref.columns] - ref
        melhora = int((dif.carga_mes < -1e-9).sum()); perde = int((dif.det < 0).sum())
        ok = (melhora >= 7 and perde == 0 and var.inicio.median() >= ref.inicio.median()
              and var.banda.median() >= ref.banda.median()
              and var.fp_mes.median() <= ref.fp_mes.median() + 1e-9)
        res[N] = ok
        print(f"\n{'=' * 90}\nRECENTRAGEM COM N = {N} DIAS ESTÁVEIS\n{'=' * 90}")
        print(R.resumo(var))
        print(f"  recentragens aplicadas por composição: {var.recentragens.tolist()}")
        print(f"  carga cai em {melhora}/8 | perdem detecção: {perde} | FP cai em "
              f"{int((dif.fp_mes < -1e-9).sum())}/8 | passa: {ok}")
        print(f"  ciclo do p (nível A) em nov/2025: {np.round(ciclo_nov, 1).tolist()}")
        print("  diferença pareada:"); print(dif[["det", "inicio", "banda", "fp_mes", "h_fp_mes",
                                                  "h_neutro_mes", "carga_mes"]].round(2).to_string())
        var.to_csv(R.CACHE / f"recentragem_N{N}.csv")
    ciclo_ref = [100 * R.detector(*R.sinais(d))["A"]["p"].to_numpy()[nov].mean() for d in R.DIAS]
    print(f"\n  referência — ciclo do p em nov/2025: {np.round(ciclo_ref, 1).tolist()}")
    print(f"\nVEREDITO: {'ACEITO' if all(res.values()) else 'REPROVADO'}  ({res})")


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════ segundo teste, decidido depois
N_RESET = 30          # 30 instantes de reinício: 0,25^30 ~ 1e-18, o acumulador vai a zero


def com_reset():
    """A MESMA recentragem, descartando a evidência acumulada contra a referência velha.

    POR QUE EXISTE -- e por que é posterior. O primeiro teste reprovou, e o
    diagnóstico mostrou que a recentragem FUNCIONA no sinal: em 19-30/11/2025 o p
    cru acima do limiar vai de 100% para 8,3% (PDI_0301 de 22,8 para 0,18; as
    pressões do header de óleo PI_0339/PI_0340 de ~10 para 0,27). O que mantém o
    canal aceso é o CUSUM: nos N dias entre a volta da máquina e a recentragem ele
    somou ~20 por amostra contra a referência velha, e leva semanas para descer.
    Quando a referência muda, a evidência acumulada contra a antiga perde o
    sentido -- deveria ter sido descartada junto. Mesmo critério, rodado uma vez."""
    ref = R.distribuicao()
    nov = ((R.idx >= pd.Timestamp("2025-11-01", tz="UTC")) & (R.idx < pd.Timestamp("2025-12-01", tz="UTC"))
           & R.mask.to_numpy())
    res = {}
    for N in NS:
        L, ciclo = [], []
        for dia in R.DIAS:
            sig, usadas = walkforward_recentrado(dia, N)
            rx = np.zeros(len(R.idx), dtype=bool)
            for _, _, s0 in usadas:
                i = R.idx.searchsorted(s0); rx[i:i + N_RESET] = True
            d = R.detector(*sig, reset_extra=rx)
            m = R.mede(d["fin"]); m.pop("cls")
            L.append(dict(dia=dia, **m)); ciclo.append(100 * d["A"]["p"].to_numpy()[nov].mean())
        var = pd.DataFrame(L).set_index("dia")
        dif = var[ref.columns] - ref
        melhora = int((dif.carga_mes < -1e-9).sum()); perde = int((dif.det < 0).sum())
        ok = (melhora >= 7 and perde == 0 and var.inicio.median() >= ref.inicio.median()
              and var.banda.median() >= ref.banda.median()
              and var.fp_mes.median() <= ref.fp_mes.median() + 1e-9)
        res[N] = ok
        print(f"\n{'=' * 90}\nRECENTRAGEM N = {N} + DESCARTE DO ACUMULADO\n{'=' * 90}")
        print(R.resumo(var))
        print(f"  carga cai em {melhora}/8 | perdem detecção: {perde} | FP cai em "
              f"{int((dif.fp_mes < -1e-9).sum())}/8 | passa: {ok}")
        print(f"  ciclo do p em nov/2025: {np.round(ciclo, 1).tolist()}")
        print(dif[["det", "inicio", "banda", "fp_mes", "h_fp_mes", "h_neutro_mes", "carga_mes"]].round(2).to_string())
        var.to_csv(R.CACHE / f"recentragem_reset_N{N}.csv")
    print(f"\nVEREDITO (segundo teste): {'ACEITO' if all(res.values()) else 'REPROVADO'}  ({res})")
