#!/usr/bin/env python3
"""A RUNNING_A, a tag que abre e fecha a máscara do detector, é coerente com o que a máquina faz?

DIAGNÓSTICO (03/10/2026). Não muda o detector nem veredito.

O QUE O USUÁRIO INFORMOU. A RUNNING_A é um status de operação DERIVADO das velocidades (NGP_A, a turbina do
gerador de gás, e a turbina de potência; o catálogo tem `NGP_A`, `NPT_A` e `NCPSR_A`). Ela não está em nenhum
catálogo do supervisor, e a regra exata (limiares) não é conhecida aqui. Os exports do portal que o projeto
tem (`portalintegridade/*recorded.xlsx`) trazem NGP_A, NPT_A e NCPSR_A, mas com só 1.000 amostras por mês
(~8 h no começo de cada mês): NÃO dá para reconstruir a RUNNING_A pelas velocidades. O que dá é ver se ela
casa com a temperatura do escape (T5) e com o que o detector faz nas partidas.

TRÊS CHECAGENS.
  1. COERÊNCIA com o T5: T5 quando ligada e quando desligada; trechos de discordância.
  2. As "partidas" (a RUNNING_A sobe depois de ter caído): de VERDADE (o T5 caiu abaixo de 300 °C enquanto estava
     desligada) ou FALHAS CURTAS da tag (<= 30 min com o T5 sempre > 300)? Cada partida dispara um blackout de 6 h
     e zera a memória do CUSUM, então uma falha curta custaria 6 h de cegueira.
  3. Os alarmes que nascem na borda do blackout (6,47 h depois de uma partida): vêm de partidas de verdade?

RESULTADO (03/10/2026) -- A TAG É COERENTE; A MÁSCARA É, NA PRÁTICA, O T5 > 300; NADA A CORRIGIR.
  1. T5 com RUNNING_A ligada: mediana 651 °C (p5 578); desligada: mediana 31 °C (p95 59). Desligada com T5 > 600 °C:
     1 h no total (19 trechos de 3 a 6 min). T5 > 300 sozinho dá 66,7% do tempo; com a RUNNING_A, 66,6%: ela tira
     só 4 h de 13.600 h de instantes quentes. Ligada com a máquina fria: 99 h com T5 < 100 °C em 37 trechos (o maior
     com 22 h), compatível com rotação de partida ou lavagem sem combustão, que uma tag por velocidade vê.
  2. 111 partidas desde 2025-01-01: 105 de verdade e só 6 falhas curtas (mediana 3 min), ou 36 h de cegueira
     (0,4% do tempo vigiado). HIPÓTESE REFUTADA: eu li os 84 trechos curtos de "desligada com T5 > 300" como falhas
     da tag; eram as primeiras amostras DEPOIS de paradas de verdade, com o escape ainda quente.
  3. Dos 21 episódios da composição 1, 11 nascem na borda do blackout, todos depois de partidas DE VERDADE
     (2 FP, 6 NEUTRO, 3 TP): é o efeito já conhecido, `a-borda-anda-com-a-mascara`; não vem da tag.
  · Outra hipótese refutada: 11 "partidas com ignição tardia" (T5 > 300 só > 6 h depois) são 2 a 3 sequências de
    21 a 24/08/2025, com a RUNNING_A ligando e desligando a cada ~15 min; nenhum alarme nasce depois delas.
  · Em partidas típicas o T5 já passa de 300 °C quando a RUNNING_A sobe (a mediana do intervalo é 0): a tag liga
    quando a turbina atinge a velocidade, depois da ignição.

4. VOTAÇÃO DE SINAIS INDEPENDENTES (a pergunta do usuário: "certeza que é coerente?"). O T5 sozinho é um só
   sinal e não separa "ligada" de "girando fria". Com quatro sinais que separam ligada de desligada com AUC > 0,97
   (T5 0,997; ΔP do gás combustível PDI_0317 0,995; vibração máxima 0,995; pressão do gás de selagem PI_0307 0,980),
   a maioria concorda com a RUNNING_A em 99,48% de 20.072 h.
     · RUNNING_A ligada e a maioria diz parada: 103 h em 39 trechos, quase tudo em 19 a 24/08/2025 (22,4, 21,1, 15,6,
       8,8 h...): a tag marca "ligada" com a máquina sem combustão. Nesses instantes o T5 < 300 °C da máscara protege.
     · RUNNING_A desligada e a maioria diz em marcha: 3 h em 75 trechos de até 0,1 h (bordas de parada).
     · RUNNING_A ausente (NaN): 239 h, e em nenhuma a maioria diz em marcha: tratar NaN como parada, como a máscara
       faz, está certo no histórico.
   LIMITE: os quatro sinais dizem "a máquina está em marcha", não a regra de velocidade da tag; e podem errar juntos
   numa indisponibilidade comum. Os limiares sobre NGP_A/NPT_A seguem não verificados.

Uso:  PYTHONPATH=. python running_a_coerencia.py
"""
from __future__ import annotations
import io, contextlib
import numpy as np, pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    import regua_fp as R

M = 2 / 60


def corridas(m: pd.Series) -> np.ndarray:
    a = m.fillna(False).to_numpy().astype(int); d = np.diff(np.concatenate(([0], a, [0])))
    return (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)) * M


def votacao(g: pd.DataFrame) -> None:
    """4. Maioria de sinais independentes contra a RUNNING_A."""
    from sklearn.metrics import roc_auc_score
    ix, r = g.index, g["RUNNING_A"]; on = r > 0.5; known = r.notna()
    V = g[[c for c in g.columns if c.startswith("TV_")]].max(axis=1)
    S = {"T5": g["T5_AVG_A"], "PDI_0317": g["954005_624_PDI_0317"], "vibração máx.": V, "PI_0307": g["954005_624_PI_0307"]}
    ok = {}
    for n, s in S.items():
        m = known & s.notna(); a = roc_auc_score(on[m], s[m]); a = max(a, 1 - a)
        mo, mf = s[m & on].median(), s[m & ~on].median(); print(f"  AUC {n:14s} {a:.3f} (ligada {mo:.2f} | desligada {mf:.2f})")
        if a > 0.97: ok[n] = (s, (mo + mf) / 2, 1 if mo > mf else -1)
    votos = pd.DataFrame({n: ((s - lim) * sg > 0) for n, (s, lim, sg) in ok.items()}); nv = pd.DataFrame({n: s.notna() for n, (s, _, _) in ok.items()}).sum(axis=1)
    mai = votos.sum(axis=1) > nv / 2; m = known & (nv >= 2)
    d_on, d_off = m & on & ~mai, m & ~on & mai
    print(f"  votação em {m.sum() * M:.0f} h: concordam {100 * (1 - (d_on | d_off).sum() / m.sum()):.2f}% | ligada e maioria diz parada {d_on.sum() * M:.0f} h | "
          f"desligada e maioria diz em marcha {d_off.sum() * M:.0f} h | RUNNING_A ausente {(~known).sum() * M:.0f} h, dos quais em marcha {((~known) & mai & (nv >= 2)).sum() * M:.0f} h")


def main():
    pd.set_option("display.width", 200)
    g = pd.read_parquet("grade2min.parquet")
    ix, r, t5 = g.index, g["RUNNING_A"], g["T5_AVG_A"]
    on = (r > 0.5).fillna(False); off = (r <= 0.5)
    print(f"{len(g)} instantes ({len(g) * M / 24:.0f} dias) | RUNNING_A ausente {100 * r.isna().mean():.1f}%")
    print(f"T5 ligada: mediana {t5[on].median():.0f} (p5 {t5[on].quantile(.05):.0f}) | desligada: mediana {t5[off].median():.0f} (p95 {t5[off].quantile(.95):.0f})")
    for nome, m in (("ligada e T5 < 100 (fria)", on & (t5 < 100)), ("desligada e T5 > 300", off & (t5 > 300)), ("desligada e T5 > 600", off & (t5 > 600))):
        h = corridas(m); print(f"  {nome:30s} {m.sum() * M:6.0f} h em {len(h):3d} trechos (mediano {np.median(h) if len(h) else 0:.2f} h, máx {h.max() if len(h) else 0:.1f} h)")
    est_t5 = t5 > 300
    print(f"  T5 > 300: {100 * est_t5.mean():.1f}% | com RUNNING_A: {100 * (est_t5 & on).mean():.1f}% | a RUNNING_A tira {(est_t5 & ~on).sum() * M:.0f} h")
    part = on & ~on.shift(fill_value=False); fall = (~on) & on.shift(fill_value=False)
    pos, fpos = np.flatnonzero(part.to_numpy()), np.flatnonzero(fall.to_numpy())
    L = []
    for p in pos:
        if ix[p] < pd.Timestamp("2025-01-01", tz="UTC"): continue
        fs = fpos[fpos < p]
        if not len(fs): continue
        f = fs[-1]; tm = t5.iloc[f:p + 1]
        L.append(dict(t=ix[p], off_h=(p - f) * M, t5_min=float(tm.min()) if tm.notna().any() else np.nan))
    D = pd.DataFrame(L)
    D["tipo"] = np.where((D.t5_min > 300) & (D.off_h <= 0.5), "falha curta", np.where(D.t5_min > 300, "parada quente", "partida de verdade"))
    print("\n'PARTIDAS' (RUNNING_A sobe depois de cair), desde 2025-01-01:")
    print(D.groupby("tipo").agg(n=("t", "size"), off_mediana_min=("off_h", lambda s: 60 * s.median())).round(1).to_string())
    fin = R.detector(*R.sinais(1))["fin"]; cls = R.mede(fin)["cls"]
    abre = D.t + pd.Timedelta(minutes=(180 + 14) * 2)
    n_borda = sum(1 for a, *_ in cls if ((abre - pd.Timedelta(minutes=6) <= a) & (a <= abre + pd.Timedelta(minutes=6))).any())
    tipos = [D.tipo[((abre - pd.Timedelta(minutes=6) <= a) & (a <= abre + pd.Timedelta(minutes=6))).to_numpy()].iloc[0] for a, *_ in cls
             if ((abre - pd.Timedelta(minutes=6) <= a) & (a <= abre + pd.Timedelta(minutes=6))).any()]
    votacao(g)
    print(f"\ncomposição 1: {len(cls)} episódios, {n_borda} nascem na borda do blackout; tipos de partida: {pd.Series(tipos).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
