#!/usr/bin/env python3
"""A memória do CUSUM contamina as comparações da segunda rodada?

DIAGNÓSTICO A POSTERIORI (01/10/2026), depois de F1, R1, R3, F3 e DPCA reprovarem.
Não muda veredito nenhum; serve para saber se a regra de detecção estrita (A) estava
medindo informação ou sorte de memória.

A PERGUNTA. O CUSUM do detector não tem teto: só zera na máscara. Um excesso grande
(o degrau pós-manutenção de nov/2025 levou o p a 350x o limiar) mantém o canal aceso
por dias depois que o EWMA voltou ao normal. No F1, o 09/12/2025 da composição 4 era
detectado assim -- com o p a 0,14 do limiar nas 48 h anteriores. Se boa parte das
detecções da referência for desse tipo, qualquer mudança no score embaralha QUAIS
estados estão acesos no trip, e a regra A reprova por loteria, não por perda de sinal.

AS MEDIDAS, por braço e cenário:
  det        a régua de sempre (alarme em [T-48h, T)).
  acaso      `avalia.permuta`: quantos dos 8 um detector com a mesma cobertura acerta
             sorteando instantes de operação.
  fresca X   a detecção sobrevive se a evidência acumulada antes de T-X for
             descartada: o mesmo detector com o acumulador zerado em T-X (só para
             aquele evento), e o VOTO ligado >= 120 min dentro da janela. Mede-se no
             voto, não no alarme final, porque o corte em T-X abriria um refratário
             que bloquearia a janela por construção. X = 7 dias e 48 h.
  memória    detectado e não fresco (7 d): sustentado por evidência de mais de uma
             semana antes.
  carga sem evidência   fração das horas de FP + NEUTRO em que nenhum voto se forma
             com os canais cujo EWMA está ACIMA do próprio limiar agora -- o alarme
             está de pé só pela memória.

RESULTADO (01/10/2026) -- A MEMÓRIA NÃO EXPLICA AS REPROVAÇÕES, MAS EXPLICA A CARGA.

    braço (medianas)   det  acaso  fresca7d  fresca48h  carga  carga sem evidência
    referência         6,5   2,71    6,0       5,0      133,4        57%
    F1                 7,0   2,11    6,0       5,5      102,1        49%
    R1 mín K=2         5,0   1,84    5,0       4,0       87,9        64%
    F3 +-7 d           5,0   2,64    5,0       4,0      137,0        55%
    DPCA 0-10-30       5,0   2,46    5,0       4,0      136,0        63%
    R3 G=16            6,0   3,12    6,0       5,0      138,0        57%

  · Na referência, só 5 das 51 detecções (8 composições x 8 eventos) dependem de
    evidência de mais de 7 dias -- todas no 26/02/2026, composições 8 a 22, com o t
    aceso pela memória de 1-2 semanas. Com 48 h caem 12: o 27/02/2025 vai de 8 para 3
    composições. A memória que detecta é de DIAS, não de semanas.
  · Os braços perdem detecção FRESCA: R1 perde 12 pares (7 frescos na referência), F3
    perde 9 (7), DPCA 8 (8). Só nas frescas: 46 -> 39, 40, 38. Os vereditos ficam.
  · F1 empata nas frescas (46 -> 46): perde o 09/12 na composição 4 e ganha o mesmo
    09/12 na 18. O da composição 4 era vb (vivo) + p, e o p estava aceso só pelo
    acumulador: até 04/12 00:00 o bundle de novembro via o PDI_0301 (degrau
    pós-manutenção) a ~7x o limiar; o bundle de dezembro vê o p normal (EWMA acima do
    limiar 4% do tempo), e o CUSUM carregou o artefato do bundle velho por 4 dias
    para dentro do novo. O critério de 7 dias não pega porque o artefato durou até
    5 dias antes do trip. O veredito do F1 não muda (regra pré-registrada).
  · Onde a memória pesa é o custo: em 57% das horas de carga (21% no dia 1) nenhum
    voto se forma com os canais que estão acima do limiar naquele instante.
  · O CUSUM não reinicia na troca de bundle e isso nunca foi testado. Hipótese nova,
    nascida deste diagnóstico: só com pré-registro próprio.

Uso:  PYTHONPATH=. python memoria_nas_comparacoes.py
"""
from __future__ import annotations
import io, contextlib
import numpy as np, pandas as pd
import regua_fp as R
import avalia as AV

with contextlib.redirect_stdout(io.StringIO()):
    import f1_exclusao as F1
    import r1_biblioteca as R1
    import r3_guarda as R3
    import f3_calibracao_separada as F3
    import dpca as DP

DB, idx, mask, alvo = R.DB, R.idx, R.mask, R.alvo
JAN = R.JAN
HORIZ = {"7d": pd.Timedelta(days=7), "48h": pd.Timedelta(hours=48)}
N_ZERA = 60                      # 2 h de reset: 0,25^60 ~ 1e-36, acumulador zerado
MIN_VOTO = DB.DUR_MIN // 2       # 120 min em amostras de 2 min

BRACOS = {
    "referência":  (lambda d: R.sinais(d), R.DIAS),
    "F1":          (lambda d: F1.com_exclusao(d, F1.gatilhos(d))[0], R.DIAS),
    "R1 mín K=2":  (lambda d: R1.sinais(d, "minimo", 2), R.DIAS),
    "F3 +-7 d":    (lambda d: F3.sinais(d, pd.Timedelta(days=7), "+-7 d"), R.DIAS),
    "DPCA 0-10-30": (lambda d: DP.sinais(d, (0, 5, 15), "lags 0-10-30 min"), R.DIAS),
    "R3 G=16":     (lambda d: R3.sinais(16, d), tuple(R3.DESLOC)),
}


def zera_em(t0: pd.Timestamp) -> np.ndarray:
    r = np.zeros(len(idx), bool)
    i = int(idx.searchsorted(t0))
    r[i:i + N_ZERA] = True
    return r


def voto_na_janela(voto: pd.Series, T: pd.Timestamp) -> int:
    v = voto.loc[(idx >= T - JAN) & (idx < T)]
    return int(v.sum())


def vivo(out: dict) -> pd.Series:
    """O voto refeito só com canais cujo EWMA está acima do próprio limiar agora."""
    m_d = mask
    A = {c: (out["EW"][c].where(m_d) > DB.BASE[c] * DB.K_LO[c]) for c in R.SIN}
    B = {c: (out["EW"][c].where(m_d) > DB.BASE[c] * DB.KH[c]) for c in R.SIN}
    vA = sum(A[c].astype(int) for c in R.SIN) >= DB.VOTO_LO
    vB = (sum(B[c].astype(int) for c in R.SIN) >= DB.VOTO_HI) & (B["sp"] | B["vb"])
    return (vA | vB) & m_d


def cenario(sig) -> tuple[dict, list[dict]]:
    out = R.detector(*sig)
    fin = out["fin"]
    m = R.mede(fin)
    cls = m.pop("cls")
    nulo = AV.permuta(fin, mask, m["det"], len(alvo))["nulo"]
    viv = vivo(out).to_numpy()
    f = fin.to_numpy()
    h_carga = h_mem = 0
    for a, b, kk, _ in cls:
        if kk in ("FP", "NEUTRO"):
            s = (idx >= a) & (idx <= b) & f
            h_carga += int(s.sum()); h_mem += int((s & ~viv).sum())
    ev = []
    for T in alvo:
        d = bool(fin.loc[(idx >= T - JAN) & (idx < T)].any())
        e = dict(T=T.strftime("%Y-%m-%d"), det=d, voto=voto_na_janela(out["voto"], T) >= MIN_VOTO)
        for k, X in HORIZ.items():
            o = R.detector(*sig, reset_extra=zera_em(T - X))
            e[k] = voto_na_janela(o["voto"], T) >= MIN_VOTO
        ev.append(e)
    E = pd.DataFrame(ev)
    resumo = dict(**m, acaso=nulo,
                  fresca_7d=int((E.det & E["7d"]).sum()), fresca_48h=int((E.det & E["48h"]).sum()),
                  memoria=int((E.det & ~E["7d"]).sum()),
                  carga_sem_evid=h_mem / h_carga if h_carga else np.nan,
                  proxy_ok=int((E.det == E.voto).sum()))
    return resumo, ev


def main():
    pd.set_option("display.width", 220)
    linhas, eventos = [], []
    for nome, (ger, cen) in BRACOS.items():
        for c in cen:
            r, ev = cenario(ger(c))
            linhas.append(dict(braco=nome, cenario=c, **r))
            eventos += [dict(braco=nome, cenario=c, **e) for e in ev]
            print(f"{nome:13s} {c:2d}: det {r['det']} acaso {r['acaso']:.2f} fresca7d {r['fresca_7d']} "
                  f"fresca48h {r['fresca_48h']} memória {r['memoria']} carga {r['carga_mes']:.1f} "
                  f"sem evidência {100 * r['carga_sem_evid']:.0f}% proxy {r['proxy_ok']}/8", flush=True)
    L, E = pd.DataFrame(linhas), pd.DataFrame(eventos)
    L.to_csv(R.CACHE / "memoria_nas_comparacoes.csv", index=False)
    E.to_csv(R.CACHE / "memoria_nas_comparacoes_eventos.csv", index=False)

    print("\nMEDIANAS POR BRAÇO (carga = média):")
    g = L.groupby("braco", sort=False)
    S = g[["det", "acaso", "fresca_7d", "fresca_48h", "memoria", "inicio", "banda", "fp_mes"]].median()
    S["acima_acaso"] = g.apply(lambda x: (x.det - x.acaso).median())
    S["carga"] = g.carga_mes.mean()
    S["sem_evid_%"] = 100 * g.carga_sem_evid.median()
    print(S.round(2).to_string())

    print("\nREFERÊNCIA, POR EVENTO (em quantas das 8 composições):")
    ref = E[E.braco == "referência"]
    P = ref.groupby("T").agg(det=("det", "sum"), fresca_7d=("7d", lambda s: int((s & ref.loc[s.index, "det"]).sum())),
                             fresca_48h=("48h", lambda s: int((s & ref.loc[s.index, "det"]).sum())))
    print(P.to_string())

    print("\nO QUE CADA BRAÇO PAREADO PERDE E GANHA CONTRA A REFERÊNCIA (pares composição x evento):")
    rk = ref.set_index(["cenario", "T"])
    for nome in [b for b in BRACOS if b not in ("referência", "R3 G=16")]:
        v = E[E.braco == nome].set_index(["cenario", "T"])
        perde = rk.det & ~v.det
        ganha = ~rk.det & v.det
        print(f"  {nome:13s} perde {int(perde.sum()):2d} (memória na ref: {int((perde & ~rk['7d']).sum())}, "
              f"fresca: {int((perde & rk['7d']).sum())})   ganha {int(ganha.sum()):2d} "
              f"(fresca no braço: {int((ganha & v['7d']).sum())})   "
              f"frescas 7d: ref {int((rk.det & rk['7d']).sum())} -> {int((v.det & v['7d']).sum())}")


if __name__ == "__main__":
    main()
