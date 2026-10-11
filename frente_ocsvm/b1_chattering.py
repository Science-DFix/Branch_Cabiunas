#!/usr/bin/env python3
"""B1 -- contar só ativações NOVAS (após silêncio) no canal 4 baixa o FP sem perder trip?

PRÉ-REGISTRADO EM 10/10/2026, ANTES DE RODAR. O canal 4 acende se alguma das 5 tags alarmou nas últimas
24 h; hoje cada repetição de um alarme que fica batendo (chattering) RENOVA as 24 h. O A2 mostrou que o
canal 4 é o filtro de CUSTO da pipeline (4,67 -> 2,88 FP/mês) e o A3, que `PI_6240319_AL` é a única tag
indispensável (sem ela 6/8).

OLHADO ANTES (descritivo do catálogo, 01/07/2024-20/04/2026, sem trips; régua de 24 h):
  tag              ativações  intervalo mediano  < 1 h   < 24 h   duty sozinha
  PI_6240319_AL       409          14,5 h        19%     60%      34,9%
  PAL_6240315         326          14,6 h        31%     58%      26,3%
  PDAL_6240302         51         134 h          16%     26%       6,0%
  TC382_05_A           38          49 h          38%     49%       3,3%
  PAH_6240319          31          11,9 h        23%     60%       2,6%
  canal 4 (OU das 5): duty 46,5% do índice inteiro (46,8% em operação, A0).
Ou seja: há rajadas (1/5 a 1/3 das repetições vêm em menos de 1 h), mas o grosso do duty vem de alarmes
espaçados de horas -- o chattering curto sozinho deve mexer pouco.

O TRATAMENTO (debounce por tag). Uma ativação de uma tag só conta se a ATIVAÇÃO ANTERIOR DA MESMA TAG
(contada ou não) foi há >= S horas. Uma rajada contínua vira um alarme só, no seu início, e o canal fica
aceso 24 h a partir dele -- não 24 h a partir da última repetição.
  S em {0 (= referência), 0,5, 1, 2, 4, 8, 12, 24} h.
  B1a: debounce só em `PI_6240319_AL` (as outras 4 como hoje).
  B1b: debounce nas 5 tags, mesmo S.
Fixo em tudo: 3 canais OCSVM de produção, voto >= 2, 5 tags, janela 24 h, filtro 45 min, refratário 48 h.

COMO S É ESCOLHIDO (lição do A3: escolher vendo os 8 trips infla o resultado). Por braço, LOEO sobre S:
para cada trip e, o S que detecta mais dos outros 7; empate, menor FP/mês; empate, menor S. Avalia se
detecta e.
  PRIMÁRIO, por braço: (1) detecções LOEO (de 8) e fora da amostra (de 3); (2) o S "final" -- mesmo
  critério com os 8 trips -- e a diferença de FP/mês contra a referência, com IC 95% por bootstrap em
  blocos de mês-calendário (2.000 reamostragens dos meses; FP/mês = episódios FP / dias "on" x 30,4375,
  como a régua).
  SECUNDÁRIOS: duty do canal 4 em operação por S (mecanismo); detecções, banda de 4 h e FP/mês por S;
  quais trips caem com qual S.

CRITÉRIO DE ACEITE (por braço; o braço aceito com menor FP vence):
  LOEO 8/8  E  3/3 fora da amostra com o S final  E  FP/mês do S final <= 2,60 (-10%) com o IC 95% da
  diferença inteiro abaixo de 0.
  Referência de comparação, também anotada: o A3 já achou 2,61 FP/mês com as 5 tags só mexendo na
  temporização (45 min, 6 h, 36 h). B1 que não bata isso não justifica uma regra nova no canal.

EXPECTATIVA REGISTRADA. O duty do canal 4 cai pouco até S = 2 h (< 3 pontos) e mais a partir de 8 h. FP/mês
melhora pouco: no melhor S de B1a, 2,6-2,8; o IC da diferença cruza 0. LOEO 7/8 a 8/8 em B1a; em B1b, se
algum trip cai, é 09/12/25 (depende de PAL_6240315/PDAL_6240302). Leitura prevista: REPROVADO pelo critério
de custo -- o chattering não é o que mantém o canal 4 aceso; o que mantém são alarmes recorrentes espaçados.

RESULTADO (10/10/2026) -- REPROVADO NOS DOIS BRAÇOS, COMO PREVISTO.
  Até S = 8 h o debounce quase não mexe: duty do canal 4 em operação 46,8% -> 46,5% (B1a) / 45,9% (B1b) e
  FP/mês parado em 2,885. Só S = 24 h mexe: duty 43,0% / 40,0%. Nenhum S, em nenhum braço, perde trip
  (8/8, fora 3/3, banda 7/8 em todos) -- o 09/12/25 que eu temia em B1b não caiu.
  LOEO 8/8 nos dois (todas as dobras escolhem S = 24 h).
  B1a  S = 24 h: 2,816 FP/mês, diferença -0,069 (IC 95% -0,220 a +0,000).
  B1b  S = 24 h: 2,747 FP/mês, diferença -0,137 (IC 95% -0,338 a +0,000).
  Falha no critério de custo (> 2,60 e o IC encosta em 0) e não bate os 2,61 do A3 só com temporização.
  Expectativa: acertada (duty quase parado até 2 h, FP 2,6-2,8, LOEO 7-8/8); errada só no 09/12/25.
  Leitura: o chattering curto não é o que mantém o canal 4 aceso; o que mantém são alarmes recorrentes
  espaçados de horas. Debounce não vira regra.

Uso:  python frente_ocsvm/b1_chattering.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import comum as C

S_H = [0, 0.5, 1, 2, 4, 8, 12, 24]
BRACOS = {"B1a": ["PI_6240319_AL"], "B1b": list(C.TAGS_C4)}
N_BOOT, SEED = 2000, 20261010
REF = C.REF


def debounce(t: pd.Series, s_h: float) -> pd.Series:
    """Mantém a ativação só se a anterior da mesma tag (contada ou não) foi há >= s_h horas."""
    x = np.sort(pd.DatetimeIndex(t).values)
    if s_h <= 0 or len(x) == 0:
        return pd.Series(x)
    gap = np.diff(x).astype("timedelta64[s]").astype(np.float64) / 3600.0
    return pd.Series(x[np.r_[True, gap >= s_h]])


def canal4(idx, cat, alvo: list[str], s_h: float) -> pd.Series:
    tempos = [debounce(cat.loc[cat.tag == t, "t"], s_h if t in alvo else 0) for t in C.TAGS_C4]
    return C.canal_alarme(idx, pd.concat(tempos), REF["janela_c4_h"])


def fp_por_mes(cls: pd.DataFrame, op: pd.Series) -> pd.DataFrame:
    """Episódios FP e dias 'on' por mês-calendário (blocos do bootstrap)."""
    dt = op.index.to_series().diff().median().total_seconds() / 86400.0
    dias = (op == "on").groupby(op.index.to_period("M")).sum() * dt
    fp = cls.loc[cls.classe == "falso_positivo", "start"]
    n = pd.to_datetime(fp).dt.to_period("M").value_counts()
    return pd.DataFrame({"fp": n.reindex(dias.index, fill_value=0), "dias": dias})


def roda(df, cat, ft, alvo, s_h):
    c = {k: df[k].astype(bool) for k in C.CANAIS[:3]}
    c["c4"] = canal4(df.index, cat, alvo, s_h)
    fin = C.decide(c, REF["min_votes"], REF["dur_min"], REF["refrat_h"])
    cls, m = C.avalia(fin, df["operational_state"], ft)
    P = C.por_trip(cls, ft)
    duty = float(c["c4"][df["operational_state"] == "on"].mean())
    return dict(s=s_h, det=P.deteccao.notna().to_numpy(), banda=P.banda_4h.to_numpy(),
                fp=m["falso_positivo_por_mes"], duty=duty, meses=fp_por_mes(cls, df["operational_state"]))


def escolhe(R: list[dict], fora: int | None = None) -> dict:
    cand = [(-(np.delete(r["det"], fora).sum() if fora is not None else r["det"].sum()), r["fp"], r["s"], i)
            for i, r in enumerate(R)]
    return R[min(cand)[3]]


def ic_diff(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    k = len(a)
    f = lambda m, i: m.fp.to_numpy()[i].sum() / m.dias.to_numpy()[i].sum() * 30.4375
    obs = f(a, np.arange(k)) - f(b, np.arange(k))
    bs = [f(a, i) - f(b, i) for i in (rng.integers(0, k, k) for _ in range(N_BOOT))]
    return obs, *np.percentile(bs, [2.5, 97.5])


def main():
    df, cat = C.carrega_canais(), C.catalogo()
    ft = C.trips(df.index)
    nomes = [t.strftime("%d/%m/%y") for t in ft]
    fora = (ft >= C.SPLIT).to_numpy()
    for braco, alvo in BRACOS.items():
        R = [roda(df, cat, ft, alvo, s) for s in S_H]
        ref = R[0]
        print(f"\n=== {braco}: debounce em {alvo}")
        print(f"{'S (h)':>6} {'duty c4':>8} {'det':>4} {'fora':>4} {'banda':>5} {'FP/mês':>7}  perde")
        for r in R:
            perde = [n for n, d in zip(nomes, r["det"]) if not d]
            print(f"{r['s']:>6} {r['duty']:>8.3f} {r['det'].sum():>4} {r['det'][fora].sum():>4} "
                  f"{r['banda'].sum():>5} {r['fp']:>7.3f}  {', '.join(perde) or '-'}")
        L = [(n, escolhe(R, e)) for e, n in enumerate(nomes)]
        det = [bool(r["det"][e]) for e, (n, r) in enumerate(L)]
        print(f"LOEO: {sum(det)}/8 (fora {sum(np.array(det)[fora])}/3) | S escolhido por trip: "
              + ", ".join(f"{n}={r['s']}" for n, r in L))
        fin = escolhe(R)
        obs, lo, hi = ic_diff(fin["meses"], ref["meses"])
        aceita = (sum(det) == 8 and fin["det"][fora].sum() == 3 and fin["fp"] <= 2.60 and hi < 0)
        print(f"S final = {fin['s']} h: {fin['det'].sum()}/8, fora {fin['det'][fora].sum()}/3, "
              f"{fin['fp']:.3f} FP/mês | diferença vs referência {obs:+.3f} (IC 95% {lo:+.3f} a {hi:+.3f})"
              f" | {'ACEITO' if aceita else 'REPROVADO'}")


if __name__ == "__main__":
    main()
