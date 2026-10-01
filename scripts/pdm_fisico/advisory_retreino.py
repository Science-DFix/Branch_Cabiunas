#!/usr/bin/env python3
"""O "Retrain Advisory": dá para saber, olhando a saída do detector, quando retreinar?

O REQUISITO. O cliente quer retreinar só quando pedir. Para pedir na hora certa,
precisa de um aviso com evidência. O KS e o PSI sobre as ENTRADAS não servem: toda
semana já está em drift contra o baseline (`drift_tecnicas.py`). A curva de idade
(`idade_referencia.py`) mostrou onde o envelhecimento aparece de verdade: na SAÍDA.
Com um mês a mais de idade, o canal p passa de 41% para 63% do tempo normal aceso.

AS DUAS MEDIDAS, por semana, só sobre o tempo vigiado:
  duty_t, duty_p   fração do tempo em que o canal está aceso no nível A
  troca_t, troca_p estabilidade do ranking de contribuições: a cada instante, o
                   sensor que dá o máximo do erro normalizado (é ele que acende o
                   canal). Compara a distribuição desse "sensor dominante" na
                   semana com a do baseline do próprio bundle, pela distância de
                   variação total (0 = mesma distribuição, 1 = disjuntas). Sobe
                   quando um sensor NOVO passa a comandar o canal -- o PDI_0301 em
                   nov/2025.
  e o máximo de cada par (duty_max, troca_max).

O RISCO. Canal aceso também é o que a degradação real faz. Um aviso que dispara nos
dias antes de um trip manda retreinar na pior hora: o bundle novo aprende a falha
como normal. Por isso cada medida é olhada também nas semanas pré-trip.

PARTE A -- separa? Para cada composição, o histórico é pontuado duas vezes: pelo
bundle do mês (novo, 0-30 dias) e pelo do mês anterior (velho, 30-60 dias). AUC de
cada medida, novo contra velho, nas semanas normais. Limiar = p90 das semanas
normais com bundle novo; quanto das velhas ele pega e quanto das pré-trip com bundle
novo ele dispara (o falso "retreine agora").

PARTE B -- funciona como política? Retreino SÓ quando o aviso dispara. Checagem
diária, às 00:00: a medida dos últimos 7 dias acima do limiar (com >= 48 h vigiadas
na janela) e >= 14 dias desde o último retreino -> retreina ali, com o dado até ali.
Teto de 60 dias: sem aviso, retreina mesmo assim (senão o bundle vence e o detector
fica cego -- `aprovacao_operador.py`); os retreinos forçados são contados à parte.
Mesmas 8 composições (o 1º retreino no dia da composição), mesmos limiares.

CRITÉRIO, ESCRITO ANTES DE RODAR.
  Parte A: a medida "serve" se AUC >= 0,70 e dispara em <= 20% das semanas pré-trip
           com bundle novo.
  Parte B: usa a medida de MAIOR AUC na Parte A, com limiar p75 e p90 das semanas
           normais com bundle novo (dois valores, os dois relatados). Tolerável como
           em `aprovacao_operador.py`: det, início e banda não caem mais que 0,5 na
           mediana, FP/mês e carga não sobem mais que 10%. Relata o número de
           retreinos por ano e quantos foram forçados pelo teto.
O limiar é calibrado nos mesmos dados em que é avaliado: otimista, e registrado.

RESULTADO (30/09/2026) -- O AVISO PELA SAÍDA REPROVA, nas duas partes.

PARTE A. Nenhuma medida passa (AUC >= 0,70):
    medida      AUC   mediana novo  mediana velho   dispara pré-trip (limiar p90)
    duty_t      0,60     0,006          0,742          0%  (p90 = 1,000: ver abaixo)
    duty_p      0,63     0,305          0,989          0%
    troca_t     0,67     0,616          0,727         11%
    troca_p     0,67     0,538          0,719          6%
    troca_max   0,69     0,687          0,804         13%
  · O duty SEMANAL é quase binário: com bundle novo, 16% das semanas normais têm o p
    aceso a semana inteira e 37% apagado; com o velho, 32% e 23%. A memória do CUSUM
    faz do canal um estado. A mediana separa (0,31 contra 0,99), a AUC não (0,63), e o
    p90 cai em 1,000 cravado -- o limiar ">" nunca dispara. O empate não foi previsto
    no pré-registro; o veredito vem da AUC, que não usa limiar.
  · PAREADO (champion contra challenger na MESMA semana): nas semanas normais o velho
    acende mais que o novo em só 40-43%; nas pré-trip, em 59-61%. A evidência de
    "precisa retreinar" fica MAIS forte justamente antes da falha -- o aviso explicaria
    o alarme verdadeiro como modelo velho.

PARTE B. Com a melhor medida (troca_max):
    política              det  início  banda  FP/mês  carga  retreinos/ano (aviso+teto)
    mensal automático     6,5   6,0    4,5    0,861   130,5      12
    aviso > p75           5,0   5,0    4,0    1,206   112,1      10,8  (19 + 5)
    aviso > p90           4,0   3,0    2,0    1,033    94,7       8,5  (12 + 7)
  · Perde 1,5 a 2,5 detecções sem economizar retreino. Os três trips que o p75 perde
    (17/03, 29/04 e 09/12/2025) tiveram um retreino por aviso 4, 10 e 0 dias antes --
    o de 09/12 às 00:00 do dia do trip, com a degradação dentro da referência.
  · O aviso não se concentra antes dos trips (7 de 19 a até 21 dias de um trip, contra
    34% do tempo vigiado): não é atraído pela falha, mas também não a evita. Retreina
    em hora aleatória, como o calendário, e sem as 8 composições para diluir o azar.
  · As 8 composições convergem para a mesma trajetória (carga 112,1 em 7 de 8): uma
    política por evento não depende do dia do mês, mas a evidência equivale a ~1
    cenário, não 8.
  · Uso que sobra: a troca do sensor dominante é boa INFORMAÇÃO no painel ("quem está
    acendendo o canal agora"), não gatilho de retreino.

Uso:  PYTHONPATH=. python advisory_retreino.py [A|B|resumo]
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import regua_fp as R
import aprovacao_operador as AO
import idade_referencia as IR

C, DF, IX, STABLE, FIT = AO.C, AO.DF, AO.IX, AO.STABLE, AO.FIT
FAM = {"t": C.TEMPERATURE_TAGS, "p": C.PRESSURE_TAGS}
SEM_VENC = pd.Timedelta(days=100_000)
JAN = pd.Timedelta(days=7)
MIN_INT, TETO = pd.Timedelta(days=14), pd.Timedelta(days=60)
MEDIDAS = ["duty_t", "duty_p", "duty_max", "troca_t", "troca_p", "troca_max"]
_BASE_SHARE: dict = {}


def pre_trip() -> np.ndarray:
    pre = pd.Series(False, index=R.idx)
    for t in R.alvo:
        pre.loc[t - JAN:t] = True
    return pre.to_numpy()


PRE = pre_trip()


def dominante(sc, X: pd.DataFrame) -> np.ndarray:
    """índice do sensor que dá o máximo do erro normalizado (-1 sem dado)."""
    out = np.full(len(X), -1, dtype=np.int8)
    ok = X.notna().all(axis=1).to_numpy()
    if ok.any():
        Xs = sc.scaler.transform(X[ok])
        e = (Xs - sc.pca.inverse_transform(sc.pca.transform(Xs))) ** 2 / sc.sens_p99_
        out[ok] = np.argmax(e, axis=1)
    return out


def share_base(ref_fim, fam) -> np.ndarray:
    k = (ref_fim, fam)
    if k not in _BASE_SHARE:
        fit = DF.loc[STABLE & (IX < ref_fim), C.SENSOR_TAGS].dropna().tail(FIT)
        sc = AO.bundle(ref_fim)[0 if fam == "t" else 1]
        d = dominante(sc, fit[FAM[fam]])
        _BASE_SHARE[k] = np.bincount(d[d >= 0], minlength=len(FAM[fam])) / max((d >= 0).sum(), 1)
    return _BASE_SHARE[k]


def serie_com_dominante(plano):
    """sinais + sensor dominante + id do segmento (qual bundle serve cada instante)."""
    S = AO.de_plano(plano, SEM_VENC)
    dom = {f: np.full(len(IX), -1, dtype=np.int8) for f in FAM}
    seg = np.full(len(IX), -1, dtype=np.int32)
    pubs = sorted(plano)
    for i, (pub, ref) in enumerate(pubs):
        fim = pubs[i + 1][0] if i + 1 < len(pubs) else AO.FIM
        s = (IX >= pub) & (IX < fim)
        bd = AO.bundle(ref)
        if bd is None or not s.any():
            continue
        seg[s] = i
        for j, f in enumerate(FAM):
            dom[f][s] = dominante(bd[j], DF.loc[s, FAM[f]])
    return S, dom, seg, pubs


def semanas(S, dom, seg, pubs) -> pd.DataFrame:
    out = R.detector(S.t, S.p, S.ms, S.ds)
    A = {c: out["A"][c].to_numpy() for c in ("t", "p")}
    sem = IX.floor("7D")
    L = []
    df = pd.DataFrame({"sem": sem, "seg": seg, "vig": AO.MASK, "pre": PRE,
                       "at": A["t"], "ap": A["p"], "dt": dom["t"], "dp": dom["p"],
                       "idade": S.idade})
    df = df[df.vig & (df.seg >= 0)]
    for (w, sg), G in df.groupby(["sem", "seg"]):
        if len(G) < 2 * 720:                      # >= 2 dias vigiados
            continue
        ref = pubs[sg][1]
        r = dict(semana=w, idade=float(G.idade.mean()), pre_trip=bool(G.pre.any()),
                 duty_t=float(G["at"].mean()), duty_p=float(G["ap"].mean()))
        for f, col in (("t", "dt"), ("p", "dp")):
            d = G[col].to_numpy(); d = d[d >= 0]
            sh = np.bincount(d, minlength=len(FAM[f])) / max(len(d), 1)
            r[f"troca_{f}"] = float(0.5 * np.abs(sh - share_base(ref, f)).sum())
        r["duty_max"] = max(r["duty_t"], r["duty_p"]); r["troca_max"] = max(r["troca_t"], r["troca_p"])
        L.append(r)
    return pd.DataFrame(L)


def plano_idade(dia, k):
    cs = AO.cortes(dia)
    validos = [c for c in cs if AO.bundle(c) is not None]
    P = []
    for i, c in enumerate(cs):
        ref = cs[i - k] if i - k >= 0 else cs[0]
        if AO.bundle(ref) is None:
            ref = validos[0] if validos[0] <= c else None
        if ref is not None:
            P.append((c, ref))
    return P


# ══════════════════════════════════════════════════ parte A
def parte_a():
    L = []
    for d in R.DIAS:
        for k, rot in ((0, "novo"), (1, "velho")):
            W = semanas(*serie_com_dominante(plano_idade(d, k)))
            L.append(W.assign(dia=d, bundle=rot))
        print("A", d, flush=True)
    T = pd.concat(L)
    T.to_csv(R.CACHE / "advisory_semanas.csv", index=False)
    resumo_a(T)


def resumo_a(T):
    nor = T[~T.pre_trip]
    novo, velho = nor[nor.bundle == "novo"], nor[nor.bundle == "velho"]
    pre_novo = T[T.pre_trip & (T.bundle == "novo")]
    print(f"semanas normais: {len(novo)} com bundle novo, {len(velho)} com velho; "
          f"pré-trip com novo: {len(pre_novo)}\n")
    print(f"{'medida':10s} {'AUC':>5s} {'limiar p90':>11s} {'pega velhas':>12s} {'dispara pré-trip':>17s}"
          f" {'mediana novo':>13s} {'mediana velho':>14s}  serve?")
    res = {}
    for m in MEDIDAS:
        y = np.r_[np.zeros(len(novo)), np.ones(len(velho))]
        auc = roc_auc_score(y, np.r_[novo[m], velho[m]])
        thr = float(novo[m].quantile(0.90))
        pega, fals = float((velho[m] > thr).mean()), float((pre_novo[m] > thr).mean())
        res[m] = auc
        print(f"{m:10s} {auc:5.2f} {thr:11.3f} {100 * pega:11.0f}% {100 * fals:16.0f}%"
              f" {novo[m].median():13.3f} {velho[m].median():14.3f}  {'SIM' if auc >= .70 and fals <= .20 else 'não'}")
    return max(res, key=res.get)


# ══════════════════════════════════════════════════ parte B
class Plano:
    """sinais e sensor dominante montados em ordem; cada publicação vale até o fim,
    até a próxima sobrescrever (como `aprovacao_operador.Serie`)."""

    def __init__(self):
        self.S = AO.Serie(SEM_VENC)
        self.dom = {f: np.full(len(IX), -1, dtype=np.int8) for f in FAM}
        self.qual = np.full(len(IX), -1, dtype=np.int32)
        self.refs: list[pd.Timestamp] = []

    def publica(self, pub, ref):
        self.S.publica(pub, ref)
        s = IX >= pub
        self.refs.append(ref); self.qual[s] = len(self.refs) - 1
        bd = AO.bundle(ref)
        for j, f in enumerate(FAM):
            self.dom[f][s] = dominante(bd[j], DF.loc[s, FAM[f]])

    def troca(self, q, medida) -> float:
        """distância do sensor dominante nos 7 dias antes de `q` (nan se < 48 h vigiadas)."""
        a, b = IX.searchsorted(q - JAN), IX.searchsorted(q)
        vig = AO.MASK[a:b]
        if vig.sum() * 2 / 60 < 48:
            return np.nan
        ref = self.refs[self.qual[a:b][vig][-1]]
        ds = []
        for f in (["t", "p"] if medida == "troca_max" else [medida[-1]]):
            d = self.dom[f][a:b][vig]; d = d[d >= 0]
            sh = np.bincount(d, minlength=len(FAM[f])) / max(len(d), 1)
            ds.append(0.5 * np.abs(sh - share_base(ref, f)).sum())
        return float(max(ds))


def politica(dia, medida, thr) -> dict:
    """retreino só quando o aviso dispara, com intervalo mínimo e teto (ver docstring).
    Só as medidas de troca: elas não dependem do detector, que roda uma vez, no fim."""
    assert medida.startswith("troca")
    cs = AO.cortes(dia)
    c0 = next(c for c in cs if AO.bundle(c) is not None)
    P = Plano(); P.publica(c0, c0)
    ult, n_aviso, n_teto = c0, 0, 0
    for q in pd.date_range(c0.normalize() + pd.Timedelta(days=1), AO.FIM, freq="D"):
        if q < ult + MIN_INT:
            continue
        if q >= ult + TETO:
            P.publica(q, q); ult = q; n_teto += 1
        elif P.troca(q, medida) > thr:
            P.publica(q, q); ult = q; n_aviso += 1
    m = P.S.mede()
    anos = (IX[-1] - c0).total_seconds() / (365.25 * 86400)
    m.update(retreinos_ano=(n_aviso + n_teto) / anos, por_aviso=n_aviso, por_teto=n_teto)
    return m


def parte_b(quantis=(0.75, 0.90)):
    T = pd.read_csv(R.CACHE / "advisory_semanas.csv")
    melhor = resumo_a(T)
    novo = T[(~T.pre_trip) & (T.bundle == "novo")]
    L = []
    for q in quantis:
        thr = float(novo[melhor].quantile(q))
        for d in R.DIAS:
            L.append(dict(politica=f"aviso {melhor} > p{int(q * 100)} ({thr:.3f})", dia=d, **politica(d, melhor, thr)))
            print(L[-1]["politica"], d, round(L[-1]["carga_mes"], 1), L[-1]["por_aviso"], L[-1]["por_teto"], flush=True)
    pd.DataFrame(L).to_csv(R.CACHE / f"advisory_politica_p{'_'.join(str(int(q * 100)) for q in quantis)}.csv",
                           index=False)


def resumo_b():
    ref = R.distribuicao()
    r = ref.median()
    P = pd.concat([pd.read_csv(f) for f in sorted(R.CACHE.glob("advisory_politica_p*.csv"))])
    print(f"\n{'política':34s} {'det':>5s} {'início':>7s} {'banda':>6s} {'FP/mês':>7s} {'carga':>7s}"
          f" {'retreinos/ano':>14s} {'aviso':>6s} {'teto':>5s}  veredito")
    print(f"{'mensal automático (régua)':34s} {r.det:5.1f} {r.inicio:7.1f} {r.banda:6.1f} {r.fp_mes:7.3f} {r.carga_mes:7.1f} {12:14.1f}")
    for pol, G in P.groupby("politica", sort=False):
        m = G.median(numeric_only=True)
        tol = (all(m[c] >= r[c] - 0.5 for c in ("det", "inicio", "banda"))
               and m.fp_mes <= 1.1 * r.fp_mes and m.carga_mes <= 1.1 * r.carga_mes)
        print(f"{pol:34s} {m.det:5.1f} {m.inicio:7.1f} {m.banda:6.1f} {m.fp_mes:7.3f} {m.carga_mes:7.1f}"
              f" {m.retreinos_ano:14.1f} {m.por_aviso:6.1f} {m.por_teto:5.1f}  {'TOLERÁVEL' if tol else 'não'}")
        print(f"{'':34s} faixa det {G.det.min():.0f}-{G.det.max():.0f}, banda {G.banda.min():.0f}-{G.banda.max():.0f},"
              f" carga {G.carga_mes.min():.0f}-{G.carga_mes.max():.0f}")


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "resumo"
    if q.startswith("B"):
        parte_b((float(q[1:]) / 100,)) if len(q) > 1 else parte_b()
        sys.exit()
    {"A": parte_a}.get(q, lambda: (resumo_a(pd.read_csv(R.CACHE / "advisory_semanas.csv")),
                                                 resumo_b()))()
