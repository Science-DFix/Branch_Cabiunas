#!/usr/bin/env python3
"""Os degraus nos sensores acontecem nas manutenções? E os alarmes, depois delas?

CONTEXTO (`drift_nos_dados.py`). O "Comando Manutenção" (HSX_6240001A) só liga com
a máquina parada, 79 vezes, às vezes por semanas. Os sensores dão saltos de mais de
3 sigma de uma semana para outra (vibração, mancal LNA, diferenciais de filtro). E o
canal p é comandado por diferenciais de filtro e de selagem.

AS PERGUNTAS:
  1. os saltos semanais > 3 sigma acontecem mais em semanas que atravessam uma
     manutenção do que o acaso daria?
  2. os diferenciais de filtro (PDI_0301 gás de selagem, PDI_0338 óleo) são dente
     de serra -- sobem entre manutenções e caem nelas?
  3. os episódios FP e NEUTRO nascem mais nas semanas logo após uma manutenção,
     quando o baseline do mês ainda é feito de dado de antes dela?

Se as três forem sim, a ideia para o retreino é disparar por EVENTO (depois de cada
manutenção, assim que houver dado estável suficiente), não só pelo calendário.

Manutenção = HSX ligado por >= 24 h, trechos a menos de 24 h fundidos.

RESULTADO (30/09/2026).

1. SIM, e é o achado central. Contando SEMANAS (os saltos de tags diferentes na
   mesma semana não são independentes):
       >= 5 tags saltam > 3 sigma juntas    75% das semanas com manutenção x 40%   p 0,012
       >= 10 tags                           50% x 18%                              p 0,011
       tags saltando por semana (mediana)   14 x 4
   A máquina volta de uma manutenção com um "normal" novo em muitos sensores de uma
   vez: drift ABRUPTO disparado por evento conhecido. Mas 65% dos saltos acontecem
   fora de manutenção -- há também drift sem evento marcado.
2. NÃO como dente de serra. O filtro de óleo (PDI_0338) é praticamente constante;
   o de gás de selagem (PDI_0301) sobe devagar (+0,18 sigma/mês, 6 de 8 trechos) e
   cai na manutenção grande de nov/2024 (-3,9 sigma). O caso que importa é outro:
   em nov/2025 o PDI_0301 foi de +0,245 (out) para -0,400 kgf/cm² depois da
   manutenção de 05-15/11 e ficou lá em dezembro. Diferencial NEGATIVO em filtro
   sugere instrumento rezerado ou trocado. O bundle de novembro (baseline de
   outubro) viu isso como anomalia o mês inteiro -- o p ficou aceso 96% de
   novembro --, e só o de dezembro absorveu.
3. NÃO para os FP: nenhum dos 4 nasce nos 21 dias após uma manutenção. Os NEUTRO
   um pouco mais (2-3 de 7, enriquecimento ~2x), mas são números pequenos. O
   drift pós-manutenção satura canais; não é, no ponto publicado, fonte de FP.

Uso:  PYTHONPATH=. python drift_eventos.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import binomtest
import regua_fp as R
import drift_nos_dados as DN

G, VIG, C = DN.G, DN.VIG, DN.C
CURTO = DN.CURTO
FILTROS = ["954005_624_PDI_0301", "954005_624_PDI_0338"]


def manutencoes(min_h=24.0, junta_h=24.0):
    h = (G["HSX_6240001A"].fillna(0) > 0.5).to_numpy()
    idx = G.index
    d = np.diff(np.concatenate(([0], h.astype(int), [0])))
    ini, fim = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    tr = [[idx[a], idx[b]] for a, b in zip(ini, fim)]
    junt = []
    for a, b in tr:
        if junt and (a - junt[-1][1]) <= pd.Timedelta(hours=junta_h):
            junt[-1][1] = b
        else:
            junt.append([a, b])
    return [(a, b) for a, b in junt if (b - a) >= pd.Timedelta(hours=min_h)]


def saltos_x_manutencao(man):
    tags = C.TEMPERATURE_TAGS + C.PRESSURE_TAGS + C.VIBRATION_TAGS
    X = G.loc[VIG, tags]
    per = X.index.to_period("W")
    sem = X.groupby(per)
    n = sem.size(); ok = n[n >= 2 * 720].index
    med = sem.median().loc[ok]
    dent = sem.agg(lambda v: 1.4826 * (v - v.median()).abs().median()).loc[ok]
    # cada transição entre semanas consecutivas com dado: atravessa manutenção?
    semanas = list(ok)
    atravessa = []
    for w0, w1 in zip(semanas[:-1], semanas[1:]):
        a, b = w0.start_time.tz_localize("UTC"), w1.end_time.tz_localize("UTC")
        atravessa.append(any(ma <= b and mb >= a for ma, mb in man))
    atravessa = np.array(atravessa)
    L = []
    for c in tags:
        sd = float(np.nanmedian(dent[c]))
        if not np.isfinite(sd) or sd <= 0:
            continue
        dm = (med[c].diff().iloc[1:].to_numpy()) / sd
        j = np.abs(dm) > 3
        if j.sum() == 0:
            continue
        L.append(dict(tag=CURTO(c), saltos=int(j.sum()), em_manutencao=int((j & atravessa).sum())))
    T = pd.DataFrame(L)
    base = float(atravessa.mean())
    k, n_ = int(T.em_manutencao.sum()), int(T.saltos.sum())
    return T, base, k, n_, binomtest(k, n_, base, alternative="greater").pvalue


def dente_de_serra(man):
    """nível do filtro 3 dias antes e 3 dias depois de cada manutenção, e a
    inclinação entre manutenções."""
    L = []
    for c in FILTROS:
        s = G[c].where(pd.Series(VIG, index=G.index))
        sd = 1.4826 * (s - s.median()).abs().median()
        for a, b in man:
            antes = s.loc[a - pd.Timedelta(days=3):a].dropna()
            depois = s.loc[b:b + pd.Timedelta(days=3)].dropna()
            if len(antes) < 300 or len(depois) < 300:
                continue
            L.append(dict(tag=CURTO(c), manutencao=a.strftime("%Y-%m-%d"),
                          dias=round((b - a).total_seconds() / 86400, 1),
                          antes=float(antes.median()), depois=float(depois.median()),
                          delta_sigmas=float((depois.median() - antes.median()) / sd)))
        # inclinação entre manutenções: por trecho de >= 20 dias sem manutenção
        bordas = [G.index[0]] + [x for ab in man for x in ab] + [G.index[-1]]
        inc = []
        for a, b in zip(bordas[::2], bordas[1::2]):
            seg = s.loc[a:b].dropna()
            if (b - a) < pd.Timedelta(days=20) or len(seg) < 5000:
                continue
            dias = (seg.index - seg.index[0]).total_seconds() / 86400
            inc.append(np.polyfit(dias, seg.to_numpy(), 1)[0] / sd * 30)
        L.append(dict(tag=CURTO(c), manutencao="INCLINACAO", dias=np.nan, antes=np.nan, depois=np.nan,
                      delta_sigmas=float(np.median(inc)) if inc else np.nan,
                      n_trechos=len(inc), sobe_em=int(sum(1 for x in inc if x > 0))))
    return pd.DataFrame(L)


def alarmes_pos_manutencao(man, dias=(7, 14, 21)):
    t, p, ms, ds = R.sinais(1)
    fin = R.detector(t, p, ms, ds)["fin"]
    cls = R.DB.classifica_regra_c(R.AV.episodios(fin), R.PARADAS)
    vig = pd.Series(R.mask.to_numpy(), index=R.idx)
    L = []
    for D in dias:
        janela = pd.Series(False, index=R.idx)
        for a, b in man:
            janela.loc[b:b + pd.Timedelta(days=D)] = True
        frac = float((janela & vig).sum() / vig.sum())
        for k in ("FP", "NEUTRO", "TP"):
            E = [(a, b) for a, b, kk, _ in cls if kk == k]
            dentro = sum(1 for a, _ in E if janela.loc[a])
            h_d = sum((b - a).total_seconds() / 3600 for a, b in E if janela.loc[a])
            h_t = sum((b - a).total_seconds() / 3600 for a, b in E)
            L.append(dict(janela_dias=D, tempo_vigiado_na_janela=frac, classe=k, episodios=len(E),
                          nascem_na_janela=dentro, horas_na_janela=h_d, horas_total=h_t,
                          enriq_episodios=(dentro / len(E) / frac) if len(E) and frac else np.nan,
                          enriq_horas=(h_d / h_t / frac) if h_t and frac else np.nan))
    return pd.DataFrame(L)


def main():
    pd.set_option("display.width", 200)
    man = manutencoes()
    print(f"MANUTENÇÕES (HSX ligado >= 24 h): {len(man)}")
    for a, b in man:
        print(f"   {a:%Y-%m-%d} a {b:%Y-%m-%d}  ({(b - a).total_seconds() / 86400:5.1f} dias)")
    T, base, k, n, p = saltos_x_manutencao(man)
    print(f"\n1. SALTOS SEMANAIS > 3 sigma: {n} no total; {k} em transições que atravessam manutenção"
          f" ({100 * k / n:.0f}%), contra {100 * base:.0f}% das transições que atravessam")
    print(f"   enriquecimento {k / n / base:.2f}x   p (binomial, 1 lado) = {p:.2g}")
    T["frac_em_manutencao"] = T.em_manutencao / T.saltos
    print(T.sort_values("saltos", ascending=False).round(2).to_string(index=False))
    S = dente_de_serra(man)
    print("\n2. FILTROS — nível antes e depois de cada manutenção (sigmas), e inclinação entre elas (sigma/mês)")
    print(S.round(2).to_string(index=False))
    A = alarmes_pos_manutencao(man)
    print("\n3. EPISÓDIOS QUE NASCEM NOS DIAS APÓS UMA MANUTENÇÃO (dia 1, detector atual)")
    print(A.round(2).to_string(index=False))
    T.to_csv(R.CACHE / "drift_saltos.csv", index=False); S.to_csv(R.CACHE / "drift_filtros.csv", index=False)
    A.to_csv(R.CACHE / "drift_pos_manutencao.csv", index=False)


if __name__ == "__main__":
    main()
