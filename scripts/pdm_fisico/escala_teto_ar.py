#!/usr/bin/env python3
"""C1 (autoescala), C2 (teto antes da EWMA) e C3 (PI_5134001 fora do canal p).

PRÉ-REGISTRADO EM 02/10/2026, ANTES DE RODAR (commit deste arquivo sem resultado).

DE ONDE VÊM. De `escala_degenerada.py` (commit f67f3dd), diagnóstico de mecanismo que
não olhou os 8 trips -- mas olhou os mesmos dados, e isso fica dito antes:
  C1  depois do RobustScaler, o PI_0319 concentra 97,9% da variância escalada do p
      (24 de 27 bundles); o PCA de pressão retém k = 1 em 68% dos bundles e o p vira
      o máximo de desvios univariados ao quadrado, em IQR do mês. Autoescala
      (StandardScaler, variância unitária) é o padrão de PCA-MSPC: com ela nenhuma tag
      passa de 8,3% da variância e k fica em 8-10 todo mês.
  C2  o score cru do p passa de 1.000x o limiar em 0,37% do tempo vigiado. O teto de
      20x do CUSUM é aplicado DEPOIS da EWMA, e 25% da massa positiva do CUSUM do p
      (17% do t) vem de instantes com a EWMA >= 20x o limiar.
  C3  o PI_5134001 é o header de ar de instrumentação da U-5134, outra unidade
      (`metadata.csv`); responde por 8% das horas em que a EWMA do p satura.

O DESENHO. As 8 composições de sempre, pareadas com a referência. MESMAS LINHAS de
ajuste da referência em todos os braços (dropna sobre os 36 sensores, as 20.000
últimas estáveis antes do corte).
  C1  t e p reajustados em todos os bundles com StandardScaler no lugar do
      RobustScaler. O resto do ScorerMax fica igual: PCA com 95% da variância, piso
      PHI = 0,10 no normalizador por tag, recon_p99 do próprio baseline. sp e vb não
      mudam. Proposto para t E p -- é o padrão nas duas famílias --, embora o defeito
      só tenha sido medido em p.
  C2  o score cru de cada canal limitado a 20x o limiar ANTES da EWMA, com uma EWMA
      por nível (20x o limiar A no nível A, 20x o B no B). É a mesma constante do
      CUSUM, só muda de lugar: nenhum parâmetro novo. A força F, da escalada por
      idade, continua sobre a EWMA sem teto: com o teto nela, F nunca passaria de 20 e
      a escalada (ESC_ABS = 20) seria desligada junto, confundindo o teste.
  C3  o canal p refeito com 11 das 12 tags, sem o PI_5134001 (o mesmo laço do H2).
  `confere()` prova antes: o laço do C1 com RobustScaler reproduz t e p da referência
  nas 8 composições, e o detector do C2 com teto infinito reproduz o alarme da
  referência bit a bit nas 8.
DECISÃO: `bootstrap_regua.decide`, pareado, IC de 1 - 0,05/3 = 98,33% em cada
(Bonferroni sobre os três). GANHO = detecção estrita (nenhuma composição perde
detecção; medianas de início e banda não caem) + carga cai em >= 7/8 + IC abaixo de
zero + FP não sobe. Se dois ou três derem GANHO, a união dos vencedores roda e também
tem de dar GANHO (IC 95%) para seguirem juntos; se só um der, só ele segue. Um GANHO
aqui é candidato a validação em dado novo e à engenharia, não decisão de produção.
IMPRESSO SEMPRE, SEM VEREDITO (leitura de mecanismo): a decomposição do C1 em só-p e
só-t, e o k e a fração do p cru acima de 100x o limiar sob a autoescala.

EXPECTATIVA REGISTRADA.
  C1  a carga cai: o p deixa de ser bimodal e o k para de saltar entre meses. É a
      mudança maior e a mais incerta das três -- troca a escala de t e p, o recon_p99
      e o duty de dois canais. Risco alto em A: t e p entram no voto de quase todo
      nascimento.
  C2  a carga cai, menos que no C1 (só a parte saturada da memória). Risco em A:
      detecção sustentada por memória de valor gigante; o 26/02/2026, com o t aceso
      por memória de 1-2 semanas, é o candidato a cair.
  C3  efeito pequeno (2% das horas acesas do p): o mais provável é reprovar em B1, por
      não mudar a carga em >= 7 composições. Se reprovar só por isso, tirar a tag do
      canal fica como questão de higiene para a engenharia, não de desempenho.
RESSALVA: hipóteses nascidas de olhar estes dados (não os trips) e medidas nos mesmos
8 eventos de sempre. O padrão de toda a pesquisa é a fronteira: corte de tempo aceso
tem custado nascimento.

EQUIVALÊNCIAS (02/10/2026, antes do teste): o laço do C1 com RobustScaler reproduz t e
p da referência nas 8 composições (máx. 1,1e-4 em t e 1,2e-3 em p, relativo, ponto
flutuante), e o C2 com teto infinito reproduz o alarme da referência bit a bit nas 8.

RESULTADO (02/10/2026) -- NENHUM DÁ GANHO. C1 E C2 REPROVAM EM A; C3 É NEUTRO.

    braço        det  início banda  FP/mês  Δcarga [IC 98,3%]       ΔFP     carga cai em
    referência   6,5   6,0   4,5   0,861   --
    C1 autoesc.  5,5   4,0   3,0   0,689  -34,3 [-98,1; +14,8]   -0,118   7/8   A, B2
    C2 teto      5,5   4,0   3,5   0,646   -9,9 [-48,1; +21,1]   -0,194   5/8   A, B1, B2
    C3 sem ar    6,5   6,0   4,5   0,775   -0,4  [-9,1;  +6,5]   -0,011   3/8   "seguro"
    sem veredito: C1 só p  6,0 / 4,5 / 3,0, Δcarga -31,8;  C1 só t  6,0 / 5,0 / 4,0, -7,5

  · C1: o mecanismo se corrige como previsto -- k do p fica em 7-10 em todos os
    bundles (era 1 em 68%) e o p cru acima de 100x o limiar cai de 0,37% para 0,01%
    do tempo vigiado. Mas a detecção cai em 5 das 8 composições, início 6 -> 4, banda
    4,5 -> 3, e o IC da carga cruza o zero. A composição publicada (dia 1) PIORA:
    49 -> 117 h/mês. O efeito é quase todo do p (só-p ~ C1). Expectativa (risco alto
    em A) confirmada.
  · C2: perde detecção em 6 das 8 e a carga cai em só 5. Expectativa errada QUANTO
    AO EVENTO: o 26/02/2026 fica intacto (8/8); quem cai é o 04/11/2025 (8 -> 5), o
    27/02/2025 (8 -> 6), o 17/03 e o 11/04 (a posteriori, `por evento` abaixo).
  · C3: detecção e banda idênticas nas 8 composições; carga parada (-0,4,
    cai em só 3/8); FP na mediana 0,861 -> 0,775, mas pareado -0,011. Como previsto,
    não passa em B1 por efeito pequeno. A decisão de tirar uma tag de outra
    unidade do canal fica com a engenharia, como higiene -- os números não pedem nem
    proíbem.
  · POR EVENTO (a posteriori, sem mudar veredito; em quantas das 8 composições há
    detecção / nascimento na janela):
        evento       ref     C1     C2
        2025-02-27   8/8    7/7    6/6
        2025-03-17   7/3    7/0    6/3
        2025-04-07   7/6    8/7    7/6
        2025-04-11   6/5    4/4    4/4
        2025-04-29   1/0    1/1    1/0
        2025-11-04   8/8    8/8    5/5
        2025-12-09   6/5    6/4    7/3
        2026-02-26   8/8    4/2    8/8
    O C1 perde o 26/02/2026 (o evento sustentado pela memória longa do t, ver
    `memoria_nas_comparacoes.py`); o C2 perde o 04/11/2025, o que nasce na abertura
    da máscara -- coerente com o teto cortar também os resíduos gigantes da máquina
    parada que a EWMA leva para dentro do blackout (`ewma_vigiado.py` perdia o mesmo
    evento). Não verificado instante a instante.
  · Leitura: a escala degenerada é defeito real e o C1 o corrige, mas parte das
    detecções da referência vive do mesmo mecanismo que gera a carga -- de novo a
    fronteira. Os vereditos ficam.

Uso:  PYTHONPATH=. python escala_teto_ar.py confere   # equivalências
      PYTHONPATH=. python escala_teto_ar.py           # o teste
"""
from __future__ import annotations
import io, contextlib, sys
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler

with contextlib.redirect_stdout(io.StringIO()):
    import regua_fp as R
    import aprovacao_operador as AO
    import bootstrap_regua as BR
    import troca_e_pi0319 as TP

C, DF, IX, STABLE, FIT, DB = AO.C, AO.DF, AO.IX, AO.STABLE, AO.FIT, R.DB
PHI = 0.10                                   # o mesmo do ScorerMax
TETO = 20.0                                  # o mesmo de (E/thr).clip(upper=20) no CUSUM
AR = "PI_5134001"
T_TAGS, P_TAGS = list(C.TEMPERATURE_TAGS), list(C.PRESSURE_TAGS)
P_SEM_AR = [c for c in P_TAGS if c != AR]
NIVEL = 1 - 0.05 / 3


# ══════════════════════════════════════════════════ o ScorerMax, com a escala trocável
def ajusta(F: pd.DataFrame, Escala):
    """A aritmética do ScorerMax (fit), com `Escala` no lugar do RobustScaler."""
    sc = Escala().fit(F)
    Xs = sc.transform(F)
    pca = PCA(n_components=0.95, svd_solver="full").fit(Xs)
    e = (Xs - pca.inverse_transform(pca.transform(Xs))) ** 2
    p = np.nanpercentile(e, 99, axis=0)
    sens = np.maximum(p, PHI * np.nanmedian(p))
    p99 = float(np.nanpercentile(np.max(e / sens, axis=1), 99))
    return sc, pca, sens, p99


def pontua(m, X: pd.DataFrame) -> np.ndarray:
    sc, pca, sens, p99 = m
    ok = X.notna().all(axis=1).to_numpy()
    out = np.full(len(X), np.nan)
    if ok.any():
        Xs = sc.transform(X[ok])
        e = (Xs - pca.inverse_transform(pca.transform(Xs))) ** 2
        out[ok] = np.max(e / sens, axis=1) / p99
    return out


def familias(dia: int, esc_t, esc_p, cols_p) -> tuple[np.ndarray, np.ndarray, list]:
    """t e p da composição `dia` com a escala e as tags pedidas. Mesmas linhas."""
    t, p = R.sinais(dia)[0].copy(), R.sinais(dia)[1].copy()
    ks = []
    for c0, c1 in TP.bundles(dia):
        F = DF.loc[STABLE & (IX < c0), C.SENSOR_TAGS].dropna().tail(FIT)
        s = (IX >= c0) & (IX < c1)
        if not s.any():
            continue
        mt, mp = ajusta(F[T_TAGS], esc_t), ajusta(F[cols_p], esc_p)
        t[s] = pontua(mt, DF.loc[s, T_TAGS])
        p[s] = pontua(mp, DF.loc[s, cols_p])
        ks.append((int(mt[1].n_components_), int(mp[1].n_components_)))
    return t, p, ks


def sinais(dia: int, braco: str):
    """(t, p, ms, ds) do braço, cacheado. Braços: C1, C1p, C1t, C3, C1+C3."""
    esc = {"C1": (StandardScaler, StandardScaler, P_TAGS),
           "C1p": (RobustScaler, StandardScaler, P_TAGS),
           "C1t": (StandardScaler, RobustScaler, P_TAGS),
           "C3": (RobustScaler, RobustScaler, P_SEM_AR),
           "C1+C3": (StandardScaler, StandardScaler, P_SEM_AR)}[braco]
    f = R.CACHE / f"escala_teto_ar_{braco.replace('+', '_')}_d{dia:02d}.npz"
    _, _, ms, ds = R.sinais(dia)
    if f.exists():
        z = np.load(f)
        return z["t"], z["p"], ms, ds
    t, p, ks = familias(dia, *esc)
    np.savez(f, t=t, p=p, k=np.asarray(ks))
    return t, p, ms, ds


# ══════════════════════════════════════════════════ o detector do C2
def detector_teto(t, p, ms, ds, teto: float = TETO) -> dict:
    """`regua_fp.detector` com o score cru limitado a teto x limiar antes da EWMA.

    Uma EWMA por nível, porque o limiar é por nível. A força F segue sobre a EWMA
    sem teto (ver o pré-registro)."""
    idx, m_d, rst = R.idx, R.mask, DB.reset
    z = np.load("piso_fisico_cache.npz")
    cru = pd.DataFrame({"t": t, "p": p, "sp": np.abs((z["b_all"] - ms) / ds),
                        "vb": DB.cru_pub["vb"].to_numpy()}, index=idx)
    ew = lambda s, c: s.ewm(halflife=pd.Timedelta(DB.HL[c]), times=idx).mean()
    EW = {c: ew(cru[c], c) for c in R.SIN}

    def canal(c, k):
        thr = DB.BASE[c] * k
        E = ew(cru[c].clip(upper=teto * thr), c).where(m_d)
        deg = ((E > thr).astype(int).rolling(DB.SUSTAIN, min_periods=DB.SUSTAIN).sum()
               >= DB.SUSTAIN)
        x = ((E / thr).clip(upper=20) - DB.KAPPA).fillna(0.0).to_numpy()
        cu = pd.Series(R.cusum_var(x, rst, DB.H_CUSUM) > DB.H_CUSUM, index=idx)
        return (deg | cu) & m_d

    A = {c: canal(c, DB.K_LO[c]) for c in R.SIN}
    B = {c: canal(c, DB.KH[c]) for c in R.SIN}
    vA = pd.Series(sum(A[c].astype(int) for c in R.SIN) >= DB.VOTO_LO, index=idx) & m_d
    vB = (pd.Series(sum(B[c].astype(int) for c in R.SIN) >= DB.VOTO_HI, index=idx)
          & m_d & (B["sp"] | B["vb"]))
    F = pd.concat([EW[c].where(m_d) / (DB.BASE[c] * DB.KH[c]) for c in R.SIN],
                  axis=1).max(axis=1)
    return dict(fin=R._pos(vA | vB, F), A=A, B=B)


# ══════════════════════════════════════════════════ equivalências
def confere() -> None:
    for d in R.DIAS:
        t, p, _ = familias(d, RobustScaler, RobustScaler, P_TAGS)
        rt, rp = R.sinais(d)[:2]
        rel = lambda a, b: float(np.nanmax(np.abs(a - b) / np.maximum(np.abs(b), 1e-9)))
        ok = np.isfinite(rt)
        assert (np.isfinite(t) == ok).all() and (np.isfinite(p) == np.isfinite(rp)).all()
        same = bool((detector_teto(*R.sinais(d), teto=np.inf)["fin"]
                     == R.detector(*R.sinais(d))["fin"]).all())
        print(f"  composição {d:2d}: laço C1 com RobustScaler -> t {rel(t, rt):.1e}, p {rel(p, rp):.1e} "
              f"(máx. dif. relativa) | C2 com teto infinito = referência: {same}", flush=True)
        assert same


# ══════════════════════════════════════════════════ o teste
def mecanismo(dia: int, braco: str) -> str:
    z = np.load(R.CACHE / f"escala_teto_ar_{braco}_d{dia:02d}.npz")
    k = z["k"]
    p = z["p"][R.mask.to_numpy() & np.isfinite(z["p"])]
    lim = DB.BASE["p"] * DB.K_LO["p"]
    return (f"k t {np.median(k[:, 0]):.0f} ({k[:, 0].min()}-{k[:, 0].max()}), "
            f"k p {np.median(k[:, 1]):.0f} ({k[:, 1].min()}-{k[:, 1].max()}), "
            f"p cru > 100x {100 * float(np.mean(p > 100 * lim)):.2f}%")


def main():
    pd.set_option("display.width", 220)
    # as equivalências rodam à parte (`confere`), antes do teste; ver o RESULTADO
    ref = BR.braco("referência", lambda d: R.detector(*R.sinais(d))["fin"], R.DIAS)
    print(f"\nreferência     det {ref.tabela.det.median()} início {ref.tabela.inicio.median()} "
          f"banda {ref.tabela.banda.median()} FP {ref.tabela.fp_mes.median():.3f} "
          f"carga {ref.tabela.carga_mes.mean():.1f}", flush=True)
    gera = {
        "C1": lambda d: R.detector(*sinais(d, "C1"))["fin"],
        "C2": lambda d: detector_teto(*R.sinais(d))["fin"],
        "C3": lambda d: R.detector(*sinais(d, "C3"))["fin"],
    }
    bracos, decs = {}, {}
    for nome, g in gera.items():
        bracos[nome] = BR.braco(nome, g, R.DIAS)
        decs[nome] = BR.decide(ref, bracos[nome], pareado=True, nivel=NIVEL)
        TP.linha(nome, decs[nome], ref, bracos[nome], NIVEL)

    print("\nSEM VEREDITO -- decomposição do C1 e mecanismo sob a autoescala:", flush=True)
    for nome in ("C1p", "C1t"):
        b = BR.braco(nome, lambda d, n=nome: R.detector(*sinais(d, n))["fin"], R.DIAS)
        TP.linha(nome, BR.decide(ref, b, pareado=True, nivel=NIVEL), ref, b, NIVEL)
    for d in R.DIAS:
        print(f"  C1 composição {d:2d}: {mecanismo(d, 'C1')}", flush=True)

    venc = [n for n, x in decs.items() if x["veredito"] == "GANHO"]
    print(f"\nvencedores: {venc or 'nenhum'}", flush=True)
    if len(venc) >= 2:
        def uniao(d):
            if "C1" in venc:
                sig = sinais(d, "C1+C3" if "C3" in venc else "C1")
            else:
                sig = sinais(d, "C3") if "C3" in venc else R.sinais(d)
            return (detector_teto(*sig) if "C2" in venc else R.detector(*sig))["fin"]
        u = BR.braco("+".join(venc), uniao, R.DIAS)
        TP.linha("+".join(venc), BR.decide(ref, u, pareado=True, nivel=0.95), ref, u, 0.95)
    pd.DataFrame([dict(braco=n, **x) for n, x in decs.items()]).to_csv(
        R.CACHE / "escala_teto_ar.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "confere":
        confere()
    else:
        main()
