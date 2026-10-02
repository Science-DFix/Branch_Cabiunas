#!/usr/bin/env python3
"""D30: o detector rodando a cada 30 s, sem a mediana de 2 min.

PRÉ-REGISTRADO EM 02/10/2026, ANTES DE RODAR (commit deste arquivo sem resultado).

A PERGUNTA, do usuário: e se o modelo trabalhasse a cada 30 s? O export do PI já vem
em 30 s; a pesquisa inteira reduziu cada 4 leituras a uma, pela mediana
(`build_cache.py`). `paridade_entrada.py` mostrou que pegar UMA leitura por janela de
2 min, em vez da mediana, leva o dia 1 de 0,344 para 0,603 FP/mês. Aqui a pergunta é
a outra: usar TODAS as leituras, sem reduzir.

O DESENHO. O mesmo detector v2, com o dado de 30 s e toda constante contada em
amostras convertida para o mesmo TEMPO (fator 4). Nenhum reajuste de limiar.
  dado      `sensores_full_2024_2026_30s.csv` (2.450.520 linhas, 30 s regulares, sem
            buraco): texto -> NaN e o corte de faixa física de `build_cache.py` (o
            corte não é agregação e fica); RUNNING_A instantâneo (0/1). SEM mediana.
  máscara   RUNNING_A > 0,5 e T5 > 300; blackout de 6 h = 720 amostras; régua desde
            2025-01-01.
  baseline  as 80.000 amostras estáveis e completas (36 sensores) antes de cada corte
            = as 20.000 de 2 min x 4, o mesmo tempo (~667 h). Os mesmos 8 dias de
            retreino, as mesmas datas de corte.
  t, p      a aritmética do ScorerMax (RobustScaler, PCA 0,95, piso PHI 0,10,
            recon_p99 do próprio baseline), ajustada no baseline de 30 s.
  sp        TI_0305 menos a mediana dos três irmãos; mediana e MAD do baseline de 30 s.
  vb        `piso_fisico.ref_rolante_bruta` com as horas em amostras de 30 s: base de
            400 h, guarda de 24 h, passo de 6 h; exclusão de -7 d a +2 d em torno dos
            trips (`falhas.csv`, a mesma lista da referência).
  detector  EWMA por tempo (meia-vida igual). Degrau SUSTAIN 15 -> 60 amostras.
            CUSUM: o incremento por amostra é a mesma função da EWMA e há 4x mais
            amostras por hora, então H 80 -> 320 e a carga de reinício 0,25 -> 0,25^(1/4)
            por amostra (o mesmo decaimento por minuto). KAPPA, teto de 20x, limiares,
            votos, portão, ESC_ABS, refratário de 72 h, durações de 120 e 60 min
            iguais; idade da escalada 96 h e emenda de 2 h convertidas em amostras; a
            duração de um episódio soma 0,5 min (uma amostra) em vez de 2.
  régua     o alarme de 30 s projetado na grade de 2 min (o ponto de 2 min está em
            alarme se qualquer das suas 4 amostras está), `regua_fp.mede` de sempre,
            com o denominador de operação da referência.
`confere()` prova, antes, que o MESMO código, alimentado com a grade de 2 min e o
fator 1, reproduz a referência: t, p, sp e vb contra os sinais da régua, e o alarme
bit a bit nas 8 composições. Assim só o dado e a conversão mudam no teste.
DECISÃO: `bootstrap_regua.decide`, pareado, IC 95% (hipótese única). GANHO = detecção
estrita (nenhuma composição perde detecção; medianas de início e banda não caem) +
carga cai em >= 7/8 + IC abaixo de zero + FP não sobe. Um GANHO é candidato a
validação em dado novo, não decisão de produção.
IMPRESSO SEMPRE, SEM VEREDITO: a fração do tempo vigiado com o p cru acima de 100x o
limiar, a 30 s contra 2 min.

EXPECTATIVA REGISTRADA: sem ganho. O mais provável é a carga SUBIR, na direção da
"amostra" de `paridade_entrada.py`: o erro é elevado ao quadrado leitura a leitura,
antes de qualquer suavização, e a 30 s toda falha de leitura entra; a EWMA as soma
em vez de descartá-las, como a mediana fazia. Antecedência: no máximo minutos de
diferença, porque nada no detector é mais rápido que 30 min. A detecção pode mudar em
qualquer direção, porque o baseline e o recon_p99 mudam junto.
RESSALVA: é outro detector nos mesmos 8 eventos de sempre, e a resposta vale para
esta régua, não para qualquer detector a 30 s.

EQUIVALÊNCIAS (02/10/2026, antes do teste): o mesmo código com a grade de 2 min e o
fator 1 reproduz a referência nas 8 composições -- vb idêntico (diferença 0), t até
1,1e-4, p até 1,2e-3 e sp até 7e-8 relativo (ponto flutuante), alarme bit a bit.

RESULTADO (02/10/2026) -- REPROVADO (A, B1, B2, C). A 30 s NÃO HÁ GANHO.

    braço        det  início banda  FP/mês  Δcarga [IC 95%]         ΔFP     carga cai em
    referência   6,5   6,0   4,5   0,861   --
    D30 (30 s)   7,0   4,0   2,5   0,947  -10,2 [-23,6; +1,7]    +0,183   6/8

    por composição (det / banda / carga h/mês, referência -> D30):
      d1  8->7 / 5->4 /  49->76     d11 5->6 / 3->2 / 163->154
      d4  6->4 / 2->2 / 123->118    d15 7->7 / 4->2 / 185->168
      d8  5->4 / 3->2 / 107->113    d18 6->7 / 5->3 / 180->177
                                    d22 7->7 / 5->6 / 126->68    d25 7->7 / 5->4 / 135->111

  · A detecção "de pé" sobe na mediana (6,5 -> 7,0), mas troca: 3 composições perdem,
    2 ganham. O que a operação lê -- o NASCIMENTO na janela -- piora muito: início
    6 -> 4, banda 4,5 -> 2,5. O FP sobe (+0,183): o total de episódios quase não muda
    (198 contra 192 nas 8 composições), mas mais deles caem longe dos trips (91 FP
    contra 74; 59 TP contra 65). A carga cai pouco, com IC cruzando o zero. A
    composição publicada (dia 1) piora: 49 -> 76 h/mês.
  · Expectativa: "sem ganho" confirmada. A direção da carga eu errei na média: previ
    que subiria como na "amostra" de `paridade_entrada.py`, e ela cai um pouco (sobe
    no dia 1). A diferença é que aqui o baseline também é de 30 s: o recon_p99
    absorve o ruído extra (o p cru acima de 100x cai de 0,40% para 0,33% do tempo
    vigiado). O que estraga a "amostra" é treinar num formato e servir noutro, não a
    resolução em si.
  · A escala degenerada do PCA de pressão (`escala_degenerada.py`) não depende da
    resolução: o p continua bimodal a 30 s.

Uso:  PYTHONPATH=. python trinta_segundos.py confere   # equivalências a 2 min
      PYTHONPATH=. python trinta_segundos.py           # o teste
"""
from __future__ import annotations
import io, contextlib, sys
from pathlib import Path
import numpy as np, pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    import regua_fp as R
    import aprovacao_operador as AO
    import bootstrap_regua as BR
    import escala_teto_ar as ET
    import troca_e_pi0319 as TP
    import piso_fisico as PF
    import paridade_entrada as PE
    from publica_clearml import CARGA
    import avalia as AV

AQUI = Path(__file__).resolve().parent
PKG = AQUI.parents[1] / "estrutura_pra_prod" / "Cabiunas" / "scripts"
sys.path.insert(0, str(PKG))
import cabiunas_inference as CI  # noqa: E402  (o _cusum vetorizado com carga explícita)

C, DB = AO.C, R.DB
SRC = AQUI.parents[3] / "dados" / "sensores_full_2024_2026_30s.csv"
T0 = pd.Timestamp("2025-01-01", tz="UTC")
ALVO_SP, IRMAOS = "954005_624_TI_0305", ["954005_624_TI_0301", "954005_624_TI_0303",
                                          "954005_624_TI_0307"]
FALHAS = list(pd.read_csv(AQUI / "falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC"))


# ══════════════════════════════════════════════════ o dado
def grade30() -> pd.DataFrame:
    """O export de 30 s com texto -> NaN e a faixa física. Sem mediana. Cacheado."""
    f = R.CACHE / "grade30s.parquet"
    if f.exists():
        return pd.read_parquet(f)
    cols = C.SENSOR_TAGS + ["RUNNING_A"]
    partes = []
    for ch in pd.read_csv(SRC, chunksize=250_000, usecols=["data_datetime"] + cols, low_memory=False):
        ch["data_datetime"] = pd.to_datetime(ch["data_datetime"], utc=True, errors="coerce")
        ch = ch.dropna(subset=["data_datetime"]).set_index("data_datetime")
        d = {}
        for c in cols:
            v = pd.to_numeric(ch[c], errors="coerce")
            if c != "RUNNING_A":
                lo, hi = PE.faixa(c)
                v = v.where((v >= lo) & (v <= hi))
            d[c] = v.astype("float32")
        partes.append(pd.DataFrame(d, index=ch.index))
    g = pd.concat(partes)
    g = g[~g.index.duplicated(keep="first")].sort_index()
    g = g.reindex(pd.date_range(g.index[0], g.index[-1], freq="30s", tz="UTC"))
    g.to_parquet(f)
    return g


def grade2() -> pd.DataFrame:
    g = pd.read_parquet(AQUI / "grade2min.parquet")
    return g[C.SENSOR_TAGS + ["RUNNING_A"]]


class Base:
    """Grade, máscara e o que o detector precisa dela, para um passo qualquer."""

    def __init__(self, g: pd.DataFrame, por_h: int):
        self.g, self.por_h, self.fator = g, por_h, por_h / 30
        self.idx = g.index
        op = (g["RUNNING_A"] > 0.5).fillna(False)
        self.estavel = op & (g["T5_AVG_A"] > 300)
        self.part = op & ~op.shift(fill_value=False)
        blk = self.part.rolling(int(6 * por_h), min_periods=1).max().astype(bool)
        self.sel = pd.Series(self.idx >= T0, index=self.idx)
        self.mask = self.estavel & ~blk & self.sel
        self.reset = ((~self.mask) | self.part).to_numpy()
        ok = self.estavel.to_numpy() & g[C.SENSOR_TAGS].notna().all(axis=1).to_numpy()
        self.P = np.flatnonzero(ok)                 # linhas estáveis e completas
        self.fit = int(AO.FIT * self.fator)
        # em float32, como `DET._spread_mancal` sobre a grade da régua
        self.b32 = (g[ALVO_SP] - g[IRMAOS].median(axis=1)).to_numpy()
        self.b = self.b32.astype("float64")
        self._vb = None

    def linhas_fit(self, c0):
        k = int(np.searchsorted(self.idx[self.P], c0))
        return self.P[max(0, k - self.fit):k]

    def bundles(self, dia: int):
        cs = [c for c in AO.cortes(dia) if len(self.linhas_fit(c)) >= self.fit // 4]
        return [(c, cs[k + 1] if k + 1 < len(cs) else self.idx[-1] + pd.Timedelta(minutes=60 / self.por_h))
                for k, c in enumerate(cs)]

    def vb(self) -> np.ndarray:
        if self._vb is None:
            PF.POR_H = self.por_h                   # as horas da referência em amostras
            try:
                V = self.g[C.VIBRATION_TAGS].where(self.estavel)
                hot, Xh, MED, S, _ = PF.ref_rolante_bruta(V, self.estavel, FALHAS)
            finally:
                PF.POR_H = 30
            with np.errstate(invalid="ignore", divide="ignore"):
                Z = np.abs((Xh - MED) / S)
            v = np.full(len(self.idx), np.nan)
            v[hot] = np.nanmax(np.where(np.isfinite(Z), Z, -np.inf), axis=1)
            v[~np.isfinite(v)] = np.nan
            self._vb = v
        return self._vb

    def sinais(self, dia: int) -> pd.DataFrame:
        n = len(self.idx)
        t, p = np.full(n, np.nan), np.full(n, np.nan)
        ms, ds = np.full(n, np.nan), np.full(n, np.nan)
        T, Pt = list(C.TEMPERATURE_TAGS), list(C.PRESSURE_TAGS)
        for c0, c1 in self.bundles(dia):
            F = self.g.iloc[self.linhas_fit(c0)]
            s = (self.idx >= c0) & (self.idx < c1)
            if not s.any():
                continue
            w = self.g.loc[s]
            t[s] = ET.pontua(ET.ajusta(F[T], ET.RobustScaler), w[T])
            p[s] = ET.pontua(ET.ajusta(F[Pt], ET.RobustScaler), w[Pt])
            b = pd.Series(self.b32[self.linhas_fit(c0)])
            ms[s] = float(b.median())
            ds[s] = float((b - b.median()).abs().median() * 1.4826)
        return pd.DataFrame({"t": t, "p": p, "sp": np.abs((self.b - ms) / ds), "vb": self.vb()},
                            index=self.idx)


# ══════════════════════════════════════════════════ o detector, com o passo explícito
def detector(B: Base, cru: pd.DataFrame) -> pd.Series:
    idx, m_d, rst, f = B.idx, B.mask, B.reset, B.fator
    sustain, H, carga = int(round(DB.SUSTAIN * f)), DB.H_CUSUM * f, CARGA ** (1 / f)
    EW = {c: cru[c].ewm(halflife=pd.Timedelta(h), times=idx).mean() for c, h in DB.HL.items()}

    def canal(c, k):
        thr = DB.BASE[c] * k
        E = EW[c].where(m_d)
        deg = ((E > thr).astype(int).rolling(sustain, min_periods=sustain).sum() >= sustain)
        x = ((E / thr).clip(upper=20) - DB.KAPPA).fillna(0.0).to_numpy()
        return (deg | pd.Series(CI._cusum(x, rst, carga) > H, index=idx)) & m_d

    A = {c: canal(c, DB.K_LO[c]) for c in R.SIN}
    Bn = {c: canal(c, DB.KH[c]) for c in R.SIN}
    vA = pd.Series(sum(A[c].astype(int) for c in R.SIN) >= DB.VOTO_LO, index=idx) & m_d
    vB = (pd.Series(sum(Bn[c].astype(int) for c in R.SIN) >= DB.VOTO_HI, index=idx)
          & m_d & (Bn["sp"] | Bn["vb"]))
    F = pd.concat([EW[c].where(m_d) / (DB.BASE[c] * DB.KH[c]) for c in R.SIN], axis=1).max(axis=1)
    return pos(B, vA | vB, F)


def pos(B: Base, voto: pd.Series, F: pd.Series) -> pd.Series:
    """`regua_fp._pos`, com as contagens em amostras do passo de `B`."""
    idx = B.idx
    f = F.fillna(0.0).to_numpy().tolist(); v = voto.to_numpy().copy(); vl = v.tolist()
    n_id = int(DB.ESC_IDADE * B.por_h)
    n_gap = int(pd.Timedelta(hours=AV.GAP_EP_H) / pd.Timedelta(minutes=60 / B.por_h)) + 1
    dentro, ini, ja = False, 0, False
    for i in range(len(vl)):
        if not vl[i]:
            dentro, ja = False, False; continue
        ac = f[i] > DB.ESC_ABS
        if not dentro:
            dentro, ini, ja = True, i, ac; continue
        if ac and not ja and (i - ini) >= n_id:
            v[max(ini + 1, i - n_gap):i] = False; ini = i
        ja = ac
    voto = pd.Series(v, index=idx)
    passo_min = 60 / B.por_h
    al = pd.Series(False, index=idx); bloq = ini_b = None; fortes = []
    for a, b in AV.episodios(voto):
        forte = float(F.loc[a:b].max()) > DB.ESC_ABS
        velho = ini_b is not None and (a - ini_b).total_seconds() / 3600 >= DB.ESC_IDADE
        if bloq is not None and a <= bloq and not (forte and velho):
            continue
        al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=DB.REFRAT_V2); ini_b = a
        if forte:
            fortes.append((a, b))
    fin = pd.Series(False, index=idx)
    for a, b in AV.episodios(al):
        d = (b - a).total_seconds() / 60 + passo_min
        if (any(x >= a and y <= b for x, y in fortes) and d >= DB.ESC_DUR) or d >= DB.DUR_MIN:
            fin.loc[a:b] = True
    return fin & B.sel


def na_grade_2min(fin: pd.Series) -> pd.Series:
    """O ponto de 2 min está em alarme se qualquer das suas amostras está."""
    g = fin.groupby(fin.index.floor("2min")).max()
    return g.reindex(R.idx, fill_value=False).astype(bool) & R.sel


# ══════════════════════════════════════════════════ equivalências
def confere() -> None:
    B2 = Base(grade2(), 30)
    assert (B2.mask.to_numpy() == R.mask.to_numpy()).all(), "máscara"
    vb_ref = DB.cru_pub["vb"].to_numpy()
    ok = np.isfinite(vb_ref)
    print(f"  vb: máx. dif. {np.nanmax(np.abs(B2.vb()[ok] - vb_ref[ok])):.1e}, "
          f"NaN iguais {bool((np.isfinite(B2.vb()) == ok).all())}", flush=True)
    z = np.load("piso_fisico_cache.npz")
    for d in R.DIAS:
        cru = B2.sinais(d)
        rt, rp, ms, ds = R.sinais(d)
        sp_ref = np.abs((z["b_all"] - ms) / ds)
        rel = lambda a, b: float(np.nanmax(np.abs(a - b) / np.maximum(np.abs(b), 1e-9)))
        same = bool((detector(B2, cru) == R.detector(rt, rp, ms, ds)["fin"]).all())
        print(f"  composição {d:2d}: t {rel(cru.t.to_numpy(), rt):.1e}  p {rel(cru.p.to_numpy(), rp):.1e}  "
              f"sp {rel(cru.sp.to_numpy(), sp_ref):.1e}  | alarme = referência: {same}", flush=True)
        assert same


# ══════════════════════════════════════════════════ o teste
def main():
    pd.set_option("display.width", 220)
    B30 = Base(grade30(), 120)
    print(f"grade de 30 s: {len(B30.idx)} instantes, {B30.mask.sum() / 120:.0f} h vigiadas", flush=True)
    ref = BR.braco("referência", lambda d: R.detector(*R.sinais(d))["fin"], R.DIAS)
    print(f"referência     det {ref.tabela.det.median()} início {ref.tabela.inicio.median()} "
          f"banda {ref.tabela.banda.median()} FP {ref.tabela.fp_mes.median():.3f} "
          f"carga {ref.tabela.carga_mes.mean():.1f}", flush=True)
    lim = DB.BASE["p"] * DB.K_LO["p"]
    mec = []

    def gera(d):
        cru = B30.sinais(d)
        m = B30.mask.to_numpy() & np.isfinite(cru.p.to_numpy())
        p2 = R.sinais(d)[1][R.mask.to_numpy() & np.isfinite(R.sinais(d)[1])]
        mec.append((d, float(np.mean(cru.p.to_numpy()[m] > 100 * lim)), float(np.mean(p2 > 100 * lim))))
        fin = na_grade_2min(detector(B30, cru))
        print(f"  composição {d:2d} pronta", flush=True)
        return fin

    d30 = BR.braco("D30", gera, R.DIAS)
    x = BR.decide(ref, d30, pareado=True, nivel=0.95)
    TP.linha("D30 (30 s)", x, ref, d30, 0.95)
    print("\nSEM VEREDITO -- p cru acima de 100x o limiar, fração do tempo vigiado (30 s | 2 min):")
    for d, a, b in mec:
        print(f"  composição {d:2d}: {100 * a:.2f}% | {100 * b:.2f}%")
    pd.DataFrame([dict(braco="D30", **x)]).to_csv(R.CACHE / "trinta_segundos.csv", index=False)
    d30.tabela.to_csv(R.CACHE / "trinta_segundos_composicoes.csv")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "confere":
        confere()
    else:
        main()
