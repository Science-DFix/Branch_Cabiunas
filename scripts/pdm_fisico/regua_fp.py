#!/usr/bin/env python3
"""A RÉGUA dos experimentos de redução de falso positivo.

POR QUE UMA RÉGUA NOVA. O ponto publicado -- 0,344 FP/mês, 6,6 h/mês -- é UMA
realização. Deslocando o dia do retreino dentro do mês (mesmo tamanho de
baseline, outra composição) o custo vai de 6,6 a 154,6 h/mês
(`drift_composicao.py`), e variando o tamanho em +-10% vai de 6,6 a 68,1
(`drift_vizinhanca_fit.py`). Qualquer ideia avaliada só no dia 1 está medindo a
sorte daquela composição: foi assim que o piso de força pareceu funcionar.

E há o problema do tamanho da amostra. São 6 falsos positivos. Cortar um ou dois
não se distingue de acaso, e é por isso que o custo aqui é medido em HORAS (uma
medida contínua) além da contagem.

O PROTOCOLO. Todo experimento roda nas mesmas 8 composições de baseline (retreino
nos dias 1, 4, 8, 11, 15, 18, 22 e 25) e é comparado PAREADO com o detector de
referência na mesma composição. Pré-registrado, antes de ver qualquer resultado:

  aceito  se  - a carga (h FP + h NEUTRO por mês) cai em >= 7 das 8 composições
                (teste do sinal, p = 0,035 sob o nulo -- otimista, porque as
                composições se sobrepõem e não são independentes), E
              - nenhuma composição perde detecção, e a régua de início e a banda
                acionável não caem na mediana, E
              - o FP/mês não sobe na mediana.

  A terceira condição entrou DEPOIS do primeiro experimento (`cusum_memoria.py`,
  28/09/2026): o teto do CUSUM derrubou as horas e SUBIU a contagem, porque
  episódio mais curto se parte em mais episódios. Horas e contagem podem andar em
  sentidos opostos. Endurecer a regra depois de ver um resultado não fabrica
  acerto -- afrouxar fabricaria --, mas fica registrado que foi a posteriori.

Não é um teste forte. É o mais forte que estes dados permitem, e é muito melhor
que um ponto só.

REPRODUZ O PUBLICADO. Com retreino no dia 1 esta régua devolve banda 5/8, 8/8,
0,344 FP/mês e 6,55 h/mês -- o número publicado. Os sinais `t` e `p` recalculados
diferem do cache publicado em 1e-6 relativo (máximo 2e-4): ponto flutuante, sem
efeito em nenhuma métrica.

Uso:  PYTHONPATH=. python regua_fp.py      # a distribuição do detector atual
"""
from __future__ import annotations
import io, contextlib, os
from pathlib import Path
import numpy as np, pandas as pd

# Algumas dependências imprimem resultados de experimento ao serem importadas.
with contextlib.redirect_stdout(io.StringIO()):
    import avalia as AV
    import drift_baseline as DB
    import pos_processamento as _PP
    import drift_composicao as DC
    from publica_clearml import CARGA

DB.PP = _PP          # estavel e blk, para a entrada do EWMA
DIAS = (1, 4, 8, 11, 15, 18, 22, 25)
CACHE = Path(__file__).resolve().parent / "_cache_regua"
JAN = pd.Timedelta(hours=48)

idx, mask, alvo, sel = DB.idx, DB.mask, DB.alvo, DB.sel
SIN = DB.SIN
PARADAS = DB.paradas


# ══════════════════════════════════════════════════ sinais por composição
def sinais(dia: int) -> tuple[np.ndarray, ...]:
    """t, p, med_sp, mad_sp com retreino no dia `dia` de cada mês. Cacheado."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"sinais_dia{dia:02d}.npz"
    if f.exists():
        z = np.load(f)
        return z["t"], z["p"], z["ms"], z["ds"]
    t, p, ms, ds = DC.walkforward_dia(dia)
    np.savez(f, t=t, p=p, ms=ms, ds=ds)
    return t, p, ms, ds


# ══════════════════════════════════════════════════ CUSUM com memória limitada
def cusum_var(x: np.ndarray, rst: np.ndarray, H: float,
              modo: str = "nunca", par: float = 0.0) -> np.ndarray:
    """O CUSUM do detector, com a memória opcionalmente limitada.

      nunca : o atual -- S = max(0, S + x), sem teto; zera só na máscara
      teto  : S <= par*H. Depois que a evidência some, o canal fica aceso no
              máximo (par-1)*H/|x| amostras, em vez de proporcional ao pico
      zero  : S volta a 0 ao passar de H (rezero ao sinalizar, carta clássica)
      head  : S volta a par*H ao passar de H (headstart)

    `nunca` usa a versão vetorizada do projeto. As outras precisam de laço,
    porque o teto e o rezero quebram a forma fechada; `confere_cusum()` prova que
    o laço com teto infinito é idêntico à versão vetorizada."""
    if modo == "nunca":
        return DB.cusum(x, rst)
    xs, rs = x.tolist(), rst.tolist()
    S = [0.0] * len(xs); a = 0.0; lim = par * H
    for i in range(len(xs)):
        if rs[i]:
            a *= CARGA
        else:
            a = a + xs[i]
            if a < 0.0:
                a = 0.0
            if modo == "teto" and a > lim:
                a = lim
        S[i] = a
        if a > H:
            if modo == "zero":
                a = 0.0
            elif modo == "head":
                a = lim
    return np.asarray(S)


def confere_cusum() -> float:
    """max|dif| entre o laço (teto infinito) e o CUSUM vetorizado do projeto."""
    t, p, ms, ds = sinais(1)
    E = pd.Series(t, index=idx).ewm(halflife=pd.Timedelta("1h"), times=idx).mean().where(mask)
    thr = DB.BASE["t"] * DB.K_LO["t"]
    x = ((E / thr).clip(upper=20) - DB.KAPPA).fillna(0.0).to_numpy()
    return float(np.max(np.abs(cusum_var(x, DB.reset, DB.H_CUSUM, "teto", 1e18)
                               - DB.cusum(x, DB.reset))))


# ══════════════════════════════════════════════════ o detector, por dentro
def detector(t, p, ms, ds, *, desliga: tuple[str, ...] = (),
             cusum: tuple[str, float] = ("nunca", 0.0),
             ewma_vigiado: tuple[str, ...] = (),
             blackout_h: float | None = None,
             exige_transicao: bool = False) -> dict:
    """O mesmo detector v2 de `drift_baseline.roda`, devolvendo o interior.

    `roda` só devolve o alarme final. Os experimentos precisam dos canais por
    nível (A sensível, B específico), dos votos e da força. `desliga` força
    canais a zero -- é assim que se mede se um canal é essencial num episódio.
    `cusum` = (modo, par) de `cusum_var`; o padrão é o detector atual.
    `ewma_vigiado` = canais cujo EWMA só enxerga instantes vigiados. No detector
    atual o EWMA é calculado sobre o sinal cru INTEIRO (parada e blackout
    inclusive) e só depois mascarado -- ver `transiente_diagnostico.py`.
    `blackout_h` = horas após cada partida em que o detector não enxerga (o
    atual é 6). Muda a máscara e o reset do CUSUM DENTRO do detector; o
    denominador das métricas (`mede`) continua o tempo de operação da
    referência, para o FP/mês ser por mês de MÁQUINA, não de detector.
    `exige_transicao` = um trecho de voto só vale se o detector o viu DESLIGADO
    antes, por SUSTAIN amostras, depois de já estar armado. Ver `_transicao`.

    A equivalência com `roda` é conferida em `confere_equivalencia()`: sem isso,
    um experimento poderia estar medindo outro detector."""
    if blackout_h is None:
        m_d, rst = mask, DB.reset
    else:
        n_bl = int(round(blackout_h * 30))
        blk_h = DB.PP.part.rolling(n_bl, min_periods=1).max().astype(bool)
        m_d = DB.PP.estavel & ~blk_h & sel
        rst = ((~m_d) | DB.PP.part).to_numpy()
    z = np.load("piso_fisico_cache.npz")
    spv = np.abs((z["b_all"] - ms) / ds)
    cru = pd.DataFrame({"t": t, "p": p, "sp": spv,
                        "vb": DB.cru_pub["vb"].to_numpy()}, index=idx)
    # o que a máscara enxerga, sem o corte de avaliação (sel): a entrada do EWMA
    # não pode depender de onde começa a régua
    vig = (DB.PP.estavel & ~DB.PP.blk) if ewma_vigiado else None
    EW = {c: (cru[c].where(vig) if c in ewma_vigiado else cru[c])
              .ewm(halflife=pd.Timedelta(h), times=idx).mean()
          for c, h in DB.HL.items()}

    def canal(c, k):
        if c in desliga:
            return pd.Series(False, index=idx)
        thr = DB.BASE[c] * k
        E = EW[c].where(m_d)
        deg = ((E > thr).astype(int).rolling(DB.SUSTAIN, min_periods=DB.SUSTAIN).sum()
               >= DB.SUSTAIN)
        x = ((E / thr).clip(upper=20) - DB.KAPPA).fillna(0.0).to_numpy()
        cu = pd.Series(cusum_var(x, rst, DB.H_CUSUM, *cusum) > DB.H_CUSUM, index=idx)
        return (deg | cu) & m_d

    A = {c: canal(c, DB.K_LO[c]) for c in SIN}
    B = {c: canal(c, DB.KH[c]) for c in SIN}
    vA = pd.Series(sum(A[c].astype(int) for c in SIN) >= DB.VOTO_LO, index=idx) & m_d
    vB = (pd.Series(sum(B[c].astype(int) for c in SIN) >= DB.VOTO_HI, index=idx)
          & m_d & (B["sp"] | B["vb"]))
    voto = vA | vB
    if exige_transicao:
        voto = _transicao(voto, m_d)
    F = pd.concat([EW[c].where(m_d) / (DB.BASE[c] * DB.KH[c]) for c in SIN],
                  axis=1).max(axis=1)
    fin = _pos(voto, F)
    return dict(A=A, B=B, vA=vA, vB=vB, voto=voto, F=F, EW=EW, fin=fin)


def _transicao(voto: pd.Series, m_d: pd.Series) -> pd.Series:
    """Só vale o trecho de voto que o detector VIU nascer.

    Metade dos episódios nasce no instante em que a máscara abre, qualquer que
    seja o blackout (6, 24, 32 ou 48 h -- `regimes.py`): como vb e sp ficam
    acesos a maior parte do tempo normal, o detector, ao começar a olhar, já
    encontra o voto ligado. O horário desse nascimento é ditado pela máscara.

    Regra: dentro de cada trecho vigiado, as primeiras SUSTAIN amostras são de
    armar (o degrau ainda não pode acender). Depois disso, um trecho de voto só
    é aceito se foi precedido por SUSTAIN amostras armadas com o voto DESLIGADO.
    Sem parâmetro novo: reusa o SUSTAIN."""
    v = voto.to_numpy(); m = m_d.to_numpy(); out = np.zeros(len(v), dtype=bool)
    n = DB.SUSTAIN
    ini = np.flatnonzero(m & ~np.concatenate(([False], m[:-1])))
    fim = np.flatnonzero(m & ~np.concatenate((m[1:], [False]))) + 1
    vl = v.tolist()
    for s, e in zip(ini, fim):
        off, vale, ant = 0, False, False
        for j in range(s, e):
            if vl[j]:
                if not ant:                       # nasce um trecho de voto
                    vale = off >= n
                if vale:
                    out[j] = True
                off = 0
            else:
                if j - s >= n:                    # já armado: o desligado conta
                    off += 1
            ant = vl[j]
    return pd.Series(out, index=voto.index)


def _pos(voto: pd.Series, F: pd.Series) -> pd.Series:
    """Escalada por idade, refratário, duração mínima. Idêntico a `roda`."""
    f = F.fillna(0.0).to_numpy(); v = voto.to_numpy().copy()
    n_id = int(DB.ESC_IDADE * 30); dentro, ini, ja = False, 0, False
    for i in range(len(v)):
        if not v[i]:
            dentro, ja = False, False; continue
        ac = f[i] > DB.ESC_ABS
        if not dentro:
            dentro, ini, ja = True, i, ac; continue
        if ac and not ja and (i - ini) >= n_id:
            v[max(ini + 1, i - DB.N_GAP):i] = False; ini = i
        ja = ac
    voto = pd.Series(v, index=idx)
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
        d = (b - a).total_seconds() / 60 + 2
        if (any(x >= a and y <= b for x, y in fortes) and d >= DB.ESC_DUR) or d >= DB.DUR_MIN:
            fin.loc[a:b] = True
    return fin & sel


def confere_equivalencia(dia: int = 1) -> bool:
    t, p, ms, ds = sinais(dia)
    return bool((detector(t, p, ms, ds)["fin"] == DB.roda(t, p, ms, ds)).all())


# ══════════════════════════════════════════════════ as métricas
def mede(fin: pd.Series) -> dict:
    """As três réguas de detecção e as três de custo, como publicadas."""
    eps = AV.episodios(fin)
    d = AV.avalia(fin, alvo, mask)
    mes = d["horas_op"] / 730.0
    ini = sum(1 for t in alvo if any(t - JAN <= a <= t for a, _ in eps))
    ban = sum(1 for t in alvo
              if any(t - JAN <= a <= t - pd.Timedelta(hours=DB.TMIN_BANDA) for a, _ in eps))
    cls = DB.classifica_regra_c(eps, PARADAS)
    h = lambda k: sum((b - a).total_seconds() / 3600 + 2 / 60 for a, b, kk, _ in cls if kk == k)
    n = lambda k: sum(1 for *_, kk, _ in cls if kk == k)
    return dict(det=d["det"], inicio=ini, banda=ban,
                fp_mes=n("FP") / mes, h_fp_mes=h("FP") / mes,
                h_neutro_mes=h("NEUTRO") / mes, carga_mes=(h("FP") + h("NEUTRO")) / mes,
                n_fp=n("FP"), n_neutro=n("NEUTRO"), n_tp=n("TP"), episodios=len(eps),
                cls=cls)


# ══════════════════════════════════════════════════ a régua
def distribuicao(variante=None, dias=DIAS) -> pd.DataFrame:
    """Uma linha por composição. `variante(t, p, ms, ds) -> fin`; None = referência."""
    linhas = []
    for dia in dias:
        t, p, ms, ds = sinais(dia)
        fin = variante(t, p, ms, ds) if variante else detector(t, p, ms, ds)["fin"]
        m = mede(fin); m.pop("cls")
        linhas.append(dict(dia=dia, **m))
    return pd.DataFrame(linhas).set_index("dia")


def compara(variante, nome: str, dias=DIAS, ref: pd.DataFrame | None = None) -> dict:
    """Pareado contra a referência, com a regra de aceite pré-registrada."""
    ref = distribuicao(None, dias) if ref is None else ref
    var = distribuicao(variante, dias)
    dif = var - ref
    melhora = int((dif["carga_mes"] < -1e-9).sum())
    perde_det = int((dif["det"] < 0).sum())
    ok = (melhora >= 7 and perde_det == 0
          and var["inicio"].median() >= ref["inicio"].median()
          and var["banda"].median() >= ref["banda"].median()
          and var["fp_mes"].median() <= ref["fp_mes"].median() + 1e-9)
    return dict(nome=nome, ref=ref, var=var, dif=dif, melhora=melhora,
                perde_det=perde_det, aceito=bool(ok))


def resumo(df: pd.DataFrame) -> str:
    L = []
    for c, fmt in (("det", "{:.0f}/8"), ("inicio", "{:.0f}/8"), ("banda", "{:.0f}/8"),
                   ("fp_mes", "{:.3f}"), ("h_fp_mes", "{:.1f}"),
                   ("h_neutro_mes", "{:.1f}"), ("carga_mes", "{:.1f}")):
        L.append(f"  {c:13s} mediana {fmt.format(df[c].median()):>7s}   "
                 f"faixa {fmt.format(df[c].min()):>7s} a {fmt.format(df[c].max()):>7s}")
    return "\n".join(L)


if __name__ == "__main__":
    print("equivalência com drift_baseline.roda (dia 1):", confere_equivalencia(1))
    ref = distribuicao()
    pd.set_option("display.width", 160)
    print("\nDETECTOR ATUAL (v2, ponto de deploy) nas 8 composições de baseline\n")
    print(ref[["det", "inicio", "banda", "n_fp", "fp_mes", "h_fp_mes",
               "n_neutro", "h_neutro_mes", "carga_mes"]].round(3).to_string())
    print("\n" + resumo(ref))
    ref.to_csv(CACHE / "referencia.csv")
