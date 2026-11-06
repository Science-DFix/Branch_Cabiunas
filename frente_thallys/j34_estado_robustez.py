#!/usr/bin/env python3
"""J34 -- o detector v2 com ESTADO de verdade: partida a frio (J3) e robustez (J4). Uma task.

PRÉ-REGISTRO (06/11/2026, commitado antes de enviar)
Origem: o J2 (05/11) mostrou que 3 d de bruto + buffer de 13.000 linhas estáveis das sondas TV_ + tabela de 45 d da saída
  do `preprocessar` (t, p, sp, vb, mask, reset) reproduzem a janela de 60 d bit a bit, com 409 mil células. Mas o
  buffer do J2 era tirado da grade completa (oráculo). Faltam as perguntas 2, 3, 4 e 6 da engenharia
  (`tc33003a-janela-historica-e-gargalos-de-inferencia.md`) e a lista de cenários do documento: instalação sem estado,
  reinícios, buracos, falhas de sensor, correções, trips novos, e RSS e tempo de recuperação.
MOTOR COM ESTADO (o que a engenharia implementaria), a cada corte c:
  janela de bruto [c − 3 d, c] do que o banco TEM em c; o vb usa o buffer do próprio motor (linhas estáveis < início da
  janela, as últimas 13.000); as linhas novas do `preprocessar` entram na tabela (cortada em 45 d), completando a grade
  de 2 min com linhas vazias (mask falso, reset verdadeiro, como a grade da referência) quando o banco não tem dado;
  o `prever` roda sobre a tabela; as linhas estáveis novas entram no buffer. Sem dado na janela, o corte é pulado.
  O vb é a PRÓPRIA `_z_vibracao` do pacote sobre [buffer + janela] (monkeypatch), não reimplementação.
REFERÊNCIA: ref60 do J2 (60 d de bruto a cada corte), rodada sobre os MESMOS dados de cada cenário. Cortes a cada 6 h
  desde 2025-01-08 07:34, subcortes quando um trip fica conhecido; régua e comparação desde 2025-02-01 (as do J1/J2).

J3, PARTIDA A FRIO. Instalação em T0 ∈ {01/03, 01/05, 01/07, 01/09, 01/11/2025, 01/01/2026}, rodando 100 d. Modos:
  H = 0 (só dado novo, desde T0) e carga inicial de H = 7, 30, 45, 60 d de bruto (estado montado de uma vez:
  `preprocessar` sobre [T0 − H, T0] sem buffer, a tabela recebe as linhas e o buffer as estáveis).
  Comparação, linha a linha, com o ref60 limpo. Mede: instante de CONVERGÊNCIA (a partir dele, sinais e alarme iguais
  até o fim) e, em cada linha, o estado de prontidão pela REGRA PROPOSTA:
    linha "cheia" = havia >= 12.720 amostras estáveis no dado disponível antes dela (base de 400 h + guarda);
    PLENA      = todas as linhas da tabela nos últimos 45 d são cheias e a tabela cobre 45 d;
    INDISPONÍVEL = menos de 3.000 amostras estáveis disponíveis (o vb não opina, VB_FRACAO_BASE_MIN);
    DEGRADADA  = o resto.
  (a contagem só cresce, então PLENA = a linha de 45 d atrás já era cheia.)
  A regra VALE se, em PLENA, a divergência for 0 em todas as 30 instalações. Mede também a divergência em DEGRADADA e o
  tempo de calendário até PLENA.
J4, ROBUSTEZ (série inteira; motor contra ref60 rodado sobre os mesmos dados):
  (buffer guarda 26.000 linhas estáveis: as 13.000 da base + a janela de 3 d + refazer até 17 d para trás)
  O motor parte, no 1º corte, de uma carga de 60 d de bruto (como uma implantação com histórico).
  limpo ...... motor sobre o dado limpo (V1: tem de ser bit a bit com o ref60, como o b45 do J2);
  reinicio ... o mesmo, mas o estado é gravado em parquet e relido a cada corte (reinício em todo corte);
  buracos .... linhas apagadas do banco: 1 h (25/05/25 12h), 6 h (10/07/25 06h), 1 d (01/08/25), 3 d (22/09/25),
               7 d (12/10/25);
  sondas ..... TV_351X/Y vazias 7 d (26/07–02/08/25); 1ª tag de temperatura do PCA vazia 1 d (01/09/25); as 10 TV_
               vazias 2 d (15–17/12/25);
  atraso ..... 6 h de dado que chegam atrasadas D = 6 h (26/05/25 06h), 1 d (05/08/25 06h) e 7 d (25/09/25 06h). O
               ref60 vê o banco de cada instante. Motor P0 ignora a correção; motor P1 refaz a partir do início do
               trecho corrigido (tabela e buffer desde X, janela [X − 3 d, c]). Compara-se depois da chegada (45 d);
  trip7 ...... os trips só ficam conhecidos 7 d depois (ref60 e motor com o mesmo atraso). Motor P0 segue; motor P1
               (trip7_p1) refaz desde o início da exclusão do vb (trip − 7 d) quando o trip fica conhecido.
CUSTO (serial, depois): em 20 instantes (semente 20261105) desde 2025-04-01: montar o estado de 60 d de bruto (= tempo
  de recuperação sem estado salvo); ler o estado do parquet (= reinício com estado salvo); tamanho do estado em disco;
  um passo do motor. RSS de pico do processo (ru_maxrss num subprocesso limpo), em 3 casos: só os imports; um passo do
  motor lendo estado + 3 d de bruto; uma execução do ref60 lendo 60 d de bruto.
VERIFICAÇÃO: V1 motor limpo = ref60 bit a bit (sinais e alarme). Se falhar, o resto é lido só como relato.
TESTE LOCAL DE ENCANAMENTO (06/11, até 2025-04-15, ANTES deste commit; mudou o desenho, registrado aqui): (1) o motor
  partia de buffer vazio em 08/01/2025 e o vb divergia até 04/02 (aquecimento) -> agora parte de carga de 60 d;
  (2) o ru_maxrss saiu 5,3 GB nos três casos (herdado do pai através do exec) -> VmHWM; (3) trip7: o motor difere do
  ref60 no alarme em 29/03–07/04/2025, porque o ref60 recalcula o vb do passado quando o trip fica conhecido e a tabela
  não -> entrou o trip7_p1 e o buffer passou a 26.000; (4) atraso: P1 = ref60 depois da chegada, P0 não (só relato).
EXPECTATIVAS: V1 passa (~85%). reinicio bit a bit (~90%). buracos: motor = ref60 na régua (~75%) e bit a bit (~50%; o
  EWMA através de buraco < 3 d é igual, o de 7 d começa de novo nos dois). sondas: bit a bit (~70%). atraso: P1 = ref60
  depois da chegada em 100% das linhas (~70%); P0 difere em alguma linha (~80%) mas régua igual (~70%). trip7: motor P0
  difere do ref60 no alarme (~85%, visto no teste local) e P1 = ref60 bit a bit (~65%). J3: a regra vale (0 divergência em PLENA) (~65%); com H = 60 PLENA desde T0 (~70%); com H = 0
  PLENA em 60–80 d de calendário (~60%); a convergência real chega antes da PLENA (~80%). Custo: estado em disco <= 5 MB
  (~80%); montar o estado de 60 d <= 2 s (~60%); RSS de um passo do motor <= 60% do RSS do ref60 (~50%).
LIMITES: 8 trips; injeções sintéticas em datas escolhidas em operação; o banco do SIMPred, as consultas de 15 min e a
  concorrência entre equipamentos não estão aqui; o Cica não é o servidor de produção.

Uso:
  python frente_thallys/j34_estado_robustez.py --local --teste    # encanamento local, até 2025-04-15
  python frente_thallys/j34_estado_robustez.py --enqueue          # uma task na fila default
"""
from clearml import Task  # noqa: F401  (antes do argparse: o worker injeta argumentos por patch)
import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from multiprocessing import get_context
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import j1_janela_entrada as j1   # noqa: E402

NOME = "frente_thallys | J34 estado: partida a frio e robustez"
PASSO_DIAS = 0.25
BRUTO_D, TAB_D = 3, 45
N_BUF, N_GUARDA_BUF = 13_000, 26_000       # usa as últimas 13.000; guarda 26.000 (janela + refazer até 17 d)
N_CHEIA, N_MIN = 12_720, 3_000
COLS = ["t", "p", "sp", "vb", "mask", "reset"]
INSTALA = ["2025-03-01", "2025-05-01", "2025-07-01", "2025-09-01", "2025-11-01", "2026-01-01"]
CARGAS = [0, 7, 30, 45, 60]
BURACOS = [("2025-05-25 12:00", "1h"), ("2025-07-10 06:00", "6h"), ("2025-08-01 00:00", "1D"),
           ("2025-09-22 00:00", "3D"), ("2025-10-12 00:00", "7D")]
ATRASOS = [("2025-05-26 06:00", "6h"), ("2025-08-05 06:00", "1D"), ("2025-09-25 06:00", "7D")]
SAIDA = Path(tempfile.gettempdir()) / "j34_saida"
G = j1.G


def T(x):
    import pandas as pd
    return pd.Timestamp(x, tz="UTC")


def D(x):
    import pandas as pd
    return pd.Timedelta(x)


# ───────────────────────────── cenários (o que o banco tem no instante c) ─────────────────────────────
def prepara_cenarios():
    import numpy as np
    df = G["df"]
    b = df.copy()
    for ini, dur in BURACOS:
        b = b[~((b.index >= T(ini)) & (b.index < T(ini) + D(dur)))]
    s = df.copy()
    tv = G["det"]["tags_vibracao"]; temp = G["modelos"][-1]["normalizacao"]["temperatura"]["cols"][0]
    s.loc[T("2025-07-26"):T("2025-08-02"), ["TV_351X_A", "TV_351Y_A"]] = np.nan
    s.loc[T("2025-09-01"):T("2025-09-02"), [temp]] = np.nan
    s.loc[T("2025-12-15"):T("2025-12-17"), tv] = np.nan
    G["dados"] = {"limpo": df, "buracos": b, "sondas": s, "atraso": df, "trip7": df}
    G["temp_tag"] = temp


def vista(cen, c):
    df = G["dados"][cen]
    if cen != "atraso":
        return df
    for ini, d in ATRASOS:
        if c < T(ini) + D(d):
            df = df[~((df.index >= T(ini)) & (df.index < T(ini) + D("6h")))]
    return df


def atraso_trip(cen):
    return D("7D") if cen == "trip7" else D("0s")


def cortes_cen(cen, fim=None):
    """Cortes de 6 h + subcortes nos instantes em que cada trip fica conhecido."""
    cs = j1.cortes(PASSO_DIAS)
    if fim is not None:
        cs = [c for c in cs if c <= fim]
    L = atraso_trip(cen)
    extra = [t + L for t in G["trips"] if cs[0] < t + L < cs[-1]]
    return sorted(set(cs) | set(extra))


def conhecidos(cen, s):
    L = atraso_trip(cen)
    return [t for t in G["trips"] if t + L < s]


# ───────────────────────────── vb com o buffer do motor ─────────────────────────────
def _z_com_buffer(g, estavel, tags, trips, excl):
    import numpy as np
    import pandas as pd
    orig = G["orig_z"]; V = G.get("buf_atual")
    if V is None:
        return orig(g, estavel, tags, trips, excl)
    pos = V.index.searchsorted(g.index[0])
    buf = V.iloc[max(0, pos - N_BUF):pos]
    gb = pd.concat([buf[tags], g[tags]])
    eb = pd.concat([pd.Series(True, index=buf.index), estavel.astype(bool)])
    return np.asarray(orig(gb, eb, tags, trips, excl))[len(buf):]


def _preproc(w, trips, buf):
    G["buf_atual"] = buf
    with contextlib.redirect_stdout(io.StringIO()):
        return G["ci"].preprocessar(G["modelos"], w, trips=trips)


def _prever(proc):
    with contextlib.redirect_stdout(io.StringIO()):
        return G["ci"].prever({"detector": G["det"]}, proc)


# ───────────────────────────── o motor ─────────────────────────────
class Motor:
    def __init__(self):
        import pandas as pd
        tags = G["det"]["tags_vibracao"]
        self.buf = pd.DataFrame(columns=tags, dtype="float32", index=pd.DatetimeIndex([], tz="UTC"))
        self.tab = None
        self.ultimo = None

    def _estaveis(self, w):
        ci = G["ci"]
        g = ci._grade(w); _, est, _, _ = ci._mascara(g)
        return g.loc[est.to_numpy(), G["det"]["tags_vibracao"]]

    def _anexa_buf(self, w):
        import pandas as pd
        e = self._estaveis(w)
        if len(self.buf):
            e = e[e.index > self.buf.index[-1]]
        self.buf = pd.concat([self.buf, e]).iloc[-N_GUARDA_BUF:] if len(self.buf) else e.iloc[-N_GUARDA_BUF:]

    def _anexa_tab(self, novo, c):
        import pandas as pd
        novo = novo[COLS]
        if self.tab is not None and len(self.tab) and len(novo):
            grade = pd.date_range(self.tab.index[-1] + D("2min"), novo.index[-1], freq="2min", tz="UTC")
            novo = novo.reindex(grade)
            novo["mask"] = novo["mask"].fillna(False).astype(bool)
            novo["reset"] = novo["reset"].fillna(True).astype(bool)
        tab = novo if self.tab is None else pd.concat([self.tab, novo])
        self.tab = tab.loc[c - D(f"{TAB_D}D"):]

    def carga(self, hist, trips, c):
        """Estado montado de uma vez a partir de H dias de bruto (sem buffer anterior)."""
        if not len(hist):
            return
        proc = _preproc(hist, trips, None)
        self._anexa_tab(proc, c); self._anexa_buf(hist); self.ultimo = proc.index[-1]

    def passo(self, banco, c, trips, desde=None):
        """Um corte. `desde` = refazer a partir deste instante (correção P1). Devolve as linhas novas com o alarme."""
        ini = c - D(f"{BRUTO_D}D")
        if desde is not None:
            ini = min(ini, desde - D(f"{BRUTO_D}D"))
            self.tab = self.tab.loc[:desde - D("1ns")] if self.tab is not None else None
            self.buf = self.buf.loc[:desde - D("1ns")]
            self.ultimo = self.tab.index[-1] if self.tab is not None and len(self.tab) else None
        w = banco.loc[ini:c]
        if not len(w):
            return None
        proc = _preproc(w, trips, self.buf)
        novo = proc[proc.index > self.ultimo] if self.ultimo is not None else proc
        if not len(novo):
            return None
        self._anexa_tab(novo, c)
        res = _prever(self.tab)
        self._anexa_buf(w); self.ultimo = proc.index[-1]
        lin = self.tab.loc[novo.index[0]:]
        return lin[COLS].join(res[["is_anomaly", "vigiado"]], how="left")

    def salva_le(self, pasta: Path):
        import pandas as pd
        self.tab.to_parquet(pasta / "tab.parquet"); self.buf.to_parquet(pasta / "buf.parquet")
        self.tab = pd.read_parquet(pasta / "tab.parquet"); self.buf = pd.read_parquet(pasta / "buf.parquet")


# ───────────────────────────── corridas ─────────────────────────────
def corre_ref(cen):
    import pandas as pd
    pedacos, anterior, t0 = [], None, time.time()
    for s in cortes_cen(cen):
        banco = vista(cen, s)
        w = banco.loc[s - D("60D"):s]
        proc = _preproc(w, conhecidos(cen, s), None)
        res = _prever(proc)
        novo = proc[proc.index > anterior] if anterior is not None else proc
        pedacos.append(novo[COLS].join(res[["is_anomaly", "vigiado"]], how="left")); anterior = s
    r = pd.concat(pedacos); r = r[~r.index.duplicated(keep="first")]
    r.to_parquet(SAIDA / f"ref_{cen}.parquet")
    print(f"   ref {cen}: {len(r)} linhas, {time.time() - t0:.0f} s", flush=True)


def corre_motor(cen, modo="normal"):
    """modo: normal | reinicio | p1 (correção refeita a partir do trecho)."""
    import pandas as pd
    m, pedacos, t0 = Motor(), [], time.time()
    pasta = Path(tempfile.mkdtemp(prefix=f"j34_{cen}_"))
    pend = []   # (refazer desde, instante em que a notícia chega)
    if modo == "p1" and cen == "atraso":
        pend = [(T(i), T(i) + D(d)) for i, d in ATRASOS]
    if modo == "p1" and cen == "trip7":   # o trip fica conhecido em t + 7 d (estrito); a exclusão do vb começa em t − 7 d
        pend = [(t - D(f"{G['det'].get('vb_exclusao_dias', 7.0)}D"), t + atraso_trip(cen) + D("1ns")) for t in G["trips"]]
    cs = cortes_cen(cen)
    pend = [(x, chega) for x, chega in pend if chega > cs[0]]   # trip já conhecido antes do 1º corte entra pela carga
    m.carga(vista(cen, cs[0]).loc[cs[0] - D("60D"):cs[0]], conhecidos(cen, cs[0]), cs[0])
    for s in cs[1:]:
        desde = None
        for x, chega in list(pend):
            if s >= chega:
                desde = x if desde is None else min(desde, x); pend.remove((x, chega))
        out = m.passo(vista(cen, s), s, conhecidos(cen, s), desde=desde)
        if out is not None:
            pedacos.append(out)
        if modo == "reinicio" and m.tab is not None:
            m.salva_le(pasta)
    r = pd.concat(pedacos); r = r[~r.index.duplicated(keep="first")]   # linha emitida não é revista
    nome = cen if modo == "normal" else f"{cen}_{modo}"
    r.to_parquet(SAIDA / f"motor_{nome}.parquet")
    print(f"   motor {nome}: {len(r)} linhas, {time.time() - t0:.0f} s", flush=True)


def corre_instalacao(arg):
    """J3: instalação em T0 com carga de H dias; o banco só existe desde T0 − H."""
    import pandas as pd
    t0_s, H = arg
    T0 = T(t0_s); fim = T0 + D("100D")
    banco = G["df"].loc[T0 - D(f"{H}D"):] if H else G["df"].loc[T0:]
    m, pedacos = Motor(), []
    if H:
        m.carga(banco.loc[:T0], conhecidos("limpo", T0), T0)
    for s in [c for c in cortes_cen("limpo") if T0 < c <= fim]:
        out = m.passo(banco, s, conhecidos("limpo", s))
        if out is not None:
            pedacos.append(out)
    r = pd.concat(pedacos); r = r[~r.index.duplicated(keep="first")]
    # prontidão: amostras estáveis disponíveis antes de cada linha, e 45 d antes dela (a contagem só cresce, então
    # "todas as linhas dos últimos 45 d cheias" = "a linha de 45 d atrás já era cheia")
    ci = G["ci"]
    _, est, _, _ = ci._mascara(ci._grade(banco.loc[:fim]))
    n = est.astype(int).cumsum().shift(fill_value=0)
    r["n_est"] = n.reindex(r.index, method="ffill").fillna(0).to_numpy()
    r["n_est_45"] = n.reindex(r.index - D(f"{TAB_D}D"), method="ffill").fillna(0).to_numpy()
    r.to_parquet(SAIDA / f"inst_{t0_s}_H{H}.parquet")
    return arg


def tarefa(x):
    kind, a = x
    if kind == "ref":
        corre_ref(a)
    elif kind == "motor":
        corre_motor(*a)
    else:
        corre_instalacao(a)
    return x


# ───────────────────────────── medidas ─────────────────────────────
def diverge(A, R, desde="2025-02-01", ate=None):
    import numpy as np
    A = A.loc[desde:ate]; R = R.reindex(A.index)
    out = {}
    for s in ("t", "p", "sp", "vb"):
        a, b = A[s].to_numpy(float), R[s].to_numpy(float)
        nd = np.isnan(a) != np.isnan(b); ok = ~np.isnan(a) & ~np.isnan(b)
        d = np.zeros_like(a); d[ok] = np.abs(a[ok] - b[ok])
        out[f"{s}_pct"] = round(100 * float(((d > 1e-9) | nd).mean()), 3) if len(a) else 0.0
        out[f"{s}_max"] = float(d.max()) if ok.any() else 0.0
    out["alarme_pct"] = round(100 * float((A["is_anomaly"].astype(bool) != R["is_anomaly"].fillna(False).astype(bool)).mean()), 3) if len(A) else 0.0
    out["n"] = int(len(A))
    return out


def igual(d):
    return all(v == 0 for k, v in d.items() if k.endswith("_pct"))


def regua(x):
    return j1.regua(x["is_anomaly"].astype(bool), x["vigiado"].fillna(False).astype(bool))


def analisa_j4():
    import pandas as pd
    R = {}
    pares = [("limpo", "limpo"), ("limpo_reinicio", "limpo"), ("buracos", "buracos"), ("sondas", "sondas"),
             ("atraso", "atraso"), ("atraso_p1", "atraso"), ("trip7", "trip7"), ("trip7_p1", "trip7")]
    refs = {c: pd.read_parquet(SAIDA / f"ref_{c}.parquet") for c in G["dados"]}
    for nome, cen in pares:
        A = pd.read_parquet(SAIDA / f"motor_{nome}.parquet")
        d = diverge(A, refs[cen]); rr, ra = regua(refs[cen]), regua(A)
        R[nome] = dict(div=d, bit_a_bit=igual(d), regua_motor=ra, regua_ref=rr,
                       regua_igual=(ra["det"], ra["inicio"], ra["banda"], ra["fp"], ra["lead"]) ==
                                   (rr["det"], rr["inicio"], rr["banda"], rr["fp"], rr["lead"]))
        if cen == "atraso":
            R[nome]["depois_da_chegada"] = {f"{i}+{dd}": diverge(A, refs[cen], desde=T(i) + D(dd), ate=T(i) + D(dd) + D("45D"))
                                            for i, dd in ATRASOS}
    R["efeito_no_detector"] = {c: regua(refs[c]) for c in refs}
    return R


def analisa_j3():
    import numpy as np
    import pandas as pd
    ref = pd.read_parquet(SAIDA / "ref_limpo.parquet")
    L = []
    for t0_s in INSTALA:
        for H in CARGAS:
            r = pd.read_parquet(SAIDA / f"inst_{t0_s}_H{H}.parquet")
            Rr = ref.reindex(r.index)
            dif = np.zeros(len(r), bool)
            for s in ("t", "p", "sp", "vb"):
                a, b = r[s].to_numpy(float), Rr[s].to_numpy(float)
                nd = np.isnan(a) != np.isnan(b); ok = ~np.isnan(a) & ~np.isnan(b)
                dd = np.zeros_like(a); dd[ok] = np.abs(a[ok] - b[ok]); dif |= (dd > 1e-9) | nd
            dal = r["is_anomaly"].astype(bool).to_numpy() != Rr["is_anomaly"].fillna(False).astype(bool).to_numpy()
            dif_any = dif | dal
            T0 = T(t0_s)
            plena = r["n_est_45"].to_numpy() >= N_CHEIA
            indisp = r["n_est"].to_numpy() < N_MIN
            degr = ~plena & ~indisp
            conv = None
            if dif_any.any():
                ult = np.flatnonzero(dif_any)[-1]
                conv = r.index[ult + 1] if ult + 1 < len(r) else None
            else:
                conv = r.index[0]
            pl0 = r.index[np.flatnonzero(plena)[0]] if plena.any() else None
            L.append(dict(T0=t0_s, H=H, convergencia_d=None if conv is None else round((conv - T0).total_seconds() / 86400, 2),
                          plena_d=None if pl0 is None else round((pl0 - T0).total_seconds() / 86400, 2),
                          div_em_plena_pct=round(100 * float(dif_any[plena].mean()), 3) if plena.any() else None,
                          div_em_degradada_pct=round(100 * float(dif_any[degr].mean()), 3) if degr.any() else None,
                          alarme_div_degradada_pct=round(100 * float(dal[degr].mean()), 3) if degr.any() else None,
                          horas_indisponivel=round(float(indisp.sum()) * 2 / 60, 1),
                          horas_degradada=round(float(degr.sum()) * 2 / 60, 1)))
    T3 = pd.DataFrame(L); T3.to_csv(SAIDA / "j3_instalacoes.csv", index=False)
    ok = T3["div_em_plena_pct"].dropna()
    return dict(regra_vale=bool((ok == 0).all() and len(ok) == T3["plena_d"].notna().sum()),
                por_carga={str(H): dict(plena_d_med=_med(x.plena_d), convergencia_d_med=_med(x.convergencia_d),
                                        convergencia_d_max=_max(x.convergencia_d), div_plena_max=_max(x.div_em_plena_pct),
                                        div_degradada_med=_med(x.div_em_degradada_pct), alarme_div_degradada_med=_med(x.alarme_div_degradada_pct),
                                        horas_indisponivel_med=_med(x.horas_indisponivel))
                           for H, x in T3.groupby("H")})


def _med(s):
    s = s.dropna(); return None if not len(s) else round(float(s.median()), 3)


def _max(s):
    s = s.dropna(); return None if not len(s) else round(float(s.max()), 3)


# ───────────────────────────── custo e RSS ─────────────────────────────
FILHO = r"""
import sys, time, json
def hwm():   # pico de RSS deste processo (VmHWM); o ru_maxrss herda o do pai através do exec
    return next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmHWM")) / 1024
modo, pasta = sys.argv[1], sys.argv[2]
sys.path[:0] = sys.argv[3].split(":")
import pandas as pd, numpy as np
import j34_estado_robustez as J
r0 = hwm()
if modo != "imports":
    import cabiunas_inference as ci
    J.G.update(ci=ci, modelos=ci.carregar_modelos(sys.argv[4]))
    J.G["det"] = J.G["modelos"][-1]["detector"]; J.G["orig_z"] = ci._z_vibracao; ci._z_vibracao = J._z_com_buffer
    trips = json.loads(open(pasta + "/trips.json").read()); trips = [pd.Timestamp(t) for t in trips]
    t0 = time.perf_counter()
    if modo == "motor":
        m = J.Motor(); m.tab = pd.read_parquet(pasta + "/tab.parquet").iloc[:-1]; m.buf = pd.read_parquet(pasta + "/buf.parquet")
        m.ultimo = m.tab.index[-1]
        w = pd.read_parquet(pasta + "/bruto3.parquet"); m.passo(w, w.index[-1], trips)
    else:
        w = pd.read_parquet(pasta + "/bruto60.parquet"); p = J._preproc(w, trips, None); J._prever(p)
    dt = time.perf_counter() - t0
else:
    dt = 0.0
print(json.dumps(dict(rss_mb=hwm(), rss_imports_mb=r0, s=dt)))
"""


def custo(teste: bool, pk: Path):
    import numpy as np
    import pandas as pd
    df = G["df"]
    rng = np.random.default_rng(j1.SEMENTE)
    cand = df.loc["2025-04-01":].index if not teste else df.index[-200:]
    inst = [pd.Timestamp(x) for x in sorted(rng.choice(cand, size=3 if teste else 20, replace=False))]
    L = []
    filho = SAIDA / "filho.py"; filho.write_text(FILHO)
    caminho = ":".join([str(AQUI), str(pk / "scripts")])
    for fim in inst:
        trips = conhecidos("limpo", fim)
        m = Motor(); t0 = time.perf_counter()
        m.carga(df.loc[fim - D("60D"):fim], trips, fim); t1 = time.perf_counter()
        pasta = Path(tempfile.mkdtemp(prefix="j34_custo_"))
        m.tab.to_parquet(pasta / "tab.parquet"); m.buf.to_parquet(pasta / "buf.parquet")
        t2 = time.perf_counter()
        pd.read_parquet(pasta / "tab.parquet"); pd.read_parquet(pasta / "buf.parquet"); t3 = time.perf_counter()
        disco = sum(f.stat().st_size for f in pasta.glob("*.parquet")) / 1e6
        df.loc[fim - D("3D"):fim].to_parquet(pasta / "bruto3.parquet")
        df.loc[fim - D("60D"):fim].to_parquet(pasta / "bruto60.parquet")
        (pasta / "trips.json").write_text(json.dumps([str(t) for t in trips]))
        rss = {}
        for modo in ("imports", "motor", "ref60"):
            o = subprocess.run([sys.executable, str(filho), modo, str(pasta), caminho, str(pk / "modelos")],
                               capture_output=True, text=True)
            rss[modo] = json.loads(o.stdout.strip().splitlines()[-1]) if o.returncode == 0 else dict(erro=o.stderr[-300:])
        L.append(dict(fim=str(fim), montar_60d_s=t1 - t0, ler_estado_s=t3 - t2, estado_disco_mb=disco,
                      linhas_tab=len(m.tab), linhas_buf=len(m.buf),
                      rss_imports_mb=rss["imports"].get("rss_mb"), rss_motor_mb=rss["motor"].get("rss_mb"),
                      rss_ref60_mb=rss["ref60"].get("rss_mb"), passo_motor_s=rss["motor"].get("s"),
                      ref60_s=rss["ref60"].get("s"), erro=(str({k: v["erro"] for k, v in rss.items() if "erro" in v}) if any("erro" in v for v in rss.values()) else None)))
    C = pd.DataFrame(L); C.to_csv(SAIDA / "custo.csv", index=False)
    return {k: round(float(C[k].median()), 3) for k in C.columns if k not in ("fim", "erro") and C[k].notna().any()} | \
           {"erros": C["erro"].dropna().unique().tolist()[:3]}


# ───────────────────────────── principal ─────────────────────────────
def executa(local: bool, insumos: str | None, teste: bool, procs: int):
    SAIDA.mkdir(parents=True, exist_ok=True)
    t00 = time.time()
    pk, a97, _ = j1.monta(local, insumos)
    j1.prepara(pk, a97, teste)
    ci = G["ci"]; G["orig_z"] = ci._z_vibracao; ci._z_vibracao = _z_com_buffer
    if teste:   # o recorte do teste acaba em 15/04/2025: injeções dentro dele, só para exercitar o encanamento
        BURACOS[:] = [("2025-03-10 00:00", "3D")]; ATRASOS[:] = [("2025-03-05 06:00", "1D")]
    prepara_cenarios()
    res = dict(ambiente=j1.ambiente(), temp_tag=G["temp_tag"]); print("ambiente:", res["ambiente"], flush=True)

    inst = INSTALA if not teste else ["2025-01-20"]
    trabalhos = [("ref", c) for c in G["dados"]]
    trabalhos += [("motor", (c, "normal")) for c in G["dados"]]
    trabalhos += [("motor", ("limpo", "reinicio")), ("motor", ("atraso", "p1")), ("motor", ("trip7", "p1"))]
    trabalhos += [("inst", (t, H)) for t in inst for H in CARGAS]
    print(f"{len(trabalhos)} corridas em paralelo", flush=True)
    with get_context("fork").Pool(min(procs, len(trabalhos))) as pool:
        for x in pool.imap_unordered(tarefa, trabalhos):
            pass
    res["J4"] = analisa_j4()
    for k, v in res["J4"].items():
        if k != "efeito_no_detector":
            print(f"   J4 {k:15s} bit a bit {v['bit_a_bit']} | régua igual {v['regua_igual']} | "
                  f"motor {v['regua_motor']['det']}/8 FP {v['regua_motor']['fp']} ref {v['regua_ref']['det']}/8 FP {v['regua_ref']['fp']} | "
                  + " ".join(f"{s} {v['div'][s + '_pct']}%" for s in ("t", "p", "sp", "vb", "alarme")), flush=True)
    if teste:
        INSTALA[:] = inst
    res["J3"] = analisa_j3(); print("J3:", json.dumps(res["J3"], indent=1), flush=True)
    print("CUSTO (serial)", flush=True)
    res["custo"] = custo(teste, pk); print(json.dumps(res["custo"], indent=1), flush=True)
    res["duracao_min"] = round((time.time() - t00) / 60, 1)
    (SAIDA / "j34_resultado.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    from clearml import Task as Tk
    tk = Tk.current_task()
    if tk is not None:
        for f in sorted(SAIDA.glob("*")):
            if f.suffix != ".py":
                tk.upload_artifact(f.stem, artifact_object=str(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enqueue", action="store_true")
    ap.add_argument("--local", action="store_true"); ap.add_argument("--teste", action="store_true")
    ap.add_argument("--insumos", default=j1.DS_INSUMOS); ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    if a.enqueue:
        t = Task.init(project_name="TesteMLCab", task_name=NOME)
        t.set_base_docker("python:3.11-slim", docker_setup_bash_script=["apt-get update -qq", "apt-get install -y -qq libgomp1"])
        t.set_packages(j1.PACOTES); t.set_parameters({"General/insumos": a.insumos, "General/procs": str(a.procs)})
        # o padrão do ClearML pega o remoto `origin` (SIMPred), onde este commit não existe: fixa o Branch_Cabiunas
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=AQUI, text=True).strip()
        t.set_script(repository="git@github.com:Science-DFix/Branch_Cabiunas.git", branch="exp/ocsvm-validacao", commit=commit)
        t.execute_remotely(queue_name="default")
    if os.environ.get("CLEARML_TASK_ID"):
        tk = Task.init(project_name="TesteMLCab", task_name=NOME)
        a.insumos = tk.get_parameter("General/insumos") or a.insumos
        a.procs = int(tk.get_parameter("General/procs") or a.procs)
    executa(a.local, a.insumos, a.teste, a.procs)


if __name__ == "__main__":
    main()
