"""Motor com estado do TC-33003A: funções de referência para a execução incremental do detector.

Faz, a cada atualização, o mesmo que o lote de 60 d de `cabiunas_inference.py` (`preprocessar` + `prever`), lendo só
3 d de bruto. Não altera o pacote: usa as funções dele, e a única intervenção é passar ao `_z_vibracao` do pacote o
buffer de amostras estáveis do próprio motor como base (troca temporária, desfeita ao sair).

Estado por equipamento (3,3 MB em parquet):
  - `buf`: últimas 26.000 linhas estáveis das sondas de vibração (a base usa 13.000; o resto permite refazer até ~17 d);
  - `tab`: últimos 45 d dos sinais (t, p, sp, vb, mask, reset), em grade de 2 min;
  - `ultimo`: instante da última linha processada.

Uso (a entrada é a grade de 2 min, `ci.preparar_grade` sobre o export de 30 s, só com janelas fechadas):

    m = MotorEstado(ci, modelos)
    m.carga(grade.loc[c0 - 60 d : c0], trips, c0)          # instalação nova: 60 d de bruto
    linhas = m.passo(grade.loc[c - 3 d : c], c, trips)     # cada atualização; devolve as linhas novas com o alarme
    linhas = m.passo(grade.loc[x - 3 d : c], c, trips, desde=x)   # refazer: dado corrigido/atrasado desde x, ou
                                                                  # trip registrado tarde (x = trip − 7 d)
    m.salva(pasta); m = MotorEstado.le(ci, modelos, pasta)  # reinício

Validado contra o lote de 60 d com igualdade exata (sinais |Δ| <= 1e-9, máscara e alarme idênticos): ver
`teste_equivalencia.py` nesta pasta. Origem: J2 e J34 (frente_thallys).
"""
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

BRUTO_D = 3            # dias de bruto lidos por atualização
TAB_D = 45             # dias de sinais guardados (memória da decisão: CUSUM e cadeia do refratário)
N_BUF = 13_000         # linhas estáveis do buffer usadas como base do vb (>= 12.000 da base + 720 da guarda)
N_GUARDA_BUF = 26_000  # linhas estáveis guardadas (base + janela de 3 d + refazer até ~17 d)
COLS = ["t", "p", "sp", "vb", "mask", "reset"]


class MotorEstado:
    def __init__(self, ci, modelos):
        self.ci, self.modelos = ci, modelos
        self.det = modelos[-1]["detector"]
        self.tags = self.det["tags_vibracao"]
        self.buf = pd.DataFrame(columns=self.tags, dtype="float32", index=pd.DatetimeIndex([], tz="UTC"))
        self.tab = None
        self.ultimo = None

    # ── vb com o buffer do motor ──
    @contextlib.contextmanager
    def _vb_com_buffer(self, buf):
        ci, orig = self.ci, self.ci._z_vibracao

        def z(g, estavel, tags, trips, excl):
            if buf is None:
                return orig(g, estavel, tags, trips, excl)
            pos = buf.index.searchsorted(g.index[0])
            b = buf.iloc[max(0, pos - N_BUF):pos]
            gb = pd.concat([b[tags], g[tags]])
            eb = pd.concat([pd.Series(True, index=b.index), estavel.astype(bool)])
            return np.asarray(orig(gb, eb, tags, trips, excl))[len(b):]

        ci._z_vibracao = z
        try:
            yield
        finally:
            ci._z_vibracao = orig

    def _preprocessar(self, w, trips, buf):
        with self._vb_com_buffer(buf), contextlib.redirect_stdout(io.StringIO()):
            return self.ci.preprocessar(self.modelos, w, trips=trips)

    def _prever(self, proc):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.ci.prever({"detector": self.det}, proc)

    # ── estado ──
    def _estaveis(self, w):
        g = self.ci._grade(w)
        _, est, _, _ = self.ci._mascara(g)
        return g.loc[est.to_numpy(), self.tags]

    def _anexa_buf(self, w):
        e = self._estaveis(w)
        if len(self.buf):
            e = e[e.index > self.buf.index[-1]]
            self.buf = pd.concat([self.buf, e]).iloc[-N_GUARDA_BUF:]
        else:
            self.buf = e.iloc[-N_GUARDA_BUF:]

    def _anexa_tab(self, novo, c):
        novo = novo[COLS]
        if self.tab is not None and len(self.tab) and len(novo):
            # instantes sem dado no banco entram como linhas vazias, como na grade da referência
            grade = pd.date_range(self.tab.index[-1] + pd.Timedelta("2min"), novo.index[-1], freq="2min", tz="UTC")
            novo = novo.reindex(grade)
            novo["mask"] = novo["mask"].fillna(False).astype(bool)
            novo["reset"] = novo["reset"].fillna(True).astype(bool)
        tab = novo if self.tab is None else pd.concat([self.tab, novo])
        self.tab = tab.loc[c - pd.Timedelta(f"{TAB_D}D"):]

    # ── operações ──
    def carga(self, hist, trips, c):
        """Instalação nova ou recuperação sem estado salvo: monta o estado de uma vez a partir do histórico de bruto."""
        if not len(hist):
            return
        proc = self._preprocessar(hist, trips, None)
        self._anexa_tab(proc, c)
        self._anexa_buf(hist)
        self.ultimo = proc.index[-1]

    def passo(self, banco, c, trips, desde=None):
        """Uma atualização no instante c. `banco` precisa cobrir [c − 3 d, c] (ou [desde − 3 d, c] ao refazer).
        `desde`: refazer a partir deste instante (dado corrigido ou atrasado; trip registrado tarde: trip − 7 d).
        Devolve as linhas novas (sinais, is_anomaly, vigiado) ou None. Linha já emitida não deve ser reescrita na tela:
        quem chama guarda o que emitiu num registro só-acréscimo."""
        ini = c - pd.Timedelta(f"{BRUTO_D}D")
        if desde is not None:
            ini = min(ini, desde - pd.Timedelta(f"{BRUTO_D}D"))
            self.tab = self.tab.loc[:desde - pd.Timedelta("1ns")] if self.tab is not None else None
            self.buf = self.buf.loc[:desde - pd.Timedelta("1ns")]
            self.ultimo = self.tab.index[-1] if self.tab is not None and len(self.tab) else None
        w = banco.loc[ini:c]
        if not len(w):
            return None
        proc = self._preprocessar(w, trips, self.buf)
        novo = proc[proc.index > self.ultimo] if self.ultimo is not None else proc
        if not len(novo):
            return None
        self._anexa_tab(novo, c)
        res = self._prever(self.tab)
        self._anexa_buf(w)
        self.ultimo = proc.index[-1]
        lin = self.tab.loc[novo.index[0]:]
        return lin[COLS].join(res[["is_anomaly", "vigiado"]], how="left")

    def salva(self, pasta):
        pasta = Path(pasta)
        self.tab.to_parquet(pasta / "tab.parquet")
        self.buf.to_parquet(pasta / "buf.parquet")

    @classmethod
    def le(cls, ci, modelos, pasta):
        pasta = Path(pasta)
        m = cls(ci, modelos)
        m.tab = pd.read_parquet(pasta / "tab.parquet")
        m.buf = pd.read_parquet(pasta / "buf.parquet")
        m.ultimo = m.tab.index[-1] if len(m.tab) else None
        return m
