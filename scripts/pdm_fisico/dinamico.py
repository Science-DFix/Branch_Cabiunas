"""Scorers DINAMICOS para os canais t e p: SFA e CVA.

Por que existem. O PCA do detector e estatico: trata cada amostra de 2 min como
independente. Nao distingue "o ponto de operacao andou" (mudanca de carga,
religamento) de "a dinamica do processo ficou anormal" -- e e onde nascem 8 dos
12 falsos positivos, na borda do blackout de 6 h. Os dois modelos daqui separam
isso explicitamente:

  SFA  (Shang et al. 2015, slow feature analysis). Decompoe o processo em
       caracteristicas ordenadas por LENTIDAO. Da quatro estatisticas:
         T2d, T2e  -- NIVEL: o estado saiu do normal? (ponto de operacao)
         S2d, S2e  -- DINAMICA: a velocidade de variacao saiu do normal?
       Mudanca de ponto de operacao acende T2 e deixa S2 quieto; falha que altera
       o comportamento temporal acende S2.

  CVA  (Russell et al. 2000; CVDA de Pilario & Cao 2018). Modelo de estado a
       partir da correlacao entre PASSADO e FUTURO empilhados. Tres estatisticas:
         T2   -- estados canonicos fora do normal
         Q    -- residuo: o passado nao se explica pelos estados (quebra de
                 correlacao entre sensores, vies de um sensor)
         D    -- dissimilaridade: o futuro nao segue o que o passado previa
                 (a dinamica mudou). Sensivel a falha incipiente.

Contrato. Mesma interface do ScorerMax: fit(baseline) -> self, score(df) ->
DataFrame com cada estatistica normalizada pelo p99 do proprio baseline (>1 =
fora do normal) e uma coluna `score` = maximo delas. Numpy puro + RobustScaler
do sklearn: nao quebra o contrato de deploy, que so aceita sklearn.

Duas regras que, se violadas, estragam os dois modelos em silencio:

  1. CONTIGUIDADE. Derivada (SFA) e janela de defasagens (CVA) so existem entre
     amostras realmente consecutivas (passo exato de 2 min, sem NaN). O baseline
     do detector e "20.000 pontos estaveis" -- cheio de buracos de parada. Uma
     diferenca atravessando um buraco vira um falso sinal rapido enorme. Aqui
     toda defasagem e montada DENTRO de trechos contiguos; o inicio de cada
     trecho fica NaN ate ter historia suficiente.
  2. CAUSALIDADE. A estatistica atribuida ao instante t usa so amostras <= t.
     Para o D do CVA, que compara passado e futuro, a janela inteira termina em
     t -- o valor sai com atraso de f passos, nunca com dado do futuro.

Hiperparametros FIXADOS ANTES de olhar os trips (pre-registro -- com 8 rotulos,
cada grau de liberdade a mais e sobreajuste): ver os defaults de cada classe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

PASSO = pd.Timedelta("2min")
Q_NORM = 99.0            # normalizacao pelo p99 do baseline, como o ScorerMax


# ─────────────────────────────────────────────────────────── utilidades
def _posicao_no_trecho(idx: pd.DatetimeIndex, ok: np.ndarray) -> np.ndarray:
    """Para cada linha, quantas amostras contiguas validas a precedem no mesmo
    trecho (0 = inicio de trecho). -1 onde a linha e invalida.

    Trecho novo quando: linha anterior invalida, ou o passo nao e exatamente
    2 min (buraco no historiador ou fronteira entre pedacos do baseline)."""
    n = len(idx)
    pos = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return pos
    # Diferenca como Timedelta, nao via asi8: o parquet do detector vem em
    # datetime64[us] e asi8 devolve a unidade NATIVA -- comparar com PASSO.value
    # (ns) marcava todo passo como buraco e nenhum trecho existia.
    dt_ok = np.concatenate(([False], np.asarray((idx[1:] - idx[:-1]) == PASSO)))
    cont = 0
    for i in range(n):
        if not ok[i]:
            cont = -1
            continue
        cont = cont + 1 if (cont >= 0 and dt_ok[i] and ok[i - 1]) else 0
        pos[i] = cont
    return pos


def _inv_sqrt(C: np.ndarray, rel: float = 1e-8) -> np.ndarray:
    """C^{-1/2} por autodecomposicao, com piso nos autovalores. Defasagens
    empilhadas sao quase colineares; inverter sem piso amplifica ruido numerico."""
    w, U = np.linalg.eigh((C + C.T) / 2)
    w = np.maximum(w, rel * w.max())
    return (U / np.sqrt(w)) @ U.T


def _p99(x: np.ndarray) -> float:
    v = np.nanpercentile(x, Q_NORM)
    return float(v) if np.isfinite(v) and v > 0 else 1.0


def _base_ortonormal(A: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(A)
    return q


# ─────────────────────────────────────────────────────────── SFA
class ScorerSFA:
    """Slow Feature Analysis com as quatro estatisticas de Shang et al. (2015).

    Passos do fit:
      1. escala robusta (mediana/IQR), como o PCA do detector;
      2. branqueamento: z = W x, cov(z) = I;
      3. derivada dz nos pares contiguos; autodecomposicao de cov(dz) em ordem
         CRESCENTE de autovalor -> caracteristicas da mais lenta para a mais
         rapida; omega_j e a lentidao de cada uma;
      4. divisao em M dominantes (lentas) e o resto. Criterio de Shang: e
         residual a caracteristica MAIS RAPIDA que a variavel de entrada mais
         rapida -- nao carrega nada que alguma entrada nao tenha, e ruido.

    `n_lentas` fixa M a mao (None = criterio acima). `piso_rel` descarta so
    direcoes NUMERICAMENTE degeneradas (autovalor < piso_rel x o maior).

    Nao cortar por fracao de variancia: a primeira versao usava 99,9% e, na
    pressao real, ficou com 2 de 12 direcoes -- tags quase constantes tem IQR
    minusculo, o RobustScaler as infla e elas concentram a variancia. As 10
    direcoes cortadas somem do SFA (nao ha estatistica de residuo para elas),
    ou seja, 10 sensores ficariam invisiveis. O SFA classico branqueia tudo.
    """

    def __init__(self, n_lentas: int | None = None, piso_rel: float = 1e-6):
        self.n_lentas = n_lentas
        self.piso_rel = piso_rel

    # --- fit ---
    def fit(self, baseline: pd.DataFrame) -> "ScorerSFA":
        B = baseline.dropna()
        self.cols = list(B.columns)
        self.scaler = RobustScaler().fit(B)
        X = self.scaler.transform(B)
        pos = _posicao_no_trecho(B.index, np.ones(len(B), bool))

        C = np.cov(X, rowvar=False)
        w, U = np.linalg.eigh(C)
        w, U = w[::-1], U[:, ::-1]
        k = max(int(np.sum(w > self.piso_rel * w[0])), 2)
        self.W = U[:, :k] / np.sqrt(w[:k])            # x -> z branqueado (d x k)
        Z = X @ self.W

        rows = np.flatnonzero(pos >= 1)
        if len(rows) < 10 * k:
            raise ValueError(f"baseline com poucos pares contiguos ({len(rows)}) para SFA")
        dZ = Z[rows] - Z[rows - 1]
        om, P = np.linalg.eigh(np.cov(dZ, rowvar=False))   # crescente: lenta -> rapida
        self.P, self.omega = P, np.maximum(om, 1e-12)

        if self.n_lentas is not None:
            M = int(self.n_lentas)
        else:
            dX = X[rows] - X[rows - 1]
            lent_x = dX.var(axis=0) / np.maximum(X.var(axis=0), 1e-12)
            M = int(np.sum(self.omega <= lent_x.max()))
        self.M = int(np.clip(M, 1, k - 1))

        st = self._estatisticas(Z @ P, rows, dZ @ P, len(B))
        self._lim = {c: _p99(v) for c, v in st.items()}
        return self

    def _estatisticas(self, S, rows, dS, n):
        """T2 sobre as caracteristicas; S2 sobre as derivadas, pesadas pela
        lentidao de cada uma (derivada de caracteristica lenta pesa mais)."""
        M = self.M
        out = {"T2d": np.sum(S[:, :M] ** 2, axis=1),
               "T2e": np.sum(S[:, M:] ** 2, axis=1)}
        s2d = np.full(n, np.nan); s2e = np.full(n, np.nan)
        s2d[rows] = np.sum(dS[:, :M] ** 2 / self.omega[:M], axis=1)
        s2e[rows] = np.sum(dS[:, M:] ** 2 / self.omega[M:], axis=1)
        out["S2d"], out["S2e"] = s2d, s2e
        return out

    # --- score ---
    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        Xdf = df[self.cols]
        ok = Xdf.notna().all(axis=1).to_numpy()
        n = len(Xdf)
        res = {c: np.full(n, np.nan) for c in ("T2d", "T2e", "S2d", "S2e")}
        if ok.any():
            X = np.zeros((n, len(self.cols)))
            X[ok] = self.scaler.transform(Xdf[ok])
            S = (X @ self.W) @ self.P
            pos = _posicao_no_trecho(Xdf.index, ok)
            rows = np.flatnonzero(pos >= 1)
            dS = S[rows] - S[rows - 1]
            st = self._estatisticas(S, rows, dS, n)
            for c in res:
                v = st[c] / self._lim[c]
                v[~ok] = np.nan
                res[c] = v
        out = pd.DataFrame(res, index=df.index)
        out["nivel"] = out[["T2d", "T2e"]].max(axis=1)
        out["dinamica"] = out[["S2d", "S2e"]].max(axis=1)
        out["score"] = out[["nivel", "dinamica"]].max(axis=1)
        return out

    def subespaco(self) -> np.ndarray:
        """Base ortonormal, no espaco escalado, das M caracteristicas lentas."""
        return _base_ortonormal(self.W @ self.P[:, :self.M])


# ─────────────────────────────────────────────────────────── CVA
class ScorerCVA:
    """Canonical Variate Analysis com T2, Q (Russell 2000) e D (CVDA, Pilario &
    Cao 2018).

    Vetores empilhados, com defasagens espacadas de `passo` amostras:
        passado p_t = [x_t, x_{t-s}, ..., x_{t-(p-1)s}]      (inclui o presente)
        futuro  f_t = [x_{t+s}, ..., x_{t+f*s}]
    SVD de H = Spp^{-1/2} Spf Sff^{-1/2} = U Sigma V'. Os n primeiros pares
    canonicos sao os estados:
        z = J p,  J = U_n' Spp^{-1/2}                    -> T2 = |z|^2
        e = (I - U_n U_n') Spp^{-1/2} p                  -> Q  = |e|^2
        r = V_n' Sff^{-1/2} f - Sigma_n J p              -> D  = r' Srr^{-1} r
    D e atribuido ao instante do ULTIMO elemento do futuro (causal).

    Defaults pre-registrados: p = f = 6 defasagens a cada 5 amostras (10 min) ->
    janela de 1 h de cada lado, na escala do EWMA de t e p. `n_estados` None =
    menor n cujos Sigma^2 somam 90% do total.
    """

    def __init__(self, p: int = 6, f: int = 6, passo: int = 5,
                 n_estados: int | None = None, frac: float = 0.90):
        self.p, self.f, self.passo = p, f, passo
        self.n_estados, self.frac = n_estados, frac

    def _matrizes(self, X, pos):
        """Passado e futuro alinhados no instante FINAL da janela (causal).

        Para a linha r (fim da janela), o futuro e [x_r, x_{r-s}, ...] e o
        passado comeca f*s amostras antes. Assim nenhuma estatistica atribuida a
        r usa amostra posterior a r."""
        s, p, f = self.passo, self.p, self.f
        rows = np.flatnonzero(pos >= (p - 1) * s + f * s)
        t = rows - f * s                                  # o "presente" do par
        pas = np.hstack([X[t - l * s] for l in range(p)])
        fut = np.hstack([X[t + j * s] for j in range(1, f + 1)])   # termina em r
        return rows, pas, fut

    def fit(self, baseline: pd.DataFrame) -> "ScorerCVA":
        B = baseline.dropna()
        self.cols = list(B.columns)
        self.scaler = RobustScaler().fit(B)
        X = self.scaler.transform(B)
        pos = _posicao_no_trecho(B.index, np.ones(len(B), bool))
        rows, Pm, Fm = self._matrizes(X, pos)
        if len(rows) < 5 * Pm.shape[1]:
            raise ValueError(f"baseline com poucas janelas contiguas ({len(rows)}) "
                             f"para CVA de dimensao {Pm.shape[1]}")
        self.mu_p, self.mu_f = Pm.mean(0), Fm.mean(0)
        Pc, Fc = Pm - self.mu_p, Fm - self.mu_f
        N = len(rows) - 1
        Spp, Sff, Spf = Pc.T @ Pc / N, Fc.T @ Fc / N, Pc.T @ Fc / N
        Ip, If = _inv_sqrt(Spp), _inv_sqrt(Sff)
        U, sig, Vt = np.linalg.svd(Ip @ Spf @ If, full_matrices=False)
        sig = np.clip(sig, 0, 1)
        if self.n_estados is not None:
            n = int(self.n_estados)
        else:
            n = int(np.searchsorted(np.cumsum(sig ** 2) / np.sum(sig ** 2), self.frac) + 1)
        self.n = int(np.clip(n, 1, len(sig) - 1))
        self.sig = sig
        Un, Vn = U[:, :self.n], Vt[:self.n].T
        self.J = Un.T @ Ip                              # n x mp
        self.L = (np.eye(len(Ip)) - Un @ Un.T) @ Ip     # mp x mp
        self.G = Vn.T @ If                              # n x mf
        R = Fc @ self.G.T - (Pc @ self.J.T) * sig[:self.n]
        self.Srr_inv = np.linalg.pinv(np.cov(R, rowvar=False).reshape(self.n, self.n))
        st = self._estatisticas(Pm, Fm)
        self._lim = {c: _p99(v) for c, v in st.items()}
        return self

    def _estatisticas(self, Pm, Fm):
        Pc, Fc = Pm - self.mu_p, Fm - self.mu_f
        Z = Pc @ self.J.T
        E = Pc @ self.L.T
        R = Fc @ self.G.T - Z * self.sig[:self.n]
        return {"T2": np.sum(Z ** 2, axis=1),
                "Q": np.sum(E ** 2, axis=1),
                "D": np.einsum("ij,jk,ik->i", R, self.Srr_inv, R)}

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        Xdf = df[self.cols]
        ok = Xdf.notna().all(axis=1).to_numpy()
        n = len(Xdf)
        res = {c: np.full(n, np.nan) for c in ("T2", "Q", "D")}
        if ok.any():
            X = np.zeros((n, len(self.cols)))
            X[ok] = self.scaler.transform(Xdf[ok])
            rows, Pm, Fm = self._matrizes(X, _posicao_no_trecho(Xdf.index, ok))
            if len(rows):
                st = self._estatisticas(Pm, Fm)
                for c in res:
                    res[c][rows] = st[c] / self._lim[c]
        out = pd.DataFrame(res, index=df.index)
        out["score"] = out[["T2", "Q", "D"]].max(axis=1)
        return out

    def subespaco(self) -> np.ndarray:
        """Base ortonormal do espaco dos estados (no espaco do passado empilhado)."""
        return _base_ortonormal(self.J.T)
