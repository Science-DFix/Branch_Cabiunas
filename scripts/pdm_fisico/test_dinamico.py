"""Testes dos scorers dinamicos em processo SINTETICO com resposta conhecida.

Rode de dentro de scripts/pdm_fisico:  python -m pytest test_dinamico.py -q

Cada teste fixa uma propriedade que, se quebrar, invalida o uso no detector:
calibracao no normal, separacao nivel x dinamica (SFA), residuo e
dissimilaridade (CVA), causalidade e contiguidade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dinamico import ScorerCVA, ScorerSFA, _posicao_no_trecho

D, K = 8, 3                       # sensores, fatores latentes
RNG = np.random.default_rng(42)
MIX = RNG.normal(0, 1, (K, D))


def processo(n, phi=0.98, offset=0.0, seed=0, inicio="2025-01-01"):
    """Fatores AR(1) com variancia estacionaria UNITARIA para qualquer phi --
    mudar phi muda so a dinamica, nao o nivel nem a variancia."""
    rng = np.random.default_rng(seed)
    F = np.zeros((n, K))
    e = rng.normal(0, np.sqrt(1 - phi ** 2), (n, K))
    F[0] = rng.normal(0, 1, K)
    for i in range(1, n):
        F[i] = phi * F[i - 1] + e[i]
    X = (F + offset) @ MIX + rng.normal(0, 0.1, (n, D))
    idx = pd.date_range(inicio, periods=n, freq="2min", tz="UTC")
    return pd.DataFrame(X, index=idx, columns=[f"S{j}" for j in range(D)])


def com_buracos(df, n_buracos=15, seed=1):
    """Remove blocos, imitando o baseline de 'pontos estaveis' do detector."""
    rng = np.random.default_rng(seed)
    keep = np.ones(len(df), bool)
    for a in rng.integers(0, len(df) - 200, n_buracos):
        keep[a:a + int(rng.integers(30, 200))] = False
    return df[keep]


@pytest.fixture(scope="module")
def base():
    return com_buracos(processo(20_000, seed=0))


@pytest.fixture(scope="module")
def sfa(base):
    return ScorerSFA().fit(base)


@pytest.fixture(scope="module")
def cva(base):
    return ScorerCVA().fit(base)


def gera_normal(n=6000, **kw):
    return processo(n, seed=7, inicio="2025-03-01", **kw)


# ─────────────────────────────────────────── calibracao no normal
@pytest.mark.parametrize("nome", ["sfa", "cva"])
def test_calibrado_no_normal(nome, request):
    sc = request.getfixturevalue(nome)
    s = sc.score(gera_normal())["score"].dropna()
    # score = max de 4 (SFA) ou 3 (CVA) estatisticas, cada uma no p99 -> ate ~4%
    assert s.gt(1).mean() < 0.06, s.gt(1).mean()


# ─────────────────────────────────────────── SFA: nivel x dinamica
def test_sfa_mudanca_de_ponto_de_operacao_acende_nivel_nao_dinamica(sfa):
    # 6 desvios num fator: T2 soma M caracteristicas e o p99 de chi2(M=3) ~ 11,
    # entao 3 desvios ficam perto do limiar -- 6 e o desvio franco
    out = sfa.score(gera_normal(offset=np.array([6.0, 0, 0])))
    assert out["nivel"].median() > 2
    assert out["dinamica"].gt(1).mean() < 0.06


def test_sfa_dinamica_anormal_acende_dinamica(sfa):
    out = sfa.score(gera_normal(phi=0.5))       # mesma variancia, 25x mais rapido
    assert out["dinamica"].gt(1).mean() > 0.5
    assert out["dinamica"].gt(1).mean() > 3 * out["nivel"].gt(1).mean()


# ─────────────────────────────────────────── CVA: Q e D
def test_cva_vies_em_um_sensor_acende_Q(cva):
    X = gera_normal()
    X["S3"] += 1.5                               # quebra a correlacao entre sensores
    out = cva.score(X)
    assert out["Q"].median() > 2


def test_cva_dinamica_anormal_acende_D(cva):
    out = cva.score(gera_normal(phi=0.5))
    assert out["D"].gt(1).mean() > 0.3


# ─────────────────────────────────────────── causalidade
@pytest.mark.parametrize("nome", ["sfa", "cva"])
def test_causal_futuro_nao_altera_passado(nome, request):
    sc = request.getfixturevalue(nome)
    X = gera_normal(3000)
    a = sc.score(X)
    Y = X.copy()
    Y.iloc[2000:] += 50.0                        # perturba so o futuro de t=1999
    b = sc.score(Y)
    pd.testing.assert_frame_equal(a.iloc[:2000], b.iloc[:2000])


# ─────────────────────────────────────────── contiguidade
def test_posicao_reinicia_em_buraco_e_nan():
    idx = pd.date_range("2025-01-01", periods=6, freq="2min", tz="UTC")
    idx = idx.delete(3)                                  # buraco de 4 min
    ok = np.array([True, True, True, True, False])
    assert _posicao_no_trecho(idx, ok).tolist() == [0, 1, 2, 0, -1]


def test_posicao_independe_da_resolucao_do_indice():
    """O parquet do detector vem em datetime64[us]; o sintetico, em ns."""
    idx = pd.date_range("2025-01-01", periods=5, freq="2min", tz="UTC")
    ok = np.ones(5, bool)
    for unidade in ("ns", "us", "ms", "s"):
        assert _posicao_no_trecho(idx.as_unit(unidade), ok).tolist() == [0, 1, 2, 3, 4]


def test_sfa_nao_deriva_atraves_de_buraco(sfa):
    X = gera_normal(1000)
    X = pd.concat([X.iloc[:500], X.iloc[600:] + 0.0])    # buraco de 200 min
    out = sfa.score(X)
    assert np.isnan(out["S2d"].iloc[500])                 # 1a amostra apos o buraco


def test_sfa_acha_o_numero_de_fatores_latentes(sfa):
    """Criterio de Shang: caracteristica mais rapida que a entrada mais rapida
    e ruido. No sintetico ha K fatores lentos -- o criterio tem de achar K."""
    assert sfa.M == K


def test_sfa_fit_ignora_saltos_entre_trechos():
    """Somar um degrau a cada trecho do baseline cria saltos enormes NAS
    FRONTEIRAS. Se a derivada atravessasse buraco, a lentidao explodiria."""
    b = com_buracos(processo(20_000, seed=0))
    pos = _posicao_no_trecho(b.index, np.ones(len(b), bool))
    trecho = np.cumsum(pos == 0)
    b2 = b + (trecho % 2)[:, None] * 0.0                  # controle: sem salto
    b3 = b.copy(); b3.iloc[:, 0] += (trecho % 2) * 30.0   # salto de 30 so na fronteira
    o2 = ScorerSFA(n_lentas=2).fit(b2).omega
    o3 = ScorerSFA(n_lentas=2).fit(b3).omega
    assert o3.max() < 10 * o2.max()


# ─────────────────────────────────────────── contrato com a avaliacao
@pytest.mark.parametrize("nome", ["sfa", "cva"])
def test_subespaco_ortonormal(nome, request):
    B = request.getfixturevalue(nome).subespaco()
    assert np.allclose(B.T @ B, np.eye(B.shape[1]), atol=1e-8)
