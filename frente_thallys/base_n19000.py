#!/usr/bin/env python3
"""Base "atual n = 19.000" do Thallys (EXP 60-62): os canais t e p trocados pelo PCA(0,95) fiel, em walk-forward mensal
com as últimas 19.000 amostras estáveis, pela decisão de produção (`decide_v2`).

Os escores vêm do mesmo caminho do EXP 34 (`walk_forward(com_sk(PCA95), grade, tags, estável, 01/02/2025, fim,
n_treino=n, semente=0)`) e ficam em frente_thallys/dados/escores_sub34/{t,p}_A_{n}.parquet. A base se monta como
`base_n` do EXP 60: EWMA com a meia-vida do detector, sobre os escores, no lugar de t e p.

Conferências (números do Thallys, EXP 60 e PDF de 21/10):
  n = 20.000 coincide com o controle atual em >= 99,9% dos instantes vigiados desde 01/02/2025 (V2 do EXP 60)
  n = 19.000: 5/8, FP 11, 1.505 h; com a regra de carga do EXP 62 (principal): 5/8, FP 6

Uso:  python frente_thallys/base_n19000.py escores | confere
"""
from __future__ import annotations
import argparse, contextlib, importlib.util, io, sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import reproduz as R  # noqa: E402  (antes do nv: fixa o NV_DADOS)

import types  # noqa: E402
import pandas as pd  # noqa: E402

# `nv.modelos.familias` importa lightgbm no topo só para uma classe que o PCA(0,95) não usa (familias.py:65), e o
# ambiente local não o tem: módulo vazio no lugar. Se algum caminho chamar `lgb.*`, quebra com AttributeError.
sys.modules.setdefault("lightgbm", types.ModuleType("lightgbm"))

CACHE = R.DADOS / "escores_sub34"
NS = (20000, 19000)


def _exp(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, R.REPO_T / "experiments" / arq)
    m = importlib.util.module_from_spec(spec); sys.modules[nome] = m; spec.loader.exec_module(m); return m


def escores():
    E34 = _exp("exp34", "34_subespacos_sem_vazamento.py")
    from nv import io as nvio
    from nv.walk import walk_forward
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool)
    ci, _ = R.modelos_ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(R.DADOS / "modelos_local")
    tags = {c: list(modelos[-1]["normalizacao"][f]["cols"]) for c, f in {"t": "temperatura", "p": "pressao"}.items()}
    _, vig = R.carrega_controle(); fim = vig.index[-1]
    CACHE.mkdir(exist_ok=True)
    for c in ("t", "p"):
        for n in NS:
            s = walk_forward(E34.CANDS["A pca95"], g, tags[c], est, E34.INICIO8, fim, n_treino=n, semente=0)
            s.to_frame("s").to_parquet(CACHE / f"{c}_A_{n}.parquet"); print(c, n, s.notna().sum(), flush=True)


def alarme_n(n, det=None):
    """Série de alarme da base com t e p do PCA fiel em n amostras (como `base_n` do EXP 60), na grade do controle."""
    from nv.plugin import decide_v2, substitui
    if det is None:
        _, det = R.modelos_ci()
    proc = pd.read_parquet(R.DADOS / "sinais_v2_8ev_local_treino.parquet")[["t", "p", "sp", "vb", "mask", "reset"]].copy()
    proc["mask"] = proc["mask"].astype(bool)
    for c in ("t", "p"):
        s = pd.read_parquet(CACHE / f"{c}_A_{n}.parquet")["s"]
        proc = substitui(proc, c, s.ewm(halflife=pd.Timedelta(det["halflife"][c]), times=s.index).mean().where(s.notna()))
    with contextlib.redirect_stdout(io.StringIO()):
        a = decide_v2(proc, det)["is_anomaly"].astype(bool)
    al, _ = R.carrega_controle()
    return a.reindex(al.index).fillna(False)


def confere():
    al, vig = R.carrega_controle()
    a20 = alarme_n(20000); vg = vig.loc[R.INICIO8:]; vg = vg[vg]
    v2 = float((a20.reindex(vg.index) == al.reindex(vg.index)).mean())
    print(f"n=20000 x controle atual: coincidência {100 * v2:.3f}% dos instantes vigiados -> {'PASSA' if v2 >= 0.999 else 'FALHA'}")
    m = R.mede(alarme_n(19000), vig)
    print(f"n=19000: {m['det']}/8 FP {m['fp']} {m['horas']:.0f} h -> {'PASSA' if (m['det'], m['fp'], round(m['horas'])) == (5, 11, 1505) else 'FALHA'}")
    E62 = _exp("exp62", "62_regra_de_carga.py")
    from nv import io as nvio
    ci, _ = R.modelos_ci(); a19 = alarme_n(19000)
    g = nvio.carrega_grade(); t5 = g["T5_AVG_A"].where(nvio.mascaras(g)["estavel"])
    eps = [(a, b) for a, b in ci._episodios(a19.loc[R.INICIO8:])]
    L = [E62.L_de(t5, a, 30, 2)[0] for a, _ in eps]
    m = R.mede(E62.aplica(a19, eps, L, 0.5), vig)
    print(f"n=19000 + carga (EXP 62 principal): {m['det']}/8 FP {m['fp']} {m['horas']:.0f} h -> {'PASSA' if (m['det'], m['fp']) == (5, 6) else 'FALHA'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("acao", choices=["escores", "confere"])
    {"escores": escores, "confere": confere}[ap.parse_args().acao]()
