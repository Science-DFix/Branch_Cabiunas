#!/usr/bin/env python3
"""Reprodução local do detector v2 do Thallys (TC-33003A), com o retreino mensal feito do zero. Sem ClearML.

Fonte: clone SIMPred-Petrobras/cabiunas-models em ~/FRENTE_THALLYS/repro_thallys_regra_carga_9fp_4fp
(`exp/modos-de-falha` para o harness `src/nv` e os experimentos; `feat/pacote-consolidado` para o pacote).
Dados em frente_thallys/dados/ (fora do git):
  nvdados/          árvore que o `nv.io` espera: pacote e modelos do git, registro de trips (os `trips_conhecidos`
                    do último bundle), CSV de produção = Dataset a97ba56b de 2025 em diante, grade = Dataset 8b06a98f
  nvdados_treino/   a mesma árvore, com modelos/ -> modelos_local (os bundles refeitos aqui)
  raw30s_a97_2024_2026.parquet   o a97ba56b inteiro, texto do PI -> NaN (entrada do retreino)

Conferido em 25/10/2026 (DIARIO_DE_BORDO.md):
  bundles   16 bundles iguais aos do git (diferença máx. ~1e-14)
  sinais    t, p, sp iguais à `scripts/ref_sinais_v2_8ev.json`; vb difere em 2 de 390.762 instantes (dado de 2024)
  controle  7/8, 9 FP, 0,807/mês, 1.457 h, 407 h fora (= o publicado)

Uso:  python frente_thallys/reproduz.py bundles | sinais | controle
"""
from __future__ import annotations
import argparse, contextlib, io, json, os, subprocess, sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
DADOS = AQUI / "dados"
REPO_T = Path.home() / "FRENTE_THALLYS" / "repro_thallys_regra_carga_9fp_4fp"
A97 = Path.home() / ".clearml/cache/storage_manager/datasets/ds_a97ba56ba14840fbb1125c2a82f883c9/sensores_full_2024_2026_30s.csv"
MESES = [f"{a}-{m:02d}" for a, m in [(2025, m) for m in range(1, 13)] + [(2026, m) for m in range(1, 5)]]
INICIO8 = "2025-02-01"

os.environ.setdefault("NV_DADOS", str(DADOS / "nvdados_treino"))  # antes de importar o nv: o nv.io lê na importação
sys.path[:0] = [str(REPO_T / "src"), str(REPO_T / "scripts")]

import pandas as pd  # noqa: E402


def bundles():
    pk = DADOS / "nvdados" / "estrutura_pra_prod" / "Cabiunas"
    for m in MESES:
        subprocess.run([sys.executable, "constroi_bundle.py", "--historico", str(DADOS / "raw30s_a97_2024_2026.parquet"),
                        "--mes", m, "--trips", str(pk / "registro_trips.csv"), "--saida", str(DADOS / "modelos_local")],
                       cwd=pk / "scripts", check=True)


def sinais(saida=None):
    """Replay causal semanal com os bundles da árvore em NV_DADOS; grava em `saida` (padrão: o controle de referência)."""
    from nv.controle_sinais import sinais_causais
    from worker_verifica_harness import verifica
    x = pd.read_csv(A97, index_col=0, parse_dates=[0], dtype=str, nrows=1_100_000)  # 2024 inteiro = ~1,05 M linhas
    x = x.apply(pd.to_numeric, errors="coerce"); x.index = x.index.tz_localize("UTC")
    x = x.loc["2024-09-01":"2024-12-31 23:59:59"]                                    # o complemento de 2024 do Thallys
    S = sinais_causais(pd.Timestamp("2025-01-08 07:34", tz="UTC"), cache=None, extra=x)
    S.to_parquet(saida or DADOS / "sinais_v2_8ev_local_treino.parquet")
    ok, linhas = verifica(S, json.load(open(REPO_T / "scripts" / "ref_sinais_v2_8ev.json")))
    print("\n".join(linhas)); print("impressão digital:", "PASSA" if ok else "FALHA (esperado: só vb, mask e reset por 1-2 instantes)")


def modelos_ci():
    from nv.controle_sinais import _ci, PKG
    ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    return ci, det


def controle():
    from nv.plugin import decide_v2
    _, det = modelos_ci()
    proc = pd.read_parquet(DADOS / "sinais_v2_8ev_local_treino.parquet")
    proc["mask"] = proc["mask"].astype(bool); proc["reset"] = proc["reset"].astype(bool)
    with contextlib.redirect_stdout(io.StringIO()):
        res = decide_v2(proc, det)
    pd.DataFrame({"alarme": res["is_anomaly"].astype(int), "vigiado": res["vigiado"].astype(int)}).rename_axis("ts") \
        .to_csv(DADOS / "controle_atual_local_treino.csv.gz")
    al, vig = carrega_controle()
    r = mede(al, vig)
    print(f"det {r['det']}/8 · início {r['inicio']} · banda {r['banda']} · FP {r['fp']} · FP/mês {r['fp_mes']:.3f} · "
          f"horas {r['horas']:.0f} · fora {r['fora']:.0f}")


def carrega_controle():
    """(alarme, vigiado) na grade de 2 min, como `nv.avalia.carrega_controle`."""
    s = pd.read_csv(DADOS / "controle_atual_local_treino.csv.gz")
    s["ts"] = pd.to_datetime(s["ts"], utc=True); s = s.set_index("ts").sort_index()
    s = s.reindex(pd.date_range(s.index[0], s.index[-1], freq="2min"))
    return s["alarme"].fillna(0).astype(bool), s["vigiado"].fillna(0).astype(bool)


def trips():
    from nv.causal import carrega_registro
    return [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]


def mede(al, vig):
    """A régua do Thallys (`nv.avalia.avalia`, desde 01/02/2025), nos campos do EXP 60."""
    from nv.avalia import avalia
    r = avalia(al, vig, trips(), inicio_periodo=pd.Timestamp(INICIO8, tz="UTC"))
    return dict(det=r.det, inicio=r.inicio, banda=r.banda, fp=r.fp, fp_mes=r.fp_mes, horas=r.horas_alarme,
                fora=r.horas_alarme_fora, ev={k[:10]: int(v["det"]) for k, v in r.por_evento.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("acao", choices=["bundles", "sinais", "controle"]); ap.add_argument("--saida")
    a = ap.parse_args()
    {"bundles": bundles, "sinais": lambda: sinais(a.saida and Path(a.saida)), "controle": controle}[a.acao]()
