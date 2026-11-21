#!/usr/bin/env python3
"""Teste de equivalência: motor com estado (`motor_estado.py`) contra o lote de 60 d do pacote, com igualdade exata.

    python teste_equivalencia.py --pacote <pasta Cabiunas> --csv <export de 30 s> [--reinicio] [--procs 8]

Repete a operação ao longo do histórico, com atualizações a cada 6 h (mais uma no instante de cada trip):
  - referência: em cada atualização, `preprocessar` + `prever` sobre os últimos 60 d de bruto, com os trips já
    ocorridos; ficam as linhas novas desde a atualização anterior;
  - motor: carga de 60 d na primeira atualização e depois `passo` com 3 d de bruto; com --reinicio, o estado é
    gravado e relido do disco em toda atualização.
Compara linha a linha a partir de --compara-desde: sinais t, p, sp, vb (|Δ| <= 1e-9 e mesmos vazios), máscara e
alarme idênticos. Sai com código 0 se tudo for igual, 1 se não. Grava o resumo em JSON (--saida).

Precisa só do pacote (`scripts/cabiunas_inference.py`, `modelos/`, `registro_trips.csv`), pandas, numpy e pyarrow.
"""
import os

# Uma thread de BLAS: mesma saída e ~1/10 da CPU no passo do motor (J19). Tem de vir antes de importar o numpy.
# Quem já definiu a variável no ambiente prevalece.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motor_estado import COLS, MotorEstado   # noqa: E402

PASSO = pd.Timedelta("6h")
JANELA_REF = pd.Timedelta("60D")
G = {}


def cortes(inicio, fim, trips):
    c = [inicio]
    while c[-1] + PASSO < fim:
        c.append(c[-1] + PASSO)
    c.append(fim)
    return sorted(set(c) | {t for t in trips if inicio < t < fim})


def conhecidos(s):
    return [t for t in G["trips"] if t < s]


def _ref_um(par):
    """Uma atualização da referência (independente das outras: dá para paralelizar)."""
    anterior, s = par
    m = G["motor_ref"]
    proc = m._preprocessar(G["df"].loc[s - JANELA_REF:s], conhecidos(s), None)
    res = m._prever(proc)
    novo = proc[proc.index > anterior] if anterior is not None else proc
    return novo[COLS].join(res[["is_anomaly", "vigiado"]], how="left")


def referencia(cs, procs):
    pares = list(zip([None] + cs[:-1], cs))
    if procs > 1:
        import multiprocessing as mp
        with ProcessPoolExecutor(procs, mp_context=mp.get_context("fork")) as ex:
            ped = list(ex.map(_ref_um, pares, chunksize=16))
    else:
        ped = [_ref_um(p) for p in pares]
    r = pd.concat(ped)
    return r[~r.index.duplicated(keep="first")]


def motor(cs, reinicio):
    ci, modelos, df = G["ci"], G["modelos"], G["df"]
    m = MotorEstado(ci, modelos)
    m.carga(df.loc[cs[0] - JANELA_REF:cs[0]], conhecidos(cs[0]), cs[0])
    pasta = Path(tempfile.mkdtemp(prefix="motor_estado_"))
    ped, t_passo = [], []
    for s in cs[1:]:
        t0 = time.perf_counter()
        out = m.passo(df, s, conhecidos(s))
        t_passo.append(time.perf_counter() - t0)
        if out is not None:
            ped.append(out)
        if reinicio and m.tab is not None:
            m.salva(pasta)
            m = MotorEstado.le(ci, modelos, pasta)
    m.salva(pasta)
    estado_mb = sum(f.stat().st_size for f in pasta.glob("*.parquet")) / 1e6
    r = pd.concat(ped)
    return r[~r.index.duplicated(keep="first")], dict(passo_s_mediana=round(float(np.median(t_passo)), 3),
                                                      estado_mb=round(estado_mb, 2))


def compara(A, R):
    R = R.reindex(A.index)
    out = dict(linhas=len(A))
    for s in ("t", "p", "sp", "vb"):
        a, b = A[s].to_numpy(float), R[s].to_numpy(float)
        nan_dif = np.isnan(a) != np.isnan(b); ok = ~np.isnan(a) & ~np.isnan(b)
        d = np.zeros_like(a); d[ok] = np.abs(a[ok] - b[ok])
        out[f"{s}_linhas_dif"] = int(((d > 1e-9) | nan_dif).sum())
        out[f"{s}_max_abs"] = float(d.max()) if ok.any() else 0.0
    for s in ("mask", "is_anomaly"):
        out[f"{s}_linhas_dif"] = int((A[s].fillna(False).astype(bool) != R[s].fillna(False).astype(bool)).sum())
    out["igual"] = all(out[k] == 0 for k in out if k.endswith("_linhas_dif"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pacote", required=True, help="pasta Cabiunas (scripts/, modelos/, registro_trips.csv)")
    ap.add_argument("--csv", required=True, help="export de 30 s")
    ap.add_argument("--inicio", default=None, help="primeira atualização (padrão: início do dado + 60 d)")
    ap.add_argument("--fim", default=None, help="última atualização (padrão: fim do dado)")
    ap.add_argument("--compara-desde", default=None, help="padrão: primeira atualização + 7 d")
    ap.add_argument("--reinicio", action="store_true", help="grava e relê o estado em toda atualização")
    ap.add_argument("--procs", type=int, default=1, help="processos para a referência")
    ap.add_argument("--saida", default="teste_equivalencia.json")
    a = ap.parse_args()

    pk = Path(a.pacote)
    sys.path.insert(0, str(pk / "scripts"))
    import cabiunas_inference as ci
    t00 = time.time()
    df = ci.preparar_grade(ci.carregar_dados(a.csv))
    modelos = ci.carregar_modelos(pk / "modelos")
    trips = sorted(ci.carregar_trips(pk / "registro_trips.csv"))
    G.update(ci=ci, df=df, modelos=modelos, trips=trips, motor_ref=MotorEstado(ci, modelos))
    ini = pd.Timestamp(a.inicio, tz="UTC") if a.inicio else df.index[0] + JANELA_REF
    fim = pd.Timestamp(a.fim, tz="UTC") if a.fim else df.index[-1]
    cs = cortes(ini, fim, trips)
    desde = pd.Timestamp(a.compara_desde, tz="UTC") if a.compara_desde else ini + pd.Timedelta("7D")
    print(f"{len(df)} linhas de 2 min, {len(modelos)} bundles, {len(trips)} trips; {len(cs)} atualizações de {ini} a "
          f"{fim}; compara desde {desde}", flush=True)

    t0 = time.time(); R = referencia(cs, a.procs); t_ref = time.time() - t0
    print(f"referência: {len(R)} linhas em {t_ref:.0f} s", flush=True)
    t0 = time.time(); M, custo = motor(cs, a.reinicio); t_mot = time.time() - t0
    print(f"motor{' (reinício a cada atualização)' if a.reinicio else ''}: {len(M)} linhas em {t_mot:.0f} s; "
          f"{custo}", flush=True)

    res = compara(M.loc[desde:], R)
    res.update(reinicio=a.reinicio, atualizacoes=len(cs), inicio=str(ini), fim=str(fim), compara_desde=str(desde),
               ref_s=round(t_ref), motor_s=round(t_mot), **custo, total_min=round((time.time() - t00) / 60, 1))
    falta = M.loc[desde:].index.difference(R.index)
    res["linhas_motor_sem_ref"] = len(falta)
    Path(a.saida).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False), flush=True)
    print("IGUAL" if res["igual"] and not len(falta) else "DIFERENTE", flush=True)
    return 0 if res["igual"] and not len(falta) else 1


if __name__ == "__main__":
    raise SystemExit(main())
