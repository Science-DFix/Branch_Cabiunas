#!/usr/bin/env python3
"""Retreino do zero: reconstroi, com o `constroi_bundle.py` do pacote, o bundle de cada mes de um intervalo.

PARA QUE. O retreino mensal ajusta um PCA em 20.000 pontos e leva segundos; a pergunta "quanto custa treinar
tudo do zero" pede a serie inteira de bundles a partir do historico, como num primeiro deploy. Cada mes roda num
processo novo, exatamente como o operador rodaria o retreino mensal; o pacote nao e alterado.
Grava em --saida/meses.json o codigo de saida e a duracao de cada mes (0 = bundle, 1 = historico insuficiente,
4 = bundle com alerta).
Uso:  python retreino_do_zero.py --constroi .../constroi_bundle.py --historico dados.csv --trips trips.csv \\
          --saida /tmp/modelos --de 2025-01 --ate 2026-04
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    for a in ("--constroi", "--historico", "--trips", "--saida", "--de", "--ate"):
        ap.add_argument(a, required=True)
    a = ap.parse_args()
    saida = Path(a.saida); saida.mkdir(parents=True, exist_ok=True)
    meses = []
    for m in pd.period_range(a.de, a.ate, freq="M").strftime("%Y-%m"):
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, a.constroi, "--historico", a.historico, "--mes", m,
                            "--saida", str(saida), "--trips", a.trips], capture_output=True, text=True)
        meses.append(dict(mes=m, codigo=r.returncode, duracao_s=round(time.perf_counter() - t0, 2),
                          erro=r.stderr.strip().splitlines()[-1] if r.returncode not in (0, 1, 4) and r.stderr.strip() else ""))
        print(f"{m}: codigo {r.returncode}  {meses[-1]['duracao_s']} s", flush=True)
    (saida / "meses.json").write_text(json.dumps(meses, indent=1))
    return 0 if all(x["codigo"] in (0, 1, 4) for x in meses) else 2


if __name__ == "__main__":
    sys.exit(main())
