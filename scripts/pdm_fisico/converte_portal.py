#!/usr/bin/env python3
"""Converte os mensais *_interpolated_PortalIntegridade.xlsx (30 s, 86 tags) em parquet.

Mantem o timestamp BRUTO do arquivo, sem fuso. Ele JA E UTC, na convencao da grade2min: a correlacao com as
colunas comuns e maxima com deslocamento 0 (descontar 3 h desalinha) -- ver DIARIO_PORTAL.md. Texto (No Data, Comm Fail, I/O Timeout...) vira NaN; a contagem de cada texto por tag vai para
parquet/_textos.csv, que e a medida de qualidade da tag.

Uso:  python converte_portal.py --raiz /caminho/dados_antigos     (requer python-calamine)
Depois, para a serie unica de 30 s, concatenar parquet/20*.parquet sem os *_OLD (ver DIARIO_PORTAL.md).
"""
import argparse, glob, re
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--raiz", required=True, help="pasta que contem PI/ (os zips do Drive extraidos)")
RAIZ = Path(ap.parse_args().raiz)
OUT = RAIZ / "parquet"; OUT.mkdir(exist_ok=True)
arqs = sorted(glob.glob(str(RAIZ / "PI/20*/*_interpolated_PortalIntegridade.xlsx")))
arqs += [a for a in sorted(glob.glob(str(RAIZ / "PI/OLD/2025/*_interpolated_PortalIntegridade.xlsx")))]
textos = []
for a in arqs:
    p = Path(a); m = re.match(r"(\d{2})_(\d{4})", p.name)
    nome = f"{m.group(2)}-{m.group(1)}" + ("_OLD" if "/OLD/" in a else "")
    dest = OUT / f"{nome}.parquet"
    if dest.exists():
        continue
    d = pd.read_excel(a, engine="calamine")
    d = d.rename(columns=lambda c: c.replace("bapiha02-", "")).set_index("data_datetime")
    for c in d.columns:
        num = pd.to_numeric(d[c], errors="coerce")
        txt = d[c][num.isna() & d[c].notna()].astype(str)
        for k, n in txt.str.extract(r"'Name': '([^']+)'")[0].fillna(txt).value_counts().items():
            textos.append(dict(arquivo=nome, tag=c, texto=k, n=int(n)))
        d[c] = num.astype("float32")
    d.to_parquet(dest)
    print(nome, d.shape, d.index.min(), d.index.max(), flush=True)
    pd.DataFrame(textos).to_csv(OUT / "_textos.csv", mode="a", header=not (OUT / "_textos.csv").exists(), index=False)
    textos = []
