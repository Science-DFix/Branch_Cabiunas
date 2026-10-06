#!/usr/bin/env python3
"""Grade de 2 min do detector + 15 tags do Portal de Integridade: a segunda base de teste.

PARA QUE. A triagem do `tags_ausentes.py` sobre o export interpolated (86 tags, 30 s, 2022-01..2025-10)
destacou vibracao do compressor, mancal de escora e oleo de dreno nas 48 h antes dos trips (6 eventos).
Esta base deixa o detector rodar com essas tags para testar se o teto de FP e do dado. As 38 colunas da
`grade2min.parquet` NAO mudam (conferido no fim, bit a bit); as novas entram por left join no mesmo indice.

COMO. Igual ao `preparar_grade` do pacote: faixa fisica -> NaN, depois mediana de cada janela de 2 min
(rotulo a esquerda). Fuso: o export ja esta em UTC, na mesma convencao da grade -- a correlacao com as
colunas comuns e maxima com deslocamento 0 (descontar 3 h desalinha). Fora da cobertura do export
(2025-11 em diante) as tags novas ficam NaN.

FAIXAS DAS TAGS NOVAS (a do pacote para `TV_` e 0..200 e apagaria o deslocamento axial, que e negativo):
    VI_030x (vibracao compressor)         0 .. 200
    TI_0318..0322 (escora, oleo de dreno)  0 .. 200   (leituras de -5 C com a maquina quente sao defeito)
    ZI_0301/02 (posicao axial)            -2 .. 2
    TV_350A/354A (deslocamento axial)     -0,9 .. 0,9 (-1,0 e sentinela)
    NGP_A, NPT_A (rotacao, %)              0 .. 120

Uso:  python monta_grade_portal.py --grade grade2min.parquet --portal _todos_30s_bruto.parquet --saida grade2min_portal.parquet
"""
from __future__ import annotations
import argparse
import pandas as pd

NOVAS = {
    "954005_624_VI_0301": (0, 200), "954005_624_VI_0302": (0, 200),
    "954005_624_VI_0304": (0, 200), "954005_624_VI_0305": (0, 200),
    "954005_624_TI_0318": (0, 200), "954005_624_TI_0319": (0, 200), "954005_624_TI_0320": (0, 200),
    "954005_624_TI_0321": (0, 200), "954005_624_TI_0322": (0, 200),
    "954005_624_ZI_0301": (-2, 2), "954005_624_ZI_0302": (-2, 2),
    "TV_350A_A": (-0.9, 0.9), "TV_354A_A": (-0.9, 0.9),
    "NGP_A": (0, 120), "NPT_A": (0, 120),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grade", required=True)
    ap.add_argument("--portal", required=True, help="export interpolated de 30 s, indice sem fuso (UTC)")
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    g = pd.read_parquet(a.grade)
    p = pd.read_parquet(a.portal, columns=list(NOVAS))
    p.index = p.index.tz_localize("UTC")
    p = p.loc[g.index[0] - pd.Timedelta("2min"):g.index[-1] + pd.Timedelta("2min")]
    for c, (lo, hi) in NOVAS.items():
        fora = ~p[c].between(lo, hi) & p[c].notna()
        print(f"{c:22} fora da faixa: {int(fora.sum()):>7} de {int(p[c].notna().sum())}")
        p[c] = p[c].where(~fora)
    m = p.resample("2min", label="left", closed="left").median().astype("float32")
    m.index = m.index.as_unit(g.index.unit)   # o parquet da grade e datetime64[us]; o resample sai em ns
    out = pd.concat([g, m.reindex(g.index)], axis=1)
    assert out.index.equals(g.index) and out[g.columns].equals(g), "as 38 colunas originais mudaram"
    out.to_parquet(a.saida)
    cob = out[list(NOVAS)].notna().mean()
    print(f"\n{a.saida}: {out.shape}  {out.index[0]} .. {out.index[-1]}")
    print(f"cobertura das tags novas na grade: {cob.min():.1%} a {cob.max():.1%} "
          f"(ultimo instante com dado: {out[list(NOVAS)].dropna(how='all').index[-1]})")


if __name__ == "__main__":
    main()
