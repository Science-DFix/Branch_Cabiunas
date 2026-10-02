#!/usr/bin/env python3
"""O pacote de produção recebe a mesma entrada que a pesquisa validou?

DIAGNÓSTICO DE DEPLOY (02/10/2026), não candidato: não muda o detector nem veredito
nenhum. Pergunta se o NÚMERO validado vale para o dado que vai chegar.

O PROBLEMA. Todo resultado da pesquisa foi medido sobre `grade2min.parquet`, que o
`build_cache.py` monta a partir do export de 30 s com três tratamentos:
  - objetos de status do PI ("No Data", "Out of Serv") viram NaN;
  - faixa física por tipo de sensor (corta o sentinela de termopar em -40,5 °C);
  - MEDIANA das 4 amostras de cada janela de 2 min (robusta a spike isolado), e
    RUNNING_A como fração da janela com a máquina ligada.
O CSV do pacote (`dados/2025_2026/data_..._raw.csv`) é essa mesma grade -- difere do
parquet em 3e-5 --, e foi sobre ele que a paridade do deploy foi provada. Mas nenhum
desses três passos está no pacote: `cabiunas_inference._regrade` casa cada ponto da
grade com a amostra MAIS PRÓXIMA, e o `prepara_dados.py` que converteria o export do
portal está pendente (INTEGRACAO_DASHBOARD, seção 5, item 1). Quem montar a grade de
2 min do jeito óbvio entrega ao detector uma amostra por janela, não a mediana.

O QUE SE MEDE. O detector de PRODUÇÃO (`cabiunas_inference`, os 16 bundles do pacote,
`registro_trips.csv`) sobre três entradas do mesmo período, 2024-11-01 a 2026-04-30:
  pesquisa       a grade da pesquisa (`grade2min.parquet`)
  amostra+faixa  o export de 30 s, uma amostra por ponto de 2 min (`_regrade` sobre
                 o dado de 30 s), com o corte de faixa física
  amostra crua   idem, sem o corte de faixa (só texto -> NaN)
A régua é a de sempre (`regua_fp.mede`), na grade da pesquisa.

RESULTADO (02/10/2026) -- O NÚMERO VALIDADO SÓ VALE PARA A GRADE DA PESQUISA.

    entrada          det  início  banda  FP/mês  carga h/mês  episódios
    pesquisa          8     6      5    0,344      48,9         21    <- o publicado, exato
    amostra+faixa     8     6      5    0,603      67,2         24
    amostra crua      8     6      5    0,603      95,7         25

  · Sobre a grade da pesquisa, o pacote reproduz o publicado exatamente (é a
    composição do dia 1, a dos 16 bundles do pacote).
  · Uma amostra por janela de 2 min em vez da mediana: FP +75%, carga +37%. A
    detecção não muda -- ruído a mais compra alarme, não trip.
  · Sem o corte de faixa, a carga quase dobra (+96%).
  · Onde: a EWMA do p chega a 3,0x a da grade da pesquisa no p99 dos instantes
    vigiados (t 1,6x); sp e vb quase não mudam. A máscara difere em 0,12%.
  · Consequência: os três passos do `build_cache.py` (texto -> NaN, faixa física,
    mediana de 2 min sobre o dado de 30 s, RUNNING_A como fração) são PARTE DO
    MODELO. Têm de estar no pacote -- no `prepara_dados.py` pendente ou, melhor,
    no próprio `cabiunas_inference`, que hoje recebe a grade pronta e não confere.

Uso:  PYTHONPATH=. python paridade_entrada.py
"""
from __future__ import annotations
import io, contextlib, sys
from pathlib import Path
import numpy as np, pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    import regua_fp as R

AQUI = Path(__file__).resolve().parent
PKG = AQUI.parents[1] / "estrutura_pra_prod" / "Cabiunas"
sys.path.insert(0, str(PKG / "scripts"))
import cabiunas_inference as CI  # noqa: E402

SRC = AQUI.parents[3] / "dados" / "sensores_full_2024_2026_30s.csv"
INI, FIM = pd.Timestamp("2024-11-01", tz="UTC"), pd.Timestamp("2026-04-30 23:58", tz="UTC")


def faixa(c: str) -> tuple[float, float]:
    """A mesma de `build_cache.py`."""
    if c.startswith("TC382") or c.startswith("T5"):
        return (-15.0, 900.0)
    if "_TI_" in c or c.startswith("TI_"):
        return (-15.0, 900.0)
    if c.startswith("TV_"):
        return (0.0, 200.0)
    if "_PDI" in c or c.startswith("PDI"):
        return (-5.0, 200.0)
    if "_PI_" in c or c.startswith("PI_"):
        return (-1.5, 200.0)
    return (-1e9, 1e9)


def amostrada(com_faixa: bool) -> pd.DataFrame:
    """O export de 30 s reduzido a uma amostra por ponto de 2 min, como o
    `_regrade` do pacote faria se recebesse o dado de 30 s. Cacheado."""
    f = R.CACHE / f"entrada_amostrada_{'faixa' if com_faixa else 'crua'}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    cols = [c for c in pd.read_csv(SRC, nrows=0).columns
            if c not in ("data_datetime", "any_sensor_constant_run")]
    partes = []
    for ch in pd.read_csv(SRC, chunksize=250_000, low_memory=False):
        ch["data_datetime"] = pd.to_datetime(ch["data_datetime"], utc=True, errors="coerce")
        ch = ch.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
        ch = ch[(ch.index >= INI - pd.Timedelta(hours=1)) & (ch.index <= FIM + pd.Timedelta(hours=1))]
        if not len(ch):
            continue
        d = {}
        for c in cols:
            v = pd.to_numeric(ch[c], errors="coerce").astype("float64")
            if com_faixa:
                lo, hi = faixa(c)
                v = v.where((v >= lo) & (v <= hi))
            d[c] = v
        partes.append(CI._regrade(pd.DataFrame(d, index=ch.index)))
    g = pd.concat(partes)
    g = g[~g.index.duplicated(keep="first")].sort_index()
    g = g.loc[(g.index >= INI) & (g.index <= FIM)]
    g.to_parquet(f)
    return g


def pesquisa() -> pd.DataFrame:
    g = pd.read_parquet(AQUI / "grade2min.parquet")
    return g.loc[(g.index >= INI) & (g.index <= FIM)].astype("float64")


def roda(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    modelos = CI.carregar_modelos(PKG / "modelos")
    trips = CI.carregar_trips(PKG / "registro_trips.csv")
    proc = CI.preprocessar(modelos, df, trips=trips)
    res = CI.prever(modelos, proc)
    fin = res["is_anomaly"].reindex(R.idx, fill_value=False).astype(bool) & R.sel
    return fin, proc


def main():
    pd.set_option("display.width", 200)
    entradas = {"pesquisa": pesquisa, "amostra+faixa": lambda: amostrada(True),
                "amostra crua": lambda: amostrada(False)}
    linhas, procs = [], {}
    for nome, ger in entradas.items():
        df = ger()
        fin, proc = roda(df)
        procs[nome] = proc
        m = R.mede(fin); m.pop("cls")
        linhas.append(dict(entrada=nome, **m))
        print(f"{nome:14s} det {m['det']} início {m['inicio']} banda {m['banda']} "
              f"FP/mês {m['fp_mes']:.3f} carga {m['carga_mes']:.1f} h/mês "
              f"episódios {m['episodios']}", flush=True)
    L = pd.DataFrame(linhas).set_index("entrada")
    L.to_csv(R.CACHE / "paridade_entrada.csv")

    print("\nQUANTO OS SINAIS MUDAM (instantes vigiados, contra a entrada da pesquisa):")
    ref = procs["pesquisa"]
    for nome in list(entradas)[1:]:
        p = procs[nome].reindex(ref.index)
        vig = ref["mask"].astype(bool) & p["mask"].fillna(False).astype(bool)
        print(f"  {nome}: máscara difere em {100 * float((ref['mask'] != p['mask']).mean()):.2f}% dos instantes")
        for c in ("t", "p", "sp", "vb"):
            a, b = ref.loc[vig, c], p.loc[vig, c]
            ok = a.notna() & b.notna()
            r = (b[ok] / a[ok].clip(lower=1e-9))
            print(f"    {c:2s}  razão EWMA amostra/mediana: p50 {r.median():.3f}  p90 {r.quantile(.9):.3f}  "
                  f"p99 {r.quantile(.99):.3f}   | EWMA > 2x limiar base: "
                  f"{100 * float((a[ok] > 2 * R.DB.BASE[c]).mean()):.1f}% -> {100 * float((b[ok] > 2 * R.DB.BASE[c]).mean()):.1f}%")


if __name__ == "__main__":
    main()
