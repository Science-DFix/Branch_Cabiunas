#!/usr/bin/env python3
"""O pacote corrigido reproduz o publicado a partir do export de 30 s?

TESTE DE PARIDADE (02/10/2026) da correção nascida de `paridade_entrada.py`: o pacote
passa a montar ele mesmo a grade de 2 min do treino (`cabiunas_inference.preparar_grade`:
texto -> NaN, faixa física, mediana de 2 min, RUNNING_A como fração), na inferência, no
monitor de drift, no diagnóstico de entrada e no `constroi_bundle`. Não é candidato:
não muda o detector; garante que produção veja o dado que a validação viu.

O QUE TEM DE VALER -- cada item é um assert:
  1. grade       `preparar_grade` sobre o export de 30 s == `grade2min.parquet`, coluna a
                 coluna, instante a instante (a grade em que tudo foi treinado e medido).
  2. regressão   quem já entrega a grade de 2 min recebe o MESMO alarme e o MESMO monitor
                 de antes: o pacote corrigido contra a versão anterior (`ANTES`, lida do
                 git), sobre o CSV do próprio pacote.
  3. inferência  o pacote sobre o export de 30 s dá o alarme da grade da pesquisa bit a
                 bit -- 0,344 FP/mês, 48,9 h/mês, 21 episódios na composição publicada.
  4. bundle      `constroi_bundle` a partir do export de 30 s gera os mesmos artefatos
                 que a partir do parquet, e que a versão anterior (sha256 dos JSON).
E O QUE NÃO SE RESOLVE, impresso: uma grade de 2 min montada errado (uma leitura por
janela, `paridade_entrada.amostrada`) continua passando -- já está em 2 min e o pacote
não tem como saber como ela foi feita. Por isso o export de 30 s vira a entrada
preferida.

RESULTADO (02/10/2026) -- PARIDADE OK NOS QUATRO.
  1. grade       612.630 instantes x 38 colunas, 0 células diferentes, mesmo índice.
  2. regressão   alarme idêntico sobre o CSV de 2 min do pacote (49.049 instantes em
                 alarme), monitor de drift e diagnóstico de entrada idênticos.
  3. inferência  a partir do export de 30 s: det 8, início 6, banda 5, 0,344 FP/mês,
                 48,9 h/mês, 21 episódios -- o publicado; alarme idêntico instante a
                 instante ao da grade da pesquisa.
  4. bundle      2026-03 (baseline 31/01-28/02/2026): os 6 JSON com o mesmo sha256 a
                 partir do export de 30 s, do parquet e na versão anterior.
  · O que não se resolve: a grade de 2 min montada com uma leitura por janela segue
    dando 0,603 FP/mês e 67,2 h/mês sem aviso. O contrato de entrada deve passar a
    ser o export de 30 s.
  · O export de 30 s traz texto do PI no meio das colunas numéricas (o pandas avisa
    `DtypeWarning` em quase todas); `preparar_grade` o converte em NaN, como o treino.

Uso:  PYTHONPATH=. python paridade_pacote.py
"""
from __future__ import annotations
import io, contextlib, hashlib, importlib.util, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    import regua_fp as R
    import paridade_entrada as PE

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
PKG = RAIZ / "estrutura_pra_prod" / "Cabiunas"
CI = PE.CI                                   # o pacote corrigido (scripts/ no sys.path)
ANTES = "7aec11e"                            # o último commit antes da correção
PY = sys.executable


def antigo(nome: str, pasta: Path):
    """O módulo do pacote como estava em ANTES, gravado em `pasta` e importado."""
    src = subprocess.run(["git", "show", f"{ANTES}:estrutura_pra_prod/Cabiunas/scripts/{nome}.py"],
                         cwd=RAIZ, capture_output=True, text=True, check=True).stdout
    f = pasta / f"{nome}_antes.py"
    f.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"{nome}_antes", f)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m, f


def alarme(ci, df: pd.DataFrame) -> pd.Series:
    modelos = ci.carregar_modelos(PKG / "modelos")
    trips = ci.carregar_trips(PKG / "registro_trips.csv")
    return ci.prever(modelos, ci.preprocessar(modelos, df, trips=trips))["is_anomaly"]


def regua(fin: pd.Series) -> dict:
    m = R.mede(fin.reindex(R.idx, fill_value=False).astype(bool) & R.sel)
    m.pop("cls")
    return m


def sha_json(d: Path) -> dict:
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in sorted(d.glob("*.json"))}


def bundle(script: Path, historico: Path, saida: Path) -> Path:
    r = subprocess.run([PY, str(script), "--historico", str(historico), "--mes", "2026-03",
                        "--saida", str(saida), "--trips", str(PKG / "registro_trips.csv")],
                       cwd=script.parent, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    (d,) = list(saida.glob("model_*"))
    return d


def main():
    tmp = Path(tempfile.mkdtemp(prefix="paridade_pacote_"))
    ci_antes, _ = antigo("cabiunas_inference", tmp)
    _, cb_antes = antigo("constroi_bundle", tmp)

    print("1) GRADE: preparar_grade(export de 30 s) contra grade2min.parquet", flush=True)
    bruto = CI.carregar_dados(PE.SRC)
    g30 = CI.preparar_grade(bruto)
    gp = pd.read_parquet(AQUI / "grade2min.parquet")
    cols = [c for c in gp.columns if c in g30.columns]
    a, b = g30.reindex(gp.index)[cols], gp[cols]
    difere = ~((a == b) | (a.isna() & b.isna()))
    print(f"   {len(gp)} instantes x {len(cols)} colunas; células diferentes: {int(difere.values.sum())}; "
          f"índices iguais: {g30.index.equals(gp.index)}", flush=True)
    assert g30.index.equals(gp.index) and int(difere.values.sum()) == 0

    print("\n2) REGRESSÃO: CSV de 2 min do pacote, versão corrigida contra a anterior", flush=True)
    csv = CI.achar_csv(PKG / "dados")
    df2 = CI.carregar_dados(csv)
    novo, velho = alarme(CI, df2), alarme(ci_antes, df2)
    modelos = CI.carregar_modelos(PKG / "modelos")
    mon_n, mon_v = CI.monitor_drift(modelos, df2), ci_antes.monitor_drift(modelos, df2)
    dia_n, dia_v = CI.diagnostico_entrada(modelos, df2), ci_antes.diagnostico_entrada(modelos, df2)
    iguais = bool(novo.equals(velho))
    print(f"   alarme idêntico: {iguais} ({int(novo.sum())} instantes em alarme) | monitor de drift "
          f"idêntico: {mon_n == mon_v} | diagnóstico de entrada idêntico: {dia_n == dia_v}", flush=True)
    assert iguais and mon_n == mon_v and dia_n == dia_v

    print("\n3) INFERÊNCIA: o pacote sobre o export de 30 s, 2024-11-01 a 2026-04-30", flush=True)
    j = (bruto.index >= PE.INI) & (bruto.index < PE.FIM + pd.Timedelta("2min"))
    f30 = alarme(CI, bruto.loc[j])
    fpq = alarme(CI, PE.pesquisa())
    m30, mpq = regua(f30), regua(fpq)
    for nome, m in (("grade da pesquisa", mpq), ("export de 30 s", m30)):
        print(f"   {nome:18s} det {m['det']} início {m['inicio']} banda {m['banda']} "
              f"FP/mês {m['fp_mes']:.3f} carga {m['carga_mes']:.1f} h/mês episódios {m['episodios']}",
              flush=True)
    same = bool(f30.reindex(fpq.index).fillna(False).astype(bool).equals(fpq.astype(bool)))
    print(f"   alarme idêntico instante a instante: {same}", flush=True)
    assert same and round(m30["fp_mes"], 3) == 0.344 and m30["episodios"] == 21

    print("\n4) BUNDLE de 2026-03: constroi_bundle a partir do export de 30 s e do parquet", flush=True)
    script = PKG / "scripts" / "constroi_bundle.py"
    b30 = bundle(script, PE.SRC, tmp / "b30")
    bpq = bundle(script, AQUI / "grade2min.parquet", tmp / "bpq")
    (tmp / "scripts_antes").mkdir()
    cb = tmp / "scripts_antes" / "constroi_bundle.py"
    cb.write_text(cb_antes.read_text(encoding="utf-8"), encoding="utf-8")
    bav = bundle(cb, AQUI / "grade2min.parquet", tmp / "bav")
    h30, hpq, hav = sha_json(b30), sha_json(bpq), sha_json(bav)
    for k in hpq:
        print(f"   {k:28s} 30 s {h30.get(k)}  parquet {hpq[k]}  antes {hav.get(k)}", flush=True)
    print(f"   mesma pasta de baseline: {b30.name == bpq.name == bav.name} ({bpq.name})", flush=True)
    assert h30 == hpq == hav and b30.name == bpq.name == bav.name

    print("\nO QUE NÃO SE RESOLVE: grade de 2 min montada errado (uma leitura por janela)", flush=True)
    m = regua(alarme(CI, PE.amostrada(True)))
    print(f"   det {m['det']} FP/mês {m['fp_mes']:.3f} carga {m['carga_mes']:.1f} h/mês -- passa sem aviso; "
          f"entregar o export de 30 s", flush=True)
    print("\nPARIDADE OK", flush=True)


if __name__ == "__main__":
    main()
