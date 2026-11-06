#!/usr/bin/env python3
"""J5 -- quanto histórico o retreino mensal do v2 precisa LER: bundle do histórico inteiro x só os últimos N dias.

PRÉ-REGISTRO (06/11/2026, commitado antes de enviar)
Origem: na reunião com a engenharia apareceu a queixa do "volume de linhas" no retreino. O `constroi_bundle.py` lê o
  arquivo de histórico INTEIRO e usa só os últimos 20.000 pontos estáveis antes do corte (dia 1 do mês). Os 16 bundles
  do git têm baseline com span de 27 a 64 d e folga de 0 a 3 d; os alertas do script disparam em span > 70 ou
  folga > 7. Logo 77 d (= 70 + 7) cobririam, por construção, todo bundle dentro da faixa validada.
Pergunta: o bundle gerado lendo só [corte − N d, corte) é IDÊNTICO ao gerado do histórico inteiro? Qual o menor N que
  dá idêntico em todos os meses, e quanto cai o volume lido?

INSUMOS: os do J1 (pacote de produção, Dataset f5e1b16b; bruto de 30 s, Dataset a97ba56b, com o complemento de
  set–dez/2024 como no J1). Bruto convertido a número uma vez (o `preparar_grade` faz o mesmo `to_numeric`).
PROCEDIMENTO. Para cada mês M de 2025-01 até o último com corte <= fim do dado (+1 d): roda o `main()` do
  `constroi_bundle.py` do pacote, sem alteração, com `--historico` = parquet do bruto de 30 s e `--trips` = registro do
  pacote, para N = inteiro (todo o bruto, como hoje), 77, 60, 45 e 30 d. Fatias começam à 0 h (a grade de 2 min do
  `resample` fica alinhada igual). Por (M, N): código de saída, sha256 de cada arquivo gerado, linhas e células do bruto
  lido, tempo do `main()`. Do inteiro: span, folga e a janela mínima necessária = ceil(corte − 1º ponto do baseline).
  IDÊNTICO = mesmo código de saída e mesmo sha256 em todos os arquivos.
VERIFICAÇÃO V1 (relato, não bloqueia): o bundle "inteiro" contra o do git do mesmo mês -- datas do baseline e máx |Δ|
  dos números de `*_transformacao.json`, `normalizacao.json` e `spread_mancal.json` (o git ainda traz os .pkl e um
  `detector.json` mais antigo, por isso não há sha256 aqui; o histórico do Thallys pode diferir em 2024).
EXPECTATIVAS: 77 d idêntico em todos os meses (~95%); 60 d idêntico em todos (~25%: o 2025-07 pede 65 d); 45 d e 30 d
  falham em algum mês com ERRO de pontos insuficientes, não com bundle diferente (~85%); janela mínima máxima entre 60
  e 70 d (~70%); volume lido com 77 d <= 20% do inteiro no último mês (~80%); V1 com datas do baseline iguais ao git
  desde 2025-03 (~70%).
LIMITES: 17 meses de um equipamento; parada longa (> 70 d de span) não ocorreu no período, e nela 77 d dá ERRO onde o
  inteiro dá bundle com alerta -- a regra proposta para esse caso (estender a leitura até ter os 20.000 pontos) é
  argumento, não medida.

RESULTADO (06/11/2026 17:20; task 4feb79c4, commit 20d8856, 1 min no Cica, 16 meses 2025-01..2026-04). Bruto inteiro =
  1.748.160 linhas de 30 s (68 M células). O teste local não coube em 7 GB de RAM; a análise foi conferida com
  registros sintéticos antes do envio.
  Bundle idêntico (sha256 de todos os arquivos) ao do inteiro: 77 d 16/16 · 60 d 15/16 (falha 2025-07) ·
    45 d 12/16 · 30 d 6/16. TODA falha é rc 1 (pontos insuficientes); nenhuma janela gerou bundle diferente.
  Janela mínima necessária por mês: 28 a 65 d (máx 2025-07); span 27–64 d, folga 0–3 d.
  Volume lido com 77 d: 221.760 linhas = 12,7% do inteiro hoje, e fixo (o inteiro cresce ~86 mil linhas/mês);
    tempo do main() 0,35 s contra 2,3 s (mediana).
  V1: os 16 bundles do inteiro batem com os do git -- mesmas datas de baseline, |Δ| = 0 nos números de todos os JSON.
PLACAR: ACERTEI 77 d 16/16; 45/30 d falham por ERRO; janela mínima máx entre 60 e 70 d (65); volume <= 20% (12,7%);
  V1 desde 2025-03 (16/16). ERREI (dei 25%) 60 d em todos -- cai em 2025-07.
LEITURA: o retreino só precisa ler [dia 1 − 77 d, dia 1) e o resultado é idêntico; numa parada longa, estender a
  leitura até ter os 20.000 pontos (argumento; não ocorreu no período).

Uso:
  python frente_thallys/j5_retreino_janela.py --local --teste    # encanamento local, 2 meses
  python frente_thallys/j5_retreino_janela.py --enqueue          # uma task na fila default
"""
from clearml import Task  # noqa: F401  (antes do argparse: o worker injeta argumentos por patch)
import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from multiprocessing import get_context
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import j1_janela_entrada as j1   # noqa: E402

NOME = "frente_thallys | J5 retreino: janela de leitura do bundle"
JANELAS = [None, 77, 60, 45, 30]
SAIDA = Path(tempfile.gettempdir()) / "j5_saida"
G = j1.G


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def roda(arg):
    """Um `constroi_bundle.main()` para (mês, N); devolve o registro."""
    import pandas as pd
    import constroi_bundle as cb
    mes, n = arg
    corte = pd.Timestamp(mes + "-01", tz="UTC")
    tmp = Path(tempfile.mkdtemp(prefix=f"j5_{mes}_{n}_"))
    if n is None:
        hist = G["parquet_inteiro"]; linhas, celulas = G["n_inteiro"]
    else:
        w = G["raw"].loc[(G["raw"].index >= corte - pd.Timedelta(days=n)) & (G["raw"].index < corte)]
        hist = tmp / "hist.parquet"; w.to_parquet(hist); linhas, celulas = len(w), int(w.size)
    argv = sys.argv
    sys.argv = ["constroi_bundle.py", "--historico", str(hist), "--mes", mes, "--saida", str(tmp / "modelos"),
                "--trips", str(G["pk"] / "registro_trips.csv")]
    out = io.StringIO(); t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(out):
            rc = cb.main()
    finally:
        sys.argv = argv
    dt = time.perf_counter() - t0
    d = next((tmp / "modelos").glob("model_*"), None) if (tmp / "modelos").exists() else None
    shas = {f.name: _sha(f) for f in sorted(d.iterdir())} if d else {}
    reg = dict(mes=mes, n=("inteiro" if n is None else n), rc=rc, linhas=linhas, celulas=celulas, s=round(dt, 2),
               sha=shas, pasta=(d.name if d else None), log=out.getvalue()[-600:])
    if d is not None:
        m = json.loads((d / "modelo.json").read_text())
        ini = pd.Timestamp(m["baseline_inicio"])
        reg.update(span=m["baseline_span_dias"], folga=m["baseline_folga_dias"],
                   janela_min_d=math.ceil((corte - ini) / pd.Timedelta(days=1)))
        if n is None:   # guarda o bundle do inteiro para o V1
            dest = SAIDA / "bundles" / mes; dest.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                (dest / f.name).write_bytes(f.read_bytes())
    return reg


def _nums(x):
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        yield float(x)
    elif isinstance(x, dict):
        for k in sorted(x):
            yield from _nums(x[k])
    elif isinstance(x, list):
        for v in x:
            yield from _nums(v)


def v1(mes: str):
    """Bundle 'inteiro' deste mês contra o do git: datas do baseline e máx |Δ| dos números."""
    git = next((d for d in (G["pk"] / "modelos").iterdir()
                if json.loads((d / "modelo.json").read_text()).get("mes_servido") == mes), None)
    nosso = SAIDA / "bundles" / mes
    if git is None or not nosso.exists():
        return dict(mes=mes, git=None)
    mg, mn = (json.loads((p / "modelo.json").read_text()) for p in (git, nosso))
    r = dict(mes=mes, git=git.name, datas_iguais=(mg["baseline_inicio"], mg["baseline_fim"]) ==
             (mn["baseline_inicio"], mn["baseline_fim"]))
    for f in ("temperatura_transformacao.json", "pressao_transformacao.json", "normalizacao.json", "spread_mancal.json"):
        a, b = (list(_nums(json.loads((p / f).read_text()))) for p in (git, nosso))
        r[f] = (max(abs(x - y) for x, y in zip(a, b)) if len(a) == len(b) else f"tamanhos {len(a)} x {len(b)}")
    return r


def analisa(regs):
    import pandas as pd
    df = pd.DataFrame(regs)
    inteiro = df[df.n == "inteiro"].set_index("mes")
    res = dict(meses=sorted(inteiro.index), por_n={})
    for n in JANELAS[1:]:
        x = df[df.n == n].set_index("mes")
        ident = [m for m in x.index if x.loc[m, "rc"] == inteiro.loc[m, "rc"] and x.loc[m, "sha"] == inteiro.loc[m, "sha"]]
        res["por_n"][n] = dict(
            identicos=f"{len(ident)}/{len(x)}",
            diferentes=sorted(set(x.index) - set(ident)),
            rc_diferente={m: int(x.loc[m, "rc"]) for m in x.index if x.loc[m, "rc"] != inteiro.loc[m, "rc"]},
            linhas_ultimo_mes=int(x.loc[x.index.max(), "linhas"]),
            frac_linhas_ultimo_mes=round(x.loc[x.index.max(), "linhas"] / inteiro.loc[x.index.max(), "linhas"], 4),
            s_mediana=float(x.s.median()))
    res["inteiro"] = dict(linhas_ultimo_mes=int(inteiro.loc[inteiro.index.max(), "linhas"]),
                          celulas_ultimo_mes=int(inteiro.loc[inteiro.index.max(), "celulas"]),
                          s_mediana=float(inteiro.s.median()), rc=inteiro.rc.astype(int).to_dict(),
                          span=inteiro.span.to_dict(), folga=inteiro.folga.to_dict(),
                          janela_min_d=inteiro.janela_min_d.to_dict(),
                          janela_min_max=int(inteiro.janela_min_d.max()))
    ok = [n for n in JANELAS[1:] if not res["por_n"][n]["diferentes"]]
    res["menor_n_identico"] = min(ok) if ok else None
    return res


def executa(local: bool, insumos: str | None, teste: bool, procs: int):
    import pandas as pd
    SAIDA.mkdir(parents=True, exist_ok=True); t00 = time.time()
    pk, a97, _ = j1.monta(local, insumos)
    j1.prepara(pk, a97, False)
    raw = G["raw"].apply(pd.to_numeric, errors="coerce")
    G.update(raw=raw, pk=pk)
    G["parquet_inteiro"] = SAIDA / "inteiro.parquet"; raw.to_parquet(G["parquet_inteiro"])
    G["n_inteiro"] = (len(raw), int(raw.size))
    fim = G["df"].index[-1]
    meses = [m.strftime("%Y-%m") for m in pd.date_range(pd.Timestamp("2025-01-01", tz="UTC"), fim + pd.Timedelta(days=1), freq="MS")]
    if teste:
        meses = ["2025-04", "2025-07"]
    res = dict(ambiente=j1.ambiente(), bruto=dict(ini=str(raw.index[0]), fim=str(raw.index[-1]), linhas=len(raw),
                                                  colunas=raw.shape[1], fim_real=str(fim)))
    print("ambiente:", res["ambiente"], "| bruto:", res["bruto"], "| meses:", meses, flush=True)
    trabalhos = [(m, n) for m in meses for n in JANELAS]
    with get_context("fork").Pool(min(procs, len(trabalhos))) as pool:
        regs = list(pool.imap_unordered(roda, trabalhos))
    regs.sort(key=lambda r: (r["mes"], str(r["n"])))
    for r in regs:
        print(f"   {r['mes']} N={str(r['n']):8s} rc {r['rc']} linhas {r['linhas']:>9} {r['s']:6.1f} s "
              f"span {r.get('span')} folga {r.get('folga')} jmin {r.get('janela_min_d')}", flush=True)
    res["analise"] = analisa(regs); print(json.dumps(res["analise"], indent=1, default=str), flush=True)
    res["V1"] = [v1(m) for m in meses]; print("V1:", json.dumps(res["V1"], indent=1, default=str), flush=True)
    res["registros"] = [{k: v for k, v in r.items() if k != "log"} for r in regs]
    res["logs_erro"] = {f"{r['mes']}_{r['n']}": r["log"] for r in regs if r["rc"] not in (0,)}
    res["duracao_min"] = round((time.time() - t00) / 60, 1)
    (SAIDA / "j5_resultado.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    from clearml import Task as Tk
    tk = Tk.current_task()
    if tk is not None:
        tk.upload_artifact("j5_resultado", artifact_object=str(SAIDA / "j5_resultado.json"))
        tk.upload_artifact("bundles_inteiro", artifact_object=str(SAIDA / "bundles"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enqueue", action="store_true")
    ap.add_argument("--local", action="store_true"); ap.add_argument("--teste", action="store_true")
    ap.add_argument("--insumos", default=j1.DS_INSUMOS); ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    if a.enqueue:
        t = Task.init(project_name="TesteMLCab", task_name=NOME)
        t.set_base_docker("python:3.11-slim", docker_setup_bash_script=["apt-get update -qq", "apt-get install -y -qq libgomp1"])
        t.set_packages(j1.PACOTES); t.set_parameters({"General/insumos": a.insumos, "General/procs": str(a.procs)})
        # o padrão do ClearML pega o remoto `origin` (SIMPred), onde este commit não existe: fixa o Branch_Cabiunas
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=AQUI, text=True).strip()
        t.set_script(repository="git@github.com:Science-DFix/Branch_Cabiunas.git", branch="exp/ocsvm-validacao", commit=commit)
        t.execute_remotely(queue_name="default")
    if os.environ.get("CLEARML_TASK_ID"):
        tk = Task.init(project_name="TesteMLCab", task_name=NOME)
        a.insumos = tk.get_parameter("General/insumos") or a.insumos
        a.procs = int(tk.get_parameter("General/procs") or a.procs)
    executa(a.local, a.insumos, a.teste, a.procs)


if __name__ == "__main__":
    main()
