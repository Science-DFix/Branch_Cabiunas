#!/usr/bin/env python3
"""Mede, num worker do ClearML, quanto leva cada etapa do PACOTE DE PRODUCAO do TC-33003A.

POR QUE EXISTE. Os tempos que a engenharia recebeu (retreino ~4 s, inferencia ~1,6 s com 2 min;
~8 s e ~4 s com o export de 30 s) foram medidos num notebook. Este script mede o mesmo codigo
num worker do ClearML, etapa por etapa, e publica o resultado.

O QUE MEDE (4 cenarios, 3 execucoes cada, cada uma num processo novo -- tempo e pico de RAM
limpos):
  inferencia, grade de 2 min, 90 d     inferencia, export de 30 s, 90 d
  retreino,   grade de 2 min, 484 d    retreino,   export de 30 s, 240 d
Por cenario, a mediana de cada etapa: leitura, diagnostico, grade, mascara, PCA de t e p, vb
(referencia de 400 h), canais (EWMA, degrau, CUSUM), voto/pos-processamento, monitor de drift;
no retreino, a leitura/grade, o ajuste de cada familia (RobustScaler + PCA + p99) e o resto.
Quem cronometra sao wrappers em volta das funcoes do pacote -- o pacote nao e alterado.

AUTOCONTIDO. O worker nao tem o repositorio (a branch nem foi publicada e `dados/` esta no
.gitignore), entao o pacote e os dados vao num Dataset do ClearML, como em `roda_clearml.py`:
scripts/, modelos/, registro_trips.csv e quatro CSV (2 min inteiro e 90 d; 30 s com 90 d e 240 d,
cortados do export da pesquisa).

Uso:  python mede_tempos_clearml.py --local     # monta numa pasta temporaria e mede aqui
      python mede_tempos_clearml.py --remote    # sobe o Dataset, enfileira em `default` e sai
"""
from __future__ import annotations
import json, os, platform, resource, shutil, subprocess, sys, tempfile, time, warnings
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PKG = AQUI.parents[1] / "estrutura_pra_prod" / "Cabiunas"
SRC30 = AQUI.parents[3] / "dados" / "sensores_full_2024_2026_30s.csv"
PROJETO, FILA = "TesteMLCab", "default"
N_EXEC = 3
CENARIOS = {                    # rotulo -> (tipo, arquivo, o que e)
    "inferência, 2 min, 90 d":  ("inf", "2min_90d.csv"),
    "inferência, 30 s, 90 d":   ("inf", "30s_90d.csv"),
    "retreino, 2 min, 484 d":   ("ret", "2min_484d.csv"),
    "retreino, 30 s, 240 d":    ("ret", "30s_240d.csv"),
}


# ══════════════════════════════════════════════════ o dataset
def _cauda(src: Path, n: int, dest: Path) -> None:
    with open(dest, "wb") as out:             # direto para o arquivo: nada passa pela memoria do Python
        subprocess.run(["head", "-n", "1", str(src)], stdout=out, check=True)
        subprocess.run(["tail", "-n", str(n), str(src)], stdout=out, check=True)


def monta(dest: Path) -> Path:
    """O pacote e os quatro CSV, numa pasta."""
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PKG / "scripts", dest / "scripts", ignore=shutil.ignore_patterns("_pdf", "__pycache__"))
    shutil.copytree(PKG / "modelos", dest / "modelos")
    shutil.copy(PKG / "registro_trips.csv", dest / "registro_trips.csv")
    d = dest / "dados"; d.mkdir()
    csv2 = next((PKG / "dados" / "2025_2026").glob("data_*_raw.csv"))
    shutil.copy(csv2, d / "2min_484d.csv")
    _cauda(csv2, 64_800, d / "2min_90d.csv")           # 90 d x 720 por dia
    _cauda(SRC30, 259_200, d / "30s_90d.csv")          # 90 d x 2.880 por dia
    _cauda(SRC30, 691_200, d / "30s_240d.csv")         # 240 d x 2.880 por dia
    return dest


def pico_ram_mb() -> float:
    """VmHWM, o pico deste processo. `ru_maxrss` de um filho herda o pico do pai e mentiria."""
    for l in open("/proc/self/status"):
        if l.startswith("VmHWM:"):
            return int(l.split()[1]) / 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ══════════════════════════════════════════════════ um cenario, num processo novo
def cenario(tipo: str, arq: str, raiz: Path, saida_json: Path) -> None:
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(raiz / "scripts"))
    import cabiunas_inference as ci
    import constroi_bundle as cb

    acum: dict[str, float] = {}

    def envolve(mod, nome, rotulo):
        f = getattr(mod, nome)

        def g(*a, **k):
            t0 = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                acum[rotulo] = acum.get(rotulo, 0.0) + time.perf_counter() - t0
        setattr(mod, nome, g)

    envolve(ci, "_grade", "grade de 2 min (preparar + regrade)")
    envolve(ci, "_mascara", "máscara de operação")
    envolve(ci, "_recon_pca", "PCA de t e p (reconstrução)")
    envolve(ci, "_z_vibracao", "vb: referência rolante de 400 h")
    envolve(ci, "_acende", "canais: EWMA já feita; degrau + CUSUM")
    envolve(ci, "_cusum", "   CUSUM (dentro dos canais)")
    envolve(ci, "_episodios", "episódios")
    envolve(cb, "_ler_historico", "leitura do CSV + grade (preparar_grade)")
    envolve(cb, "ajusta_familia", "ajuste de uma família (RobustScaler + PCA + p99)")

    etapas: list[tuple[str, float, bool]] = []          # (rotulo, segundos, e_subetapa)
    arq_ = raiz / "dados" / arq
    t_ini = time.perf_counter()

    def etapa(rot, f):
        antes = dict(acum)
        t0 = time.perf_counter(); r = f(); dt = time.perf_counter() - t0
        etapas.append((rot, dt, False))
        for k, v in acum.items():
            if v - antes.get(k, 0.0) > 1e-4 and rot.startswith(("4", "5")):
                etapas.append((f"   ↳ {k}", v - antes.get(k, 0.0), True))
        return r

    extra = {}
    if tipo == "inf":
        df = etapa("1 leitura do CSV", lambda: ci.carregar_dados(arq_))
        modelos = etapa("2 carregar bundles e registro de trips", lambda: ci.carregar_modelos(raiz / "modelos"))
        trips = ci.carregar_trips(raiz / "registro_trips.csv")
        etapa("3 diagnóstico da entrada", lambda: ci.diagnostico_entrada(modelos, df))
        proc = etapa("4 preprocessar (grade, máscara, PCA t/p, sp, vb, EWMA)", lambda: ci.preprocessar(modelos, df, trips=trips))
        res = etapa("5 prever (canais, voto, escalada, refratário, duração)", lambda: ci.prever(modelos, proc))
        etapa("6 resumo de episódios", lambda: ci.resumo_episodios(res))
        etapa("7 monitor de drift", lambda: ci.monitor_drift(modelos, df))
        extra = dict(linhas=int(len(df)), instantes_em_alarme=int(res["is_anomaly"].sum()))
    else:
        saida = Path(tempfile.mkdtemp())
        argv0 = sys.argv
        sys.argv = ["constroi_bundle.py", "--historico", str(arq_), "--mes", "2026-03",
                    "--saida", str(saida), "--trips", str(raiz / "registro_trips.csv")]
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            rc = cb.main()
        sys.argv = argv0
        total_ret = time.perf_counter() - t_ini
        ler = acum.get("leitura do CSV + grade (preparar_grade)", 0.0)
        aj = acum.get("ajuste de uma família (RobustScaler + PCA + p99)", 0.0)
        etapas += [("1 leitura do CSV + grade (preparar_grade)", ler, False),
                   ("2 ajuste das 2 famílias, t e p (RobustScaler + PCA + p99)", aj, False),
                   ("3 resto (baseline estável, spread do mancal, escrita dos JSON)", total_ret - ler - aj, False)]
        extra = dict(codigo_saida=int(rc), arquivos_gerados=len(list(saida.rglob("*.json"))))
        shutil.rmtree(saida, ignore_errors=True)

    total = time.perf_counter() - t_ini
    saida_json.write_text(json.dumps(dict(
        etapas=etapas, total_s=total, pico_ram_mb=pico_ram_mb(),
        **extra)), encoding="utf-8")


# ══════════════════════════════════════════════════ o orquestrador
def sistema() -> dict:
    sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
    import numpy, pandas, sklearn
    return dict(maquina=sh("lscpu | grep 'Model name' | sed 's/.*: *//'") or platform.processor(),
                nucleos_logicos=os.cpu_count(), ram_total=sh("free -h | awk '/Mem:/ {print $2}'"),
                so=platform.platform(), python=platform.python_version(),
                numpy=numpy.__version__, pandas=pandas.__version__, scikit_learn=sklearn.__version__,
                host=platform.node())


def mede(raiz: Path) -> tuple["pd.DataFrame", "pd.DataFrame", dict]:
    import numpy as np, pandas as pd
    linhas, totais = [], []
    for rotulo, (tipo, arq) in CENARIOS.items():
        runs = []
        for i in range(N_EXEC):
            j = Path(tempfile.mkdtemp()) / "r.json"
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--cenario", tipo, arq,
                                str(raiz), str(j)], capture_output=True, text=True)
            if r.returncode != 0 or not j.exists():
                raise RuntimeError(f"{rotulo} execução {i + 1} falhou:\n{r.stderr[-1500:]}")
            runs.append(json.loads(j.read_text(encoding="utf-8")))
            print(f"  {rotulo} #{i + 1}: {runs[-1]['total_s']:.1f} s, pico {runs[-1]['pico_ram_mb']:.0f} MB", flush=True)
        nomes = [e[0] for e in runs[0]["etapas"]]
        for k, nome in enumerate(nomes):
            v = [r["etapas"][k][1] for r in runs if k < len(r["etapas"]) and r["etapas"][k][0] == nome]
            linhas.append(dict(cenario=rotulo, etapa=nome, mediana_s=round(float(np.median(v)), 3),
                               execucao_1=round(v[0], 3), execucao_2=round(v[1], 3), execucao_3=round(v[2], 3)))
        t = [r["total_s"] for r in runs]; m = [r["pico_ram_mb"] for r in runs]
        totais.append(dict(cenario=rotulo, total_mediana_s=round(float(np.median(t)), 2),
                           execucoes_s=" / ".join(f"{x:.2f}" for x in t),
                           pico_ram_mb=round(float(np.max(m))), **{k: runs[0][k] for k in runs[0]
                           if k in ("linhas", "instantes_em_alarme", "codigo_saida", "arquivos_gerados")}))
    return pd.DataFrame(linhas), pd.DataFrame(totais), sistema()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--cenario":
        _, _, tipo, arq, raiz, j = sys.argv
        return cenario(tipo, arq, Path(raiz), Path(j))

    remoto = "--remote" in sys.argv
    from clearml import Dataset, Task
    ds_id = ""
    if Task.running_locally() and remoto:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=AQUI, capture_output=True,
                                text=True).stdout.strip()
        suja = bool(subprocess.run(["git", "status", "--porcelain", "--", str(PKG)], cwd=AQUI,
                                   capture_output=True, text=True).stdout.strip())
        pasta = monta(Path(tempfile.mkdtemp(prefix="pacote_tempos_")))
        ds = Dataset.create(dataset_name=f"TC33003A_pacote_producao_tempos_{commit}", dataset_project=PROJETO)
        ds.add_files(str(pasta)); ds.upload(); ds.finalize()
        ds_id = ds.id
        print("dataset:", ds_id, "commit", commit, "(pacote com alteração não commitada)" if suja else "")
    task = Task.init(project_name=PROJETO, task_name="pacote-producao::tempos_por_etapa",
                     task_type=Task.TaskTypes.testing, reuse_last_task_id=False, auto_connect_frameworks=False,
                     tags=["pacote-producao", "tempos"])
    cfg = task.connect(dict(dataset_id=ds_id, n_execucoes=N_EXEC,
                            observacao="pacote de producao (estrutura_pra_prod/Cabiunas), nao o codigo de pesquisa"))
    if Task.running_locally() and remoto:
        # as versoes que o requirements.txt do pacote pede (o notebook de desenvolvimento nao as segue)
        task.set_packages(["numpy>=2.4,<3", "pandas>=3.0,<4", "scikit-learn>=1.8,<1.9", "clearml"])
        task.execute_remotely(queue_name=FILA, exit_process=True)
        return

    if Task.running_locally():                       # --local: monta e mede aqui
        raiz = monta(Path(tempfile.mkdtemp(prefix="pacote_tempos_")))
    else:                                            # no worker
        raiz = Path(Dataset.get(dataset_id=cfg["dataset_id"]).get_local_copy())
    tab, tot, sis = mede(raiz)
    lg = task.get_logger()
    lg.report_table("tempos", "por etapa (mediana de 3 execuções)", iteration=0, table_plot=tab)
    lg.report_table("tempos", "totais e pico de RAM", iteration=0, table_plot=tot)
    for r in tot.itertuples():
        lg.report_single_value(f"{r.cenario} | total (s)", float(r.total_mediana_s))
        lg.report_single_value(f"{r.cenario} | pico RAM (MB)", float(r.pico_ram_mb))
    lg.report_text(json.dumps(sis, indent=2, ensure_ascii=False))
    task.upload_artifact("sistema", sis)
    task.upload_artifact("por_etapa", tab)
    task.upload_artifact("totais", tot)
    task.flush(wait_for_uploads=True)
    print(json.dumps(sis, indent=2, ensure_ascii=False))
    print(tot.to_string(index=False))
    print(tab.to_string(index=False))
    task.close()


if __name__ == "__main__":
    main()
