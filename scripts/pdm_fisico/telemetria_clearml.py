#!/usr/bin/env python3
"""Roda o pacote de producao do TC-33003A sob `telemetria_execucao.py` num worker do ClearML e publica o consumo.

Mede, por cenario (inferencia e retreino, grade de 2 min e export de 30 s; 3 execucoes cada), CPU, RAM, disco,
espaco temporario, duracao, codigos de saida. Usa o Dataset ja subido (pacote + CSV) mais um Dataset-filho com o wrapper.
Uso:  python telemetria_clearml.py --remote --dataset ID_PAI     # cria o filho, enfileira em `default` e sai
"""
from __future__ import annotations
import json, statistics as st, subprocess, sys, tempfile
from pathlib import Path

PROJETO, FILA, IMAGEM, N = "TesteMLCab", "default", "python:3.12-slim", 3
CEN = [("inferência, 2 min, 90 d", "inf", "2min_90d"), ("inferência, 30 s, 90 d", "inf", "30s_90d"),
       ("retreino, 2 min, 484 d", "ret", "2min_484d"), ("retreino, 30 s, 240 d", "ret", "30s_240d"),
       ("retreino do zero, 2 min, 16 meses", "zero", "2min_484d"), ("retreino do zero, 30 s, 16 meses", "zero", "30s_240d")]
MESES = ("2025-01", "2026-04")   # os 16 meses servidos pelos bundles versionados em modelos/


def _numeros(o) -> list[float]:
    if isinstance(o, dict):
        return [x for k in sorted(o) for x in _numeros(o[k])]
    if isinstance(o, list):
        return [x for v in o for x in _numeros(v)]
    return [float(o)] if isinstance(o, (int, float)) and not isinstance(o, bool) else []


def compara(gerado: Path, versionado: Path) -> list[dict]:
    """Bundle a bundle, a maior diferenca absoluta entre os numeros dos JSON gerados do zero e os versionados."""
    out = []
    for d in sorted(p for p in gerado.iterdir() if p.is_dir()):
        ref = versionado / d.name
        lin = dict(bundle=d.name, existe_versionado=ref.is_dir(), json_comparados=0, max_dif_abs=None, json_diferente="")
        if ref.is_dir():
            difs = []
            for j in sorted(d.glob("*.json")):
                if not (ref / j.name).exists():
                    continue
                a, b = _numeros(json.loads(j.read_text())), _numeros(json.loads((ref / j.name).read_text()))
                lin["json_comparados"] += 1
                if len(a) != len(b):
                    lin["json_diferente"] += f"{j.name}(n {len(a)}x{len(b)}) "
                    continue
                difs.append(max((abs(x - y) for x, y in zip(a, b)), default=0.0))
            lin["max_dif_abs"] = max(difs, default=None)
        out.append(lin)
    return out


def mede(raiz: Path, so_zero: bool = False) -> tuple[list[dict], dict, dict]:
    log = Path(tempfile.mkdtemp()) / "t.jsonl"; zero = {}
    for rot, tipo, c in CEN:
        if not (raiz / f"dados/{c}.csv").exists() or (so_zero and tipo != "zero"):
            continue
        for i in range(N):
            saida = Path(tempfile.mkdtemp())
            if tipo == "inf":
                cmd = [sys.executable, str(raiz / "scripts/tc33003a_exemplo.py"), "--csv", str(raiz / f"dados/{c}.csv"),
                       "--modelos", str(raiz / "modelos"), "--dias", "60", "--json", str(saida / "estado.json")]
                extra = ["--ok", "0,2,3", "--json-estado", str(saida / "estado.json")]
            elif tipo == "zero":
                cmd = [sys.executable, str(raiz / "retreino_do_zero.py"), "--constroi", str(raiz / "scripts/constroi_bundle.py"),
                       "--historico", str(raiz / f"dados/{c}.csv"), "--trips", str(raiz / "registro_trips.csv"),
                       "--saida", str(saida), "--de", MESES[0], "--ate", MESES[1]]
                extra = ["--saidas", str(saida)]; zero[rot] = saida
            else:
                cmd = [sys.executable, str(raiz / "scripts/constroi_bundle.py"), "--historico", str(raiz / f"dados/{c}.csv"),
                       "--mes", "2026-03", "--saida", str(saida), "--trips", str(raiz / "registro_trips.csv")]
                extra = ["--saidas", str(saida)]
            r = subprocess.run([sys.executable, str(raiz / "telemetria_execucao.py"), "--rotulo", rot, "--log", str(log), *extra, "--", *cmd],
                               capture_output=True, text=True)
            print(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode, flush=True)
    linhas = [json.loads(l) for l in log.read_text().splitlines()]
    return linhas, linhas[0]["sistema"], zero


def main():
    from clearml import Dataset, Task
    ds_id = sys.argv[sys.argv.index("--dataset") + 1] if "--dataset" in sys.argv else ""
    remoto = "--remote" in sys.argv
    pacote = sys.argv[sys.argv.index("--pacote") + 1] if "--pacote" in sys.argv else ""   # pasta pronta (ex.: a do Drive)
    commit = ""
    if Task.running_locally() and not pacote:   # no worker o script chega avulso: nao ha repositorio
        PKG = Path(__file__).resolve().parents[2] / "estrutura_pra_prod" / "Cabiunas"   # o pacote deste commit
        git = lambda *a: subprocess.run(["git", "-C", str(PKG), *a], capture_output=True, text=True)
        commit = git("rev-parse", "--short", "HEAD").stdout.strip() + ("" if git("diff", "--quiet", "HEAD", "--", ".").returncode == 0 else "+mod")
    nome = "pacote-producao::telemetria_de_recursos" + ("_drive" if pacote else f"_{commit}")
    if Task.running_locally() and remoto:
        if pacote:
            f = Dataset.create(dataset_name="TC33003A_pacote_drive_telemetria", dataset_project=PROJETO)
            f.add_files(pacote)
        else:
            pai = Dataset.get(dataset_id=ds_id)   # o pai traz os CSV; scripts/ e modelos/ vem do commit atual, nao do pai
            f = Dataset.create(dataset_name=f"TC33003A_pacote_producao_telemetria_{commit}", dataset_project=PROJETO, parent_datasets=[pai.id])
            for arq in sorted(PKG.glob("scripts/*.*")):
                f.add_files(str(arq), dataset_path="scripts")
            f.add_files(str(PKG / "modelos"), dataset_path="modelos")
        for arq in ("telemetria_execucao.py", "retreino_do_zero.py"):
            f.add_files(str(Path(__file__).resolve().parent / arq))
        f.upload(); f.finalize()
        ds_id = f.id; print("dataset filho:", ds_id)
    Task.force_store_standalone_script(True)
    task = Task.init(project_name=PROJETO, task_name=nome, task_type=Task.TaskTypes.testing,
                     reuse_last_task_id=False, auto_connect_frameworks=False, tags=["pacote-producao", "telemetria"])
    cfg = task.connect(dict(dataset_id=ds_id, n_execucoes=N, commit_pacote=commit, so_zero="--so-zero" in sys.argv))
    if Task.running_locally() and remoto:
        task.set_packages(["numpy>=2.4,<3", "pandas>=3.0,<4", "scikit-learn>=1.8,<1.9", "clearml"]); task.set_base_docker(IMAGEM)
        task.execute_remotely(queue_name=FILA, exit_process=True); return
    raiz = Path(Dataset.get(dataset_id=cfg["dataset_id"]).get_local_copy())
    so_zero = str(cfg["so_zero"]).lower() == "true"
    linhas, sis, zero = mede(raiz, so_zero)
    import pandas as pd
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("sistema", "atraso", "saidas_mb", "comando")} for r in linhas])
    med = df.groupby("rotulo", sort=False).median(numeric_only=True).round(2).reset_index()
    lg = task.get_logger()
    lg.report_table("telemetria", "medianas por cenário", iteration=0, table_plot=med)
    lg.report_table("telemetria", "todas as execuções", iteration=0, table_plot=df)
    lg.report_text(json.dumps(sis, indent=2, ensure_ascii=False))
    task.upload_artifact("sistema", sis); task.upload_artifact("telemetria_medianas", med); task.upload_artifact("telemetria_execucoes", df)
    for rot, saida in zero.items():   # a ultima repeticao de cada cenario do zero: meses e paridade com modelos/
        meses = pd.DataFrame(json.loads((saida / "meses.json").read_text()))
        par = pd.DataFrame(compara(saida, raiz / "modelos"))
        lg.report_table("retreino do zero", f"meses -- {rot}", iteration=0, table_plot=meses)
        lg.report_table("retreino do zero", f"paridade com modelos/ -- {rot}", iteration=0, table_plot=par)
        task.upload_artifact(f"meses {rot}", meses); task.upload_artifact(f"paridade {rot}", par)
        print(rot); print(meses.to_string()); print(par.to_string())
    task.flush(wait_for_uploads=True)
    print(json.dumps(sis, ensure_ascii=False)); print(med.T.to_string()); print("codigos:", df.groupby("rotulo", sort=False).codigo_saida.apply(list).to_dict())
    task.close()


if __name__ == "__main__":
    main()
