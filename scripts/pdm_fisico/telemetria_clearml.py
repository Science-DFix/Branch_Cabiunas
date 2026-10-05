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
       ("retreino, 2 min, 484 d", "ret", "2min_484d"), ("retreino, 30 s, 240 d", "ret", "30s_240d")]


def mede(raiz: Path) -> tuple[list[dict], dict]:
    log = Path(tempfile.mkdtemp()) / "t.jsonl"; out = []
    for rot, tipo, c in CEN:
        for i in range(N):
            saida = Path(tempfile.mkdtemp())
            if tipo == "inf":
                cmd = [sys.executable, str(raiz / "scripts/tc33003a_exemplo.py"), "--csv", str(raiz / f"dados/{c}.csv"),
                       "--modelos", str(raiz / "modelos"), "--dias", "60", "--json", str(saida / "estado.json")]
                extra = ["--ok", "0,2,3", "--json-estado", str(saida / "estado.json")]
            else:
                cmd = [sys.executable, str(raiz / "scripts/constroi_bundle.py"), "--historico", str(raiz / f"dados/{c}.csv"),
                       "--mes", "2026-03", "--saida", str(saida), "--trips", str(raiz / "registro_trips.csv")]
                extra = ["--saidas", str(saida)]
            r = subprocess.run([sys.executable, str(raiz / "telemetria_execucao.py"), "--rotulo", rot, "--log", str(log), *extra, "--", *cmd],
                               capture_output=True, text=True)
            print(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode, flush=True)
    linhas = [json.loads(l) for l in log.read_text().splitlines()]
    return linhas, linhas[0]["sistema"]


def main():
    from clearml import Dataset, Task
    ds_id = sys.argv[sys.argv.index("--dataset") + 1] if "--dataset" in sys.argv else ""
    remoto = "--remote" in sys.argv
    if Task.running_locally() and remoto:
        pai = Dataset.get(dataset_id=ds_id)
        f = Dataset.create(dataset_name="TC33003A_pacote_producao_telemetria", dataset_project=PROJETO, parent_datasets=[pai.id])
        f.add_files(str(Path(__file__).resolve().parent / "telemetria_execucao.py")); f.upload(); f.finalize()
        ds_id = f.id; print("dataset filho:", ds_id)
    Task.force_store_standalone_script(True)
    task = Task.init(project_name=PROJETO, task_name="pacote-producao::telemetria_de_recursos", task_type=Task.TaskTypes.testing,
                     reuse_last_task_id=False, auto_connect_frameworks=False, tags=["pacote-producao", "telemetria"])
    cfg = task.connect(dict(dataset_id=ds_id, n_execucoes=N))
    if Task.running_locally() and remoto:
        task.set_packages(["numpy>=2.4,<3", "pandas>=3.0,<4", "scikit-learn>=1.8,<1.9", "clearml"]); task.set_base_docker(IMAGEM)
        task.execute_remotely(queue_name=FILA, exit_process=True); return
    raiz = Path(Dataset.get(dataset_id=cfg["dataset_id"]).get_local_copy())
    linhas, sis = mede(raiz)
    import pandas as pd
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("sistema", "atraso", "saidas_mb", "comando")} for r in linhas])
    med = df.groupby("rotulo", sort=False).median(numeric_only=True).round(2).reset_index()
    lg = task.get_logger()
    lg.report_table("telemetria", "medianas por cenário", iteration=0, table_plot=med)
    lg.report_table("telemetria", "todas as execuções", iteration=0, table_plot=df)
    lg.report_text(json.dumps(sis, indent=2, ensure_ascii=False))
    task.upload_artifact("sistema", sis); task.upload_artifact("telemetria_medianas", med); task.upload_artifact("telemetria_execucoes", df)
    task.flush(wait_for_uploads=True)
    print(json.dumps(sis, ensure_ascii=False)); print(med.T.to_string()); print("codigos:", df.groupby("rotulo", sort=False).codigo_saida.apply(list).to_dict())
    task.close()


if __name__ == "__main__":
    main()
