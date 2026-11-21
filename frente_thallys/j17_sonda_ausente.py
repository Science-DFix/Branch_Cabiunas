#!/usr/bin/env python3
"""J17 -- o motor com estado quebra quando uma sonda de vibração falta no CSV? A correção iguala o lote?

PRÉ-REGISTRO (20/11/2026, commitado antes de enviar)
Origem: o Thallys pôs o motor no pacote (feat/pacote-respostas-engenharia, e733279) e, no a020477, fez o
  `preprocessar` tratar tag ausente do CSV como coluna vazia (o canal degrada em vez de KeyError). Ele concluiu, só
  lendo o código, que o motor ainda quebra nesse caso. Lendo também: `MotorEstado._estaveis` seleciona
  `g.loc[..., self.tags]` direto da grade, fora do `preprocessar`, então a sonda ausente dá KeyError já na `carga`.
Tratamento: `_estaveis` passa a usar `.reindex(columns=self.tags)` (sonda ausente = coluna vazia, como no pacote).
  Única mudança no motor; nada do detector muda.
Desenho: pacote = scripts da e733279 (Dataset criado por --cria-pacote) + modelos e registro do f5e1b16b (modelos
  idênticos entre e17bf27 e e733279). CSV = export de 30 s desde 2025-01-01 (como no J15), em três versões:
  completo; sem TV_351X_A; sem as 10 TV_. Comparação = `teste_equivalencia.py` (motor × lote de 60 d a cada 6 h,
  igualdade exata), período inteiro, 8 processos na referência.
    A1  motor ATUAL (cópia do pacote e733279/scripts) × sem TV_351X_A
    A2  motor CORRIGIDO × sem TV_351X_A
    A3  motor CORRIGIDO × sem as 10 TV_
    A4  motor CORRIGIDO × completo (regressão do J15)
EXPECTATIVAS: A1 cai com KeyError (~90%); A2 IGUAL (~80%); A3 IGUAL (~70%; sem vb o caminho é menos exercitado);
  A4 IGUAL (~95%; a mudança é nula com as colunas todas).
CRITÉRIO: A1 quebra e A2, A3 e A4 IGUAIS -> a correção entra no motor e vai ao Thallys para puxar no pacote.
  A1 não quebra -> a leitura de código estava errada; desfazer a correção. A2/A3/A4 diferente -> a correção não sai
  até achar a diferença. Se o próprio lote falhar sem as sondas, registrar: é do pacote (a020477), não do motor.

RESULTADO (task 896d034c, Cica, 13,6 min; pacote e733279). 1.708 atualizações de 02/03/2025 a 30/04/2026,
  300.960 linhas comparadas desde 09/03/2025:
    A1 atual × sem TV_351X_A ...... QUEBRA: KeyError "['TV_351X_A'] not in index" (na carga).
    A2 corrigido × sem TV_351X_A .. IGUAL: 0 linhas diferentes em t, p, sp, vb, máscara e alarme; estado 2,84 MB.
    A3 corrigido × sem as 10 TV_ .. IGUAL (vb vazio nos dois); estado 1,28 MB, passo 0,06 s.
    A4 corrigido × completo ....... IGUAL (= J15: |Δ| máx t 5,3e-15, p 8,9e-15; estado 2,99 MB, passo 0,116 s).
  O lote do a020477 rodou nos três CSVs sem erro.
PLACAR: A1 quebra SIM; A2 SIM; A3 SIM; A4 SIM -> as 4 expectativas certas.
LEITURA: a leitura de código do Thallys estava certa; a correção (1 linha em `_estaveis`) fecha o caso e não muda
  nada com as colunas completas. Critério cumprido: o motor_estado.py corrigido vai ao Thallys para puxar no pacote
  (a branch dele não é alterada por nós).
  _(registrado em 21/11/2026)_

Uso:
  python frente_thallys/j17_sonda_ausente.py --cria-pacote   # uma vez, local: Dataset com os scripts da e733279
  python frente_thallys/j17_sonda_ausente.py --enqueue
"""
from clearml import Task  # noqa: F401  (antes do argparse: o worker injeta argumentos por patch)
import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import j1_janela_entrada as j1      # noqa: E402

NOME = "frente_thallys | J17 sonda de vibração ausente no motor"
DS_SCRIPTS = "bed2e62f62914fa087af7e0a28c9e296"   # Dataset com Cabiunas/scripts da e733279 (--cria-pacote)
COMMIT_PACOTE = "e733279"
REPO_PACOTE = Path.home() / "FRENTE_THALLYS" / "repro_thallys_regra_carga_9fp_4fp"
SAIDA = Path(tempfile.gettempdir()) / "j17_saida"
TESTE_NOSSO = AQUI / "motor_estado" / "teste_equivalencia.py"
TV = ["TV_351X_A", "TV_351Y_A", "TV_352X_A", "TV_352Y_A", "TV_353X_A", "TV_353Y_A", "TV_354X_A", "TV_354Y_A",
      "TV_355X_A", "TV_355Y_A"]


def cria_pacote():
    from clearml import Dataset
    tmp = Path(tempfile.mkdtemp(prefix="j17_pacote_"))
    tar = tmp / "p.tar"
    subprocess.run(["git", "-C", str(REPO_PACOTE), "archive", "-o", str(tar), COMMIT_PACOTE,
                    "estrutura_pra_prod/Cabiunas/scripts"], check=True)
    with tarfile.open(tar) as t:
        t.extractall(tmp, filter="data")
    src = tmp / "estrutura_pra_prod" / "Cabiunas" / "scripts"
    ds = Dataset.create(dataset_name=f"frente_thallys J17 scripts do pacote ({COMMIT_PACOTE}, "
                                     "feat/pacote-respostas-engenharia)", dataset_project="TesteMLCab")
    ds.add_files(str(src), dataset_path="scripts", wildcard="*.py", recursive=False)
    ds.upload(); ds.finalize(); print("scripts J17:", ds.id, flush=True)


def sem_colunas(csv: Path, tirar: list[str], destino: Path) -> Path:
    """Cópia do CSV sem as colunas `tirar` (módulo csv: o texto do PI tem vírgula entre aspas)."""
    import csv as _csv
    with open(csv, newline="") as f, open(destino, "w", newline="") as g:
        r, w = _csv.reader(f), _csv.writer(g)
        cab = next(r)
        fica = [i for i, c in enumerate(cab) if c not in tirar]
        assert len(fica) == len(cab) - len(tirar), "coluna a tirar não está no CSV"
        w.writerow([cab[i] for i in fica])
        for v in r:
            w.writerow([v[i] for i in fica])
    return destino


def roda(teste: Path, pk: Path, csv: Path, rot: str, env: dict) -> dict:
    js = SAIDA / f"{rot}.json"; t0 = time.time()
    o = subprocess.run([sys.executable, str(teste), "--pacote", str(pk), "--csv", str(csv), "--procs", "8",
                        "--saida", str(js)], capture_output=True, text=True, env=env)
    print(f"== {rot} (código {o.returncode}, {time.time() - t0:.0f} s)\n{o.stdout[-2500:]}", flush=True)
    if o.stderr.strip():
        print(o.stderr[-2500:], flush=True)
    r = json.loads(js.read_text()) if js.exists() else {}
    r["codigo"] = o.returncode
    if o.returncode not in (0, 1) or not js.exists():
        err = o.stderr.strip().splitlines()
        r["erro"] = err[-1] if err else "sem stderr"
    return r


def executa(local: bool, insumos: str | None, scripts: str):
    SAIDA.mkdir(parents=True, exist_ok=True); t00 = time.time()
    pk, _, _ = j1.monta(local, insumos)
    if not local:
        from clearml import Dataset
        novo = Path(Dataset.get(dataset_id=scripts).get_local_copy()) / "scripts"
    else:
        novo = Path(os.environ["FT_SCRIPTS_J17"])
    (pk / "scripts").unlink(); (pk / "scripts").symlink_to(novo)
    csv = next((pk / "dados" / "2025_2026").glob("data_*_raw.csv"))
    sem1 = sem_colunas(csv, ["TV_351X_A"], SAIDA / "sem_tv351x.csv")
    sem10 = sem_colunas(csv, TV, SAIDA / "sem_tv.csv")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CLEARML_", "TRAINS_"))}
    teste_atual = pk / "scripts" / "teste_equivalencia.py"   # usa o motor_estado.py do pacote (sem a correção)
    res = dict(ambiente=j1.ambiente(), pacote=COMMIT_PACOTE)
    res["A1_atual_sem1"] = roda(teste_atual, pk, sem1, "A1_atual_sem1", env)
    res["A2_corrigido_sem1"] = roda(TESTE_NOSSO, pk, sem1, "A2_corrigido_sem1", env)
    res["A3_corrigido_sem10"] = roda(TESTE_NOSSO, pk, sem10, "A3_corrigido_sem10", env)
    res["A4_corrigido_completo"] = roda(TESTE_NOSSO, pk, csv, "A4_corrigido_completo", env)
    res["duracao_min"] = round((time.time() - t00) / 60, 1)
    (SAIDA / "j17_resultado.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str), flush=True)
    from clearml import Task as Tk
    tk = Tk.current_task()
    if tk is not None:
        tk.upload_artifact("j17_resultado", artifact_object=str(SAIDA / "j17_resultado.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enqueue", action="store_true"); ap.add_argument("--local", action="store_true")
    ap.add_argument("--cria-pacote", action="store_true")
    ap.add_argument("--insumos", default=j1.DS_INSUMOS); ap.add_argument("--scripts", default=DS_SCRIPTS)
    a = ap.parse_args()
    if a.cria_pacote:
        return cria_pacote()
    if a.enqueue:
        t = Task.init(project_name="TesteMLCab", task_name=NOME)
        t.set_base_docker("python:3.11-slim", docker_setup_bash_script=["apt-get update -qq", "apt-get install -y -qq libgomp1"])
        t.set_packages(j1.PACOTES); t.set_parameters({"General/insumos": a.insumos, "General/scripts": a.scripts})
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=AQUI, text=True).strip()
        t.set_script(repository="git@github.com:Science-DFix/Branch_Cabiunas.git", branch="exp/ocsvm-validacao", commit=commit)
        t.execute_remotely(queue_name="default")
    if os.environ.get("CLEARML_TASK_ID"):
        tk = Task.init(project_name="TesteMLCab", task_name=NOME)
        a.insumos = tk.get_parameter("General/insumos") or a.insumos
        a.scripts = tk.get_parameter("General/scripts") or a.scripts
    executa(a.local, a.insumos, a.scripts)


if __name__ == "__main__":
    main()
