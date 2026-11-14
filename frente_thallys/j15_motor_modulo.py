#!/usr/bin/env python3
"""J15 -- o motor com estado como módulo separado (`motor_estado/`) reproduz o lote de 60 d com igualdade exata?

PRÉ-REGISTRO (14/11/2026, commitado antes de enviar)
Origem: o Thallys conferiu na consolidada tudo o que não depende do motor com estado, mas não conseguiu conferir J34,
  J6-J10, o estado de 3,3 MB e os tempos, porque o motor vivia espalhado nos scripts de teste (j34_estado_robustez.py,
  com o estado global G). Ele propôs a forma de entrega: funções de referência num módulo separado, com o teste de
  equivalência, sem mexer no `cabiunas_inference.py`; tolerância: igualdade exata, comparando no mesmo ambiente.
O que muda: `motor_estado/motor_estado.py` é a classe Motor do J34 sem o G (recebe ci e modelos), com a troca do
  `_z_vibracao` desfeita ao sair de cada chamada; `motor_estado/teste_equivalencia.py` roda sozinho com
  --pacote/--csv (para o Thallys rodar na consolidada). Lógica idêntica à do J34; nada do detector muda.
Desenho: CSV de produção montado por j1.monta (export de 30 s desde 2025-01-01; sem o trecho de 2024 que o J34
  acrescentava), pacote e17bf27 com os 16 bundles. Atualizações a cada 6 h + no instante de cada trip, desde início
  + 60 d; referência = lote de 60 d em cada atualização com os trips já ocorridos; compara desde a 1a atualização
  + 7 d. Duas corridas: motor normal e motor com reinício (grava e relê o estado) em toda atualização.
EXPECTATIVAS: normal igual (sinais |Δ| <= 1e-9, máscara e alarme) (~85%; o J34 limpo foi bit a bit com a mesma
  lógica); reinício igual (~90%); estado em disco entre 2,5 e 4 MB (~80%); passo mediano <= 0,3 s (~70%).
CRITÉRIO: as duas corridas IGUAL -> o módulo é a entrega de referência e vai ao Thallys com o commit. Se não, o
  módulo não sai até achar a diferença (o J34 continua valendo como estava).
LIMITES: só o cenário limpo e o reinício; buracos, sondas, atrasos e trips tardios (J34) e as recuperações (J6-J9)
  não são refeitos aqui: ficam para o Thallys rodar, ou para um teste seguinte, sobre o módulo.

Uso:
  python frente_thallys/j15_motor_modulo.py --enqueue         # uma task na fila default
"""
from clearml import Task  # noqa: F401  (antes do argparse: o worker injeta argumentos por patch)
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import j1_janela_entrada as j1      # noqa: E402

NOME = "frente_thallys | J15 motor com estado como módulo (equivalência)"
SAIDA = Path(tempfile.gettempdir()) / "j15_saida"
TESTE = AQUI / "motor_estado" / "teste_equivalencia.py"


def executa(local: bool, insumos: str | None, teste: bool):
    SAIDA.mkdir(parents=True, exist_ok=True); t00 = time.time()
    pk, _, _ = j1.monta(local, insumos)
    res = dict(ambiente=j1.ambiente()); print("ambiente:", res["ambiente"], flush=True)
    csv = next((pk / "dados" / "2025_2026").glob("data_*_raw.csv"))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CLEARML_", "TRAINS_"))}
    extra = ["--fim", "2025-05-15"] if teste else []
    for rot, flag in (("normal", []), ("reinicio", ["--reinicio"])):
        js = SAIDA / f"{rot}.json"
        o = subprocess.run([sys.executable, str(TESTE), "--pacote", str(pk), "--csv", str(csv), "--procs", "8",
                            "--saida", str(js)] + flag + extra, capture_output=True, text=True, env=env)
        print(f"== {rot} (código {o.returncode})\n{o.stdout[-3000:]}", flush=True)
        if o.stderr.strip():
            print(o.stderr[-2000:], flush=True)
        res[rot] = json.loads(js.read_text()) if js.exists() else dict(erro=o.stderr[-800:])
        res[rot]["codigo"] = o.returncode
    res["duracao_min"] = round((time.time() - t00) / 60, 1)
    (SAIDA / "j15_resultado.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    from clearml import Task as Tk
    tk = Tk.current_task()
    if tk is not None:
        tk.upload_artifact("j15_resultado", artifact_object=str(SAIDA / "j15_resultado.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enqueue", action="store_true")
    ap.add_argument("--local", action="store_true"); ap.add_argument("--teste", action="store_true")
    ap.add_argument("--insumos", default=j1.DS_INSUMOS)
    a = ap.parse_args()
    if a.enqueue:
        t = Task.init(project_name="TesteMLCab", task_name=NOME)
        t.set_base_docker("python:3.11-slim", docker_setup_bash_script=["apt-get update -qq", "apt-get install -y -qq libgomp1"])
        t.set_packages(j1.PACOTES); t.set_parameters({"General/insumos": a.insumos})
        # o padrão do ClearML pega o remoto `origin` (SIMPred), onde este commit não existe: fixa o Branch_Cabiunas
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=AQUI, text=True).strip()
        t.set_script(repository="git@github.com:Science-DFix/Branch_Cabiunas.git", branch="exp/ocsvm-validacao", commit=commit)
        t.execute_remotely(queue_name="default")
    if os.environ.get("CLEARML_TASK_ID"):
        tk = Task.init(project_name="TesteMLCab", task_name=NOME)
        a.insumos = tk.get_parameter("General/insumos") or a.insumos
    executa(a.local, a.insumos, a.teste)


if __name__ == "__main__":
    main()
