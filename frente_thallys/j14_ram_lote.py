#!/usr/bin/env python3
"""J14 -- de onde vem a RAM do lote de 90 d em 2026 (~805 MB contra ~497 MB em 2025)? E rodar com 1 thread muda algo?

PRÉ-REGISTRO (14/11/2026, commitado antes de enviar)
Origem: na J11 (task 8b7268f0) o lote de 90 d deu RAM pico ~497 MB nos instantes de 2025 e ~805 MB nos de 2026, com
  CSV do mesmo tamanho (~106 MB). O Thallys rodou 90 d a 30 s em quatro datas de 2025 e 2026 na consolidada e deu
  499-501 MB em todas: o código não explica. Ele sugere comparar df.dtypes e memory_usage(deep=True) logo depois do
  carregar_dados. Ele também mediu que a álgebra linear com 1 thread dá saída idêntica byte a byte e 21% menos CPU.
Perguntas:
  A. Em que etapa do lote a memória de 2026 se separa da de 2025? Os dtypes do CSV mudam entre os anos?
  B. Com 1 thread (OMP/OPENBLAS/MKL_NUM_THREADS=1) a saída do exemplo é idêntica (sha256) e a CPU cai?
Desenho: os mesmos 10 instantes da J11 (semente 20261105, desde 2025-04-01), CSV de 90 d do export de 30 s cortado como
  na J11. Por instante:
  - sonda (`j14_sonda_ram.py`, processo novo): os passos do `tc33003a_exemplo.py` (e17bf27) com RSS e pico depois de
    cada etapa, dtypes, memória profunda e colunas object (com exemplos de texto) após carregar_dados e após
    preparar_grade;
  - o exemplo inteiro com o medidor de telemetria (como J11), com as threads padrão e com 1 thread; sha256 do
    `tc33003a_inferencia.csv` de cada execução.
EXPECTATIVAS: reproduz a J11 (pico 2025 ~500 MB, 2026 ~800 MB) (~90%). A diferença já aparece no carregar_dados,
  com colunas object (texto) no CSV de 2026 (~60%); se não, aparece no preprocessar ou prever (~25%).
  1 thread: saída idêntica nos 10 (~85%) e CPU menor (~70%).
CRITÉRIO: a etapa em que o pico de 2026 passa o de 2025 em > 100 MB é a origem. Se for o carregar_dados com colunas
  object, a diferença é do CSV, não do pacote, e o orçamento do documento passa a usar o pico de 2025 com essa
  ressalva.
LIMITES: o CSV é o nosso export (sensores_full_2024_2026_30s.csv), não o da consolidada; Cica, não o SIMPred.

RESULTADO (task 760f35b1, Cica, 1,8 min; os mesmos 10 instantes da J11)
  Telemetria do exemplo inteiro: pico 495-497 MB nos 4 instantes de 2025 e 795-808 MB nos 6 de 2026 (= J11).
  Após carregar_dados:  2025: 37 float64 + 1 int64 + 1 bool, 81 MB (memória profunda); RSS ~235 MB.
                        2026: 38 object + 1 bool, 336-383 MB; RSS ~565 MB.
    As colunas viram object por 11 a 14 células de texto do PI na janela de 90 d ("{'Name': 'Comm Fail', 'Value':
    313, 'IsSystem': True}" e "Out of Serv"). Após preparar_grade os dois anos ficam iguais: 39 float32, 10,6 MB.
    Daí em diante o RSS não sobe mais de 30 MB em nenhuma etapa.
  1 thread: saída idêntica (sha256) em 10/10; CPU 6,7 -> 2,2 s (mediana); duração igual (2,26 s); RAM igual.
  (O "pico por etapa" da sonda saiu 5.255 MB em tudo: ru_maxrss herdado do pai pelo exec, o mesmo erro do teste
  local da J34; vale o RSS atual por etapa e o pico da telemetria.)
PLACAR: reproduz a J11 SIM; diferença no carregar_dados por colunas object (texto) SIM; 1 thread idêntica SIM;
  CPU menor SIM (-67%, o Thallys mediu -21%).
LEITURA: os ~300 MB a mais de 2026 vêm do CSV, não do pacote: um punhado de estados do sistema do PI em texto faz o
  pandas ler as 38 colunas como object, e o pico acontece no read_csv, antes do preparar_grade (que já transforma texto
  em NaN). O Thallys não vê isso porque o CSV dele não tem essas células. Como o export de produção do PI pode trazê-las,
  o orçamento deve usar ~800 MB, ou a leitura deve converter para número já no read_csv (mudança no pacote: decisão do
  Thallys). Com 1 thread a CPU cai a um terço sem mudar a saída nem o tempo.
  _(registrado em 14/11/2026 13:41)_

Uso:
  python frente_thallys/j14_ram_lote.py --enqueue         # uma task na fila default
"""
from clearml import Task  # noqa: F401  (antes do argparse: o worker injeta argumentos por patch)
import argparse
import hashlib
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
import j11_telemetria_e_sem_vb as j11   # noqa: E402

NOME = "frente_thallys | J14 RAM do lote 2025 x 2026 + 1 thread"
SAIDA = Path(tempfile.gettempdir()) / "j14_saida"
G = j1.G
UMA = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def env_limpo(extra=None):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("CLEARML_", "TRAINS_"))}
    e.update(extra or {})
    return e


def instantes(teste):
    """Os mesmos da J11 (monta_estados): mesma semente, mesmo candidato."""
    import numpy as np
    import pandas as pd
    df = G["df"]
    rng = np.random.default_rng(j1.SEMENTE)
    cand = df.loc[pd.Timestamp("2025-04-01", tz="UTC"):df.index[-1] - pd.Timedelta("1D")].index
    return sorted(cand[i] for i in rng.choice(len(cand), size=2 if teste else j11.N_INST, replace=False))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def executa(local: bool, insumos: str | None, teste: bool):
    import pandas as pd
    SAIDA.mkdir(parents=True, exist_ok=True); t00 = time.time()
    pk, a97, _ = j1.monta(local, insumos)
    res = dict(ambiente=j1.ambiente()); print("ambiente:", res["ambiente"], flush=True)
    j1.prepara(pk, a97, teste)
    inst = instantes(teste); G.clear()
    print("instantes:", [str(t) for t in inst], flush=True)
    exemplo, scripts, modelos = pk / "scripts" / "tc33003a_exemplo.py", pk / "scripts", pk / "modelos"
    trips_csv, saida_ex = pk / "registro_trips.csv", (pk / "scripts").resolve() / "tc33003a_inferencia.csv"
    log = SAIDA / "telemetria.jsonl"; L = []
    for k, fim in enumerate(inst):
        pa = SAIDA / f"i{k:02d}"; pa.mkdir(exist_ok=True)
        j11.recorta_csv(a97, [dict(fim=str(fim), pasta=str(pa))])
        for n in (3, 60):
            (pa / f"bruto{n}.csv").unlink()
        csv = pa / "bruto90.csv"; x = dict(fim=str(fim), ano=fim.year, csv_mb=round(csv.stat().st_size / 1e6, 1))
        o = subprocess.run([sys.executable, str(AQUI / "j14_sonda_ram.py"), str(csv), str(scripts), str(modelos),
                            str(trips_csv), str(pa / "sonda.json")], capture_output=True, text=True, env=env_limpo())
        x["sonda"] = json.loads((pa / "sonda.json").read_text()) if o.returncode == 0 else dict(erro=o.stderr[-800:])
        for rot, extra in (("padrao", None), ("1thread", UMA)):
            o = subprocess.run([sys.executable, str(j11.TELEM), "--rotulo", f"{rot}|{fim}", "--log", str(log), "--ok",
                                "0", "--", sys.executable, str(exemplo), "--csv", str(csv), "--dias", "1",
                                "--ignorar-validade"], capture_output=True, text=True, env=env_limpo(extra))
            x[f"sha_{rot}"] = sha(saida_ex) if o.returncode == 0 else f"erro {o.returncode}: {o.stderr[-300:]}"
        csv.unlink()
        s = x["sonda"]
        if "etapas" in s:
            print(f"   {fim} ({x['csv_mb']} MB de CSV): " + " | ".join(f"{e['etapa']} {e['pico_mb']:.0f}"
                  for e in s["etapas"]), flush=True)
            print(f"      após carregar: {s['apos_carregar']['dtypes']} mem {s['apos_carregar']['mem_profunda_mb']} MB, "
                  f"object {s['apos_carregar']['colunas_object'][:6]} {s['apos_carregar']['texto_nas_object']}", flush=True)
            print(f"      após grade: {s['apos_grade']['dtypes']} mem {s['apos_grade']['mem_profunda_mb']} MB", flush=True)
        else:
            print(f"   {fim}: sonda falhou: {s['erro']}", flush=True)
        print(f"      sha padrão {x['sha_padrao']} | 1 thread {x['sha_1thread']}", flush=True)
        L.append(x)
    R = pd.DataFrame([json.loads(l) for l in log.read_text().splitlines()])
    R["modo"] = R.rotulo.str.split("|").str[0]; R["fim"] = R.rotulo.str.split("|").str[1]
    R["cpu_s"] = R.cpu_usuario_s + R.cpu_sistema_s
    R.to_csv(SAIDA / "telemetria.csv", index=False)
    for modo, g in R.groupby("modo", sort=False):
        print(f"   {modo}: duração {g.duracao_s.median():.2f} s | CPU {g.cpu_s.median():.1f} s | RAM pico "
              f"{g.ram_pico_maior_processo_mb.median():.0f} MB ({g.ram_pico_maior_processo_mb.min():.0f}-"
              f"{g.ram_pico_maior_processo_mb.max():.0f})", flush=True)
    res["instantes"] = L
    res["sha_identico"] = sum(x["sha_padrao"] == x["sha_1thread"] for x in L)
    print(f"   saída idêntica com 1 thread: {res['sha_identico']}/{len(L)}", flush=True)
    res["duracao_min"] = round((time.time() - t00) / 60, 1)
    (SAIDA / "j14_resultado.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    from clearml import Task as Tk
    tk = Tk.current_task()
    if tk is not None:
        tk.upload_artifact("j14_resultado", artifact_object=str(SAIDA / "j14_resultado.json"))
        tk.upload_artifact("j14_telemetria", artifact_object=str(SAIDA / "telemetria.csv"))


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
