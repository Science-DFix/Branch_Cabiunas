#!/usr/bin/env python3
"""J11 -- (A) custo do lote e do motor com o medidor de telemetria do Thallys; (B) detecção sem o canal de vibração.

PRÉ-REGISTRO (11/11/2026, commitado antes de enviar)
Origem: a segunda avaliação do Thallys (11/11) pede (ponto 5) que o custo da resposta à engenharia seja medido com o
  MESMO método da telemetria da branch (`scripts/pdm_fisico/telemetria_execucao.py`, exp/reducao-falsos-positivos
  cc6edc9; copiado sem alteração em `terceiros/`, sha256 dac2f9f35ac22d1c...) e citado numa tabela só, reconciliando
  com os 167 MB e 5,4 s do relatório de 07/10. O J10 mediu só as partes do detector (sem leitura do CSV, sem
  diagnóstico/drift/contrato) e o bruto em objetos do Python, o que não se compara com a telemetria dele.
  Ele também afirma (ponto 3) que "sem o vb a detecção cai para 1 de 8"; nunca medimos.
Perguntas: (A) quanto custa, medido pelo medidor dele, a execução do pacote com 60 e 90 d de export de 30 s e um
  passo do motor? (B) com o vb vazio em todos os instantes, quantos trips o detector pega e com quantos FP?

PARTE A. 10 instantes sorteados (semente 20261105) desde 2025-04-01. Para cada um:
  - lote_60, lote_90: `telemetria_execucao.py -- python tc33003a_exemplo.py --csv <N d de export de 30 s, no formato
    do CSV de produção> --dias 1 --ignorar-validade` (o comando do pacote, inteiro: leitura do CSV, preparar_grade uma
    vez, bundles, diagnóstico, drift, preprocessar, prever, saída);
  - motor: `telemetria_execucao.py -- python j11_motor_cmd.py` (CSV de 3 d do export, preparar_grade, bundles,
    estado de disco de 45 d + buffer, um passo), sem o clearml importado.
  O CSV termina na última janela de 2 min fechada. O estado do motor é montado antes, fora da medida (60 d, como J34).
  Medidas do medidor: duração, CPU (usuário + sistema), pico de RAM do maior processo (rusage), bytes lidos.
  Mediana e p95 nos 10 instantes.
PARTE B. Controle publicado (`sinais_causais` + `prever`, = reproduz.controle: 7/8, 9 FP) contra o mesmo com
  vb = NaN em todas as linhas, régua do Thallys (nv.avalia, trips desde 27/02/2025, período desde 01/02/2025).
EXPECTATIVAS: (A) lote_90 entre 3 e 8 s e RAM pico entre 150 e 400 MB (~70%; o relatório de 07/10 dá 5,4 s e
  167 MB); lote_60 abaixo de lote_90 nos dois (~90%); motor <= metade do tempo de lote_60 (~75%) e RAM <= lote_60
  (~75%). (B) controle 7/8 FP 9 (~95%); sem vb <= 3/8 (~70%; o Thallys diz 1/8).
LIMITES: Cica, não o servidor do SIMPred; leitura do banco fora (o CSV vem do disco local); o motor importa o
  módulo da frente (j34), que é maior que um motor de produção.

Uso:
  python frente_thallys/j11_telemetria_e_sem_vb.py --enqueue         # uma task na fila default
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

NOME = "frente_thallys | J11 telemetria do pacote + sem vb"
N_INST = 10
SAIDA = Path(tempfile.gettempdir()) / "j11_saida"
TELEM = AQUI / "terceiros" / "telemetria_execucao.py"
G = j1.G


def monta_estados(teste: bool):
    """Subprocesso: sorteia os instantes e grava o estado do motor de cada um (fora da medida)."""
    import numpy as np
    import pandas as pd
    import j34_estado_robustez as J
    ci = G["ci"]; G["orig_z"] = ci._z_vibracao; ci._z_vibracao = J._z_com_buffer
    df = G["df"]
    rng = np.random.default_rng(j1.SEMENTE)
    cand = df.loc[pd.Timestamp("2025-04-01", tz="UTC"):df.index[-1] - pd.Timedelta("1D")].index
    inst = sorted(cand[i] for i in rng.choice(len(cand), size=2 if teste else N_INST, replace=False))
    L = []
    for k, fim in enumerate(inst):
        pasta = SAIDA / f"i{k:02d}"; pasta.mkdir(parents=True, exist_ok=True)
        m = J.Motor(); m.carga(df.loc[fim - pd.Timedelta("60D"):fim], J.conhecidos("limpo", fim), fim)
        m.tab.to_parquet(pasta / "tab.parquet"); m.buf.to_parquet(pasta / "buf.parquet")
        L.append(dict(k=k, fim=str(fim), pasta=str(pasta)))
    (SAIDA / "instantes.json").write_text(json.dumps(L))


def recorta_csv(a97: Path, inst: list[dict]):
    """Uma passada no export de 30 s: grava, para cada instante, os CSVs de 3, 60 e 90 d no formato de produção."""
    import pandas as pd
    alvos = []
    for x in inst:
        fim = pd.Timestamp(x["fim"]) + pd.Timedelta("2min")       # até a última janela fechada, exclusive
        for n in (3, 60, 90):
            ini = fim - pd.Timedelta("2min") - pd.Timedelta(f"{n}D")
            alvos.append((ini.strftime("%Y-%m-%d %H:%M:%S"), fim.strftime("%Y-%m-%d %H:%M:%S"),
                          Path(x["pasta"]) / f"bruto{n}.csv"))
    with open(a97) as f:
        cab = f.readline()
        arqs = {p: open(p, "w") for _, _, p in alvos}
        for p in arqs.values():
            p.write(cab)
        inis = sorted((a, b, p) for a, b, p in alvos)
        for ln in f:
            t = ln[:19]
            for a, b, p in inis:
                if a <= t < b:
                    arqs[p].write(ln)
        for p in arqs.values():
            p.close()


def mede(rotulo, cmd, log):
    o = subprocess.run([sys.executable, str(TELEM), "--rotulo", rotulo, "--log", str(log), "--ok", "0", "--"] + cmd,
                       capture_output=True, text=True)
    return o.returncode, o.stderr[-600:]


def parte_a(pk: Path, a97: Path, teste: bool, local: bool, insumos):
    import pandas as pd
    o = subprocess.run([sys.executable, __file__, "--estados"] + (["--local"] if local else []) +
                       (["--teste"] if teste else []) + ["--insumos", insumos or ""], capture_output=True, text=True)
    if o.returncode != 0:
        raise RuntimeError("estados: " + o.stderr[-1500:])
    inst = json.loads((SAIDA / "instantes.json").read_text())
    t0 = time.time(); recorta_csv(a97, inst); print(f"CSVs recortados em {time.time() - t0:.0f} s", flush=True)
    log = SAIDA / "telemetria.jsonl"
    exemplo, scripts, modelos = pk / "scripts" / "tc33003a_exemplo.py", pk / "scripts", pk / "modelos"
    erros = []
    for x in inst:
        pa = Path(x["pasta"])
        for n in (60, 90):
            c, e = mede(f"lote_{n}|{x['fim']}", [sys.executable, str(exemplo), "--csv", str(pa / f"bruto{n}.csv"),
                                                 "--dias", "1", "--ignorar-validade"], log)
            if c: erros.append((f"lote_{n}", x["fim"], c, e))
        c, e = mede(f"motor|{x['fim']}", [sys.executable, str(AQUI / "j11_motor_cmd.py"), str(pa), str(scripts),
                                          str(modelos)], log)
        if c: erros.append(("motor", x["fim"], c, e))
        print(f"   {x['fim']} medido", flush=True)
    R = pd.DataFrame([json.loads(l) for l in log.read_text().splitlines()])
    R["modo"] = R.rotulo.str.split("|").str[0]
    R["cpu_s"] = R.cpu_usuario_s + R.cpu_sistema_s
    R["csv_mb"] = [Path(json.loads(l)["comando"].split("--csv ")[1].split(" ")[0]).stat().st_size / 1e6
                   if "--csv" in json.loads(l)["comando"] else None for l in log.read_text().splitlines()]
    R.to_csv(SAIDA / "telemetria.csv", index=False)
    res = {}
    for modo, g in R[R.codigo_saida == 0].groupby("modo", sort=False):
        res[modo] = dict(n=len(g), duracao_s_med=round(float(g.duracao_s.median()), 2),
                         duracao_s_p95=round(float(g.duracao_s.quantile(0.95)), 2),
                         cpu_s_med=round(float(g.cpu_s.median()), 2),
                         ram_pico_mb_med=round(float(g.ram_pico_maior_processo_mb.median()), 1),
                         ram_pico_mb_max=round(float(g.ram_pico_maior_processo_mb.max()), 1),
                         lidos_mb_med=round(float(g.bytes_logicos_lidos_mb_amostrado.median()), 1))
        r = res[modo]
        print(f"   {modo:9s} {r['duracao_s_med']} s (p95 {r['duracao_s_p95']}) | CPU {r['cpu_s_med']} s | RAM pico "
              f"{r['ram_pico_mb_med']} MB (máx {r['ram_pico_mb_max']}) | lidos {r['lidos_mb_med']} MB | n {r['n']}", flush=True)
    res["falhas"] = [dict(modo=a, fim=b, codigo=c, erro=e) for a, b, c, e in erros]
    if erros:
        print("FALHAS:", res["falhas"][:3], flush=True)
    return res


def parte_b():
    import numpy as np
    import pandas as pd
    import contextlib
    import io
    from nv.controle_sinais import sinais_causais
    S = sinais_causais(pd.Timestamp("2025-01-08 07:34", tz="UTC"), cache=None, extra=G["extra"])
    S["mask"] = S["mask"].astype(bool); S["reset"] = S["reset"].astype(bool)
    out = {}
    for nome, X in (("controle", S), ("sem_vb", S.assign(vb=np.nan))):
        with contextlib.redirect_stdout(io.StringIO()):
            r = G["ci"].prever({"detector": G["det"]}, X)
        out[nome] = j1.regua(r["is_anomaly"].astype(bool), r["vigiado"].astype(bool))
        print(f"   {nome}: {out[nome]['det']}/8 FP {out[nome]['fp']} ({out[nome]['fp_mes']}/mês), início "
              f"{out[nome]['inicio']}, banda {out[nome]['banda']}", flush=True)
    return out


def executa(local: bool, insumos: str | None, teste: bool):
    SAIDA.mkdir(parents=True, exist_ok=True); t00 = time.time()
    pk, a97, _ = j1.monta(local, insumos)
    res = dict(ambiente=j1.ambiente()); print("ambiente:", res["ambiente"], flush=True)
    print("PARTE A (telemetria)", flush=True)
    res["A"] = parte_a(pk, a97, teste, local, insumos)
    print("PARTE B (sem vb)", flush=True)
    j1.prepara(pk, a97, teste)
    res["B"] = parte_b()
    res["duracao_min"] = round((time.time() - t00) / 60, 1)
    (SAIDA / "j11_resultado.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    from clearml import Task as Tk
    tk = Tk.current_task()
    if tk is not None:
        tk.upload_artifact("j11_resultado", artifact_object=str(SAIDA / "j11_resultado.json"))
        tk.upload_artifact("j11_telemetria", artifact_object=str(SAIDA / "telemetria.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enqueue", action="store_true"); ap.add_argument("--estados", action="store_true")
    ap.add_argument("--local", action="store_true"); ap.add_argument("--teste", action="store_true")
    ap.add_argument("--insumos", default=j1.DS_INSUMOS)
    a = ap.parse_args()
    if a.estados:                       # subprocesso da parte A: monta os estados do motor e sai
        pk, a97, _ = j1.monta(a.local, a.insumos or j1.DS_INSUMOS)
        j1.prepara(pk, a97, a.teste); monta_estados(a.teste); return
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
