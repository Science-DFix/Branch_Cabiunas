#!/usr/bin/env python3
"""J19 -- os três pontos abertos do J18: (1) por que os sinais diferem ~12 h no início da PLENA; (2) que regra de PLENA
zera a divergência com a referência; (3) o motor com 1 thread.

PRÉ-REGISTRO (21/11/2026, commitado antes de enviar)
Origem: J18 (task 4e026612; repetido em fb9618a6, idêntico). Pela regra do `estado_saida` (posições estáveis antes do
  bloco de 6 h >= 12.720), 9 das 30 instalações divergem da referência em PLENA: 8 com ~12 h de sinais diferentes
  logo depois da entrada em PLENA (alarme igual) e 01/01/2026 com carga de 30 d (alarme 02/01–23/01, sinais iguais).
  Leitura do código (e733279): o vb cru (`_z_vibracao`) usa a base das últimas 12.000 posições estáveis antes do bloco
  menos a guarda; com >= 12.720 posições desde o início do dado, a base do motor é a mesma da referência. Mas o vb
  entra no detector suavizado por EWMA (`det["halflife"]["vb"]`), e o EWMA carrega a diferença dos blocos anteriores
  (base truncada) por algumas meias-vidas.
Pacote e dados: os do J18 (scripts e733279 do Dataset bed2e62f; modelos, registro e CSV do f5e1b16b). Nada do pacote
  muda.
PARTE 1+2 (as 30 instalações do J34/J18; `j18.parte_b`, motor do J34 contra o lote de 60 d):
  V1: o resumo da parte B sai igual ao J18 (PLENA 19,12/12,25/0/0/0 d; divergências 3/4/2/0/0).
  (1) nas linhas em PLENA que divergem: quais sinais (t, p, sp, vb), |Δ| máximo, horas desde a entrada em PLENA até a
      última divergência, e a meia-vida implícita do decaimento de |Δvb| (ajuste de log2|Δ| contra o tempo, linhas com
      |Δ| > 1e-9), comparada com a meia-vida do vb no bundle.
  (2) regras de PLENA, todas sobre as posições do pacote (limiar 12.720):
      R0   pacote: posições no bloco da linha;
      R12  posições no bloco de 12 h antes da linha (PLENA atrasada 12 h);
      R24  idem, 24 h;
      R45  R0 e a linha a >= 45 d do início do dado da instalação (memória da decisão);
      R45_24  R45 e R24.
      Para cada regra: PLENA em (mediana, máx) por carga; instalações com divergência em PLENA (sinais ou alarme) e
      com divergência de alarme.
PARTE 3 (motor do módulo `motor_estado/`, o que vai ao pacote; `j19_motor_cmd.py`; mesmos 10 instantes e medidor do
  J11/J18; export de 30 s): passo com as threads padrão × com OMP/OPENBLAS/MKL_NUM_THREADS=1. Duração, CPU, RAM e
  igualdade da saída (sinais, máscara, alarme) entre as duas.
EXPECTATIVAS: (1) nos 8 casos só o vb diverge (~80%); a meia-vida implícita fica a ±25% da do bundle (~65%).
  (2) R12 ou R24 zera a divergência de sinais nos 8 (~75%) e não zera o alarme de 01/01/2026 com carga de 30 d (~90%);
  R45 zera o alarme (~80%); R45_24 zera tudo (~70%) com PLENA sem carga em ~45–65 d e com carga de 60 d desde T0 (~85%).
  (3) saída idêntica em 10/10 (~90%); CPU cai >= 50% (~75%); duração igual ±10% (~70%).
CRITÉRIO: (1) se só o vb diverge e o decaimento bate com a meia-vida, a divergência é o EWMA levando a base truncada
  para dentro da PLENA: explicado, não é erro do motor. Senão, fica aberta. (2) recomendar ao Thallys a regra mais
  curta das testadas que zera as divergências em PLENA nas 30, com o custo em dias; a decisão de pôr no `estado_saida`
  é dele. (3) se idêntica e CPU menor, o 1 thread vai para o comando do motor (nosso), e o Thallys puxa.
LIMITES: os do J18 (8 trips, 6 datas, Cica).

RESULTADO (task ade8113e, Cica, 10,6 min, commit 2f53f9c). V1: as 30 instalações repetem o J18 (PLENA 19,12/12,25/0/0/0 d;
  divergências 3/4/2/0/0).
  (1) Nos 8 casos de sinais, SÓ o vb diverge (t, p, sp: 0 linhas); |Δvb| máx 0,004–0,25; a última divergência sai
      10,9–13,9 h depois da entrada em PLENA; meia-vida implícita do decaimento 0,50 h nos 8 = meia-vida do vb no
      bundle (30 min).
  (2) regra              instalações com divergência   PLENA em, mediana (máx), por carga 0 / 7 / 30 / 45 / 60 d
      R0 (pacote)        9 (8 sinais, 1 alarme)        19,1 (52,5) / 12,3 (47,5) / 0 (1,3) / 0 / 0
      R12                7 (6 sinais, 1 alarme)        19,6 / 12,8 / 0 / 0 / 0
      R24                1 (só o alarme)               20,1 (53,5) / 13,3 (48,5) / 0 (2,3) / 0 / 0
      R45                1 (só o alarme)               45 (52,5) / 38 / 15 / 0 / 0
      R45_24             1 (só o alarme)               45 (53,5) / 38 / 15 / 0 / 0
      O alarme que sobra é sempre 01/01/2026 com carga de 30 d: 11.848 linhas em R0/R12/R24, 4.419 (16–23/01) em R45.
      A regra antiga do J34 (PLENA em 33 d nesse caso) é a única medida com 0 nas 30 (J18).
  (3) motor do módulo, 10 passos: saída idêntica 10/10; CPU 4,68 -> 0,40 s; duração 0,504 -> 0,403 s; RAM 141 -> 140 MB.
PLACAR: (1) só vb SIM; meia-vida ±25% SIM (0,50 = 0,50). (2) R12 ou R24 zera os sinais SIM (R24; R12 não);
  não zera o alarme de 01/01/26 SIM; R45 zera o alarme NÃO (4.419 linhas); R45_24 zera tudo NÃO; R45_24 sem carga em
  45–65 d e com 60 d desde T0 SIM. (3) idêntica SIM; CPU >= 50% SIM (-91%); duração ±10% NÃO (-20%).
LEITURA: (1) explicado: é o EWMA do vb (meia-vida 30 min) levando para dentro da PLENA a diferença dos blocos com base
  truncada; não é erro do motor, e nenhum alarme muda. (2) nenhuma regra testada zera as 30 (o critério previa uma).
  R24 (PLENA 24 h depois da base cheia) zera os sinais em todas, ao custo de +1 d. O alarme de 01/01/2026 com carga de
  30 d não sai por regra de dias até 45 d: é a cadeia da decisão montada sobre a carga curta, que só se alinha à
  referência em 23/01. Só a carga >= 45 d (ou a regra antiga, 33 d nesse caso) resolve. Recomendação ao Thallys:
  R24 no `estado_saida` e carga inicial >= 45 d (recomendada 60 d) como condição da instalação. (3) o 1 thread entra no
  `teste_equivalencia.py` e no docstring do `motor_estado.py` (o módulo não pode definir: o numpy já foi importado).
  _(registrado em 21/11/2026 12:51)_

Uso:
  python frente_thallys/j19_plena_threads.py --enqueue
"""
from clearml import Task  # noqa: F401  (antes do argparse: o worker injeta argumentos por patch)
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import j1_janela_entrada as j1          # noqa: E402
import j11_telemetria_e_sem_vb as j11   # noqa: E402
import j18_e733279_ram_plena as j18     # noqa: E402

NOME = "frente_thallys | J19 PLENA: causa das 12 h, regra e motor com 1 thread"
SAIDA = Path(tempfile.gettempdir()) / "j19_saida"
G = j1.G
UMA = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
J18_B = {"0": (19.12, 3), "7": (12.25, 4), "30": (0.0, 2), "45": (0.0, 0), "60": (0.0, 0)}


# ───────────────────────────── parte 3 ─────────────────────────────
def mede(rotulo, cmd, log, env):
    o = subprocess.run([sys.executable, str(j11.TELEM), "--rotulo", rotulo, "--log", str(log), "--ok", "0", "--"] + cmd,
                       capture_output=True, text=True, env=env)
    return o.returncode, o.stderr[-600:]


def parte_3(pk: Path, a97: Path, teste: bool, local: bool, insumos):
    import numpy as np
    import pandas as pd
    base = {k: v for k, v in os.environ.items() if not k.startswith(("CLEARML_", "TRAINS_"))}
    base = {k: v for k, v in base.items() if k not in UMA}
    env = dict(base, J11_PK=str(pk), J11_A97=str(a97), J11_NV_PAI=str(j11.NV_PAI[0]))
    o = subprocess.run([sys.executable, j11.__file__, "--estados"] + (["--local"] if local else []) +
                       (["--teste"] if teste else []) + ["--insumos", insumos or ""], capture_output=True, text=True,
                       env=env)
    if o.returncode != 0:
        raise RuntimeError("estados: " + o.stderr[-1500:])
    inst = json.loads((j11.SAIDA / "instantes.json").read_text())
    log = SAIDA / "telemetria.jsonl"
    erros, iguais = [], []
    for x in inst:
        pa = Path(x["pasta"]); j11.recorta_csv(a97, [x])
        saidas = {}
        for modo, e in (("padrao", base), ("1thread", dict(base, **UMA))):
            saidas[modo] = pa / f"out_{modo}.parquet"
            c, err = mede(f"motor_{modo}|{x['fim']}", [sys.executable, str(AQUI / "j19_motor_cmd.py"), str(pa),
                                                     str(pk / "scripts"), str(pk / "modelos"), str(saidas[modo])], log, e)
            if c: erros.append((modo, x["fim"], c, err))
        if all(p.exists() for p in saidas.values()):
            A, B = (pd.read_parquet(p) for p in saidas.values())
            ok = A.index.equals(B.index) and list(A.columns) == list(B.columns)
            for col in A.columns if ok else []:
                a, b = A[col].to_numpy(), B[col].to_numpy()
                ok &= bool(np.array_equal(a, b, equal_nan=a.dtype.kind == "f"))
            iguais.append(dict(fim=x["fim"], igual=ok, linhas=len(A)))
        for n in (3, 60, 90):
            (pa / f"bruto{n}.csv").unlink()
    R = pd.DataFrame([json.loads(ln) for ln in log.read_text().splitlines()])
    R["modo"] = R.rotulo.str.split("|").str[0]; R["cpu_s"] = R.cpu_usuario_s + R.cpu_sistema_s
    R.to_csv(SAIDA / "telemetria.csv", index=False)
    res = {}
    for modo, g in R[R.codigo_saida == 0].groupby("modo", sort=False):
        res[modo] = dict(n=len(g), duracao_s_med=round(float(g.duracao_s.median()), 3),
                         duracao_s_p95=round(float(g.duracao_s.quantile(0.95)), 3),
                         cpu_s_med=round(float(g.cpu_s.median()), 2),
                         ram_pico_mb_med=round(float(g.ram_pico_maior_processo_mb.median()), 1),
                         ram_pico_mb_max=round(float(g.ram_pico_maior_processo_mb.max()), 1))
        r = res[modo]
        print(f"   {modo:15s} {r['duracao_s_med']} s (p95 {r['duracao_s_p95']}) | CPU {r['cpu_s_med']} s | RAM "
              f"{r['ram_pico_mb_med']} MB (máx {r['ram_pico_mb_max']}) | n {r['n']}", flush=True)
    res["saida_igual"] = f"{sum(i['igual'] for i in iguais)}/{len(iguais)}"
    res["comparacoes"] = iguais
    res["falhas"] = [dict(modo=a, fim=b, codigo=c, erro=e) for a, b, c, e in erros]
    print("   saída 1 thread = padrão:", res["saida_igual"], flush=True)
    if erros:
        print("FALHAS:", res["falhas"][:3], flush=True)
    return res


# ───────────────────────────── partes 1 e 2 ─────────────────────────────
def analisa_12(J):
    import numpy as np
    import pandas as pd
    ci = G["ci"]
    ref = pd.read_parquet(J.SAIDA / "ref_limpo.parquet")
    passo = pd.Timedelta(hours=ci.VB_PASSO_H)
    hl_vb = pd.Timedelta(G["det"]["halflife"]["vb"]).total_seconds() / 3600
    regras = ("R0", "R12", "R24", "R45", "R45_24")
    L, diag = [], []
    for t0_s in J.INSTALA:
        for H in J.CARGAS:
            r = pd.read_parquet(J.SAIDA / f"inst_{t0_s}_H{H}.parquet")
            T0 = J.T(t0_s); fim = T0 + J.D("100D"); inicio = T0 - J.D(f"{H}D")
            banco = G["df"].loc[inicio:] if H else G["df"].loc[T0:]
            est = ci.instantes_estaveis(banco.loc[:fim])
            Rr = ref.reindex(r.index)

            def pos(atraso_h):
                q = r.index - pd.Timedelta(hours=atraso_h)
                return est.searchsorted(ci.EPOCA + ((q - ci.EPOCA) // passo) * passo, side="left")
            cheia = {h: pos(h) >= ci.VB_POSICOES_PLENA for h in (0, 12, 24)}
            mem45 = (r.index >= inicio + J.D("45D"))
            pl = dict(R0=cheia[0], R12=cheia[12], R24=cheia[24], R45=cheia[0] & mem45, R45_24=cheia[24] & mem45)
            difs = {}
            for s in ("t", "p", "sp", "vb"):
                a, b = r[s].to_numpy(float), Rr[s].to_numpy(float)
                nd = np.isnan(a) != np.isnan(b); ok = ~np.isnan(a) & ~np.isnan(b)
                dd = np.zeros_like(a); dd[ok] = np.abs(a[ok] - b[ok]); dd[nd] = np.inf
                difs[s] = dd
            dsin = np.zeros(len(r), bool)
            for s in difs:
                dsin |= difs[s] > 1e-9
            dal = r["is_anomaly"].astype(bool).to_numpy() != Rr["is_anomaly"].fillna(False).astype(bool).to_numpy()
            linha = dict(T0=t0_s, H=H)
            for k in regras:
                m = pl[k]
                linha[f"{k}_plena_d"] = None if not m.any() else round((r.index[np.flatnonzero(m)[0]] - T0).total_seconds() / 86400, 2)
                linha[f"{k}_div_sinais"] = int((dsin & m).sum()); linha[f"{k}_div_alarme"] = int((dal & m).sum())
            L.append(linha)
            m0 = pl["R0"]
            if (dsin & m0).any():
                q = np.flatnonzero(dsin & m0)
                t_pl = r.index[np.flatnonzero(m0)[0]]
                horas = ((r.index[q] - t_pl).total_seconds() / 3600).to_numpy()
                d = dict(T0=t0_s, H=H, linhas=len(q), horas_ate_ultima=round(float(horas.max()), 2),
                         sinais={s: dict(n=int((difs[s][q] > 1e-9).sum()), max=float(np.nanmax(difs[s][q])))
                                 for s in difs})
                v = difs["vb"][q]; ok = np.isfinite(v) & (v > 1e-9)
                if ok.sum() >= 10:
                    incl = np.polyfit(horas[ok], np.log2(v[ok]), 1)[0]
                    d["meia_vida_implicita_h"] = round(float(-1 / incl), 3) if incl < 0 else None
                diag.append(d)
    T = pd.DataFrame(L); T.to_csv(SAIDA / "j19_regras.csv", index=False)
    resumo = {}
    for k in regras:
        resumo[k] = dict(
            inst_div_sinais=int((T[f"{k}_div_sinais"] > 0).sum()), inst_div_alarme=int((T[f"{k}_div_alarme"] > 0).sum()),
            inst_div_qualquer=int(((T[f"{k}_div_sinais"] > 0) | (T[f"{k}_div_alarme"] > 0)).sum()),
            plena_d={str(H): dict(med=round(float(x[f"{k}_plena_d"].median()), 2), max=round(float(x[f"{k}_plena_d"].max()), 2))
                     for H, x in T.groupby("H")})
    so_vb = all(all(v["n"] == 0 for s, v in d["sinais"].items() if s != "vb") for d in diag if d["linhas"] < 1000)
    return dict(meia_vida_vb_bundle_h=hl_vb, diagnostico=diag, so_vb_nos_casos_curtos=so_vb, regras=resumo)


def partes_12(pk, a97, teste, procs):
    j18.SAIDA = SAIDA
    B = j18.parte_b(pk, a97, teste, procs)
    v1 = {H: (B["por_carga"][H]["plena_d_med"], B["por_carga"][H]["div_plena_n"]) for H in B["por_carga"]}
    import j34_estado_robustez as J
    return dict(v1_igual_j18=(v1 == J18_B), v1=v1, B=B, **analisa_12(J))


# ───────────────────────────── principal ─────────────────────────────
def executa(local: bool, insumos: str | None, scripts: str, teste: bool, procs: int):
    SAIDA.mkdir(parents=True, exist_ok=True); t00 = time.time()
    pk, a97, _ = j1.monta(local, insumos)
    j11.NV_PAI[:] = [sys.path[0]]
    scripts_e17 = (pk / "scripts").resolve()
    if local:
        novo = Path(os.environ["FT_SCRIPTS_J17"])
    else:
        from clearml import Dataset
        novo = Path(Dataset.get(dataset_id=scripts).get_local_copy()) / "scripts"
    (pk / "scripts").unlink(); shutil.copytree(novo, pk / "scripts")
    i = next(k for k, c in enumerate(sys.path) if c and Path(c).resolve() == scripts_e17)
    sys.path[i] = str(pk / "scripts")
    import cabiunas_inference as ci
    assert Path(ci.__file__).resolve().parent == (pk / "scripts").resolve() and hasattr(ci, "estado_saida"), ci.__file__
    res = dict(ambiente=j1.ambiente()); print("ambiente:", res["ambiente"], flush=True)
    print("PARTE 3 (motor do módulo, 1 thread x padrão)", flush=True)
    res["P3"] = parte_3(pk, a97, teste, local, insumos)
    print("PARTES 1 e 2 (30 instalações)", flush=True)
    res["P12"] = partes_12(pk, a97, teste, procs)
    print(json.dumps({k: v for k, v in res["P12"].items() if k != "B"}, indent=1, ensure_ascii=False, default=str), flush=True)
    res["duracao_min"] = round((time.time() - t00) / 60, 1)
    (SAIDA / "j19_resultado.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    from clearml import Task as Tk
    tk = Tk.current_task()
    if tk is not None:
        for f in ("j19_resultado.json", "telemetria.csv", "j19_regras.csv", "j18_instalacoes.csv"):
            tk.upload_artifact(Path(f).stem, artifact_object=str(SAIDA / f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enqueue", action="store_true"); ap.add_argument("--local", action="store_true")
    ap.add_argument("--teste", action="store_true"); ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--insumos", default=j1.DS_INSUMOS); ap.add_argument("--scripts", default=j18.DS_SCRIPTS)
    a = ap.parse_args()
    if a.enqueue:
        t = Task.init(project_name="TesteMLCab", task_name=NOME)
        t.set_base_docker("python:3.11-slim", docker_setup_bash_script=["apt-get update -qq", "apt-get install -y -qq libgomp1"])
        t.set_packages(j1.PACOTES)
        t.set_parameters({"General/insumos": a.insumos, "General/scripts": a.scripts, "General/procs": str(a.procs)})
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=AQUI, text=True).strip()
        t.set_script(repository="git@github.com:Science-DFix/Branch_Cabiunas.git", branch="exp/ocsvm-validacao", commit=commit)
        t.execute_remotely(queue_name="default")
    if os.environ.get("CLEARML_TASK_ID"):
        tk = Task.init(project_name="TesteMLCab", task_name=NOME)
        a.insumos = tk.get_parameter("General/insumos") or a.insumos
        a.scripts = tk.get_parameter("General/scripts") or a.scripts
        a.procs = int(tk.get_parameter("General/procs") or a.procs)
    executa(a.local, a.insumos, a.scripts, a.teste, a.procs)


if __name__ == "__main__":
    main()
