#!/usr/bin/env python3
"""J18 -- na e733279: (A) a RAM do lote medida de novo; (B) a coluna "PLENA em" do J34 pela regra nova.

PRÉ-REGISTRO (21/11/2026, commitado antes de enviar)
Origem: pendência do diário (J17, 21/11). A seção 6 da resposta à engenharia (563 MB / 800 MB do lote, J11) foi medida
  no e17bf27. A e733279 (feat/pacote-respostas-engenharia, do Thallys) traz o `carregar_dados` em blocos com texto do PI
  -> NaN (8364998), o exemplo com 1 thread e o `estado_saida`. E a coluna "PLENA em" da seção 3 foi medida no J34 com a
  regra antiga (tabela de 45 d cheia); a regra nova (3ab1070) conta só as posições estáveis da base do vb antes do
  bloco de 6 h: INDISPONÍVEL < 3.720 <= AQUECENDO < 12.720 <= PLENA.
Pacote: scripts da e733279 = Dataset bed2e62f (o mesmo do J17), copiados para a árvore do pacote; modelos, registro e
  CSV do f5e1b16b (como J11/J17). Nada é alterado no código do Thallys.
PARTE A (= J11 parte A, mesma semente e mesmos 10 instantes, export de 30 s, medidor do Thallys, processo novo):
  lote_60 e lote_90 com o `tc33003a_exemplo.py` do e17bf27 (controle, refaz o J11 na mesma máquina) e da e733279;
  motor (j11_motor_cmd, motor do J34) com os scripts da e733279. Mediana, p95, RAM pico, CPU.
PARTE B (= J3 do J34, mesmas 30 instalações: T0 em 6 datas x carga H 0/7/30/45/60 d, 100 d, motor x ref60 limpo),
  com o `cabiunas_inference` da e733279. Em cada linha emitida: posições = ci.posicoes_base_vb(instantes estáveis do
  banco disponível, instante da linha) e estado pelos limiares do pacote (VB_POSICOES_OPINA/PLENA). Também a regra
  antiga (J34), como controle. Mede: PLENA em (d), divergência (sinais |Δ|>1e-9 ou alarme) em PLENA e em AQUECENDO,
  horas em INDISPONÍVEL e em AQUECENDO.
  Nota: no motor a contagem vem do buffer (26.000 posições); abaixo de 12.720 ela é igual à do banco disponível.
VERIFICAÇÃO: V1 ref60 limpo = 7/8 FP 9. V2 regra antiga recomputada = J34 (PLENA 64/57/35/20/13 d; 0 em PLENA).
  V3 lote e17bf27 = J11 (563 / 800 MB, ±10%). Se V1 ou V2 falhar, a parte B é só relato; se V3 falhar, a A é só relato.
EXPECTATIVAS:
  A: e733279 lote_90 RAM mediana <= 600 MB (~75%; o Thallys mediu 877 -> 513 MB); lote_60 <= 450 MB (~70%);
     duração igual ±15% (~70%); CPU menor (1 thread) (~80%); motor 141 MB ±10% (~80%).
  B: PLENA mais cedo que na regra antiga em todas as cargas (~85%); H = 60 PLENA desde T0 em 6/6 (~75%); H = 0 PLENA
     em 15–30 d (~60%); divergência em PLENA > 0 em pelo menos 1 das 30 (~60%: a regra nova não pede a tabela de 45 d,
     e o caso 01/01/2026 H = 30 tinha alarme 3 semanas a mais com sinais iguais).
CRITÉRIO: B com 0 divergência em PLENA nas 30 -> a coluna nova entra no documento e a regra do pacote basta.
  Divergência > 0 -> entra a coluna nova com as linhas que divergem e o documento diz que PLENA (pacote) garante a
  base do vb, não a igualdade com a referência; a recomendação de carga de 60 d em blocos fica.
  A: os números da e733279 substituem os da seção 6 (com o e17bf27 ao lado).
LIMITES: os do J11 e do J34 (Cica, não o servidor; 8 trips; 6 datas de instalação).

RESULTADO (task 4e026612, Cica, 11,7 min, commit d98bc7c)
  V1 ref60 limpo 7/8 FP 9 (início 6, banda 4, 1.451 h). V2 regra antiga: 0 em PLENA nas 30; PLENA 64/57/35/20/13 d
  (= J34). V3 e17bf27: lote_60 563 MB, lote_90 799 MB (= J11). As três passam.
  A (mediana de 10, processo novo, export de 30 s):
                   e17bf27                               e733279
    lote_60   1,61 s | CPU 5,7 s | 563 MB (366–574)   1,31 s | CPU 1,3 s | 366 MB (364–375)
    lote_90   2,31 s | CPU 6,8 s | 799 MB (497–808)   1,81 s | CPU 1,8 s | 495 MB (488–499)
    motor     --                                     0,50 s | CPU 4,7 s | 139 MB (139–146)
    Na e733279, 2025 e 2026 ocupam o mesmo (lote_90 492 × 497 MB): o texto do PI deixou de pesar.
  B (30 instalações, 100 d; PLENA pela regra do pacote):
    carga   PLENA em (med; mín–máx)   antiga   INDISPONÍVEL (med; máx)   AQUECENDO (med)   PLENA ≠ referência
    0 d     19 d (18–52)              64 d     126 h (648 h)             333 h              3 de 6 (sinais, ~12 h, alarme igual)
    7 d     12 d (11–48)              57 d     0 (540 h)                 294 h              4 de 6 (idem)
    30 d    0 (0–1,25 d)              35 d     0                         0                  2 de 6 (01/07/25: sinais ~12 h;
                                                                                             01/01/26: ALARME 02/01–23/01, 16,5%)
    45 d    0                         20 d     0                         0                  0 de 6
    60 d    0                         13 d     0                         0                  0 de 6
    As divergências de sinais começam no 1º bloco de 6 h da PLENA e duram ~12 h (327–419 linhas), sem mudar alarme.
    A de alarme é o caso já visto no J34 (tabela de 30 d sem a cadeia do refratário; sinais iguais).
PLACAR: A lote_90 <= 600 MB SIM (495); lote_60 <= 450 MB SIM (366); duração igual ±15% NÃO (19–22% mais rápido);
  CPU menor SIM (5,7 -> 1,3 s; 6,8 -> 1,8 s); motor 141 ±10% SIM (139). B PLENA mais cedo em todas as cargas SIM;
  H = 60 PLENA desde T0 6/6 SIM; H = 0 em 15–30 d SIM pela mediana (19 d; 2 de 6 passam: 38 e 52 d);
  divergência em PLENA > 0 SIM (9 de 30).
LEITURA: critério B -> divergência > 0: a coluna nova entra no documento e o texto diz que PLENA (pacote) garante a base
  do vb, não a igualdade com a referência; com carga >= 45 d, PLENA desde T0 e 0 divergência nas 12. A recomendação
  (60 d em blocos) fica. A: os números da e733279 substituem os da seção 6 (o conserto do carregar_dados tira ~300 MB
  do lote em 2026, e o 1 thread corta a CPU a 1/4). Não diagnosticado: a causa das ~12 h de sinais diferentes no
  início da PLENA (hipótese: vb com base recém-completa difere da do ref60, que vê 60 d).
  _(registrado em 21/11/2026 09:47)_

Uso:
  python frente_thallys/j18_e733279_ram_plena.py --enqueue
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
from multiprocessing import get_context
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import j1_janela_entrada as j1      # noqa: E402
import j11_telemetria_e_sem_vb as j11  # noqa: E402

NOME = "frente_thallys | J18 e733279: RAM do lote e PLENA em (regra nova)"
DS_SCRIPTS = "bed2e62f62914fa087af7e0a28c9e296"   # scripts da e733279 (criado no J17)
SAIDA = Path(tempfile.gettempdir()) / "j18_saida"
G = j1.G


# ───────────────────────────── parte A ─────────────────────────────
def parte_a(pk: Path, a97: Path, scripts_e17: Path, teste: bool, local: bool, insumos):
    import pandas as pd
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CLEARML_", "TRAINS_"))}
    env.update(J11_PK=str(pk), J11_A97=str(a97), J11_NV_PAI=str(j11.NV_PAI[0]))
    o = subprocess.run([sys.executable, j11.__file__, "--estados"] + (["--local"] if local else []) +
                       (["--teste"] if teste else []) + ["--insumos", insumos or ""], capture_output=True, text=True,
                       env=env)
    if o.returncode != 0:
        raise RuntimeError("estados: " + o.stderr[-1500:])
    inst = json.loads((j11.SAIDA / "instantes.json").read_text())
    log = SAIDA / "telemetria.jsonl"
    versoes = {"e17bf27": scripts_e17, "e733279": pk / "scripts"}
    erros = []
    for x in inst:
        pa = Path(x["pasta"]); j11.recorta_csv(a97, [x])
        for v, sc in versoes.items():
            for n in (60, 90):
                c, e = j11.mede(f"lote_{n}|{v}|{x['fim']}", [sys.executable, str(sc / "tc33003a_exemplo.py"), "--csv",
                                str(pa / f"bruto{n}.csv"), "--modelos", str(pk / "modelos"), "--dias", "1",
                                "--ignorar-validade"], log)
                if c: erros.append((f"lote_{n}|{v}", x["fim"], c, e))
        c, e = j11.mede(f"motor|e733279|{x['fim']}", [sys.executable, str(AQUI / "j11_motor_cmd.py"), str(pa),
                                                     str(pk / "scripts"), str(pk / "modelos")], log)
        if c: erros.append(("motor|e733279", x["fim"], c, e))
        for n in (3, 60, 90):
            (pa / f"bruto{n}.csv").unlink()
        print(f"   {x['fim']} medido", flush=True)
    R = pd.DataFrame([json.loads(ln) for ln in log.read_text().splitlines()])
    R["modo"] = R.rotulo.str.split("|").str[:2].str.join("|"); R["fim"] = R.rotulo.str.split("|").str[2]
    R["cpu_s"] = R.cpu_usuario_s + R.cpu_sistema_s
    R.to_csv(SAIDA / "telemetria.csv", index=False)
    res = {}
    for modo, g in R[R.codigo_saida == 0].groupby("modo", sort=False):
        res[modo] = dict(n=len(g), duracao_s_med=round(float(g.duracao_s.median()), 2),
                         duracao_s_p95=round(float(g.duracao_s.quantile(0.95)), 2),
                         cpu_s_med=round(float(g.cpu_s.median()), 2),
                         ram_pico_mb_med=round(float(g.ram_pico_maior_processo_mb.median()), 1),
                         ram_pico_mb_min=round(float(g.ram_pico_maior_processo_mb.min()), 1),
                         ram_pico_mb_max=round(float(g.ram_pico_maior_processo_mb.max()), 1),
                         ram_2025_med=round(float(g[g.fim < "2026"].ram_pico_maior_processo_mb.median()), 1),
                         ram_2026_med=round(float(g[g.fim >= "2026"].ram_pico_maior_processo_mb.median()), 1))
        r = res[modo]
        print(f"   {modo:15s} {r['duracao_s_med']} s (p95 {r['duracao_s_p95']}) | CPU {r['cpu_s_med']} s | RAM "
              f"{r['ram_pico_mb_med']} MB ({r['ram_pico_mb_min']}–{r['ram_pico_mb_max']}; 2025 {r['ram_2025_med']}, "
              f"2026 {r['ram_2026_med']}) | n {r['n']}", flush=True)
    res["falhas"] = [dict(modo=a, fim=b, codigo=c, erro=e) for a, b, c, e in erros]
    if erros:
        print("FALHAS:", res["falhas"][:3], flush=True)
    return res


# ───────────────────────────── parte B ─────────────────────────────
def _div(r, Rr):
    import numpy as np
    dif = np.zeros(len(r), bool)
    for s in ("t", "p", "sp", "vb"):
        a, b = r[s].to_numpy(float), Rr[s].to_numpy(float)
        nd = np.isnan(a) != np.isnan(b); ok = ~np.isnan(a) & ~np.isnan(b)
        dd = np.zeros_like(a); dd[ok] = np.abs(a[ok] - b[ok]); dif |= (dd > 1e-9) | nd
    dal = r["is_anomaly"].astype(bool).to_numpy() != Rr["is_anomaly"].fillna(False).astype(bool).to_numpy()
    return dif | dal, dal


def analisa_b(J):
    import numpy as np
    import pandas as pd
    ci = G["ci"]
    ref = pd.read_parquet(J.SAIDA / "ref_limpo.parquet")
    passo = pd.Timedelta(hours=ci.VB_PASSO_H)
    L, linhas_div = [], []
    for t0_s in J.INSTALA:
        for H in J.CARGAS:
            r = pd.read_parquet(J.SAIDA / f"inst_{t0_s}_H{H}.parquet")
            T0 = J.T(t0_s); fim = T0 + J.D("100D")
            banco = G["df"].loc[T0 - J.D(f"{H}D"):] if H else G["df"].loc[T0:]
            est = ci.instantes_estaveis(banco.loc[:fim])
            bloco = ci.EPOCA + ((r.index - ci.EPOCA) // passo) * passo
            pos = est.searchsorted(bloco, side="left")
            dif, dal = _div(r, ref.reindex(r.index))
            ind = pos < ci.VB_POSICOES_OPINA; aq = ~ind & (pos < ci.VB_POSICOES_PLENA); pl = pos >= ci.VB_POSICOES_PLENA
            antiga = r["n_est_45"].to_numpy() >= J.N_CHEIA
            dias = lambda m: None if not m.any() else round((r.index[np.flatnonzero(m)[0]] - T0).total_seconds() / 86400, 2)  # noqa: E731
            pct = lambda x, m: round(100 * float(x[m].mean()), 3) if m.any() else None  # noqa: E731
            L.append(dict(T0=t0_s, H=H, plena_d=dias(pl), plena_antiga_d=dias(antiga),
                          div_em_plena_pct=pct(dif, pl), alarme_div_em_plena_pct=pct(dal, pl),
                          div_em_aquecendo_pct=pct(dif, aq), alarme_div_em_aquecendo_pct=pct(dal, aq),
                          div_em_plena_antiga_pct=pct(dif, antiga),
                          horas_indisponivel=round(float(ind.sum()) * 2 / 60, 1),
                          horas_aquecendo=round(float(aq.sum()) * 2 / 60, 1)))
            if (dif & pl).any():
                q = np.flatnonzero(dif & pl)
                linhas_div.append(dict(T0=t0_s, H=H, n=len(q), primeira=str(r.index[q[0]]), ultima=str(r.index[q[-1]]),
                                       so_alarme=bool(not (dif & pl & ~dal).any())))
    T3 = pd.DataFrame(L); T3.to_csv(SAIDA / "j18_instalacoes.csv", index=False)
    med = lambda s: None if not s.dropna().size else round(float(s.dropna().median()), 2)  # noqa: E731
    mx = lambda s: None if not s.dropna().size else round(float(s.dropna().max()), 3)  # noqa: E731
    por = {str(H): dict(plena_d_med=med(x.plena_d), plena_d_min=None if not x.plena_d.dropna().size else round(float(x.plena_d.min()), 2),
                        plena_d_max=mx(x.plena_d), plena_antiga_d_med=med(x.plena_antiga_d),
                        div_plena_max=mx(x.div_em_plena_pct), div_plena_n=int((x.div_em_plena_pct > 0).sum()),
                        div_aquecendo_med=med(x.div_em_aquecendo_pct), horas_indisp_med=med(x.horas_indisponivel),
                        horas_indisp_max=mx(x.horas_indisponivel), horas_aquecendo_med=med(x.horas_aquecendo))
           for H, x in T3.groupby("H")}
    ok_ant = T3["div_em_plena_antiga_pct"].dropna()
    return dict(por_carga=por, linhas_div_em_plena=linhas_div,
                v2_regra_antiga_vale=bool((ok_ant == 0).all()),
                div_em_plena_zero_nas_30=bool((T3["div_em_plena_pct"].fillna(0) == 0).all()),
                regua_ref=J.regua(ref))


def parte_b(pk, a97, teste, procs):
    import j34_estado_robustez as J
    J.SAIDA = SAIDA / "j34"; J.SAIDA.mkdir(parents=True, exist_ok=True)
    j1.prepara(pk, a97, teste)
    ci = G["ci"]; G["orig_z"] = ci._z_vibracao; ci._z_vibracao = J._z_com_buffer
    J.prepara_cenarios()
    if teste:
        J.INSTALA[:] = ["2025-01-20"]
    trab = [("ref", "limpo")] + [("inst", (t, H)) for t in J.INSTALA for H in J.CARGAS]
    print(f"{len(trab)} corridas em paralelo", flush=True)
    with get_context("fork").Pool(min(procs, len(trab))) as pool:
        for _ in pool.imap_unordered(J.tarefa, trab):
            pass
    return analisa_b(J)


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
    # cópia real (não link): o exemplo acha ../modelos e ../registro_trips.csv pelo caminho resolvido
    (pk / "scripts").unlink(); shutil.copytree(novo, pk / "scripts")
    i = next(k for k, c in enumerate(sys.path) if c and Path(c).resolve() == scripts_e17)   # j1.monta pôs o e17bf27
    sys.path[i] = str(pk / "scripts")
    import cabiunas_inference as ci
    assert Path(ci.__file__).resolve().parent == (pk / "scripts").resolve() and hasattr(ci, "estado_saida"), ci.__file__
    res = dict(ambiente=j1.ambiente(), opina=ci.VB_POSICOES_OPINA, plena=ci.VB_POSICOES_PLENA)
    print("ambiente:", res["ambiente"], flush=True)
    print("PARTE A (telemetria)", flush=True)
    res["A"] = parte_a(pk, a97, scripts_e17, teste, local, insumos)
    print("PARTE B (PLENA em, regra nova)", flush=True)
    res["B"] = parte_b(pk, a97, teste, procs)
    print(json.dumps(res["B"], indent=1, ensure_ascii=False, default=str), flush=True)
    res["duracao_min"] = round((time.time() - t00) / 60, 1)
    (SAIDA / "j18_resultado.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    from clearml import Task as Tk
    tk = Tk.current_task()
    if tk is not None:
        for f in ("j18_resultado.json", "telemetria.csv", "j18_instalacoes.csv"):
            tk.upload_artifact(Path(f).stem, artifact_object=str(SAIDA / f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enqueue", action="store_true"); ap.add_argument("--local", action="store_true")
    ap.add_argument("--teste", action="store_true"); ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--insumos", default=j1.DS_INSUMOS); ap.add_argument("--scripts", default=DS_SCRIPTS)
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
