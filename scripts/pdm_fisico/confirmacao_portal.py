#!/usr/bin/env python3
"""C-PORTAL: confirmação de episódio pelos canais do Portal de Integridade (vibração do compressor, mancal de
escora, óleo de dreno). O teto de FP é do dado?

PRÉ-REGISTRADO EM 05/10/2026, ANTES DE RODAR (commit deste arquivo sem resultado).

DE ONDE VEM. `tags_ausentes.py` sobre o export interpolated do Portal (86 tags, 30 s, até 2025-10-24), 6 eventos:
pico nas 48 h antes do trip contra 2.000 janelas sorteadas em operação --
    vc vibração do compressor (VI_0301/02/04/05)      4,47x  p < 0,001
    es mancal de escora (TI_0318/0319)                3,69x  p = 0,001
    od óleo de dreno (TI_0320/0321/0322)              2,98x  p = 0,0005
HIPÓTESE NASCIDA DE OLHAR ESTES EVENTOS, e dito antes: os canais foram escolhidos nos mesmos trips em que serão
testados. Resultado favorável aqui é INDICATIVO; a validação é o export de 2025-11 em diante (eventos de 11/2025,
12/2025 e 02/2026) ou o modo sombra. Ficam de fora: deslocamento axial (p = 0,05) e desempenho (p = 0,19).

A REGRA (veto pós-refratário, como o P1b: o alarme final só perde episódios, nunca ganha -- o refratário não muda).
  Canal por família, com a composição do detector (retreino no dia `d` de cada mês, os 20.000 instantes vigiados
  anteriores ao corte, com todas as tags da família presentes):
    vc, es: max_tag |x - mediana| / (1,4826 MAD)          (o `vb` do detector, para estas tags)
    od    : max_tag |(x - mediana dos irmãos) - med| / MAD (o `sp`, cada mancal contra os outros dois)
  Só em instantes vigiados (operação estável fora do blackout); EWMA com meia-vida de 1 h. O limiar de cada
  família é o p99 da EWMA do próprio canal no baseline do mês. C(t) = max das três famílias de EWMA / limiar.
  Para cada episódio do alarme final: se C tem algum valor no episódio e max C < θ, o episódio é APAGADO.
  Sem nenhum valor de C (sem dado), o episódio fica. θ = 1,0 (o p99 do baseline), fixado agora.
  θ = 0,8 e θ = 1,2 rodam e são IMPRESSOS COMO SENSIBILIDADE, sem veredito e sem escolha.

A JANELA. As tags novas acabam em 2025-10-24. Os DOIS braços são medidos de 2025-01-01 a 2025-10-24: alarme,
eventos e tempo de operação cortados no mesmo instante (o detector roda inteiro e só a medida é cortada; ele é
causal, então o corte não muda nada antes dele). Cinco eventos: 27/02, 17/03, 07/04, 11/04, 29/04.

A DECISÃO. `bootstrap_regua.decide`, pareado, nas 8 composições (`regua_fp.DIAS`), IC de 97,5% (Bonferroni sobre
os dois critérios), blocos mensais de 2024-02 a 2025-10.
  PRIMÁRIO (a regra de sempre): GANHO = nenhuma composição perde detecção, medianas de início e banda não caem,
    carga (h FP + NEUTRO) cai em >= 7/8 com IC abaixo de zero e FP/mês não sobe.
  SECUNDÁRIO (contagem, o do P1): mesma condição de detecção, início e banda, IC do ΔFP/mês abaixo de zero e a
    carga média não sobe.

CONTROLES (o experimento não vale se falharem).
  1. A régua reproduz o publicado no dia 1, janela inteira: 8/8 e 0,344 FP/mês.
  2. θ = -inf reproduz a referência bit a bit nas 8 composições.
  3. A variante é subconjunto da referência (só apaga).
  4. Cobertura do canal C nos instantes vigiados da janela >= 95%.

CHECAGEM DE MECANISMO. Por composição: episódios apagados por classe (TP, FP, NEUTRO) e horas de cada.

EXPECTATIVA REGISTRADA. Os canais têm sinal antes dos trips, mas episódios FP/NEUTRO também nascem em operação
anormal (partida, carga), onde vibração e temperatura de mancal sobem por motivo de processo. Espero que o veto
apague mais NEUTRO do que FP, e que algum TP fraco seja apagado em pelo menos uma composição.
Chance de passar o primário: baixa (~20%); o secundário: ~30%.

Uso:  python confirmacao_portal.py --dados PASTA     # local; PASTA tem grade2min.parquet, grade2min_portal.parquet,
                                                     #   falhas.csv e piso_fisico_cache.npz (Dataset 341cecbd);
                                                     #   são ligados em scripts/pdm_fisico, onde a régua os lê
      python confirmacao_portal.py --remote          # enfileira no Cica; o worker clona esta branch
"""
from __future__ import annotations
import io, contextlib, os, sys
from pathlib import Path

DATASET_ID = "341cecbd24b24df08cdb2d575c47cbd8"     # 8b06a98f + grade2min_portal.parquet
PROJETO, FILA, IMAGEM = "TesteMLCab", "default", "python:3.12-slim"
AQUI = Path(__file__).resolve().parent

FAMILIAS = {
    "vc": ("zmax", ["954005_624_VI_0301", "954005_624_VI_0302", "954005_624_VI_0304", "954005_624_VI_0305"]),
    "es": ("zmax", ["954005_624_TI_0318", "954005_624_TI_0319"]),
    "od": ("spread", ["954005_624_TI_0320", "954005_624_TI_0321", "954005_624_TI_0322"]),
}
THETA, SENSIB, NIVEL = 1.0, (0.8, 1.2), 0.975
FIT, HL = 20_000, "1h"
INI, FIM = "2025-01-01", "2025-10-24"


def _bruto(tipo, X, ref):
    """valor cru do canal em X, com mediana/MAD tirados de `ref` (o baseline)."""
    import numpy as np, pandas as pd
    if tipo == "zmax":
        med = ref.median(); mad = ((ref - med).abs().median() * 1.4826).replace(0, np.nan)
        return ((X - med) / mad).abs().max(axis=1, skipna=False)
    cols = list(X.columns)
    dif = lambda D: pd.concat([D[c] - D[[x for x in cols if x != c]].median(axis=1) for c in cols], axis=1)
    sr, sx = dif(ref), dif(X)
    med = sr.median(); mad = ((sr - med).abs().median() * 1.4826).replace(0, np.nan)
    return ((sx - med) / mad).abs().max(axis=1, skipna=False)


def canal(R, G, vig, dia: int):
    """C(t) = max_familia EWMA/limiar, com a composição do retreino no dia `dia`. Cacheado como os sinais."""
    import numpy as np, pandas as pd
    f = R.CACHE / f"portal_C_dia{dia:02d}.npz"
    if f.exists():
        return pd.Series(np.load(f)["C"], index=R.idx)
    idx = R.idx
    meses = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC")
    cortes = [c for c in (m + pd.Timedelta(days=dia - 1) for m in meses) if idx[0] < c < idx[-1]]
    norm = []
    for nome, (tipo, tags) in FAMILIAS.items():
        raw = pd.Series(np.nan, index=idx); thr = pd.Series(np.nan, index=idx)
        for i, c0 in enumerate(cortes):
            c1 = cortes[i + 1] if i + 1 < len(cortes) else idx[-1] + pd.Timedelta("2min")
            base = G.loc[vig & (idx < c0), tags].dropna().tail(FIT)
            if len(base) < FIT // 4:
                continue
            s = (idx >= c0) & (idx < c1)
            raw[s] = _bruto(tipo, G.loc[s, tags], base).where(vig[s]).to_numpy()
            rb = _bruto(tipo, base, base)
            thr[s] = float(np.nanpercentile(rb.ewm(halflife=pd.Timedelta(HL), times=rb.index).mean(), 99))
        E = raw.ewm(halflife=pd.Timedelta(HL), times=idx).mean().where(raw.notna())
        norm.append((E / thr).rename(nome))
    C = pd.concat(norm, axis=1).max(axis=1)
    np.savez(f, C=C.to_numpy())
    return C


def veta(fin, C, theta: float):
    import numpy as np, pandas as pd
    if theta == -np.inf:
        return fin
    v = fin.to_numpy().copy(); c = C.to_numpy(); idx = fin.index
    for a, b in AV.episodios(fin):
        i0, i1 = idx.get_loc(a), idx.get_loc(b) + 1
        seg = c[i0:i1]
        if np.isfinite(seg).any() and np.nanmax(seg) < theta:
            v[i0:i1] = False
    return pd.Series(v, index=idx)


ARQS = ("grade2min.parquet", "grade2min_portal.parquet", "falhas.csv", "piso_fisico_cache.npz")


def prepara(pasta) -> None:
    """A régua lê os dados da PRÓPRIA pasta (`publica_clearml` faz chdir para ela ao ser importado).
    Liga ali os arquivos do Dataset; um arquivo já presente tem de ser idêntico, senão o resultado é outro."""
    import filecmp
    for a in ARQS:
        src, dst = Path(pasta) / a, AQUI / a
        if dst.exists():
            assert dst.resolve() == src.resolve() or filecmp.cmp(dst, src, shallow=False), f"{dst} difere do Dataset"
        else:
            dst.symlink_to(src)


def roda(log=print):
    global AV
    import numpy as np, pandas as pd
    with contextlib.redirect_stdout(io.StringIO()):
        import regua_fp as R
    AV = R.AV
    pd.set_option("display.width", 200)
    G = pd.read_parquet("grade2min_portal.parquet", columns=[t for _, ts in FAMILIAS.values() for t in ts])
    assert G.index.equals(R.idx), "grade2min_portal.parquet não está no índice da grade"
    vig = (R.DB.PP.estavel & ~R.DB.PP.blk)

    # controle 1: o publicado, janela inteira
    m1 = R.mede(R.detector(*R.sinais(1))["fin"])
    log(f"controle 1 (dia 1, janela inteira): det {m1['det']}/8, FP/mês {m1['fp_mes']:.3f}")
    assert m1["det"] == 8 and abs(m1["fp_mes"] - 0.344) < 5e-4, "a régua não reproduz o publicado"

    fins = {d: R.detector(*R.sinais(d))["fin"] for d in R.DIAS}
    Cs = {d: canal(R, G, vig, d) for d in R.DIAS}

    # a janela: corta a MEDIDA dos dois braços em FIM (alarme, eventos, tempo de operação)
    corte = R.idx < pd.Timestamp(FIM, tz="UTC")
    R.mask = R.mask & corte
    R.alvo = pd.Series([t for t in R.alvo if t < pd.Timestamp(FIM, tz="UTC")])
    with contextlib.redirect_stdout(io.StringIO()):
        import bootstrap_regua as BR
    BR.MESES = pd.date_range("2024-02-01", "2025-11-01", freq="MS", tz="UTC")
    BR.OP = BR._horas_op()
    cob = np.mean([np.isfinite(Cs[d][R.mask].to_numpy()).mean() for d in R.DIAS])
    log(f"janela {INI} .. {FIM}: eventos {[t.strftime('%d/%m') for t in R.alvo]}  |  "
        f"controle 4, cobertura de C nos instantes vigiados: {cob:.1%}")
    assert cob >= 0.95, "cobertura do canal abaixo de 95%"

    gera = lambda th: (lambda d: veta(fins[d], Cs[d], th) & corte)
    ref = BR.braco("referência", gera(-np.inf), R.DIAS)
    for d in R.DIAS:                                                      # controles 2 e 3
        assert ((fins[d] & corte) == gera(-np.inf)(d)).all()
        assert not (gera(THETA)(d) & ~(fins[d] & corte)).any()
    log("controles 2 e 3: θ = -inf reproduz a referência; a variante só apaga")

    tabela, mec = [], []
    for th in (THETA,) + SENSIB:
        var = BR.braco(f"θ={th}", gera(th), R.DIAS)
        dec = BR.decide(ref, var, pareado=True, nivel=NIVEL)
        r, v = ref.tabela, var.tabela
        sec = bool((v.det.values >= r.det.values).all() and v.inicio.median() >= r.inicio.median()
                   and v.banda.median() >= r.banda.median() and dec["fp_hi"] < 0
                   and v.carga_mes.mean() <= r.carga_mes.mean())
        tabela.append(dict(braco=f"θ={th}" + ("" if th == THETA else " (sensib.)"),
                           det=v.det.median(), inicio=v.inicio.median(), banda=v.banda.median(),
                           fp_mes=round(v.fp_mes.median(), 3), carga=round(v.carga_mes.mean(), 1),
                           d_carga=f"{dec['d_carga']:+.1f} [{dec['ic_lo']:+.1f}; {dec['ic_hi']:+.1f}]",
                           d_fp=f"{dec['d_fp']:+.3f} [{dec['fp_lo']:+.3f}; {dec['fp_hi']:+.3f}]",
                           carga_cai=f"{int((v.carga_mes.values < r.carga_mes.values).sum())}/8",
                           perde_det=int((v.det.values < r.det.values).sum()),
                           primario=dec["veredito"] if th == THETA else "-",
                           secundario=("GANHO-CONTAGEM" if sec else "não") if th == THETA else "-"))
        for d in R.DIAS:
            cr = {(a, b): k for a, b, k, _ in R.mede(gera(-np.inf)(d))["cls"]}
            cv = {(a, b) for a, b, k, _ in R.mede(gera(th)(d))["cls"]}
            for k in ("TP", "FP", "NEUTRO"):
                ap = [(a, b) for (a, b), kk in cr.items() if kk == k and (a, b) not in cv]
                mec.append(dict(theta=th, dia=d, classe=k, apagados=len(ap),
                                horas=round(sum((b - a).total_seconds() / 3600 for a, b in ap), 1)))
    r = ref.tabela
    tabela.insert(0, dict(braco="referência", det=r.det.median(), inicio=r.inicio.median(), banda=r.banda.median(),
                          fp_mes=round(r.fp_mes.median(), 3), carga=round(r.carga_mes.mean(), 1)))
    tab = pd.DataFrame(tabela).fillna("")
    mec = pd.DataFrame(mec).pivot_table(index=["theta", "dia"], columns="classe", values=["apagados", "horas"])
    log("\nRESULTADO (mediana nas 8 composições; carga = média, h/mês)\n" + tab.to_string(index=False))
    log("\nMECANISMO (episódios da referência apagados, por composição)\n" + mec.to_string())
    log("\nREFERÊNCIA por composição\n" + r[["det", "inicio", "banda", "n_fp", "fp_mes", "n_neutro", "carga_mes"]]
        .round(3).to_string())
    return tab, mec, r


def main():
    remoto = "--remote" in sys.argv
    if remoto or "--dados" not in sys.argv:
        from clearml import Dataset, Task                      # antes de qualquer leitura de argv no worker
        task = Task.init(project_name=PROJETO, task_name="detector-v2::confirmacao_portal (C-PORTAL)",
                         task_type=Task.TaskTypes.testing, reuse_last_task_id=False,
                         auto_connect_frameworks=False, tags=["portal", "pre-registrado", "regua_fp"])
        cfg = task.connect(dict(dataset_id=DATASET_ID, theta=THETA, janela=f"{INI}..{FIM}"))
        if Task.running_locally() and remoto:
            task.set_packages(["numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "matplotlib",
                               "tzdata", "clearml"])
            task.set_base_docker(IMAGEM)
            task.execute_remotely(queue_name=FILA, exit_process=True)
            return
        prepara(Dataset.get(dataset_id=cfg["dataset_id"]).get_local_copy())
        sys.path.insert(0, str(AQUI))
        lg = task.get_logger()
        tab, mec, ref = roda(lambda s: print(s, flush=True))
        lg.report_table("C-PORTAL", "resultado", iteration=0, table_plot=tab)
        lg.report_table("C-PORTAL", "mecanismo", iteration=0, table_plot=mec.reset_index())
        lg.report_table("C-PORTAL", "referência por composição", iteration=0, table_plot=ref.reset_index())
        task.upload_artifact("resultado", tab); task.upload_artifact("mecanismo", mec.reset_index())
        task.flush(wait_for_uploads=True); task.close()
        return
    prepara(sys.argv[sys.argv.index("--dados") + 1])
    sys.path.insert(0, str(AQUI))
    roda()


if __name__ == "__main__":
    main()
