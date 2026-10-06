#!/usr/bin/env python3
"""V-PORTAL: os canais do Portal de Integridade como 5º canal do VOTO (não como veto).

PRÉ-REGISTRADO EM 06/10/2026, ANTES DE RODAR (commit deste arquivo sem resultado).

DE ONDE VEM. O C-PORTAL (`confirmacao_portal.py`, task a7ebb645) usou o mesmo canal como VETO de episódio e reprovou:
o FP caiu (0,972 -> 0,416/mês, IC < 0), mas 4 das 8 composições perderam detecção -- no nível do episódio inteiro o
canal não separa TP de FP. A triagem (`tags_ausentes.py`) mostrou sinal nas 48 h ANTES dos trips; a pergunta aqui é a
outra: o canal ajuda o detector a NASCER (antecipar), votando junto com t, p, sp e vb?
Mesma ressalva: canal escolhido nos mesmos 5 eventos em que é testado; resultado favorável é INDICATIVO, a validação é
o export de 2025-11 em diante ou o modo sombra. E este é o segundo desenho sobre os mesmos eventos (o primeiro reprovou).

O CANAL `pt`. O C do C-PORTAL, sem mudança: max sobre vibração do compressor (VI_0301/02/04/05), mancal de escora
(TI_0318/0319) e óleo de dreno (TI_0320/21/22, contra a mediana dos irmãos) de EWMA(1 h) / p99 do próprio baseline,
com a composição do retreino no dia `d`. Já está na escala "1 = p99 do baseline": BASE_pt = 1.

A REGRA (o mínimo que muda).
  · Nível A (sensível): >= 3 de 5 canais (t, p, sp, vb, pt) acima de K_LO.   [era >= 3 de 4]
  · Nível B (específico): >= 2 de 5 acima de KH, exigindo sp OU vb OU pt.    [era >= 2 de 4, sp ou vb]
  · Para pt: K_LO = KH = θ = 1,0 (o p99 do baseline), fixado agora. θ = 0,8 e 1,2 como SENSIBILIDADE, sem veredito.
  · pt tem SUSTAIN e CUSUM como os outros canais (a função de canal é a mesma).
  · NÃO MUDA: a força F (escalada por idade, "forte" do refratário) continua sobre t, p, sp, vb; refratário de 72 h,
    duração mínima, blackout, máscara, limiares dos quatro canais atuais.

A JANELA. A do C-PORTAL: os dois braços medidos de 2025-01-01 a 2025-10-24, 5 eventos (27/02, 17/03, 07/04, 11/04,
29/04), 8 composições, bootstrap pareado com blocos mensais de 2024-02 a 2025-10, IC de 97,5%.

A DECISÃO.
  PRIMÁRIO (a régua de sempre, `bootstrap_regua.decide`): nenhuma composição perde detecção, medianas de início e
    banda não caem, carga cai em >= 7/8 com IC abaixo de zero, FP/mês mediano não sobe.
  SECUNDÁRIO (o alvo deste desenho, DETECÇÃO): nenhuma composição perde detecção, a mediana de início OU a de banda
    SOBE, o FP/mês mediano não sobe e o IC da Δcarga tem limite inferior <= 0 (a carga não sobe com certeza).
  Se só o secundário passar: "antecipa sem custo de FP", candidato a validação em dado novo, não decisão de produção.

CONTROLES (o experimento não vale se falharem).
  1. A régua reproduz o publicado no dia 1, janela inteira: 8/8 e 0,344 FP/mês.
  2. Com pt desligado, este detector reproduz `regua_fp.detector` bit a bit nas 8 composições.
  3. Cobertura de pt nos instantes vigiados da janela >= 95%.

CHECAGEM DE MECANISMO. Por composição: episódios novos e perdidos por classe; para cada evento, o lead (h) do primeiro
episódio na janela de 48 h, referência contra variante.

EXPECTATIVA REGISTRADA. pt fica acima de 1 em ~19% do tempo vigiado (C-PORTAL, composição 1), mais que os canais
atuais no limiar baixo. Com >= 3 de 5 o nível A fica mais fácil: espero MAIS episódios e FP/mês subindo, o que
reprova os dois critérios. Chance de passar o primário: ~5%; o secundário: ~20%.

Uso:  python voto_portal.py --dados PASTA      # local (ver confirmacao_portal.py)
      python voto_portal.py --remote           # enfileira no Cica; o worker clona esta branch
"""
from __future__ import annotations
import io, contextlib, os, sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import confirmacao_portal as CP                      # canal pt, prepara, janela -- o mesmo do C-PORTAL

PROJETO, FILA, IMAGEM = CP.PROJETO, CP.FILA, CP.IMAGEM
THETA, SENSIB, NIVEL = 1.0, (0.8, 1.2), 0.975


def detector_pt(R, t, p, ms, ds, C, theta: float | None):
    """`regua_fp.detector` (caminho padrão) com o 5º canal pt; theta=None desliga pt (controle 2)."""
    import numpy as np, pandas as pd
    DB, idx = R.DB, R.idx
    m_d, rst = R.mask, DB.reset
    z = np.load("piso_fisico_cache.npz")
    spv = np.abs((z["b_all"] - ms) / ds)
    cru = pd.DataFrame({"t": t, "p": p, "sp": spv, "vb": DB.cru_pub["vb"].to_numpy()}, index=idx)
    EW = {c: cru[c].ewm(halflife=pd.Timedelta(h), times=idx).mean() for c, h in DB.HL.items()}
    EW["pt"] = C
    BASE = dict(DB.BASE, pt=1.0)
    KLO = dict(DB.K_LO, pt=theta)
    KHI = dict(DB.KH, pt=theta)

    def canal(c, k):
        thr = BASE[c] * k
        E = EW[c].where(m_d)
        deg = ((E > thr).astype(int).rolling(DB.SUSTAIN, min_periods=DB.SUSTAIN).sum() >= DB.SUSTAIN)
        x = ((E / thr).clip(upper=20) - DB.KAPPA).fillna(0.0).to_numpy()
        cu = pd.Series(R.cusum_var(x, rst, DB.H_CUSUM) > DB.H_CUSUM, index=idx)
        return (deg | cu) & m_d

    sin = list(DB.SIN) + ([] if theta is None else ["pt"])
    A = {c: canal(c, KLO[c]) for c in sin}
    B = {c: canal(c, KHI[c]) for c in sin}
    vA = pd.Series(sum(A[c].astype(int) for c in sin) >= DB.VOTO_LO, index=idx) & m_d
    porta = B["sp"] | B["vb"] | (B["pt"] if "pt" in B else False)
    vB = pd.Series(sum(B[c].astype(int) for c in sin) >= DB.VOTO_HI, index=idx) & m_d & porta
    F = pd.concat([EW[c].where(m_d) / (DB.BASE[c] * DB.KH[c]) for c in DB.SIN], axis=1).max(axis=1)
    return R._pos(vA | vB, F)


def roda(log=print):
    import numpy as np, pandas as pd
    with contextlib.redirect_stdout(io.StringIO()):
        import regua_fp as R
    pd.set_option("display.width", 220)
    G = pd.read_parquet("grade2min_portal.parquet", columns=[t for _, ts in CP.FAMILIAS.values() for t in ts])
    assert G.index.equals(R.idx)
    vig = (R.DB.PP.estavel & ~R.DB.PP.blk)

    m1 = R.mede(R.detector(*R.sinais(1))["fin"])
    log(f"controle 1 (dia 1, janela inteira): det {m1['det']}/8, FP/mês {m1['fp_mes']:.3f}")
    assert m1["det"] == 8 and abs(m1["fp_mes"] - 0.344) < 5e-4

    sig = {d: R.sinais(d) for d in R.DIAS}
    Cs = {d: CP.canal(R, G, vig, d) for d in R.DIAS}
    fins = {}
    for d in R.DIAS:
        ref = R.detector(*sig[d])["fin"]
        assert (detector_pt(R, *sig[d], Cs[d], None) == ref).all(), f"controle 2 falhou na composição {d}"
        fins[(d, None)] = ref
        for th in (THETA,) + SENSIB:
            fins[(d, th)] = detector_pt(R, *sig[d], Cs[d], th)
    log("controle 2: com pt desligado o detector reproduz regua_fp.detector nas 8 composições")

    FIMt = pd.Timestamp(CP.FIM, tz="UTC")
    corte = R.idx < FIMt
    R.mask = R.mask & corte
    R.alvo = pd.Series([t for t in R.alvo if t < FIMt])
    with contextlib.redirect_stdout(io.StringIO()):
        import bootstrap_regua as BR
    BR.MESES = pd.date_range("2024-02-01", "2025-11-01", freq="MS", tz="UTC")
    BR.OP = BR._horas_op()
    BR.W_BOOT = BR.pesos()
    cob = np.mean([np.isfinite(Cs[d][R.mask].to_numpy()).mean() for d in R.DIAS])
    log(f"janela {CP.INI} .. {CP.FIM}: eventos {[t.strftime('%d/%m') for t in R.alvo]}  |  "
        f"controle 3, cobertura de pt nos instantes vigiados: {cob:.1%}")
    assert cob >= 0.95

    gera = lambda th: (lambda d: fins[(d, th)] & corte)
    ref = BR.braco("referência", gera(None), R.DIAS)
    r = ref.tabela
    tabela, mec, leads = [], [], []
    J = pd.Timedelta(hours=48)

    def lead(fin, t):
        eps = [a for a, _ in R.AV.episodios(fin) if t - J <= a <= t]
        return round((t - min(eps)).total_seconds() / 3600, 1) if eps else None

    for th in (THETA,) + SENSIB:
        var = BR.braco(f"θ={th}", gera(th), R.DIAS)
        v = var.tabela
        dec = BR.decide(ref, var, pareado=True, nivel=NIVEL)
        sec = bool((v.det.values >= r.det.values).all()
                   and (v.inicio.median() > r.inicio.median() or v.banda.median() > r.banda.median())
                   and v.fp_mes.median() <= r.fp_mes.median() + 1e-9 and dec["ic_lo"] <= 0)
        tabela.append(dict(braco=f"θ={th}" + ("" if th == THETA else " (sensib.)"),
                           det=v.det.median(), inicio=v.inicio.median(), banda=v.banda.median(),
                           fp_mes=round(v.fp_mes.median(), 3), carga=round(v.carga_mes.mean(), 1),
                           d_carga=f"{dec['d_carga']:+.1f} [{dec['ic_lo']:+.1f}; {dec['ic_hi']:+.1f}]",
                           d_fp=f"{dec['d_fp']:+.3f} [{dec['fp_lo']:+.3f}; {dec['fp_hi']:+.3f}]",
                           perde_det=int((v.det.values < r.det.values).sum()),
                           ganha_det=int((v.det.values > r.det.values).sum()),
                           primario=dec["veredito"] if th == THETA else "-",
                           secundario=("GANHO-DETECÇÃO" if sec else "não") if th == THETA else "-"))
        for d in R.DIAS:
            cr = {(a, b): k for a, b, k, _ in R.mede(gera(None)(d))["cls"]}
            cv = {(a, b): k for a, b, k, _ in R.mede(gera(th)(d))["cls"]}
            for k in ("TP", "FP", "NEUTRO"):
                mec.append(dict(theta=th, dia=d, classe=k,
                                novos=sum(1 for e, kk in cv.items() if kk == k and e not in cr),
                                perdidos=sum(1 for e, kk in cr.items() if kk == k and e not in cv)))
            if th == THETA:
                for t in R.alvo:
                    leads.append(dict(dia=d, evento=t.strftime("%d/%m"),
                                      lead_ref=lead(gera(None)(d), t), lead_pt=lead(gera(th)(d), t)))
    tabela.insert(0, dict(braco="referência", det=r.det.median(), inicio=r.inicio.median(),
                          banda=r.banda.median(), fp_mes=round(r.fp_mes.median(), 3),
                          carga=round(r.carga_mes.mean(), 1)))
    tab = pd.DataFrame(tabela).fillna("")
    mec = pd.DataFrame(mec).pivot_table(index=["theta", "dia"], columns="classe", values=["novos", "perdidos"])
    leads = pd.DataFrame(leads).pivot_table(index="evento", columns="dia", values=["lead_ref", "lead_pt"])
    log("\nRESULTADO (mediana nas 8 composições; carga = média, h/mês)\n" + tab.to_string(index=False))
    log("\nMECANISMO (episódios novos e perdidos contra a referência, por composição)\n" + mec.to_string())
    log("\nLEAD do primeiro episódio na janela de 48 h (h; vazio = não nasceu)\n" + leads.to_string())
    return tab, mec, leads


def main():
    remoto = "--remote" in sys.argv
    if remoto or "--dados" not in sys.argv:
        from clearml import Dataset, Task
        task = Task.init(project_name=PROJETO, task_name="detector-v2::voto_portal (V-PORTAL)",
                         task_type=Task.TaskTypes.testing, reuse_last_task_id=False,
                         auto_connect_frameworks=False, tags=["portal", "pre-registrado", "regua_fp"])
        cfg = task.connect(dict(dataset_id=CP.DATASET_ID, theta=THETA, janela=f"{CP.INI}..{CP.FIM}"))
        if Task.running_locally() and remoto:
            task.set_packages(["numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "matplotlib",
                               "tzdata", "clearml"])
            task.set_base_docker(IMAGEM)
            task.execute_remotely(queue_name=FILA, exit_process=True)
            return
        CP.prepara(Dataset.get(dataset_id=cfg["dataset_id"]).get_local_copy())
        lg = task.get_logger()
        tab, mec, leads = roda(lambda s: print(s, flush=True))
        lg.report_table("V-PORTAL", "resultado", iteration=0, table_plot=tab)
        lg.report_table("V-PORTAL", "mecanismo", iteration=0, table_plot=mec.reset_index())
        lg.report_table("V-PORTAL", "leads", iteration=0, table_plot=leads.reset_index())
        task.upload_artifact("resultado", tab)
        task.flush(wait_for_uploads=True); task.close()
        return
    CP.prepara(sys.argv[sys.argv.index("--dados") + 1])
    roda()


if __name__ == "__main__":
    main()
