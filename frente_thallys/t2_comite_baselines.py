#!/usr/bin/env python3
"""T2 -- comitê de baselines no detector v2 do Thallys: a mediana de t, p e sp de vários retreinos mensais.

PRÉ-REGISTRO (25/10/2026, commitado antes de construir os membros; nenhum membro além do de produção foi olhado)
Pergunta: o v2 é frágil à escolha do baseline do retreino mensal. O próprio `constroi_bundle.py` mede: mudar o corte
do dia 1 para o dia 8 leva 8/8 -> 5/8 e 6,6 -> 77 h/mês de alarme falso; no PCA fiel, n = 20.000 -> 19.000 leva
7/8 FP 9 -> 5/8 FP 11. Parte dos FP pode ser "sorte do corte". A mediana de vários baselines por mês reduz FP e
deixa o detector menos dependente de um corte?

MEMBRO (n, atraso): o bundle de cada mês 2025-01..2026-04 feito pelo `constroi_bundle.py` do pacote, sem editá-lo,
com FIT_POINTS = n e o baseline terminando >= `atraso` dias antes do dia 1. Implementação: as linhas de
[dia 1 - atraso, dia 1) saem da entrada do ajuste. O mês servido, e portanto a vigência, não muda; só os dados do
ajuste. Seis membros: (19.000, 0), (20.000, 0) = produção, (21.000, 0), (20.000, 7), (20.000, 14), (20.000, 21).
Os sinais de cada membro saem do replay causal semanal (`reproduz.py sinais`), como o controle.
COMITÊ: em cada instante, t, p e sp = mediana entre os membros. vb, mask e reset não dependem do bundle e vêm do de
produção (verificado: V4). A decisão é a de produção (`decide_v2`, detector.json do último bundle), sem mudar k,
voto, CUSUM, refratário nem duração.
PRINCIPAL: mediana de 5 membros: (19.000, 0), (20.000, 0), (21.000, 0), (20.000, 7), (20.000, 14).
VIZINHANÇA (5): só n {19, 20, 21 mil; atraso 0}; só atraso {0, 7, 14, 21; n 20.000}; os 6 membros; média dos 5
principais em vez da mediana; 3 membros {(20.000, 0), (20.000, 7), (20.000, 14)}.
CRITÉRIOS (fixados antes):
  (a) META: principal com perda <= 1 detecção contra o publicado (7/8) E FP <= 6.
  (b) ESTABILIDADE: o principal tem detecção >= a mediana das detecções dos 6 membros sozinhos E FP <= a mediana dos FP
      deles; E entre as 5 vizinhas a detecção varia no máximo 1 (máx - mín <= 1).
  (c) VIZINHANÇA: >= 3 das 5 vizinhas com detecção >= 6/8 e FP <= 7.
  PROMISSOR = (a) e (b) e (c). ESTABILIZA = (b) e (c) sem (a): ganho de robustez, não de FP. NÃO = o resto.
  Nada se adota sem a sombra: o comitê muda o detector congelado.
SECUNDÁRIOS (só relato): cada membro sozinho (det, FP, horas); a regra de carga do EXP 62 (principal) sobre o comitê
principal e sobre cada membro; o "T1 OU carga" (pós-hoc do T1) sobre o comitê e sobre a base n = 19.000.
EXPECTATIVAS:
  Os membros sozinhos variam: detecção de 5 a 8 (~70%), FP de 8 a 14 (~60%). A mediana corta picos de resíduo de um
  baseline ruim, mas também alguns precursores. Prevejo FP 9 -> 7-9 (~55%); detecção do principal 6-7 (~65%);
  (a) vale (~20%); (b) vale (~50%); (c) vale (~55%). Veredito NÃO (~50%), ESTABILIZA (~35%), PROMISSOR (~15%).
LIMITES: 8 trips; uma base de dados; os membros compartilham a maior parte dos dados de ajuste (não são independentes);
o atraso de 21 d encosta no SPAN_MAX/FOLGA_MAX do pacote (alerta de faixa histórica, código 4), registrado e não
bloqueado.
VERIFICAÇÕES: V1 os bundles do membro (20.000, 0) são idênticos, byte a byte, aos de modelos_local; V2 os sinais do
membro (20.000, 0) são idênticos aos do controle (t, p, sp, vb, mask, reset); V3 o comitê de 1 membro (produção) dá
7/8, FP 9, 1.457 h; V4 vb, mask e reset são idênticos entre os 6 membros.

RESULTADO (25/10/2026; local, pré-registro d9e1a20). V1-V4 PASSAM. NÃO: (a) não, (b) ok, (c) 0/5.
  MEMBROS SOZINHOS: (19 mil, 0) 5/8 FP 13 | (20 mil, 0) = produção 7/8 FP 9 | (21 mil, 0) 6/8 FP 12 |
    (20 mil, 7 d) 7/8 FP 16 | (20 mil, 14 d) 7/8 FP 17 | (20 mil, 21 d) 7/8 FP 22.
    O ponto publicado é o MELHOR dos 6. Qualquer desvio do baseline, em tamanho ou em atraso, sobe o FP (12 a 22) ou
    derruba a detecção (5-6/8). O 9 FP publicado é o vale de uma região que sobe dos dois lados.
    Alerta do pacote (folga > 7 d): os 16 meses de 14 d e de 21 d e 1 mês de 7 d. O pré-registro só previa o de 21 d.
  PRINCIPAL (mediana de 5): 7/8 FP 11 1.577 h (0,986/mês), início 6, banda 5.
  VIZINHANÇA: só n 7/8 FP 9 | só atraso 8/8 FP 15 | 6 membros 8/8 FP 13 | média 7/8 FP 15 | 3 atrasos 8/8 FP 13.
    A mediana não corta FP: cada baseline erra em episódios diferentes, e a mediana herda os de quem está no meio.
  SECUNDÁRIOS (pré-registrados como relato):
    A carga do EXP 62 reduz o FP em TODOS os 6 membros: 13 -> 7, 9 -> 4, 12 -> 5, 16 -> 6, 17 -> 10, 22 -> 13. A
      detecção só cai num deles (19 mil: 5 -> 4). São 5 bases que o Thallys não viu: a carga é a alavanca estável.
    Comitê + carga: 7/8 FP 3 (1.189 h). Comitê + (T1 OU carga): 7/8 FP 1 (1.049 h).
    Base n = 19.000 + (T1 OU carga): 5/8 FP 3 (1.218 h), contra 5/8 FP 6 só com a carga. Sem perda de detecção, e é a
      segunda base em que a combinação (vista pós-hoc no T1) aparece.
PLACAR: ACERTEI: detecção dos membros 5-8 (~70%; 5-7); detecção do principal 6-7 (~65%; 7); (a) não vale (~80%);
  (b) vale (~50%); veredito NÃO (~50%). ERREI: FP dos membros 8-14 (~60%; 9-22); FP do principal 7-9 (~55%; 11);
  (c) vale (~55%; 0/5). Não previ que a produção fosse o melhor membro.

Uso:  python frente_thallys/t2_comite_baselines.py membros   # 6 x 16 bundles + 6 replays (~15 min)
      python frente_thallys/t2_comite_baselines.py avalia
"""
from __future__ import annotations
import argparse, contextlib, filecmp, io, os, subprocess, sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(AQUI), str(AQUI.parent / "frente_ocsvm")]
import reproduz as R  # noqa: E402  (antes do nv: fixa o NV_DADOS)
import base_n19000 as B  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MEMBROS = [(19000, 0), (20000, 0), (21000, 0), (20000, 7), (20000, 14), (20000, 21)]
PROD = (20000, 0)
PRINC = dict(membros=[(19000, 0), (20000, 0), (21000, 0), (20000, 7), (20000, 14)], agrega="mediana")
VIZ = {"só n": dict(membros=[(19000, 0), (20000, 0), (21000, 0)], agrega="mediana"),
       "só atraso": dict(membros=[(20000, 0), (20000, 7), (20000, 14), (20000, 21)], agrega="mediana"),
       "6 membros": dict(membros=MEMBROS, agrega="mediana"),
       "média dos 5": dict(PRINC, agrega="média"),
       "3 atrasos": dict(membros=[(20000, 0), (20000, 7), (20000, 14)], agrega="mediana")}
PK = R.DADOS / "nvdados" / "estrutura_pra_prod" / "Cabiunas"
PASTA = R.DADOS / "membros"


def nome(m):
    return f"n{m[0]}_atraso{m[1]}"


def membros():
    sys.path.insert(0, str(PK / "scripts"))
    import constroi_bundle as cb
    G = cb._ler_historico(R.DADOS / "raw30s_a97_2024_2026.parquet")      # a grade do ajuste, uma vez só
    for n, atraso in MEMBROS:
        d = PASTA / nome((n, atraso)); (d / "modelos").mkdir(parents=True, exist_ok=True)
        cb.FIT_POINTS = n
        for mes in R.MESES:
            corte = pd.Timestamp(mes + "-01", tz="UTC")
            g = G[~((G.index >= corte - pd.Timedelta(days=atraso)) & (G.index < corte))] if atraso else G
            cb._ler_historico = lambda _c, g=g: g
            sys.argv = ["constroi_bundle.py", "--historico", "grade.parquet", "--mes", mes,
                        "--trips", str(PK / "registro_trips.csv"), "--saida", str(d / "modelos")]
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                rc = cb.main()
            if rc != 0:
                print(f"   {nome((n, atraso))} {mes}: código {rc} | " + " ".join(l for l in buf.getvalue().splitlines() if "ALERTA" in l or "ERRO" in l))
        arv = d / "arvore"; c = arv / "estrutura_pra_prod" / "Cabiunas"; c.mkdir(parents=True, exist_ok=True); (arv / "scripts").mkdir(exist_ok=True)
        for x in ("scripts", "dados", "registro_trips.csv"):
            (c / x).unlink(missing_ok=True); (c / x).symlink_to(PK / x)
        (c / "modelos").unlink(missing_ok=True); (c / "modelos").symlink_to(d / "modelos")
        (arv / "scripts" / "pdm_fisico").unlink(missing_ok=True)
        (arv / "scripts" / "pdm_fisico").symlink_to((R.DADOS / "nvdados" / "scripts" / "pdm_fisico").resolve())
        subprocess.run([sys.executable, str(AQUI / "reproduz.py"), "sinais", "--saida", str(d / "sinais.parquet")],
                       env=dict(os.environ, NV_DADOS=str(arv)), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(nome((n, atraso)), "pronto", flush=True)


def proc_de(m):
    p = pd.read_parquet(PASTA / nome(m) / "sinais.parquet")[["t", "p", "sp", "vb", "mask", "reset"]].copy()
    p["mask"] = p["mask"].astype(bool); p["reset"] = p["reset"].astype(bool); return p


def comite(P, cfg):
    base = P[PROD].copy()
    for c in ("t", "p", "sp"):
        X = pd.concat([P[m][c] for m in cfg["membros"]], axis=1)
        base[c] = X.median(axis=1, skipna=False) if cfg["agrega"] == "mediana" else X.mean(axis=1, skipna=False)
    return base


def alarme(proc, det, idx):
    from nv.plugin import decide_v2
    with contextlib.redirect_stdout(io.StringIO()):
        return decide_v2(proc, det)["is_anomaly"].astype(bool).reindex(idx).fillna(False)


def mais_regras(al, vig, ci):
    """Secundários: carga do EXP 62 (principal) e T1 OU carga, pelas funções já conferidas."""
    import t1_canal4_no_v2 as T1
    import comum as C
    E62 = B._exp("exp62", "62_regra_de_carga.py")
    from nv import io as nvio
    g = nvio.carrega_grade(); t5 = g["T5_AVG_A"].where(nvio.mascaras(g)["estavel"])
    eps = [(a, b) for a, b in ci._episodios(al.loc[R.INICIO8:])]
    L = [E62.L_de(t5, a, 30, 2)[0] for a, _ in eps]
    conf = T1.confirma(T1.canal4(al.index, C.catalogo(), T1.PRINC), eps, 2.0)
    m_c = R.mede(E62.aplica(al, eps, L, 0.5), vig)
    m_t = R.mede(T1.aplica(al, eps, [cf and not (l < 0.5) for cf, l in zip(conf, L)]), vig)
    return m_c, m_t


def avalia():
    ci, det = R.modelos_ci()
    al0, vig = R.carrega_controle(); idx = al0.index
    P = {m: proc_de(m) for m in MEMBROS}
    ok = {}
    a = PASTA / nome(PROD) / "modelos"; b = R.DADOS / "modelos_local"
    ok["V1"] = sorted(os.listdir(a)) == sorted(os.listdir(b)) and all(
        filecmp.cmp(a / d / f, b / d / f, shallow=False) for d in os.listdir(b) for f in os.listdir(b / d))
    ref = pd.read_parquet(R.DADOS / "sinais_v2_8ev_local_treino.parquet")[["t", "p", "sp", "vb", "mask", "reset"]]
    ok["V2"] = bool(P[PROD].astype(float).equals(ref.astype(float)))
    m_prod = R.mede(alarme(comite(P, dict(membros=[PROD], agrega="mediana")), det, idx), vig)
    ok["V3"] = (m_prod["det"], m_prod["fp"], round(m_prod["horas"])) == (7, 9, 1457)
    ok["V4"] = all(P[m][c].astype(float).equals(P[PROD][c].astype(float)) for m in MEMBROS for c in ("vb", "mask", "reset"))
    print(f"verificações: {ok}", flush=True)
    if not all(ok.values()):
        print("verificação falhou -- parando"); return

    print("\nMEMBROS SOZINHOS (det/8 · FP · horas | + carga EXP 62)")
    dets, fps = [], []
    for m in MEMBROS:
        al = alarme(P[m], det, idx); r = R.mede(al, vig); rc, _ = mais_regras(al, vig, ci); dets.append(r["det"]); fps.append(r["fp"])
        print(f"   {nome(m):16s} {r['det']}/8 · FP {r['fp']:2d} · {r['horas']:5.0f} h | + carga {rc['det']}/8 · FP {rc['fp']:2d}", flush=True)
    al_p = alarme(comite(P, PRINC), det, idx); r = R.mede(al_p, vig)
    print(f"\nPRINCIPAL (mediana de 5): {r['det']}/8 · FP {r['fp']} · {r['horas']:.0f} h · FP/mês {r['fp_mes']:.3f} · "
          f"início {r['inicio']} · banda {r['banda']} | perdidos contra o publicado {[e for e in m_prod['ev'] if m_prod['ev'][e] and not r['ev'][e]]}")
    print("\nVIZINHANÇA")
    rv = {}
    for k, cfg in VIZ.items():
        rv[k] = R.mede(alarme(comite(P, cfg), det, idx), vig)
        print(f"   {k:12s} {rv[k]['det']}/8 · FP {rv[k]['fp']:2d} · {rv[k]['horas']:5.0f} h")
    a_ = r["fp"] <= 6 and m_prod["det"] - r["det"] <= 1
    dv = [x["det"] for x in rv.values()]
    b_ = r["det"] >= np.median(dets) and r["fp"] <= np.median(fps) and max(dv) - min(dv) <= 1
    c_ = sum(x["det"] >= 6 and x["fp"] <= 7 for x in rv.values()) >= 3
    cl = "PROMISSOR" if (a_ and b_ and c_) else "ESTABILIZA" if (b_ and c_) else "NÃO"
    print(f"\nCLASSIFICAÇÃO: (a) meta {'ok' if a_ else 'não'} | (b) estabilidade {'ok' if b_ else 'não'} "
          f"(membros: det mediana {np.median(dets)}, FP mediana {np.median(fps)}; vizinhas det {min(dv)}-{max(dv)}) | "
          f"(c) vizinhança {sum(x['det'] >= 6 and x['fp'] <= 7 for x in rv.values())}/5 -> {cl}")
    rc, rt = mais_regras(al_p, vig, ci)
    print(f"\nSECUNDÁRIO: comitê + carga {rc['det']}/8 · FP {rc['fp']} · {rc['horas']:.0f} h | comitê + (T1 OU carga) "
          f"{rt['det']}/8 · FP {rt['fp']} · {rt['horas']:.0f} h")
    a19 = B.alarme_n(19000, det); rc, rt = mais_regras(a19, vig, ci)
    print(f"SECUNDÁRIO: base n=19.000 + (T1 OU carga) {rt['det']}/8 · FP {rt['fp']} · {rt['horas']:.0f} h "
          f"(só carga: {rc['det']}/8 · FP {rc['fp']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("acao", choices=["membros", "avalia"])
    {"membros": membros, "avalia": avalia}[ap.parse_args().acao]()
