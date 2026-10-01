#!/usr/bin/env python3
"""F1: o sensor que o monitor sinalizou sai do canal até o próximo bundle.

PRÉ-REGISTRADO em `bootstrap_regua.py` (commit a6ddd6b), antes de rodar. Regra F
(segurança): adotado se a detecção estrita vale e as estimativas pontuais de carga e
de FP (média entre as 8 composições) não sobem. Ganho só se o IC da carga excluir 0.

OS GATILHOS, os mesmos de `cabiunas_inference.monitor_drift`, checados todo dia às
00:00 UTC em cada composição, contra o bundle em vigor naquele dia:
  degrau   a tag está a >= 10 sigma do centro do baseline (sigma = IQR/1,349) na
           última semana vigiada E na anterior, mesmo sinal; NENHUMA outra tag do
           mesmo canal >= 10 sigma na última semana; e a checagem cai até 21 dias
           depois do fim de uma manutenção (HSX >= 24 h). [A condição "nenhuma outra
           >= 10 sigma" foi escolhida depois de ver o caso de nov/2025, n = 1.]
  travado  o IQR das 3 últimas semanas abaixo de 10% da mediana do IQR semanal das
           semanas anteriores (até 13 semanas, >= 4 com dado), sem condição de
           manutenção.
A EXCLUSÃO. Do instante do gatilho até o próximo corte, o canal (t ou p) é pontuado
por um bundle refeito SEM o sensor: mesmo baseline, PCA e p99 recalculados juntos.
Gatilhos seguintes no mesmo período acumulam. O canal sp e o vb não mudam.

RESULTADO (01/10/2026) -- REPROVADO pela regra pré-registrada (A).

    composição      1     4     8    11    15    18    22    25
    det           8->8  6->5  5->5  5->5  7->7  6->7  7->7  7->7
    banda         5->5  2->2  3->3  3->4  4->5  5->6  5->5  5->5
    carga          49->48 123->83 107->103 163->136 185->113 180->113 126->108 135->114
  Δcarga -31,3 [-61,1; -8,9] (IC exclui zero) · ΔFP -0,108 [-0,223; 0,000]
  A composição 4 perde o trip de 09/12/2025, e isso reprova pela regra estrita.

  · OS GATILHOS NÃO SÃO O QUE SE ESPERAVA. Degrau com as condições do F1: só três
    (PDIT_0305 em 24/01/2025 numa composição, PDI_0302 em 30/04/2025, PDI_0301 em
    25/11/2025). O grosso vem do "travado": PI_0319 em 53 dias (abr/2025 a
    jan/2026) e PDI_0301 em 22 (dez/2025 a jan/2026) -- 55 de 427 dias com operação
    (13%), checando todo dia. A validação semanal anterior (6 de 59 semanas, só após
    nov/2025) subestimou.
  · O PI_0319 NÃO ESTÁ TRAVADO. É sensor de estados: o IQR semanal é ~0,3, ~0,007 ou
    ~45 conforme a linha de gás do motor de partida esteja pressurizada. As semanas
    de 45 inflam o "típico" e as planas parecem travamento. É falso alarme do
    monitor de produção, em ~13% dos dias.
  · DECOMPOSIÇÃO (a posteriori): só o gatilho de degrau dá Δcarga -4,6 [-14,7; 0,0]
    e AINDA perde o 09/12 na composição 4. ~85% do corte de carga vem de tirar o
    PI_0319 do canal p.
  · O MECANISMO DA PERDA. Na composição 4, o 09/12 não tinha precursor no p: nas 48 h
    antes do trip o EWMA do p está em 0,14-0,17 do limiar. O canal estava aceso
    pela MEMÓRIA do CUSUM, acumulada enquanto o bundle de novembro pontuou o degrau
    pós-manutenção (o p passou de 350x o limiar no fim de novembro e seguia a ~7x,
    100% PDI_0301, até a troca de 04/12). O bundle de dezembro vê o p normal; o
    acumulador levou o artefato 4 dias para dentro dele (`memoria_nas_comparacoes.py`).
    A exclusão encurta o artefato, o CUSUM esvazia antes, e o voto (vb + p) não fecha.
    A regra estrita conta como perda; fica registrado que era detecção sustentada por
    artefato de instrumento. A regra não foi mudada depois de ver isso.
  · CONSEQUÊNCIAS. Não excluir sensor do canal. Ficam: (1) a notificação à
    instrumentação, que o monitor já produz; (2) corrigir o falso "travado" do
    PI_0319 no monitor. Tirar o PI_0319 do canal p de vez (fisicamente irrelevante
    com a máquina rodando) é hipótese NOVA, nascida deste resultado: só com
    pré-registro próprio.

Uso:  PYTHONPATH=. python f1_exclusao.py
"""
from __future__ import annotations
import io, contextlib
import numpy as np, pandas as pd
import regua_fp as R
import aprovacao_operador as AO
import bootstrap_regua as BR
with contextlib.redirect_stdout(io.StringIO()):
    import drift_eventos as DE

C, DF, IX, STABLE, FIT, DC = AO.C, AO.DF, AO.IX, AO.STABLE, AO.FIT, AO.DC
FAM = {"t": list(C.TEMPERATURE_TAGS), "p": list(C.PRESSURE_TAGS)}
COLS = FAM["t"] + FAM["p"]
L_SIG, POS_MANUT = 10.0, pd.Timedelta(days=21)
TRAV_FRAC, TRAV_SEM, TRAV_REF = 0.10, 3, 4
SEMANA = pd.Timedelta(days=7)
CURTO = lambda c: c.replace("954005_624_", "")
MAN = DE.manutencoes()
DIAS = pd.date_range(pd.Timestamp("2024-01-08", tz="UTC"), IX[-1].normalize(), freq="D")


def estatisticas() -> dict:
    """mediana, IQR e contagem vigiada da janela (D - 7 d, D] para cada dia D."""
    f = R.CACHE / "f1_estat_diarias.npz"
    if f.exists():
        z = np.load(f); return {k: z[k] for k in z.files}
    X = DF[COLS].to_numpy(dtype=float); vig = AO.MASK
    n = len(DIAS); k = len(COLS)
    med, q25, q75 = (np.full((n, k), np.nan) for _ in range(3)); cnt = np.zeros(n)
    for i, D in enumerate(DIAS):
        a, b = IX.searchsorted(D - SEMANA, side="right"), IX.searchsorted(D, side="right")
        v = X[a:b][vig[a:b]]
        cnt[i] = len(v)
        if len(v):
            med[i] = np.nanmedian(v, axis=0)
            q25[i], q75[i] = np.nanpercentile(v, [25, 75], axis=0)
    np.savez(f, med=med, iqr=q75 - q25, cnt=cnt)
    return dict(med=med, iqr=q75 - q25, cnt=cnt)


ST = estatisticas()


def gatilhos(dia: int) -> list[dict]:
    cs = [c for c in AO.cortes(dia) if AO.bundle(c) is not None]
    out = []
    for i, D in enumerate(DIAS):
        if D < cs[0] or ST["cnt"][i] < 720:
            continue
        c = max(x for x in cs if x <= D)
        bd = AO.bundle(c)
        i0 = i - 7
        for j, f in enumerate(("t", "p")):
            sc = bd[j]; cols = FAM[f]
            pos = [COLS.index(x) for x in cols]
            sig = sc.scaler.scale_ / 1.349
            d1 = (ST["med"][i, pos] - sc.scaler.center_) / sig
            d0 = ((ST["med"][i0, pos] - sc.scaler.center_) / sig) if i0 >= 0 and ST["cnt"][i0] >= 720 \
                else np.full(len(cols), np.nan)
            alto = np.abs(d1) >= L_SIG
            pers = np.isfinite(d0) & (np.abs(d0) >= L_SIG) & alto & (np.sign(d0) == np.sign(d1))
            pos_man = [(D - b) for _, b in MAN if pd.Timedelta(0) <= D - b <= POS_MANUT]
            for q in np.flatnonzero(pers):
                if alto.sum() == 1 and pos_man:
                    out.append(dict(dia_check=D, corte=c, canal=f, tag=cols[q], tipo="degrau",
                                    sigma=float(d1[q]), dias_pos_manut=pos_man[0].days))
            # travado: semanas que terminam em D, D-7, ..., D-84
            ks = [i - 7 * k for k in range(13)]
            if ks[TRAV_SEM - 1] < 0:
                continue
            ok = [kk >= 0 and ST["cnt"][kk] >= 2 * 720 for kk in ks]
            if not all(ok[:TRAV_SEM]):
                continue
            ref = [kk for kk, o in zip(ks[TRAV_SEM:], ok[TRAV_SEM:]) if o]
            if len(ref) < TRAV_REF:
                continue
            tip = np.median(ST["iqr"][ref][:, pos], axis=0)
            r = ST["iqr"][ks[:TRAV_SEM]][:, pos] / np.where(tip > 0, tip, np.nan)
            for q in np.flatnonzero(np.all(r < TRAV_FRAC, axis=0)):
                out.append(dict(dia_check=D, corte=c, canal=f, tag=cols[q], tipo="travado",
                                sigma=float(d1[q]), dias_pos_manut=None))
    return out


_FIT: dict = {}


def fit_rows(c):
    if c not in _FIT:
        _FIT[c] = DF.loc[STABLE & (IX < c), C.SENSOR_TAGS].dropna().tail(FIT)
    return _FIT[c]


def com_exclusao(dia: int, gat: list[dict]):
    """os sinais da composição com a exclusão aplicada; devolve também os intervalos."""
    t, p, ms, ds = (x.copy() for x in R.sinais(dia))
    cs = [c for c in AO.cortes(dia) if AO.bundle(c) is not None]
    prox = {c: (cs[k + 1] if k + 1 < len(cs) else AO.FIM) for k, c in enumerate(cs)}
    intervalos = []
    G = pd.DataFrame(gat)
    if G.empty:
        return (t, p, ms, ds), intervalos
    for (c, f), H in G.groupby(["corte", "canal"]):
        H = H.sort_values("dia_check")
        fora: list[str] = []
        momentos = []
        for _, g in H.iterrows():
            if g.tag not in fora:
                fora.append(g.tag); momentos.append((g.dia_check, list(fora)))
        for k, (ini, excl) in enumerate(momentos):
            fim = momentos[k + 1][0] if k + 1 < len(momentos) else prox[c]
            cols = [x for x in FAM[f] if x not in excl]
            sc = DC.ScorerMax().fit(fit_rows(c)[cols])
            s = (IX >= ini) & (IX < fim)
            sinal = sc.score(DF.loc[s, cols])["pca_recon"].to_numpy()
            (t if f == "t" else p)[s] = sinal
            intervalos.append(dict(dia=dia, canal=f, fora=", ".join(CURTO(x) for x in excl),
                                   de=ini, ate=fim, dias=round((fim - ini).total_seconds() / 86400, 1)))
    return (t, p, ms, ds), intervalos


def main():
    pd.set_option("display.width", 200)
    ref = BR.braco("referência", lambda d: R.detector(*R.sinais(d))["fin"], R.DIAS)
    todos, inter, fins, linhas = [], [], [], []
    for d in R.DIAS:
        gat = gatilhos(d)
        todos += [dict(dia=d, **g) for g in gat]
        sig, iv = com_exclusao(d, gat)
        inter += iv
        fin = R.detector(*sig)["fin"]
        m = R.mede(fin); m.pop("cls")
        fins.append(fin); linhas.append(dict(cenario=d, **m))
        print(f"dia {d:2d}: {len(gat)} checagens com gatilho, {len(iv)} intervalos de exclusão", flush=True)
    var = BR.Braco("F1", fins, pd.DataFrame(linhas).set_index("cenario"))
    G = pd.DataFrame(todos)
    if len(G):
        prim = G.sort_values("dia_check").groupby(["dia", "corte", "canal", "tag", "tipo"]).first().reset_index()
        print("\nPRIMEIRO GATILHO por composição, bundle, canal e sensor:")
        print(prim.assign(tag=prim.tag.map(CURTO), dia_check=prim.dia_check.dt.strftime("%Y-%m-%d"),
                          corte=prim.corte.dt.strftime("%Y-%m-%d"))[
            ["dia", "corte", "canal", "tag", "tipo", "dia_check", "sigma", "dias_pos_manut"]].round(1).to_string(index=False))
    I = pd.DataFrame(inter)
    if len(I):
        print("\nINTERVALOS DE EXCLUSÃO:")
        print(I.assign(de=I.de.dt.strftime("%Y-%m-%d"), ate=I.ate.dt.strftime("%Y-%m-%d")).to_string(index=False))
    r, v = ref.tabela, var.tabela
    print("\nPOR COMPOSIÇÃO (referência -> F1):")
    for d in R.DIAS:
        print(f"  dia {d:2d}: det {r.det[d]} -> {v.det[d]}  início {r.inicio[d]} -> {v.inicio[d]}  banda {r.banda[d]} -> {v.banda[d]}"
              f"  FP/mês {r.fp_mes[d]:.3f} -> {v.fp_mes[d]:.3f}  carga {r.carga_mes[d]:.1f} -> {v.carga_mes[d]:.1f}")
    x = BR.decide(ref, var, pareado=True, nivel=0.95)
    print(f"\nDECISÃO (regra F, pré-registrada): A={x['A']}  Δcarga {x['d_carga']:+.1f} [{x['ic_lo']:+.1f}; {x['ic_hi']:+.1f}]"
          f"  ΔFP {x['d_fp']:+.3f} [{x['fp_lo']:+.3f}; {x['fp_hi']:+.3f}]  -> {x['veredito']}")
    G.to_csv(R.CACHE / "f1_gatilhos.csv", index=False); I.to_csv(R.CACHE / "f1_intervalos.csv", index=False)


if __name__ == "__main__":
    main()
