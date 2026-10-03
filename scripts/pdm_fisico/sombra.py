#!/usr/bin/env python3
"""Modo sombra prospectivo: o único holdout honesto do detector.

PRÉ-REGISTRADO EM 02/10/2026, ANTES DE HAVER UM ÚNICO DADO PROSPECTIVO.

POR QUE EXISTE. Todo número do detector (0,344 FP/mês, 8/8, banda 5/8) vem dos mesmos 8
eventos em que os parâmetros foram escolhidos: o LOEO dá 7/8, e só ~3,8 das ~6,5 detecções
ficam acima do acaso. Dado de antes de congelar a configuração não valida nada. Dado de DEPOIS
valida, desde que nada mude no meio -- é o que este script faz cumprir.

O QUE FAZ.
  registra   roda o pacote de produção sobre o CSV recebido, grava num ledger (pasta):
               congelado.json   a configuração no 1.º registro: commit, sha256 do código de inferência, do
                                de retreino e do detector.json (sem a lista de trips, que cresce)
               episodios.csv    um alarme por linha, com first_seen_at (quando ESTE processo o viu
                                pela primeira vez) e `notificavel_em` = início + 2 h (o episódio só
                                vira alarme depois de DUR_MIN; antes é "atenção")
               serie.csv.gz     por instante: vigiado e alarme (para horas vigiadas e cobertura)
               execucoes.csv    uma linha por execução: bundle, veredito da entrada e do drift
             Se o código ou o detector.json mudaram desde o congelamento, RECUSA (código 5): o
             período prospectivo deixou de valer. `--refreza` abre um período novo, e fica registrado.
  avalia     só dado a partir da data de congelamento: detecção, início, banda, banda NOTIFICÁVEL
             (início + 2 h), FP bruto por mês de operação com IC de Poisson, e o teste binomial de
             detecção contra o acaso medido na própria série.

CRITÉRIOS FIXADOS AGORA (avaliados quando houver >= 12 meses de operação OU >= 8 trips-alvo
no período, o que vier depois; antes disso só se descreve, não se decide):
  FP     o limite SUPERIOR de 95% (Poisson) da taxa de FP por mês de operação <= 1,15, o
         orçamento de FP usado na seleção do ponto (`ORC_FP`). Com 12 meses isso aceita até 8 FP.
  DET    o teste binomial exato, unilateral a 5%, rejeita "detecção por acaso", com a cobertura
         medida na série prospectiva. Com 8 eventos exige >= 6 detecções.
  Cada um é reportado separadamente. NÃO se combinam nem se reescolhem depois de ver o resultado.

O QUE ESTE MODO NÃO FAZ. Com ~0,5 trip por mês, 6 meses são ~3 eventos (poder 54% de provar
detecção se a verdade for 0,81); 12 meses, ~6 (69%); ~16 meses, ~8 (82%). Para FP, 12 meses
dão 4 a 10 episódios: distingue "abaixo de ~1,2" de "acima", mas não 0,34 de 0,6. Detecção só
se decide em ~16 meses. O FP é BRUTO (sem a Regra C, que precisa das paradas reais do PI).

HIPÓTESE PROSPECTIVA P1b (03/10/2026). Em `portao_de_forca.py` o portão de força sobre o alarme final
(apara o início do episódio até a força F chegar a θ; sem F >= θ, descarta; o que sobra tem de cumprir a
duração mínima) reprovou no pré-registro (θ = 2: perde uma detecção na composição 1) e passou, como
sensibilidade, com θ = 1,5. Aqui o instante em que F chega a 1,5, 2 e 3 em cada episódio vai para o ledger
(`t_f15`, `t_f20`, `t_f30`), e `avalia` imprime o resultado COM θ = 1,5, escolhido agora, antes de haver dado.
CRITÉRIO, fixado agora, avaliado quando houver >= 12 meses ou >= 8 trips: o portão é confirmado se NENHUM
trip do período prospectivo perde a detecção E o FP bruto com o portão é menor que sem ele. Um trip
perdido o reprova, sem reescolher θ. Os outros θ ficam registrados, sem decisão.

Uso:  python sombra.py registra --csv ARQ --ledger PASTA [--refreza]
      python sombra.py avalia   --ledger PASTA [--trips registro_trips.csv]
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
PKG = AQUI.parents[1] / "estrutura_pra_prod" / "Cabiunas"
sys.path.insert(0, str(PKG / "scripts"))
import cabiunas_inference as ci  # noqa: E402

JAN = pd.Timedelta(hours=48)
TMIN = pd.Timedelta(hours=4)
LAT = pd.Timedelta(hours=2)        # DUR_MIN: o episódio só vira alarme depois disso
ORC_FP = 1.15
COD_MUDOU = 5


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def configuracao() -> dict:
    """O que não pode mudar durante o período: o código e o detector (sem a lista de trips) e o código do retreino."""
    mods = ci.carregar_modelos(PKG / "modelos")
    det = {k: v for k, v in mods[-1]["detector"].items() if k != "trips_conhecidos"}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=AQUI, capture_output=True,
                            text=True).stdout.strip()
    return dict(commit=commit, sha_codigo=sha((PKG / "scripts" / "cabiunas_inference.py").read_bytes()),
                sha_retreino=sha((PKG / "scripts" / "constroi_bundle.py").read_bytes()),
                sha_detector=sha(json.dumps(det, sort_keys=True).encode()))


def upsert(path: Path, novo: pd.DataFrame, chave: str) -> pd.DataFrame:
    if path.exists():
        velho = pd.read_csv(path)
        novo = pd.concat([velho[~velho[chave].isin(novo[chave])], novo], ignore_index=True)
    novo.to_csv(path, index=False)
    return novo


def registra(a) -> int:
    led = Path(a.ledger); led.mkdir(parents=True, exist_ok=True)
    agora = pd.Timestamp.now(tz="UTC")
    cfg = configuracao(); fr = led / "congelado.json"
    if fr.exists():
        antigo = json.loads(fr.read_text(encoding="utf-8"))
        mudou = [k for k in ("sha_codigo", "sha_retreino", "sha_detector") if antigo["configuracao"].get(k) != cfg[k]]
        if mudou and not a.refreza:
            print(f"RECUSADO: {', '.join(mudou)} mudou desde o congelamento de {antigo['desde']} "
                  f"(commit {antigo['configuracao']['commit']} -> {cfg['commit']}). O período prospectivo deixou de "
                  f"valer. Use --refreza para abrir um período novo.")
            return COD_MUDOU
        if mudou and a.refreza:
            hist = led / "periodos_anteriores.json"
            ant = json.loads(hist.read_text()) if hist.exists() else []
            hist.write_text(json.dumps(ant + [antigo], indent=1, ensure_ascii=False), encoding="utf-8")
            fr.unlink()
    df = ci.preparar_grade(ci.carregar_dados(a.csv))
    modelos = ci.carregar_modelos(PKG / "modelos"); trips = ci.carregar_trips(PKG / "registro_trips.csv")
    diag = ci.diagnostico_entrada(modelos, df)
    if diag["veredito"] == "cego":
        print("ENTRADA CEGA:", diag["mensagem"]); return 3
    res = ci.prever(modelos, ci.preprocessar(modelos, df, trips=trips))
    if not fr.exists():
        desde = res.index[-1].normalize() + pd.Timedelta(days=1) if a.desde is None else pd.Timestamp(a.desde, tz="UTC")
        fr.write_text(json.dumps(dict(desde=str(desde), congelado_em=str(agora), configuracao=cfg),
                                 indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"CONFIGURAÇÃO CONGELADA. Período prospectivo a partir de {desde:%Y-%m-%d}: {cfg}")
    eps = ci.resumo_episodios(res)
    linhas = []
    for e in eps.itertuples():
        w = res.loc[e.inicio:e.fim]
        ok_al, fr = w["is_anomaly"].to_numpy(), w["forca"].fillna(0.0).to_numpy()
        t_f = {n: (str(w.index[int(np.argmax(ok_al & (fr >= th)))]) if (ok_al & (fr >= th)).any() else "")
               for n, th in (("t_f15", 1.5), ("t_f20", 2.0), ("t_f30", 3.0))}
        linhas.append(dict(id=f"TC-33003A:{e.inicio:%Y%m%dT%H%M%SZ}", inicio=str(e.inicio), fim=str(e.fim),
                           horas=float(e.horas), forca_max=float(e.forca_max), canais_max=int(e.canais_max), **t_f,
                           notificavel_em=str(e.inicio + LAT), em_curso=bool(e.fim >= res.index[-1] - pd.Timedelta("2min")),
                           last_seen_at=str(agora)))
    novo = pd.DataFrame(linhas)
    epi = led / "episodios.csv"
    if epi.exists() and len(novo):
        ant = pd.read_csv(epi).set_index("id")["first_seen_at"]
        novo["first_seen_at"] = novo["id"].map(ant).fillna(str(agora))
    else:
        novo["first_seen_at"] = str(agora)
    upsert(epi, novo, "id")
    s = pd.DataFrame({"ts": res.index.astype(str), "vigiado": res["vigiado"].astype(int).to_numpy(),
                      "alarme": res["is_anomaly"].astype(int).to_numpy()})
    s = s[(s.vigiado > 0) | (s.alarme > 0)]
    sp = led / "serie.csv.gz"
    if sp.exists():
        v = pd.read_csv(sp); s = pd.concat([v[~v.ts.isin(s.ts)], s], ignore_index=True)
    s.sort_values("ts").to_csv(sp, index=False)
    drift = ci.monitor_drift(modelos, df)
    upsert(led / "execucoes.csv", pd.DataFrame([dict(executado_em=str(agora), dado_ate=str(res.index[-1]),
                                                     bundle=modelos[-1]["dir"].name, entrada=diag["veredito"],
                                                     drift=drift["veredito"], episodios=len(eps))]), "executado_em")
    print(f"registrado: {len(eps)} episódio(s) no histórico da entrada, dado até {res.index[-1]:%Y-%m-%d %H:%M}")
    return 0


def poisson_ic(k: int, meses: float, nivel=0.95) -> tuple[float, float]:
    lo = stats.chi2.ppf((1 - nivel) / 2, 2 * k) / 2 / meses if k > 0 else 0.0
    hi = stats.chi2.ppf(1 - (1 - nivel) / 2, 2 * (k + 1)) / 2 / meses
    return lo, hi


def avalia(a) -> int:
    led = Path(a.ledger); fr = json.loads((led / "congelado.json").read_text(encoding="utf-8"))
    desde = pd.Timestamp(fr["desde"])
    epi = pd.read_csv(led / "episodios.csv", parse_dates=["inicio", "fim", "notificavel_em", "t_f15", "t_f20", "t_f30"])
    ser = pd.read_csv(led / "serie.csv.gz", parse_dates=["ts"])
    ser = ser[ser.ts >= desde]; epi = epi[epi.inicio >= desde]
    alvo = [t for t in ci.carregar_trips(a.trips or PKG / "registro_trips.csv") if t >= desde]
    h_vig = float(ser.vigiado.sum()) * 2 / 60; meses = h_vig / 730
    print(f"período prospectivo: desde {desde:%Y-%m-%d} (commit {fr['configuracao']['commit']}) | "
          f"{h_vig:.0f} h vigiadas = {meses:.2f} meses de operação | {len(epi)} episódios | {len(alvo)} trips")
    def metricas(e):
        det = ini = ban = ban_n = 0
        for T in alvo:
            det += len(e[(e.inicio <= T) & (e.fim >= T - JAN)]) > 0
            nasc = e[(e.inicio >= T - JAN) & (e.inicio <= T)]
            ini += len(nasc) > 0
            ban += int(((nasc.inicio <= T - TMIN)).any())
            ban_n += int(((nasc.notificavel_em <= T - TMIN)).any())
        jan = [(T - JAN, T) for T in alvo]
        return det, ini, ban, ban_n, int(sum(1 for _, r in e.iterrows() if not any(r.inicio <= t1 and r.fim >= t0 for t0, t1 in jan)))

    det, ini, ban, ban_n, fp = metricas(epi)
    print(f"detecção {det}/{len(alvo)} | início {ini}/{len(alvo)} | banda {ban}/{len(alvo)} | "
          f"banda NOTIFICÁVEL (início + 2 h) {ban_n}/{len(alvo)}")
    # P1b, hipótese prospectiva: portão de força depois do refratário (ver o docstring)
    g = epi.dropna(subset=["t_f15"]).copy()
    dur = (g.fim - g.t_f15).dt.total_seconds() / 60 + 2
    g = g[dur >= np.where(g.forca_max > 20, 60, 120)]
    g["inicio"] = g.t_f15; g["notificavel_em"] = g.t_f15 + LAT
    gdet, gini, gban, gban_n, gfp = metricas(g)
    print(f"COM O PORTÃO P1b (θ = 1,5, hipótese prospectiva): detecção {gdet}/{len(alvo)} | início {gini}/{len(alvo)} | "
          f"banda {gban}/{len(alvo)} | notificável {gban_n}/{len(alvo)} | FP bruto {gfp} (sem o portão: {fp}) | "
          f"episódios {len(g)} de {len(epi)}")
    if meses > 0:
        lo, hi = poisson_ic(fp, meses)
        print(f"FP bruto: {fp} em {meses:.2f} meses = {fp / meses:.2f}/mês, IC95% [{lo:.2f}; {hi:.2f}] "
              f"(critério: limite superior <= {ORC_FP})")
        ok_fp = hi <= ORC_FP
    else:
        ok_fp = False
    # cobertura: fração do tempo vigiado com alarme em [t-48h, t)
    v = ser.set_index("ts")
    al = v["alarme"].reindex(pd.date_range(v.index.min(), v.index.max(), freq="2min", tz="UTC"), fill_value=0)
    cob = float((al.rolling(int(JAN / pd.Timedelta("2min")), min_periods=1).max().reindex(v.index)[v["vigiado"] > 0]).mean())
    if alvo:
        p = stats.binomtest(det, len(alvo), cob, alternative="greater").pvalue
        print(f"detecção contra o acaso (cobertura medida {100 * cob:.0f}%): p = {p:.3f}")
    meses_ok, eventos_ok = meses >= 12, len(alvo) >= 8
    print(f"\nDECIDE-SE? meses de operação >= 12: {meses_ok} | trips >= 8: {eventos_ok} "
          f"({'sim: critério FP ' + ('ATENDIDO' if ok_fp else 'NÃO atendido') if (meses_ok or eventos_ok) else 'não: só descrever'})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("registra"); r.add_argument("--csv", required=True); r.add_argument("--ledger", required=True)
    r.add_argument("--desde", default=None, help="início do período (padrão: dia seguinte ao fim do dado, no 1.º registro)")
    r.add_argument("--refreza", action="store_true")
    v = sub.add_parser("avalia"); v.add_argument("--ledger", required=True); v.add_argument("--trips", default=None)
    a = ap.parse_args()
    return registra(a) if a.cmd == "registra" else avalia(a)


if __name__ == "__main__":
    raise SystemExit(main())
