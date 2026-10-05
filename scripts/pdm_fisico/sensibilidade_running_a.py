#!/usr/bin/env python3
"""As falhas da RUNNING_A interferem no resultado do detector? Sensibilidade da máscara.

SENSIBILIDADE (03/10/2026), não candidato: nenhum destes braços vira proposta sem pré-registro próprio.

A PERGUNTA, do usuário: "acha que isso pode interferir no resultado?". `running_a_coerencia.py` mostrou que a tag
(status derivado das velocidades) concorda com quatro sinais independentes em 99,5% do tempo, com 103 h de
"ligada com a máquina sem combustão" (quase tudo em 19 a 24/08/2025) e 6 falhas curtas em 111 partidas. A tag entra
na máscara (RUNNING_A > 0,5 e T5 > 300) e define o RELÓGIO do blackout de 6 h (cada subida é uma partida) e os
resets do CUSUM. Em vez de opinar, refaz-se o detector com definições alternativas e compara-se o resultado.

OS BRAÇOS (o resto do detector é idêntico; `confere()` prova que a reconstrução com a máscara original e o blackout
de 6 h reproduz a referência bit a bit nas 8 composições):
  V1  só T5: máscara = T5 > 300; o relógio do blackout é a subida do T5 acima de 300. Sem a RUNNING_A.
  V2  RUNNING_A sem falhas curtas: lacunas desligadas de até 30 min com o T5 sempre > 300 viram "ligada".
  V3  sem 19 a 24/08/2025: a janela em que a tag marcou "ligada" sem combustão fica fora da máscara.
  V4  relógio pela ignição: ligada e T5 > 300 (a partida só começa quando as duas valem); corrige o relógio nas
      partidas frias longas, sem abrir mão da tag.
A medida: as da régua de sempre (detecção, início, banda, FP/mês, carga), pareadas nas 8 composições.

EXPECTATIVA REGISTRADA. Nenhuma mudança relevante: os desacordos são ~0,8% das horas e já cobertos pelo T5 > 300; o
V1 e o V4 mexem no relógio por minutos na maioria das partidas (o T5 já passa de 300 °C quando a tag sobe).
Se algum braço mudar detecção ou carga mais que o ruído das composições, a tag interfere.

RESULTADO DA SENSIBILIDADE (03/10/2026) -- A EXPECTATIVA ESTAVA ERRADA.
    braço                      det  início banda  FP/mês  carga   perde det  maior mudança de carga numa composição
    referência                 6,5   6,0   4,5   0,861   133,4      --          --
    V1 só T5                   6,0   5,0   3,5   0,990   105,3      3          66 h/mês
    V2 sem falhas curtas       6,0   5,0   3,5   0,904   137,4      3          14 h/mês
    V3 sem 19-24/08/2025       6,5   6,0   4,5   0,861   145,8      0          29 h/mês
    V4 relógio pela ignição    6,5   6,0   4,5   0,904   102,6      0          70 h/mês  (detecção idêntica nas 8)
  Mexer em minutos no relógio do blackout muda a carga em 23% (V4) e custa detecção em V1 e V2. Duas leituras:
  (a) a tag interfere de forma sistemática (o V4 seria um achado); (b) o detector é caoticamente sensível a qualquer
  perturbação pequena (poucos episódios gigantes de ~600 h aparecem e somem). Falta o controle negativo abaixo.

PRÉ-REGISTRO DO CONTROLE (03/10/2026), ANTES DE RODAR. `jitter`: cada partida (subida da RUNNING_A) é atrasada
por um número aleatório de 0 a 5 amostras (0 a 10 min), o MESMO para as 8 composições dentro de um sorteio; a
máscara e o resto ficam como na referência. 12 sorteios. Mede-se, por sorteio, a carga média das 8 composições, o
número de composições que perdem detecção e o FP/mês mediano, contra a referência.
  LEITURA: se o espalhamento da carga sob esse jitter de <= 10 min for da ordem do efeito do V4 (-31 h/mês), o V4 não
  distingue nada de uma perturbação aleatória (hipótese b); se for muito menor, o V4 é um efeito real (a).
  EXPECTATIVA REGISTRADA: o jitter produz desvio-padrão de 15 a 25 h/mês na carga média e perdas de detecção em
  alguns sorteios; o V4 cai dentro dessa faixa. Chance de o V4 ficar fora (a): ~25%.

Uso:  PYTHONPATH=. python sensibilidade_running_a.py confere    # a reconstrução reproduz a referência
      PYTHONPATH=. python sensibilidade_running_a.py            # a comparação
      PYTHONPATH=. python sensibilidade_running_a.py jitter     # o controle negativo (12 sorteios, em paralelo)
"""
from __future__ import annotations
import io, contextlib, sys
import numpy as np, pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    import regua_fp as R

PP = R.DB.PP
g = pd.read_parquet("grade2min.parquet")
T5 = g["T5_AVG_A"]; HOT = (T5 > 300).fillna(False)
ON = (g["RUNNING_A"] > 0.5).fillna(False)
ORIG = (PP.part.copy(), PP.estavel.copy())
ssub = lambda s: s.reindex(PP.part.index)


def sobe(m: pd.Series) -> pd.Series:
    return m & ~m.shift(fill_value=False)


def preenche_falhas(on: pd.Series, hot: pd.Series, max_amostras: int = 15) -> pd.Series:
    """Lacunas desligadas de até `max_amostras` (30 min) com o T5 sempre quente viram ligadas."""
    v = on.to_numpy().copy(); h = hot.to_numpy()
    off = ~v; d = np.diff(np.concatenate(([0], off.astype(int), [0]))); ini, fim = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    for a, b in zip(ini, fim):
        if 0 < a and b < len(v) and (b - a) <= max_amostras and h[a:b].all() and v[a - 1] and v[b]:
            v[a:b] = True
    return pd.Series(v, index=on.index)


def bracos() -> dict[str, tuple[pd.Series, pd.Series]]:
    op2 = preenche_falhas(ON, HOT)
    j = (g.index >= pd.Timestamp("2025-08-19", tz="UTC")) & (g.index < pd.Timestamp("2025-08-25", tz="UTC"))
    est_ref = ORIG[1]
    op4 = ON & HOT
    return {
        "V1 só T5": (sobe(HOT), HOT),
        "V2 sem falhas curtas": (sobe(op2), op2 & HOT),
        "V3 sem 19-24/08/2025": (ORIG[0], est_ref & ~pd.Series(j, index=g.index)),
        "V4 relógio pela ignição": (sobe(op4), op4),
    }


def roda(d: int, part: pd.Series, estavel: pd.Series) -> pd.Series:
    PP.part, PP.estavel = ssub(part), ssub(estavel)
    try:
        return R.detector(*R.sinais(d), blackout_h=6.0)["fin"]
    finally:
        PP.part, PP.estavel = ORIG


def confere():
    for d in R.DIAS:
        ok = bool((roda(d, *ORIG) == R.detector(*R.sinais(d))["fin"]).all())
        print(f"  composição {d:2d}: máscara original + blackout de 6 h reproduz a referência: {ok}", flush=True)
        assert ok


def main():
    pd.set_option("display.width", 200)
    B = bracos(); linhas = []
    for d in R.DIAS:
        for nome, (p, e) in {"referência": ORIG, **B}.items():
            fin = R.detector(*R.sinais(d))["fin"] if nome == "referência" else roda(d, p, e)
            m = R.mede(fin); m.pop("cls"); linhas.append(dict(dia=d, braco=nome, **m))
        print(f"composição {d:2d} ok", flush=True)
    T = pd.DataFrame(linhas); T.to_csv(R.CACHE / "sensibilidade_running_a.csv", index=False)
    ref = T[T.braco == "referência"].set_index("dia")
    print("\nMEDIANA DAS 8 COMPOSIÇÕES (carga = média)")
    out = []
    for nome in ["referência"] + list(B):
        x = T[T.braco == nome].set_index("dia")
        out.append(dict(braco=nome, det=x.det.median(), inicio=x.inicio.median(), banda=x.banda.median(), fp_mes=x.fp_mes.median(),
                        carga=x.carga_mes.mean(), episodios=int(x.episodios.sum()),
                        perde_det=int((x.det < ref.det).sum()), ganha_det=int((x.det > ref.det).sum()),
                        carga_dif_max=float((x.carga_mes - ref.carga_mes).abs().max())))
    print(pd.DataFrame(out).round(3).to_string(index=False))
    print("\nPOR COMPOSIÇÃO (det / carga h/mês):")
    print(T.pivot(index="dia", columns="braco", values="det").to_string())
    print(T.pivot(index="dia", columns="braco", values="carga_mes").round(0).to_string())


def _tarefa(a):
    k, d = a
    pos = np.flatnonzero(ORIG[0].to_numpy())
    desl = np.random.default_rng(1000 + k).integers(0, 6, size=len(pos))          # 0 a 10 min, igual nas 8 composições
    v = np.zeros(len(g), bool); v[np.minimum(pos + desl, len(v) - 1)] = True
    m = R.mede(roda(d, pd.Series(v, index=g.index), ORIG[1])); m.pop("cls")
    return k, d, m


def jitter(n_sorteios: int = 12):
    import multiprocessing as mp
    ref = pd.read_csv(R.CACHE / "sensibilidade_running_a.csv"); ref = ref[ref.braco == "referência"].set_index("dia")
    tarefas = [(k, d) for k in range(n_sorteios) for d in R.DIAS]
    with mp.get_context("fork").Pool(6) as p:
        res = p.map(_tarefa, tarefas)
    L = []
    for k in range(n_sorteios):
        x = pd.DataFrame([dict(dia=d, **m) for kk, d, m in res if kk == k]).set_index("dia")
        L.append(dict(sorteio=k, carga=x.carga_mes.mean(), d_carga=x.carga_mes.mean() - ref.carga_mes.mean(), fp_med=x.fp_mes.median(),
                      perde_det=int((x.det < ref.det).sum()), det_med=x.det.median(),
                      maior_dif=float((x.carga_mes - ref.carga_mes).abs().max())))
    J = pd.DataFrame(L); J.to_csv(R.CACHE / "sensibilidade_running_a_jitter.csv", index=False)
    print(J.round(2).to_string(index=False))
    d = J.d_carga
    print(f"\nΔ carga média sob o jitter (<= 10 min): média {d.mean():+.1f}, DP {d.std():.1f}, mín {d.min():+.1f}, máx {d.max():+.1f} h/mês")
    print(f"sorteios com perda de detecção: {int((J.perde_det > 0).sum())}/{n_sorteios} (até {int(J.perde_det.max())} composições)")
    print(f"maior mudança de carga numa composição: mediana {J.maior_dif.median():.0f}, máx {J.maior_dif.max():.0f} h/mês")
    print("para comparar: V4 -30,8 h/mês (0 perdas) | V1 -28,1 (3 perdas) | V2 +4,0 (3 perdas) | V3 +12,4 (0 perdas)")
    print(f"sorteios com Δ carga <= -30,8 (tão bom quanto o V4): {int((d <= -30.8).sum())}/{n_sorteios}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "confere":
        confere()
    elif len(sys.argv) > 1 and sys.argv[1] == "jitter":
        jitter()
    else:
        main()
