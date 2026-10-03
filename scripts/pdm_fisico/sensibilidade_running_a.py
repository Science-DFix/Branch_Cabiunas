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

Uso:  PYTHONPATH=. python sensibilidade_running_a.py confere    # a reconstrução reproduz a referência
      PYTHONPATH=. python sensibilidade_running_a.py            # a comparação
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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "confere":
        confere()
    else:
        main()
