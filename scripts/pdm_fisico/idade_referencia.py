#!/usr/bin/env python3
"""Por que retreinar todo mês? A curva de degradação pela idade do bundle.

É a "Fase 0" do plano do especialista (treina em M, avalia M+1 ... M+6 sem
retreino), feita do jeito que 8 eventos permitem. Avaliar cada bundle sozinho nos 6
meses seguintes daria recall por mês -- e quase todo mês tem zero trip. Aqui é o
contrário: o histórico INTEIRO é servido por bundles de uma idade fixa k. Com
k = 2, o mês M é pontuado pelo bundle treinado no corte de M-2. Cada par (bundle,
mês) de idade k aparece uma vez, como lá, mas a detecção é medida nos 8 eventos de
uma vez, e o estado do detector (EWMA, CUSUM, refratário) corre contínuo.

k = 0 é o mensal de hoje. "congelado" = o primeiro bundle válido serve tudo. Nos
primeiros k meses, sem bundle de k meses atrás, serve o mais antigo que existe (a
idade média real sai numa coluna). Sem vencimento: é exatamente o que a validade
de 62 dias impede em produção.

DUTY. Fração do tempo NORMAL -- máquina vigiada, fora dos 7 dias antes de cada
trip -- em que o canal fica aceso no nível A. É a medida direta do "resíduo
cresce por deriva, não por saúde": o canal acende sem nada estar errado.

PREVISÃO, ESCRITA ANTES. Duty de t e p e carga sobem com a idade; a detecção sobe
um pouco no começo (referência velha é mais sensível, como na guarda) e depois
para de ajudar.

RESULTADO (30/09/2026). Mediana das 8 composições.

    idade do bundle   idade média  det  início  banda  FP/mês  carga  duty t  duty p
    0 (mensal, hoje)     15 d      6,5   6,0    4,5    0,861   130,5   33%     41%
    1 mês                46 d      7,5   5,0    4,0    0,947   231,4   52%     63%
    2 meses              76 d      7,0   5,0    4,0    0,904   219,4   60%     66%
    3 meses             107 d      8,0   3,5    3,0    0,732   172,2   57%     66%
    6 meses             198 d      7,0   3,0    2,0    1,249   261,6   60%     92%
    congelado           581 d      7,0   3,5    2,5    1,292   183,8    8%    100%

  · UM MÊS A MAIS de idade leva o canal p de 41% para 63% do tempo NORMAL aceso, e o t
    de 33% para 52%. A carga sobe 77%. É o "resíduo cresce por deriva, não por saúde",
    medido diretamente.
  · A detecção "de pé" SOBE com a idade, e é o sinal do problema, não de melhora: com
    os canais acesos quase sempre, qualquer trip encontra alarme de pé. As réguas que
    pedem o alarme NASCENDO antes da falha caem: início 6 → 5 → 3, banda 4,5 → 4 → 2.
    O alarme vira fundo de tela.
  · A carga não é monótona (1 mês 231, 3 meses 172): com tudo aceso, os episódios se
    fundem e a contagem por episódio perde sentido. O que anda numa direção só até 6
    meses é o duty do p (41 → 63 → 66 → 66 → 92%) e as réguas de nascimento; o duty
    do t satura em ~55-60% já no 2º mês.
  · Congelado: o t quase não acende (8%) e o p fica aceso 100% -- a escala do primeiro
    PCA (`recon_p99`) decide tudo, como em `o normalizador é o ponto frágil`.

Uso:  PYTHONPATH=. python idade_referencia.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import regua_fp as R
import aprovacao_operador as AO

SEM_VENCIMENTO = pd.Timedelta(days=100_000)
NORMAL = None


def normal() -> np.ndarray:
    global NORMAL
    if NORMAL is None:
        alvo = getattr(R, "alvo", None)
        if alvo is None:
            alvo = R.AV.alvo
        pre = pd.Series(False, index=R.idx)
        for t in alvo:
            pre.loc[t - pd.Timedelta(days=7):t] = True
        NORMAL = AO.MASK & ~pre.to_numpy()
    return NORMAL


def serve(dia: int, k: int | None) -> dict:
    cs = AO.cortes(dia)
    validos = [c for c in cs if AO.bundle(c) is not None]
    if k is None:
        plano = [(cs[0], validos[0])]
    else:
        plano = []
        for i, c in enumerate(cs):
            ref = cs[i - k] if i - k >= 0 else cs[0]
            if AO.bundle(ref) is None:
                ref = min(validos[0], c) if validos[0] <= c else None
            if ref is not None:
                plano.append((c, ref))
    S = AO.de_plano(plano, SEM_VENCIMENTO)
    out = R.detector(S.t, S.p, S.ms, S.ds)
    m = R.mede(out["fin"]); m.pop("cls")
    nm = normal()
    for c in ("t", "p", "sp", "vb"):
        m[f"duty_{c}"] = float(out["A"][c].to_numpy()[nm].mean())
    m["idade_media"] = float(np.nanmean(S.idade[AO.MASK]))
    return m


def main():
    L = []
    for k in (0, 1, 2, 3, 6, None):
        for d in R.DIAS:
            L.append(dict(idade="congelado" if k is None else f"{k} meses", dia=d, **serve(d, k)))
        print(L[-1]["idade"], "ok", flush=True)
    T = pd.DataFrame(L)
    T.to_csv(R.CACHE / "idade_referencia.csv", index=False)
    cols = ["idade_media", "det", "inicio", "banda", "fp_mes", "carga_mes", "duty_t", "duty_p"]
    print(T.groupby("idade", sort=False)[cols].median().round(3).to_string())
    print("\nfaixa entre composições:")
    print(T.groupby("idade", sort=False)[["det", "carga_mes"]].agg(["min", "max"]).round(1).to_string())


if __name__ == "__main__":
    main()
