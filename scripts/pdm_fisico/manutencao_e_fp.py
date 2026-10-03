#!/usr/bin/env python3
"""Os falsos positivos se concentram logo depois de uma manutenção?

DIAGNÓSTICO A POSTERIORI (02/10/2026), nos mesmos 8 eventos. Não é candidato e não muda
veredito: serve para decidir se vale pré-registrar uma "carência pós-manutenção".

POR QUE NÃO COM AS PLANILHAS. `Histórico Operações TSC33003A.xlsx` tem execução real só até
24/03/2025 (3 operações datadas dentro dos 16 meses de avaliação) e `Historico Ordens
Turbina A.xlsx` termina em 08/05/2025, com datas-base de planejamento (33 de 115 ordens ocupam
28 dias ou mais). O PI cobre os 16 meses: o modo de manutenção `HSX_6240001A` ligado por
>= 24 h, com trechos a menos de 24 h fundidos (a mesma definição de `monitor_drift`).

A PERGUNTA E O CONTROLE NEGATIVO. Se a carga (h de FP + NEUTRO) se concentra nos K dias após o
fim de uma manutenção, o que importa é por QUANTO: comparada com a mesma lista de episódios
DESLOCADA no tempo (rotação circular da janela de avaliação, 2.000 deslocamentos), que preserva
duração e agrupamento mas apaga a relação com a manutenção. Sem o controle, "20% da carga cai
na pós-manutenção" não diz nada se a pós-manutenção ocupa 20% do tempo.

E O CUSTO DE UMA CARÊNCIA: quantos dos 8 trips caem nos K dias após uma manutenção, e quantas
horas de operação a carência cegaria.

RESULTADO (02/10/2026) -- NÃO HÁ SINAL; A CARÊNCIA PÓS-MANUTENÇÃO NÃO SE PRÉ-REGISTRA.
  7 manutenções (HSX >= 24 h; duas delas, 05-09/11 e 10-15/11/2025, são um trabalho só).

    K dias   tempo vigiado   carga na janela   por acaso   razão   p (desloc.)   trips dentro
        3         3,2%            1,8%            4,4%      0,42      0,78            0
        7         8,3%            2,1%           10,1%      0,21      0,94            0
       14        17,2%           13,8%           18,7%      0,74      0,71            0
       30        29,9%           16,1%           34,3%      0,47      0,90            2

  · A carga fica ABAIXO do acaso em todas as janelas; em nenhuma o deslocamento aleatório fica
    abaixo do observado. Os FP não se concentram depois de manutenção.
  · O custo de uma carência: 30 dias cegariam 2.538 h vigiadas (30% do tempo) e 2 dos 8 trips
    (29/04/2025 a 15,7 d; 09/12/2025 a 23,7 d) cairiam dentro dela; 7 dias cegariam 700 h para
    cortar ~2% da carga.
  · Poder baixo (7 manutenções): não prova efeito zero, mas não dá nenhum motivo para um
    pré-registro. O degrau de nov/2025 foi um episódio só.

Uso:  PYTHONPATH=. python manutencao_e_fp.py
"""
from __future__ import annotations
import io, contextlib
from pathlib import Path
import numpy as np, pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    import regua_fp as R

AQUI = Path(__file__).resolve().parent
PKG = AQUI.parents[1] / "estrutura_pra_prod" / "Cabiunas"
KS = (3, 7, 14, 30)
N_DESL = 2000
SEMENTE = 0


def manutencoes() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    csv = next((PKG / "dados" / "2025_2026").glob("data_*_raw.csv"))
    h = pd.read_csv(csv, index_col=0, parse_dates=[0], usecols=["ts", "HSX_6240001A"])["HSX_6240001A"]
    h.index = h.index.tz_convert("UTC") if h.index.tz is not None else h.index.tz_localize("UTC")
    on = (h.fillna(0) > 0.5).to_numpy().astype(int)
    d = np.diff(np.concatenate(([0], on, [0])))
    ini, fim = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    tr = []
    for a, b in zip(h.index[ini], h.index[fim]):
        if tr and a - tr[-1][1] <= pd.Timedelta(hours=24):
            tr[-1][1] = b
        else:
            tr.append([a, b])
    return [(a, b) for a, b in tr if b - a >= pd.Timedelta(hours=24)]


def pos_manutencao(man, k_dias: int) -> np.ndarray:
    """Instantes da grade dentro dos k dias seguintes ao fim de uma manutenção."""
    m = np.zeros(len(R.idx), bool)
    for _, b in man:
        m |= (R.idx > b) & (R.idx <= b + pd.Timedelta(days=k_dias))
    return m


def carga_por_instante(cls) -> np.ndarray:
    """1 nos instantes dentro de episódio FP ou NEUTRO."""
    c = np.zeros(len(R.idx), bool)
    for a, b, k, _ in cls:
        if k in ("FP", "NEUTRO"):
            c[(R.idx >= a) & (R.idx <= b)] = True
    return c


def main():
    pd.set_option("display.width", 200)
    man = manutencoes()
    print(f"manutenções (HSX_6240001A >= 24 h): {len(man)}")
    for a, b in man:
        print(f"  {a:%Y-%m-%d %H:%M} .. {b:%Y-%m-%d %H:%M}  ({(b - a).total_seconds() / 86400:.1f} d)")

    avaliacao = np.flatnonzero(np.asarray(R.sel))
    op = R.mask.to_numpy()
    rng = np.random.default_rng(SEMENTE)
    desl = rng.integers(1, len(avaliacao) - 1, size=N_DESL)

    print("\nOS 8 TRIPS: dias desde o fim da manutenção anterior")
    for T in R.alvo:
        ant = [b for _, b in man if b < T]
        print(f"  {T:%Y-%m-%d}: " + (f"{(T - max(ant)).total_seconds() / 86400:6.1f} d" if ant else "sem manutenção anterior"))

    cls_por = {}
    for d in R.DIAS:
        m = R.mede(R.detector(*R.sinais(d))["fin"])
        cls_por[d] = m["cls"]

    linhas = []
    for k in KS:
        pm = pos_manutencao(man, k)
        ocupa = float(pm[op].mean())                       # fatia do tempo vigiado na pós-manutenção
        cega = float(pm.astype(bool)[op].sum()) / 30
        trips_dentro = sum(1 for T in R.alvo if pm[R.idx.searchsorted(T) - 1])
        obs, nulos, h_obs = [], [], []
        for d, cls in cls_por.items():
            c = carga_por_instante(cls)
            tot = c[avaliacao].sum()
            if tot == 0:
                continue
            frac = float(c[avaliacao][pm[avaliacao]].sum() / tot)
            ca = c[avaliacao]
            nul = np.array([np.roll(ca, s)[pm[avaliacao]].sum() / tot for s in desl])
            obs.append(frac); nulos.append(nul); h_obs.append(float(c[avaliacao][pm[avaliacao]].sum() / 30))
        obs = np.array(obs); nul = np.mean(nulos, axis=0)
        linhas.append(dict(k_dias=k, tempo_vigiado_pos_manut=100 * ocupa, h_vigiadas_cegadas=cega,
                           carga_na_pos_manut=100 * obs.mean(), nulo_medio=100 * nul.mean(),
                           razao=obs.mean() / nul.mean(), p_desl=float((nul >= obs.mean()).mean()),
                           comp_acima_do_nulo=int((obs > nul.mean()).sum()), trips_dentro=trips_dentro,
                           h_carga_pos_manut_por_composicao=np.mean(h_obs)))
    T = pd.DataFrame(linhas).set_index("k_dias")
    print("\nCARGA (FP + NEUTRO) NOS K DIAS APÓS UMA MANUTENÇÃO, média das 8 composições")
    print("  carga_na_pos_manut / nulo_medio: % da carga que cai na janela, contra o que cairia por acaso")
    print(T.round(2).to_string())
    T.to_csv(R.CACHE / "manutencao_e_fp.csv")


if __name__ == "__main__":
    main()
