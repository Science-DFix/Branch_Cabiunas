#!/usr/bin/env python3
"""E1 -- triagem dos FP por causa (descritiva; nenhuma regra é escolhida aqui).

Pergunta (do usuário: "comece pelo item 1"): D1-D3 procuraram uma assinatura do sinal que separe trip de FP e não
acharam. Aqui a pergunta é outra: POR QUE cada FP aconteceu? Se um grupo de FP tiver causa de qualidade de dado ou de
operação que NÃO aparece nas detecções, uma regra física (que não olha os trips para ser escolhida) pode ser
pré-registrada depois. As detecções entram só como controle: causa que também aparece em trip não vira regra.

Episódios: OCSVM de produção e CNN-1D mensal (rodada 1d), régua e camada de decisão de produção, desde 01/2025.
Janela de cada episódio: 6 h antes do início até o fim do episódio. Sensores olhados = os do(s) grupo(s) cujo canal
de modelo está aceso no episódio (>= 50% dele).
Causas candidatas (cada uma é um sim/não por episódio):
  falha_txt   algum valor "Comm Fail"/"Out of Serv" (texto no lugar da medida) nos sensores do grupo aceso
  reta        >= 50% dos pontos "on" da janela em trecho de reta (2ª diferença ~0, 1ª != 0) em algum sensor do grupo
              aceso -- interpolação do historiador entre registros esparsos
  constante   algum sensor do grupo aceso parado (1ª diferença 0) por >= 10 min na janela (o veto de produção só pega
              >= 30 min)
  partida     episódio começa até 6 h depois de a máquina entrar em "on"
  rajada      rajada de alarmes de temperatura (>= 5 tags em 10 min) nas 24 h em volta (mesma regra do relatório)
  pos_2702    depois de 27/02/2026, quando PI_6240319_AL para de alarmar
  cronico     a tag que acendeu o canal 4 alarmou >= 10 vezes nos 7 dias antes

RESULTADO (19/10/2026), desde 01/2025, OCSVM (36 FP, 8 det) | CNN-1D mensal (45 FP, 8 det):
  causa         det  FP        |  det  FP
  falha_txt      0    2        |   0    2     limpa, mas só 2 FP (14 e 17/01/2026)
  reta           0    0        |   0    0     nenhum episódio; PI_0308 tem 12,9% em reta, mas não puxa FP
  constante      1    8        |   3   16     não limpa; PDIT_0305 fica parado 32% do tempo (sensor quase estático)
  partida (6h)   3   18        |   2   20     MAIOR GRUPO: 15/36 FP começam na 1ª hora "on"; mas T4, T6, T7 também
                                              (1,1 / 0,6 / 0,0 h após a partida, antecedência 20 / 14 / 8 h)
  rajada         1    5        |   1    5     não limpa (T4)
  pos_2702       0    6        |   0    5     "limpa" só porque não há trip depois de 26/02/2026: não testável
  cronico        4   13        |   4   15     não limpa
  sem causa nenhuma: 9/36 | 17/45 -- quase todos na série de vibração de 31/01 a 14/02/2026 e em 01-03/2025.
Leitura: qualidade de dado explica só 2 FP. O grupo grande é a PARTIDA (metade dos FP na 1ª-2ª hora depois de a
máquina voltar a "on", que já vem após 60 min de TRANSIENT_PADDING), mas 3 dos 8 trips também começam ali; uma
regra só sobrevive se ADIAR o alerta (o voto continua aceso depois da janela?) em vez de descartar. Candidato a E2.

Uso: python frente_ocsvm/e1_triagem_fp.py   (precisa de relatorio_ocsvm_cnn1d_mensal/cache.pkl; gera_figuras.py o cria)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import comum as C
sys.path.insert(0, str(C.AQUI.parent / "relatorio_ocsvm_cnn1d_mensal"))
from gera_figuras import CSV, INICIO, TV, composicao, episodios  # noqa: E402

GRUPO = {"temp": ["TC382_03_A", "T5_AVG_A"], "vib": TV, "óleo": ["954005_624_PI_0308", "954005_624_PDIT_0305"]}
CAUSAS = ["falha_txt", "reta", "constante", "partida", "rajada", "pos_2702", "cronico"]
SAIDA = C.AQUI / "dados" / "e1_triagem.csv"


def sensores(inicio="2024-12-25"):
    cols = ["data_datetime"] + sorted({s for v in GRUPO.values() for s in v})
    raw = pd.read_csv(CSV, usecols=cols, dtype=str).set_index("data_datetime")
    raw.index = pd.to_datetime(raw.index)
    raw = raw.loc[pd.Timestamp(inicio):].sort_index()
    num = raw.apply(pd.to_numeric, errors="coerce")
    txt = num.isna() & raw.notna()
    return num, txt


def flags_sensor(x: pd.Series):
    d1, d2 = x.diff().abs(), x.diff().diff().abs()
    reta = (d2 < 1e-9) & (d1 > 1e-12)
    parado = d1 < 1e-12
    run = parado.groupby((~parado).cumsum()).cumsum()  # pontos seguidos parados
    return reta, run >= 20  # 20 x 30 s = 10 min


def triagem(cls, canais3, c4, cat, op, F):
    comp = composicao(cls, canais3, c4, cat).set_index("start")
    on = op.eq("on")
    partidas = on.index[on & ~on.shift(fill_value=False)]
    L = []
    for t0, e in comp.iterrows():
        w = slice(t0 - pd.Timedelta(hours=6), e.end)
        grupos = [] if e.modelos == "nenhum" else e.modelos.split("+")
        sens = [s for g in grupos for s in GRUPO[g]]
        ow = on.loc[w]
        p = partidas[partidas <= t0]
        a4 = cat[cat.tag.isin(e.tags4.split(",")) & (cat.t <= t0) & (cat.t >= t0 - pd.Timedelta(days=7))]
        L.append(dict(start=t0, classe=e.classe, modelos=e.modelos, gatilho=e.gatilho,
                      falha_txt=bool(sens) and bool(F["txt"].loc[w, sens].any().any()),
                      reta=bool(sens) and any(F["reta"][s].loc[w][ow].mean() >= 0.5 for s in sens),
                      constante=bool(sens) and any(F["parado"][s].loc[w].any() for s in sens),
                      partida=len(p) > 0 and (t0 - p[-1]) <= pd.Timedelta(hours=6),
                      rajada=bool(e.rajada_temp), pos_2702=t0 > pd.Timestamp("2026-02-27"),
                      cronico=len(a4) >= 10, h_partida=round((t0 - p[-1]).total_seconds() / 3600, 1) if len(p) else np.nan,
                      n_alarmes_7d=len(a4)))
    return pd.DataFrame(L)


def resumo(T, nome):
    print(f"\n=== {nome}: {T.classe.value_counts().to_dict()}")
    tab = T.groupby("classe")[CAUSAS].sum().astype(int).T
    print(tab.reindex(columns=["deteccao", "falso_positivo", "inconclusivo"]).to_string())
    fp, det = T[T.classe == "falso_positivo"], T[T.classe == "deteccao"]
    # causas "limpas": nenhuma detecção tem a causa
    limpas = [c for c in CAUSAS if not det[c].any()]
    sem_causa = ~fp[CAUSAS].any(axis=1)
    so_limpa = fp[limpas].any(axis=1) if limpas else pd.Series(False, index=fp.index)
    print(f"causas que nenhuma detecção tem: {limpas or 'nenhuma'}")
    print(f"FP com alguma causa limpa: {int(so_limpa.sum())}/{len(fp)}; FP sem nenhuma causa: {int(sem_causa.sum())}/{len(fp)}")
    print("medianas det x FP: h desde partida", det.h_partida.median(), "x", fp.h_partida.median(),
          "| alarmes da tag do c4 em 7 d", det.n_alarmes_7d.median(), "x", fp.n_alarmes_7d.median())
    print("\ndetecções:")
    print(det[["start", "modelos", "gatilho", "h_partida", "n_alarmes_7d"] + CAUSAS].to_string(index=False))


def main():
    oc, cn, c4, cat, _ = pd.read_pickle(C.AQUI.parent / "relatorio_ocsvm_cnn1d_mensal" / "cache.pkl")
    idx, op = oc.index, oc["operational_state"]
    ft = C.trips(idx)
    num, txt = sensores()
    num, txt = num.reindex(idx), txt.reindex(idx, fill_value=False)
    F = {"txt": txt, "reta": {}, "parado": {}}
    for s in num.columns:
        F["reta"][s], F["parado"][s] = flags_sensor(num[s])
    on_rec = (idx >= INICIO) & op.eq("on").values
    print("fração dos pontos 'on' desde 01/2025 em reta / parado >= 10 min, por sensor:")
    print(pd.DataFrame({"reta": {s: F["reta"][s][on_rec].mean() for s in num.columns},
                        "parado10": {s: F["parado"][s][on_rec].mean() for s in num.columns},
                        "texto": txt[on_rec].sum()}).round(3).to_string())
    out = []
    for nome, can in (("OCSVM", oc), ("CNN-1D mensal", cn)):
        cls, _ = episodios(can, c4, op, ft)
        T = triagem(cls[cls.start >= INICIO], can, c4, cat, op, F)
        resumo(T, nome)
        out.append(T.assign(modelo=nome))
    pd.concat(out).to_csv(SAIDA, index=False)
    print(f"\ntabela por episódio: {SAIDA}")


if __name__ == "__main__":
    main()
