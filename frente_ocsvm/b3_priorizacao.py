#!/usr/bin/env python3
"""B3 -- uma prioridade ALTA/BAIXA por episódio, fixada a priori, rebaixa parte dos FP sem rebaixar trip?

PRÉ-REGISTRADO EM 14/10/2026, ANTES DE RODAR. B1, B2 e C1 mostraram que cortar alerta nesses 8 trips vira
ajuste ao período. A alternativa da gestão de alarmes é NÃO suprimir: todo episódio continua anunciado, mas
com prioridade. Se uma regra sem parâmetro ajustado põe todos os trips em ALTA e boa parte dos FP em BAIXA, a
equipe passa a ter ~2,88 FP/mês anunciados, mas só uma fração deles exige ação imediata.

OLHADO ANTES (descritivo, sem separar por classe): os 75 episódios finais da referência (8 detecções, 25
inconclusivos, 42 FP) duram pouco (mediana 1,2 h, p90 3,8 h, máx 9,6 h; 150 h de alerta em 14,6 meses, ~10 h/mês).
O refratário de 48 h já faz o papel de "condição conhecida" depois do 1º anúncio. Por isso a ideia original
("ativo nas primeiras N h, depois condição conhecida") não tem o que cortar e fica de fora: o B3 testa
prioridade pela COMPOSIÇÃO do episódio.

REGRAS (fixas, sem parâmetro escolhido nos trips; um episódio é ALTA se a condição vale em algum instante dele):
  R1  "dois modelos"  >= 2 dos 3 canais OCSVM (temp, vib, óleo) ativos ao mesmo tempo, com ou sem o canal 4.
      Lógica: o canal 4 sozinho é fraco (enriquecimento 1,23x, p = 0,19, A2); dois modelos independentes
      concordando é a evidência mais forte que a pipeline tem.
  R2  ">= 3 canais"   >= 3 dos 4 canais ativos ao mesmo tempo. REFERÊNCIA CONHECIDA: o voto >= 3 como gatilho
      deu 5/8 (seção 3); aqui só muda a prioridade, então o esperado é rebaixar ~3 trips.
Base: a pipeline de referência inteira (A0), com os mesmos episódios. A prioridade não muda detecção nem FP/mês.

O QUE SE MEDE, por regra: trips com pelo menos um episódio de detecção em ALTA (de 8; fora da amostra, de 3);
episódios de FP e inconclusivos em BAIXA (n e %); FP/mês só de ALTA; horas de alerta ALTA por mês.

CRITÉRIO (por regra):
  APROVADO   8/8 trips com episódio de detecção em ALTA  E  >= 30% dos episódios de FP em BAIXA.
  REPROVADO  qualquer trip só com episódios de detecção em BAIXA, OU < 30% dos FP rebaixados.
Duas regras = duas chances; uma aprovação só vale se também tiver 3/3 fora da amostra.

EXPECTATIVA REGISTRADA. As duas REPROVADAS por perder trip. R2 rebaixa ~3 trips (como o voto >= 3). R1 rebaixa
mais ainda (<= 4/8 em ALTA), porque os 3 canais OCSVM coincidem entre si só ~1,2% do tempo (seção 1) e as
detecções vêm de 1 modelo + canal 4. Se der assim, a prioridade não separa trip de FP com a informação que a
pipeline tem, pelo mesmo motivo de o corte não separar.

RESULTADO (14/10/2026) -- AS DUAS REPROVADAS:
    regra            trips ALTA  fora ALTA  FP em BAIXA   inconcl. BAIXA  FP/mês ALTA  h/mês ALTA
    R1 dois modelos  3/8         3/3        18/42 (43%)   6/25            1,648        7,1
    R2 >= 3 canais   3/8         3/3        19/42 (45%)   7/25            1,580        6,6
  · As duas rebaixam os MESMOS 5 trips: todos os de dentro da amostra (27/02, 17/03, 07/04, 11/04, 29/04/25).
    Os 3 de fora ficam em ALTA.
  · Expectativa: acertei o veredito e o R1 (<= 4/8); errei o R2 (previ ~5/8, deu 3/8).
  · A POSTERIORI (não pré-registrado): dentro da amostra, 0/5 trips e 13/19 FP em ALTA; fora, 3/3 trips e
    11/23 FP em ALTA (12/23 = 52% rebaixados). Não é deriva: o tempo com >= 2 OCSVM juntos é MENOR fora (0,94%)
    que dentro (1,6%). Hipótese: o trecho antes dos trips de dentro entrou no treino como normal (a exclusão
    é de 24 h em volta do alarme; a detecção nasce até 48 h antes) e os modelos aprenderam a não acender
    juntos ali. Fora, onde o modelo não viu o precursor, os trips aparecem como coincidência de 2 modelos.
    Com 3 trips isso é só hipótese; o teste é a pipeline congelada em dados novos (21/04/2026 em diante).

Uso:  python frente_ocsvm/b3_priorizacao.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import comum as C


def main():
    df = C.carrega_canais()
    idx, op = df.index, df["operational_state"]
    ft = C.trips(idx)
    can = {c: df[c].astype(bool) for c in C.CANAIS}
    fin = C.decide(can, **{k: C.REF[k] for k in ("min_votes", "dur_min", "refrat_h")})
    cls, m = C.avalia(fin, op, ft)
    n_ocsvm = sum(can[c].astype(int) for c in C.CANAIS[:3])
    n_todos = n_ocsvm + can[C.CANAIS[3]].astype(int)
    regras = {"R1 dois modelos": n_ocsvm >= 2, "R2 >=3 canais": n_todos >= 3}
    meses = C.compute_operational_period_days(pd.DataFrame({"operational_state": op})) / 30.4375
    alerta_h = fin.astype(float) * 30 / 3600  # pontos de 30 s

    L, perdas = [], {}
    for nome, cond in regras.items():
        alta = np.array([bool(cond[(idx >= s) & (idx <= e)].any()) for s, e in zip(cls.start, cls.end)])
        c = cls.assign(alta=alta)
        det = c[c.classe == "deteccao"]
        trips_alta = {pd.Timestamp(t) for t in det.loc[det.alta, "falha_associada"]}
        fp, inc = c[c.classe == "falso_positivo"], c[c.classe == "inconclusivo"]
        horas_alta = sum(alerta_h[(idx >= s) & (idx <= e)].sum() for s, e in zip(c.start[alta], c.end[alta]))
        perdas[nome] = [t.strftime("%d/%m/%y") for t in ft if t not in trips_alta]
        L.append(dict(regra=nome, trips_alta=len(trips_alta),
                      fora_alta=sum(t in trips_alta for t in ft if t >= C.SPLIT),
                      FP_baixa=f"{int((~fp.alta).sum())}/{len(fp)} ({(~fp.alta).mean():.0%})",
                      inc_baixa=f"{int((~inc.alta).sum())}/{len(inc)}",
                      FP_mes_alta=round(int(fp.alta.sum()) / meses, 3),
                      h_mes_alta=round(horas_alta / meses, 1),
                      ok=len(trips_alta) == len(ft) and (~fp.alta).mean() >= 0.30))
    T = pd.DataFrame(L)
    pd.set_option("display.width", 200)
    print(f"referência: {m['falhas_detectadas']}/8, {m['n_episodios_falso_positivo']} FP "
          f"({m['falso_positivo_por_mes']:.3f}/mês), {alerta_h.sum() / meses:.1f} h/mês de alerta\n")
    print(T.to_string(index=False))
    for nome, p in perdas.items():
        print(f"{nome}: trips só em BAIXA -> {', '.join(p) or '-'}")
    for _, r in T.iterrows():
        print(f"{r.regra}: {'APROVADO' if r.ok and r.fora_alta == 3 else 'REPROVADO'}")


if __name__ == "__main__":
    main()
