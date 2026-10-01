#!/usr/bin/env python3
"""Champion/challenger: o que fazer quando o cliente não promove e o bundle vence?

O DESENHO. Todo mês o pipeline treina um challenger no corte (segundos de CPU) e o
deixa rodando em sombra. O cliente promove quando quiser. Se ele não promover, o
champion envelhece e, aos 62 dias do fim do baseline, vence (`checa_validade`).

AS QUATRO POLÍTICAS DE DEGRADAÇÃO, com o MESMO comportamento de cliente de
`aprovacao_operador.py` (cada mês promovido com probabilidade q, atraso de 0 a 14
dias, 5 sementes x 8 composições, mesmo sorteio):
  cego              o de hoje: vencido, a inferência para        (já medido lá)
  segue velho       o champion vencido continua, marcado         (já medido lá,
                    "degradado"                                    "sem vencimento")
  promove sozinho   no vencimento, o challenger mais recente entra sem aprovação;
                    o cliente aceitou de antemão que silêncio até o dia 62 = sim
  só sp e vb        vencido, os canais t e p (os resíduos de PCA, que são os que
                    envelhecem) saem; o detector segue com o espalhamento dos
                    mancais e a vibração, cuja referência é rolante e não vence

CRITÉRIO, ESCRITO ANTES DE RODAR: o mesmo TOLERÁVEL de `aprovacao_operador.py`
contra o mensal automático (det, início e banda não caem mais que 0,5; FP e carga
não sobem mais que 10%; nenhum dia cego -- aqui nenhuma das duas novas tem dia cego
por construção).

PREVISÃO. "Promove sozinho" fica perto do mensal (a idade máxima é 62 dias e só em
alguns meses); "só sp e vb" perde detecção -- sem o vb o detector cai a 1/8, mas sem
t e p ninguém mediu.

RESULTADO (30/09/2026). Mediana de 40 cenários por linha.

    cliente promove 3 de 4 meses   det  início  banda  FP/mês  carga  dias cegos
    mensal automático              6,5   6,0    4,5    0,861   130,5     0
    cego (hoje)                    6,0   5,0    4,0    0,861   164,6    28,5
    segue velho                    7,0   5,0    4,0    0,861   182,4     0
    promove sozinho no dia 62      7,0   5,0    4,0    0,861   178,3     0    (5 promoções)
    só sp e vb                     6,5   5,0    4,0    0,861   177,4     0    (28,5 d degradado)
    cliente promove 1 de 2 meses
    cego (hoje)                    5,0   4,0    3,0    0,689   132,3    98,7
    segue velho                    7,0   5,0    4,0    0,861   197,9     0
    promove sozinho no dia 62      7,0   5,0    4,0    0,861   166,8     0    (8 promoções)
    só sp e vb                     7,0   5,0    4,0    0,732   153,4     0    (98,7 d degradado)

  · As três políticas novas acabam com o tempo cego e seguram a detecção. Nenhuma é
    tolerável: o início cai de 6 para 5 e a carga sobe 18% a 52%. O custo não está no
    vencimento, está nos 30-62 dias em que o champion envelhece esperando a promoção
    -- a curva de idade dá +77% de carga com um mês a mais. Política de degradação
    trata a cegueira, não o envelhecimento.
  · "Só sp e vb" parece a melhor com o cliente omisso (carga 153), mas SOZINHOS, no
    histórico inteiro, sp e vb pegam 2,5 de 8 (início 2, banda 2, FP 0,56/mês,
    carga 75). Segurou a detecção aqui porque ficou ligado só 8% a 28% do tempo. É
    quase cego, com o voto desenhado para quatro canais (o nível A, >= 3 de 4, fica
    impossível); um voto refeito para dois seria detector novo, sobre os mesmos 8
    eventos.
  · O que resolve é encurtar a espera: promoção por exceção em <= 3 dias foi
    TOLERÁVEL (`aprovacao_operador.py`). Com isso, a política de degradação vira rede
    de proteção para o caso raro; a melhor delas é "promove sozinho no dia 62".

Uso:  PYTHONPATH=. python degradacao.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import regua_fp as R
import aprovacao_operador as AO

VENCE = pd.Timedelta(days=62)


def aprovacoes(dia, q, semente):
    """o mesmo sorteio de `aprovacao_operador.irregular` (modo pronto)."""
    rng = np.random.default_rng(1000 * semente + dia)
    plano = []
    for c in AO.cortes(dia):
        aprova, lag = rng.random() < q, int(rng.integers(0, 15))
        if aprova:
            plano.append((c + lag * AO.DIA, c))
    return plano


def promove_sozinho(dia, q, semente) -> dict:
    cs = AO.cortes(dia)
    ap = sorted(aprovacoes(dia, q, semente))
    plano, forcados = [], 0
    for i, (pub, ref) in enumerate(ap):
        plano.append((pub, ref))
        prox = ap[i + 1][0] if i + 1 < len(ap) else AO.FIM
        r = ref
        while r + VENCE < prox:                     # vence antes da próxima aprovação
            quando = r + VENCE
            novo = max(c for c in cs if c <= quando)  # o challenger em sombra
            plano.append((quando, novo)); forcados += 1
            r = novo
    m = AO.de_plano(plano).mede()
    m["promocoes_automaticas"] = forcados
    return m


def so_sp_vb(dia, q, semente) -> dict:
    S = AO.de_plano(aprovacoes(dia, q, semente))
    vencido = S.cego.copy()
    S.t[vencido] = 0.0; S.p[vencido] = 0.0          # t e p apagados, sp e vb seguem
    S.cego[:] = False
    m = S.mede()
    m["dias_degradado"] = float((vencido & AO.MASK).sum() * 2 / 60 / 24)
    return m


def main():
    L = []
    for q in (0.75, 0.50):
        for nome, f in (("promove sozinho", promove_sozinho), ("só sp e vb", so_sp_vb)):
            for d in R.DIAS:
                for s in range(5):
                    L.append(dict(politica=f"q={q:.2f} {nome}", dia=d, semente=s, **f(d, q, s)))
            print(q, nome, "ok", flush=True)
    T = pd.DataFrame(L)
    T.to_csv(R.CACHE / "degradacao.csv", index=False)
    resumo()


def resumo():
    ref = R.distribuicao(); r = ref.median()
    A = pd.read_csv(R.CACHE / "aprovacao_irregular.csv")
    A = A[A.politica.str.contains("pronto")]
    V = pd.read_csv(R.CACHE / "aprovacao_sem_vencimento.csv")
    T = pd.read_csv(R.CACHE / "degradacao.csv")
    print(f"{'política':40s} {'det':>5s} {'início':>7s} {'banda':>6s} {'FP/mês':>7s} {'carga':>7s} {'cego':>6s}  veredito")
    print(f"{'mensal automático':40s} {r.det:5.1f} {r.inicio:7.1f} {r.banda:6.1f} {r.fp_mes:7.3f} {r.carga_mes:7.1f} {0:6.1f}")
    for q in (0.75, 0.50):
        blocos = [(f"q={q:.2f} cego (hoje)", A[A.politica.str.startswith(f"irregular q={q:.2f}")]),
                  (f"q={q:.2f} segue velho", V[V.politica.str.startswith(f"irregular q={q:.2f}")]),
                  (f"q={q:.2f} promove sozinho", T[T.politica == f"q={q:.2f} promove sozinho"]),
                  (f"q={q:.2f} só sp e vb", T[T.politica == f"q={q:.2f} só sp e vb"])]
        for nome, G in blocos:
            m = G.median(numeric_only=True)
            cego = m.get("dias_cegos", 0.0)
            tol = (all(m[c] >= r[c] - 0.5 for c in ("det", "inicio", "banda"))
                   and m.fp_mes <= 1.1 * r.fp_mes and m.carga_mes <= 1.1 * r.carga_mes
                   and G.get("dias_cegos", pd.Series([0])).max() == 0)
            extra = (f"  promoções automáticas {m.promocoes_automaticas:.0f}" if "promocoes_automaticas" in G
                     and G.promocoes_automaticas.notna().any() else "")
            extra += (f"  dias degradados {m.dias_degradado:.1f}" if "dias_degradado" in G
                      and G.dias_degradado.notna().any() else "")
            print(f"{nome:40s} {m.det:5.1f} {m.inicio:7.1f} {m.banda:6.1f} {m.fp_mes:7.3f} {m.carga_mes:7.1f}"
                  f" {cego:6.1f}  {'TOLERÁVEL' if tol else 'não'}{extra}")


if __name__ == "__main__":
    main()
