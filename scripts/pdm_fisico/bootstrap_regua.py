#!/usr/bin/env python3
"""Pré-registro da régua de 8 composições e bootstrap por blocos para o custo.

REGISTRADO EM 01/10/2026, ANTES de rodar o F1 (exclusão do sensor sinalizado) e o
R1 (biblioteca de referências). Nada abaixo foi escolhido olhando o resultado
deles. O que foi escolhido depois de ver dado está marcado como tal.

POR QUE MUDAR A RÉGUA. Com 8 eventos, nenhum teste estatístico certifica que uma
variante "não perde" detecção: perder 1 evento dá p = 1,0 no teste exato pareado e
perder 2 dá p = 0,5; no bootstrap, quem perde 1 evento na amostra perde >= 1 em
1 - (7/8)^8 = 66% das reamostras, e quem perde 0 perde em 0%. "Não distinguível"
na detecção aprovaria exatamente o movimento ao longo da fronteira que já se sabe
inútil (menos carga com menos detecção). Por isso a detecção continua com a regra
ESTRITA, e o bootstrap entra só no CUSTO, que é medido de forma contínua no tempo e
onde ele tem o que dizer.

══════════════════════════════════════════════════ A REGRA DE DECISÃO

A. DETECÇÃO (estrita, sem estatística).
   Desenho pareado (mesmas 8 composições): nenhuma composição perde detecção "de
   pé", e início e banda não caem na mediana. Desenho não pareado (cadência,
   semanal, com outros cenários): as medianas de det, início e banda não caem.

B. CARGA (h FP + h NEUTRO por mês de operação).
   B1. Pareado: cai em >= 7 das 8 composições (como já era).
   B2. O intervalo de confiança da diferença de carga exclui zero, a favor.
       · bloco = mês-calendário, de fev/2024 a abr/2026: 27 blocos, os mesmos para
         todas as composições (o mês alinhado ao bundle muda com a composição; o
         calendário é o que permite reamostrar todas juntas);
       · pareamento: os mesmos meses reamostrados nos dois braços e nas 8
         composições ao mesmo tempo;
       · estatística: a MÉDIA, entre as composições, da carga de cada uma, com a
         carga calculada como razão (soma das horas de alarme / soma das horas de
         operação dos meses sorteados). A média, não a mediana, que tem
         distribuição ruim em bootstrap;
       · B = 10.000 reamostras, semente 20261001, IC de percentil.
C. FP/mês: não sobe na mediana (como já era). O IC é relatado, não decide.

D. MULTIPLICIDADE. Cada teste declara UMA variante primária, julgada com IC de 95%.
   As demais da mesma família são secundárias, julgadas com Bonferroni: IC de
   1 - 0,05/k (k = tamanho da família) e não podem virar o veredito sozinhas.

E. GANHO = A e B1 e B2 e C. Andar ao longo da fronteira não conta: menos carga com
   menos detecção é reprovado por A, qualquer que seja o IC.

F. REGRA DE SEGURANÇA (não reivindica ganho): adotada se A vale e as estimativas
   pontuais de carga e de FP (média entre composições) não sobem. Só se fala em
   ganho se também passar B2.

══════════════════════════════════════════════════ F1, REGISTRADO ANTES

Excluir do canal o sensor que o monitor sinalizou, até o próximo bundle. Gatilhos:
  · degrau persistente (>= 10 sigma em duas semanas seguidas, mesmo sinal), com
    NENHUM outro sensor do mesmo canal >= 10 sigma na semana e a semana terminando
    até 21 dias depois de uma manutenção (HSX >= 24 h). REGISTRO: a condição "nenhum
    outro >= 10 sigma" foi escolhida DEPOIS de ver o caso de nov/2025 (n = 1): com
    "nenhum outro >= 3 sigma" a regra não dispara nenhuma vez no histórico, porque
    a máquina volta da manutenção com vários sensores em nível novo. Os 10 sigma são
    o limiar que o monitor já usava;
  · sensor travado (a regra do monitor), sem condição de manutenção.
Implementação: a partir do instante do gatilho, o bundle do canal é refeito SEM o
sensor (PCA e p99 recalculados juntos, porque o par é atômico), até o próximo
corte. O contrato ganha a notificação "SENSOR fora do canal X desde DD/MM, provável
rezero/instrumento". Julgado pela regra F (segurança).

══════════════════════════════════════════════════ R1, REGISTRADO ANTES

Biblioteca de referências. Para o mês M de cada composição: o bundle do mês, intacto,
mais os K - 1 bundles mensais anteriores, refeitos com +-7 dias em volta dos trips já
ocorridos antes do corte deles excluídos do ajuste. A exclusão só vale para as
referências mais velhas: no bundle do mês ela derrubaria o p99 e traria de volta o
efeito já medido de 6,6 -> 68 h/mês, que confundiria o teste.
  mínimo   o score de t e de p é o MÍNIMO, entre as K referências, do score
           normalizado de cada uma; sp com o centro/escala da referência que deu
           o mínimo em t.
  estado   no início de cada semana (segunda 00:00 UTC), escolhe a referência cuja
           média do baseline está mais perto (Mahalanobis, covariância Ledoit-Wolf
           do próprio baseline) da média dos 7 dias vigiados anteriores, SÓ nas
           variáveis de ponto de operação: T5_AVG_A, TI_0315 e TI_0317 (T7 de
           exaustão), PI_0315 (gás combustível), PDI_0317 (delta P gás/PCD).
           Fora: mancal, óleo, selagem e vibração, os sensores que carregam os
           trips. A semana inteira usa só essa referência; o CUSUM não reinicia
           na troca, como não reinicia na troca mensal.
  Família de 4: mínimo K=2 (PRIMÁRIA, IC 95%), mínimo K=3, estado K=2, estado K=3
  (secundárias, IC 98,75%).
  EXPECTATIVA REGISTRADA: o "estado" é parente da normalização por regime, já
  reprovada -- expectativa baixa. O "mínimo" deve cortar carga; a dúvida é a
  detecção (a memória do CUSUM é a cola do voto, e toda redução de tempo aceso
  medida até aqui custou nascimento).

══════════════════════════════════════════════════ O QUE ESTE SCRIPT FAZ

Implementa A-F e aplica aos experimentos já feitos, para (1) conferir que o
bootstrap reproduz os números publicados quando não se reamostra, (2) saber de
quanto a carga precisa cair para o IC excluir zero, e (3) reclassificar. Alcance
limitado, registrado: com A estrita, quem perdeu detecção continua reprovado
qualquer que seja o IC. A reclassificação só pode mudar o veredito de quem passou
em A e falhou no custo.

══════════════════════════════════════════════════ RESULTADO (01/10/2026)

CONFERE. Sem reamostrar, a carga por blocos reproduz a do `mede` com erro de
3e-14 h/mês (8.476 h de operação em 27 meses). Referência: carga média 133,4 h/mês
(mediana 130,5).

QUANTO A CARGA PRECISA CAIR. Meia-largura do IC 95% da diferença: mediana de 34,6
h/mês (faixa 5,1 a 72,7). Nos desenhos pareados, de 5 a 50; nos não pareados
(retreino semanal/quinzenal, outros cenários), de 50 a 73. Para o R1 (pareado),
uma redução abaixo de ~20-35 h/mês não vai excluir zero.

RECLASSIFICAÇÃO: NENHUM VEREDITO MUDA.
    variante                   det  ini  banda   Δcarga [IC 95%]          ΔFP [IC 95%]            veredito
    teto do CUSUM 1,25-4H     5-5,5 4,5   3,5   -37 a -50, IC < 0         +0,24 a +0,52, IC > 0   A, C
    teto 8H                    5,5  5,0   3,5   -29,9 [-72,5; +1,9]       +0,18 [+0,04; +0,35]    A, B2, C
    headstart / rezero        4-4,5 3,5-4 3,0   -80 a -82, IC < 0         +0,66 a +0,70, IC > 0   A, C
    EWMA vigiado               5,5  4,0   3,0   -22,8 [-38,7; -9,4]       -0,25 [-0,53; +0,02]    A
    blackout 24/32/48 h       4-5  2,5-3 1,5-2,5 -11 a +8, IC inclui 0    -0,15 a -0,21           A, B1, B2
    transição observada        4,0  2,0   2,0   -82,6 [-137,9; -36,6]     -0,41 [-0,79; -0,07]    A
    recentragem (4 formas)     7,0  5,5   4,5   -4 a +4, IC estreito      +0,08 a +0,17           A, B1, B2, C
    semanal / quinzenal       4-5  2-4   1-3    -22 / -13, IC inclui 0    +0,20 / +0,21           A, B2, C
    semanal, guarda 11 d       6,0  5,0   4,0   -67,2 [-127,2; -19,1]     -0,08 [-0,36; +0,17]    A
    semanal, guarda 21 d       7,0  7,0   5,0   +17,4 [-61,4; +84,1]      -0,13 [-0,37; +0,03]    B2

  · O padrão da fronteira, agora com estatística: o teto do CUSUM, o EWMA
    vigiado, a transição observada e a guarda de 11 dias cortam a carga de forma
    SIGNIFICATIVA (o teto e a guarda, mesmo com Bonferroni) e reprovam todos pela
    detecção. Corte de carga real, comprado com nascimento.
  · O teto do CUSUM SOBE o FP com IC inteiro acima de zero: "horas e contagem
    andam em sentidos opostos" deixa de ser leitura e vira resultado.
  · Os aumentos de carga relatados antes não se distinguem de zero: a guarda de
    21 dias (+20%, IC -61 a +84) e o blackout de 48 h. A guarda de 21 dias é a
    única que passa em A (melhor nas três réguas) e reprova só por B2 -- não há
    redução de carga para mostrar, mas também não há aumento demonstrável.
  · A recentragem tem efeito pequeno e bem medido (IC de ~+-6 h/mês): não muda
    nada, para nenhum lado, além de subir o FP.

Uso:  PYTHONPATH=. python bootstrap_regua.py
"""
from __future__ import annotations
import io, contextlib
import numpy as np, pandas as pd
import regua_fp as R

MESES = pd.date_range("2024-02-01", "2026-05-01", freq="MS", tz="UTC")     # 28 bordas, 27 blocos
B, SEMENTE = 10_000, 20261001
DOIS_MIN = pd.Timedelta("2min")


def _horas_op() -> np.ndarray:
    m = pd.Series(R.mask.to_numpy().astype(bool), index=R.idx)
    return np.array([m[(m.index >= a) & (m.index < b)].sum() * 2 / 60 for a, b in zip(MESES[:-1], MESES[1:])])


OP = _horas_op()


def blocos(fin: pd.Series) -> dict:
    """horas de carga, horas e contagem de FP, por mês, a partir do alarme final."""
    eps = R.AV.episodios(fin)
    cls = R.DB.classifica_regra_c(eps, R.PARADAS)
    hc, hf, nf = (np.zeros(len(MESES) - 1) for _ in range(3))
    for a, b, k, _ in cls:
        if k == "TP":
            continue
        fim = b + DOIS_MIN
        for i, (m0, m1) in enumerate(zip(MESES[:-1], MESES[1:])):
            h = (min(fim, m1) - max(a, m0)).total_seconds() / 3600
            if h > 0:
                hc[i] += h
                if k == "FP":
                    hf[i] += h
        if k == "FP":
            i = MESES.searchsorted(a, side="right") - 1
            if 0 <= i < len(nf):
                nf[i] += 1
    return dict(h_carga=hc, h_fp=hf, n_fp=nf)


class Braco:
    """um braço = uma lista de alarmes finais, um por cenário (composição)."""

    def __init__(self, nome: str, fins: list[pd.Series], tabela: pd.DataFrame):
        self.nome, self.tabela = nome, tabela
        bl = [blocos(f) for f in fins]
        self.H = np.vstack([b["h_carga"] for b in bl])            # cenários x meses
        self.F = np.vstack([b["n_fp"] for b in bl])

    def carga(self, W):          # W: reamostras x meses (contagens)
        return (W @ self.H.T) / (W @ OP)[:, None] * 730.0         # reamostras x cenários

    def fp(self, W):
        return (W @ self.F.T) / (W @ OP)[:, None] * 730.0


def pesos(b=B, semente=SEMENTE) -> np.ndarray:
    rng = np.random.default_rng(semente)
    n = len(MESES) - 1
    return rng.multinomial(n, np.full(n, 1 / n), size=b).astype(float)


W_BOOT = pesos()


def ic(ref: Braco, var: Braco, nivel=0.95, f="carga") -> tuple[float, float, float]:
    """diferença (var - ref) da média entre cenários, com IC de percentil."""
    g = (lambda br, W: br.carga(W)) if f == "carga" else (lambda br, W: br.fp(W))
    um = np.ones((1, len(MESES) - 1))
    ponto = float(g(var, um).mean() - g(ref, um).mean())
    d = g(var, W_BOOT).mean(axis=1) - g(ref, W_BOOT).mean(axis=1)
    a = (1 - nivel) / 2
    return ponto, float(np.quantile(d, a)), float(np.quantile(d, 1 - a))


def decide(ref: Braco, var: Braco, pareado=True, nivel=0.95) -> dict:
    r, v = ref.tabela, var.tabela
    if pareado:
        A = bool((v.det.values >= r.det.values).all() and v.inicio.median() >= r.inicio.median()
                 and v.banda.median() >= r.banda.median())
        B1 = bool((v.carga_mes.values < r.carga_mes.values).sum() >= 7)
    else:
        A = bool(v.det.median() >= r.det.median() and v.inicio.median() >= r.inicio.median()
                 and v.banda.median() >= r.banda.median())
        B1 = True
    dc, lo, hi = ic(ref, var, nivel, "carga")
    df, flo, fhi = ic(ref, var, nivel, "fp")
    B2 = hi < 0
    C = bool(v.fp_mes.median() <= r.fp_mes.median())
    if A and B1 and B2 and C:
        vered = "GANHO"
    elif A and dc <= 0 and df <= 0:
        vered = "seguro"
    else:
        vered = "reprovado (" + ", ".join(n for n, ok in (("A", A), ("B1", B1), ("B2", B2), ("C", C)) if not ok) + ")"
    return dict(A=A, B1=B1, B2=B2, C=C, d_carga=dc, ic_lo=lo, ic_hi=hi, d_fp=df, fp_lo=flo, fp_hi=fhi,
                det=f"{v.det.median():.1f}", inicio=f"{v.inicio.median():.1f}", banda=f"{v.banda.median():.1f}",
                carga=float(v.carga_mes.mean()), veredito=vered)


def braco(nome, gerador, cenarios) -> Braco:
    fins, linhas = [], []
    for c in cenarios:
        fin = gerador(c)
        m = R.mede(fin); m.pop("cls")
        fins.append(fin); linhas.append(dict(cenario=c, **m))
    return Braco(nome, fins, pd.DataFrame(linhas).set_index("cenario"))


def confere(ref: Braco) -> float:
    """sem reamostrar, a carga por blocos tem de reproduzir a do `mede`."""
    um = np.ones((1, len(MESES) - 1))
    return float(np.max(np.abs(ref.carga(um)[0] - ref.tabela.carga_mes.values)))


# ══════════════════════════════════════════════════ os experimentos já feitos
def familias():
    with contextlib.redirect_stdout(io.StringIO()):
        import cusum_memoria as CM, recentragem as RC, retreino_semanal as RS
    det = lambda **kw: (lambda d: R.detector(*R.sinais(d), **kw)["fin"])

    def recentra(N, reset):
        def g(d):
            if not reset:
                sig, _ = RC.sinais(d, N)
                return R.detector(*sig)["fin"]
            sig, usadas = RC.walkforward_recentrado(d, N)
            rx = np.zeros(len(R.idx), dtype=bool)
            for _, _, s0 in usadas:
                i = R.idx.searchsorted(s0); rx[i:i + RC.N_RESET] = True
            return R.detector(*sig, reset_extra=rx)["fin"]
        return g

    def guarda(g):
        def f(d):
            z = np.load(R.CACHE / f"guarda{g}_d{d:02d}.npz")
            return R.detector(z["t"], z["p"], z["ms"], z["ds"])["fin"]
        return f

    return {
        "memória do CUSUM": [(CM.rotulo(m, q), det(cusum=(m, q)), True) for m, q in CM.VARIANTES],
        "EWMA vigiado": [("4 canais", det(ewma_vigiado=("t", "p", "sp", "vb")), True),
                         ("só t e p", det(ewma_vigiado=("t", "p")), True)],
        "regimes (blackout)": [(f"blackout {h:.0f} h", det(blackout_h=h), True) for h in (24.0, 32.0, 48.0)],
        "transição observada": [("exige transição", det(exige_transicao=True), True)],
        "recentragem": [(f"N={N}{' + reset' if rs else ''}", recentra(N, rs), True) for rs in (False, True) for N in (3, 7)],
        "retreino": [("semanal", lambda d: R.detector(*RS.sinais(7, d))["fin"], False),
                     ("quinzenal", lambda d: R.detector(*RS.sinais(14, 2 * d))["fin"], False),
                     ("semanal guarda 11 d", guarda(11), False),
                     ("semanal guarda 21 d", guarda(21), False)],
    }


def main():
    pd.set_option("display.width", 220)
    ref = braco("referência", lambda d: R.detector(*R.sinais(d))["fin"], R.DIAS)
    print(f"confere (carga por blocos sem reamostrar vs mede): erro máx {confere(ref):.2e} h/mês;"
          f" horas de operação {OP.sum():.0f} em {len(OP)} meses")
    print(f"referência: carga média {ref.tabela.carga_mes.mean():.1f} h/mês (mediana {ref.tabela.carga_mes.median():.1f})\n")
    L = []
    for fam, itens in familias().items():
        k = len(itens)
        for nome, g, par in itens:
            cen = R.DIAS if par else range(7)
            v = braco(nome, g, cen)
            r95 = decide(ref, v, par, 0.95)
            rb = decide(ref, v, par, 1 - 0.05 / k)
            L.append(dict(familia=fam, variante=nome, k=k, **{kk: r95[kk] for kk in
                     ("det", "inicio", "banda", "carga", "d_carga", "ic_lo", "ic_hi", "d_fp", "fp_lo", "fp_hi",
                      "A", "B1", "C")}, B2_95=r95["B2"], ic_bonf_hi=rb["ic_hi"], veredito=r95["veredito"]))
            x = L[-1]
            print(f"{fam:20s} {nome:22s} det {x['det']:>4s} ini {x['inicio']:>4s} banda {x['banda']:>4s} | "
                  f"Δcarga {x['d_carga']:+7.1f} [{x['ic_lo']:+7.1f}; {x['ic_hi']:+7.1f}] "
                  f"Bonf.≤{x['ic_bonf_hi']:+7.1f} | ΔFP {x['d_fp']:+.3f} [{x['fp_lo']:+.3f}; {x['fp_hi']:+.3f}] | {x['veredito']}",
                  flush=True)
    T = pd.DataFrame(L)
    T.to_csv(R.CACHE / "bootstrap_regua.csv", index=False)
    meia = ((T.ic_hi - T.ic_lo) / 2)
    print(f"\nmeia-largura do IC 95% da Δcarga: mediana {meia.median():.1f} h/mês "
          f"(faixa {meia.min():.1f}-{meia.max():.1f}), contra carga de referência {ref.tabela.carga_mes.mean():.1f}")


if __name__ == "__main__":
    main()
