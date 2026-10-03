#!/usr/bin/env python3
"""Quanto do desempenho publicado é sobreajuste dos parâmetros?

PRÉ-REGISTRADO EM 02/10/2026, ANTES DE RODAR (commit deste arquivo sem resultado).

A PERGUNTA, do usuário: "você, como especialista, está tendo overfitting?". Os ~13 parâmetros do
detector foram escolhidos olhando os mesmos 8 eventos, e o ponto publicado é o MELHOR das 8
composições de baseline (8/8 e 0,344 FP/mês no dia 1; mediana 6,5/8 e 0,86). Aqui se mede o
tamanho do excesso de otimismo, em duas dimensões distintas.

O ESPAÇO REAJUSTÁVEL (243 configurações, o ponto publicado incluído):
  K do nível A  {t 1,10; p 1,20; sp 0,90; vb 2,00} x {0,8; 1,0; 1,25} por canal   = 81
  K do nível B  {1,7; 1,7; 1,7; 2,2} x {0,85; 1,0; 1,2}, os quatro juntos          =  3
Voto, CUSUM, refratário, duração e escalada ficam fixos. Cada configuração roda nas 8
composições de baseline; guarda-se, por evento: detectado, nasce na janela, nasce na banda; e
FP/mês e carga.

OS DOIS TESTES, a partir da mesma tabela.
  A  ENTRE COMPOSIÇÕES. Para cada composição X, escolhe-se a configuração que maximiza a detecção
     em X, depois a banda, depois minimiza a carga, depois a que mais se parece com a publicada.
     Avalia-se essa escolha em X (dentro da amostra) e nas outras 7 (fora). Mede: (i) o ganho
     dentro da amostra; (ii) se a escolha vence a publicada fora; (iii) o OTIMISMO = métrica em X
     menos a média nas outras, para a escolha e para a publicada (dia 1 contra as demais).
     LIMITE, dito antes: as 8 composições compartilham os MESMOS 8 eventos. Isto mede o
     sobreajuste à composição do baseline, NÃO aos eventos.
  B  DEIXANDO UM EVENTO DE FORA (o que importa). Para cada evento e: escolhe-se a configuração
     que maximiza as detecções dos OUTROS 7 (somadas nas 8 composições), depois minimiza a carga
     média, depois a mais próxima da publicada. Avalia-se se ela detecta e, na fração de
     composições. Compara-se com a publicada, que viu o evento e (viés a favor dela): a diferença é
     uma estimativa do otimismo de escolher parâmetros em poucos eventos.

EXPECTATIVA REGISTRADA.
  A  dentro da amostra o reajuste melhora (a publicada está na grade, então não pode piorar); fora,
     NÃO vence a publicada de forma sistemática (carga não cai em >= 7 de 8 sem perder detecção); e
     o otimismo da publicada (dia 1: 8/8, 0,344, 49 h/mês contra 6,0/8, 0,95 e 146 h/mês nas
     demais) é MAIOR que o típico de uma escolha reajustada em outra composição.
  B  a configuração escolhida sem o evento e detecta e em menos composições que a publicada, na
     média dos 8 eventos; o 04/11/2025 é o que mais cai.
  Se o reajuste vencer fora de forma sistemática, isso é um candidato a estudar, não decisão.

RESULTADO (02/10/2026). A expectativa se confirmou no que mede os eventos e FALHOU no que mede o dia 1.

  B  POR EVENTO (o que importa). Escolher os parâmetros sem ver um evento não custa nada nesta
     vizinhança: detecção do evento deixado de fora 0,797 (reajustada) contra 0,797 (publicada,
     que o viu), banda 0,500 contra 0,500. A escolha sem o evento é a MESMA configuração (153) em 7
     dos 8; a melhor da grade tem 53/64 detecções-evento, a publicada 51. LIMITE: a grade é local
     (+-25%), então só exclui sobreajuste dos K; estrutura, voto, CUSUM e refratário ficam fora. E o
     04/11/2025 NÃO é o que mais cai (1,00); o 29/04/2025 é detectado em 12% das composições.
  A  ENTRE COMPOSIÇÕES. Reajustar em X e avaliar nas outras 7: dentro da amostra detecção 6,38 ->
     6,88 e carga 133 -> 86 h/mês; fora, detecção -0,57, carga -24 h/mês, FP +0,07. Vence a
     publicada em 1 de 8. O "otimismo" (métrica em X menos nas outras) MISTURA sobreajuste com
     dificuldade da composição (na 15 dá -32 h/mês: ela é só mais difícil). A medida limpa é
     comparar a publicada com uma configuração TÍPICA da grade na mesma composição:
       configuração típica, dia 1: carga 54 h/mês, detecção 7,0 | outras 7: 117 h/mês, 5,9.
     O número publicado (49 h/mês, 8/8) é sobretudo a composição do dia 1, que é fácil: nem uma
     configuração típica nem o reajuste dentro da amostra levam as outras a ele (65 a 125).
     O que o ajuste faz: no dia 1, +1,0 detecção e +5 h/mês contra a típica; nas outras 7, +0,29
     detecção a +28 h/mês PIOR (67 a 97% da grade tem carga menor que a publicada em 4 a 25).
  C  (A POSTERIORI) deixando uma COMPOSIÇÃO de fora. A configuração 153 (K do nível A: t 1,10,
     p 1,50, sp 1,125, vb 1,60; K do nível B x0,85) escolhida nas outras 7 domina a publicada em 7
     de 8: carga -26 h/mês (-36 sem o dia 1), detecção +0,12, banda igual. Mas o FP sobe +0,25/mês
     (mediana 0,861 -> 0,990), de modo que REPROVA em C da regra pré-registrada: corta horas e
     parte episódios em mais episódios, como o teto do CUSUM. Só perde no dia 1 (carga 93 contra 49).
  LEITURA. Há sobreajuste, mas não onde se temia: (1) não aos eventos, nos K; (2) sim ao DIA 1: o
  critério "8/8 no dia 1" empurra o ponto para um canto sensível e caro nas demais composições;
  (3) o número de título é sobretudo sorte da composição. Nada disso muda veredito: a 153 não passa
  em C, e é a posteriori. Sugere congelar os DOIS no modo sombra, e não escolher agora.

Uso:  PYTHONPATH=. python sobreajuste.py roda       # a tabela (paralelo, ~10 min)
      PYTHONPATH=. python sobreajuste.py analisa    # os dois testes sobre a tabela
      PYTHONPATH=. python sobreajuste.py loco       # C (a posteriori): deixando uma composição de fora
"""
from __future__ import annotations
import io, contextlib, itertools, multiprocessing as mp, sys
import numpy as np, pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    import regua_fp as R
    import avalia as AV

DB = R.DB
SIN = list(R.SIN)
BASE_LO = dict(DB.K_LO); BASE_HI = dict(DB.KH)
M_LO, M_HI = (0.8, 1.0, 1.25), (0.85, 1.0, 1.2)
JAN, TMIN = pd.Timedelta(hours=48), pd.Timedelta(hours=4)
TAB = R.CACHE / "sobreajuste_tabela.csv"


def configs() -> list[tuple]:
    return [(m, h) for m in itertools.product(M_LO, repeat=4) for h in M_HI]


def roda_config(i: int) -> list[dict]:
    mult, h = configs()[i]
    for c, m in zip(SIN, mult):
        DB.K_LO[c] = BASE_LO[c] * m
    for c in SIN:
        DB.KH[c] = BASE_HI[c] * h
    linhas = []
    try:
        for d in R.DIAS:
            fin = R.detector(*R.sinais(d))["fin"]
            eps = AV.episodios(fin)
            met = R.mede(fin); met.pop("cls")
            det = "".join("1" if bool(fin.loc[(R.idx >= T - JAN) & (R.idx < T)].any()) else "0" for T in R.alvo)
            ban = "".join("1" if any(T - JAN <= a <= T - TMIN for a, _ in eps) else "0" for T in R.alvo)
            linhas.append(dict(cfg=i, m_t=mult[0], m_p=mult[1], m_sp=mult[2], m_vb=mult[3], m_hi=h, dia=d,
                               det=met["det"], inicio=met["inicio"], banda=met["banda"], fp_mes=met["fp_mes"],
                               carga_mes=met["carga_mes"], det_ev=det, ban_ev=ban))
    finally:
        for c in SIN:
            DB.K_LO[c] = BASE_LO[c]; DB.KH[c] = BASE_HI[c]
    return linhas


def roda():
    n = len(configs())
    print(f"{n} configurações x {len(R.DIAS)} composições", flush=True)
    out = []
    with mp.get_context("fork").Pool(6) as p:
        for k, linhas in enumerate(p.imap_unordered(roda_config, range(n), chunksize=2), 1):
            out += linhas
            if k % 20 == 0:
                print(f"  {k}/{n}", flush=True)
    pd.DataFrame(out).sort_values(["cfg", "dia"]).to_csv(TAB, index=False)
    print("->", TAB)


def analisa():
    T = pd.read_csv(TAB, dtype={"det_ev": str, "ban_ev": str})
    pub = int(T[(T.m_t == 1) & (T.m_p == 1) & (T.m_sp == 1) & (T.m_vb == 1) & (T.m_hi == 1)].cfg.iloc[0])
    dist = (np.abs(np.log(T.groupby("cfg")[["m_t", "m_p", "m_sp", "m_vb", "m_hi"]].first())).sum(axis=1))
    P = T.pivot(index="cfg", columns="dia")
    ev = lambda col: np.array([[list(map(int, P[(col, d)].loc[c])) for d in R.DIAS] for c in P.index])   # cfg x dia x evento
    DET, BAN = ev("det_ev"), ev("ban_ev")
    carga = P["carga_mes"].to_numpy(); fp = P["fp_mes"].to_numpy(); detn = P["det"].to_numpy()
    idx = {c: k for k, c in enumerate(P.index)}; kp = idx[pub]
    nC = len(P.index)
    print(f"{nC} configurações; publicada = cfg {pub}\n")

    print("=" * 100); print("A. ENTRE COMPOSIÇÕES (mesmos 8 eventos: mede o sobreajuste à composição do baseline)"); print("=" * 100)
    linhas = []
    for x, d in enumerate(R.DIAS):
        key = sorted(range(nC), key=lambda c: (-detn[c, x], -BAN[c, x].sum(), carga[c, x], dist.iloc[c]))
        c = key[0]; outras = [y for y in range(len(R.DIAS)) if y != x]
        linhas.append(dict(
            X=d, cfg=int(P.index[c]), igual_a_publicada=c == kp,
            det_dentro=detn[c, x], det_pub_X=detn[kp, x], carga_dentro=carga[c, x], carga_pub_X=carga[kp, x],
            d_det_fora=detn[c, outras].mean() - detn[kp, outras].mean(), d_carga_fora=carga[c, outras].mean() - carga[kp, outras].mean(),
            d_fp_fora=fp[c, outras].mean() - fp[kp, outras].mean(),
            otim_det=detn[c, x] - detn[c, outras].mean(), otim_carga=carga[c, outras].mean() - carga[c, x]))
    A = pd.DataFrame(linhas); pd.set_option("display.width", 220)
    print(A.round(2).to_string(index=False))
    venc = int(((A.d_carga_fora < 0) & (A.d_det_fora >= 0)).sum())
    print(f"\n  dentro da amostra: detecção {A.det_pub_X.mean():.2f} -> {A.det_dentro.mean():.2f}, carga {A.carga_pub_X.mean():.0f} -> {A.carga_dentro.mean():.0f} h/mês")
    print(f"  FORA, escolha reajustada menos publicada (média das outras 7): detecção {A.d_det_fora.mean():+.2f}, "
          f"carga {A.d_carga_fora.mean():+.0f} h/mês, FP {A.d_fp_fora.mean():+.3f}/mês")
    print(f"  composições em que a escolha reajustada vence a publicada fora (carga menor E detecção >=): {venc}/8")
    out = [y for y in range(1, len(R.DIAS))]
    print(f"\n  OTIMISMO (métrica na composição de ajuste menos a média nas outras 7)")
    print(f"    escolhas reajustadas: detecção {A.otim_det.mean():+.2f}, carga {A.otim_carga.mean():+.0f} h/mês (média das 8)")
    print(f"    publicada, dia 1:     detecção {detn[kp, 0] - detn[kp, out].mean():+.2f}, "
          f"carga {carga[kp, out].mean() - carga[kp, 0]:+.0f} h/mês, FP {fp[kp, out].mean() - fp[kp, 0]:+.3f}/mês")

    print("\n" + "=" * 100); print("B. DEIXANDO UM EVENTO DE FORA (a escolha de parâmetros não vê o evento)"); print("=" * 100)
    linhas = []
    for e, T_ in enumerate(R.alvo):
        outros = [k for k in range(len(R.alvo)) if k != e]
        score = DET[:, :, outros].sum(axis=(1, 2)); cm = carga.mean(axis=1)
        key = sorted(range(nC), key=lambda c: (-score[c], cm[c], dist.iloc[c])); c = key[0]
        linhas.append(dict(evento=T_.strftime("%Y-%m-%d"), cfg=int(P.index[c]), igual_a_publicada=c == kp,
                           detecta_reajustada=DET[c, :, e].mean(), detecta_publicada=DET[kp, :, e].mean(),
                           banda_reajustada=BAN[c, :, e].mean(), banda_publicada=BAN[kp, :, e].mean()))
    B = pd.DataFrame(linhas); print(B.round(2).to_string(index=False))
    print(f"\n  detecção do evento deixado de fora (fração das 8 composições): reajustada {B.detecta_reajustada.mean():.3f} "
          f"| publicada (que viu todos) {B.detecta_publicada.mean():.3f} | diferença {B.detecta_reajustada.mean() - B.detecta_publicada.mean():+.3f}")
    print(f"  banda:                                                         reajustada {B.banda_reajustada.mean():.3f} "
          f"| publicada {B.banda_publicada.mean():.3f} | diferença {B.banda_reajustada.mean() - B.banda_publicada.mean():+.3f}")
    print(f"  eventos em que a escolha sem o evento coincide com a publicada: {int(B.igual_a_publicada.sum())}/8")
    sc = DET.sum(axis=(1, 2)); print(f"\n  melhor da grade vendo os 8 eventos: {sc.max()}/{DET.shape[1] * DET.shape[2]} detecções-evento; publicada: {sc[kp]}; "
                                    f"configurações no máximo: {int((sc == sc.max()).sum())} de {nC}")
    A.to_csv(R.CACHE / "sobreajuste_A.csv", index=False); B.to_csv(R.CACHE / "sobreajuste_B.csv", index=False)


def loco():
    """C (A POSTERIORI, não pré-registrada): deixando uma COMPOSIÇÃO de fora. Escolhe-se a configuração
    nas outras 7 (mais detecções somadas, depois menor carga média, depois a mais próxima da
    publicada) e avalia-se na composição deixada de fora, contra a publicada. Os eventos continuam
    os mesmos 8: isto testa a generalização entre composições, não entre eventos."""
    T = pd.read_csv(TAB, dtype={"det_ev": str, "ban_ev": str})
    pub = int(T[(T.m_t == 1) & (T.m_p == 1) & (T.m_sp == 1) & (T.m_vb == 1) & (T.m_hi == 1)].cfg.iloc[0])
    dist = np.abs(np.log(T.groupby("cfg")[["m_t", "m_p", "m_sp", "m_vb", "m_hi"]].first())).sum(axis=1)
    P = T.pivot(index="cfg", columns="dia"); detn = P["det"].to_numpy(); carga = P["carga_mes"].to_numpy()
    banda = P["banda"].to_numpy(); fp = P["fp_mes"].to_numpy(); nC = len(P.index); kp = list(P.index).index(pub)
    linhas = []
    for y, d in enumerate(R.DIAS):
        res = [k for k in range(len(R.DIAS)) if k != y]
        key = sorted(range(nC), key=lambda c: (-detn[c, res].sum(), carga[c, res].mean(), dist.iloc[c])); c = key[0]
        linhas.append(dict(deixada=d, cfg=int(P.index[c]), d_det=detn[c, y] - detn[kp, y], d_banda=banda[c, y] - banda[kp, y],
                           d_carga=carga[c, y] - carga[kp, y], d_fp=fp[c, y] - fp[kp, y]))
    L = pd.DataFrame(linhas); print(L.round(2).to_string(index=False))
    print(f"\n  na composição deixada de fora, escolha sem ela menos publicada: detecção {L.d_det.mean():+.2f}, banda {L.d_banda.mean():+.2f}, "
          f"carga {L.d_carga.mean():+.1f} h/mês, FP {L.d_fp.mean():+.3f}/mês")
    print(f"  carga menor em {int((L.d_carga < 0).sum())}/8 | detecção não menor em {int((L.d_det >= 0).sum())}/8 | "
          f"carga menor E detecção não menor: {int(((L.d_carga < 0) & (L.d_det >= 0)).sum())}/8 | sem o dia 1: carga {L.d_carga[L.deixada != 1].mean():+.1f}")
    L.to_csv(R.CACHE / "sobreajuste_C.csv", index=False)


if __name__ == "__main__":
    {"roda": roda, "analisa": analisa, "loco": loco}[sys.argv[1]]()
