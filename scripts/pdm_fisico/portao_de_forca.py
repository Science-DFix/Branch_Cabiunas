#!/usr/bin/env python3
"""P1: portão de força no nascimento ("aparar a cabeça" do episódio até a força chegar a θ).

PRÉ-REGISTRADO EM 03/10/2026, ANTES DE RODAR (commit deste arquivo sem resultado).

DE ONDE VEM. De `anatomia_fp_p.py` e `vivo_no_nascimento.py` (commit 58e5b24), diagnósticos que olharam
os mesmos 8 eventos: nos 54 episódios distintos das 8 composições (FP 25, NEUTRO 15, TP 14), a força nas
2 h de confirmação separa TP de FP fracamente (mediana 8,1 contra 3,5; AUC 0,70; p = 0,045, nominal:
não passa Bonferroni). É a mesma direção do "piso de força" de 19/09 (`piso_no_nascimento.py`), que foi
descartado porque o teste usava os FP que o criaram. HIPÓTESE NASCIDA DE OLHAR ESTES DADOS, e dito antes.

A REGRA. Força F = max_c EWMA_c / (BASE_c x KH_c), a mesma da escalada por idade (`regua_fp.detector`).
Para cada episódio de voto (os instantes de voto agrupados com folga <= 2 h, como `avalia.episodios`):
  · se F nunca chega a θ dentro dele, o episódio inteiro é descartado;
  · senão, o voto é apagado do início do episódio até o primeiro instante com voto e F >= θ. O episódio
    NASCE nesse instante; o resto do pós-processamento (escalada, refratário, duração mínima de 120 min,
    contada daí) é o de sempre.
POR QUE APARAR E NÃO DESCARTAR O EPISÓDIO PELA FORÇA DAS 2 PRIMEIRAS HORAS: descartar perderia o TP que
cresce devagar (força 1,2 a 2,5 no início de 3 dos 14 TP distintos); aparar só atrasa o nascimento dele.
Decidido pelo raciocínio, antes de olhar qualquer resultado. θ = 0 reproduz a referência bit a bit.

A CONFIGURAÇÃO. UM único θ, fixado agora: θ = 2,0. É 2x o limiar do nível B: ao menos um canal a duas
vezes o seu limiar específico. θ = 1,5 e θ = 3,0 rodam e são IMPRESSOS COMO SENSIBILIDADE, sem veredito e
sem escolha. Voto, limiares, CUSUM, refratário e duração não mudam.

A DECISÃO. `bootstrap_regua.decide`, pareado, nas 8 composições de sempre, IC de 97,5% (Bonferroni sobre os
dois critérios abaixo). Dois critérios, escritos agora:
  PRIMÁRIO (a regra de sempre, sem alteração): GANHO = nenhuma composição perde detecção, medianas de
    início e banda não caem, carga (h de FP + NEUTRO) cai em >= 7/8 com IC abaixo de zero e FP/mês não sobe.
  SECUNDÁRIO (o alvo desta ideia, a CONTAGEM): GANHO-CONTAGEM = a mesma condição de detecção, início e banda,
    mais o IC do ΔFP/mês abaixo de zero e a carga média não sobe. Existe porque o mecanismo corta
    episódios, não horas, e o primário poderia reprovar uma melhora real de contagem.
Se só o secundário passar, o veredito é "melhora a contagem sem cortar horas", candidato a validação em dado
novo e a discussão, não decisão de produção.

CHECAGEM DE MECANISMO (sempre, a regra C não limita duração): por composição, quantos episódios da
referência somem, quantos nascem mais tarde e quanto (mediana, h), quantas horas de FP e NEUTRO somem, e se
aparece episódio NEUTRO novo e longo no lugar de um FP removido.

EXPECTATIVA REGISTRADA.
  · A contagem de FP cai (a curva dos episódios distintos remove 11 de 25 FP com θ = 2). A carga em horas
    cai pouco: as maiores forças de FP (35 a 170) são os episódios longos, que o portão não toca.
  · Risco alto no critério de detecção estrita: o TP com força 1,2 (17 h) pode ser o único alarme de um
    trip em alguma composição. Chance de passar o primário: baixa (~20%); o secundário, média (~35%).
  · O padrão da pesquisa é a fronteira: cortar episódio tem custado nascimento.
RESSALVA: é uma hipótese a posteriori nos mesmos 8 eventos e nos mesmos dias dos 25 FP que a inspiraram.

RESULTADO DO P1 (03/10/2026) -- REPROVA NOS DOIS CRITÉRIOS; O MECANISMO MOSTRA O PORQUÊ.
    braço         det  início banda  FP/mês  Δcarga [IC 97,5%]    ΔFP [IC 97,5%]       carga cai em
    referência    6,5   6,0   4,5   0,861
    P1 θ=2,0      7,0   5,5   3,5   0,775   -2,8 [-13,7; +8,0]   -0,011 [-0,17; +0,12]   4/8   A, B1, B2
    θ=1,5 (sens.) 6,5   5,5   3,5   0,818   +1,9 [-3,1; +8,5]    -0,011                  5/8
    θ=3,0 (sens.) 6,0   4,5   3,0   0,818   -7,3 [-37; +36]      +0,000                  6/8
  · A composição 1 perde uma detecção (8 -> 7); início 6,0 -> 5,5 e banda 4,5 -> 3,5. Secundário: detecção,
    início e banda não ok, e o FP NÃO cai (-0,011), ao contrário do que a curva estática dos 54 episódios
    distintos prometia (11 de 25 FP removidos).
  · MECANISMO: somem só 11 FP nas 8 composições, mas APARECEM 17 episódios FP/NEUTRO novos (1.191 h, 419 só na
    composição 1) contra 1.019 h que somem. Um episódio fraco removido deixa de abrir o refratário de 72 h, e
    os votos seguintes, que ele bloqueava, viram episódios. A conta por episódio ignorava o acoplamento.

PRÉ-REGISTRO DO P1b (03/10/2026), ANTES DE RODAR -- o portão DEPOIS do refratário.
  DE ONDE VEM: da falha do P1 acima. Mesmo θ = 2,0 (NÃO reajustado), mesma força, mesma ideia; muda ONDE se
  aplica. Os episódios do alarme final (pós escalada, refratário e duração mínima) são tratados como acima:
  sem F >= θ, o episódio inteiro deixa de ser anunciado; senão o início é aparado até F >= θ, e o que sobra
  só vale se ainda cumprir a duração mínima (120 min; 60 se F > 20). O episódio fraco CONTINUA bloqueando o
  refratário como hoje; só deixa de ser anunciado. Logo o conjunto de alarmes é SUBCONJUNTO da referência:
  não nascem episódios novos (verificado por assert), e a contagem por episódio deixa de ser enganosa.
  DECISÃO: a mesma, em dois critérios, com IC de 98,75% (a família agora tem 4 comparações: P1 e P1b, cada
  um com dois critérios). θ = 1,5 e 3,0 só como sensibilidade. θ = 0 reproduz a referência bit a bit.
  EXPECTATIVA: o FP cai de verdade (a curva estática passa a valer): ~0,86 -> ~0,5-0,6. O risco é o de
  sempre, a detecção estrita: o TP de força 1,2 pode ser o único alarme de um trip em alguma composição
  (a composição 1 já perdeu uma detecção no P1). Chance de passar o primário ~15%; o secundário, ~25%.

Uso:  PYTHONPATH=. python portao_de_forca.py confere   # θ = 0 reproduz a referência
      PYTHONPATH=. python portao_de_forca.py pos       # o P1b (portão depois do refratário)
      PYTHONPATH=. python portao_de_forca.py           # o teste
"""
from __future__ import annotations
import io, contextlib, sys
import numpy as np, pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    import regua_fp as R
    import avalia as AV
    import bootstrap_regua as BR
    import troca_e_pi0319 as TP

THETA = 2.0
SENSIB = (1.5, 3.0)
NIVEL = 0.975
_CACHE: dict[int, dict] = {}


def saida(d: int) -> dict:
    """O detector de referência por composição, uma vez; o portão se aplica sobre o voto e a força dele."""
    if d not in _CACHE:
        _CACHE[d] = R.detector(*R.sinais(d))
    return _CACHE[d]


def aparar(voto: pd.Series, F: pd.Series, theta: float) -> pd.Series:
    """Apaga o voto do início de cada episódio até F >= theta; sem F >= theta, o episódio inteiro."""
    if theta <= 0:
        return voto
    v = voto.to_numpy().copy()
    f = np.nan_to_num(F.to_numpy(), nan=0.0)
    idx = voto.index
    for a, b in AV.episodios(voto):
        i0, i1 = idx.get_loc(a), idx.get_loc(b) + 1
        ok = v[i0:i1] & (f[i0:i1] >= theta)
        if not ok.any():
            v[i0:i1] = False
        else:
            v[i0:i0 + int(np.argmax(ok))] = False
    return pd.Series(v, index=idx)


def com_portao(d: int, theta: float) -> pd.Series:
    o = saida(d)
    return R._pos(aparar(o["voto"], o["F"], theta), o["F"])


def aparar_pos(fin: pd.Series, F: pd.Series, theta: float) -> pd.Series:
    """P1b: o mesmo portão, sobre o alarme FINAL (depois do refratário); o refratário não muda."""
    if theta <= 0:
        return fin
    v = fin.to_numpy().copy()
    f = np.nan_to_num(F.to_numpy(), nan=0.0)
    idx = fin.index
    for a, b in AV.episodios(fin):
        i0, i1 = idx.get_loc(a), idx.get_loc(b) + 1
        ok = v[i0:i1] & (f[i0:i1] >= theta)
        if not ok.any():
            v[i0:i1] = False
            continue
        j = i0 + int(np.argmax(ok))
        forte = f[i0:i1].max() > R.DB.ESC_ABS
        if (i1 - j) * 2 < (R.DB.ESC_DUR if forte else R.DB.DUR_MIN):
            v[i0:i1] = False
        else:
            v[i0:j] = False
    return pd.Series(v, index=idx)


def com_portao_pos(d: int, theta: float) -> pd.Series:
    o = saida(d)
    return aparar_pos(o["fin"], o["F"], theta)


def confere() -> None:
    for d in R.DIAS:
        ok = bool((com_portao(d, 0.0) == saida(d)["fin"]).all())
        sub = not bool((com_portao_pos(d, THETA) & ~saida(d)["fin"]).any())
        print(f"  composição {d:2d}: θ = 0 reproduz a referência bit a bit: {ok} | P1b é subconjunto da referência: {sub}", flush=True)
        assert ok and bool((com_portao_pos(d, 0.0) == saida(d)["fin"]).all()) and sub


def mecanismo(theta: float, fn=None) -> None:
    fn = fn or com_portao
    """Quem some, quem nasce tarde, e se um FP curto virou NEUTRO longo."""
    cab = ["dia", "somem FP", "somem TP", "FP atrasados", "TP atrasados", "atraso TP (h, med)",
           "h FP+NEUTRO que somem", "episódios NEUTRO/FP novos", "h dos novos"]
    L = []
    for d in R.DIAS:
        ref = R.mede(saida(d)["fin"])["cls"]; var = R.mede(fn(d, theta))["cls"]
        so = lambda ep, k: [(a, b) for a, b, kk, _ in ep if kk == k]
        r = dict(dia=d)
        somem = {k: 0 for k in ("FP", "TP")}; atrasados = {k: 0 for k in ("FP", "TP")}; atraso_tp = []; h_somem = 0.0
        for a, b, k, _ in ref:
            sob = [(x, y) for x, y, _, _ in var if x <= b and y >= a]
            if not sob:
                if k in somem: somem[k] += 1
                if k in ("FP", "NEUTRO"): h_somem += (b - a).total_seconds() / 3600
            else:
                x0 = min(x for x, _ in sob)
                if x0 > a + pd.Timedelta(minutes=2):
                    if k in atrasados: atrasados[k] += 1
                    if k == "TP": atraso_tp.append((x0 - a).total_seconds() / 3600)
        novos = [(a, b) for a, b, k, _ in var if k in ("FP", "NEUTRO") and not any(x <= b and y >= a for x, y, _, _ in ref)]
        L.append([d, somem["FP"], somem["TP"], atrasados["FP"], atrasados["TP"],
                  round(float(np.median(atraso_tp)), 1) if atraso_tp else 0.0, round(h_somem), len(novos),
                  round(sum((b - a).total_seconds() / 3600 for a, b in novos))])
    print(pd.DataFrame(L, columns=cab).to_string(index=False), flush=True)


def main(pos: bool = False):
    fn, nome, nivel = (com_portao_pos, "P1b", 0.9875) if pos else (com_portao, "P1", NIVEL)
    ref = BR.braco("referência", lambda d: saida(d)["fin"], R.DIAS)
    t = ref.tabela
    print(f"referência     det {t.det.median()} início {t.inicio.median()} banda {t.banda.median()}"
          f" FP {t.fp_mes.median():.3f} carga {t.carga_mes.mean():.1f}", flush=True)
    br = BR.braco(f"{nome} θ={THETA}", lambda d: fn(d, THETA), R.DIAS)
    x = BR.decide(ref, br, pareado=True, nivel=nivel)
    TP.linha(f"{nome} θ={THETA}", x, ref, br, nivel)
    v, r = br.tabela, ref.tabela
    det_ok = bool((v.det.values >= r.det.values).all() and v.inicio.median() >= r.inicio.median()
                  and v.banda.median() >= r.banda.median())
    sec = bool(det_ok and x["fp_hi"] < 0 and v.carga_mes.mean() <= r.carga_mes.mean())
    print(f"\n{nome} -- PRIMÁRIO (regra de sempre): {x['veredito']}")
    print(f"SECUNDÁRIO (contagem): detecção/início/banda {'ok' if det_ok else 'NÃO ok'} | ΔFP {x['d_fp']:+.3f} "
          f"[{x['fp_lo']:+.3f}; {x['fp_hi']:+.3f}] | carga {r.carga_mes.mean():.1f} -> {v.carga_mes.mean():.1f} "
          f"=> {'GANHO-CONTAGEM' if sec else 'reprovado'}", flush=True)
    print(f"\nMECANISMO (θ = {THETA}):", flush=True)
    mecanismo(THETA, fn)
    print("\nSENSIBILIDADE, sem veredito:", flush=True)
    for th in SENSIB:
        b2 = BR.braco(f"θ={th}", lambda d, th=th: fn(d, th), R.DIAS)
        TP.linha(f"θ={th}", BR.decide(ref, b2, pareado=True, nivel=nivel), ref, b2, nivel)
    pd.DataFrame([dict(theta=THETA, **x, secundario=sec)]).to_csv(R.CACHE / f"portao_de_forca_{nome}.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "confere":
        confere()
    else:
        main(pos=len(sys.argv) > 1 and sys.argv[1] == "pos")
