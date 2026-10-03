#!/usr/bin/env python3
"""Gestão de alarme com reanúncio por piora: o episódio "conhecido" que piora volta a ser anunciado.

PRÉ-REGISTRADO EM 03/10/2026, ANTES DE RODAR (commit deste arquivo sem resultado).

DE ONDE VEM. `gestao_alarme.py` (29/09) mediu a gestão de alarme na apresentação: com N = 24 h, a carga
ativa cai 81% (130,5 -> 24,5 h/mês), os nascimentos e as 2,07 notificações/mês não mudam, e o custo é
~1 trip sem alarme ATIVO nas 48 h (o 17/03/2025, nascido 195 h antes). O único reanúncio que existe é a
escalada por idade (F > 20 depois de 96 h). Aqui se pergunta se um reanúncio por PIORA recupera esse trip
sem encher o operador de notificações. NÃO TOCA O DETECTOR: é decisão de projeto da integração.

AS TRÊS APRESENTAÇÕES (todas sobre o alarme final de `regua_fp.detector`, que não muda):
  atual   o episódio fica ativo do início ao fim.
  R0      ativo nas primeiras N = 24 h; depois "condição conhecida". Reanúncio só pela escalada por idade
          (que já cria um nascimento novo no detector).
  R1      R0 mais REANÚNCIO POR PIORA: depois das N h, o episódio volta a ficar ativo por mais N h quando a
          força F (a da escalada, max_c EWMA/(BASE x KH)) chega a >= K x F_ref, com F_ref = a força máxima
          da última janela ativa, e F >= F_MIN em termos absolutos. K = 3 e F_MIN = 3, fixados agora; a
          cada reanúncio, F_ref passa a ser a força máxima da nova janela ativa.
N, K e F_MIN são parâmetros de PROJETO, não ajustados nos 8 eventos: um valor só de cada, sem varredura.

MEDIDAS, por composição (mediana das 8; as 8 composições de sempre):
  carga ativa      h/mês de FP + NEUTRO com o alarme "ativo" (a carga que o operador vê).
  notificações     por mês: nascimentos + reanúncios; e só as de episódios FP + NEUTRO (as falsas).
  de pé ativo      trips com alarme ATIVO em algum instante de [T-48 h, T); e quantos pares composição x trip
                   a apresentação PERDE em relação ao atual.
CRITÉRIOS DE ACEITE PARA A INTEGRAÇÃO (de engenharia, escritos agora, não estatísticos):
  (a) a carga ativa de R1 cai >= 60% em relação ao atual (mediana);
  (b) R1 perde estritamente MENOS pares composição x trip de pé ativo que R0, ou nenhum;
  (c) as notificações FP + NEUTRO por mês de R1 ficam <= 1,25 x as do atual.
  R1 é recomendada se (a), (b) e (c); se só (b) falhar, a troca é reportada para a engenharia escolher.
CHECAGEM DE MECANISMO: quantos reanúncios R1 faz, em episódios de que classe, e quais trips R1 recupera.

EXPECTATIVA REGISTRADA. R0 reproduz os números de 29/09. R1 recupera o 17/03/2025 (ou parte) se a força cresce
perto do trip, ao custo de reanúncios em episódios FP longos. Chance de passar (a), (b) e (c): ~40%.
RESSALVA: nada aqui reduz o número de FP (as notificações falsas seguem ~0,86/mês); só as horas.

Uso:  PYTHONPATH=. python gestao_alarme_reanuncio.py
"""
from __future__ import annotations
import io, contextlib
import numpy as np, pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    import regua_fp as R

N_H, K, F_MIN = 24.0, 3.0, 3.0
PAS_H = 2 / 60


def ativo_r0(fin: pd.Series, n_h: float = N_H) -> pd.Series:
    ativo = pd.Series(False, index=fin.index)
    for a, b in R.AV.episodios(fin):
        ativo.loc[a:min(b, a + pd.Timedelta(hours=n_h))] = True
    return ativo & fin


def ativo_r1(fin: pd.Series, F: pd.Series, n_h: float = N_H, k: float = K, fmin: float = F_MIN):
    """(ativo, reanuncios por episódio): R0 mais o reanúncio por piora."""
    idx = fin.index
    v = np.zeros(len(idx), bool)
    f = np.nan_to_num(F.to_numpy(), nan=0.0)
    n = int(n_h * 30)
    rean = []
    for a, b in R.AV.episodios(fin):
        i0, i1 = idx.get_loc(a), idx.get_loc(b) + 1
        fim_jan = min(i1, i0 + n)
        v[i0:fim_jan] = True
        fref, i, cont = f[i0:fim_jan].max(), fim_jan, 0
        while i < i1:
            alvo = np.flatnonzero(f[i:i1] >= max(fmin, k * fref))
            if not len(alvo):
                break
            j = i + int(alvo[0]); fim_jan = min(i1, j + n)
            v[j:fim_jan] = True
            fref = f[j:fim_jan].max(); i = fim_jan; cont += 1
        rean.append(cont)
    return pd.Series(v, index=idx) & fin, rean


def mede(fin: pd.Series, ativo: pd.Series, rean: list[int] | None) -> dict:
    eps = R.AV.episodios(fin)
    cls = R.DB.classifica_regra_c(eps, R.PARADAS)
    mes = float(R.mask.sum()) * 2 / 60 / 730
    h_ativa = lambda ks: sum(float(ativo.loc[a:b].sum()) * PAS_H for a, b, kk, _ in cls if kk in ks)
    rean = rean or [0] * len(eps)
    notif = lambda ks: sum(1 + r for (a, b, kk, _), r in zip(cls, rean) if kk in ks)
    pares = [bool(ativo.loc[t - R.JAN:t - pd.Timedelta(minutes=2)].any()) for t in R.alvo]
    return dict(carga_ativa=h_ativa(("FP", "NEUTRO")) / mes, notif_mes=notif(("FP", "NEUTRO", "TP")) / mes,
                notif_falsas_mes=notif(("FP", "NEUTRO")) / mes, de_pe_ativo=sum(pares), pares=pares,
                reanuncios=sum(rean), rean_fp=sum(r for (a, b, kk, _), r in zip(cls, rean) if kk in ("FP", "NEUTRO")),
                rean_tp=sum(r for (a, b, kk, _), r in zip(cls, rean) if kk == "TP"))


def main():
    pd.set_option("display.width", 200)
    L, perdas = [], {"R0": [], "R1": []}
    for d in R.DIAS:
        o = R.detector(*R.sinais(d)); fin, F = o["fin"], o["F"]
        atual = mede(fin, fin, None)
        r0 = mede(fin, ativo_r0(fin), None)
        a1, rean = ativo_r1(fin, F)
        r1 = mede(fin, a1, rean)
        for nome, m in (("atual", atual), ("R0", r0), ("R1", r1)):
            L.append(dict(dia=d, apres=nome, **{k: v for k, v in m.items() if k != "pares"}))
        for nome, m in (("R0", r0), ("R1", r1)):
            perdas[nome] += [(d, R.alvo[i].strftime("%Y-%m-%d")) for i, (x, y) in enumerate(zip(atual["pares"], m["pares"])) if x and not y]
        print(f"composição {d:2d} ok", flush=True)
    T = pd.DataFrame(L); T.to_csv(R.CACHE / "gestao_alarme_reanuncio.csv", index=False)
    print("\nPOR COMPOSIÇÃO: carga ativa (h/mês) | notificações falsas/mês | de pé ativo")
    print(T.pivot(index="dia", columns="apres", values=["carga_ativa", "notif_falsas_mes", "de_pe_ativo"]).round(2).to_string())
    g = T.groupby("apres")[["carga_ativa", "notif_mes", "notif_falsas_mes", "de_pe_ativo", "reanuncios", "rean_fp", "rean_tp"]].median().round(2)
    print("\nMEDIANA DAS 8 COMPOSIÇÕES"); print(g.loc[["atual", "R0", "R1"]].to_string())
    a, r1m = g.loc["atual"], g.loc["R1"]
    qa = 100 * (1 - r1m.carga_ativa / a.carga_ativa)
    print(f"\n(a) carga ativa de R1: {a.carga_ativa:.1f} -> {r1m.carga_ativa:.1f} h/mês = -{qa:.0f}%   (critério: >= 60%)  {'OK' if qa >= 60 else 'NÃO'}")
    print(f"(b) pares composição x trip que perdem o ALARME ATIVO: R0 {len(perdas['R0'])} | R1 {len(perdas['R1'])}   (critério: R1 < R0, ou 0)  "
          f"{'OK' if len(perdas['R1']) < len(perdas['R0']) or not perdas['R1'] else 'NÃO'}")
    print(f"(c) notificações falsas/mês: atual {a.notif_falsas_mes:.2f} | R1 {r1m.notif_falsas_mes:.2f} = {r1m.notif_falsas_mes / a.notif_falsas_mes:.2f}x   (critério: <= 1,25x)  "
          f"{'OK' if r1m.notif_falsas_mes <= 1.25 * a.notif_falsas_mes else 'NÃO'}")
    for nome in ("R0", "R1"):
        c = pd.Series([t for _, t in perdas[nome]]).value_counts()
        print(f"  trips perdidos em {nome}: " + (", ".join(f"{t} ({n} composições)" for t, n in c.items()) or "nenhum"))


if __name__ == "__main__":
    main()
