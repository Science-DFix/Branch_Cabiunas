#!/usr/bin/env python3
"""Separar os dois regimes: o detector só fala depois de H horas da partida.

POR QUÊ (`relogio_partida.py`). 5 dos 8 trips vêm até 31 h após uma partida (risco
~3,7x), e neles o detector é PIOR que um relógio (3,5/5 contra 5/5). O valor
físico dele está nas 3 falhas de regime longo (3/3). E é da borda pós-partida
que nascem 6 dos 7 NEUTRO e 2 dos 4 FP.

A PROPOSTA. Dois produtos em vez de um:
  · o DETECTOR, armado só depois de H horas de operação após cada partida --
    implementado como blackout de H horas em vez de 6, dentro do detector;
  · um AVISO DE PROCEDIMENTO, "janela de risco pós-partida", em toda partida.
    Não é previsão: dispara em toda partida. É o que a razão de risco de 3,7x
    justifica pedir à operação.

O QUE ISTO NÃO É. Não é reduzir falso positivo por mérito do detector. É
delimitar o escopo: o detector passa a responder oficialmente por 3 dos 8 trips
observados, e os outros 5 ficam com o procedimento. Com 3 eventos no regime longo,
a base é finíssima.

CRITÉRIO, ESCRITO ANTES DE RODAR. A regra geral da régua proíbe perder detecção,
mas aqui a perda dos trips pós-partida é INTENCIONAL. O critério é sobre o que o
detector continua responsável:

  H primário = 24 h. Aceito se, nas 8 composições:
    1. detecção do regime longo (de pé) nunca abaixo da referência em nenhuma
       composição, e início do regime longo não cai na mediana;
    2. carga cai em >= 7 de 8, e o FP/mês cai na mediana;
    3. platô: pelo menos um de H = 32 h ou 48 h também passa.
  Relatado sempre: trips pós-partida perdidos pelo detector, cobertura do aviso e
  quantos avisos por mês ele emite.

O denominador do FP/mês é o tempo de operação da REFERÊNCIA (por mês de máquina).
Com blackout maior o detector enxerga menos horas; dividir pelas horas dele
inflaria a taxa só por ter olhado menos.

RESULTADO (29/09/2026) -- REPROVADO nos três H.

    (mediana das 8)   longo det   FP/mês   carga   carga cai   longo cai em
    referência          3/3       0,861    130,5      --          --
    blackout 24 h       3/3       0,603    120,9     5/8        2 composições
    blackout 32 h       3/3       0,560    117,9     5/8        2 composições
    blackout 48 h       3/3       0,603    147,5     3/8        3 composições

O FP/mês cai, a carga quase não. No dia 1 o blackout de 24 h PIORA o FP (0,344 ->
0,603). O motivo: com qualquer blackout, metade dos episódios nasce no instante
em que a máscara abre -- 11 de 21 com 6 h, 7 a 8 com 24, 32 e 48 h. A BORDA ANDA
JUNTO COM O BLACKOUT, porque vb e sp estão acesos a maior parte do tempo normal e
o detector, ao começar a olhar, já encontra o voto ligado. O aviso de
procedimento emitiria 9,6 avisos por mês.

Uso:  PYTHONPATH=. python regimes.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import regua_fp as R
import pericia_fp_atual as PF

HS = (24.0, 32.0, 48.0)
POS_H = 32.0


def separa_trips():
    hrel = PF.horas_desde_religamento()
    pos = [t for t in R.alvo if hrel.iloc[R.idx.searchsorted(t) - 1] <= POS_H]
    lon = [t for t in R.alvo if t not in pos]
    return pos, lon


def mede_regime(fin, pos, lon) -> dict:
    eps = R.AV.episodios(fin)
    det = lambda ts: sum(1 for t in ts if fin.loc[t - R.JAN:t - pd.Timedelta(minutes=2)].any())
    ini = lambda ts: sum(1 for t in ts if any(t - R.JAN <= a <= t for a, _ in eps))
    ban = lambda ts: sum(1 for t in ts
                         if any(t - R.JAN <= a <= t - pd.Timedelta(hours=R.DB.TMIN_BANDA)
                                for a, _ in eps))
    m = R.mede(fin); m.pop("cls")
    return dict(det_pos=det(pos), ini_pos=ini(pos), det_longo=det(lon),
                ini_longo=ini(lon), banda_longo=ban(lon), **m)


def aviso(pos) -> dict:
    """O procedimento: janela de risco em toda partida."""
    part = PF.PP.part.to_numpy() & np.asarray(R.sel)
    meses = float(R.mask.sum()) * 2 / 60 / 730
    return dict(avisos_mes=part.sum() / meses, cobre=len(pos))


def main():
    pos, lon = separa_trips()
    print(f"pós-partida: {[t.strftime('%m-%d') for t in pos]}   "
          f"regime longo: {[t.strftime('%Y-%m-%d') for t in lon]}\n")
    ref = pd.DataFrame([dict(dia=d, **mede_regime(R.detector(*R.sinais(d))["fin"], pos, lon))
                        for d in R.DIAS]).set_index("dia")
    res = {}
    for H in HS:
        var = pd.DataFrame([dict(dia=d, **mede_regime(
            R.detector(*R.sinais(d), blackout_h=H)["fin"], pos, lon)) for d in R.DIAS]).set_index("dia")
        c1 = bool((var.det_longo >= ref.det_longo).all()
                  and var.ini_longo.median() >= ref.ini_longo.median())
        c2 = bool(int((var.carga_mes < ref.carga_mes - 1e-9).sum()) >= 7
                  and var.fp_mes.median() < ref.fp_mes.median())
        res[H] = dict(var=var, c1=c1, c2=c2, passa=c1 and c2)

    cols = ["det_pos", "ini_pos", "det_longo", "ini_longo", "banda_longo",
            "n_fp", "fp_mes", "h_fp_mes", "n_neutro", "h_neutro_mes", "carga_mes"]
    pd.set_option("display.width", 190)
    print("REFERÊNCIA (blackout 6 h)")
    print(ref[cols].round(3).to_string())
    for H in HS:
        r = res[H]
        print(f"\n{'=' * 100}\nBLACKOUT {H:.0f} h — detector só fala depois de {H:.0f} h da partida\n{'=' * 100}")
        print(r["var"][cols].round(3).to_string())
        v = r["var"]
        print(f"\n  mediana: regime longo det {v.det_longo.median():.1f}/3 (ref {ref.det_longo.median():.1f})"
              f"  início {v.ini_longo.median():.1f}/3 (ref {ref.ini_longo.median():.1f})"
              f"  banda {v.banda_longo.median():.1f}/3 (ref {ref.banda_longo.median():.1f})")
        print(f"           FP/mês {v.fp_mes.median():.3f} (ref {ref.fp_mes.median():.3f})"
              f"  carga {v.carga_mes.median():.1f} (ref {ref.carga_mes.median():.1f})"
              f"  | carga cai em {int((v.carga_mes < ref.carga_mes - 1e-9).sum())}/8"
              f"  | FP cai em {int((v.fp_mes < ref.fp_mes - 1e-9).sum())}/8")
        print(f"  critério 1 (regime longo intacto): {r['c1']}   critério 2 (custo cai): {r['c2']}"
              f"   -> passa: {r['passa']}")
    viz = res[32.0]["passa"] or res[48.0]["passa"]
    aceito = res[24.0]["passa"] and viz
    a = aviso(pos)
    print(f"\n{'=' * 100}\nVEREDITO (H = 24 h, com platô em 32 ou 48 h): "
          f"{'ACEITO' if aceito else 'REPROVADO'}")
    print(f"  aviso de procedimento: {a['avisos_mes']:.1f} avisos por mês (toda partida), "
          f"cobre os {a['cobre']} trips pós-partida por construção")
    pd.concat({f"{H:.0f}h": res[H]["var"] for H in HS} | {"ref": ref}).to_csv(R.CACHE / "regimes.csv")


if __name__ == "__main__":
    main()
