#!/usr/bin/env python3
"""Retreino só com a permissão do operador: como fica o resultado?

HOJE. O bundle é treinado no dia 1 e publicado na hora, se passar no portão técnico
(RESPOSTAS_ENGENHARIA, pergunta 15). Com aprovação humana, três coisas mudam, e cada
uma mexe numa alavanca que já foi medida:

  ATRASO      o candidato fica pronto no dia 1 e só entra quando alguém aprova. O
              bundle antigo serve por mais L dias e o novo já nasce com L dias de
              idade -- é uma guarda. A guarda de 21 dias subiu a detecção E a carga
              (`retreino_semanal.py`).
  MÊS PULADO  o operador não aprova (férias, parada, dúvida). A cadência fica
              irregular -- 2 e 3 meses já pioraram a carga em +36% a +44%
              (`cadencia_retreino.py`) -- e o bundle VENCE 62 dias depois do fim do
              baseline (`checa_validade`): a inferência para e o detector fica cego.
  VETO        o operador segura o retreino quando o detector está alarmando -- a
              regra que um operador experiente adotaria sozinho ("não mexo na
              referência com a máquina em alarme"). Protege contra absorver uma
              degradação. O risco é o contrário: se quem alarma é o próprio bundle
              velho (nov/2025, PDI_0301 rezerado, p aceso 96% do mês), o veto o mantém
              no ar, e ele segue alarmando.

E dois jeitos de montar o candidato:
  (a) pronto    treinado no dia agendado, com o dado até ali; a aprovação só publica.
  (b) na hora   treinado quando o operador libera, com o dado até a liberação. Com
                atraso FIXO isto é só outro dia do mês, e a régua já cobre (mediana
                6,5 / 0,861 FP/mês, dias 1 a 25); por isso (b) só entra onde o atraso
                varia.

OS CENÁRIOS, todos nas 8 composições da régua (dias 1, 4, ..., 25), mesmos limiares:
  automático         o de hoje (a régua)
  atraso L           (a), L = 3, 7, 14 e 21 dias
  irregular q        cada mês aprovado com probabilidade q (0,75 e 0,50), atraso
                     sorteado de 0 a 14 dias; (a) e (b) com o MESMO sorteio;
                     5 sementes por composição = 40 cenários por casela
  veto 72 h          (a), publica só depois de 72 h sem alarme; se o próximo
                     candidato chega antes, substitui o pendente
  veto 7 d           (b), treina e publica só depois de 7 dias sem alarme
O veto é SEQUENCIAL: o alarme que o operador vê é o do bundle que está no ar, que
depende das aprovações anteriores. Bundle vencido = sem alarme na tela (fica cego).

COMO O CEGO ENTRA NA CONTA. O tempo cego tem a máquina rodando e o detector mudo:
não gera FP -- o custo por mês CAI --, e perde o trip que cair nele. Por isso os dias
cegos saem numa coluna própria: custo baixo com dia cego não é ganho.

CRITÉRIO, ESCRITO ANTES DE RODAR. Uma política é TOLERÁVEL se, na mediana dos
cenários: detecção, início e banda não caem mais que 0,5 abaixo do automático (meia
detecção com 8 eventos é ruído); FP/mês e carga não sobem mais que 10%; e nenhum
cenário tem dia cego. Não é um teste de adoção -- é medir o preço de uma restrição.

PREVISÃO, ESCRITA ANTES. Atraso age como guarda: sobe detecção e carga, mais com L
maior. Mês pulado: carga sobe, e com q = 0,50 aparecem dias cegos (dois meses
seguidos sem aprovação vencem o bundle). Veto: sem previsão firme.

RESULTADO (30/09/2026). Mediana dos cenários [mínimo-máximo]; dias cegos = dias de
máquina rodando com o bundle vencido, num histórico de ~353 dias de operação.

    política                 det   início  banda  FP/mês  carga  dias cegos   veredito
    automático (hoje)        6,5    6,0    4,5    0,861   130,5      0
    atraso  3 d              6,5    6,0    4,5    0,775   141,2      0        TOLERÁVEL
    atraso  7 d              7,0    6,0    4,5    0,775   160,0      0        não (carga +23%)
    atraso 14 d              7,0    6,5    4,5    0,818   165,3      0        não (+27%)
    atraso 21 d              7,5    5,5    4,5    0,904   214,1      0        não (+64%)
    1 em 4 meses sem aprov.  6,0    5,0    4,0    0,861   164,6     28,5      não (cego em 39/40)
      ... treino na hora     6,0    5,0    4,0    0,861   140,3     18,5      não (cego em 33/40)
    1 em 2 meses sem aprov.  5,0    4,0    3,0    0,689   132,3     98,7      não (cego em 40/40)
      ... treino na hora     5,5    4,0    3,0    0,732   121,5     80,0      não
    veto 72 h sem alarme     6,5    5,5    4,5    0,861   143,9    0 [0-3]    não (cego em 3/8)
    veto 7 d, treino na hora 7,0    5,5    4,0    0,861   121,4    0 [0-6,7]  não (cego em 3/8)

  · ATRASO é guarda, como previsto: a detecção sobe meio evento e a carga sobe muito.
    Até 3 dias não custa nada.
  · MÊS PULADO é o problema. A validade de 62 dias é exatamente dois meses: um mês
    sem aprovação mais QUALQUER atraso no seguinte vence o bundle. Com 1 em 4 meses
    sem aprovação, 39 de 40 cenários ficam cegos em algum momento (mediana 28,5 dias,
    8% da operação) e a detecção cai de 6,5 para 6,0, o início de 6 para 5.
    DECOMPOSIÇÃO (decidida depois): sem vencimento, a detecção volta a 7,0, mas a
    carga vai a 182 (+40%) e 198 (+52%) -- o bundle velho alarma. O vencimento troca
    carga por cegueira; nenhum dos dois é bom. O remédio é não pular mês.
  · TREINAR NA HORA DA APROVAÇÃO é melhor que deixar o candidato pronto quando a
    aprovação é irregular: menos cego (18,5 contra 28,5 dias) e menos carga (140
    contra 165), porque o relógio da validade e a idade da referência recomeçam ali.
  · O VETO não dá a proteção que a intuição promete: no máximo meio evento a mais
    (ruído). E segura o retreino por até 27 dias, várias vezes por ano (set/2025,
    fev/2026, abr/2026), vencendo o bundle em 3 de 8 composições. O de 7 dias é o
    mais próximo do tolerável (carga -7%), reprovado só pelos dias cegos.
  · Nenhuma política com aprovação MELHORA o automático com segurança. A aprovação é
    viável se for rápida e não pular mês.

Uso:  PYTHONPATH=. python aprovacao_operador.py [atraso|irregular|veto|resumo]
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
import regua_fp as R
import drift_nos_dados as DN

DC, C, DF, IX, STABLE, FIT = DN.DC, DN.C, DN.DF, DN.IX, DN.STABLE, DN.FIT
VALIDADE = pd.Timedelta(days=63)   # checa_validade: vence quando (agora - fim).days > 62
DIA = pd.Timedelta(days=1)
FIM = IX[-1] + pd.Timedelta("2min")
MASK = R.mask.to_numpy().astype(bool)
_BUNDLE: dict = {}


def cortes(dia: int) -> list[pd.Timestamp]:
    """os mesmos cortes de `drift_composicao.walkforward_dia`."""
    base = pd.date_range(IX[0].normalize().replace(day=1), IX[-1], freq="MS", tz="UTC")
    cs = [m + pd.Timedelta(days=dia - 1) for m in base]
    return [c for c in cs if IX[0] < c < IX[-1]]


def bundle(ref_fim: pd.Timestamp):
    """o bundle treinado com os FIT pontos estáveis antes de `ref_fim` (memoizado)."""
    if ref_fim not in _BUNDLE:
        fit = DF.loc[STABLE & (IX < ref_fim), C.SENSOR_TAGS].dropna().tail(FIT)
        if len(fit) < FIT // 4:
            _BUNDLE[ref_fim] = None
        else:
            b = DC.DET._spread_mancal(fit)
            _BUNDLE[ref_fim] = (DC.ScorerMax().fit(fit[C.TEMPERATURE_TAGS]),
                                DC.ScorerMax().fit(fit[C.PRESSURE_TAGS]),
                                float(b.median()), float((b - b.median()).abs().median() * 1.4826))
    return _BUNDLE[ref_fim]


class Serie:
    """os sinais do plano de publicação, montados em ordem: cada publicação vale
    daquele instante até o fim, até a próxima sobrescrever."""

    def __init__(self, validade: pd.Timedelta = VALIDADE):
        n = len(IX)
        self.validade = validade
        self.t, self.p = np.full(n, np.nan), np.full(n, np.nan)
        self.ms, self.ds = np.full(n, np.nan), np.full(n, np.nan)
        self.cego = np.zeros(n, bool); self.idade = np.full(n, np.nan)
        self.plano: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def publica(self, pub: pd.Timestamp, ref_fim: pd.Timestamp):
        self.plano.append((pub, ref_fim))
        s = IX >= pub
        bd = bundle(ref_fim)
        if bd is None:
            self.t[s] = self.p[s] = self.ms[s] = self.ds[s] = np.nan
            self.cego[s] = False; self.idade[s] = np.nan
            return
        w = DF.loc[s]
        self.t[s] = bd[0].score(w[C.TEMPERATURE_TAGS])["pca_recon"].to_numpy()
        self.p[s] = bd[1].score(w[C.PRESSURE_TAGS])["pca_recon"].to_numpy()
        self.ms[s], self.ds[s] = bd[2], bd[3]
        self.cego[s] = IX[s] >= ref_fim + self.validade
        self.idade[s] = (IX[s] - ref_fim).total_seconds() / 86400

    def fin(self) -> pd.Series:
        f = R.detector(self.t, self.p, self.ms, self.ds)["fin"]
        return f & ~pd.Series(self.cego, index=f.index)

    def mede(self) -> dict:
        m = R.mede(self.fin()); m.pop("cls")
        m["dias_cegos"] = float((self.cego & MASK).sum() * 2 / 60 / 24)
        m["idade_media"] = float(np.nanmean(self.idade[MASK]))
        m["publicacoes"] = len(self.plano)
        return m


def de_plano(plano, validade: pd.Timedelta = VALIDADE) -> Serie:
    S = Serie(validade)
    for pub, ref in sorted(plano):
        S.publica(pub, ref)
    return S


def confere(dia: int = 1) -> bool:
    """atraso 0 tem de reproduzir a régua."""
    S = de_plano([(c, c) for c in cortes(dia)])
    t, p, ms, ds = R.sinais(dia)
    ok = all(np.allclose(a, b, equal_nan=True, rtol=1e-5, atol=1e-6)
             for a, b in ((S.t, t), (S.p, p), (S.ms, ms), (S.ds, ds)))
    return ok and bool((R.detector(S.t, S.p, S.ms, S.ds)["fin"] == R.detector(t, p, ms, ds)["fin"]).all())


# ══════════════════════════════════════════════════ as políticas
def atraso(dia: int, L: int) -> dict:
    return de_plano([(c + L * DIA, c) for c in cortes(dia)]).mede()


def irregular(dia: int, q: float, semente: int, modo: str, validade: pd.Timedelta = VALIDADE) -> dict:
    rng = np.random.default_rng(1000 * semente + dia)
    plano = []
    for c in cortes(dia):
        aprova, lag = rng.random() < q, int(rng.integers(0, 15))
        if aprova:
            pub = c + lag * DIA
            plano.append((pub, c if modo == "pronto" else pub))
    return de_plano(plano, validade).mede()


def veto(dia: int, janela: pd.Timedelta, modo: str) -> dict:
    """publica só depois de `janela` sem alarme na tela, olhando dia a dia."""
    cs = cortes(dia)
    S = Serie()
    ultimo = None
    for i, c in enumerate(cs):
        prox = cs[i + 1] if i + 1 < len(cs) else FIM
        if ultimo is not None and ultimo >= c:          # (b): ainda na espera anterior
            continue
        fin = S.fin()
        on = fin.to_numpy()
        # instante do último alarme até cada ponto: basta achar a 1ª tentativa limpa
        tent = c
        while tent < FIM:
            sel = (IX > tent - janela) & (IX <= tent)
            if not on[sel].any():
                break
            tent += DIA
        if tent >= FIM or (modo == "pronto" and tent >= prox):
            continue                                     # candidato caducou / fim do dado
        S.publica(tent, c if modo == "pronto" else tent)
        ultimo = tent
    m = S.mede()
    esp = [(pub - max(c for c in cs if c <= pub)).total_seconds() / 86400 for pub, _ in S.plano]
    m["espera_mediana_d"], m["espera_max_d"] = float(np.median(esp)), float(np.max(esp))
    m["cortes"] = len(cs)
    m["esperas_longas"] = "; ".join(f"{max(c for c in cs if c <= pub):%Y-%m} {e:.0f}d"
                                    for (pub, _), e in zip(S.plano, esp) if e > 7)
    return m


# ══════════════════════════════════════════════════ execução e resumo
def roda(qual: str):
    L = []
    if qual == "atraso":
        print("confere (atraso 0 = régua, dia 1):", confere(1), flush=True)
        for Ld in (0, 3, 7, 14, 21):
            for d in R.DIAS:
                L.append(dict(politica=f"atraso {Ld} d", dia=d, **atraso(d, Ld)))
                print(L[-1]["politica"], d, round(L[-1]["carga_mes"], 1), flush=True)
    elif qual == "irregular":
        for q in (0.75, 0.50):
            for modo in ("pronto", "na hora"):
                for d in R.DIAS:
                    for s in range(5):
                        L.append(dict(politica=f"irregular q={q:.2f} ({modo})", dia=d, semente=s,
                                      **irregular(d, q, s, modo)))
                print(f"irregular {q} {modo} ok", flush=True)
    elif qual == "sem vencimento":
        # DECOMPOSIÇÃO, decidida depois de ver o resultado: quanto do preço do mês
        # pulado é o bundle velho e quanto é o tempo cego do vencimento.
        for q in (0.75, 0.50):
            for d in R.DIAS:
                for s in range(5):
                    L.append(dict(politica=f"irregular q={q:.2f} (pronto, sem vencimento)", dia=d, semente=s,
                                  **irregular(d, q, s, "pronto", validade=pd.Timedelta(days=10_000))))
            print(f"sem vencimento {q} ok", flush=True)
    elif qual == "veto":
        for nome, jan, modo in (("veto 72 h (pronto)", pd.Timedelta(hours=72), "pronto"),
                                ("veto 7 d (na hora)", pd.Timedelta(days=7), "na hora")):
            for d in R.DIAS:
                L.append(dict(politica=nome, dia=d, **veto(d, jan, modo)))
                print(nome, d, round(L[-1]["carga_mes"], 1), L[-1]["publicacoes"], flush=True)
    pd.DataFrame(L).to_csv(R.CACHE / f"aprovacao_{qual.replace(' ', '_')}.csv", index=False)


def resumo():
    ref = R.distribuicao().reset_index().assign(politica="automático (régua)", dias_cegos=0.0)
    ref["idade_media"] = 15.0
    T = pd.concat([ref] + [pd.read_csv(R.CACHE / f"aprovacao_{q}.csv")
                           for q in ("atraso", "irregular", "sem_vencimento", "veto") if (R.CACHE / f"aprovacao_{q}.csv").exists()])
    r = ref.median(numeric_only=True)
    cols = [("det", "{:.1f}"), ("inicio", "{:.1f}"), ("banda", "{:.1f}"), ("fp_mes", "{:.3f}"),
            ("carga_mes", "{:.1f}"), ("dias_cegos", "{:.1f}"), ("idade_media", "{:.1f}")]
    print(f"{'política':30s} {'n':>3s} " + " ".join(f"{c:>18s}" for c, _ in cols) + "  veredito")
    for pol, G in T.groupby("politica", sort=False):
        m = G.median(numeric_only=True)
        cel = [(f.format(m[c]) + f" [{f.format(G[c].min())}-{f.format(G[c].max())}]") for c, f in cols]
        tol = (all(m[c] >= r[c] - 0.5 for c in ("det", "inicio", "banda"))
               and m.fp_mes <= 1.1 * r.fp_mes and m.carga_mes <= 1.1 * r.carga_mes and G.dias_cegos.max() == 0)
        print(f"{pol:30s} {len(G):3d} " + " ".join(f"{x:>18s}" for x in cel)
              + f"  {'TOLERÁVEL' if tol else 'não'}  (cenários com cego: {int((G.dias_cegos > 0).sum())})")


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "resumo"
    resumo() if q == "resumo" else roda(q)
