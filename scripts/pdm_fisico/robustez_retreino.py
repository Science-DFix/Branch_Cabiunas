#!/usr/bin/env python3
"""Ensemble de baselines: o detector deixa de depender do DIA do retreino?

Da geladeira. drift_composicao.py (commit 66049bb, 17/09) mediu e documentou,
sem remediar, a maior fragilidade conhecida do detector: com FIT_POINTS fixo em
20.000, so deslocar o dia do retreino dentro do mes muda QUAIS pontos entram no
baseline, e o resultado vai de 6,6 h/mes de alarme falso (dia 1, o publicado)
a 154,6 h/mes (dia 15) -- 23x. O numero publicado e o melhor caso de uma
distribuicao larga, e em producao ninguem controla em qual caso se cai.

O remedio padrao para variancia de composicao e ensemble: varios ajustes com
composicoes diferentes, scores combinados. Nunca testado aqui.

Variantes, cada uma avaliada com o corte nos dias 1, 8, 15 e 22:
  unico     -- o publicado: os FIT pontos estaveis antes do corte
  janelas   -- 4 PCAs: os FIT pontos que terminam no corte e 7, 14, 21 dias
               antes. Deslocar o corte em 7 dias troca UMA das 4 janelas; a
               composicao combinada muda pouco.
  tamanhos  -- 5 PCAs com 16, 18, 20, 22 e 24 mil pontos antes do corte.
Combinacao: media dos scores normalizados (cada ScorerMax ja sai em unidade de
p99 do proprio baseline). O spread do mancal usa a mediana das medianas e a
media dos MAD. vb nao muda (referencia rolante, independe do retreino).

Todo o resto e o v2 publicado, pelo mesmo codigo (publica_clearml.reproduz e
metricas). O controle -- unico no dia 1 -- tem de dar 8/8 · 6/8 · 5/8 · 0,344.

Criterio: o PIOR dia de corte e a dispersao entre dias, nao o melhor numero.

RESULTADO (30/09/2026, task geladeira::robustez_do_retreino, 6741a88e) --
o ensemble NAO remedia, e a leitura da fragilidade muda.

    variante   banda pior  inicio pior  det pior  FP/mes pior  h FP pior  carga pior
    unico          3/8         4/8        5/8        0,947       154,6      184,5
    janelas        4/8         6/8        6/8        1,206       143,2      210,3
    tamanhos       3/8         4/8        5/8        1,033       148,0      189,2

O unico reproduz o drift_composicao (dia 8: 5/8 e 77,3 h; dia 15: 154,6 h). O
ensemble de janelas melhora o pior caso de banda/inicio/deteccao mas piora FP e
carga; o de tamanhos nao melhora nada. E os dois PIORAM o dia 1 (8/8 -> 7/8,
6,6 -> 31-33 h/mes).

CORRECAO POSTERIOR (limiares_robustos.py, mesmo dia): a leitura abaixo e
PARCIAL. Limiares escolhidos pelo pior caso entre retreinos NAO recuperam as
realizacoes novas -- a instabilidade e do sinal, nao so dos limiares. O que
continua valendo: o dia 1 e in-sample e o numero publicado e otimista.

A LEITURA QUE MUDA. O dia 1 nao e um sorteio feliz de composicao: e o unico
IN-SAMPLE. Todos os limiares do v2 (756 configuracoes, vizinhanca de onze
parametros, minimax) foram escolhidos sobre os sinais do corte no dia 1.
Deslocar o corte -- ou fazer ensemble -- produz OUTRA realizacao do mesmo sinal,
e os limiares nao sao dela. O que drift_composicao mediu como "sensibilidade a
composicao" e, em boa parte, o sobreajuste dos limiares a uma realizacao.

Por isso o ensemble nao tinha como ganhar a limiar fixo: ele tambem e fora da
amostra. Reajustar limiares no ensemble reproduziria o mesmo sobreajuste.

CONSEQUENCIA PARA O NUMERO A CITAR. Nos tres cortes fora da amostra (8, 15, 22)
o detector entrega banda 3-5/8, 0,69-0,95 FP/mes e 77-155 h/mes de alarme
falso. A mediana dos quatro cortes -- 0,82 FP/mes, 77 h/mes, carga 117 h/mes --
e uma estimativa mais honesta do que producao vera do que 0,344 / 6,6 / 48,9.

Uso (de dentro de scripts/pdm_fisico, com os dados):
    python robustez_retreino.py
    python robustez_retreino.py --clearml --remote     # como task no TesteMLCab
"""
from __future__ import annotations

import argparse
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
os.chdir(AQUI)
sys.path.insert(0, AQUI)

if {"--clearml", "--remote"} & set(sys.argv) or os.environ.get("CLEARML_TASK_ID"):
    import clearml  # noqa: F401  -- antes do argparse: o worker injeta os args por patch

import numpy as np
import pandas as pd

DIAS = [1, 8, 15, 22]
FIT = 20_000
IMAGEM = "tensorflow/tensorflow:2.16.1-gpu"
VARIANTES = {
    "unico":    [(FIT, 0)],
    "janelas":  [(FIT, 0), (FIT, 7), (FIT, 14), (FIT, 21)],
    "tamanhos": [(n, 0) for n in (16_000, 18_000, 20_000, 22_000, 24_000)],
}                           # (pontos, dias de recuo do fim da janela)


def garante_dados(dataset_id):
    precisa = ["grade2min.parquet", "falhas.csv", "piso_fisico_cache.npz"]
    if all(os.path.exists(f) for f in precisa):
        return
    from clearml import Dataset
    D = Dataset.get(dataset_id=dataset_id).get_local_copy()
    for f in precisa:
        if not os.path.exists(f):
            os.symlink(os.path.join(D, f), f)


def scorer_max():
    """ScorerMax de ablacao.py, sem importar ablacao (ele roda main() no import)."""
    from cabiunas_pdm import scoring as S

    class ScorerMax(S.MultivariateScorer):
        PHI = 0.10

        def fit(self, baseline):
            super().fit(baseline)
            X = baseline.dropna()[self.cols]
            Xs = self.scaler.transform(X)
            e = (Xs - self.pca.inverse_transform(self.pca.transform(Xs))) ** 2
            p = np.nanpercentile(e, 99, axis=0)
            self.sens_p99_ = np.maximum(p, self.PHI * np.nanmedian(p))
            r, _ = self._raw_scores(X)
            self.recon_p99 = float(np.nanpercentile(r, 99))
            return self

        def _raw_scores(self, df):
            X = df[self.cols]
            ok = X.notna().all(axis=1).to_numpy()
            recon = np.full(len(X), np.nan); maha = np.full(len(X), np.nan)
            if ok.any():
                Xs = self.scaler.transform(X[ok])
                e = (Xs - self.pca.inverse_transform(self.pca.transform(Xs))) ** 2
                recon[ok] = (np.mean(e, axis=1) if getattr(self, "sens_p99_", None) is None
                             else np.max(e / self.sens_p99_, axis=1))
                maha[ok] = self.lw.mahalanobis(Xs)
            return recon, maha
    return ScorerMax


def walkforward(df, stable, dia, janelas):
    """t, p e o spread normalizado do mancal, com retreino no `dia` de cada mes
    e o ensemble descrito em `janelas`."""
    from cabiunas_pdm import config as C, detector as DET
    SM = scorer_max()
    idx = df.index
    base = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC")
    cortes = [c for c in (m + pd.Timedelta(days=dia - 1) for m in base) if idx[0] < c < idx[-1]]
    n = len(idx)
    t = np.full(n, np.nan); p = np.full(n, np.nan); sp = np.full(n, np.nan)
    b_all = DET._spread_mancal(df).to_numpy().astype("float64")
    for i, c0 in enumerate(cortes):
        c1 = cortes[i + 1] if i + 1 < len(cortes) else idx[-1] + pd.Timedelta("2min")
        s = (idx >= c0) & (idx < c1)
        if not s.any():
            continue
        w = df.loc[s]
        ts, ps, meds, mads = [], [], [], []
        for pts, recuo in janelas:
            fim = c0 - pd.Timedelta(days=recuo)
            fit = df.loc[stable & (idx < fim), C.SENSOR_TAGS].dropna().tail(pts)
            if len(fit) < FIT // 4:
                continue
            ts.append(SM().fit(fit[C.TEMPERATURE_TAGS]).score(w[C.TEMPERATURE_TAGS])["pca_recon"].to_numpy())
            ps.append(SM().fit(fit[C.PRESSURE_TAGS]).score(w[C.PRESSURE_TAGS])["pca_recon"].to_numpy())
            b = DET._spread_mancal(fit)
            meds.append(float(b.median())); mads.append(float((b - b.median()).abs().median() * 1.4826))
        if not ts:
            continue
        t[s] = np.mean(ts, axis=0); p[s] = np.mean(ps, axis=0)
        sp[s] = np.abs((b_all[s] - np.median(meds)) / np.mean(mads))
    return {"t": t, "p": p, "sp": sp}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-id", default="8b06a98f8b264820a9ecf2075a188395")
    ap.add_argument("--clearml", action="store_true")
    ap.add_argument("--remote", action="store_true")
    ap.add_argument("--fila", default="default")
    ap.add_argument("--nome", default="geladeira::robustez_do_retreino")
    a = ap.parse_args()

    task = None
    if a.clearml or a.remote:
        from clearml import Task
        task = Task.init(project_name="TesteMLCab", task_name=a.nome, task_type=Task.TaskTypes.testing,
                         reuse_last_task_id=False, auto_connect_frameworks=False)
        task.set_base_docker(docker_image=IMAGEM)
        task.set_packages(["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "tzdata",
                           "matplotlib", "clearml"])
        task.add_tags(["geladeira", "retreino", "ensemble"])
        if a.remote:
            task.execute_remotely(queue_name=a.fila, exit_process=True)

    garante_dados(a.dataset_id)
    import publica_clearml as PC
    from cabiunas_pdm import config as C

    g = pd.read_parquet("grade2min.parquet")
    df = g[C.SENSOR_TAGS]
    stable = ((g["RUNNING_A"] > 0.5) & (g["T5_AVG_A"] > 300)).fillna(False)

    linhas = []
    for nome, jan in VARIANTES.items():
        for dia in DIAS:
            sin = walkforward(df, stable, dia, jan)
            al, mask, alvo, _, idx, sel = PC.reproduz(v2=True, sinais=sin)
            res, _, _ = PC.metricas(al, mask, alvo, sel, permutacao=False)
            n = lambda s: int(s.split("/")[0])
            r = dict(variante=nome, dia=dia, det=n(res["recall"]), inicio=n(res["recall_regua_inicio"]),
                     banda=n(res["recall_banda_acionavel"]), fp_mes=res["fp_por_mes_regra_c"],
                     h_fp_mes=res["horas_fp_por_mes_regra_c"], carga=res["carga_h_por_mes"],
                     lead_ini=res["lead_medio_inicio_h"], eps=res["episodios"])
            linhas.append(r)
            print(r, flush=True)
            if nome == "unico" and dia == 1:
                assert (r["det"], r["inicio"], r["banda"], r["fp_mes"]) == (8, 6, 5, 0.344), \
                    "controle nao reproduz o ponto publicado -- nada abaixo vale"
    T = pd.DataFrame(linhas)
    T.to_csv("robustez_retreino.csv", index=False)

    R = T.groupby("variante", sort=False).agg(
        banda_pior=("banda", "min"), banda_melhor=("banda", "max"),
        inicio_pior=("inicio", "min"), det_pior=("det", "min"),
        fp_mes_pior=("fp_mes", "max"), fp_mes_mediana=("fp_mes", "median"),
        h_fp_pior=("h_fp_mes", "max"), h_fp_mediana=("h_fp_mes", "median"),
        carga_pior=("carga", "max"), carga_mediana=("carga", "median")).reset_index()
    pd.set_option("display.width", 220)
    print("\n=== por dia de corte ===\n" + T.to_string(index=False))
    print("\n=== resumo: o PIOR dia de corte decide ===\n" + R.to_string(index=False))

    if task is not None:
        lg = task.get_logger()
        lg.report_table("por dia de corte", "todas", iteration=0, table_plot=T)
        lg.report_table("resumo (pior caso)", "por variante", iteration=0, table_plot=R)
        task.upload_artifact("robustez_retreino", artifact_object="robustez_retreino.csv")
        task.flush(wait_for_uploads=True)


if __name__ == "__main__":
    main()
