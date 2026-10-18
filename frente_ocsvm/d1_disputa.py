#!/usr/bin/env python3
"""D1 -- outro modelo, por grupo de sinal e com as MESMAS regras, dá menos FP que a OCSVM sem perder trip?

PRÉ-REGISTRADO EM 16/10/2026, ANTES DE ENVIAR AS TASKS. Aprovado pelo usuário em 15/10 ("sim, pode seguir").
A validação (A0-A6, B1-B5, C1) mostrou que mexer na camada de decisão não corta FP sem perder trip. A pergunta
agora é o MODELO de cada grupo. O CNN-1D AE e o AE denso só foram testados como modelo único com todos os sinais;
nunca por grupo + regras (usuário, 15/10).

DESENHO. Camada de decisão e régua CONGELADAS: canal 4 (5 tags, 24 h), voto >= 2, filtro de 45 min, refratário de
48 h, mesma classificação de episódios. Cada candidato roda pelo caminho do AutoML (automl_pipeline.py), com os 3
configs de produção (configs/calibracao_v4_eq, exp33/34/38) trocando SÓ AUTOML_MODELS. O AutoML escolhe
percentil e debounce na mesma grade e com o mesmo corte de 01/07/2025 que escolheu os da OCSVM; o limiar nunca é
ajustado nos trips. Semente 42, mesmo src/, tasks clonadas das de produção.
  Rodada 1 (este envio): iforest e dense (já implementados; 6 tasks, 1 por vez).
  Rodada 1b: CNN-1D AE por grupo (fitter "cnn1d" em _FITTERS, adicionado em 16/10 e testado em dados sintéticos
            antes de enviar). Mesma arquitetura e mesmo tuner do AE legado (RandomSearch no val_loss do normal,
            MAX_TRIALS 10, EPOCHS 20, BATCH 1024, PATIENCE 6, semente 42), janela TIME_STEPS 60 (30 min) que nunca
            cruza buraco do normal; janelas de treino com passo STRIDE 5 (memória do worker). Score de cada ponto =
            MSE da janela que TERMINA nele (causal). Sem seed sweep (só iforest/ocsvm têm). Envio depois das 6 da
            rodada 1, 1 por vez. Critério, combinações e confirmação iguais (todos_cnn1d + 3 trocas de 1 grupo).
  Rodada 1c (acrescentada em 16/10, antes de enviar qualquer cnn1d; ideia do usuário: "treinamento por grupo de
            sinais, mensal, e usar as regras" para o CNN-1D, que antes só rodava como modelo único e treino único):
            cnn1d_mensal = cnn1d + ENABLE_WALKFORWARD_RETRAIN (retreina do zero no início de cada mês de 07/2025 a
            04/2026 com todo o normal anterior; normalização, tuner e limiar refeitos a cada mês) e
            ENABLE_MIN_DURATION_FILTER = false (o src/ não aceita os dois juntos; no C1 tirar o filtro não mudou
            nenhum episódio). Antes de 01/07/2025 o walk-forward sai com flag 0, então ESTE braço só se julga FORA
            DA AMOSTRA: PASSA = 3/3 fora E FP/mês fora <= 2,158. Comparações: OCSVM treino único (C1-c0: 3/3, 23 FP)
            e OCSVM mensal (C1-c1: 2/3, 26 FP), e o cnn1d de treino único.
  Resultado das rodadas 1, 1b e 1c (18/10): nenhuma das 17 combinações PASSA (DIARIO_DE_BORDO.md).
  Rodada 2 (PRÉ-REGISTRADA EM 18/10/2026, depois do resultado acima e antes de escrever o fitter e de enviar; pedida
            pelo usuário: "pode rodar a rodada 2"). PCA por grupo, fitter "pca" em _FITTERS, no MESMO caminho do
            AutoML (dados já clipados e normalizados só com o treino, como os outros fitters).
            Modelo: PCA(n_components=0,95 da variância, svd_solver full) ajustado no normal de treino. Score de cada
            ponto = índice combinado phi = max(T²/T²_p99, Q/Q_p99), com T² = soma(z_k²/lambda_k) nas componentes
            retidas, Q = SPE (resíduo ao quadrado) e os p99 tirados do PRÓPRIO normal de treino (mesma régua do
            PCA de referência, cabiunas_pdm/scoring.py). Se Q_p99 = 0 (todas as componentes retidas, possível nos
            grupos de 2 sensores), phi = T²/T²_p99, que é a distância de Mahalanobis. O AutoML escolhe percentil e
            debounce sobre phi na mesma grade de sempre; o limiar nunca é ajustado nos trips.
            Braços: pca (treino único, julgado nos dois recortes) e pca_mensal (pca + o mesmo walk-forward do
            cnn1d_mensal, julgado SÓ FORA da amostra). 6 tasks, 1 por vez. Combinações fixadas antes: todos_pca,
            pca_so_<grupo> (3), todos_pca_mensal, pca_mensal_so_<grupo> (3) = 8.
            Critério: o mesmo da rodada 1 (e o da 1c para o mensal). CONFIRMAÇÃO: o PCA é determinístico, então a
            troca de semente não se aplica; quem PASSA roda de novo com 0,90 e 0,99 da variância e vale se PASSA
            nos três. Ressalva: é a 4ª família testada nos mesmos 8 trips; um vencedor ainda precisaria de dados novos.
            EXPECTATIVA: NÃO. Nos grupos de 2 sensores o PCA vira uma elipse de Mahalanobis, mais rígida que a
            fronteira RBF da OCSVM (aposta: mais FP no óleo/temperatura com a deriva). Na vibração (10 sensores) o Q
            pega quebra de correlação entre mancais, que pode antecipar trips, mas também acende com a deriva; o
            mensal deve cortar FP da deriva e, como nos outros modelos, perder trip.

COMBINAÇÕES AVALIADAS (fixadas antes):
  principal   o candidato nos 3 grupos (todos_iforest, todos_dense)
  secundária  o candidato em UM grupo e a OCSVM de produção nos outros dois (2 candidatos x 3 grupos = 6).
              São 6 tentativas a mais; a confirmação em 3 sementes é a proteção contra achar sorte.
Referência: produção (8/8, banda 7/8, 42 FP = 2,885 FP/mês; fora: 3/3, 23 FP = 2,877 FP/mês).

CRITÉRIO (por combinação, semente 42):
  PASSA   8/8 no período inteiro E 3/3 fora da amostra E FP/mês <= 0,75 x produção nos dois recortes
          (<= 2,164 inteiro E <= 2,158 fora).
  NÃO     qualquer outro caso.
Quem PASSA vai para CONFIRMAÇÃO: as tasks do candidato rodam de novo com RANDOM_SEED 7 e 123; vale se PASSA nas 3
sementes. Só então se traça a curva trips x FP (varrendo o limiar sobre sequence_scores_all.csv) contra a da OCSVM,
como leitura secundária. A banda de 4 h e o duty de cada canal se reportam, mas não decidem.

EXPECTATIVA REGISTRADA. Nenhuma combinação principal PASSA. O iforest pontua por isolamento e tende a ser mais
ruidoso que a OCSVM em sinais com deriva (mais FP); o dense AE ponto a ponto aprende quase a mesma fronteira que a
OCSVM (FP a +-15%). O CNN-1D vê a forma temporal (30 min) e pode cortar FP de pico curto, mas perdeu como
modelo único; aposta: também NÃO. Na secundária, a chance maior é a vibração (o canal com mais duty, 14% fora), mas a aposta é
que nenhuma chegue a -25% mantendo 8/8.

Uso:
  python frente_ocsvm/d1_disputa.py gera                    # escreve as configs (commitar e dar push)
  python frente_ocsvm/d1_disputa.py envia --so iforest_temp # clona a task de produção do canal e enfileira
  python frente_ocsvm/d1_disputa.py avalia                  # baixa os artefatos e aplica a régua
"""
from __future__ import annotations
import argparse
import json

import pandas as pd
import comum as C
from a4_variabilidade import RAIZ, REPO, BRANCH, CANAIS, canais_da_replica

CFG_DIR = C.AQUI / "d1_configs"
TASKS_JSON = C.AQUI / "d1_tasks.json"
CNN1D = dict(AUTOML_MODELS=["cnn1d"], TIME_STEPS=60, STRIDE=5, MAX_TRIALS=10, EPOCHS=20, BATCH_SIZE=1024, PATIENCE=6)
MODELOS = {  # variante: mudanças sobre o config de produção
    "iforest": dict(AUTOML_MODELS=["iforest"]),
    "dense": dict(AUTOML_MODELS=["dense"]),
    "cnn1d": CNN1D,
    "cnn1d_mensal": dict(CNN1D, ENABLE_WALKFORWARD_RETRAIN=True, WALKFORWARD_RETRAIN_FREQ="MS",
                         ENABLE_MIN_DURATION_FILTER=False),
    "pca": dict(AUTOML_MODELS=["pca"]),
    "pca_mensal": dict(AUTOML_MODELS=["pca"], ENABLE_WALKFORWARD_RETRAIN=True, WALKFORWARD_RETRAIN_FREQ="MS",
                       ENABLE_MIN_DURATION_FILTER=False),
}
PROD = {c: v[1] for c, v in CANAIS.items()}
REF_FP = {"total": 2.885, "fora": 2.877}  # FP/mês da produção (A0, A1)
CORTE = 0.75


def mudancas(modelo: str) -> dict:
    return MODELOS[modelo]


def gera():
    CFG_DIR.mkdir(exist_ok=True)
    for modelo in MODELOS:
        for canal, (cfg, _, _) in CANAIS.items():
            d = json.loads((RAIZ / "configs/calibracao_v4_eq" / f"{cfg}.json").read_text(encoding="utf-8"))
            d.update(mudancas(modelo))
            (CFG_DIR / f"{modelo}_{canal}.json").write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n",
                                                            encoding="utf-8")
    print(f"{len(MODELOS) * len(CANAIS)} configs em {CFG_DIR}")


def envia(so: str | None):
    import subprocess
    from clearml import Task
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RAIZ, text=True).strip()
    remoto = subprocess.check_output(["git", "ls-remote", "cabiunas", f"refs/heads/{BRANCH}"], cwd=RAIZ, text=True)
    assert remoto.startswith(commit), f"faça push de {commit[:7]} para cabiunas/{BRANCH} antes"
    reg = json.loads(TASKS_JSON.read_text()) if TASKS_JSON.exists() else {}
    for modelo in MODELOS:
        for canal, (cfg, tid, _) in CANAIS.items():
            nome = f"{modelo}_{canal}"
            if (so and so not in (modelo, nome)) or nome in reg:
                continue
            t = Task.clone(source_task=tid, name=f"ocsvm-d1::{nome}::{cfg}")
            t.set_script(repository=REPO, branch=BRANCH, commit=commit)
            t.set_parameter("Args/config", f"frente_ocsvm/d1_configs/{nome}.json")
            for k, v in mudancas(modelo).items():  # só o painel; o cfg vem do JSON de Args/config
                t.set_parameter(f"pipeline_config/{k}", str(v))
            Task.enqueue(t, queue_name="default")
            reg[nome] = t.id
            print(nome, t.id)
            TASKS_JSON.write_text(json.dumps(reg, indent=2) + "\n")


def combinacoes(reg: dict) -> dict:
    reps = {"prod": PROD}
    for m in MODELOS:
        if all(f"{m}_{c}" in reg for c in CANAIS):
            reps[f"todos_{m}"] = {c: reg[f"{m}_{c}"] for c in CANAIS}
        for c in CANAIS:
            if f"{m}_{c}" in reg:
                reps[f"{m}_so_{c}"] = dict(PROD, **{c: reg[f"{m}_{c}"]})
    return reps


def avalia():
    reg = json.loads(TASKS_JSON.read_text()) if TASKS_JSON.exists() else {}
    cat = C.catalogo()
    L = []
    for rep, ids in combinacoes(reg).items():
        try:
            df = canais_da_replica(ids)
        except Exception as e:  # task não terminou ou falhou
            print(f"{rep}: indisponível ({type(e).__name__}: {e})")
            continue
        idx, op = df.index, df["operational_state"]
        ft = C.trips(idx)
        c = {k: df[k] for k in C.CANAIS[:3]}
        c["c4"] = C.canal_alarme(idx, cat.loc[cat.tag.isin(C.TAGS_C4), "t"], C.REF["janela_c4_h"])
        cls, m = C.avalia(C.decide(c, 2, C.REF["dur_min"], C.REF["refrat_h"]), op, ft)
        P = C.por_trip(cls, ft)
        fora = idx >= C.SPLIT
        dias = C.compute_operational_period_days(pd.DataFrame({"operational_state": op[fora]}))
        fp = int(((cls.start >= C.SPLIT) & (cls.classe == "falso_positivo")).sum())
        Pf = P[P.amostra == "fora"]
        on = (op == "on").values
        det, det_f = int(P.deteccao.notna().sum()), int(Pf.deteccao.notna().sum())
        fpm, fpm_f = m["falso_positivo_por_mes"], fp / (dias / 30.4375)
        ok = det_f == 3 and fpm_f <= CORTE * REF_FP["fora"] and ("mensal" in rep or (det == 8 and fpm <= CORTE * REF_FP["total"]))
        L.append(dict(rep=rep, det=f"{det}/8", banda=int(P.banda_4h.sum()), FP=m["n_episodios_falso_positivo"],
                      FP_mes=round(fpm, 3), det_fora=f"{det_f}/3", FP_fora=fp, FP_mes_fora=round(fpm_f, 3),
                      duty=" ".join(f"{df.loc[on, k].mean():.3f}" for k in C.CANAIS[:3]),
                      veredito="PASSA" if ok else "NÃO"))
    # ranking para escolher o ponto de partida do refinamento: trips fora, FP/mês fora, depois o período inteiro
    T = pd.DataFrame(L).sort_values(["det_fora", "FP_mes_fora", "det", "FP_mes"], ascending=[False, True, False, True])
    pd.set_option("display.width", 200)
    print(T.to_string(index=False))
    print(f"\nPASSA = 8/8, 3/3 fora, FP/mês <= {CORTE * REF_FP['total']:.3f} (inteiro) e "
          f"<= {CORTE * REF_FP['fora']:.3f} (fora). duty = temp vib óleo com a máquina on.\n"
          "*_mensal: só vale o fora da amostra (antes de 01/07/2025 o walk-forward sai com flag 0).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=["gera", "envia", "avalia"])
    ap.add_argument("--so")
    a = ap.parse_args()
    {"gera": gera, "envia": lambda: envia(a.so), "avalia": avalia}[a.acao]()
