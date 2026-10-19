# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Detecção de precursores de trip da turbina TC-33003A (Cabiúnas/Petrobras) com uma pipeline de **OCSVM por grupo de
sinal**, integrada ao ClearML (dataset, tracking e execução remota no Cica).

- **Pipeline vigente:** 3 canais OCSVM isolados, um por grupo de sinal (EXP33 temperatura de mancal, EXP34 vibração,
  EXP38 pressão de óleo) + canal 4 de proximidade a alarme (5 tags do catálogo, janela de 24 h). Decisão: voto ≥ 2
  de 4, filtro de 45 min, refratário de 48 h, regra de falha de comunicação (E5, 19/10/2026) → 8/8 trips,
  2,82 FP/mês. Cada episódio sai com a causa provável (`describe_alert_context`).
- **Documento vivo:** `DIARIO_DE_BORDO.md` (estado, regras de trabalho, o que já foi testado, registro). `README.md`
  e `docs/` são histórico de consulta.
- **Não reintroduzir** modelo único: a arquitetura é um modelo por grupo de sinal + camada de decisão. Desde 15/10/2026
  o AE (denso e CNN-1D) volta como **candidato por grupo** na disputa D1 (`frente_ocsvm/d1_disputa.py`), pelo caminho
  do AutoML (`_FITTERS`); o caminho antigo do AE em `pipeline.py` (modelo único) segue legado.
- **ClearML:** projeto `TesteMLCab`, fila `default` (4 workers compartilhados com a equipe; ver regras de envio no
  diário, seção 5).

## Environment Setup

```bash
source ../venv/bin/activate   # ou o Python com clearml/sklearn/tensorflow instalado
export CLEARML_CONFIG_FILE="$(pwd)/clearml.conf"   # fora do Git; credenciais também valem via ~/.bashrc
```

## Common Commands

**Treinar um canal (enfileira no Cica se `RUN_REMOTE=true` no config):**
```bash
PYTHONPATH=. python src/main.py --config configs/calibracao_v4_eq/test_grupo_exp34_mancal_vibracao_isolada.json
```

**Experimentos da frente OCSVM** (`frente_ocsvm/`, um script por teste, com pré-registro no docstring):
```bash
python frente_ocsvm/a0_reproducao.py                 # reavaliação local sobre artefatos já treinados
python frente_ocsvm/c1_retreino.py gera|envia|avalia  # testes com treino: configs -> tasks no Cica -> régua
```
Código comum (camada de decisão, régua, canal 4, trips) em `frente_ocsvm/comum.py`. Dados locais em
`frente_ocsvm/dados/` (links, fora do Git).

**Testes:**
```bash
PYTHONPATH=. python -m pytest tests/
```

## Architecture

### Execution Flow

`src/main.py` carrega o JSON em `PipelineConfig` (`src/cnn1d_ae/config.py`), cria a task do ClearML e, se
`RUN_REMOTE=true` e rodando local, chama `task.execute_remotely()` (o worker clona o repo e roda o commit). O `cfg`
vem sempre do JSON de `Args/config`; o painel `pipeline_config` do ClearML é só registro.

Com `ENABLE_AUTOML=true` e `SENSOR_GROUPS`, `pipeline.run()` chama `automl_pipeline.run_automl_group` para cada grupo:
1. Monta o normal de treino (exclui janelas de alarme e buracos longos), clipa outliers, normaliza só com o treino
2. Treina o OCSVM (subamostra de `AUTOML_OCSVM_MAX_TRAIN_SAMPLES`, semente `RANDOM_SEED`) com dados antes de
   `AUTOML_OOS_SPLIT_DATE`
3. AutoML escolhe percentil de limiar × debounce pelo desempenho nos alarmes depois do split
4. Aplica máscara operacional, filtro de duração e portões (carga, degrau, sensor congelado)
5. Salva `point_anomalies_all.csv`, `best_hyperparameters.json` e relatórios como artefatos da task

`ENABLE_WALKFORWARD_RETRAIN` retreina mês a mês (testado no C1 e reprovado).

### Key Modules

| Module | Responsibility |
|---|---|
| `config.py` | `PipelineConfig` — todos os parâmetros |
| `io.py` | Leitura de dados (ClearML Dataset ou local), fuso, auditoria de tempo |
| `preprocess.py` | Máscara de alarme, clipping, normalização só com treino, estado operacional |
| `automl_models.py` | Fitters: `ocsvm` (vigente), `iforest`, `dense` |
| `automl_pipeline.py` | `run_automl_group`: treino por grupo, AutoML de limiar/debounce, walk-forward |
| `scoring.py` | Limiar, seq→ponto, portões, voto entre canais, refratário, episódios e régua |
| `pipeline.py` | `run()`: despacha para AutoML/supervisionado; o resto é o caminho legado do AE |

### Time Handling

Timestamps são normalizados para UTC naive na leitura (`io.py`).
