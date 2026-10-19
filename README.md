# cabiunas-models

> **Frente OCSVM (10/10/2026 em diante): o documento vivo é o [`DIARIO_DE_BORDO.md`](DIARIO_DE_BORDO.md).** O restante deste README e `docs/` são histórico de consulta.

Pipeline de deteccao de anomalias em series temporais de Cabiunas com um OCSVM por grupo de sinal (temperatura, vibracao, oleo) + canal de alarme, integrado ao ClearML para versionamento de dataset, tracking de tasks e execucao remota via ClearML Agent. O caminho CNN-1D Autoencoder que deu nome ao pacote `src/cnn1d_ae/` e legado.

## Falsos positivos: o que sabemos e o que explorar (atualizado em 19/10/2026)

Detalhes em [`relatorio_falsos_positivos/`](relatorio_falsos_positivos/) e no [`DIARIO_DE_BORDO.md`](DIARIO_DE_BORDO.md).

**O que sabemos**
- Trips e FP nascem do mesmo jeito: `PI_6240319_AL`/`PAL_6240315` acendem o canal 4, e um modelo (quase sempre a
  vibração) dá o segundo voto. Nenhum sinal testado separa trip de FP:
  - tags e temporização do canal 4 (A3, B1, B2);
  - dois modelos juntos (B3) e modelo extra (D2);
  - score contínuo da OCSVM e do CNN-1D (D3, D3b);
  - "novidade" do alarme (E2).
- Causas dos 36 FP da OCSVM desde 01/2025 (E1):

  | Causa | FP |
  |---|---|
  | Pós-partida (≤ 2 h) | 16 |
  | Série de vibração 31/01–14/02/2026 | 5 |
  | Regime depois de 27/02/2026 (`PI_6240319_AL` parou de alarmar) | 4 |
  | Alarme crônico | 4 |
  | Comm Fail | 2 |
  | Rajada de alarmes de temperatura | 1 |
  | Sem causa | 4 |

- **Pós-partida não é transitório (E4):** adiar o alerta depois da partida só reduz FP em 5%, porque 15 de 17 desses FP
  reaparecem horas depois. É um regime de horas que o modelo não conhece. T4, T6 e T7 também começam na partida.
- Perder trips compra pouco FP (E3): 7/8 dá −19% (−9% fora da amostra); 6/8 dá −27% (−13% fora).

**O que já entrou na pipeline final** (`scripts/pipeline_unificada_final.py`)
- **Regra de falha de comunicação (E5):** o canal do modelo não vota se algum sensor do grupo teve "Comm Fail"/"Out of
  Serv" nas últimas 6 h. Mantém 8/8 com as mesmas antecedências; FP/mês cai de 2,885 para 2,816 (42 → 41).
- **Contexto do alerta (só descreve, não suprime):** `episodios_classificados.csv` ganha as colunas `contexto`,
  `modelos_acesos`, `tags_canal4`, `h_desde_partida`, `alarmes_canal4_7d`, `alertas_anteriores_7d`,
  `rajada_temperatura_24h` e `falha_comunicacao_6h` (`describe_alert_context` em `src/cnn1d_ae/scoring.py`).

**O que explorar a partir do retreino** (cada item com pré-registro antes de enviar ao Cica)
1. **Regime pós-partida no modelo:** incluir as primeiras horas depois da partida no normal de treino, hoje fora por
   `TRANSIENT_PADDING_MINUTES` e pela máscara operacional, ou dar a esse regime um limiar próprio. É o maior grupo de
   FP (16 de 36). Risco: T4, T6 e T7 começam na partida.
2. **Canal 4 depois de 27/02/2026:** só `PAL_6240315` sustenta o canal. Revalidar com trips novos e pedir o motivo da
   mudança do `PI_6240319_AL`.
3. **Alerta repetido em 7 dias como prioridade:** 6 de 8 trips têm outro alerta nos 7 dias anteriores, contra 22 de 41
   FP. É fraco com 8 trips; vale como hipótese para dados novos, junto com o B3 (2 modelos juntos).
4. **Retreino mensal com score salvo:** desde `77f09c4`, o walk-forward grava score/limiar do mês. Qualquer retreino
   novo já serve para análises de score.
5. **Trips depois de 20/04/2026 com a pipeline congelada:** o único teste limpo de tudo acima.

Já reprovados no retreino, não repetir sem informação nova:
- descontaminar o treino (A5);
- retreino mensal da OCSVM (C1) e do CNN-1D (1d: 8/8, mas +25% FP);
- trocar a família de modelo (D1);
- deslocar o corte do treino (A4: frágil).

## ClearML

Valores padrao do fluxo:

- Projeto: `TesteMLCab`
- Dataset: `Cabiunas 2025`
- Dataset ID: `e2765c3eef2349cda5f5cbcb0fcd5a40`
- Queue remota: `default`
- Docker remoto: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`

O arquivo `clearml.conf` deve existir localmente na raiz do projeto, mas fica fora do Git.

## Preparar Ambiente Local

```bash
cd /home/thallys/Documents/projeto-petrobras/Analise-exploratoria-dos-dados/analise_cabiunas/cabiunas-models
source ../venv/bin/activate
export CLEARML_CONFIG_FILE="$(pwd)/clearml.conf"
```

## Criar/Atualizar Dataset

O dataset ja foi criado no ClearML. Para criar uma nova versao a partir de um config, use o uploader Python:

```bash
PYTHONPATH=. ../venv/bin/python scripts/upload_cabiunas_dataset.py --config configs/calibracao_v4_eq/test_grupo_exp34_mancal_vibracao_isolada.json
```

Tambem existe o wrapper CLI legado:

```bash
scripts/create_clearml_dataset_from_config.sh configs/calibracao_v4_eq/test_grupo_exp34_mancal_vibracao_isolada.json "TesteMLCab" "Cabiunas 2025"
```

## Execucao Remota

Garanta que um worker existente esteja escutando a fila `default`.

```bash
clearml-agent daemon --queue default
```

Em outro terminal, submeta a task:

```bash
cd /home/thallys/Documents/projeto-petrobras/Analise-exploratoria-dos-dados/analise_cabiunas/cabiunas-models
source ../venv/bin/activate
export CLEARML_CONFIG_FILE="$(pwd)/clearml.conf"
PYTHONPATH=. ../venv/bin/python src/main.py --config configs/calibracao_v4_eq/test_grupo_exp34_mancal_vibracao_isolada.json
```

Com `RUN_REMOTE=true`, a maquina local apenas enfileira a task. O worker remoto baixa o codigo do Git, baixa o dataset ClearML e executa o treino.

## Execucao Local

Para rodar localmente, use um config com `RUN_REMOTE=false` ou remova essa chave temporariamente:

```bash
PYTHONPATH=. ../venv/bin/python src/main.py --config configs/calibracao_v4_eq/test_grupo_exp34_mancal_vibracao_isolada.json
```

## Validacao

```bash
python3 -m compileall src/main.py src/cnn1d_ae
python3 -m json.tool configs/calibracao_v4_eq/test_grupo_exp34_mancal_vibracao_isolada.json >/tmp/cabiunas_config_check.json
```
