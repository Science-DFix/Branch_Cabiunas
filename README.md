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
  reaparecem horas depois. É um regime de horas, e T4, T6 e T7 também começam na partida.
- **Limiar próprio para o regime pós-partida (E6):** com o modelo congelado, o mesmo percentil da produção sobre o
  normal de treino das primeiras 6 h. O limiar sobe 3× na temperatura e 74× na vibração. Mantém 8/8, mas o FP cai só
  10% (−5% fora da amostra); dos 16 FP de partida, 8 continuam. Não passa.
- Tempo dos FP de partida (pipeline final com E5, 41 FP; `h_desde_partida` de `episodios_classificados.csv`):
  15 dos 41 começam até 2 h depois de entrar em "on" (5 em < 1 h, 10 em 1–2 h), mediana 1,1 h. O mínimo é 0,8 h por causa do
  filtro de 45 min, e o "on" já vem depois de 60 min de padding, então da ignição são cerca de 2 h. Esses FP duram
  mais: mediana 1,8 h (média 2,7 h, até 9,7 h), contra 1,1 h (média 1,7 h) nos outros. 3 dos 8 trips também começam
  a ≤ 2 h (T4 1,9 h; T6 1,3 h; T7 0,8 h).
- Por que E4 e E6 falham na partida (relatório dos FP, seção 6; `relatorio_falsos_positivos/gera_partida.py`):
  - O canal 4 (janela de 24 h) fica aceso em 98% das primeiras 6 h após a partida (36% em regime), então o voto
    ≥ 2 vira "um modelo aceso". O voto ≥ 2 fica aceso 59% do tempo entre 1 e 3 h (4,9% em regime). Fora do
    refratário, 70% das partidas desde 01/2025 geram alerta (31 de 44).
  - Os modelos ficam acima do normal por 3 a 4 h depois da partida: adiar só empurra o alerta (E4).
  - O score da OCSVM RBF tem teto (3,27 / 87,6 / 1,08 × limiar). O limiar do E6 ficou a 84–97% do teto, e os trips e
    a maior parte dos FP de partida ficam perto dele: não há intensidade que os separe (E6, D3).
- O canal 4 acende na partida: `PI_6240319_AL` aparece a ≤ 3 h de 52% das 83 partidas (base 9,6%) e `PAL_6240315`
  em 66% (base 6,2%).
- Perder trips compra pouco FP (E3): 7/8 dá −19% (−9% fora da amostra); 6/8 dá −27% (−13% fora).
- `PI_6240319_AL`, o alarme que mais sustenta o canal 4, é "Falha PI Gás p/ Motor de Partida". O alarme de que o
  modelo mais depende é ligado à partida, o mesmo momento do maior grupo de FP.
- A falha de comunicação de 14/01/2026 16:18 cai no mesmo minuto de uma rajada de alarmes de temperatura. Indício de que
  as rajadas são eventos do sistema de aquisição.

**O que já entrou na pipeline final** (`scripts/pipeline_unificada_final.py`)
- **Regra de falha de comunicação (E5):** o canal do modelo não vota se algum sensor do grupo teve "Comm Fail"/"Out of
  Serv" nas últimas 6 h. Mantém 8/8 com as mesmas antecedências; FP/mês cai de 2,885 para 2,816 (42 → 41).
- **Contexto do alerta (só descreve, não suprime):** `episodios_classificados.csv` ganha as colunas `contexto`,
  `modelos_acesos`, `tags_canal4`, `h_desde_partida`, `alarmes_canal4_7d`, `alertas_anteriores_7d`,
  `rajada_temperatura_24h` e `falha_comunicacao_6h` (`describe_alert_context` em `src/cnn1d_ae/scoring.py`).

**Pedido de dados (19/10/2026, a enviar):** `pedido_de_dados/pedido_de_dados.pdf` com a planilha
`frente_ocsvm/revisao/pedido_dados_TC33003A.xlsx` (gerada por `frente_ocsvm/pedido_dados.py`). Pede:
- tipo de cada uma das 83 partidas;
- causa das 31 paradas sem alarme crítico;
- o que houve nos 41 alertas sem trip;
- 8 perguntas pontuais;
- dados depois de 20/04/2026;
- ordens de serviço, histórico de alarmes e eventos do sistema de aquisição.

Substitui a planilha de 15/10, que não foi enviada.

**O que explorar a partir do retreino** (cada item com pré-registro antes de enviar ao Cica)
1. **Regime pós-partida no modelo:** é o maior grupo de FP (16 de 36). Correção de 19/10: esse regime **já está** no
   normal de treino. Das horas 1–3 após a partida, 67% entram no treino de temperatura e vibração e 100% no de óleo;
   só os 60 min do transiente ficam fora. Mas é pouco (as primeiras 6 h são ~6% do treino), e a OCSVM (nu = 0,05)
   deixa o mais raro fora da fronteira. "Incluir no treino" não muda nada.
   - O limiar próprio (E6) já é o caso extremo e só deu −10%. Um retreino que reforce o regime na subamostra de 50 mil
     pontos ficaria aquém dele: **não vale gastar Cica com isso** sem informação nova.
   - O que falta é o tipo de partida (planilha): testar se os FP vêm de partidas normais e os trips de partidas
     após trip ou manutenção. Se sim, a informação vira canal ou estratificação do treino.
   - Achado de passagem do E6: na produção, o limiar sai do score dos pontos de treino clipados pelos quantis do
     próprio normal, e o alerta sai da série clipada pelos quantis dela mesma. São dois clippings diferentes. Não
     muda nada hoje (o E6 reproduz os limiares), mas um retreino que padronize o clipping mudaria a calibração.
   - Score com teto: na partida, trips e FP batem no teto da OCSVM RBF. Um retreino com *gamma* menor (teto mais
     alto) ou um score de distância ao normal pós-partida diria *quanto* o episódio foge. Antes de gastar Cica,
     checar local se trips e FP de partida se separam no score do PCA/T² do D1 (sem teto), olhando só a partida.
2. **Canal 4 depois de 27/02/2026:** só `PAL_6240315` sustenta o canal. Revalidar com trips novos e pedir o motivo da
   mudança do `PI_6240319_AL`.
3. **Alerta repetido em 7 dias como prioridade:** 6 de 8 trips têm outro alerta nos 7 dias anteriores, contra 22 de 41
   FP. É fraco com 8 trips; vale como hipótese para dados novos, junto com o B3 (2 modelos juntos).
4. **Retreino mensal com score salvo:** desde `77f09c4`, o walk-forward grava score/limiar do mês. Qualquer retreino
   novo já serve para análises de score.
5. **Trips depois de 20/04/2026 com a pipeline congelada:** o único teste limpo de tudo acima. A ferramenta está
   pronta e conferida: `frente_ocsvm/f1_congelado.py` aplica os 3 modelos de produção (`model.pkl`, normalização,
   limiar) e a camada de decisão a dados novos, sem retreino. No período de referência, reproduz a produção com 0
   divergências (8/8, 41 FP). As hipóteses (P1, P2, S1–S3) estão pré-registradas no docstring. D2-B fica de fora,
   porque precisaria do CNN-1D mensal retreinado nos meses novos (Cica).

Já reprovados no retreino, não repetir sem informação nova:
- descontaminar o treino (A5);
- retreino mensal da OCSVM (C1) e do CNN-1D (1d: 8/8, mas +25% FP);
- trocar a família de modelo (D1);
- deslocar o corte do treino (A4: frágil);
- regime pós-partida: adiar (E4) e limiar próprio (E6), sem retreino.

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
