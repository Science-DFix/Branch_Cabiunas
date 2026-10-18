# Diário de bordo — pipeline OCSVM (3 canais + alarme), TC-33003A

Aberto em 10/10/2026. **Daqui pra frente, este é o documento vivo da frente OCSVM.** O histórico de
agosto e setembro continua onde está e serve de consulta, não de plano:

| Para consultar | Onde |
|---|---|
| Tudo do EXP10 ao EXP39, com números e tasks | `docs/analise_automl_exp10.md` |
| Relatório operacional (8/8, 2,88 FP/mês) | `relatorio_pipeline_operacional/relatorio_pipeline_operacional.tex` (commit `c5ec664`; revisado com a seção 8 de validação em `6699db9`) |
| EXP28 a EXP37, apêndice dos 24 isolados | `relatorio_exp28_pipeline_francisco/relatorio_exp28.tex` |
| Eras anteriores (AutoML EXP5–EXP18, CNN EXP13, máscara) | `docs/*.md` |

---

## 1. Ponto de partida (congelado em 10/10/2026)

**Pipeline de referência** — `scripts/pipeline_unificada_final.py`:

| Peça | Configuração |
|---|---|
| Canal 1, temperatura | OCSVM, `TC382_03_A` + `T5_AVG_A` (config EXP33) |
| Canal 2, vibração | OCSVM, 10 `TV_*` (EXP34) |
| Canal 3, óleo | OCSVM, `PI_0308` + `PDIT_0305` (EXP38) |
| Canal 4, alarme de processo | sem modelo: alarme de `PI_6240319_AL`, `PAL_6240315`, `PDAL_6240302`, `TC382_05_A` ou `PAH_6240319` nas últimas 24 h |
| Decisão | voto ≥ 2 de 4 → filtro de duração 45 min → refratário 48 h |
| Treino | 01/07/2024 a 20/04/2026, split fora da amostra em **01/07/2025**, 50.000 amostras, ν = 0,05 |
| Tasks de produção | `b494d457` (temperatura), `9c344688` (vibração), `fc4123fb` (óleo); retreino de 08/09 idêntico a `805fbf34`/`7815d2cf`/`18a61687` |
| Catálogo de alarmes | Dataset `a97ba56ba14840fbb1125c2a82f883c9`, `alarmes_selecionados_turbina_a.csv` |

**Resultado de referência:** 8/8 trips, **2,88 FP/mês**, 25 inconclusivos, antecedência média 23,8 h.

**Por trip** (tabela 7 do relatório operacional; régua rigorosa: episódio nasce até 48 h antes):

| Trip | 1ª detecção | Antecedência | Período | Banda [T−48 h, T−4 h] |
|---|---|---|---|---|
| 27/02/2025 08:38 | 25/02/2025 22:51 | 33,8 h | dentro da amostra | sim |
| 17/03/2025 18:16 | 16/03/2025 11:04 | 31,2 h | dentro da amostra | sim |
| 07/04/2025 21:18 | 06/04/2025 02:05 | 43,2 h | dentro da amostra | sim |
| 11/04/2025 17:02 | 10/04/2025 21:15 | 19,8 h | dentro da amostra | sim |
| 29/04/2025 03:04 | 27/04/2025 14:22 | 36,7 h | dentro da amostra | sim |
| 04/11/2025 06:22 | 03/11/2025 16:40 | 13,7 h | **fora** da amostra | sim |
| 09/12/2025 08:36 | 09/12/2025 00:15 | 8,4 h | **fora** da amostra | sim |
| 26/02/2026 15:34 | 26/02/2026 11:48 | 3,8 h | **fora** da amostra | **não** (< 4 h) |

Leitura: **8/8 no total = 5/5 dentro da amostra + 3/3 fora**; na banda acionável de 4 h, **7/8** (fora da
amostra, 2/3). Os 8 horários de trip são também a referência para reconstruir o
`alarmes_francisco_falhas.csv` (não versionado), se a cópia não aparecer.

**O que sabemos que o número esconde:**
- **5 dos 8 trips estão no período de ajuste e calibração** (27/02, 17/03, 07/04, 11/04 e 29/04/2025,
  todos antes do split). Fora da amostra só há 3: 04/11/2025, 09/12/2025 e 26/02/2026.
- **Os 8/8 dependem do canal 4** (sem ele: 0/8). O canal 4 fica aceso ~47% do tempo e tem
  enriquecimento 1,23× (p = 0,19) — não se distingue do acaso sozinho. Os 3 canais modelados coincidem
  entre si só ~1,2% do tempo.
- O filtro de 45 min, a janela de 24 h, as 5 tags e o refratário de 48 h foram **escolhidos olhando os
  mesmos 8 trips**.

---

## 2. Regras de trabalho

1. **Pré-registro antes de rodar.** Cada teste ganha um script com docstring contendo pergunta, braços,
   critério de decisão e expectativa, commitado **antes** do resultado. O resultado entra depois, no
   mesmo docstring e neste diário. Mudar o critério depois de ver o resultado fica registrado como a
   posteriori.
2. **Treino só no ClearML do Cica.** Tudo que reajusta OCSVM, limiar ou portão vai como task (fila
   `default`, projeto `TesteMLCab`). Reavaliar a camada de decisão sobre artefatos já treinados pode
   rodar local.
3. **Régua.** A de sempre (`classify_episodes_regua`): episódios com fusão < 2 h; detecção = episódio
   que nasce até 48 h antes do trip; inconclusivo = parada real ≥ 2 h até 48 h depois; FP/mês por mês de
   operação vigiada. **Sempre reportar também fora da amostra (os 3 trips depois de 01/07/2025).**
4. **Nada é aceito por um número só.** Detecção com nulo (quantas acima do acaso), custo com intervalo,
   e o efeito por evento (qual trip ganha ou perde).
5. **Métrica de canal isolado não decide.** Três vezes (grade fina, iforest, ensemble por sinal) a
   melhora isolada piorou o conjunto. Só a pipeline completa na régua decide.

---

## 3. Já testado — não repetir sem informação nova

Detalhes no histórico (`docs/analise_automl_exp10.md`).

| Ideia | Resultado |
|---|---|
| Filtro de duração por canal (além do da votação) | redundante, FP parado em 2,88 |
| Grade de AutoML mais fina (153 combinações) | 8/8 → 4/8 |
| iforest/dense no AutoML; melhor modelo por sinal | cada troca perde 27/02 ou 17/03 |
| Refratário ≠ 48 h | 60 h já perde 2 trips |
| Voto ≥ 3 de 4 | 5/8 |
| Supressão de 6 h pós-religamento | perde 11/04 e 04/11 |
| Supressão cirúrgica perto de portão | mata precursores reais (07/04, 26/02) |
| EWMA no score antes do limiar (EXP18) | piora |
| Modelo único com features extras (EXP35/36) | 6/8, saturado |
| Tirar o canal 4 / tirar as tags de utilidade | 0/8 / 3/8 |
| Votação 2-de-2 entre famílias (era EXP10c) | refutada |
| Filtro de duração adaptativo / com walk-forward | refutado |

---

## 4. Abordagens a explorar

Ordem de ataque: **primeiro validar** (A), porque se a detecção estiver perto do acaso, cortar FP só
troca acertos por sorte. Depois reduzir FP na camada de decisão (B, sem treino) e, com o Cica, nos
modelos (C).

| # | Abordagem | Pergunta | Precisa treinar? | Estado |
|---|---|---|---|---|
| **A0** | Reprodução local | os artefatos em cache reproduzem 8/8 e 2,88? | não | **feito 10/10: passou** |
| **A1** | Avaliação dentro × fora da amostra | quanto do 8/8 está nos 3 trips depois do split? | não | **feito 10/10** |
| **A2** | Nulo de acaso | quantas detecções faria um canal 4 aleatório de mesmo duty, ou os alarmes deslocados no tempo? | não | **feito 10/10: acima do acaso** |
| **A3** | LOEO da camada de decisão | filtro, janela, tags e refratário escolhidos sem o evento ainda o pegam? | não | **feito 10/10: 5/8 (tags) · 8/8 (5 tags fixas)** |
| **A4** | Variabilidade de treino | deslocar o split e a amostra de 50 mil muda o resultado? | **sim (Cica)** | **feito 14/10: ROBUSTO à semente** (8/8, FP/mês 2,82–2,95); **FRÁGIL ao corte** (sp0501 4/8; sp0901 8/8) |
| **A6** | Controle do z-score do relatório (seção 6.2) | os 61,9% de FP "com sinal físico" superam janelas sorteadas de mesma duração? | não | **feito 15/10: NÃO ESPECÍFICO** (controle 96,9%) |
| **B1** | Chattering do `PI_6240319_AL` | contar só ativações novas após silêncio mínimo reduz o duty do canal 4 sem perder trips? | não | **feito 10/10: reprovado** |
| **B2** | Janela do canal 4 × filtro de duração | a janela de 24 h, varrida junto com os 45 min, tem ponto melhor? | não | **feito 10/10: reprovado** (−32% dentro vira −4% fora) |
| **B3** | Priorização em vez de supressão | gestão de alarme (ativo nas primeiras N h, depois "condição conhecida") e o contexto de catálogo do EXP30/31 como prioridade | não | **feito 14/10: REPROVADO** (R1/R2 rebaixam os 5 trips de dentro; fora 3/3 em ALTA, hipótese para dados novos) |
| **A5** | Descontaminação do treino | tirar [T−72 h, T+24 h] dos trips do normal de treino faz os trips de dentro virarem coincidência de 2 OCSVM (R1)? | **sim (Cica)** | **feito 14/10: REFUTA** (1/5 dentro em ALTA; FP/mês 2,88 → 3,50) |
| **B5** | FP como "alarme antecipado" | os episódios de FP antecedem alarme novo do catálogo (48 h) mais que o acaso? (pipeline retreinada do zero) | **sim (Cica)** | **feito 14/10: NULO** (0,70x todas · 1,07x sem canal 4; HIGH vazio por construção) |
| **C1** | Retreino mensal (walk-forward) | o −19% de FP do EXP10c se repete nos canais separados? | **sim (Cica)** | **feito 14/10: REPROVADO** (perde 26/02/26, FP fora +3 episódios; vib duty cai à metade) |
| **C2** | OCSVM + iforest como votos extras no mesmo canal | somar (não substituir) muda o voto? | **sim (Cica)** | a fazer |
| **C3** | Limiares dos portões por LOEO | o −32% de FP do EXP10c se repete? | **sim (Cica)** | a fazer |
| **D1** | Disputa entre modelos por grupo de sinal | com a camada de decisão congelada, outro modelo (iforest, AE denso, CNN-1D AE, PCA) mantém 8/8 com ≥ 25% menos FP que a OCSVM? | **sim (Cica)** | **pré-registrado 16/10** (`d1_disputa.py`); **rodadas 1, 1b e 1c feitas 18/10: NENHUMA PASSA** (12 tasks, IDs em `frente_ocsvm/d1_envios.md`); rodada 2 (PCA) feita 18/10: **NENHUMA PASSA** (melhor com 8/8: PCA só na temp, −5%). **D1 encerrado** |

**Dados locais (resolvido em 10/10):** `frente_ocsvm/dados/` (fora do git) tem links para
`point_anomalies_final.csv` e `alarmes_francisco_falhas.csv` do clone antigo
(`~/REPO_CABIUNAS/cabiunas-models`, mesmo commit `52d705a`) e para o catálogo de alarmes no cache do
ClearML (Dataset `a97ba56b`). Código comum em `frente_ocsvm/comum.py`, que chama as funções de produção
de `src/cnn1d_ae/scoring.py`. Os 76 scripts e fontes que só existiam no clone antigo foram versionados
em `de18d60`.

---

## 5. Registro

Uma entrada por teste, da mais recente para a mais antiga: data, abordagem, script, commit do
pré-registro, resultado, decisão.

### 18/10/2026 — D1 rodada 2, PCA por grupo — NENHUMA COMBINAÇÃO PASSA
- **Pré-registro:** docstring de `frente_ocsvm/d1_disputa.py`, commit `e026b73` (antes do fitter e do envio).
  PCA com 95% da variância no normal de treino; score phi = max(T²/T²_p99, Q/Q_p99) com p99 do próprio treino
  (só T², isto é Mahalanobis, quando todas as componentes ficam retidas). Braços `pca` e `pca_mensal`
  (walk-forward, julgado só fora), 6 tasks, 8 combinações, mesmo critério. Confirmação: 0,90 e 0,99 da variância
  (o PCA é determinístico, a semente não muda nada). Expectativa: NÃO.
- **Fitter `pca`** em `automl_pipeline.py` + `AUTOML_PCA_VARIANCE` no config. Teste sintético: phi igual à conta
  manual; com 2 sensores, phi·T²_p99 = Mahalanobis² (o Q_p99 dava 4e-31 de ruído e foi zerado pela regra de
  "todas as componentes retidas"); walk-forward ok; pytest 15/15.
- **Envio:** 1 por vez (temp, vib, óleo, depois os 3 mensais), IDs em `frente_ocsvm/d1_envios.md`.
- **Correção de premissa (sem mudar o critério):** o pré-registro supôs que temp e óleo teriam 2 dimensões, mas o
  AutoML treina sobre as features derivadas do grupo; a `pca_temp` reteve 17 componentes (Q_p99 = 26,9), então o caso
  "só Mahalanobis" não ocorre ali. A expectativa ("elipse rígida") partia dessa premissa errada.
- **Fila:** a task fica `completed` antes de o worker terminar de subir os artefatos (~3 min para 100 MB; o
  `task.close()` espera). A fila agora também espera o worker da task anterior ficar livre (1 task nossa por vez).
- **Tasks (commit `67c3f7c`):** pca_temp `8077dc2e`, pca_vib `3a7b65aa`, pca_oleo `47964c2e`, pca_mensal_temp
  `d42c0e47`, pca_mensal_vib `a6d764a1`, pca_mensal_oleo `37a895cc`; todas completed 18/10 11:57–12:44.
- **Placar (8 combinações): nenhuma PASSA**, como na expectativa.
  - Melhor com 8/8: `pca_so_temp`, 40 FP (2,747, −5% vs 2,885) e fora 3/3 com 23 (= OCSVM). Longe do −25%.
    `pca_so_oleo` = produção (o óleo quase não vota).
  - Vibração piora: `pca_so_vib` 6/8 e 2/3 fora (27 FP); `todos_pca` 6/8, 2/3 fora.
  - Mensal (só fora): `pca_mensal_so_temp` e `_so_oleo` 3/3 e 23 (= OCSVM); `_so_vib` 3/3 e 33 (4,128, duty da vib
    sobe para 15%); `todos_pca_mensal` 3/3 e 33.
- **Decisão:** sem candidato para a confirmação (0,90/0,99). Fecha o D1: nas 4 famílias (iforest, dense, CNN-1D,
  PCA), treino único e mensal, nenhum modelo por grupo corta 25% do FP mantendo os trips. A OCSVM segue nos 3
  grupos; reduzir FP pela troca de modelo, com estas regras e estes 8 trips, não se sustentou.
- **Funil das regras pós-treino (pergunta do usuário, `frente_ocsvm/d1_funil.py`; fora da amostra, máquina on,
  700.843 pontos):** o CNN-1D mensal corta muito PONTO e quase nada de EPISÓDIO.
  - Pontos anômalos da vib: 97.992 (OCSVM) → 43.520 (−56%); algum canal de modelo 108.633 → 66.021 (−39%).
  - Depois das regras: voto ≥ 2 58.302 → 45.825 (−21%); 45 min 44.091 → 30.578 (−31%); 48 h 11.304 → 9.951 (−12%).
  - Episódios: 35 → 38 e FP 23 → 26 (todos_cnn1d_mensal); só na vib, 36 e 24. Detecções 3/3 nos dois.
  - Leitura: o canal 4 fica aceso 36% do tempo, então basta um canal de modelo encostar nele para virar voto; o
    mensal deixa menos pontos, mas espalhados em episódios separados. A régua conta episódio (FP), não ponto.
    No temp o mensal faz o contrário: 14.544 → 33.674 pontos (2,3x).

### 17–18/10/2026 — D1, rodadas 1, 1b e 1c — NENHUMA COMBINAÇÃO PASSA
- **Estado ao retomar:** as 10 tasks de 16/10 (iforest, dense e cnn1d nos 3 grupos; cnn1d_mensal só em temp)
  terminaram `completed`; nenhuma ainda passou pela régua (`d1_disputa.py avalia`).
- **Envio:** `cnn1d_mensal_vib` e depois `cnn1d_mensal_oleo`, 1 por vez, commit `c2d183a`, semente 42. Cada ID e
  mudança de status é anotado em `frente_ocsvm/d1_envios.md` (sobrevive ao terminal fechar; o `envia` também
  grava em `d1_tasks.json` e pula o que já está lá).
- **Não commitar no meio da fila:** o `envia` exige HEAD == `cabiunas/exp/ocsvm-validacao`.
- **Placar parcial (`avalia`, 17/10, 10 tasks, semente 42): nenhuma combinação PASSA.** Melhor FP com 8/8:
  `dense_so_oleo` e `cnn1d_so_oleo`, idênticos à produção (42 FP, 2,885; o óleo quase não vota). Abaixo do corte
  só `dense_so_vib` (1,992), mas com 4/8 e 2/3 fora. Principais: todos_iforest 6/8 (3,091), todos_dense 4/8
  (3,022), todos_cnn1d 6/8 e 2/3 fora (3,503). `cnn1d_mensal_so_temp` (vale só fora): 3/3, 27 FP (3,377) > 23 da
  OCSVM. Coerente com a expectativa registrada. Falta o mensal de vib e óleo.
- **`cnn1d_mensal_vib` falhou** (`71647eac`, exit 137 = memória do worker) no 8º dos 10 retreinos. Causa medida
  local: cada `fit` do TF/Keras deixa memória que nem `clear_session`, `gc` ou `malloc_trim` devolvem (~45 MB por
  fit puro, ~200 MB por tuner de 2 trials; com 10 trials e 10 sensores ~1 GB/mês). Correção em `automl_pipeline.py`:
  no walk-forward, o `cnn1d` treina cada mês num processo filho (`spawn`) e só o modelo do último mês fica na
  memória. Teste sintético: RSS do pai estável (+8 MB/mês contra +200 MB/mês), saída igual dentro do ruído do TF
  (6e-5; duas execuções iguais diferem 4e-5); pytest 15/15. Método inalterado (a semente é refixada a cada
  chamada), então a `cnn1d_mensal_temp` (commit antigo) continua valendo. Reenvio de vib e depois óleo no commit novo.
- **Reenvio:** `cnn1d_mensal_vib` `6fbead3c` completed 18/10 02:14 (10/10 meses; correção validada no worker);
  `cnn1d_mensal_oleo` `7a7b77bf` completed 03:34. O script de fila não percebia o fim das tasks (status vem como
  enum); corrigido no scratchpad, não afeta resultado.
- **Placar final (12 tasks, semente 42): nenhuma das 17 combinações PASSA**, como na expectativa registrada.
  - Com 8/8: só `dense_so_oleo` e `cnn1d_so_oleo` (idênticas à produção, 2,885; o óleo quase não vota) e
    `cnn1d_so_vib` (3,366, pior).
  - Mensal (vale só fora): melhor `cnn1d_mensal_so_oleo`, 3/3 e 23 FP (= OCSVM, 2,877); `_so_vib` 3/3 e 24;
    `_so_temp` 3/3 e 27; `todos_cnn1d_mensal` 3/3 e 26 (3,252). Nenhum chega a 2,158.
  - Cortes grandes de FP só vêm com perda de trip: `dense_so_vib` 1,992 com 4/8 e 2/3 fora; `todos_cnn1d_mensal`
    1,786 no inteiro com 3/8 (o inteiro não vale para o mensal). O canal de vib cai de 10,5% para 2,5–3,4% de duty
    e leva os trips junto.
- **Decisão:** sem candidato para a confirmação em 3 sementes. A OCSVM segue como modelo dos 3 grupos. Fica a
  rodada 2 (PCA/T² por grupo), que estava fora deste registro e precisa de pré-registro próprio.

### 15/10/2026 — D1, disputa entre modelos por grupo de sinal — PLANEJADO, PARADO (Cica fora do ar)
- **Contexto:** depois de ler o relatório operacional com a validação (A0–A6, B1–B5, C1), o usuário perguntou
  o que mudou. Os números não mudaram (8/8, 2,885; retreino do zero idêntico); mudou o quanto se pode confiar
  neles: 5/8 no LOEO, 8,4 h de antecedência fora da amostra, 4/8 com outro corte ou outra grade, natureza dos
  FP indeterminada. Minha indicação foi parar de ajustar, congelar e testar em dados novos ao lado do PCA.
- **Decisão do usuário:** nova abordagem. Mantém 8/8 como dado e mira **reduzir FP**, com a premissa de que outro
  modelo, sob as mesmas regras, acende menos no normal. Inclui o autoencoder: o CNN-1D AE sempre foi treinado
  com todos os sinais juntos e sem regras; nunca foi testado por grupo de sinal com a camada de decisão.
- **Por que a premissa é plausível:** o FP nasce da coincidência de um canal de modelo com o canal 4 (aceso
  ~47% do tempo). O canal de vibração da OCSVM fica aceso 10,5% do tempo em operação e é o maior gerador de FP.
- **Desenho combinado (a pré-registrar antes de enviar):**
  - Camada de decisão e régua congeladas (voto ≥ 2, 45 min, 48 h, canal 4 com 5 tags/24 h). Todo candidato
    roda por grupo (temp, vib, óleo), pelo mesmo caminho do AutoML (mesmo limiar por percentil do treino,
    mesmos portões, mesmo artefato), trocando só `AUTOML_MODELS` nos 3 configs de produção.
  - Rodada 1: `iforest` e `dense` (já implementados; só config, 6 tasks) e CNN-1D AE (fitter novo em
    `_FITTERS` de `automl_pipeline.py`, usando `model.py`/`sequences.py`, janelas sem cruzar buracos; teste
    local antes do Cica; 3 tasks). Rodada 2: PCA/T² por grupo (código pequeno) e treino mensal dos melhores
    (no C1 o mensal piorou a OCSVM; no PCA ele funciona, então vale testar por modelo).
  - Critério para trocar a OCSVM: mantém 8/8 e 3/3 fora; **≥ 25% menos FP/mês** que 2,885 (ruído entre
    sementes ~±3%); curva trips × FP/mês igual ou melhor (traçada local a partir do `sequence_scores_all.csv`
    que o AutoML grava); confirmação em 3 sementes. O limiar nunca é ajustado olhando os trips.
  - Ressalva registrada: escolher entre vários modelos nos mesmos 8 trips pode repetir o otimismo do A3; o FP
    (~42 episódios) é métrica mais estável que trips, mas o vencedor ainda precisa de dados novos.
- **Levantamento do código:** `src/cnn1d_ae/automl_pipeline.py` despacha por `_FITTERS = {dense, ocsvm,
  iforest}` com a interface `fitter(cfg, x_normal, x_all, n_features) -> (train_err, all_err, modelo,
  params)`; `x_normal` é o trecho normal antes do split, `x_all` a série inteira. O caminho antigo do AE
  (`run_one_group` em `pipeline.py`) tem limiar e portões próprios e não serve para comparação justa.
- **Cica fora do ar (15/10, manhã):** o nome resolve e o relay do Tailscale Funnel aceita a conexão, mas o
  TLS é encerrado sem resposta (API, web e arquivos). Na noite de 14/10 funcionava. Deixei um vigia de 5 em
  5 min. Também neste dia: o A6 lê o CSV de sensores do cache local porque o `Dataset.get` travava.
- **Ao retomar:** (1) conferir o Cica e a fila; (2) escrever e commitar o pré-registro do D1; (3) atualizar o
  `CLAUDE.md` (AE volta como candidato por grupo); (4) enviar `iforest` e `dense`, 1 task por vez; (5) em
  paralelo, local, o fitter do CNN-1D.
- **Pendente com a equipe:** planilha `frente_ocsvm/revisao/paradas_para_revisao.xlsx` (31 paradas + 4 FP sem
  explicação), ainda não enviada.

### 15/10/2026 — A6, controle do z-score da seção 6.2 do relatório operacional — NÃO ESPECÍFICO
- **Script:** `frente_ocsvm/a6_controle_zscore.py` · pré-registro `60ffec6` (leitura do cache em `82d12c3`) ·
  local, ~5 min.
- **Pergunta:** o relatório (seção 6.2) deu 26/42 FP (61,9%) "confirmados como sinal físico" pelo pico de |z| de
  15 sensores contra as 2 h anteriores, sem controle. Janelas sorteadas de mesma duração (on, longe de qualquer
  episódio) dão menos?
- **Resultado:** reprodução exata (26/42, mesmos 4 sem explicação). Controle (840 janelas): **96,9%**
  confirmado; só janelas totalmente "on" (a posteriori): 97,3%. Diferença −35 p.p.
- **Leitura:** o critério é satisfeito por quase qualquer trecho de operação; não discrimina. Os 61,9% não
  sustentam que os FP sejam eventos físicos. Somado à seção 6.3 (enriquecimento real, mas 8 dos 10 alarmes vêm
  ANTES do episódio: eco, não antecipação) e ao B5 (sem antecipação de alarme), a conclusão da seção 6.5 do
  relatório ("FP são eventos físicos reais; não suprimir mais") fica sem sustentação. Expectativa: veredito
  certo, magnitude errada (previ 40–55%).
- **Decisão:** corrigir as seções 4.2, 6.2, 6.3 e 6.5 do relatório (nova seção 8, validação).
- **Feito no mesmo dia:** relatório operacional revisado (notas nas seções 4.2, 6.2, 6.3, 6.5 e seção 8 com
  todos os testes; 17 páginas). Planilha de revisão ganhou a aba "Alertas sem explicação" com os 4 FP sem
  sinal físico (14/01/25, 24/01/25, 21/02/25, 09/03/26), e a lista suspensa das paradas foi corrigida (uma
  opção tinha vírgula e se partia em duas no Excel).

### 14/10/2026 — B5, FP como "alarme antecipado" (pipeline retreinada do zero) — NULO
- **Script:** `frente_ocsvm/b5_alarme_antecipado.py` · pré-registro `e830050` · 3 tasks novas no Cica com os
  configs de produção sem mudança (AutoML livre, semente 42), a pedido do usuário ("fazer todo o processo").
- **Ideia (do usuário):** aceitar os FP e rever a contagem. Um FP que antecede um alarme real ainda não ativo
  seria "alarme antecipado", não falso.
- **Reprodução:** o AutoML escolheu os mesmos hiperparâmetros e a pipeline nova dá exatamente a produção (8/8,
  banda 7/8, 42 FP, mesmos leads).
- **Teste:** alarme novo = tag que vai a ACT em (s, s + 48 h] sem ter ido em [s − 24 h, s]; acaso = mesma regra
  em instantes de 5 min elegíveis a FP (on, sem trip nem parada ≥ 2 h nas 48 h seguintes). Válido se ≥ 1,5x e
  p < 0,05/3.

| Recorte | FP com alarme novo | Acaso | Enriquecimento | p | |
|---|---|---|---|---|---|
| todas as tags | 14/42 (33%) | 47,6% | 0,70 | 0,98 | NULO |
| HIGH/LOLO | 0/42 | 0,0% | — | — | NULO (vazio por construção) |
| sem as 5 do canal 4 | 7/42 (17%) | 15,6% | 1,07 | 0,49 | NULO |

- **Leitura:** os FP não antecipam alarme além do acaso; com todas as tags ficam até abaixo dele. O recorte HIGH é
  vazio porque 252/266 ativações HIGH/LOLO ocorrem com a máquina já parada: acompanham paradas, que a régua já
  conta como "inconclusivo". Expectativa certa (NULO nos 3).
- **Decisão:** a 4ª classe não se sustenta; os 2,885 FP/mês continuam sendo FP. Com B1–B5 e C1, a camada de
  decisão e a régua estão esgotadas com a informação disponível.

### 14/10/2026 — A5, treino sem [T−72 h, T+24 h] dos trips — REFUTA a hipótese do B3
- **Script:** `frente_ocsvm/a5_descontaminacao.py` · pré-registro `1996465` · 3 tasks no Cica · chave nova
  `TRAIN_EXCLUDE_WINDOWS` no `src/` (desligada por padrão). Percentil e debounce fixos nos de produção.
- **Conferência antes:** as 72 h antes dos 5 trips de dentro estavam ~100% no normal de treino (11/04: 29% em
  temp/vib); as dos 3 de fora, 0%.

| | Trips | Banda 4 h | FP/mês | Duty vib | R1: dentro em ALTA | R1: fora em ALTA | FP em BAIXA |
|---|---|---|---|---|---|---|---|
| produção | 8/8 | 7/8 | 2,885 | 0,105 | 0/5 | 3/3 | 18/42 |
| A5 | 8/8 | 7/8 | **3,503** | 0,147 | **1/5** | 3/3 | 24/51 |

- **Critério:** 1/5 → **REFUTA**. Expectativa errada (previ 3/5).
- **Leitura:** tirar o precursor do treino não faz os trips de dentro virarem coincidência de 2 modelos. O
  efeito que aparece é colateral: o limiar (percentil do erro de treino) desce, a vibração acende mais e o FP
  sobe 21%. O padrão dentro × fora do B3 continua sem explicação (o tipo de trip não basta: fora há 2 de óleo e
  1 de mancal, todos em ALTA).
- **Contexto descoberto no mesmo dia:** o pacote de produção (detector PCA de 4 sinais, `cabiunas-models`)
  já exclui ±7 d em volta dos trips conhecidos, mas só da referência rolante da vibração, e retreina todo mês.
  A OCSVM herdou do PCA só o voto ≥ 2 e o refratário de 48 h.
- **Decisão:** não adotar a exclusão no treino da OCSVM. Próximo: levantar quais regras do detector PCA a OCSVM
  não herdou e testar a OCSVM com elas.

### 14/10/2026 — B3, prioridade ALTA/BAIXA por composição do episódio — REPROVADO (com hipótese fora da amostra)
- **Script:** `frente_ocsvm/b3_priorizacao.py` · pré-registro `1f463bf` · local, sem treino.
- **Desenho:** nada é suprimido; cada episódio da referência recebe prioridade. R1 = ALTA se ≥ 2 dos 3 OCSVM
  ativos juntos; R2 = ALTA se ≥ 3 dos 4 canais. A ideia original (teto de N horas) saiu antes de rodar:
  os episódios finais duram 1,2 h na mediana (~10 h/mês de alerta) e o refratário de 48 h já faz esse papel.
- **Resultado:**

| Regra | Trips em ALTA | Fora em ALTA | FP em BAIXA | FP/mês só ALTA | h/mês ALTA |
|---|---|---|---|---|---|
| referência (sem prioridade) | 8/8 | 3/3 | – | 2,885 | 10,3 |
| R1 dois modelos | **3/8** | 3/3 | 18/42 (43%) | 1,648 | 7,1 |
| R2 ≥ 3 canais | **3/8** | 3/3 | 19/42 (45%) | 1,580 | 6,6 |

- **Critério:** as duas rebaixam 5 trips → **REPROVADO**. Expectativa: acertei o veredito e o R1; errei o R2
  (previ ~5/8).
- **A posteriori (não vale como aprovação):** os 5 trips rebaixados são **exatamente os de dentro da amostra**.
  Fora, R1 põe 3/3 trips em ALTA e rebaixa 12/23 FP (52%). Não é deriva: o tempo com ≥ 2 OCSVM juntos é menor
  fora (0,94%) que dentro (1,6%). Hipótese: o trecho de 48 h antes dos trips de dentro entrou no treino como
  normal (a exclusão em volta do alarme é de só 24 h), e os modelos aprenderam a não acender juntos ali.
- **Decisão:** sem aprovação com esses dados. A hipótese "R1 separa trip de FP fora da amostra" vira
  **pré-registro para a validação em dados novos** (pipeline congelada, 21/04/2026 em diante): se os trips
  novos caírem em ALTA e ~metade dos FP em BAIXA, a prioridade R1 vale para a operação.

### 14/10/2026 — C1, retreino mensal (walk-forward) × treino único — REPROVADO
- **Script:** `frente_ocsvm/c1_retreino.py` · pré-registro `f573a86` · tasks em `frente_ocsvm/c1_tasks.json`
  (6 no Cica, 3 em paralelo por braço com a fila vazia). Os 3 grupos por sinal continuam separados (1 OCSVM por
  grupo). Filtro de 4,5 min desligado nos dois braços (o `src/` proíbe esse filtro junto com o walk-forward).
  Percentil e debounce fixos nos de produção. Comparação **fora da amostra**, porque o walk-forward só pontua
  a partir de 01/07/2025.
- **Resultado:**

| Braço | Trips fora | Banda 4 h | Antecedência (h) | Episódios FP fora | FP/mês fora | Duty temp · vib · óleo |
|---|---|---|---|---|---|---|
| produção | 3/3 | 2/3 | 13,7 · 8,4 · 3,8 | 23 | 2,877 | 0,021 · 0,140 · 0,004 |
| c0 (treino único, sem filtro 4,5 min) | 3/3 | 2/3 | 13,7 · 8,4 · 3,8 | 23 | 2,877 | 0,023 · 0,144 · 0,004 |
| c1 (walk-forward mensal) | **2/3** | 2/3 | 13,7 · 8,4 · **–** | **26** | **3,252** | 0,024 · **0,072** · 0,003 |

- **Critério:** c1 perde 26/02/2026 e tem 3 episódios de FP a **mais** → **REPROVADO**.
- **Expectativa × resultado:** errei a direção. Previ efeito nulo (±2 episódios); o retreino mensal piora.
- **Leitura:** o retreino mensal corta à metade o duty da vibração fora da amostra (14,4% → 7,2%). Com isso
  some a coincidência que pegava o trip mais tardio (26/02, 3,8 h), e o FP não cai: os episódios de FP não
  vinham do excesso de vibração. O −19% pontual do EXP10c não se transfere para a camada de decisão por canal.
  c0 = produção, ponto a ponto no nível de episódio: o filtro de 4,5 min por canal é redundante (confirma a seção 3).
- **Decisão:** retreino mensal fica fora. Junto com o A4 (o corte do treino muda o resultado), o retreino da
  pipeline é um risco operacional e não uma alavanca de FP: retreinar exige revalidar os 8 trips antes de trocar
  o modelo. Restam C2 (iforest como voto extra), C3 (limiares dos portões por LOEO) e B3 (priorização).

### 14/10/2026 — A4 secundário, data de corte do treino (sp0501, sp0901) — FRÁGIL AO CORTE
- **Script:** `frente_ocsvm/a4_variabilidade.py` (configs pré-registradas em `5728fe4`), semente 42, só
  `AUTOML_OOS_SPLIT_DATE` muda. 6 tasks no Cica, 3 em paralelo por réplica com a fila vazia; ids em
  `frente_ocsvm/a4_tasks.json`. Liberadas pelo usuário em 14/10 ("faça split").
- **Resultado** (mesma camada de decisão de referência; "fora" ainda contado com o corte de 01/07/2025):

| Réplica | Trips | Banda 4 h | FP/mês | Duty temp · vib · óleo | AutoML temp · vib · óleo | Perde |
|---|---|---|---|---|---|---|
| produção (01/07/2025) | 8/8 | 7/8 | 2,885 | 0,023 · 0,105 · 0,005 | p97,5/d1 · p95/d4 · p99,9/d12 | – |
| sp0501 (01/05/2025) | **4/8** | 3/8 | **3,297** | 0,023 · **0,010** · **0,063** | p97,5/d1 · **p99,5/d1** · p99,5/d12 | 27/02/25, 07/04/25, 29/04/25, 04/11/25 |
| sp0901 (01/09/2025) | 8/8 | 7/8 | 3,022 | 0,022 · 0,078 · 0,008 | p97,5/d1 · p95/d4 · p99,5/d12 | – |

- **Expectativa × resultado:** acertei ("pelo menos um dos dois ≤ 6/8"): sp0501 cai para 4/8.
- **Leitura:** a fragilidade é de um lado só. Com o corte em 01/05/2025 o AutoML da vibração troca p95/d4
  por p99,5/d1 e o canal quase se cala (duty 10% → 1%); o óleo, com o mesmo percentil, sobe de 0,8% para
  6,3% (limiar tirado de outro trecho de treino). Sem a vibração, o voto ≥ 2 perde as coincidências que
  pegavam 4 trips, e o óleo ruidoso sobe o FP. Cortando mais tarde (01/09) nada muda além de +2 episódios
  de FP (3,02, perto do efeito mínimo crível). O 8/8 depende de o AutoML da vibração escolher p95/d4, e
  essa escolha não é estável quando a janela de calibração encolhe.
- **Decisão:** A4 fecha como **robusto à semente, frágil ao corte**. Para C1 (retreino mensal), esse é o
  risco central: cada retreino reescolhe os hiperparâmetros do AutoML, e um corte "ruim" derruba a vibração.
  Registrar em cada braço do C1 o percentil/debounce escolhido por canal, não só det e FP/mês.

### 14/10/2026 — A4, variabilidade de treino (5 sementes) — ROBUSTO
- **Script:** `frente_ocsvm/a4_variabilidade.py` · pré-registro `5728fe4` · redução `c0fff2d` · tasks em
  `frente_ocsvm/a4_tasks.json` (15 tasks no Cica, 3 em paralelo por réplica com a fila vazia, ~50 min cada).
- **Resultado** (camada de decisão de referência: canal 4 com 5 tags/24 h, voto ≥ 2, 45 min, 48 h):

| Réplica | Trips | Fora (3) | Banda 4 h | FP/mês | AutoML temp · vib · óleo |
|---|---|---|---|---|---|
| produção (semente 42) | 8/8 | 3/3 | 7/8 | 2,885 | p97,5/d1 · p95/d4 · p99,9/d12 |
| s43 | 8/8 | 3/3 | 7/8 | 2,953 | p97,5/d1 · p95/d4 · p99,5/d12 |
| s44 | 8/8 | 3/3 | 7/8 | 2,816 | idem |
| s45 | 8/8 | 3/3 | 7/8 | 2,885 | idem |
| s46 | 8/8 | 3/3 | 7/8 | 2,816 | idem |
| s47 | 8/8 | 3/3 | 7/8 | 2,953 | idem |

- **Critério:** as 5 sementes ≥ 7/8 e FP/mês em 2,30–3,46 → **ROBUSTO** (mín. 8/8; FP/mês 2,816–2,953,
  ±2,4% da produção). Os três valores de FP/mês distam ~0,069 entre si, ou seja, **±1 episódio** em 14,5 meses.
- **Expectativa × resultado:** errei. Previ FRÁGIL (mediana 7/8, alguma semente ≤ 6/8, perda de 27/02 ou
  17/03). Acertei só que o AutoML troca de percentil: o óleo foi p99,5 nas 5 sementes (a produção, p99,9, é
  a exceção), sem efeito na detecção. Duty dos canais quase idêntico (temp 0,022–0,024; vib 0,101–0,105).
- **Leitura:** a semente da subamostra de 50 mil não move o resultado; o ruído de treino dos OCSVM é ~1
  episódio de FP. O que sustenta o 8/8 continua sendo a seleção das tags do canal 4 (A3), não o treino.
- **Decisão:** C1–C3 podem ser julgados com **uma semente por braço**; efeito mínimo crível ≈ 2 episódios
  (~0,14 FP/mês, ~5%). Os splits (sp0501, sp0901, secundário) ficam **adiados** por decisão do usuário em
  13/10 (seis tasks para pergunta secundária); o próximo uso do Cica é o C1.

### 11–13/10/2026 — A4, envio de 10/10 cancelado; retomado em versão reduzida
- **10/10:** as 21 tasks foram enfileiradas de uma vez e ocuparam os 4 workers do Cica, deixando tasks da
  equipe esperando. **11/10:** paradas e depois **apagadas** do ClearML, sem resultado aproveitado.
  `a4_tasks.json` foi zerado (o `envia` recria os clones). No mesmo dia o Cica estava instável: os agentes
  reiniciavam e a task em curso virava "user abort" em ~30–40 s.
- **Regra de envio (13/10):** **1 task por vez** (um worker só), e só com a fila vazia e ≥ 2 workers
  livres; a próxima só vai quando a anterior terminar. Não dá para fixar um worker: os 4 escutam a fila
  `default`. Se alguma não completar, parar e conferir os workers (`register_time`), `Args/config` e
  `RANDOM_SEED` do clone antes de reenviar.
- **Redução (decidida antes de qualquer resultado):** sementes em ordem, s43 → s47. Como o critério é "alguma
  semente ≤ 6/8 ou FP/mês fora de ±20% → FRÁGIL", **parar na 1ª semente FRÁGIL**, porque o veredito do
  primário já fica decidido. Os splits (sp0501, sp0901, secundário) só rodam se as 5 sementes saírem ROBUSTO.

### 10/10/2026 — B2, janela do canal 4 × filtro × refratário — REPROVADO
- **Script:** `frente_ocsvm/b2_janela_filtro.py` · pré-registro `6843a0e` · 1 min 15 s local · grade em
  `frente_ocsvm/dados/b2_grade.pkl`.
- **Desenho:** 280 configurações (filtro 30–90 min × janela 3–36 h × refratário 24–48 h, 5 tags fixas).
  Escolha **só antes de 01/07/2025**; teste nos 10 meses depois. Não cego: o A3 já tinha mostrado 6 h/36 h.
- **Escolhida:** 45 min, janela **4 h**, refratário **36 h**.

| | Dentro (5 trips) | Fora (3 trips, 10 meses) | Total |
|---|---|---|---|
| Referência (45 min, 24 h, 48 h) | 5/5, 2,894 FP/mês | 3/3, banda 2/3, 2,877 | 8/8, 2,885 |
| Escolhida (45 min, 4 h, 36 h) | 5/5, 1,980 (−32%) | 3/3, banda 3/3, 2,752 (**−4,3%**) | 8/8, 2,404 |

- **Primário:** Δ FP/mês fora −0,125, IC 95% −0,401 a +0,000. Nenhum trip perdido; banda fora 2/3 → 3/3.
- **Secundário:** 140 de 280 configurações dão 8/8; LOEO **7/8** (perde 29/04/25 com janela de 3 h).
- **Critério:** falha em três condições (−4,3% < 10%; IC encosta em 0; LOEO 7/8).
- **Expectativa × resultado:** acertei janela curta, refratário < 48 h, 3/3 fora e IC cruzando 0; errei o
  LOEO (previ 8/8) e subestimei o ganho dentro (previ 2,3–2,6; deu 1,98).
- **Leitura:** o −32% dentro vira −4% fora. Os 2,40–2,61 FP/mês achados no A3 eram em sua maior parte
  ajuste aos meses usados na escolha. Mexer na temporização da decisão não é uma alavanca de custo
  comprovada.
- **Decisão:** manter a referência. A camada de decisão já rendeu o que dava (B1 e B2 reprovados);
  próximo passo útil é B3 (priorização, sem cortar alarme) ou a frente com treino (A4/C1–C3, no Cica).

### 10/10/2026 — B1, debounce do chattering no canal 4 — REPROVADO
- **Script:** `frente_ocsvm/b1_chattering.py` · pré-registro `a74f660` · 28 s local.
- **Tratamento:** ativação só conta se a mesma tag ficou ≥ S h em silêncio; S ∈ {0…24 h}; B1a só em
  `PI_6240319_AL`, B1b nas 5 tags; S escolhido por LOEO.
- **Resultado:** até S = 8 h nada muda (duty do canal 4 46,8% → ≥ 45,9%; FP/mês 2,885). Só S = 24 h mexe.
  Nenhum S perde trip (8/8, fora 3/3, banda 7/8). LOEO 8/8 nos dois braços (sempre S = 24 h).
  B1a: 2,816 FP/mês, Δ −0,069 (IC 95% −0,220 a +0,000). B1b: 2,747, Δ −0,137 (IC −0,338 a +0,000).
- **Critério:** falha no custo (> 2,60 e IC encosta em 0); também não bate os 2,61 que o A3 achou só
  mexendo na temporização.
- **Expectativa × resultado:** acertada (efeito pequeno, rejeitado); errada só no 09/12/25, que não caiu.
- **Leitura:** o canal 4 fica aceso por alarmes recorrentes espaçados de horas, não por rajadas.
  O debounce é seguro (não perde trip) mas inócuo.
- **Decisão:** não adotar. Próximo: B2 (janela do canal 4 × filtro de duração), que o A3 indica ter
  margem (2,61 com as 5 tags) — mas a escolha tem de vir de LOEO.

### 10/10/2026 — A3, LOEO aninhado da camada de decisão — 5/8 COM TAGS LIVRES, 8/8 COM AS 5 FIXAS
- **Script:** `frente_ocsvm/a3_loeo_decisao.py` · pré-registro `7667782` · grade salva em
  `frente_ocsvm/dados/a3_grade.pkl` (fora do git) · 11,5 min local, 6 núcleos.
- **Grade:** 3.968 configurações; 928 dão 8/8. A referência (45 min, 24 h, 48 h, 5 tags; 2,885 FP/mês)
  é só a **118ª mais barata** entre as de 8/8; mínimo 2,404 FP/mês (50 min, 24 h, 48 h, 2 tags).
- **PRIMÁRIO, grade inteira: LOEO 5/8** (banda 4/8), FP/mês mediano das escolhidas 2,404. Perde 27/02/25,
  17/03/25 e 09/12/25. Escolhendo sem o trip, o critério corta tags (fica com 2–3) ou encurta a janela
  para 6 h, e justamente a tag/janela que o trip deixado de fora precisava sai.
- **SECUNDÁRIO, 5 tags fixas: LOEO 8/8** (banda 8/8); as 8 dobras escolhem a mesma (45 min, 6 h, 36 h), a
  2,61 FP/mês.
- **Dependência por tag** (temporização de referência): só `PI_6240319_AL` é indispensável (sem ela 6/8);
  sem `PAL_6240315` ainda 8/8 a 2,54; `PDAL_6240302`, `TC382_05_A` e `PAH_6240319` saem sem perder trip.
  27/02/25 e 26/02/26 só são pegos por subconjuntos com `PI_6240319_AL` (16 de 31); 09/12/25 precisa de
  `PAL_6240315` ou `PDAL_6240302`.
- **Expectativa × resultado:** previ 6–7/8 no primário (deu **5/8**: errada, pior) e 7/8 com tags fixas
  (deu **8/8**: errada, melhor).
- **Leitura (regra pré-registrada, ≤ 5/8):** o 8/8 é em boa parte **seleção do subconjunto de tags** do
  canal 4. Filtro, janela e refratário **não** são o que o sustenta: com as 5 tags dadas, a temporização
  escolhida sem o evento pega todos. E o otimismo está subestimado, porque as 5 tags já vieram de 47.
- **Decisão:** as tags do canal 4 passam a ser o ponto fraco da validação. Tratar no B1 (chattering do
  `PI_6240319_AL`, a única tag indispensável) e nunca reportar o 8/8 sem esta ressalva.

### 10/10/2026 — A2, nulos de acaso — ACIMA DO ACASO NOS TRÊS
- **Script:** `frente_ocsvm/a2_nulo.py` · pré-registro `f6a656b`.
- **N1, instantes sorteados:** chance de "detectar" um instante qualquer q = 0,30. 8/8 p = 7×10⁻⁵;
  fora da amostra 3/3 p = 0,017.
- **N2, paradas comuns como trip:** 28 de 67 (41,8%) contra 8/8 nos trips, p = 0,0018. A pipeline
  separa trip de parada comum (fora da amostra só 3 trips: p = 0,098).
- **N3, canal 4 deslocado:** mediana 4/8, nunca 8/8 em 40 deslocamentos; FP/mês igual (~2,9). Canal 4
  sempre ligado: 8/8 a 4,67 FP/mês.
- **Leitura:** o canal 4 não acrescenta detecção (os OCSVM em OU já pegam 8/8); ele é o **filtro de
  custo** que leva 4,67 → 2,88 FP/mês sem perder trip, e só funciona **alinhado** com os trips.
  Expectativas erradas: N2 (achei que antecipava paradas em geral) e N3 (achei que o alinhamento importava
  pouco).
- **Ressalva aberta:** as 5 tags e a janela de 24 h foram escolhidas vendo os 8 trips → é o A3.

### 10/10/2026 — A1, dentro × fora da amostra
- **Script:** `frente_ocsvm/a1_dentro_fora.py` · pré-registro `f6a656b`.
- **Resultado:** dentro 5/5 (banda 5/5), fora 3/3 (banda 2/3). FP/mês **igual**: 2,89 dentro, 2,88 fora
  (expectativa de subir fora: errada). Mas a composição muda: duty da vibração 6,4% → 14,0%, do canal 4
  60% → 36%. Antecedência mediana 33,8 h dentro, **8,4 h fora**.
- **Leitura:** custo estável no tempo; detecções fora da amostra são tardias.

### 10/10/2026 — A0, reprodução local — PASSOU
- **Script:** `frente_ocsvm/a0_reproducao.py` · pré-registro `9880cfb`.
- **Resultado:** canal 4 recalculado do catálogo em cache idêntico à tabela (0 de 1.895.041 instantes
  diferentes); decisão recalculada idêntica (0 diferentes); 8/8, 42 FP, 25 inconclusivos, 2,8847 FP/mês;
  as 8 antecedências iguais à tabela 7. Roda em ~13 s, sem ClearML.
- **Achado de passagem:** o canal 4 fica aceso **46,8% do tempo em operação**.
- **Correção registrada:** a 1ª rodada "falhou" por comparar a 1ª detecção com a tabela 7 ao minuto (a
  tabela arredonda; a grade é de 30 s). A conferência passou a tolerar 1 min e exigir as antecedências.
- **Decisão:** a base local vale para A1, A2, A3 e B1–B3.
