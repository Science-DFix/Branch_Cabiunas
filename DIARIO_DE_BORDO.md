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
| Limiar próprio para as primeiras horas após a partida (E6) | 8/8, mas só −10% de FP |

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
| **D1** | Disputa entre modelos por grupo de sinal | com a camada de decisão congelada, outro modelo (iforest, AE denso, CNN-1D AE, PCA) mantém 8/8 com ≥ 25% menos FP que a OCSVM? | **sim (Cica)** | **pré-registrado 16/10** (`d1_disputa.py`); **rodadas 1, 1b e 1c feitas 18/10: NENHUMA PASSA** (12 tasks, IDs em `frente_ocsvm/d1_envios.md`); rodada 2 (PCA) feita 18/10: **NENHUMA PASSA** (melhor com 8/8: PCA só na temp, −5%). D1 encerrado no critério; **rodada 1d** (CNN-1D mensal desde 01/2025) feita 18/10: 8/8 nos 3 grupos, mas +25% FP: **NÃO PASSA** (`docs/cnn1d_mensal_por_grupo.md`) |
| **D2** | CNN-1D de vibração como canal extra | somar (A) ou exigir concordância com a OCSVM na vibração (B) corta FP mantendo 8/8? | não | **feito 18/10: NÃO PASSA** (A +64% FP; B −6%, 8/8, antecedência intacta) |
| **D3** | Score contínuo no momento do alerta | magnitude, persistência ou carga separam trip de FP? | não | **feito 18/10: NÃO SEPARA** (AUC 0,40–0,47; fora 0,40–0,61) |
| **D3b** | Score contínuo do CNN-1D mensal | idem D3, com CNN-1D mensal desde 01/2025 | **sim (Cica)** | **feito 19/10: NÃO SEPARA** (AUC 0,34–0,40; fora 0,25–0,56) |
| **E1** | Triagem dos FP por causa (descritiva) | qualidade de dado, partida, rajada, regime pós-27/02 ou alarme crônico explicam FP que nenhum trip tem? | não | **feito 19/10:** qualidade de dado explica só 2 FP; metade dos FP começa na 1ª–2ª h após a partida, mas 3 trips também → candidato a E2 (adiar, não descartar) |
| **E2** | Novidade do alarme do canal 4 | o canal 4 aceso há pouco tempo (N1) ou poucas ativações em 7 dias (N2) separam trip de FP? | não | **feito 19/10: NÃO SEPARA** (AUC 0,55 / 0,34; fora 0,49 / 0,54) |
| **E3** | Curva de compromisso trips × FP/mês | quanto FP se economiza aceitando perder trips (sensibilidade r ≥ k; envelope da grade A3)? | não | **feito 19/10 (descritivo):** k=3 → 7/8, −19% (fora −9%); k=4–12 → 6/8, −27% (fora −13%); A3 só PI+PAL 8/8 −17% (otimista) |
| **E4** | Adiar o alerta após a partida | votos não contam nas primeiras h horas depois da partida; corta FP sem perder trip? | não | **feito 19/10: NÃO** (h=1: −5%, fora 0%; LOEO 7/8; 15 de 17 FP de partida só são adiados) |
| **E5** | Regra de falha de comunicação (higiene) | ignorar o canal do modelo com Comm Fail no grupo nas últimas 6 h mantém 8/8 sem subir FP? | não | **feito 19/10: ADOTADA** (8/8, FP 42 → 41; na pipeline final + contexto do alerta) |
| **G1** | Sinal que manda em cada grupo (oclusão por sensor e família, AUC por feature, janelas na partida) | que sensor/feature acende cada canal nos FP de partida, nos trips e nas partidas normais? | não | **feito 24/10 (descritivo):** nenhum sensor dita temp/vib; óleo pelo PDIT; **FP de partida = janelas de 4 h/24 h atravessando a parada** (92% parada vs 27%) → próximo: janelas que reiniciam na partida (Cica) |
| **H0** | Prévia local das janelas por trecho (modelos congelados) | sem o artefato da janela, o FP de partida some sem perder trip? | não | **feito 24/10: NÃO PROMISSOR** (5/8, fora 1/3; FP 41 → 30, de partida 16 → 3): **os 3 trips de partida (T4, T6, T7) eram pegos pelo mesmo artefato** |
| **H1** | Retreino com janelas por trecho | idem H0, com o modelo treinado nas features novas | **sim (Cica)** | **pré-registrado 24/10** (`h1_janela_trecho.py`, `ce9f96f`); aguardando o Cica |

**Para trazer do detector do Thallys (anotado em 25/10).** O foco agora é reduzir FP na pipeline dele, que reproduz
localmente (entrada de 25/10 na seção 5). Estes itens valem para a nossa:

| # | O que trazer | Por quê / o que dele sustenta | Precisa treinar? |
|---|---|---|---|
| **T-a** | Carga na notificação, na definição dele: percentil do T5 nas 2 h do alerta contra os 30 d anteriores de operação estável | É o único discriminante dele que se sustentou fora da amostra (AUC 0,75, p = 0,051). O nosso D3 usou outra definição (AUC 0,40–0,47). É barato: só os 41 FP e os 8 trips da pipeline final | não | **→ feito 25/10: NÃO** (6/8, FP 26; placebo p 0,31; carga baixa = partida na OCSVM)
| **T-b** | Blackout como **vigilância fechada + estado reiniciado**, não como supressão do alerta | Nele, 6 h levaram 6,06 → 4,35 FP/mês com 8/8. Em nós, a supressão de 6 h e o H0 perdem T4, T6 e T7, que só são pegos pelo artefato de partida. Só vale se um modelo novo pegar esses 3 trips depois de 6 h | sim |
| **T-c** | Robustez em várias bases + placebo de permutação (suprimir o mesmo número de episódios ao acaso) para qualquer regra de supressão | Foi o que separou "funciona na base vista" de "frágil" no EXP60/62 | não |
| **T-d** | Rótulos na tela: "nascido na partida" e "carga alta/baixa" | Contexto sem esconder alarme; entra no `describe_alert_context` | não |
| **T-e** | Lição do refratário fantasma: na dele, um episódio que não chega a alarme arma 72 h de bloqueio e silenciou o 11/04 (n = 19.000) | Na nossa, o filtro de 45 min vem **antes** do refratário (`pipeline_unificada_final.py:213-214`): não temos fantasma. Manter essa ordem | não |
| **T-f** | Lição do corte do retreino: nele, mudar o dia do corte (1 → 8) leva 8/8 → 5/8, e o n do baseline (20.000 → 19.000) leva 7/8 → 5/8 | Igual ao nosso A4 (frágil ao corte do split). Qualquer número publicado precisa vir com a variação ao corte | — |

_(registrado em 25/10/2026 13:15)_

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

### 25/10/2026 — FECHAMENTO da sessão de 24–25/10 (frente Thallys + T-a) — RESUMO E PENDÊNCIAS
- **Pedido do usuário:** "anote tudo o que fizemos, vamos fechar essa seção. até a mensagem para o thallys".
- **O que foi feito**, em ordem. Os detalhes estão nas entradas abaixo e nos docstrings.
  1. **Fonte do relatório 9 → 4 FP:** a branch `exp/modos-de-falha`, com os Datasets `86e3990f` e `b71d7c5f`. A
     versão do pacote não muda o resultado.
  2. **Reprodução local do v2, com treino do zero** (`frente_thallys/reproduz.py`). Os 16 bundles são iguais aos do
     git, e o controle dá 7/8, 9 FP, 1.457 h. O EXP 62 dá 9 → 4. O retreino local é uma exceção autorizada só para o
     PCA do v2.
  3. **T1:** o nosso canal 4 como confirmação. NÃO (FP 9 → 7, placebo p 0,246).
  4. **Base n = 19.000 regerada** (`base_n19000.py`). Confere com o Thallys.
  5. **T2:** comitê de baselines. NÃO (7/8 FP 11).
     - O publicado é o **melhor de 6 retreinos**.
     - A carga reduz o FP em todos os 6.
  6. **T3:** carga baixa OU sem canal 4 → atenção. **ROBUSTO, raspando.**
     - Bases principais: 4/8 FP 4, 6/8 FP 2, 6/8 FP 4.
     - Custos: 17/03 (carga) e 26/02/26 (canal 4).
  7. **T-a:** a carga na OCSVM. NÃO (6/8, FP 26, placebo p 0,31). Na OCSVM, carga baixa = partida.
- **Leituras que ficam:**
  - O v2 publicado está num vale estreito, tanto no corte do retreino quanto no n.
  - A carga é a alavanca estável do v2.
  - Na OCSVM, tudo passa pela partida.
- **Pendências:**
  - **G168 e B800**, para o teste decisivo do T3. Estão no Dataset `86e3990f`, que depende do Cica. Mas talvez existam
    na pasta local do Thallys: `data_local/controle_v2/<base>/serie.csv.gz`, ~2 MB.
    → mensagem pronta em `frente_thallys/mensagem_thallys_2026-10-25.md`. **Ainda não enviada**; o envio é com o
    usuário.
  - **H1** (janelas por trecho na OCSVM): pré-registrado em `ce9f96f`, aguardando o Cica, 1 task por vez.
  - **Ideias não testadas no v2:** refratário só por alarme notificado; reaviso e encerramento de episódio longo.
  - **Push:** 21 commits locais na `exp39_validacao_canal4` ainda **sem push** para `cabiunas/exp/ocsvm-validacao`.
    O push fica para o fluxo combinado, antes de enviar task ao Cica.
- **Onde está:**
  - código em `frente_thallys/` e `frente_ocsvm/ta_carga_thallys.py`;
  - dados em `frente_thallys/dados/`, fora do git (bundles, sinais, membros, logs);
  - clone do Thallys em `~/FRENTE_THALLYS/repro_thallys_regra_carga_9fp_4fp`.

_(registrado em 25/10/2026 22:30)_

### 25/10/2026 — T-a, carga do Thallys como "atenção" na pipeline OCSVM — NÃO
- **Pedido do usuário:** "vamos ... levar a carga para a nossa pipeline (T-a) ... e também a combinação com o nosso
  canal 4".
- **Script:** `frente_ocsvm/ta_carga_thallys.py`, pré-registro `d7b72dc`.
  - Base: a pipeline final com E5 (8/8, 41 FP), a partir dos canais de `point_anomalies_final.csv`.
  - Carga na definição do Thallys, em janela causal [n − 2 h, n], com n = início + 45 min.
  - V1–V3 passam. A V3 confirma que o L daqui reproduz o `L_de` do EXP 62.
- **Resultado:**
  - Braço C (carga): rebaixa 27 de 74 episódios. Fica 6/8, fora 2/3, FP 26 (fora 2,001). Perde T1 (L 0,43) e T7
    (nasce na partida, L 0,19). Placebo p = 0,310. Vizinhança 0/8.
  - Braço C4 (carga OU sem canal 4): 6/8, FP 25, p = 0,248. O canal 4 está aceso na notificação em 72 dos 74
    episódios.
- **Mecanismo:** 16 dos 27 rebaixados nascem a ≤ 2,3 h da partida, com o T5 ainda na rampa. Na OCSVM, carga baixa é,
  em boa parte, "acabou de ligar".
- **Leitura:**
  - A carga separa os episódios do v2, que nascem em regime por causa do blackout de 6 h, mas não os nossos. É
    coerente com o D3 (AUC 0,40–0,47).
  - Não é propriedade dos trips da máquina. É dos episódios de um detector que não olha a partida.
- **Decisão:** não adotar.
  - O T-a sai da lista de "trazer do Thallys".
  - Fica a hipótese de que a carga só serviria na OCSVM depois de resolver a partida (H1, Cica).

_(registrado em 25/10/2026 14:41)_

### 25/10/2026 — Conclusão da frente Thallys até o T3, e pendência do Cica — NOTA
- **Pedido do usuário:** "Anote esse último trecho, é importante!"
- **Conclusão:**
  - A regra "carga baixa OU sem alarme de processo → atenção" (T3) é a melhor candidata até agora para reduzir FP no
    detector do Thallys.
  - É a primeira que passa no critério completo dele em bases que ele não viu: 2 a 4 FP, 0,18–0,36/mês, abaixo da
    meta de ~0,5.
  - A margem é fina: vizinhança, placebo e acréscimo sobre a carga passam exatamente no limite.
  - Os custos são previsíveis e se repetem: o 17/03/25 cai pela carga e o 26/02/26 cai pelo canal 4.
- **Uso:** só no modo sombra, na forma "atenção", nunca como supressão.
- **Pendência:** antes de levar ao Thallys, falta o teste decisivo nas bases G168 e B800, onde a carga dele perdeu o
  17/03.
  - Elas só existem no Dataset `86e3990f` do ClearML. O gerador não está em nenhuma branch.
  - **Depende do Cica voltar:** o Thallys não consegue passá-las por fora.
  - Decisão do usuário: aguardar o Cica e, enquanto isso, levar a carga para a nossa pipeline (T-a).

_(registrado em 25/10/2026 14:33)_

### 25/10/2026 — T3 (frente Thallys), carga baixa OU sem canal 4 → atenção — ROBUSTO, RASPANDO
- **Pedido do usuário:** "pode seguir para T3".
- **Script:** `frente_thallys/t3_carga_ou_canal4.py`, pré-registro `6886520`.
- **Bases:**
  - Principais: os membros do T2 dentro da faixa validada do pacote (19 mil; 21 mil; atraso de 7 d).
  - Estresse: atrasos de 14 e 21 d, que o pacote marcaria para revisão.
  - Já vistas, só relato: publicado, comitê e n = 19.000.
  - V1–V3 passam.
- **Resultado** (sem regra → só carga → combinação):

  | Base | Sem regra | Só carga | Combinação | p do placebo |
  |---|---|---|---|---|
  | 19 mil | 5/8 · 13 | 4/8 · 7 | **4/8 · 4** | 0,100 |
  | 21 mil | 6/8 · 12 | 6/8 · 5 | **6/8 · 2** | 0,004 |
  | atraso 7 d | 7/8 · 16 | 7/8 · 6 | **6/8 · 4** | 0,008 |
  | (estresse) 14 d | 7/8 · 17 | 7/8 · 10 | 7/8 · 8 | 0,010 |
  | (estresse) 21 d | 7/8 · 22 | 7/8 · 13 | 6/8 · 12 | 0,124 |
  | (vista) publicado | 7/8 · 9 | 7/8 · 4 | 7/8 · 2 | — |

- **Critério:**
  - (a) passa: FP ≤ 6 com perda ≤ 1.
  - (b) vizinhança 6/8, no limite. Falham θ 0,6 e a janela de 12 h do canal 4.
  - (c) placebo em 2/3, no limite: o 19 mil dá p = 0,100.
  - (d) acréscimo sobre a carga em 2/3, no limite.
- **Custos que se repetem:**
  - o 17/03/25 cai pela carga (o mesmo do Thallys em G168 e B800);
  - o 26/02/26 cai pelo canal 4 (o mesmo do T1).
- **Leitura:** é a primeira regra que passa no critério completo do Thallys em bases que ele não viu, com 2 a 4 FP
  (0,18–0,36/mês). A margem é fina, e faltam G168 e B800.
- **Para a nossa pipeline:** a carga na definição do Thallys (T-a) entra como próximo teste local da OCSVM.
- **Decisão:** candidata para a sombra, na forma "atenção". Não suprimir. Antes de levar ao Thallys, pedir G168 e B800
  (Dataset `86e3990f`).

_(registrado em 25/10/2026 14:25)_

### 25/10/2026 — T2 (frente Thallys), comitê de baselines — NÃO; a carga se sustenta em 5 bases novas
- **Pedido do usuário:** "pode rodar".
- **Script:** `frente_thallys/t2_comite_baselines.py`, pré-registro `d9e1a20`. Os 6 membros foram feitos com o
  `constroi_bundle.py` do pacote, só variando n e o atraso do baseline. V1–V4 passam.
- **Membros sozinhos:** o ponto publicado (20 mil, corte no dia 1) é o **melhor dos 6**: 7/8 FP 9. Os outros:
  - n = 19 mil: 5/8 FP 13;
  - n = 21 mil: 6/8 FP 12;
  - atraso de 7, 14 e 21 d: 7/8 com FP 16, 17 e 22.
  - Os 9 FP são um vale: qualquer desvio do baseline sobe o FP ou derruba a detecção.
- **Comitê (mediana de 5):** 7/8 FP 11. Na vizinhança, FP de 9 a 15. Não passa no critério.
- **Secundários pré-registrados:**
  - A **carga do EXP 62 reduz o FP nos 6 membros**: 13→7, 9→4, 12→5, 16→6, 17→10, 22→13. A detecção só cai num
    deles (19 mil: 5→4). São 5 bases que o Thallys não viu: a regra de carga é a alavanca estável do v2.
  - Comitê + carga: 7/8 FP 3.
  - Comitê + (T1 OU carga): 7/8 FP 1.
  - Base n = 19.000 + (T1 OU carga): 5/8 FP 3, contra FP 6 só com a carga, sem perder detecção. É a segunda base em
    que a combinação, vista pós-hoc no T1, aparece.
- **Para a nossa pipeline:** a lição é a mesma do A4. O número publicado de um detector com retreino precisa vir com
  a variação ao corte, e ali o publicado era o melhor ponto da vizinhança.
- **Decisão:** o comitê não se adota. Próximo candidato: pré-registrar "carga OU sem canal 4 → atenção" (T3) e
  avaliar nos membros, que ainda não foram vistos com a combinação.

_(registrado em 25/10/2026 14:18)_

### 25/10/2026 — Base n = 19.000 do Thallys regerada localmente — CONFERE
- **Pedido do usuário:** "regere a base n=19000 e pré-registre o 2".
- **Script:** `frente_thallys/base_n19000.py` (`c7f6e3c`).
  - Os escores são o PCA(0,95) fiel em walk-forward mensal, pelo mesmo caminho do EXP 34, sobre a grade
    `8b06a98f`. Levam ~6 s.
  - Substituem t e p no proc, como no `base_n` do EXP 60.
  - O `lightgbm`, que só é importado, foi trocado por um módulo vazio: o PCA não o usa.
- **Conferência:** igual ao Thallys nos três números.
  - n = 20.000 coincide com o controle em 100% dos instantes vigiados.
  - n = 19.000 dá 5/8, FP 11, 1.505 h.
  - Com a carga do EXP 62, dá 5/8, FP 6, 1.360 h.
- **Pré-registro do T2** (comitê de baselines): `frente_thallys/t2_comite_baselines.py`, `d9e1a20`. Ainda não rodou.

_(registrado em 25/10/2026 13:58)_

### 25/10/2026 — T1 (frente Thallys), nosso canal 4 como confirmação dos episódios do v2 — NÃO
- **Pedido do usuário:** "pode seguir com o 1, pré-registre antes!"
- **Script:** `frente_thallys/t1_canal4_no_v2.py`, pré-registro `e0c91dc`. Roda local, sobre o controle refeito com
  treino do zero (`frente_thallys/reproduz.py`).
- **Regra:** um episódio do v2 sem alarme de processo (as 5 tags do canal 4, janela de 24 h) na janela de notificação
  vira "atenção". O fuso do catálogo foi conferido antes: lag 0 h contra as 159 partidas.
- **Resultado:** 7/8 FP 9 → **7/8 FP 7** (0,628/mês); início 6 → 5; banda 5 → 4.
  - Não passa no critério (FP ≤ 6).
  - Placebo de permutação: p = 0,246, ou seja, não é específico.
  - O canal 4 confirma 15 dos 18 episódios, inclusive os 6 FP de borda.
  - Somem os FP de 26/10/25 e 16/01/26. O episódio que nasce na janela do trip de 26/02/26 é rebaixado: o trip segue
    detectado pelo episódio travado de 29/01, mas perde o início.
- **Secundário, pós-hoc:** T1 OU carga do EXP 62 dá 7/8 FP 2. Só vale testar nas outras bases, porque nelas o EXP 62
  perde o 17/03.
- **Para a nossa pipeline:** o canal 4 é filtro de custo só onde os modelos acendem sozinhos. No v2 ele quase sempre
  já está aceso.
- **Decisão:** não adotar. Próximo da lista: o comitê de baselines (item 2).

_(registrado em 25/10/2026 13:27)_

### 25/10/2026 — Detector do Thallys reproduzido localmente, com treino do zero — REPRODUZ
- **Pedido do usuário:** "nós conseguimos reproduzir localmente?" e "minha ideia é treinar o modelo mensal para os
  sinais".
- **Exceção autorizada:** o retreino mensal do v2 é PCA, ~1,5 s por mês, e roda local enquanto o Cica está fora.
  Não vale para OCSVM nem para rede neural.
- **Entradas, só locais:**
  - Bruto: Dataset `a97ba56b` (30 s, UTC, texto → NaN).
  - Pacote: `git archive` da `feat/pacote-consolidado`.
  - Registro: `trips_conhecidos` do último bundle (9 trips).
  - Grade: Dataset `8b06a98f`.
  - Scripts no scratchpad da sessão `919a7d85`: `gera_sinais.py`, `controle_atual.py`, `exp62_atual.py`.
- **Resultado:**
  - Os 16 bundles mensais refeitos com `constroi_bundle.py` são iguais aos do git (diferença máxima ~1e-14).
  - Sinais: t, p e sp idênticos à impressão digital; vb difere em 2 de 390.762 instantes, por causa do dado de 2024.
  - Controle: 7/8, 9 FP, 0,807/mês, 1.457 h, igual ao publicado.
  - EXP62 na base atual: 9 → 4 FP com 7/8 (1.319 h; 0,359/mês), placebo p = 0,018, vizinhança igual à do PDF.
- **Falta:** as bases G168 e B800, cujo gerador está fora do git, e a n = 19.000, que dá para regerar pelo EXP34.
- **Decisão:** a redução de FP passa a ser estudada sobre a pipeline dele. O que dá para trazer para a nossa está
  na seção 4 (T-a a T-f).

_(registrado em 25/10/2026 13:15)_

### 24/10/2026 — Fonte do relatório do Thallys (9 → 4 FP) — VERIFICADO: `exp/modos-de-falha` + Datasets do ClearML
- **Pedido do usuário:** "veja se voce consegue averiguar se foi essa branch responsaveis por fonecer essas
  informações" e "voce consegue checar essas informações aqui mesmo? ou realmente precisa do thallys?".
- **Objeto:** `Thallys_doc/RELATORIO_9FP_PARA_4FP_TC33003A.pdf` (21/10, 18:27). Clone em
  `~/FRENTE_THALLYS/repro_thallys_regra_carga_9fp_4fp` (SIMPred-Petrobras/cabiunas-models, `exp/modos-de-falha`,
  `d77de7d`). Só leitura de git; nada rodado.
- **A branch é a `exp/modos-de-falha`, não a `feat/pacote-consolidado`:**
  - Os 6 commits dos EXP60–62 (`06d80b8` … `50eace8`) são de 21/10, entre 14:12 e 15:18, antes do PDF.
  - Os blocos de RESULTADO de `experiments/60_borda_adiada.py`, `61_carga_na_notificacao.py` e
    `62_regra_de_carga.py` batem número a número com o PDF.
  - Os workers 60–62 usam os insumos `86e3990f`, citados no PDF.
  - A `feat/pacote-consolidado` (08/10) não tem `experiments/`: é o pacote de produção.
- **A versão do pacote não muda o 9 → 4:**
  - O Dataset `b71d7c5f` é a pasta `estrutura_pra_prod/Cabiunas` da máquina do Thallys, fora do git.
  - Entre as branches do pacote, `modelos/` é idêntico (`37d0830c`). As funções `prever`, `_episodios`,
    `carregar_modelos`, `carregar_dados` e `carregar_trips` também são iguais (comparação da AST, sem comentários).
  - Só o `preprocessar` muda: a correção do EWMA sobre buraco, `197bff5` de 07/10.
    - O V3 do EXP54 (rodado localmente em 20/10, "NaN diferentes 0") indica que ele usou a versão com a correção, a
      mesma da `pacote-consolidado`.
    - Ressalva: isso pressupõe um buraco nos dados na janela testada.
  - Os EXP60–62 leem os sinais e o controle prontos dos insumos (`sinais_v2_8ev.parquet` e
    `controle/serie.csv.gz`). Do pacote, só usam `carregar_modelos` e `_episodios`.
- **Fora do git:**
  - O texto do relatório de 21/10.
  - O conversor para PDF (Chrome headless) só entrou em 22/10 (`exp/sota` `5efc068`, `experiments/sota/pdf.py`).
  - A tabela dos 9 FP junta o EXP27b (em UTC) com os EXP60/61 (em hora local).
- **Depende do Cica:** baixar `86e3990f` e `b71d7c5f` e rodar os EXP60–62. Não estão no cache local.
- **Decisão:** reproduzir com a `exp/modos-de-falha` mais os Datasets, sem consultar o Thallys. Só seria preciso
  falar com ele para regenerar os sinais do zero.

### 24/10/2026 — H0, prévia local das janelas por trecho (modelos congelados) — NÃO PROMISSOR: PERDE OS 3 TRIPS DE PARTIDA
- **Pedido do usuário:** "deixe guardado para o cica e vamos testar localmente".
- **Script:** `frente_ocsvm/h0_previa_janela_trecho.py`, pré-registro `ce9f96f`. As correções `4fcc195` (estado com o
  sensor-alvo) e `09f9922` (memória) vieram antes de qualquer resultado.
- **O que faz:** os 3 OCSVM de produção, sem retreino, recebem as features com janelas que não atravessam a troca de
  estado (`DERIVED_RESET_AT_STATE_CHANGE`). Normalização, limiar, portões e decisão com E5 ficam iguais. A base
  reproduz 8/8 e 41 FP.
- **Resultado:** 5/8 (fora 1/3), 30 FP, 2,061 FP/mês (fora 2,251).
  - FP de partida: 16 → 3. Aparecem 5 FP novos, nenhum na partida.
  - Perde exatamente T4, T6 e T7, os 3 trips que começam na partida. O canal que votava neles apaga: T4 temperatura
    1,00 → 0,00; T6 vibração 1,00 → 0,75; T7 vibração 1,00 → 0,00. Os outros 5 trips ficam idênticos.
- **Leitura:**
  - Nos 3 trips de partida, o voto do modelo é o mesmo artefato dos FP de partida: o modelo vê a partida, não um
    precursor diferente.
  - Dois desses trips estão fora da amostra, então o 3/3 fora depende do artefato.
  - Os 3 trips caem 8–21 h depois da partida. A partida é o período de risco, e "partida + canal 4" acerta esses 3 ao
    custo de ~13 FP.
- **Expectativa:** certa nos FP de partida e nos FP novos; errada no total (previ 33–38) e nos trips (previ 8/8).
- **Decisão:** nada adotado. O H1 (retreino, Cica) continua guardado, com o pré-registro intacto. Agora se espera
  que ele também perca T4, T6 e T7.

### 24/10/2026 — G1, sinal que manda em cada grupo — DESCRITIVO: ARTEFATO DAS JANELAS LONGAS
- **Pedido do usuário:** "vamos fazer diagnostico local, encontrar nossas galinha dos ovos de ouro em cada sinal!
  devemos usar dados que são usados no cica".
- **Script:** `frente_ocsvm/g1_sinal_dominante.py`, pré-registro `761e7c1`. A base é a do Cica: Dataset `a97ba56b`,
  configs `6842ac7`, features de produção e clipping do F1. Os artefatos vêm do cache local, porque o Cica está fora
  do ar desde 24/10. Conferência com o score do E6: erro 0,0 nos 3 grupos.
- **Oclusão por sensor:** nenhum sensor dita a temperatura (T5 e TC382 são redundantes) nem a vibração (peso
  espalhado; o que mais pesa é o `TV_354Y_A`). No óleo, o `PDIT_0305` dita.
- **Oclusão por família:** manda a textura de 4 h (curtose, assimetria, crista), até na temperatura.
- **Janelas longas:** atravessam a parada.
  - A 2 h da partida, a janela de 24 h está 92% fora de "on" nos FP de partida, contra 27% nas partidas sem alerta.
  - O |z| das escalas de 4 h e 24 h vai a 5–33× o do regime e só volta em 24 h.
  - A posteriori: parada ≥ 12 h antes de 11 de 16 FP de partida e de 12 de 49 partidas sem alerta. Também antes do
    trip de 04/11/2025, que veio depois de 79 h parado.
- **Feature a feature:** 15 da vibração (desvio e crista de 1 h dos mancais 4 e 5) e 12 do óleo (`PDIT_0305`, com só
  2 trips) passam nos dois AUC ≥ 0,75. Não há controle de múltiplas comparações, então é hipótese.
- **Expectativa:** certa em 2 de 5. Certa: vibração sem sensor dominante; o artefato existe. Errada: temperatura
  redundante, óleo pelo PDIT, e muitas features passaram, não no máximo 1.
- **Decisão:** nada adotado. Próximos itens no README (retreino):
  - janelas que reiniciam na partida, para corrigir a feature em vez de suprimir;
  - depois, as features por origem com escolha fixa.

### 21/10/2026 — FP de partida em detalhe (relatório dos FP, seção 6) — DESCRITIVO
- **Pedido do usuário:** "explore esses falso positivos em uma pagina do nosso relatorio, tempo médio o porque regras
  proximo do religamento nao funcionaram tão bem".
- **Script:** `relatorio_falsos_positivos/gera_partida.py`. Pipeline final com E5 (41 FP) e o score do E6 em cache.
- **Tempo:** 15 FP a ≤ 2 h de "on" (mediana 1,1 h, ~2 h da ignição). Duram mais que os outros: mediana 1,8 h
  contra 1,1 h, e até 9,7 h. Na mesma janela começam T4, T6 e T7 (1,9 / 1,3 / 0,8 h).
- **Por quê:**
  - O canal 4 fica aceso em 98% das primeiras 6 h, então o voto vira "1 modelo". 70% das partidas fora do
    refratário geram alerta.
  - O regime dos modelos dura 3–4 h, por isso adiar não basta (E4).
  - O score da OCSVM RBF satura: o limiar do E6 fica a 84–97% do teto, e trips e FP de partida ficam perto do teto.
- **Decisão:** nada adotado. Novos pontos a explorar (README): score sem teto, que exige retreino e deve ser checado
  antes no PCA do D1, e a regra S1 do canal 4, para os dados novos.

### 19/10/2026 — E6, limiar próprio para o regime pós-partida — NÃO
- **Pedido do usuário:** "pode seguir" (item 4 do plano, o regime pós-partida no modelo).
- **Script:** `frente_ocsvm/e6_limiar_pos_partida.py`, pré-registro `6944f8a`. Local, com os modelos congelados do F1.
- **Medido antes do pré-registro:**
  - O regime pós-partida **já está** no normal de treino: 67% das horas 1–3 após a partida (temperatura e vibração),
    100% no óleo. Mas é raro (~6% do treino nas primeiras 6 h). "Incluir no treino", o plano original, não muda nada.
  - O canal 4 acende na partida: `PI_6240319_AL` a ≤ 3 h de 52% das 83 partidas (base 9,6%); `PAL_6240315`, 66%
    (base 6,2%).
- **Tratamento:** nas primeiras H = 6 h após a partida, o limiar de cada canal vira o mesmo percentil da produção,
  calculado só sobre o normal de treino desse trecho.
- **1ª rodada inválida:** a conferência dos limiares falhou. Na produção, o score de treino usa o clipping do próprio
  normal, e o alerta usa o da série inteira. Corrigido; a regra não mudou.
- **Resultado:** os limiares sobem 3,1× (temperatura), 74× (vibração) e 1,05× (óleo).
  - 8/8 e 3/3, com antecedências iguais.
  - FP 41 → 37 (−10%); fora da amostra 2,752 → 2,627 (−5%; limite 2,158). Iguais com H = 3 e 12 h.
  - Dos 16 FP de partida, 8 somem.
- **Expectativa:** acertei o veredito; errei o motivo (previ perder T4/T6/T7, não perdeu nenhum) e o tamanho.
- **Decisão:** não adotar. Retreino reforçando o regime ficaria aquém deste limiar, que é o caso extremo. Não gastar
  Cica com isso sem o tipo de partida.

### 19/10/2026 — F1, pipeline congelada para dados novos — FERRAMENTA PRONTA, HIPÓTESES PRÉ-REGISTRADAS
- **Pedido do usuário:** "pode seguir" (passo 2 do plano). Script `frente_ocsvm/f1_congelado.py`, commit `3d10b16`.
- **O que faz:** aplica os 3 modelos de produção (`model.pkl`, normalização e limiar das tasks b494d457 / 9c344688 /
  fc4123fb; configs do commit `6842ac7`) e a camada de decisão com E5 a qualquer série, sem retreino.
  - As duas estatísticas que a produção tira da série inteira ficam congeladas em `frente_ocsvm/f1_congelado.json`:
    os quantis de clipping e o limiar de transiente do RUNNING_A.
- **Conferência:** CONFERE. 0 instantes divergentes nos 3 canais (alerta e estado); 8/8, 41 FP, 25 inconclusivos.
  ~27 min e ~7 GB.
- **Pré-registrado para os dados depois de 20/04/2026:**
  - P1 (≥ 75% dos trips novos) e P2 (FP/mês ≤ 4,3).
  - S1: canal 4 sem `PI_6240319_AL`.
  - S2: prioridade "dois modelos".
  - S3: prioridade "alerta repetido em 7 d".
  - D2-B fica fora: precisa de retreino do CNN-1D.
- **Mudança em `src/`:** `build_operational_state` ganhou `transient_diff_abs_threshold` (opcional; o padrão não muda).

### 19/10/2026 — Pedido de dados (item 3) — PRONTO, A ENVIAR
- **Pedido do usuário:** "passe para item 3".
- **Documento:** `pedido_de_dados/pedido_de_dados.pdf`, 3 páginas: por que pedimos, os 7 pedidos por prioridade e o que
  faremos com cada resposta.
- **Planilha:** `frente_ocsvm/revisao/pedido_dados_TC33003A.xlsx`, gerada por `frente_ocsvm/pedido_dados.py`. Revisão
  cega: nada indica o que o modelo acha de cada linha.
  - Abas: 83 partidas (tipo), 31 paradas (causa), 41 alertas sem trip da pipeline final com E5 (o que houve), 8
    perguntas pontuais e os 8 trips como referência.
  - Substitui a `paradas_para_revisao.xlsx` de 15/10, que não foi enviada.
- **Achados de passagem:**
  - `PI_6240319_AL` = "Falha PI Gás p/ Motor de Partida" (`PAH_6240319` é do mesmo gás). O principal alarme do canal 4
    é ligado à partida.
  - A falha de comunicação de 14/01/2026 16:18 coincide com a rajada de alarmes de temperatura do mesmo minuto.
- **Falta:** o usuário enviar à equipe da planta.

### 19/10/2026 — E5, regra de falha de comunicação, e contexto do alerta — ADOTADOS
- **Pedido do usuário:** "vamos de item 2". Pré-registro `d2cf017`, script `frente_ocsvm/e5_falha_comunicacao.py`.
- **E5:** com 6 h, 8/8 e antecedências iguais; FP 42 → 41 (2,885 → 2,816/mês). Somem os FP de 14 e 17/01/2026, mas
  surge um em 15/01, que o refratário do de 14/01 cobria.
- **Na pipeline final** (`scripts/pipeline_unificada_final.py`):
  - a regra entra antes do voto (`compute_comm_fail_mask`);
  - `describe_alert_context` grava a causa provável de cada episódio, só com dados até o momento do alerta.
  - Rodada real: 8/8, 2,82 FP/mês, 25 inconclusivos. Testes: 20/20.
- **Contexto na produção:** pós-partida aparece em 3 det, 16 FP e 10 inc; repetição em 5 det, 16 FP e 10 inc.
  Descreve, mas não separa.
  - Sem ordem de prioridade, há "outro alerta nos 7 dias antes" em 6/8 trips e 22/41 FP. É uma hipótese para dados novos.
- **README:** nova seção "Falsos positivos: o que sabemos e o que explorar", com os pontos para retreino.

### 19/10/2026 — E4, adiar o alerta após a partida — NÃO
- **Pedido do usuário:** "vamos de item 1". Pré-registro `f7a2895`, script `frente_ocsvm/e4_adia_partida.py`.
- **Resultado:** com h = 1 h, 8/8 e −5% de FP/mês (fora 0%); LOEO 7/8. Com h ≥ 3 h, perde T4.
- **Mecanismo:** dos 17 FP de partida, 15 reaparecem até 8 h depois. A anomalia pós-partida é um regime de horas, não um transitório.
- **Expectativa:** errada no tamanho (previ −15 a −25%), certa no veredito.
- **Próximo:** se o usuário pedir, atacar no modelo (incluir no normal de treino as horas pós-partida, ou um limiar próprio para elas), com retreino no Cica.

### 19/10/2026 — Relatório dedicado aos FP
- **Pedido do usuário:** juntar o levantamento dos FP num segundo LaTeX, com título informativo.
- **Pasta:** `relatorio_falsos_positivos/` ("De onde vêm os falsos positivos da detecção de trips da TC-33003A").
  Figuras em `gera_figuras_fp.py`; usa `frente_ocsvm/dados/e1_triagem.csv` e algumas figuras do primeiro relatório.
- **Conteúdo:** composição dos episódios, as 9 hipóteses já testadas, triagem por causa com a categoria principal de
  cada FP, curva de compromisso, dados faltantes e apêndice com os 36 FP da OCSVM.
  - Categorias da OCSVM: partida (≤ 2 h) 16; série de vibração 01–02/2026 5; regime pós-27/02 4; crônico 4;
    Comm Fail 2; rajada 1; sem causa 4.
- **Primeiro relatório:** as subseções E1–E3 saíram de lá; ficou só uma remissão para o relatório novo.

### 19/10/2026 — E3, curva de compromisso — DESCRITIVO
- **Pedido do usuário:** "vamos para item 3". Script `frente_ocsvm/e3_curva_compromisso.py`, figura
  `relatorio_ocsvm_cnn1d_mensal/figs/curva_compromisso.png`, subseção nova no relatório.
- **Sensibilidade** (canal do modelo só conta se score/limiar ≥ k):
  - k = 3: 7/8 (perde T2), 2,335 FP/mês (−19%); fora 3/3 e −9%.
  - k = 4–12: 6/8, −27%; fora −13%.
  - k = 16: 5/8, −36%; fora −13%.
  - Voto ≥ 3: 0/8.
- **Envelope A3** (otimista, LOEO 5/8): 8/8 só com PI_6240319_AL + PAL_6240315 dá 2,404 (−17%), fora −22%.
- **Leitura:** não há ponto de graça. Fora da amostra os trips resistem até k = 16, mas os FP também. O ponto é decisão da operação.

### 19/10/2026 — E2, novidade do alarme do canal 4 — NÃO SEPARA
- **Pedido do usuário:** "vamos para item 2". Pré-registro `208f86d`, script `frente_ocsvm/e2_novidade_alarme.py`.
- **Medidas** no momento do alerta (OCSVM de produção, período inteiro, 8 det × 42 FP):
  - N1: há quanto tempo o canal 4 está aceso. Mediana 3,8 h nas detecções e 14,6 h nos FP; AUC 0,55, p 0,35.
  - N2: quantas ativações das 5 tags em 7 dias. Mediana 9,5 nas detecções e 7,5 nos FP; AUC 0,34, ou seja, contra a hipótese.
- **Leitura:** os trips não chegam com alarme "novo". Três deles (16/03/2025, 27/04/2025 e 03/11/2025) têm o canal 4 aceso
  há 26–74 h. Expectativa certa (nulo); a Etapa 2 não roda.

### 19/10/2026 — E1, triagem dos FP por causa — DESCRITIVO
- **Pedido do usuário:** "comece pelo item 1" (descobrir por que cada FP aconteceu, sem olhar os trips para escolher regra).
- **Script:** `frente_ocsvm/e1_triagem_fp.py` (local, usa o `cache.pkl` do relatório). Tabela por episódio em
  `frente_ocsvm/dados/e1_triagem.csv` (fora do git).
- **Resultado** (desde 01/2025; OCSVM 36 FP | CNN-1D 45 FP):
  - "Comm Fail"/"Out of Serv" nos sensores do grupo aceso: 2 | 2 FP, nenhum trip. É a única causa de dado limpa, mas é pequena.
  - Trecho de reta (interpolação): nenhum episódio. Sensor parado ≥ 10 min: não limpa (o PDIT_0305 fica parado 32% do tempo).
  - **Partida:** 18 | 20 FP começam até 6 h depois de a máquina voltar a "on", 15 | 15 deles na 1ª hora. Mas T4, T6 e T7
    também (1,1 / 0,6 / 0,0 h após a partida, com antecedência de 20 / 14 / 8 h).
  - Rajada de temperatura e alarme crônico não são limpas. Os 6 | 5 FP depois de 27/02/2026 não são testáveis (não há trip depois do T8).
  - Sem causa nenhuma: 9 | 17, concentrados na série de vibração de 31/01–14/02/2026.
- **Decisão:** nenhuma regra sai daqui. Candidato a teste pré-registrado (E2): **adiar** o alerta nas primeiras horas
  após a partida (sem descartar); o trip só se mantém se o voto continuar aceso depois da janela.

### 18–19/10/2026 — D3b, score contínuo do CNN-1D mensal — NÃO SEPARA
- **Pedido do usuário:** "faça para CNN1D".
- **Código:** o walk-forward passa a salvar `sequence_scores_all.csv` com r = score/limiar do próprio mês (limiar 1;
  `automl_pipeline.py`). As flags não mudam.
  - Teste mínimo local: iforest em 6 meses, só o grupo de temperatura. O CSV sai a partir do início do
    walk-forward, e os 137 pontos marcados têm r > 1. pytest 15/15.
- **Braço:** `cnn1d_mensal_jan_sc` tem config idêntica ao `cnn1d_mensal_jan` (diff vazio) e é rodado de novo neste
  commit. São 3 tasks, uma por vez, ~8 h.
- **Script:** `frente_ocsvm/d3b_score_cnn1d.py`. As medidas, a Etapa 1, a Etapa 2 e o critério são os da D3, no
  recorte de 01/2025 ao fim. Expectativa: nulo → NÃO.
- **Execução:** temp `ffe9825c` e óleo `de4e9963` em paralelo, a pedido do usuário; vib `dc73ba24` sozinha, por
  último. As três ficaram `completed` e com o arquivo de score (conferido a cada 30 min).
  - Reprodução da 1d exata: 8 detecções, 45 FP, 23 inconclusivos e as mesmas antecedências.
- **Resultado:** nenhuma medida separa.
  - Magnitude: AUC 0,34 (p 0,92), mediana de r 2,6 nas detecções contra 3,4 nos FP.
  - Persistência: AUC 0,39 (p 0,84).
  - Carga: AUC 0,40 (p 0,18).
  - Fora da amostra (descritivo): 0,25, 0,56 e 0,38.
- **Leitura:** como na OCSVM, os FP do CNN-1D têm score maior e mais persistente que os trips. Nenhuma das duas
  famílias traz no score a informação que separaria trip de FP.

### 18/10/2026 — D3, o score contínuo separa trip de FP? — NÃO SEPARA
- **Script:** `frente_ocsvm/d3_score_continuo.py` (pré-registro no docstring, commitado antes de rodar). Local, sem
  treino, com a OCSVM de produção, a única família com score contínuo salvo. O CNN-1D mensal descarta o score no
  walk-forward.
- **Pedido do usuário:** item 2 do relatório: usar informação que não seja "qual alarme tocou".
- **Medidas no momento do alerta** (t0 + 45 min, só dados até ali):
  - F1 magnitude, a mediana de score/limiar no voto;
  - F2 persistência, a fração das 6 h anteriores acima do limiar;
  - F3 mudança de carga, pelo T5_AVG_A.
- **Etapas:**
  - Etapa 1: Mann-Whitney, 8 detecções × 42 FP, Bonferroni 0,05/3.
  - Etapa 2, só para o que separar: portão por trecho de voto, com corte escolhido por LOEO.
- **Expectativa:** nada separa na Etapa 1 → NÃO.
- **Resultado** (pré-registro `eef43f6`): nenhuma medida separa; a Etapa 2 não roda.
  - Magnitude: AUC 0,41 (p 0,78), mediana de score/limiar 26 nas detecções contra 44 nos FP.
  - Persistência: AUC 0,40 (p 0,83).
  - Carga: AUC 0,47 (p 0,40).
  - Fora da amostra (descritivo, 3 trips): AUC 0,56, 0,61 e 0,40.
- **Leitura:** os FP têm score tão alto e tão persistente quanto os trips. O score contínuo da OCSVM não traz a
  informação que faltava.
  - A posteriori: a persistência é baixa nos 5 trips de dentro e alta nos 3 de fora, o mesmo padrão do B3. Mesmo
    fora, os FP também são persistentes.
  - A coluna AUC fora saiu NaN na 1ª execução (faltava dropna). Foi corrigida, e o veredito não muda.

### 18/10/2026 — Relatório LaTeX: OCSVM × CNN-1D mensal e o limite dos dados
- **Arquivo:** `relatorio_ocsvm_cnn1d_mensal/relatorio_ocsvm_cnn1d_mensal.pdf` (24 p.). Figuras, tabelas e números
  vêm de `relatorio_ocsvm_cnn1d_mensal/gera_figuras.py`, que não treina nada.
- **Conteúdo:**
  - canais, períodos e treino das duas famílias, com o C1, as rodadas 1b/1c/1d e a D2;
  - série temporal conjunta e zoom dos 8 trips;
  - diagnóstico dos 62 episódios: 60 nascem de PI_6240319_AL/PAL_6240315 + 1 modelo, e trips e FP têm a mesma
    composição;
  - problemas dos dados e o que é preciso obter.
- **Achados novos registrados no relatório:**
  - a antecedência de 3,8 h do T8 é efeito do refratário: um episódio que começou 51,8 h antes (inconclusivo)
    bloqueou alertas até 26/02 11:47;
  - PI_6240319_AL para de alarmar depois de 27/02/2026;
  - há 19 rajadas de alarmes de temperatura no mesmo segundo, uma delas perto do T4;
  - o CSV traz valores "Comm Fail"/"Out of Serv" no lugar da medida, e ~17% do PI_0308 está em trechos interpolados.

### 18/10/2026 — D2, CNN-1D de vibração como canal extra da OCSVM — NÃO PASSA
- **Script:** `frente_ocsvm/d2_canal_extra.py` (pré-registro no docstring, commitado antes de rodar). Local, sem
  treino: usa `cnn1d_mensal_jan_vib` (`892e93c1`) somado à produção.
- **Pedido do usuário:** testar o CNN-1D de vibração como canal extra, não como substituto.
- **Diagnóstico descritivo, feito antes do pré-registro** (produção, 01/2025 ao fim, 62 episódios): 61 têm o canal 4
  aceso, quase sempre por PI_6240319_AL e/ou PAL_6240315. São as duas tags mais recorrentes (409 e 326 ativações
  desde 07/2024; só ~8% delas perto de trip). FP e detecção têm a mesma composição: canal 4 + 1 modelo, quase sempre
  a vibração.
- **Braços:** A = 5º canal, voto ≥ 2 de 5; B = a vibração só vota se OCSVM e CNN-1D acenderem juntos. Critério da
  D1, no recorte de 01/2025 ao fim e fora da amostra.
- **Expectativa:** A mantém 8/8 e sobe o FP; B corta FP, mas arrisca o trip de 26/02/2026. Os dois provavelmente NÃO.
- **Resultado** (commit do pré-registro `37e0ca1`): ref 8/8, 36 FP (3,051) e fora 3/3, 23 FP (2,877).
  **A:** 8/8, 59 FP (5,001) e fora 39 FP: +64%. **B:** 8/8, 34 FP (2,882) e fora 21 FP (2,627): −6% e −9%, com as
  antecedências idênticas à produção. Corte: 2,288 e 2,158. **Nenhum passa.**
- **Leitura:** somar um canal só soma votos. A confirmação é segura (não perde trip nem antecedência), mas corta
  pouco, porque as duas famílias concordam na vibração. Errei em B: o 26/02 sobreviveu, e o corte ficou bem menor
  que o previsto.

### 18/10/2026 — D1 rodada 1d, CNN-1D mensal desde 01/2025 (os 8 trips) — NÃO PASSA
- **Pergunta do usuário:** o CNN-1D mensal pega os 8 trips com antecedência? Na 1c o walk-forward começava em
  01/07/2025, então só os 3 trips fora eram julgáveis (antes disso o braço não emite flag).
- **Pré-registro:** docstring de `frente_ocsvm/d1_disputa.py`, commit `45b8b6c` (antes do código e do envio).
  `cnn1d_mensal_jan` = cnn1d_mensal com `WALKFORWARD_START_DATE` = 2025-01-01 (1º retreino com o normal de 07–12/2024).
  O AutoML continua escolhendo percentil/debounce só com alarmes depois de 01/07/2025, então os 5 trips de
  fev–abr/2025 entram fora da amostra para este braço. Recorte: 01/2025 ao fim, com a OCSVM no mesmo recorte.
  Expectativa: 6 a 8 de 8, FP maior que a OCSVM, NÃO pelo critério.
- **Código:** `WALKFORWARD_START_DATE` no config (None = split, comportamento antigo; as 18 configs anteriores não
  mudam) e no `run_automl_group`. pytest 15/15; as 3 configs carregam com 16 retreinos mensais.
- **Envio:** temp, vib, óleo, 1 por vez; IDs em `frente_ocsvm/d1_envios.md`. Tempo estimado: vib ~3,5 h, temp e
  óleo ~1,5 h cada.
- **Execução:** temp `bbc71b16` (1h10), óleo `aab72b80` (3h45, enviado em paralelo com a vib, com ok do usuário),
  vib `892e93c1` (2h50, sozinha). A 1ª vib, `f0701a16`, ficou `completed` mas sem artefatos do grupo: o filho do
  walk-forward morreu no 16º mês ("process pool terminated abruptly"). Os 4 workers são a mesma máquina e a mesma
  GPU; o paralelo com o óleo é a causa provável. Regra: CNN-1D nunca em paralelo, e conferir os artefatos do grupo
  depois do `completed`.
- **Resultado** (`d1_jan.py`; tabela trip a trip em `docs/cnn1d_mensal_por_grupo.md`, seção 7): **nenhuma passa**.
  - Recorte 01/2025→ (OCSVM 8/8, 36 FP, 3,051): todos 8/8, 45 (3,814); só vib 8/8, 42 (3,560); só óleo 8/8, 36
    (= produção); só temp **7/8** (perde 29/04/2025), 44 (3,729).
  - Fora: todos 3/3, 26 FP; só vib 24; só óleo 23; só temp 27. Corte 2,288 / 2,158.
  - **Pergunta respondida:** o CNN-1D mensal pega os 8 trips com antecedência, também os 5 de 2025 fora da amostra.
    Ganha em 09/12/2025 (8,4 → 21,1 h) e perde em 07/04 (43,2 → 31,4 h) e 29/04/2025 (36,7 → 12,4 h), mas tem +25% de FP.
- **Decisão:** a OCSVM segue em produção. O D1 fecha também para o CNN-1D mensal; reduzir FP fica para a camada de
  decisão. Documentação separada do CNN-1D: `docs/cnn1d_mensal_por_grupo.md`.

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
