# Diário de bordo — pipeline OCSVM (3 canais + alarme), TC-33003A

Aberto em 10/10/2026. **Daqui pra frente, este é o documento vivo da frente OCSVM.** O histórico de
agosto e setembro continua onde está e serve de consulta, não de plano:

| Para consultar | Onde |
|---|---|
| Tudo do EXP10 ao EXP39, com números e tasks | `docs/analise_automl_exp10.md` |
| Relatório operacional (8/8, 2,88 FP/mês) | `relatorio_pipeline_operacional/relatorio_pipeline_operacional.tex` (commit `c5ec664`) |
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
| **A4** | Variabilidade de treino | deslocar o split e a amostra de 50 mil muda o resultado? | **sim (Cica)** | a fazer |
| **B1** | Chattering do `PI_6240319_AL` | contar só ativações novas após silêncio mínimo reduz o duty do canal 4 sem perder trips? | não | **pré-registrado 10/10** (`b1_chattering.py`) |
| **B2** | Janela do canal 4 × filtro de duração | a janela de 24 h, varrida junto com os 45 min, tem ponto melhor? | não | a fazer |
| **B3** | Priorização em vez de supressão | gestão de alarme (ativo nas primeiras N h, depois "condição conhecida") e o contexto de catálogo do EXP30/31 como prioridade | não | a fazer |
| **C1** | Retreino mensal (walk-forward) | o −19% de FP do EXP10c se repete nos canais separados? | **sim (Cica)** | a fazer |
| **C2** | OCSVM + iforest como votos extras no mesmo canal | somar (não substituir) muda o voto? | **sim (Cica)** | a fazer |
| **C3** | Limiares dos portões por LOEO | o −32% de FP do EXP10c se repete? | **sim (Cica)** | a fazer |

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
