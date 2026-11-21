# TC-33003A: resposta da equipe de modelos sobre janela histórica e gargalos de inferência

Resposta ao documento `tc33003a-janela-historica-e-gargalos-de-inferencia.md` (análise de 13/10/2026).
Versão de 21/11/2026. O que depende da próxima versão do pacote (v2.1, ainda não avaliada) está marcado como
**[FORA DESTA VERSÃO]**.

**Unidade das células.** Todas as contagens de células deste documento são em **linhas de 2 min** × colunas (60 d =
60 × 720 × 37 ≈ 1,6 M). No export de 30 s, a parte de bruto é 4× maior: um bloco de 30 d tem ~3,2 M células, e o
limite de 1 M comporta ~9 d por carga. O histórico que alimenta a pipeline do detector no Cica é a linha de 2 min
já montada pela mediana das leituras de 30 s, como faz `preparar_grade` (J16). Por isso, os números de carga em blocos
(itens 3 e 4) valem como estão.

**"Bit a bit"** neste documento quer dizer: sinais com |Δ| ≤ 1e-9 e máscara e alarme idênticos em cada instante. As
diferenças medidas em t e p são da ordem de 1e-14 (arredondamento de ponto flutuante); sp, vb, máscara e alarme saem
exatamente iguais (J15).

## Resumo

- **O detector não precisa de 60 nem de 90 dias de dado bruto a cada atualização.** Na execução, ele precisa de três
  coisas:
  1. **3 d de bruto** (37 colunas) do export de 30 s do PI, reduzidos à grade de 2 min por `preparar_grade` (mediana
     de cada janela de 2 min), só com janelas fechadas;
  2. um **buffer de amostras estáveis das 10 sondas de vibração** (usa 13.000, guarda 26.000);
  3. **45 d dos sinais já calculados** (6 colunas: t, p, sp, vb, máscara, reset).
- **Um motor com estado reproduz a referência em lote bit a bit.** Medimos com reinícios, buracos de 1 h a 7 d,
  sondas fora do ar, dados atrasados, trips conhecidos tarde e trocas de bundle.
  - Estado: 3,3 MB por equipamento.
  - Por execução: **0,5 s e 139 MB**, contra 1,8 s e 495 MB do lote de 90 d de hoje e 1,3 s e 366 MB do lote de 60 d
    (pacote `e733279`; J18).
  - Leitura por atualização: ~409 mil células (3 d de bruto, 84 mil; buffer do vb, 130 mil; 45 d de sinais, 194 mil),
    contra 1,68 M de 60 d de bruto. Com o bruto de 3 d contado a 30 s, ~660 mil: continua abaixo de 1 M.
- **Correções, trips novos e bundles novos se resolvem refazendo o trecho afetado, não recarregando tudo.** O que já
  foi emitido para a tela não é reescrito.
  - Até 17 d atrás: refazer pelo estado, em ~0,6 s.
  - Mais antigo: voltar a uma foto diária do estado e refazer em blocos de 30 d, em até 2 s, com cada carga em
    ≤ 879 mil células.
- **Instalação nova:** carga inicial de 60 d de bruto, lida em 2 blocos de 30 d. A saída é igual desde o primeiro
  instante, e cada carga fica em ≤ 879 mil células (em linhas de 2 min; a 30 s, blocos de ~8 d).
- **Sem a vibração, o detector perde quase todos os trips:** 2 de 8, contra 7 de 8. Hoje o pacote continua votando
  nesse caso. Propomos marcar a saída como `INDISPONÍVEL`. Com a base do vb incompleta, ele pega mais trips mas quase
  dobra os falsos alarmes (J12); propomos marcar como `AQUECENDO`.
- **Se o lote for mantido:** 60 d é o menor histórico bit a bit e 45 d o menor sem diferença de alarme. Passar de 90
  para 60 d não muda nenhum resultado. Nenhuma janela de lote ≥ 45 d cabe no limite de 1 M células.

**O que propomos à engenharia:**
0. **Antes de tudo: gerar bundles novos.** Os bundles da branch vão até 29/03/2026 e valem 62 d; hoje a inferência
   devolve o código 2 (bundle vencido) e não roda. O primeiro passo no SIMPred é rodar o `constroi_bundle.py` com dado
   novo.
1. **Já, sem mudar o detector:** alinhar o `HISTORY_DAYS` de 90 para 60 d. O resultado não muda, e o tempo e a
   memória por execução caem ~30%. Se a latência permitir, rodar a cada 15 min em vez de a cada linha (7,5× menos
   execuções).
2. **Trocar o teste de 30 d de calendário por um estado de saída** (`INDISPONÍVEL`, `DEGRADADA`, `AQUECENDO` ou
   `PLENA`; item 3). Ele junta os códigos de saída e o diagnóstico de entrada que o pacote já tem com a contagem de
   amostras estáveis da base do vb.
3. **Implementar o motor com estado** (itens 1 e 4) e validar contra a referência em lote com igualdade exata (item 5).
   É o único caminho medido que fica abaixo do limite de 1 M células em todas as cargas.
4. **Carregar o histórico em blocos de até 30 d**, com o limite de células valendo por carga: na instalação nova e na
   recuperação de correção antiga.
5. **Guardar o que foi emitido num registro só-acréscimo**, com o instante de emissão de cada episódio (item 4).

**Como medimos.** Dezessete testes, cada um com pré-registro versionado antes de rodar. Todos rodaram como tasks no
cluster Cica (ClearML; Threadripper 7970X, 64 núcleos, 131 GB):

- dados de jan/2025 a abr/2026 do TC-33003A;
- os 16 bundles mensais históricos, cada um no seu mês;
- reprodução cronológica com cortes a cada 6 h, usando só o que o banco tinha em cada instante;
- referência: o pacote da equipe de modelos sem alteração (`cabiunas_inference.py`, branch
  `feat/pacote-consolidado`, commit `e17bf27`). O J17 e o J18 usam a `feat/pacote-respostas-engenharia`, commit
  `e733279`: leitura do CSV com texto do PI convertido em NaN, 1 thread e o estado de saída já implementados;
- custo medido com o mesmo medidor de telemetria usado no relatório do pacote de 07/10 (`telemetria_execucao.py`).

Os resultados de igualdade não dependem da máquina; tempos e memória devem ser medidos de novo no ambiente do SIMPred.
Os testes estão em `frente_thallys/j*.py` no repositório `Branch_Cabiunas`, branch `exp/ocsvm-validacao`.

| Teste | Pergunta | Task no Cica |
|---|---|---|
| J1 | Janela de entrada: qualidade de 30 a 90 d | `397a6e11` |
| J2 | Estado mínimo: que parte pede histórico | `31f31a1a` |
| J34 | Motor com estado: reinício, buracos, sondas, atrasos, trips tardios, instalação nova | `24c98ea5` |
| J5 | Janela de leitura do retreino mensal (anexo) | `4feb79c4` |
| J6 | Bundle mensal atrasado | `4c542c2b` |
| J7 | Limite do "refazer" pelo estado | `e9ce6fc5` |
| J8 | Correção antiga dentro do limite de 1 M células | `0355fa6d` |
| J9 | Carga inicial de instalação nova em blocos | `c9411e84` |
| J10 | Custo só do cálculo; grade de 2 min montada em trechos de 3 d | `1d3d16f1` |
| J11 | Custo por execução com o medidor de telemetria; detecção sem a vibração | `8b7268f0` |
| J12 | Detecção com a base do vb incompleta | `ce91ed2f` |
| J13 | Base do vb curta: ganho real ou mais horas de alarme? | `e7574f0d` |
| J14 | Memória do lote por período; execução com 1 thread | `760f35b1` |
| J15 | Motor com estado como módulo separado, contra o lote de 60 d | `244691f9` |
| J16 | A grade de 2 min que alimenta a pipeline é a mediana do export de 30 s? | `003a69f8` |
| J17 | Motor com estado com sonda de vibração ausente no CSV | `896d034c` |
| J18 | Custo do lote e "`PLENA` em" pela regra do pacote, na `e733279` | `4e026612` |

## 1. Que componente determina o histórico necessário?

**Só a referência de vibração (`vb`) e a memória da decisão pedem histórico longo. Os demais sinais não.** (J1, J2)

- **t, p e sp** nunca divergiram, nem com 30 d de janela. O EWMA (meia-vida de até 1 h) converge em 3 d de bruto.
- **vb:** a base de 400 h estáveis mais a guarda de 24 h é posicional: precisa de 12.720 amostras estáveis antes do
  bloco de 6 h. É o único sinal que pede histórico bruto, e só das 10 sondas TV_ mais as 2 colunas da máscara
  (`RUNNING_A`, `T5_AVG_A`).
  - Um buffer com as últimas 13.000 amostras estáveis dessas 12 colunas, guardado à parte, dá o `vb` bit a bit com
    3 d de bruto.
  - Isso responde à observação sobre mediana e MAD móveis: mantemos a amostra exata de que eles dependem, sem
    aproximação.
- **Decisão:** tem duas memórias longas, as duas cobertas por 45 d de sinais.
  - **CUSUM:** fica na decisão, não nos sinais, e lembra até a maior corrida sem parada (35,7 d no histórico).
  - **Refratário de 72 h:** é armado por todo episódio de voto que nasce fora de um bloqueio, **inclusive os de
    atenção**, que não chegam à duração mínima (120 min, ou 60 min se forte) e não viram alarme. Um episódio que nasce
    dentro do bloqueio é descartado e não rearma, exceto o forte e antigo. Como cada bloqueio depende do episódio
    anterior, a decisão lembra a cadeia de episódios, não só as últimas 72 h.
  - O comentário em `cabiunas_inference.py:659` ("depois de um alarme") está errado; o código está certo e é a
    referência. Armar só depois de alarme muda o detector: na base atual, 7/8 com 9 FP vira 7/8 com 15 FP.
  - Por isso a decisão roda sobre **45 d de sinais** (6 colunas, não bruto), e assim sai bit a bit. Com 30 d, 0,2% dos
    instantes têm alarme diferente. Com 14 d, a régua cai para 6/8 e FP 16 e perde o trip de 26/02/2026.

**Reduzir separadamente:**
- **a base do vb não deve ser reduzida.** Mudar o tamanho dela muda o modelo (J12, J13 e varredura da equipe de
  modelos; régua de referência 7/8, 9 FP, 1.457 h de alarme):

  | Base do vb | Trips | FP | Horas de alarme |
  |---|---|---|---|
  | 420 h | 7/8 | 9 | 1.452 |
  | **400 h (atual)** | **7/8** | **9** | **1.457** |
  | 380 h | 8/8 | 11 | 1.515 |
  | 300 h | 8/8 | 17 | 1.929 |
  | 200 h | 8/8 | 17 | 2.331 |
  | 100 h | 8/8 | 15 | 2.196 |

  A base curta pega o trip de 04/11/2025, que está na beira da base (−5% já o pega), mas espalha alarme pelo período
  todo. Com as horas de alarme igualadas às da base atual, ela não a supera (J13). A guarda de 24 h não foi testada.
- a memória da decisão é 45 d de sinais, ~194 mil células, e não depende do bruto.

Na janela de lote (J1, contra todo o histórico; régua de referência: 7/8 trips, 9 FP no período):

| Janela | vb diferente | Alarme diferente | Régua | Horas estáveis no pior instante |
|---|---|---|---|---|
| 90 d | 0 | 0 | 7/8 · FP 9 | 1.152 h |
| 60 d | 0 | 0 | 7/8 · FP 9 | 518 h |
| 45 d | 0,30% (máx 3,5e-6) | 0 | 7/8 · FP 9 | 279 h |
| 35 d | 4,7% | 2,9% | 7/8 · FP 14 | 218 h |
| 30 d | 9,9% | 4,6% | 7/8 · FP 16 | 140 h |

## 2. Qual o menor histórico validado para o último ponto?

| Modo de execução | Histórico | Resultado |
|---|---|---|
| Lote (como hoje) | 60 d de bruto | bit a bit em todos os instantes (J1) |
| Lote | 45 d de bruto | alarme idêntico; vb difere em 0,30% dos pontos, máx 3,5e-6 (J1) |
| Incremental (motor com estado) | 3 d de bruto + buffer + 45 d de sinais | bit a bit (J2, J34) |

**Cobertura de sensores e amostras estáveis:**
- **O critério é contagem de amostras estáveis, não tempo de calendário.** Uma linha está "cheia" quando há ≥ 12.720
  amostras estáveis antes dela. Isso substitui o teste atual de 30 d entre a primeira e a última linha, que, como o
  documento aponta, não mede cobertura.
- **Sondas fora do ar:** continua bit a bit no motor. Testado com TV_351X/Y vazias por 7 d, uma tag de temperatura
  do PCA vazia por 1 d e as 10 TV_ vazias por 2 d (J34).
- **Buracos no historiador** de 1 h, 6 h, 1 d, 3 d e 7 d: bit a bit (J34).

## 3. Um período inicial sem previsão é aceitável?

A decisão é da engenharia e da operação. Hoje o pacote **não para** por falta de histórico: sem base do vb ele
continua votando com t, p e sp. Mas, sem o vb, a detecção cai de 7 para **2 de 8 trips** (FP de 9 para 8; J11). Uma
saída nesse período parece normal e quase não detecta. Por isso propomos o estado **`INDISPONÍVEL`** enquanto a base do
vb não se forma. **É uma regra nova.** A `e733279` já a implementa (`estado_saida`, campo novo no contrato do
dashboard); o `e17bf27` não.

Propomos quatro estados de saída. Os nomes foram escolhidos para não colidir com o `degradado` do
`diagnostico_entrada()`:

| Estado | Quando |
|---|---|
| **`INDISPONÍVEL`** | bundle vencido (código 2), entrada cega (código 3) ou menos de 3.720 posições estáveis antes do bloco do vb (as sondas não opinam) |
| **`DEGRADADA`** | família sem todas as tags (`diagnostico_entrada()` = `degradado`) |
| **`AQUECENDO`** | entre 3.720 e 12.720 posições estáveis antes do bloco do vb: a base está incompleta e a detecção muda |
| **`PLENA`** | base do vb cheia. Sozinha, não garante saída igual à referência (ver a tabela abaixo) |

As duas contagens são na mesma unidade, posições estáveis antes do bloco de 6 h, e incluem a guarda de 24 h
(720 posições): a sonda opina com 720 + 3.000 (`VB_FRACAO_BASE_MIN` = 0,25 de 12.000) e a base fica cheia com
720 + 12.000.

Vale o primeiro estado que se aplica, nesta ordem: `INDISPONÍVEL` > `DEGRADADA` > `AQUECENDO` > `PLENA`. O estado
resume campos que o pacote já tem, mais a contagem da base do vb:

| Campo | Valores | Entra no estado como |
|---|---|---|
| código de saída da inferência | 0 ok; 2 bundle vencido (não roda); 3 entrada cega (não roda) | 2 e 3 → `INDISPONÍVEL` |
| `diagnostico_entrada()` | `ok` / `degradado` / `cego` | `degradado` → `DEGRADADA` |
| posições estáveis antes do bloco do vb (novo) | contagem | < 3.720 → `INDISPONÍVEL`; 3.720 a 12.720 → `AQUECENDO` |

Os códigos 1 (retreino recusado) e 4 (bundle gerado fora da faixa histórica) são do `constroi_bundle.py`, não da
inferência. Retreino recusado mantém o bundle anterior até ele vencer (62 d); daí a inferência passa a devolver 2.

**O que fazer em `AQUECENDO`.** Com a base do vb incompleta o período todo (pior caso), o detector pega 8/8 trips, mas
com 15 a 19 FP contra 9 e 32% a 61% mais horas de alarme (J12; tabela no item 1). A saída **não deve ser tratada como
`PLENA`**. Mostrar o alarme com aviso de falsos alarmes ~2× maiores, ou não mostrar, é decisão da operação.

**Memória da decisão.** Além da base do vb, a decisão precisa de 45 d de sinais (item 1). Isso não é um estado: é uma
condição da instalação. Com a carga inicial recomendada (60 d), a saída é igual à referência desde o primeiro
instante. Sem carga inicial, a saída só fica igual depois de 18 d (até 36 d; tabela abaixo).

Medimos quanto dura cada fase em 30 instalações simuladas de 100 d (6 datas × 5 cargas), todas com as tags
completas, contando as posições como o pacote conta (J18; motor do J34 contra o lote de 60 d).

| Carga inicial de bruto | `PLENA` em (mediana; mín–máx) | Em `INDISPONÍVEL` (mediana; máx) | Em `AQUECENDO` (mediana) | Igual à referência desde (J34) | `PLENA` diferente da referência |
|---|---|---|---|---|---|
| nenhuma (só dado novo) | 19 d (18–52) | 126 h (648 h) | 333 h | 18 d (até 36 d) | 3 de 6 instalações |
| 7 d | 12 d (11–48) | 0 (540 h) | 294 h | 12 d | 4 de 6 |
| 30 d | o 1º instante (até 1,3 d) | 0 | 0 | 0 (exceto 1 caso) | 2 de 6 |
| **45 d** | o 1º instante | 0 | 0 | **o 1º instante** | **0 de 6** |
| **60 d** | o 1º instante | 0 | 0 | o 1º instante | **0 de 6** |

- **`PLENA` garante a base do vb, não a igualdade com a referência.** Em 9 das 30 instalações a saída em `PLENA`
  difere da referência:
  - em 8 (cargas de 0, 7 e 30 d), os sinais diferem nas ~12 h seguintes à entrada em `PLENA` (327 a 419 linhas de
    2 min, 0,5–0,7% das linhas em `PLENA`), **sem mudar nenhum alarme**. A causa ainda não foi diagnosticada;
  - em 1 (instalação em 01/01/2026 com carga de 30 d), os sinais são iguais, mas o alarme fica aceso de 02/01 a
    23/01, onde a referência tem dois episódios curtos (16,5% das linhas em `PLENA`). A tabela de 30 d não tem a
    memória da decisão (45 d).
- **Com carga de 45 ou 60 d**, `PLENA` vale desde o primeiro instante e a saída é igual à referência nas 12
  instalações.
- Sem carga, a mediana é de 19 d, mas o período importa: as instalações em 01/05 e 01/11/2025, com pouca operação
  estável, levam 52 e 38 d.
- A regra anterior (base cheia em todas as linhas dos últimos 45 d) dava `PLENA` em 64, 57, 35, 20 e 13 d, sem
  nenhuma divergência em `PLENA`, porque esperava também a memória da decisão. A regra do pacote mede só a base do
  vb; a memória da decisão fica garantida pela carga inicial de 60 d (condição da instalação, acima).

A tabela acima usa 6 datas de instalação em operação normal. O J9 repetiu a carga em 20 datas desde abr/2025,
comparando 14 d de emissões com o motor contínuo:

| Carga inicial | Igual ao motor contínuo | Montagem (mediana) | Maior carga |
|---|---|---|---|
| 45 d numa carga | 16/20 | 0,40 s | 1,20 M células |
| 45 d em blocos de 30 d | 16/20 (as mesmas 4 falhas) | 0,91 s | 799 mil |
| 60 d numa carga | 20/20 | 0,51 s | 1,60 M células |
| **60 d em 2 blocos de 30 d** | **20/20** | **1,07 s** | **879 mil** |

- **45 d não bastam em período de baixa disponibilidade.** As 4 instalações que falham começam entre 25/05 e
  14/06/2025, e só o vb diverge. A causa ainda não foi diagnosticada. A hipótese é pouca operação estável no período,
  com a base de 400 h alcançando antes do início da carga.
- **Montar em blocos não muda nada.** Os blocos falham exatamente onde a carga inteira falha, no mesmo instante e com a
  mesma diferença, e acertam onde ela acerta.

**Recomendação:** instalar com carga inicial de **60 d de bruto, lida em blocos**. A saída fica igual à do motor
contínuo desde o primeiro instante (20/20), e a montagem leva ~1,1 s.
- Em linhas de 2 min, 2 blocos de 30 d: cada carga lê ≤ 879 mil células, abaixo do limite padrão do worker.
- No export de 30 s, cada bloco teria ~3,2 M células; para ficar abaixo de 1 M, blocos de ~8 d. Medimos que blocos de
  30 d e de 14 d (J8) não mudam o resultado; blocos de 8 d não foram medidos.

O estado de saída continua valendo como salvaguarda: em qualquer período, ele marca `AQUECENDO` enquanto a base do vb
não estiver cheia.

## 4. Que estado persistir e como restaurar?

**Estado por equipamento** (J2, J34):
1. buffer do vb: até 26.000 amostras estáveis × 10 sondas;
2. tabela de 45 d dos sinais: t, p, sp, vb, máscara, reset;
3. instante do último ponto processado.

Total: 3,3 MB em disco; reler leva 5 ms.

**Entrada.** A entrada do motor é o export de 30 s do PI. A linha de 2 min é a mediana das leituras de cada janela,
montada por `preparar_grade`; o motor herda isso se usar o `preprocessar` do pacote.

- **Só janelas fechadas.** A linha de 2 min de `t` só entra quando a janela [t, t + 2 min) terminou. Em 50 cortes
  (J10), a grade montada a partir de 3 d de bruto com janelas fechadas é idêntica à montada a partir do histórico
  inteiro (50/50). Se a última janela entrar com 2 das 4 leituras, a linha sai diferente em ~29 das 39 colunas, com
  até 128% de diferença relativa.
- **Leitura de 30 s que chega depois** numa janela já processada é um **dado atrasado**: segue a regra de refazer
  (tabela abaixo).
- **Não usar uma grade de 2 min pronta de outra fonte.** Uma linha montada com a leitura mais próxima ou interpolada
  passa sem erro e aumenta os falsos alarmes de 9 para 12 no período (+33%) e as horas de alarme fora de trip de 407
  para 620 (+52%).
- **A grade de 2 min da pipeline atual serve.** O CSV de 2 min que alimenta a pipeline do detector no Cica (Dataset
  `bb85d853`, jan/2025 a abr/2026) é a mediana do export de 30 s. Foram comparadas 349.200 linhas × 38 colunas, e a
  diferença máxima foi de 3,05e-5. Isso equivale a 1 passo de float32 na faixa de 256 a 512, ou seja, arredondamento
  do texto do CSV. Os NaN coincidem em todas as células. A leitura mais próxima e a média da janela diferem em
  centenas de unidades (J16).

**Atualização normal:** ler 3 d do export de 30 s, montar a grade de 2 min com `preparar_grade`, calcular, anexar as
linhas novas à tabela e rodar a decisão sobre a tabela.

**Eventos de recuperação:**

| Evento | O que fazer | Medido |
|---|---|---|
| Reinício | reler o estado salvo | bit a bit, 5 ms (J34) |
| Buraco ou dado atrasado | refazer desde o início do trecho que chegou | bit a bit após a chegada; ignorar muda o alarme, FP 9 → 10 (J34) |
| Correção de dado até 17 d atrás | refazer desde o trecho corrigido | exato em 20/20 até 17 d; 0,49–0,65 s; lê P + 3 d de bruto (J7) |
| Trip registrado tarde | refazer desde trip − 7 d (exclusão do vb é [t − 7 d, t + 2 d]) | bit a bit com trip conhecido 7 d depois; ignorar dá FP 11 (J34) |
| Bundle novo (dia 1) ou atrasado | refazer desde o dia 1 do mês quando o bundle chega | bit a bit após a chegada; ignorar muda 2,7–7,8% dos alarmes e com 7 d de atraso perde um trip (J6) |
| Correção mais antiga que 17 d | voltar à última foto diária anterior ao trecho e refazer em blocos de 30 d | exato em 160/160 (idades de 21 a 60 d); 0,66–1,96 s; cada carga ≤ 879 mil células (J8) |

**O que já foi emitido fica congelado.** Refazer recalcula o estado e as linhas daí para a frente. O que já foi
emitido para a tela fica congelado num registro que só acrescenta, e a linha recalculada de um instante passado não
substitui a emitida. Sem isso, um trip registrado depois faz aparecer no passado um alarme que nunca foi mostrado; no
histórico, isso acontece com o episódio de 03/11/2025, antes do trip de 04/11. Propomos duas séries:
- **emitida:** só cresce; é o que a operação viu e o que vale para auditoria e para a régua de tempo real;
- **recalculada:** a do estado corrigido, para análise.

No contrato (`contrato_dashboard`), cada episódio ganha o campo **`emitido_em`**: o instante em que foi mostrado pela
primeira vez. É o campo que permite validar pelo instante de emissão (item 5).

**Por que o limite de 17 d:** refazer corta o buffer do vb no início do trecho. Até ~17 d de operação contínua sobram
as 12.720 posições estáveis de que a base precisa; acima disso, não (21 d: exato em 13/20; 30 d: 6/20). O limite em
dias é mais seguro de implementar que a contagem exata de posições, que acertou 119/120.

**Fotos diárias:** uma foto por dia, guardada por 60 d, ocupa ~200 MB por equipamento (conta, não medido). Na
recuperação em blocos de 30 d (J8):

| Idade da correção | Tempo (mediana) | Maior carga | Total lido |
|---|---|---|---|
| 21 d | 0,66 s | 659 mil | 646 mil |
| 30 d | 1,12 s | 879 mil | 966 mil |
| 45 d | 1,41 s | 879 mil | 1,37 M |
| 60 d | 1,96 s | 879 mil | 1,84 M |

A partir de 30 d o **total** lido passa de 1 M células. O carregamento precisa ser por bloco, com o limite valendo por
carga.

**Sobre a revisão no pré-processador:** com o motor, uma revisão vira "refazer desde o primeiro timestamp revisado".
Fica entre 0,5 e 2 s e não exige recriar o feature builder. A sobreposição de timestamps com valores iguais não muda
nada no resultado e não precisaria disparar revisão. Quanto isso ocorre em produção é medida do lado da engenharia.

**O bundle deve sair no dia 1.** O atraso em si custa de +1 a +5 FP em 15 meses, qualquer que seja o procedimento (J6).

**[FORA DESTA VERSÃO] Campos da notificação.** A próxima versão do pacote acrescenta, por episódio, uma prioridade pela carga e
um "onde olhar". Para isso a tabela de 45 d precisa de mais colunas:
- prioridade: `carga_t5` nos 30 d anteriores a cada episódio;
- onde olhar: `sp_z`, `t_dominante` e `p_dominante` nas 2 h iniciais do episódio.

O efeito no tamanho do estado será medido quando essa versão for avaliada. Esta resposta vale para o pacote `e17bf27`.

## 5. Igualdade exata ou tolerância?

**Decidido com a equipe de modelos (14/11):**

- **Implementação incremental contra a referência em lote**, com os mesmos dados e bundles: exigir igualdade.
  - Sinais com |Δ| ≤ 1e-9.
  - Máscara e alarme idênticos em cada instante.
  - Mesma régua no período de validação: 7/8 trips, 9 FP.
  - Medimos que isso é alcançável em todos os cenários do item 4. Uma tolerância maior só esconderia erro de
    implementação, porque as diferenças que aparecem (ignorar uma correção, perder a cadeia do refratário) mudam
    episódios e FP, não casas decimais.
- **Redução de janela ou mudança do detector:** qualquer diferença de alarme conta como mudança do modelo. Ela precisa
  de nova validação de detecção, antecedência e FP, e não de aceite por tolerância.
- **Semântica online:** a série em lote preenche o início do episódio quando a duração mínima se confirma. Ao vivo, o
  alerta sai em média ~2,0 h depois desse início (1,0 h nos episódios fortes), então a antecedência operacional é a do
  lote menos 1–2 h.
  - Exemplo: o trip de 11/04/2025 cai de 4,1 h para 2,1 h de antecedência (J1, 18 episódios).
  - Propomos registrar os dois instantes, o início estimado e o instante de emissão (`emitido_em`), e comparar a
    implementação pelo instante de emissão.

## 6. Orçamento de RAM, latência e recuperação

O orçamento é decisão da engenharia. Cada execução abaixo é o comando inteiro num processo novo: leitura do CSV do
export de 30 s, `preparar_grade` uma vez, bundles, diagnóstico, drift, decisão e saída. Mediana de 10 instantes
sorteados desde abr/2025, no Cica. Os números são do pacote `e733279` (J18), que já lê o CSV convertendo o texto
do PI em NaN e roda com 1 thread; entre parênteses, o pacote `e17bf27` medido na mesma task (= J11).

| Por equipamento, por execução | Lote 60 d | Lote 90 d (hoje) | Motor com estado |
|---|---|---|---|
| Entrada lida | 71 MB de CSV (60 d) | 106 MB de CSV (90 d) | 3,6 MB de CSV (3 d) + estado |
| Duração | **1,3 s** (p95 1,4) (`e17bf27`: 1,6 s) | **1,8 s** (p95 2,0) (2,3 s) | **0,5 s** (p95 0,56) |
| CPU (usuário + sistema) | 1,3 s (5,7 s) | 1,8 s (6,8 s) | 4,7 s (sem 1 thread; ver abaixo) |
| RAM pico do processo | **366 MB** (364–375) (563 MB) | **495 MB** (488–499) (799 MB) | **139 MB** (139–146) |
| Estado em disco | — | — | 3,3 MB (+ ~200 MB de fotos diárias) |
| Instalação nova | — | — | ≤ 1,1 s (60 d em 2 blocos de 30 d; J9) |
| Recuperação após reinício | recarga completa | recarga completa | 5 ms (J34) |
| Recuperação após correção | recarga completa | recarga completa | 0,5–0,7 s (≤ 17 d); 0,7–2 s (até 60 d, em blocos; J7, J8) |

- **Os números do `e17bf27` batem com o relatório do pacote de 07/10**, medido com o mesmo medidor: inferência de
  90 d com entrada de 30 s, 750 MB e 5,4 s. A memória confere; o tempo medido aqui é menor (2,3 s), e a causa da
  diferença não foi identificada. Os 167 MB desse relatório são da inferência com entrada de 2 min.
- **A memória do lote não depende mais do conteúdo do CSV.** No `e17bf27`, 11 a 14 células com estado do sistema do
  PI em texto ("Comm Fail", "Out of Serv") faziam a leitura tratar todas as colunas como texto: o lote de 90 d ia de
  ~497 MB nos instantes de 2025 a ~805 MB nos de 2026 (J14). A `e733279` converte essas células em NaN na leitura, e
  os dois anos ficam iguais (492 e 497 MB; J18). O orçamento pode usar ~500 MB (90 d) e ~370 MB (60 d).
- **CPU com 1 thread.** A álgebra linear usa vários núcleos se deixada livre. Com 1 thread (`OMP_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) a saída é a mesma byte a byte (J14), e a `e733279` já roda assim:
  a CPU caiu para ~1/4 e a duração ~20%. O motor medido ainda não fixa 1 thread, por isso a CPU dele passa da
  duração; esperamos que caia do mesmo modo, mas não medimos.
- **Instalação e recuperação** (J7, J8, J9) foram medidas com o rastreamento de memória do Python ligado, que deixa o
  cálculo ~3× mais lento. Os valores são **limites superiores**.

**CPU por dia e por equipamento**, num processo novo por execução (J18):

| Execução | A cada linha de 2 min (720/dia) | A cada 15 min (96/dia) |
|---|---|---|
| Lote 90 d | 1.270 s | 170 s |
| Lote 60 d | 940 s | 126 s |
| Motor com estado (sem 1 thread) | 3.370 s | 449 s |

Com um processo novo por execução, a partida domina a CPU do motor. Um consumidor que fique de pé, com os bundles
carregados, paga só o cálculo: 0,13 s por execução no motor, contra 0,31 s no lote de 60 d (J10, sem a leitura do
arquivo).

**Medidas que ficam com a engenharia**, porque dependem da aplicação e não do detector:
- tempo de consulta ao banco e número de consultas de 15 min;
- atraso de consumo com vários equipamentos no mesmo consumidor;
- frequência real das revisões e recargas.

## Decisões em aberto

1. **Granularidade do histórico consultado (resolvida em 18/11).** A pipeline do detector no Cica lê a linha de 2 min
   montada pela mediana (Dataset `bb85d853`, J16), então valem os números deste documento (2 blocos de 30 d na
   instalação).
   - Se outro worker passar a ler o export de 30 s, as cargas de bruto ficam 4× maiores e os blocos passam a ~8 d.
   - O motor com estado fica abaixo de 1 M nos dois casos (~409 mil ou ~660 mil por atualização).
2. **Forma de entrega.** Proposta da equipe de modelos: funções de referência do motor com estado num módulo
   separado, com o teste de equivalência contra o lote, sem alterar o `cabiunas_inference.py`. O módulo existe
   (`frente_thallys/motor_estado/`) e reproduz o lote de 60 d nas 1.708 atualizações de mar/2025 a abr/2026, com e
   sem reinício a cada atualização (J15).
3. **Tolerância da validação:** igualdade exata (item 5), comparando sempre no mesmo ambiente.
4. **[FORA DESTA VERSÃO] Campos da notificação:** medir o estado com as colunas novas (item 4) quando a v2.1 for
   avaliada.

## Limites

- Um equipamento e um período (jan/2025 a abr/2026). Os regimes cobertos são os que aparecem nele: corridas longas,
  paradas, falhas de sensores e 8 trips.
- Tempos e memória são do Cica, não do servidor do SIMPred. Nos testes locais (notebook i5, 8 GB), os tempos foram
  ~2× maiores. A leitura do banco não está nos números: o CSV vem do disco local.
- O motor medido no J11 e no J18 importa o módulo de testes, maior que um motor de produção. O número é um limite superior.
- O "refazer" foi medido sobre o mesmo dado (J7, J8), para isolar o mecanismo. O J34 mostrou que, com dado corrigido,
  o resultado é o da referência.
- J7, J8 e J9 rodaram sem trips, para medir só o mecanismo do buffer. O efeito dos trips está no J34.

## Anexo: janela de leitura do retreino mensal (J5)

O documento não pergunta sobre o retreino. Incluímos porque o volume de linhas lido no retreino foi levantado na
reunião entre as equipes.

O `constroi_bundle.py` hoje lê o histórico inteiro, mas usa só os últimos 20.000 pontos estáveis antes do dia 1.

| Janela de leitura | Bundle idêntico (sha256) |
|---|---|
| 77 d | 16/16 meses |
| 60 d | 15/16 meses |
| 45 d | 12/16 meses |
| 30 d | 6/16 meses |

- Toda falha é recusa explícita por "pontos insuficientes"; nenhuma janela gerou um bundle diferente.
- Com 77 d são 221.760 linhas de 30 s, fixas, contra 1,75 M do histórico inteiro, que cresce ~86 mil por mês. O tempo
  cai de 2,3 s para 0,35 s.
- Se uma parada longa deixar menos de 20.000 pontos estáveis em 77 d, a leitura deve se estender até tê-los. Isso é
  argumento, não medida: não ocorreu no período.

**Recomendação:** o retreino do dia 1 lê só [dia 1 − 77 d, dia 1), com 12,7% das linhas de hoje e o mesmo bundle.
