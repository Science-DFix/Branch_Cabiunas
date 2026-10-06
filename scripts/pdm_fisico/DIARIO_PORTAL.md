# Diário de bordo — benchmark do pacote e tags do Portal de Integridade

Branch: `exp/portal-tags-extras`, no repositório **Science-DFix/Branch_Cabiunas** (remoto `cabiunas`).
Ela parte de `cc6edc9`, a ponta da `exp/reducao-falsos-positivos` do Thallys. Nada desta frente foi enviado ao
`origin` (SIMPred-Petrobras/cabiunas-models).
Período: 05 e 06/10/2026. ClearML: servidor Cica, projeto `TesteMLCab`, fila `default`.

---

## Resumo

| # | O quê | Resultado |
|---|---|---|
| 1 | Benchmark de recursos do pacote (pedido da engenharia) | Medido no `cc6edc9`. O treino é um PCA linear: segundos e até 2 GB de RAM. O retreino do zero dos 16 bundles leva de 35 a 87 s e reproduz os bundles versionados a ~1e-15. |
| 2 | Dados antigos do Portal (86 tags, 30 s, 2022-01 a 2025-10) | Já estão em UTC (**não** descontar 3 h). Dez/2022 a set/2023 estão congelados. De 2024-03 a 2025-10 têm cadência real: é o export que o `tags_ausentes.py` pedia. |
| 3 | Triagem das tags extras (`tags_ausentes.py`) | Vibração do compressor 4,5×, escora 3,7× e óleo de dreno 3,0× antes dos trips (p ≤ 0,001). É triagem, não validação. |
| 4 | Segunda base no ClearML | Dataset `341cecbd`: as 38 tags originais (intactas bit a bit) + 15 do Portal. |
| 5 | **C-PORTAL**: canal do Portal como **veto** de episódio | **Reprovado.** O FP cai de 0,972 para 0,416/mês, mas 4 de 8 composições perdem detecção. |
| 6 | **V-PORTAL**: canal do Portal como 5º canal do **voto** | **Reprovado.** A detecção fica igual e o FP sobe de 0,972 para 1,180/mês. |

**Conclusão:** com os 5 eventos cobertos (2025-01 a 2025-10), as tags do Portal não furam o teto de FP, nem como
veto nem como voto. A pergunta continua aberta e só se responde com **dado novo**: o export interpolated de 2025-11 em
diante (eventos de 11/2025, 12/2025 e 02/2026) ou o modo sombra. Desenhar uma terceira variante sobre os mesmos
eventos seria ajuste ao acaso.

---

## 1. Benchmark de recursos do pacote

**Pedido da engenharia:** pico de RAM, CPU média e pico, duração total e por etapa, disco, espaço temporário, tamanho
do bundle, falhas e retries, consumo durante o retreino e atraso das predições.

**Achado prévio:** o Dataset pai `e8c79d82` (CSVs + pacote) traz o pacote de `f67fae3`, de antes do perf da grade
(`101de36`) e dos portões e EWMA (`849c3f5`). As medições de 05/10 enviadas no texto à engenharia mediram essa versão
antiga, não a ponta da branch.

**Mudanças (commit `710bd6f`):**
- `telemetria_clearml.py`: o Dataset-filho passa a sobrescrever `scripts/` e `modelos/` com os do commit atual, e a
  task registra `commit_pacote`.
- `retreino_do_zero.py`: reconstrói os bundles de 2025-01 a 2026-04, um processo por mês.

**Tasks:**
- Telemetria: `5823c8ce` (Dataset-filho `ab202b61`).
- Tempo por etapa: `f2e54448` (`mede_tempos_clearml.py --dataset ab202b61`).
- Máquina: Threadripper 7970X, 64 núcleos lógicos, contêiner Python 3.12.

Medianas de 3 execuções, pacote `cc6edc9`:

| Cenário | Duração | CPU média / pico | RAM (maior processo) | Leitura lógica | Escrita |
|---|---|---|---|---|---|
| Inferência, 2 min, 90 d | 1,2 s | 4,5 / 43 núcleos | 186 MB | 37 MB | 7,2 MB |
| Inferência, 30 s, 90 d | 3,2 s | 2,3 / 43 | 748 MB | 168 MB | 7,2 MB |
| Retreino mensal, 2 min, 484 d | 2,2 s | 5,3 / 43 | 524 MB | 167 MB | 0,02 MB |
| Retreino mensal, 30 s, 240 d | 5,8 s | 2,6 / 42 | 1,96 GB | 364 MB | 0,02 MB |
| Retreino do zero, 2 min, 16 meses | 34,7 s | 5,5 / 43 | 523 MB | 5,2 GB | 0,36 MB |
| Retreino do zero, 30 s, 16 meses | 87,3 s | 2,7 / 43 | 1,96 GB | 11,7 GB | 0,18 MB |

- **Falhas:** 0 nas 18 execuções. **Espaço temporário:** 0. **Disco físico lido:** 0, porque o CSV estava em cache.
- **Pico de CPU:** é uma rajada de BLAS de menos de 0,1 s. A média é o número que representa o uso.
- **Por etapa:** o ajuste do PCA leva 0,03 s; o resto é leitura e grade do CSV (1,5 s com 2 min, 4,9 s com 30 s).
- **Bundle:** cerca de 9,6 KB por mês (6 JSON).
- **Retreino do zero:** os meses sem 20.000 pontos estáveis saem com código 1, como previsto (2 de 16 com 2 min,
  9 de 16 com 30 s). Os bundles gerados batem com `modelos/` a ~1e-15.

**Não medido aqui:** o que a aplicação da engenharia consome em paralelo, e o atraso de chegada do dado. Os dois
dependem do servidor e do pipeline deles. O `telemetria_execucao.py` roda em qualquer máquina para isso.

## 2. Dados antigos do Portal de Integridade

**Fonte:** zips do Drive (`PI-20261005T222531Z-1-00{1,2}.zip`) + `Lista Variáveis por Equipamentos.xlsx`, em
`~/FRENTE_THALLYS/dados_antigos/`, fora do repositório.

**Conversão:** `converte_portal.py --raiz dados_antigos` (usa `python-calamine`; cerca de 13 s por mês contra
minutos com o openpyxl). Gera um parquet por mês em `dados_antigos/parquet/`. A série única de 30 s é
`_todos_30s_bruto.parquet`: os 46 mensais principais concatenados, sem os `*_OLD`.

| Item | Achado |
|---|---|
| Formato | Os mensais `*_interpolated_PortalIntegridade.xlsx` têm 86 tags a 30 s. O cabeçalho repete a `TI_0324`. A grade de tempo está completa: 4.032.000 linhas, sem lacuna nem duplicata. |
| Cópias `OLD/` | 2025-01 a 03 são idênticas às principais. 2025-04 `OLD` está 88% vazia e foi ignorada. `OLD/Interpolated` e `OLD/Recorded` são outros conjuntos (36 tags `UTGCAB_6240_*`, ou `recorded`). |
| **Fuso** | **Já é UTC**, na mesma convenção da `grade2min.parquet`. Cada mensal vai das 03:00 UTC do dia 1 às 02:59 do mês seguinte, ou seja, o mês local. Correlação com a grade em lag 0: T5 0,985, TI_0305 0,983, TV_351X 0,988. Em −3 h, a TI_0305 cai para 0,854. **Não descontar 3 h** para juntar com a base atual. |
| **Período congelado** | De dez/2022 a set/2023, a T5 e a TI_0305 aparecem 100% preenchidas mas mudam **0 vezes por dia**. A RUNNING_A e outras 44 tags estão ausentes. **Descartar.** |
| Cadência | De 2024-03 a 2025-10, as tags extras mudam de 2.000 a 2.880 vezes por dia: cadência real, não `recorded` preenchido. É o export pedido em `tags_ausentes.py` (`a269c63`). |
| Colunas | Contém as 38 da grade + 48 tags que o detector não usa. |
| RUNNING_A (dúvida do Renan) | Com as velocidades a 30 s (15.788 h): desligada ⇒ NGP, NPT e NCPSR sempre 0 (até o p99); ligada ⇒ NGP ≥ 72 no p1. Concorda 99,2% do tempo. O erro é "ligada" falsa (~103 h, quase tudo em ago/2025), já protegida pela máscara `T5 > 300`. Máscara alternativa possível: `NGP_A > ~70 e T5 > 300`. |

**Triagem** (`tags_ausentes.py`, sem mudanças, sobre a grade de 2 min do Portal de 2023-10 a 2025-10; cobre 6 dos 9
eventos do `falhas.csv`):

| Canal | Pico antes do trip / normal | p |
|---|---|---|
| Vibração do compressor (VI_0301/02/04/05) | 4,47× | < 0,001 |
| Mancal de escora (TI_0318/0319) | 3,69× | 0,001 |
| Óleo de dreno (TI_0320/21/22) | 2,98× | 0,0005 |
| Deslocamento axial (ZI_0301/02, TV_350A/354A) | 1,96× | 0,05 |
| Desempenho (vazão, torque, rotações, P/T de sucção) | 1,84× | 0,19 |
| Controle `t` | 5,30× | 0,019 |
| Controles `sp` / `vb` | 2,2× / 1,9× | 0,11 / 0,12 |

## 3. Segunda base de teste (Dataset `341cecbd`)

`monta_grade_portal.py` (commit `fd1fc98`) parte da `grade2min.parquet` e acrescenta, por left join no mesmo índice,
15 tags: VI_0301/02/04/05, TI_0318 a 0322, ZI_0301/02, TV_350A/354A, NGP_A e NPT_A.
- **Regra de agregação:** a do pacote. Faixa física → NaN, depois mediana de 2 min com rótulo à esquerda. Com rótulo
  à esquerda e sem deslocamento, 55% dos pontos coincidem exatamente com a grade (10 a 17% em ±2 min).
- **Faixas próprias para as tags novas.** A do pacote para `TV_` (0 a 200) apagaria o deslocamento axial, que é
  negativo (−1,0 é sentinela). Faixas usadas: VI 0–200; TI 0–200; ZI ±2; TV_350A/354A ±0,9; NGP/NPT 0–120.
- **Controle:** o script só grava se as 38 colunas e o índice originais saírem idênticos bit a bit. Na primeira
  versão, ele barrou uma diferença de unidade no índice (µs contra ns).
- **Cobertura em operação estável:** 97% a 100% em todo mês de 2024-01 a 2025-09; 77% em 2025-10 (o export para em
  24/10); 0 de 2025-11 em diante.
- **Dataset ClearML** `341cecbd24b24df08cdb2d575c47cbd8` (`TC33003A_detector_fisico_entradas_portal`), filho de
  `8b06a98f`. Herda `grade2min.parquet`, `falhas.csv` e `piso_fisico_cache.npz`, e acrescenta
  `grade2min_portal.parquet`.

## 4. Testes no detector v2 (régua de 8 composições do Thallys)

**Protocolo comum aos dois testes:**
- Comparação pareada contra o detector de referência nas 8 composições (`regua_fp.DIAS`).
- Janela 2025-01-01 a 2025-10-24 nos dois braços, porque as tags novas acabam aí. Cinco eventos: 27/02, 17/03,
  07/04, 11/04 e 29/04.
- `bootstrap_regua.decide` com IC de 97,5% (Bonferroni sobre primário e secundário), blocos mensais de 2024-02 a
  2025-10.
- Pré-registro commitado **antes** de rodar.
- Os scripts rodam no Cica clonando esta branch. Eles ligam os arquivos do Dataset dentro de `scripts/pdm_fisico/`,
  porque o `publica_clearml.py` faz `chdir` para lá na importação.

**Canal `pt`**: o máximo sobre as três famílias de EWMA(1 h) / p99 do próprio baseline, com a composição do retreino
no dia `d`. Vibração do compressor e escora entram como z robusto máximo; óleo de dreno entra como spread contra os
irmãos.

**Referência na janela** (mediana): detecção 4,0 de 5, início 3,0, banda 2,0, 0,972 FP/mês, carga 112,8 h/mês.

### 4.1 C-PORTAL — veto de episódio (`confirmacao_portal.py`)

- **Pré-registro:** `e451d2f`.
- **Correção de execução:** `48adb96`. Os pesos do bootstrap foram refeitos para 21 meses, sem mudar a regra.
- **Task:** `a7ebb645`. A primeira tentativa (`6686f612`) falhou nessa correção, depois de passar os quatro controles.
- **Regra:** episódio do alarme final em que pt nunca chega a θ = 1,0 é apagado.

| Braço | Detecção | Início | Banda | FP/mês | Carga | Δcarga [IC] | ΔFP/mês [IC] | Perde detecção |
|---|---|---|---|---|---|---|---|---|
| θ = 1,0 | 2,5 | 1,5 | 1,0 | 0,416 | 97,5 | −15,3 [−34,2; −2,5] | −0,469 [−0,870; −0,150] | 4 de 8 |

- **Resultado:** o FP e a carga caem de verdade (ICs abaixo de zero, carga cai em 8/8), mas somem de 1 a 4 TP por
  composição. Primário: **reprovado** (A). Secundário: **não**. θ = 0,8 e 1,2 dão o mesmo quadro.
- **Leitura:** no nível do episódio inteiro, o canal não separa TP de FP.

### 4.2 V-PORTAL — 5º canal do voto (`voto_portal.py`)

- **Pré-registro:** `a43b6d9`. **Task:** `7f36f43e`.
- **Regra:** nível A ≥ 3 de 5, nível B ≥ 2 de 5 com portão sp|vb|pt, K_LO = KH = 1,0 para pt. Força, refratário e
  limiares atuais não mudam.

| Braço | Detecção | Início | Banda | FP/mês | Carga | Δcarga [IC] | ΔFP/mês [IC] | Perde / ganha detecção |
|---|---|---|---|---|---|---|---|---|
| θ = 1,0 | 4,0 | 3,0 | 2,0 | 1,180 | 130,1 | +17,3 [−35,8; +68,6] | +0,295 [−0,039; +0,769] | 0 / 1 |

- **Resultado:** a detecção fica igual, o FP sobe 21% e a carga sobe 15%. Primário: **reprovado** (B1, B2, C).
  Secundário: **não**.
- **Mecanismo:** surgem de 1 a 7 FP novos por composição, e os TP trocam de lugar. O evento de 07/04 passa a nascer
  6,6 h antes em todas as composições, o que é ganho em umas e perda em outras. A expectativa registrada (mais
  episódios, FP subindo) se confirmou.

## 5. Próximos passos

1. **Pedir o export interpolated (86 tags, 30 s) de 2025-11 em diante.** Cobre os 3 eventos que faltam, é a única
   validação sem circularidade e é pré-requisito para qualquer uso em produção (o feed atual só tem 38 tags).
   Com ele, rodar `monta_grade_portal.py` e os dois scripts **sem mudar nada** (regra, θ e critérios congelados).
2. **Não desenhar uma terceira variante** sobre os 5 eventos de 2025-01 a 2025-10.
3. **Benchmark:** o texto à engenharia precisa trocar os números pelos do `cc6edc9` (seção 1) e dizer qual versão foi
   medida.
4. **Avisar o Thallys:** `tags_ausentes.py` e `running_a_coerencia.py` esperavam este dado. Os limiares de
   velocidade da RUNNING_A agora são verificáveis.

## Como reproduzir

```bash
# dados antigos -> parquet (local)
python scripts/pdm_fisico/converte_portal.py --raiz ~/FRENTE_THALLYS/dados_antigos
# segunda base (precisa da grade2min.parquet do Dataset 8b06a98f)
python scripts/pdm_fisico/monta_grade_portal.py --grade grade2min.parquet \
    --portal ~/FRENTE_THALLYS/dados_antigos/parquet/_todos_30s_bruto.parquet --saida grade2min_portal.parquet
# testes no Cica (o worker clona esta branch e baixa o Dataset 341cecbd)
python scripts/pdm_fisico/confirmacao_portal.py --remote
python scripts/pdm_fisico/voto_portal.py --remote
# benchmark
python scripts/pdm_fisico/telemetria_clearml.py --remote --dataset e8c79d82e7c74425afea6c7620be3a18
python scripts/pdm_fisico/mede_tempos_clearml.py --remote --dataset ab202b618ea845d986aa0754c98e0345
```

## IDs

| Tipo | ID | Conteúdo |
|---|---|---|
| Dataset | `8b06a98f` | entradas do detector (grade2min, falhas, piso_fisico_cache) |
| Dataset | `341cecbd` | filho do anterior + `grade2min_portal.parquet` |
| Dataset | `e8c79d82` | pacote `f67fae3` + 4 CSVs do benchmark |
| Dataset | `ab202b61` | filho do anterior, pacote `cc6edc9` + medidores |
| Task | `5823c8ce` | telemetria (6 cenários) |
| Task | `f2e54448` | tempo por etapa |
| Task | `a7ebb645` | C-PORTAL |
| Task | `7f36f43e` | V-PORTAL |
