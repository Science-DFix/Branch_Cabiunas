# CNN-1D AE por grupo de sinal, com retreino mensal e régua

Documento próprio do braço CNN-1D da disputa D1 (`frente_ocsvm/d1_disputa.py`). Registra o modelo, a configuração,
o que foi feito com a camada de decisão e a régua, e os resultados. O diário geral continua em `DIARIO_DE_BORDO.md`;
aqui fica só o CNN-1D.

**Estado em 18/10/2026:** rodadas 1b (treino único) e 1c (mensal desde 07/2025) concluídas; nenhuma combinação passa
no critério. Rodada 1d (mensal desde 01/2025, para julgar os 8 trips) **em execução no Cica** (seção 7).

---

## 1. Ideia

O CNN-1D AE já tinha sido testado no projeto, mas sempre como **modelo único com todos os sinais juntos** e sem a
camada de decisão. Ele perdeu para a OCSVM nesse formato. A pergunta da D1 é outra: com a mesma arquitetura vigente
(**um modelo por grupo de sinal + regras**), um CNN-1D no lugar da OCSVM em cada grupo dá menos falso positivo (FP)
sem perder trip? E com **retreino mensal**, que acompanha a deriva dos sensores?

O CNN-1D olha uma janela de 30 min (forma temporal), não um ponto isolado como a OCSVM. A aposta era cortar FP de
pico curto.

---

## 2. Modelo

**Autoencoder convolucional 1D** (`build_cnn1d_autoencoder` em `src/cnn1d_ae/model.py`), o mesmo do caminho legado:

```
entrada (60 passos x n_features)
 -> Conv1D(f1, k1, stride s1, relu) -> Dropout
 -> Conv1D(f2, k2, stride s2, relu)
 -> Conv1DTranspose(f2, k2, stride s2, relu) -> Dropout
 -> Conv1DTranspose(f1, k1, stride s1, relu)
 -> Conv1DTranspose(n_features, 3)            (reconstrução)
perda MSE, Adam
```

**Busca de hiperparâmetros** (`run_tuner` em `src/cnn1d_ae/tuning.py`): keras-tuner RandomSearch, 10 trials, objetivo
`val_loss` (só dados normais), semente fixada antes da busca e antes do refit final.

| hiperparâmetro | espaço |
|---|---|
| filters_1, filters_2 | 8, 16, 32, 64 |
| kernel_1, kernel_2 | 3, 5, 7, 9 |
| stride_1, stride_2 | 1, 2 |
| dropout | 0,0 a 0,4 (passo 0,1) |
| lr | 1e-4, 3e-4, 1e-3, 3e-3 |

Treino: 20 épocas, batch 1024, early stopping com paciência 6 (restaura os melhores pesos), ReduceLROnPlateau.

**Integração com o AutoML por grupo** (`_fit_score_cnn1d` em `src/cnn1d_ae/automl_pipeline.py`, fitter `"cnn1d"` em
`_FITTERS`, ao lado de `ocsvm`, `iforest`, `dense` e `pca`):
- Janelas de `TIME_STEPS = 60` pontos (30 min a 30 s) que **nunca cruzam buraco** da série (`_segments`).
- Janelas de treino com passo `STRIDE = 5` (memória do worker); validação = últimos 10% do normal (temporal).
- **Score causal:** o erro de cada ponto é o MSE da janela que **termina** nele. Os 59 primeiros pontos de cada trecho
  herdam o erro da 1ª janela; trechos mais curtos que 60 pontos ficam com erro 0 (nunca alertam).
- O resto é o mesmo de qualquer fitter: o AutoML escolhe percentil do limiar e debounce, aplica máscara operacional e
  portões.

---

## 3. Configuração

Configs em `frente_ocsvm/d1_configs/` (gerados por `d1_disputa.py gera`): cópia exata dos 3 configs de produção
(`configs/calibracao_v4_eq/test_grupo_exp33/34/38`) trocando só o que está na tabela abaixo.

| grupo (config) | sensores do modelo | alarmes de avaliação do AutoML | portões |
|---|---|---|---|
| temperatura (EXP33) | TC382_03_A, T5_AVG_A | TC382_03_A, T5_AVG_A | carga, degrau (T5_AVG_A), sensor congelado |
| vibração (EXP34) | TV_351X/Y … TV_355X/Y (10) | TC382_03_A, T5_AVG_A | carga, degrau (T5_AVG_A), sensor congelado |
| óleo (EXP38) | 954005_624_PI_0308, 954005_624_PDIT_0305 | PALL_6240309 | sensor congelado |

**Comum aos 3 (produção, inalterado):**
- Dados: `sensores_full_2024_2026_30s.csv` (Dataset ClearML `a97ba56b`), 30 s, 07/2024 a 04/2026.
- Features derivadas por sensor: valor, delta de 1 passo, e mediana, desvio e tendência móveis em 4 janelas
  (12, 120, 480, 2880 pontos = 6 min, 1 h, 4 h, 24 h), mais curtose, assimetria e fator de crista nas janelas de
  120 pontos ou mais.
- Normal de treino: exclui 24 h em torno de cada alarme e buracos longos; outliers clipados nos quantis 0,1% e 99,9%;
  z-score com estatísticas só do treino.
- AutoML: corte `AUTOML_OOS_SPLIT_DATE = 2025-07-01`; grade de percentis (temp: 95–99,9; vib e óleo: 69–99,9) × debounce
  (1, 4, 12 pontos); escolhe pelo score composto (detecção dos alarmes do grupo depois do corte, penalidade 2,0 para
  alerta no normal, detecção mínima de 30%). Semente 42.

**O que muda em cada braço:**

| braço | mudanças sobre a produção |
|---|---|
| `cnn1d` (rodada 1b) | `AUTOML_MODELS = ["cnn1d"]`, `TIME_STEPS 60`, `STRIDE 5`, `MAX_TRIALS 10`, `EPOCHS 20`, `BATCH_SIZE 1024`, `PATIENCE 6` |
| `cnn1d_mensal` (1c) | o de cima + `ENABLE_WALKFORWARD_RETRAIN = true`, `WALKFORWARD_RETRAIN_FREQ = "MS"`, `ENABLE_MIN_DURATION_FILTER = false` |
| `cnn1d_mensal_jan` (1d) | o de cima + `WALKFORWARD_START_DATE = "2025-01-01"` |

**Retreino mensal (walk-forward, `_walkforward_fit_periods`):** no início de cada mês, retreina do zero com **todo o
normal anterior** (janela expansiva): normalização, tuner, modelo e limiar refeitos. Cada mês é pontuado pelo modelo
treinado até a véspera. Antes do 1º mês o braço não emite flag. O filtro de duração interno do AutoML fica desligado
porque o `src/` não aceita os dois juntos (o filtro de 45 min da camada de decisão continua valendo, seção 4).

**Hiperparâmetros escolhidos** (semente 42; no mensal, os do último mês):

| task | percentil | debounce | filtros | kernels | strides | dropout | lr |
|---|---|---|---|---|---|---|---|
| cnn1d temp | 97,5 | 12 | 16/16 | 3/7 | 2/2 | 0,0 | 3e-3 |
| cnn1d vib | 95,0 | 1 | 16/16 | 3/7 | 2/2 | 0,0 | 3e-3 |
| cnn1d óleo | 99,9 | 12 | 64/16 | 5/9 | 1/2 | 0,4 | 1e-4 |
| cnn1d_mensal temp | 95,0 | 1 | 16/16 | 3/7 | 2/2 | 0,0 | 3e-3 |
| cnn1d_mensal vib | 95,0 | 12 | 16/16 | 3/7 | 2/2 | 0,0 | 3e-3 |
| cnn1d_mensal óleo | 99,9 | 12 | 16/16 | 3/7 | 2/2 | 0,0 | 3e-3 |

---

## 4. Camada de decisão (regras) e régua (avaliação)

São duas coisas diferentes:

**Regras** (`frente_ocsvm/comum.py`, `decide`) — o que geraria o alarme em produção, **congeladas** em toda a D1:
1. 4 canais: os 3 modelos de grupo + **canal 4** (alarmes de processo das 5 tags `PI_6240319_AL`, `PAL_6240315`,
   `PDAL_6240302`, `TC382_05_A`, `PAH_6240319`, aceso por 24 h após cada ocorrência).
2. **Voto ≥ 2** de 4 canais.
3. **Filtro de 45 min:** o voto precisa durar pelo menos 45 min contínuos.
4. **Refratário de 48 h** depois de cada alerta.

**Régua** (`C.avalia`, `C.por_trip`) — só avaliação, depois do fato (em produção ninguém sabe qual alerta vira trip):
- Alertas agrupados em episódios (intervalo de até 2 h junta).
- Episódio que começa nas **48 h antes de um trip** = **detecção**.
- Não é detecção, mas vem seguido (até 48 h depois do fim) de uma parada real de 2 h ou mais = **inconclusivo** (pode
  ser evento físico não catalogado; não conta a favor nem contra).
- Nenhum dos dois = **FP**.
- Métricas: trips detectados (de 8), antecedência (h), FP e FP/mês de operação; recortes "inteiro" e "fora da amostra"
  (depois de 01/07/2025: 3 trips).

**Critério da D1** (fixado antes de enviar): 8/8, 3/3 fora e FP/mês ≤ 75% da OCSVM nos dois recortes (≤ 2,164 inteiro e
≤ 2,158 fora). Para o mensal, que só emite flag a partir do início do walk-forward, só vale o recorte em que ele emite.

---

## 5. O que foi feito

| data | etapa | commit |
|---|---|---|
| 16/10 | fitter `cnn1d` no AutoML por grupo, testado em dados sintéticos; pré-registro da rodada 1b | `6f36d72` |
| 16/10 | `cnn1d` aceita walk-forward; pré-registro da rodada 1c (ideia do usuário: "por grupo, mensal, com as regras") | `c2d183a` |
| 16/10 | 1b: 3 tasks; 1c: só temperatura (vib e óleo ficaram para depois) | — |
| 17/10 | `cnn1d_mensal_vib` falha por falta de memória (exit 137) no 8º mês | — |
| 17/10 | correção: cada mês do walk-forward do `cnn1d` treina num **processo filho** | `1c369e5` |
| 18/10 | 1c completa (vib e óleo); placar: nada passa | `c9d52b4` |
| 18/10 | funil das regras pós-treino (`frente_ocsvm/d1_funil.py`) | `a78b124` |
| 18/10 | pré-registro da rodada 1d; `WALKFORWARD_START_DATE`; envio | `45b8b6c`, `4773a46` |

**A falha de memória.** O TF/Keras não devolve a memória de cada `fit` (medido local: ~45 MB por fit puro e ~200 MB
por tuner de 2 trials, mesmo com `clear_session`, `gc` e `malloc_trim`). Com 10 trials e 10 sensores, a vibração
acumulava ~1 GB por mês e estourou o worker no 8º. A correção (`_cnn1d_period_entry`) treina cada mês num processo
`spawn`, que devolve tudo ao sair; só o modelo do último mês fica na memória. Resultado igual dentro do ruído do TF
(diferença 6e-5; duas execuções idênticas diferem 4e-5), porque o tuner e o refit refixam a semente.

**Tasks no ClearML** (projeto TesteMLCab; lista completa em `frente_ocsvm/d1_envios.md`):

| braço | temp | vib | óleo |
|---|---|---|---|
| cnn1d | `14348f72` | `ee9c1a9e` | `80fbbc23` |
| cnn1d_mensal | `2c19e1b9` | `6fbead3c` (a 1ª, `71647eac`, falhou por memória) | `7a7b77bf` |
| cnn1d_mensal_jan | `bbc71b16` | `892e93c1` (a 1ª, `f0701a16`, morreu no 16º mês: rodou em paralelo com o óleo na mesma máquina/GPU) | `aab72b80` |

---

## 6. Resultados

### 6.1 Placar (semente 42)

| combinação | trips | FP/mês inteiro | trips fora | FP fora | FP/mês fora |
|---|---|---|---|---|---|
| **OCSVM (produção)** | 8/8 | 2,885 | 3/3 | 23 | 2,877 |
| cnn1d nos 3 grupos | 6/8 | 3,503 | 2/3 | 28 | 3,502 |
| cnn1d só na vib | 8/8 | 3,366 | 3/3 | 27 | 3,377 |
| cnn1d só no óleo | 8/8 | 2,885 | 3/3 | 23 | 2,877 |
| cnn1d só na temp | 7/8 | 3,159 | 3/3 | 24 | 3,002 |
| cnn1d_mensal nos 3 grupos* | — | — | 3/3 | 26 | 3,252 |
| cnn1d_mensal só na vib* | — | — | 3/3 | 24 | 3,002 |
| cnn1d_mensal só no óleo* | — | — | 3/3 | 23 | 2,877 |
| cnn1d_mensal só na temp* | — | — | 3/3 | 27 | 3,377 |

\* Mensal desde 07/2025: só vale o fora da amostra. Nenhuma combinação passa no critério.

### 6.2 Antecedência por trip (h)

| trip | amostra | OCSVM | cnn1d (3 grupos) | cnn1d_mensal (3 grupos) |
|---|---|---|---|---|
| 27/02/2025 | dentro | 33,8 | 33,8 | não avaliado |
| 17/03/2025 | dentro | 31,2 | 31,2 | não avaliado |
| 07/04/2025 | dentro | 43,2 | 15,5 | não avaliado |
| 11/04/2025 | dentro | 19,8 | 19,3 | não avaliado |
| 29/04/2025 | dentro | 36,7 | perdeu | não avaliado |
| 04/11/2025 | fora | 13,7 | 13,7 | 13,9 |
| 09/12/2025 | fora | 8,4 | 21,2 | **21,1** |
| 26/02/2026 | fora | 3,8 | perdeu | 3,8 |

O mensal pega os 3 trips fora e antecipa o de 09/12/2025 em 21 h (a OCSVM, em 8 h).

### 6.3 Funil das regras (fora da amostra, máquina ligada, 700.843 pontos)

| etapa | OCSVM | cnn1d_mensal (3 grupos) | cnn1d_mensal só na vib |
|---|---|---|---|
| pontos anômalos na vib | 97.992 | 43.520 (−56%) | 43.520 |
| pontos anômalos na temp | 14.544 | 33.674 (+132%) | 14.544 |
| algum canal de modelo aceso | 108.633 | 66.021 (−39%) | 55.744 (−49%) |
| voto ≥ 2 | 58.302 | 45.825 (−21%) | 37.009 (−37%) |
| após 45 min | 44.091 | 30.578 (−31%) | 27.737 (−37%) |
| após refratário 48 h | 11.304 | 9.951 (−12%) | 9.224 (−18%) |
| **episódios de FP** | **23** | **26** | **24** |

**Leitura:** o mensal corta muito **ponto** anômalo na vibração, mas quase nada de **episódio**. O canal 4 fica aceso
36% do tempo com a máquina ligada, então basta um canal de modelo encostar nele para virar voto; o mensal deixa menos
pontos, mas espalhados em episódios separados, e a régua conta episódio. Na temperatura o mensal faz o contrário
(2,3 vezes mais pontos), e é isso que piora o FP quando ele entra nesse grupo.

---

## 7. Rodada 1d — mensal desde 01/2025

**Pergunta (do usuário):** o CNN-1D mensal pega os 8 trips com antecedência? Na 1c ele só emitia flag a partir de
07/2025, então os 5 trips de fev–abr/2025 não eram julgáveis.

**Desenho** (pré-registro no docstring de `d1_disputa.py`, commit `45b8b6c`): walk-forward desde 01/01/2025 (1º
retreino com o normal de 07–12/2024; 16 retreinos). O AutoML continua escolhendo limiar e debounce só com alarmes
depois de 07/2025, então os 5 trips de 2025 entram **fora da amostra** para este braço. Recorte de julgamento:
01/2025 ao fim, com a OCSVM no mesmo recorte. Avaliação: `frente_ocsvm/d1_jan.py`.

**Referência da OCSVM no recorte:** 8/8, 36 FP, 3,051 FP/mês. Para passar: 8/8 e FP/mês ≤ 2,288, mantendo 3/3 fora
com ≤ 2,158.

**Expectativa registrada:** detecta 6 a 8 de 8, com mais FP que a OCSVM; NÃO pelo critério.

**Ressalvas:** (1) a OCSVM viu os 5 trips de 2025 no treino, este braço não; empate nesses 5 favorece o braço;
(2) o 1º retreino tem só ~6 meses de normal.

**Resultado (18/10, 21:30): NÃO PASSA em nenhuma combinação.** A expectativa registrada se confirmou.

Antecedência por trip (h; — = não detectado). `todos` = CNN-1D mensal nos 3 grupos; `só X` = CNN-1D só no grupo X,
OCSVM nos outros; canal 4 igual em todas.

| trip | OCSVM | todos | só temp | só vib | só óleo |
|---|---|---|---|---|---|
| 27/02/2025 | 33,8 | 33,8 | 33,8 | 33,8 | 33,8 |
| 17/03/2025 | 31,2 | 31,2 | 31,2 | 31,2 | 31,2 |
| 07/04/2025 | 43,2 | 31,4 | 15,5 | 32,8 | 43,2 |
| 11/04/2025 | 19,8 | 19,8 | 19,8 | 19,8 | 19,8 |
| 29/04/2025 | 36,7 | 12,4 | — | 12,4 | 36,7 |
| 04/11/2025 | 13,7 | 13,9 | 13,7 | 13,7 | 13,9 |
| 09/12/2025 | 8,4 | 21,1 | 8,4 | 21,1 | 8,4 |
| 26/02/2026 | 3,8 | 3,8 | 3,8 | 3,8 | 3,8 |

| | trips 01/2025→ | FP | FP/mês | trips fora | FP fora (FP/mês) |
|---|---|---|---|---|---|
| OCSVM | 8/8 | 36 | 3,051 | 3/3 | 23 (2,877) |
| todos | 8/8 | 45 | 3,814 | 3/3 | 26 (3,252) |
| só temp | **7/8** | 44 | 3,729 | 3/3 | 27 (3,377) |
| só vib | 8/8 | 42 | 3,560 | 3/3 | 24 (3,002) |
| só óleo | 8/8 | 36 | 3,051 | 3/3 | 23 (2,877) |

Corte: 8/8 e ≤ 2,288 FP/mês no recorte, 3/3 e ≤ 2,158 fora.

**Leitura:**
- **Responde à pergunta: sim, o CNN-1D mensal pega os 8 trips com antecedência** (todos e só vib), também nos 5 de
  2025, que este braço não viu no treino. A menor antecedência continua sendo 3,8 h (26/02/2026), igual à OCSVM.
- **Antecedência:** ganha no trip de 09/12/2025 (8,4 → 21,1 h, vem da vibração) e perde nos de 07/04 (43,2 → 31,4 h)
  e 29/04/2025 (36,7 → 12,4 h).
- **Sozinho na temperatura perde o trip de 29/04/2025:** ali a temperatura da OCSVM era um dos 2 votos. Nos 3 grupos,
  a vibração do CNN-1D cobre esse trip (12,4 h).
- **FP:** sobe em todas as combinações (36 → 45 nos 3 grupos, +25%), em vez de cair os 25% exigidos. O óleo é neutro
  (idêntico à produção; quase não vota).
- **Conclusão:** o CNN-1D mensal é **equivalente em detecção** e **pior em FP**. O ganho de antecedência em 09/12 não
  compensa pelo critério pré-registrado. Reduzir FP depende da camada de decisão, não da troca de modelo.

---

## 8. Limitações

- Só 8 trips, 3 fora da amostra. Qualquer escolha feita olhando esses trips tende a parecer melhor do que é.
- O CNN-1D é a 3ª família testada nos mesmos trips (antes iforest e dense; depois PCA). Um vencedor ainda precisaria
  de dados novos.
- Sem varredura de semente para o CNN-1D (o AutoML só faz isso para iforest e OCSVM); a confirmação em 3 sementes só
  rodaria para quem passasse.
- Os quantis do clip de outliers são calculados sobre todo o normal, inclusive depois de cada retreino (vale também
  para a OCSVM). É um vazamento pequeno, igual para os dois lados; não foi corrigido para não mudar o método no meio
  da comparação.

---

## 9. Como reproduzir

```bash
python frente_ocsvm/d1_disputa.py gera                      # configs em frente_ocsvm/d1_configs/
python frente_ocsvm/d1_disputa.py envia --so cnn1d_mensal_vib   # clona a task de produção do canal e enfileira
python frente_ocsvm/d1_disputa.py avalia                    # placar de todas as combinações (seção 6.1)
python frente_ocsvm/d1_funil.py saida.csv                   # funil das regras (seção 6.3)
python frente_ocsvm/d1_jan.py                               # rodada 1d (seção 7)
```

O `envia` exige o commit local igual ao de `cabiunas/exp/ocsvm-validacao` (o worker clona do remoto) e pula o que já
está em `frente_ocsvm/d1_tasks.json`. Para enviar 1 por vez, respeitando a fila do Cica, usar
`frente_ocsvm/d1_fila.py <nomes>` (espera fila vazia, 2 workers livres e o worker da task anterior liberado; anota
cada evento em `frente_ocsvm/d1_envios.md`).
