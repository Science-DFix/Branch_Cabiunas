---
name: avalia-normalidade
description: Avalia se um modelo de normalidade NÃO supervisionado (PCA, SFA, CVA, DPCA, MSET ou qualquer scorer com fit/score) do detector físico de 4 sinais do TC-330.03A está "aprendendo" — o substituto de curva de loss para modelos estatísticos sem gradiente. Mede generalização treino→mês seguinte, curva de aprendizado por tamanho de baseline, estabilidade do subespaço, sensibilidade por falha sintética injetada e separação contra os trips. Use SEMPRE que o usuário quiser saber se um modelo/sinal novo aprende, comparar scorers (ex. PCA × SFA × CVA), diagnosticar deriva ou calibração do limiar, validar um canal antes de colocá-lo no detector, ou disser coisas como "não temos loss", "ver se o modelo está aprendendo", "avaliar o modelo", "curva de aprendizado", mesmo sem citar esta skill.
---

# Avaliação de modelo de normalidade (sem loss)

Os canais `t` e `p` do detector são **modelos de normalidade** ajustados em fechado
(PCA hoje; SFA/CVA na branch `exp/sinal-sfa-cva`). Não há época nem gradiente, então
não há curva de loss. As perguntas que a loss responderia continuam válidas, e cada
uma tem um equivalente medível em **dado normal fora do ajuste** — que existe em
abundância, ao contrário dos 8 trips rotulados:

| pergunta de quem treina rede | equivalente aqui | nível |
|---|---|---|
| a loss de treino caiu? | o score no próprio baseline fica ~1% acima de 1,0 (por construção, via p99) | L1 |
| a loss de validação acompanha? | excedência do limiar no **mês seguinte**, walk-forward: nominal 1% | L1 |
| está sobreajustando? | razão score mediano validação / treino | L1 |
| mais dado ajudaria? | curva: excedência de validação × tamanho do baseline | L2 |
| os pesos convergem? | ângulo principal entre subespaços de meses seguidos | L3 |
| o modelo aprendeu algo útil? | taxa de detecção e atraso de falhas **sintéticas** injetadas | L4 |
| acerta o alvo? | score nas 48 h pré-trip × janelas aleatórias (AUC, p) | L5 |

L1–L4 não usam rótulo e por isso são o critério de **aprendizado**. L5 usa os 8 trips e
é o único nível sujeito ao sobreajuste que o repositório documenta — leia-o como
indício, não como prova.

## Como rodar

O script está em `scripts/avalia_normalidade.py` (ao lado deste arquivo). Rode da raiz
do repositório.

**Primeiro, sempre, o `--demo`** — valida o próprio ferramental em dado sintético (~7 s):

```bash
python .claude/skills/avalia-normalidade/scripts/avalia_normalidade.py --demo
```

**No dado real**, o dataset do detector está no ClearML (servidor `cica`, projeto
`TesteMLCab`), com `grade2min.parquet`, `falhas.csv` e `piso_fisico_cache.npz`:

```bash
python .claude/skills/avalia-normalidade/scripts/avalia_normalidade.py \
  --dataset-id 8b06a98f8b264820a9ecf2075a188395 \
  --familia temperatura \
  --scorer-path .claude/skills/avalia-normalidade/scripts:scripts/pdm_fisico \
  --scorer scorers:ScorerMax --col pca_recon \
  --desde 2024-06-01 --saida avaliacao_pca_t.json
```

- `--familia temperatura|pressao` (ou `--tags a,b,c`) escolhe o canal.
- `--scorer modulo:Classe` é qualquer classe com `fit(df) -> self` e
  `score(df) -> DataFrame`, cuja coluna `--col` é normalizada (>1 = fora do normal).
  Para SFA/CVA, aponte para a classe nova e **use a mesma normalização** (p99 do
  baseline), senão L1 não é comparável.
- Para L3, o scorer precisa expor `pca.components_` ou um método `subespaco()` que
  devolva uma base ortonormal (d × k). Sem isso, L3 fica vazio — não é erro.
- `--parquet`/`--falhas` locais substituem o `--dataset-id`.
- Acesso ao `cica` exige estar na tailnet `tail4d7f36` (ver memória do projeto). Sem
  acesso, diga isso ao usuário em vez de tentar outro caminho.

Não use `ablacao:ScorerMax` direto: `ablacao.py` chama `main()` no import e roda a
ablação inteira. Por isso `scorers.py` traz uma cópia fiel.

## Comparando modelos

Para comparar PCA × SFA × CVA, rode o script **uma vez por scorer, com os mesmos
argumentos** (mesmo `--n-fit`, `--desde`, `--seed`), e monte uma tabela lado a lado
com: excedência de validação (mediana), razão treino/validação, ângulo mediano,
menor magnitude de degrau com ≥80% de detecção, atraso da rampa em 3 MAD, e AUC/p de
L5. Não compare um scorer rodado com `--seed` diferente — L4 sorteia trechos.

**Scorers dinâmicos (`scripts/pdm_fisico/dinamico.py`)**: avalie **cada estatística
separada** com `--col`, não só o `score` combinado — o ganho deles está justamente em
mandar mudança de regime para as estatísticas de nível e manter as de dinâmica limpas:

| scorer | nível (absorve mudança de ponto de operação) | dinâmica / resíduo |
|---|---|---|
| `dinamico:ScorerSFA` | `nivel` (T2d, T2e) | `dinamica` (S2d, S2e) |
| `dinamico:ScorerCVA` | `T2` | `Q`, `D` |

No `--demo` (degraus de carga), a excedência de validação fica em 13–21% nas de nível e
em 1–2% em `dinamica`, `Q` e `D`. É esse padrão que se procura no dado real.

## Como ler o veredito

O script termina com um bloco `VEREDITO`. Traduza para o usuário assim:

- **Excedência de validação ≫ 1%** — o baseline não representa o mês seguinte
  (deriva). É o achado do `conformal.py`: espalhamento real ~1,9× o do baseline. Em
  um modelo novo, isso pesa mais que qualquer ganho em L5: vai virar alarme falso.
- **Razão validação/treino > 1,5** — gap de generalização; o modelo decorou o baseline.
- **Curva que ainda cai no maior tamanho** — mais baseline ajudaria; mas mexer em
  `FIT_POINTS` muda *quais* pontos entram e o detector é muito sensível a isso (o dia
  do retreino vale 23× nas horas de FP). Sugira como experimento, não como ajuste.
- **Ângulo > 45° entre meses** — subespaço instável: o modelo está aprendendo regime
  ou ruído, não estrutura.
- **Sensibilidade** — um bom canal detecta degrau de 2–3 MAD em minutos e rampa de 3
  MAD em horas. Se "congelado" é detectado muito bem, o canal vai acender em sensor
  travado — não é falha da máquina.
- **L5** — enriquecimento e AUC com **8 eventos** têm intervalo enorme. p < 0,05 aqui
  é raro e não é o critério de adoção.

Um modelo "aprende" quando L1 fica perto do nominal, L3 é estável e L4 detecta falhas
pequenas. Só então vale colocá-lo no detector e medir as três réguas (de pé, início,
banda), a carga e o minimax da vizinhança — isso é outro passo
(`publica_clearml.py --offline`), não esta skill.

## Cuidados

- O `--demo` tem degraus de carga de propósito, então o veredito dele acusa deriva:
  é o comportamento esperado do PCA estático sob mudança de regime, e é exatamente o
  que SFA/CVA deveriam melhorar.
- Não commite os JSONs de saída nem dados: a regra do repositório é **só código**.
- A injeção sintética mede sensibilidade a falhas *simples*. Os 8 precursores reais
  são heterogêneos (`trajetoria.py`); boa sensibilidade sintética é necessária, não
  suficiente.
