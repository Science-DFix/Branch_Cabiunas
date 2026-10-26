# Mensagem para o Thallys — 25/10/2026

Preparada no fechamento da sessão de 24–25/10. Ainda não enviada: o envio é com o usuário. Contexto em
`DIARIO_DE_BORDO.md` (entradas de 24 e 25/10) e nos docstrings de `frente_thallys/t1..t3` e
`frente_ocsvm/ta_carga_thallys.py`.

## Texto para enviar

> Thallys, conseguimos reproduzir aqui o teu detector v2 do zero, sem o Cica: refizemos os 16 bundles mensais com o
> `constroi_bundle.py` e eles saem iguais aos do git (~1e-14). O controle bate (7/8, 9 FP, 1.457 h), e o EXP62 também
> (9→4, placebo 0,018, mesma vizinhança). Regeramos também a base n=19.000 (5/8, FP 11 → 6 com a carga).
>
> Testamos a tua regra de carga em 5 retreinos novos (n de 19 e 21 mil, baseline atrasado 7/14/21 d): ela reduz FP em
> todos e só perde detecção em um. Combinada com "sem alarme de processo nas 24 h → atenção", ficou com 2 a 4 FP nas
> bases dentro da faixa validada. Os custos são previsíveis: o 17/03 (pela carga) e o 26/02/26 (pelo alarme de
> processo).
>
> O que falta é testar nas bases G168 e B800, onde a carga perdeu o 17/03. Pelo `worker_exp60.py`, elas saíram da tua
> pasta local `data_local/controle_v2/G168/serie.csv.gz` e `.../B800/serie.csv.gz`. Se ainda estiverem aí, consegue
> mandar por e-mail ou Drive? Dá ~2 MB e não precisa do Cica. E como elas foram geradas (o que é G168 e B800)?
>
> Mais umas perguntas rápidas:
> 1. O filtro de partida que derrubou o FP foi o blackout de 6 h (6,06 → 4,35/mês) ou o EXP60?
> 2. O corte no dia 1 e o n = 20.000 foram escolhidos olhando os trips? Aqui o publicado saiu como o melhor de 6
>    variações: as outras dão 12 a 22 FP ou 5–6/8.
> 3. Tem tag de carga melhor que o T5 (potência, vazão de gás, rotação)?
> 4. Já tem dado depois de 20/04 ou registro da sombra? E a consulta à operação (os 27 períodos) teve resposta?
> 5. Na v2.1, por que manter o refratário armado por episódio de atenção? Pensamos em testar sem ele.

## Por que cada item

| # | Pedido ou pergunta | O que destrava |
|---|---|---|
| Arquivos G168/B800 | `serie.csv.gz` das duas variantes | O teste decisivo do T3: é onde a carga perdeu o 17/03. O `cria_insumos` do `worker_exp60.py` os leu de `data_local/controle_v2/` da máquina dele; pode não depender do Cica |
| Definição G168/B800 | Como foram gerados | Regerar aqui, como a base n = 19.000 |
| 1 | Qual filtro de partida | Confirmar a leitura de 25/10: o blackout de 6 h já está no v2 publicado |
| 2 | Escolha do dia 1 e do n | No T2, o publicado é o melhor de 6 retreinos: risco de viés de seleção e de fragilidade na operação |
| 3 | Tag de carga melhor | A regra saiu frágil no indicador e no limiar |
| 4 | Dados novos, sombra, consulta à operação | A única validação de verdade, para as duas frentes. O FP de 26/10/25 pode ser acerto (parada e 20 h de manutenção) |
| 5 | Refratário fantasma | Item 4 da lista de 25/10 (refratário só por alarme notificado) |
| — | Menor, não precisa perguntar | O complemento de 2024 saiu de xlsx interpolados; por isso o nosso vb difere em 2 de 390.762 instantes |

_(registrado em 25/10/2026 22:30)_
