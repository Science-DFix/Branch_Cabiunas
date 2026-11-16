# TC-33003A: testes da resposta à engenharia (frente Thallys)

Só o necessário para reproduzir os testes J1 a J15 que sustentam a resposta da equipe de modelos ao documento
`tc33003a-janela-historica-e-gargalos-de-inferencia.md` (janela histórica e gargalos de inferência).

- **Comece por `frente_thallys/REPRODUZIR.md`:** versões exatas, Datasets do ClearML, três formas de rodar e a tabela
  dos testes.
- **Motor com estado (entrega de referência):** `frente_thallys/motor_estado/` (`motor_estado.py` e
  `teste_equivalencia.py`, que roda sozinho com `--pacote` e `--csv`).
- **Pré-registro e resultado** de cada teste estão no docstring do próprio script (`frente_thallys/jN_*.py`).
- **Registro cronológico:** `DIARIO_DE_BORDO.md` (seção 5; as entradas "frente Thallys"). O diário é o da frente
  OCSVM inteira e foi copiado como está.

Esta branch não tem histórico: é uma cópia de `frente_thallys/` da branch `exp/ocsvm-validacao` no commit `91d7f7a`.
A única mudança é o nome da branch que o `--enqueue` passa ao ClearML (`frente-thallys/reproducao`).
