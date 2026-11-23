# Frente Thallys: como reproduzir os testes da resposta à engenharia (TC-33003A)

Os testes J1 a J20 respondem às perguntas da engenharia (TC-33003A). Cada
script tem no docstring o pré-registro (perguntas, desenho, expectativas, critério), commitado antes de rodar, e o
resultado registrado depois. O registro cronológico está em `DIARIO_DE_BORDO.md`, na raiz do repositório.

## Versões exatas

| Peça | Origem | Conferência |
|---|---|---|
| Pacote do detector | `SIMPred-Petrobras/cabiunas-models`, branch `feat/pacote-consolidado`, commit `e17bf27` (`estrutura_pra_prod/Cabiunas`) | sha256 de `scripts/cabiunas_inference.py` = `a4926a2c55fa0317…` |
| `nv` (régua, `sinais_causais`) | `SIMPred-Petrobras/cabiunas-models`, branch `exp/modos-de-falha`, commit `d77de7d`, `src/nv` | os 22 `.py` idênticos aos do Dataset de insumos |
| Dado bruto | export de 30 s, `sensores_full_2024_2026_30s.csv` (1,06 GB), Dataset ClearML `a97ba56ba14840fbb1125c2a82f883c9` | sha256 `3ee22102e463bfbc…` |
| Insumos montados | Dataset ClearML `f5e1b16b56324d6ab242e49bce224d07`: pacote (`Cabiunas/scripts`, `modelos`, `registro_trips.csv`), `nv` e a série do controle publicado (`controle_atual/serie.csv.gz`, do Dataset `86e3990f`) | |
| Pacote da J17 e da J18 | `SIMPred-Petrobras/cabiunas-models`, branch `feat/pacote-respostas-engenharia`, commit `e733279` (motor no pacote, `estado_saida`, tag ausente degrada o canal); só `scripts/` no Dataset ClearML `bed2e62f62914fa087af7e0a28c9e296`, com `modelos/` e registro do `f5e1b16b` (modelos idênticos ao `e17bf27`) | |
| Grade de 2 min da pipeline (J16) | Dataset ClearML `bb85d853f92545beb771d303354302ab` (`data_2025-01-01_2026-04-30_raw.csv`) | |
| Python | 3.11; `requirements.txt` desta pasta (as versões fixas das tasks) | |

## Três formas de rodar

**1. Só o motor com estado, com os seus dados (sem ClearML).** É o caminho para conferir a entrega de referência:

    pip install pandas numpy pyarrow
    python frente_thallys/motor_estado/teste_equivalencia.py --pacote <pasta Cabiunas> --csv <export de 30 s> \
        [--reinicio] [--procs 8] [--inicio AAAA-MM-DD] [--fim AAAA-MM-DD]

Ele compara o motor com o lote de 60 d a cada 6 h (mais o instante de cada trip) e sai com código 0 se forem iguais
(sinais com |Δ| ≤ 1e-9, máscara e alarme idênticos). No Cica (J15), com o nosso CSV: 1.708 atualizações, 0 linhas
diferentes, com e sem `--reinicio`. Na J17, também sem uma sonda TV_ e sem as 10 no CSV (com o `motor_estado.py` corrigido).

**2. Qualquer J, com os Datasets do ClearML** (precisa de credenciais do servidor do Cica):

    pip install -r frente_thallys/requirements.txt
    python frente_thallys/jN_....py --enqueue     # manda uma task para a fila default do Cica
    python frente_thallys/jN_....py               # roda na sua máquina, baixando os Datasets

**3. Qualquer J, localmente, com os seus clones** (`--local`). Aponte os caminhos por variável de ambiente:

    export FT_PKG=<cabiunas-models em e17bf27>/estrutura_pra_prod/Cabiunas
    export FT_NV=<cabiunas-models em d77de7d>/src/nv
    export FT_A97=<pasta>/sensores_full_2024_2026_30s.csv
    export FT_ATUAL=<pasta>/serie.csv.gz          # série do controle publicado (comparação V1 do J1)
    python frente_thallys/jN_....py --local [--teste]

`--teste` corta o período (até abr–mai/2025) para conferir o encanamento em poucos minutos.

## Os testes

| Teste | Script | Task no Cica | Pergunta |
|---|---|---|---|
| J1 | `j1_janela_entrada.py` | `397a6e11` | Janela de entrada do lote: qualidade de 30 a 90 d |
| J2 | `j2_estado_minimo.py` | `31f31a1a` | Estado mínimo: que parte pede histórico |
| J34 | `j34_estado_robustez.py` | `24c98ea5` | Motor com estado: reinício, buracos, sondas, atrasos, trips tardios, instalação nova |
| J5 | `j5_retreino_janela.py` | `4feb79c4` | Janela de leitura do retreino mensal |
| J6 | `j6_bundle_atrasado.py` | `4c542c2b` | Bundle mensal atrasado |
| J7 | `j7_custo_refazer.py` | `e9ce6fc5` | Limite do "refazer" pelo estado |
| J8 | `j8_foto_blocos.py` | `0355fa6d` | Correção antiga dentro de 1 M células (foto diária + blocos) |
| J9 | `j9_carga_blocos.py` | `c9411e84` | Carga inicial de instalação nova em blocos |
| J10 | `j10_custo_sem_tracemalloc.py` | `1d3d16f1` | Custo só do cálculo; grade de 2 min em trechos de 3 d |
| J11 | `j11_telemetria_e_sem_vb.py` (+ `j11_motor_cmd.py`) | `8b7268f0` | Custo com o teu `telemetria_execucao.py` (cópia sem alteração em `terceiros/`); detecção sem o vb |
| J12 | `j12_base_vb_parcial.py` | `ce91ed2f` | Detecção com a base do vb incompleta |
| J13 | `j13_base_vb_ganho.py` | `e7574f0d` | Base do vb curta: ganho real ou mais horas de alarme? |
| J14 | `j14_ram_lote.py` (+ `j14_sonda_ram.py`) | `760f35b1` | RAM do lote 2025 × 2026; execução com 1 thread |
| J15 | `j15_motor_modulo.py` (usa `motor_estado/`) | `244691f9` | Motor com estado como módulo separado contra o lote de 60 d |
| J16 | `j16_grade_2min_fonte.py` | `003a69f8` | A grade de 2 min que alimenta a pipeline é a mediana do export de 30 s? |
| J17 | `j17_sonda_ausente.py` (usa `motor_estado/` e o pacote `e733279`) | `896d034c` | Sonda de vibração ausente do CSV derruba o motor? A correção iguala o lote? |
| J18 | `j18_e733279_ram_plena.py` (usa `j11`, `j34` e o pacote `e733279`) | `4e026612` | RAM do lote no `e733279` × `e17bf27`; "`PLENA` em" pela regra do `estado_saida` nas 30 instalações do J34 |
| J19 | `j19_plena_threads.py` (+ `j19_motor_cmd.py`; usa `j18` e o pacote `e733279`) | `ade8113e` | Causa das ~12 h de vb diferente no início da PLENA; regras de PLENA alternativas; motor do módulo com 1 thread |
| J20 | `j20_plena_r60_24.py` (usa `j19`, `j18` e o pacote `e733279`) | `802694ac` | Regra R60_24 de PLENA (135039e) nas 30 instalações: 0 divergindo? |

**Conferência desta branch (21/11/2026).** J16 e J17 foram rodados de novo no Cica a partir da própria
`frente-thallys/reproducao` (commit `22bf4c1`), com resultado idêntico ao original: J16 task `3ec83759`
(|Δ| máx 3,05e-5, NaN idênticos), J17 task `3c0b7822` (A1 quebra; A2, A3 e A4 iguais em 1.708 atualizações).
A J18 rodou primeiro a partir da `exp/ocsvm-validacao` (`d98bc7c`, task `4e026612`) e de novo a partir desta branch
(`3f415c7`, task `fb9618a6`): parte B idêntica (as 30 instalações e o placar); parte A dentro do ruído de medida
(RAM ±2 MB, duração ±0,1 s; lote 90 d na `e733279` 495,3 × 495,6 MB).

## Cuidados

- **Na `e733279` a RAM do lote já não depende do CSV (J18):** 366 MB (60 d) e 495 MB (90 d), iguais em 2025 e 2026.
- **O resultado da J14 depende do nosso CSV.** Os ~800 MB de 2026 vêm de 11 a 14 células de texto do PI ("Comm Fail",
  "Out of Serv") no export `a97ba56b`. Com um CSV sem essas células, o lote de 90 d fica em ~500 MB.
- Os scripts J1–J13 usam o motor antigo (classe `Motor` em `j34_estado_robustez.py`, com estado global). O
  `motor_estado/` é a mesma lógica sem o estado global; a J15 mostrou que reproduz o lote. Os cenários do J34 e as
  recuperações J6–J9 ainda não foram portados para o teste do módulo.
- **1 thread no motor (J19):** o `teste_equivalencia.py` agora fixa OMP/OPENBLAS/MKL_NUM_THREADS = 1 antes do numpy
  (saída idêntica, CPU do passo 4,7 → 0,4 s). O `motor_estado.py` não define isso: fica com o executável que o chama.
- **`motor_estado.py` corrigido na J17:** `_estaveis` usa `.reindex(columns=self.tags)`. Antes, uma sonda TV_ ausente
  do CSV dava `KeyError` na carga; agora vira coluna vazia, como no `preprocessar` do pacote (`a020477`). Com as
  colunas completas, nada muda (J17, A4).
- **A J14 é anterior à correção do `carregar_dados`** (`8364998`, já na `feat/pacote-respostas-engenharia`): com ela,
  o texto do PI vira NaN na leitura e o pico cai (877 → 513 MB no pandas 2.3, medido pelo Thallys).
- "Bit a bit" nos resultados = sinais com |Δ| ≤ 1e-9 e máscara e alarme idênticos (t e p diferem na ordem de 1e-14).
- Tempos e memória são do Cica (Threadripper 7970X, 64 núcleos, 131 GB). Resultados de igualdade e de régua não
  dependem da máquina.
- Subprocessos dentro de task do ClearML precisam rodar sem as variáveis `CLEARML_*`: com `CLEARML_TASK_ID` herdado, o
  ClearML troca os argumentos do filho pelos da task (foi a causa das duas primeiras falhas da J11).
