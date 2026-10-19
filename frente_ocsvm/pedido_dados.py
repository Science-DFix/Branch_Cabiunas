#!/usr/bin/env python3
"""Planilha do pedido de dados à equipe da planta (item 3 do relatorio_falsos_positivos; substitui a de 15/10).

Abas para preencher (revisão CEGA: nada indica quais linhas são FP, trip ou "pós-partida" para nós):
  Partidas              as 83 partidas da máquina (07/2024 a 04/2026): tipo de partida
  Paradas               as 31 paradas sem alarme crítico (mesma lista de paradas_revisao.py): programada ou não
  Alertas sem trip      os 41 alertas da pipeline final (com a regra E5) não seguidos de trip nem parada
  Perguntas pontuais    vibração 01-02/2026, PI_6240319_AL, PAL_6240315, rajadas, falhas de comunicação, unidades
Mais a aba de referência com os 8 trips curados.
Saídas em frente_ocsvm/revisao/:
  pedido_dados_TC33003A.xlsx   para a equipe
  pedido_dados_interno.csv     partidas e alertas com a nossa leitura (h após partida, contexto), só para nós

Uso: python frente_ocsvm/pedido_dados.py   (precisa de runs_pipeline_unificada_final/episodios_classificados.csv)
"""
from __future__ import annotations
import json

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment
from openpyxl.worksheet.datavalidation import DataValidation

import comum as C
from e1_triagem_fp import sensores
from paradas_revisao import AMARELO, FONTE, NEGRITO, OPCOES, OPCOES_FP, paradas

SAIDA = C.AQUI / "revisao"
EPIS = C.AQUI.parent / "runs_pipeline_unificada_final" / "episodios_classificados.csv"
RAJADAS = json.loads((C.AQUI.parent / "relatorio_ocsvm_cnn1d_mensal" / "numeros.json").read_text())["rajadas_temp"]
OPCOES_PARTIDA = ["Normal (após parada programada)", "Após trip", "Após manutenção", "Teste / partida abortada",
                  "Não sei"]
NOMES = {"temperatura": "temperatura de mancal", "vibracao": "vibração de mancal", "oleo_pressao": "pressão de óleo"}
BR = pd.Timedelta(hours=3)


def partidas(df) -> pd.DataFrame:
    op = df["operational_state"]
    on = op.eq("on")
    t_on = op.index[on & ~on.shift(fill_value=False)]
    t_off = op.index[~on & on.shift(fill_value=False)]
    L = []
    for t in t_on:
        antes = t_off[t_off < t]
        L.append(dict(t=t, parada_antes_h=round((t - antes[-1]).total_seconds() / 3600, 1) if len(antes) else None))
    return pd.DataFrame(L)


def falhas_comunicacao() -> list:
    _, txt = sensores(inicio="2024-06-25")
    t = txt.index[txt.any(axis=1)]
    t = t[t <= pd.Timestamp("2026-04-20 23:59")]
    grupos = (pd.Series(t).diff() > pd.Timedelta(hours=1)).cumsum()
    return [(g.iloc[0], g.iloc[-1]) for _, g in pd.Series(t).groupby(grupos.values)]


def cabecalho(w, cols, larg):
    for j, (c, l) in enumerate(zip(cols, larg), 1):
        k = w.cell(1, j, c)
        k.font, k.alignment = NEGRITO, Alignment(wrap_text=True, vertical="top")
        w.column_dimensions[k.column_letter].width = l
    w.freeze_panes = "B2"


def linhas(w, rows, amarelo_desde, datas=(2, 3)):
    for i, linha in enumerate(rows):
        for j, v in enumerate(linha, 1):
            k = w.cell(i + 2, j, v)
            k.font, k.alignment = FONTE, Alignment(wrap_text=True, vertical="top")
            if j in datas:
                k.number_format = "dd/mm/yyyy hh:mm"
            if j >= amarelo_desde:
                k.fill = AMARELO


def lista(w, opcoes, col, n):
    dv = DataValidation(type="list", formula1='"' + ",".join(opcoes) + '"', allow_blank=True)
    w.add_data_validation(dv)
    dv.add(f"{col}2:{col}{n + 1}")


def main():
    df = C.carrega_canais()
    a = pd.read_csv(C.DADOS / "alarmes_selecionados_turbina_a.csv")
    a["t"] = pd.to_datetime(a["Data da Ocorrência"], errors="coerce")
    a = a[a.Status.astype(str).str.startswith("ACT")]
    desc = a.drop_duplicates("Tag Alarme").set_index("Tag Alarme")["Descrição Alarme"]
    P = partidas(df)
    S = paradas()
    rev, ref = S[~S.critico].reset_index(drop=True), S[S.critico].reset_index(drop=True)
    E = pd.read_csv(EPIS, parse_dates=["start", "end"])
    fp = E[E.classe == "falso_positivo"].reset_index(drop=True)
    cf = falhas_comunicacao()

    wb = Workbook()
    ws = wb.active
    ws.title = "Instruções"
    texto = [
        ("Pedido de dados -- turbina TC-33003A (Cabiúnas), período 07/2024 a 04/2026", NEGRITO),
        ("", FONTE),
        ("Contexto: o modelo de detecção de precursores de trip acerta os 8 trips do período, mas emite cerca de 3 alertas "
         "por mês que não são seguidos de trip nem de parada. Para entender esses alertas precisamos de informação que "
         "não está nos sinais. O documento 'pedido_de_dados.pdf' explica o porquê de cada pedido.", FONTE),
        ("", FONTE),
        ("Abas para preencher (colunas amarelas; onde houver lista, escolher uma opção):", NEGRITO),
        (f"1. Partidas: as {len(P)} vezes em que a máquina voltou a operar. Tipo de partida e observação.", FONTE),
        (f"2. Paradas: as {len(rev)} paradas sem alarme crítico. Programada, operacional ou falha.", FONTE),
        (f"3. Alertas sem trip: {len(fp)} alertas do modelo sem trip nem parada nas 48 h seguintes. O que estava "
         "acontecendo? Manobra, problema de instrumento, evento de processo, nada de anormal.", FONTE),
        ("4. Perguntas pontuais: perguntas sobre períodos e alarmes específicos.", FONTE),
        ("A aba 'Trips curados (referência)' mostra os 8 trips já reconhecidos, só para comparação.", FONTE),
        ("", FONTE),
        ("Fonte: de onde veio a informação (livro de turno, nota/ordem SAP, registro de partidas e paradas, memória da "
         "equipe...). 'Não sei' também é uma resposta útil.", FONTE),
        ("Horários: 'Brasília' = UTC-3; 'UTC' é o relógio dos dados. Partidas, paradas e alertas são derivados dos "
         "sinais, não de registro da operação, e podem ter alguns minutos de diferença.", FONTE),
    ]
    for i, (s, f) in enumerate(texto, 1):
        ws.cell(i, 1, s).font = f
        ws.cell(i, 1).alignment = Alignment(wrap_text=True)
    ws.column_dimensions["A"].width = 130

    w = wb.create_sheet("1. Partidas")
    cabecalho(w, ["Nº", "Partida (Brasília)", "Partida (UTC)", "Máquina parada antes (h)", "Tipo de partida",
                  "Observação", "Fonte"], [5, 18, 18, 12, 32, 45, 25])
    linhas(w, [[i + 1, (r.t - BR).to_pydatetime(), r.t.to_pydatetime(), r.parada_antes_h, "", "", ""]
               for i, r in P.iterrows()], 5)
    lista(w, OPCOES_PARTIDA, "E", len(P))

    w = wb.create_sheet("2. Paradas")
    cabecalho(w, ["Nº", "Início (Brasília)", "Início (UTC)", "Duração da parada (h)", "Em operação antes (h)",
                  "Alarmes do catálogo de 2 h antes a 15 min depois", "Classificação", "Motivo / observação", "Fonte"],
              [5, 18, 18, 12, 12, 70, 30, 40, 25])
    linhas(w, [[i + 1, (r.t - BR).to_pydatetime(), r.t.to_pydatetime(), r.duracao_h, r.rodando_h, r.alarmes or "-",
                "", "", ""] for i, r in rev.iterrows()], 7)
    lista(w, OPCOES, "G", len(rev))

    w = wb.create_sheet("3. Alertas sem trip")
    cabecalho(w, ["Nº", "Início (Brasília)", "Fim (Brasília)", "Duração (h)", "Sinais que acenderam no modelo",
                  "Alarmes do catálogo nas 24 h antes do alerta", "Classificação", "O que estava acontecendo?",
                  "Fonte"], [5, 18, 18, 10, 30, 55, 32, 45, 25])
    rows = []
    for i, r in fp.iterrows():
        tags = [t for t in str(r.tags_canal4).split(",") if t and t != "nan"]
        rows.append([i + 1, (r.start - BR).to_pydatetime(), (r.end - BR).to_pydatetime(),
                     round((r.end - r.start).total_seconds() / 3600, 1),
                     ", ".join(NOMES.get(m, m) for m in str(r.modelos_acesos).split("+") if m and m != "nan") or "-",
                     "; ".join(f"{t} ({desc.get(t, '')})" for t in tags) or "-", "", "", ""])
    linhas(w, rows, 7)
    lista(w, OPCOES_FP, "G", len(fp))

    w = wb.create_sheet("4. Perguntas pontuais")
    cabecalho(w, ["Nº", "Pergunta", "Período / datas (Brasília)", "Por que perguntamos", "Resposta", "Fonte"],
              [5, 60, 40, 55, 50, 25])
    br = lambda t: (pd.Timestamp(t) - BR).strftime("%d/%m/%Y %H:%M")
    Q = [
        ("Houve inspeção, ordem de serviço ou intervenção na vibração dos mancais (TV_351 a TV_355) nesse período?",
         "31/01/2026 a 14/02/2026",
         "O modelo de vibração ficou aceso duas semanas seguidas, gerando 5 alertas sem trip. Parece um único evento."),
        ("PI_6240319_AL (Falha PI Gás p/ Motor de Partida) deixou de alarmar depois de 27/02/2026. Houve reparo, "
         "troca de instrumento, mudança de setpoint ou desabilitação do alarme?",
         "a partir de 27/02/2026",
         "É o alarme mais usado pelo modelo (409 ativações desde 07/2024). Sem ele o modelo muda de comportamento."),
        ("PI_6240319_AL e PAH_6240319 são do gás do motor de partida. É esperado que alarmem a cada partida, ou "
         "fora dela? O alarme é considerado 'incômodo' pela operação?",
         "07/2024 a 04/2026",
         "Metade dos alertas sem trip começa logo depois de uma partida, e esse alarme está em quase todos."),
        ("PAL_6240315 (Pr. Baixa Gás Combustível) alarmou 326 vezes desde 07/2024. É um alarme conhecido como "
         "recorrente? Há explicação operacional (pressão de suprimento, manobra)?",
         "07/2024 a 04/2026", "É o segundo alarme mais usado pelo modelo e toca sem trip na maioria das vezes."),
        ("Nessas datas, 5 ou mais alarmes de temperatura (TC382_01 a 06, TI_6240315/17, T5_AVG) tocaram no mesmo "
         "minuto sem mudança no sinal. Foram eventos do sistema de aquisição (reinício, teste, troca de cartão)?",
         "; ".join(br(t) for t in RAJADAS),
         "Parecem eventos de instrumentação. A de 14/01/2026 coincide com uma falha de comunicação."),
        ("Nessas datas os sensores gravaram 'Comm Fail'/'Out of Serv'. Qual foi a causa?",
         "; ".join(f"{br(a_)} a {br(b_)}" for a_, b_ in cf),
         "Confirmar que são falhas do sistema de aquisição (a regra nova do modelo já ignora esses trechos)."),
        ("Quais são as unidades de engenharia e as faixas normais dos tags usados pelo modelo? (TC382_03_A, T5_AVG_A, "
         "TV_351X/Y a TV_355X/Y, 954005_624_PI_0308, 954005_624_PDIT_0305)", "-",
         "Os dados vêm sem unidade; isso dificulta a leitura física dos alertas."),
        ("O 954005_624_PI_0308 (pressão de óleo) é gravado com compressão/interpolação no historiador? É possível "
         "exportá-lo na resolução nativa?", "-", "Cerca de 13% dos pontos estão em trechos de reta (interpolados)."),
    ]
    linhas(w, [[i + 1, q, per, porque, "", ""] for i, (q, per, porque) in enumerate(Q)], 5, datas=())

    w = wb.create_sheet("Trips curados (referência)")
    cabecalho(w, ["Nº", "Parada (Brasília)", "Parada (UTC)", "Duração da parada (h)", "Em operação antes (h)",
                  "Alarmes do catálogo de 2 h antes a 15 min depois", "Trip curado"], [5, 18, 18, 12, 12, 70, 14])
    linhas(w, [[i + 1, (r.t - BR).to_pydatetime(), r.t.to_pydatetime(), r.duracao_h, r.rodando_h, r.alarmes or "-",
                r.curado] for i, r in ref.iterrows()], 99)

    SAIDA.mkdir(exist_ok=True)
    wb.save(SAIDA / "pedido_dados_TC33003A.xlsx")
    # nossa leitura (não vai para a equipe)
    interno = pd.concat([P.assign(aba="partidas"),
                         fp[["start", "end", "contexto", "h_desde_partida", "modelos_acesos", "tags_canal4"]]
                         .assign(aba="alertas_sem_trip")])
    interno.to_csv(SAIDA / "pedido_dados_interno.csv", index=False)
    print(f"partidas {len(P)} | paradas p/ revisão {len(rev)} (trips de referência {len(ref)}) | alertas sem trip "
          f"{len(fp)} | perguntas {len(Q)} | falhas de comunicação {len(cf)} trechos | rajadas {len(RAJADAS)}")
    print("FP por contexto:", fp.contexto.str.replace(r" \(.*", "", regex=True).value_counts().to_dict())


if __name__ == "__main__":
    main()
