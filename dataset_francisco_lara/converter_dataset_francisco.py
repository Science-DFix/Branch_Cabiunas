"""Converte o dataset do Francisco (ClearML dataset_id
8b06a98f8b264820a9ecf2075a188395, `TC33003A_detector_fisico_entradas`)
para o formato que nossa pipeline (src/cnn1d_ae/io.py) espera:

1. `grade2min.parquet` (37 sensores, grade de 2min, index tz-aware UTC
   com o MESMO relogio de parede que ja usamos -- ver
   docs/analise_pca_monitoramento_sistema.md, achado de fuso) vira
   `sensores_francisco_grade2min.csv` com coluna `data_datetime`
   (naive, mesmo valor de relogio, sem deslocamento).
2. `falhas.csv` (9 eventos curados, colunas evento/dur_h/repeticoes/
   alarmes) vira `alarmes_francisco_falhas.csv` no formato
   Tag Alarme/Data da Ocorrência/Status que io.py exige -- uma linha
   com tag generico `FALHA_CURADA` por evento, mais uma linha com tag
   especifico por categoria (`FALHA_MANCAL`/`FALHA_SELAGEM`/
   `FALHA_OLEO_LUB`, inferido do texto da coluna `alarmes`).

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/dataset_francisco_lara/converter_dataset_francisco.py
"""
import pandas as pd
import os
from clearml import Dataset

OUT = os.path.dirname(os.path.abspath(__file__))
ds = Dataset.get(dataset_id="8b06a98f8b264820a9ecf2075a188395")
root = ds.get_local_copy()

print("lendo grade2min.parquet...", flush=True)
df_sensores = pd.read_parquet(os.path.join(root, "grade2min.parquet"))
df_sensores.index = df_sensores.index.tz_localize(None)
df_sensores.index.name = "data_datetime"
out_sensores = os.path.join(OUT, "sensores_francisco_grade2min.csv")
df_sensores.to_csv(out_sensores)
print(f"salvo: {out_sensores}  ({len(df_sensores)} linhas, {len(df_sensores.columns)} sensores)", flush=True)

print("\nlendo falhas.csv...", flush=True)
df_falhas = pd.read_csv(os.path.join(root, "falhas.csv"))
df_falhas["evento"] = pd.to_datetime(df_falhas["evento"]).dt.tz_localize(None)


def categoria(texto: str) -> str:
    t = str(texto).lower()
    if "manc" in t:
        return "FALHA_MANCAL"
    if "selo" in t or "selagem" in t:
        return "FALHA_SELAGEM"
    if "óleo" in t or "oleo" in t:
        return "FALHA_OLEO_LUB"
    return "FALHA_OUTRA"


rows = []
for _, r in df_falhas.iterrows():
    rows.append({"Tag Alarme": "FALHA_CURADA", "Data da Ocorrência": r["evento"], "Status": "ACT"})
    rows.append({"Tag Alarme": categoria(r["alarmes"]), "Data da Ocorrência": r["evento"], "Status": "ACT"})

df_alarm_out = pd.DataFrame(rows)
out_alarm = os.path.join(OUT, "alarmes_francisco_falhas.csv")
df_alarm_out.to_csv(out_alarm, index=False)
print(f"salvo: {out_alarm}")
print(df_alarm_out.to_string())
