import pandas as pd

OUT = "/tmp/claude-1000/-home-dvar-REPO-CABIUNAS/85b4269a-b5f0-450b-9615-b38e2751d331/scratchpad/relatorio_exp20"
df = pd.read_csv(f"{OUT}/tabela_acertos.csv", parse_dates=["alarme"])

def esc(s):
    return str(s).replace("_", "\\_")

rows = []
for _, r in df.iterrows():
    alarme_str = r["alarme"].strftime("%Y-%m-%d %H:%M")
    tag = esc(r["tag"])
    if r["hit"]:
        if r["tipo"] == "antecedencia":
            status = "Detectado"
            lead = f"{r['lead_horas']:.1f}\\,h"
        else:
            status = "Detectado"
            lead = f"{r['lead_horas']:.1f}\\,h (pós)"
    else:
        status = "Perdido"
        lead = "---"
    rows.append(f"{alarme_str} & {tag} & {status} & {lead} \\\\")

table_body = "\n".join(rows)

with open(f"{OUT}/relatorio_exp20.tex", encoding="utf-8") as f:
    content = f.read()

content = content.replace("INPUTROWS", table_body)

with open(f"{OUT}/relatorio_exp20.tex", "w", encoding="utf-8") as f:
    f.write(content)

print("done,", len(rows), "rows")
