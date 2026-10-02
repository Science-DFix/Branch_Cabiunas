#!/usr/bin/env python3
"""Fontes das respostas a engenharia -> PDF (Chrome headless): producao e pesquisa."""
import subprocess
from pathlib import Path

AQUI = Path(__file__).resolve().parent
DOCS = {"respostas_engenharia.html": "RESPOSTAS_ENGENHARIA_TC33003A.pdf",
        "respostas_engenharia_pesquisa.html": "RESPOSTAS_ENGENHARIA_MODELO_PESQUISA_TC33003A.pdf"}
for fonte, saida in DOCS.items():
    pdf = AQUI.parents[1] / saida
    subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", f"file://{AQUI / fonte}"], capture_output=True, text=True, timeout=240)
    print(f"-> {pdf.name} ({pdf.stat().st_size / 1024:.0f} KB)" if pdf.exists() else f"falhou: {fonte}")
