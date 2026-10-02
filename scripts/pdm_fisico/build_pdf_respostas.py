#!/usr/bin/env python3
"""respostas_engenharia.html -> RESPOSTAS_ENGENHARIA_TC33003A.pdf (Chrome headless)."""
import subprocess
from pathlib import Path

AQUI = Path(__file__).resolve().parent
SAIDA = AQUI.parents[1] / "RESPOSTAS_ENGENHARIA_TC33003A.pdf"
subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                f"--print-to-pdf={SAIDA}", f"file://{AQUI / 'respostas_engenharia.html'}"],
               capture_output=True, text=True, timeout=240)
print(f"-> {SAIDA} ({SAIDA.stat().st_size / 1024:.0f} KB)" if SAIDA.exists() else "falhou")
