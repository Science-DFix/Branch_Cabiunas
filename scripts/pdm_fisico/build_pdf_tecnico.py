#!/usr/bin/env python3
"""relatorio_tecnico.html -> RELATORIO_TECNICO_DETECTOR_TC33003A.pdf (Chrome headless)."""
import subprocess, sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
SAIDA = Path(sys.argv[1]) if len(sys.argv) > 1 else AQUI.parents[1] / "RELATORIO_TECNICO_DETECTOR_TC33003A.pdf"
subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                f"--print-to-pdf={SAIDA}", f"file://{AQUI / 'relatorio_tecnico.html'}"],
               capture_output=True, text=True, timeout=240)
print(f"-> {SAIDA} ({SAIDA.stat().st_size / 1024:.0f} KB)" if SAIDA.exists() else "falhou")
