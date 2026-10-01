#!/usr/bin/env python3
"""Monta o relatório de experimentos e gera o PDF.

As fontes são `parte1.html` e `parte2.html`. O estilo (cores, tipografia, tabelas)
vem do cabeçalho de `estrutura_pra_prod/Cabiunas/scripts/_pdf/integracao.html`, e o
PDF sai pelo mesmo `build_pdf.py` (Chrome headless) dos outros relatórios.

Uso:  python3 monta.py            # -> RELATORIO_EXPERIMENTOS_TC33003A.pdf na raiz
"""
import subprocess, sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
PDF = RAIZ / "estrutura_pra_prod/Cabiunas/scripts/_pdf"

estilo = (PDF / "integracao.html").read_text(encoding="utf-8")
estilo = estilo[:estilo.index("<header>")].replace(
    "<title>Integração TC-33003A</title>", "<title>Relatório de experimentos TC-33003A</title>")
html = AQUI / "relatorio_experimentos.html"
html.write_text(estilo + (AQUI / "parte1.html").read_text(encoding="utf-8")
                + (AQUI / "parte2.html").read_text(encoding="utf-8"), encoding="utf-8")
saida = RAIZ / "RELATORIO_EXPERIMENTOS_TC33003A.pdf"
sys.exit(subprocess.run([sys.executable, str(PDF / "build_pdf.py"), str(html), str(saida)]).returncode)
