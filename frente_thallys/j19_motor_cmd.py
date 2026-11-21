#!/usr/bin/env python3
"""Comando de UM passo do motor do módulo `motor_estado/` (o que vai ao pacote), para o medidor de telemetria (J19).

Igual ao `j11_motor_cmd.py`, mas com `MotorEstado` em vez da classe do J34, e grava a saída do passo para comparar
execuções (1 thread × padrão). As threads vêm do ambiente de quem chama.

    python j19_motor_cmd.py <pasta com bruto3.csv, tab.parquet, buf.parquet> <pasta scripts> <pasta modelos> <saida.parquet>
"""
import sys
from pathlib import Path

pasta, scripts, modelos, saida = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
sys.path[:0] = [str(Path(__file__).resolve().parent / "motor_estado"), scripts]
import cabiunas_inference as ci                  # noqa: E402
from motor_estado import MotorEstado             # noqa: E402

df = ci.preparar_grade(ci.carregar_dados(pasta / "bruto3.csv"))
mod = ci.carregar_modelos(modelos)
trips = sorted(ci.carregar_trips(Path(modelos).parent / "registro_trips.csv"))
m = MotorEstado.le(ci, mod, pasta)
m.tab = m.tab.iloc[:-1]; m.ultimo = m.tab.index[-1]
out = m.passo(df, df.index[-1], trips)
out.to_parquet(saida)
print(f"motor: {len(out)} linha(s) nova(s)")
