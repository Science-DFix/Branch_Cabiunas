"""Envia as que faltam do D1, 1 por vez: só com fila vazia e >=2 workers livres; anota cada evento em d1_envios.md."""
import json, subprocess, sys, time
from datetime import datetime, timezone, timedelta
from clearml.backend_api.session.client import APIClient
RAIZ = "/home/dvar/FRENTE_THALLYS/wt-ocsvm"
LOG = f"{RAIZ}/frente_ocsvm/d1_envios.md"
REG = f"{RAIZ}/frente_ocsvm/d1_tasks.json"
FALTAM = sys.argv[1:]
c = APIClient()
agora = lambda: datetime.now(timezone(timedelta(hours=-3))).strftime("%d/%m %H:%M")
def anota(nome, tid, ev):
    with open(LOG, "a") as f: f.write(f"| {agora()} | {nome} | {tid} | {ev} |\n")
    print(agora(), nome, tid, ev, flush=True)
def livre():
    ws = [w for w in c.workers.get_all() if w.id.startswith("clearml-worker")]
    q = c.queues.get_all(name="default")[0]
    return len(q.entries) == 0 and sum(w.task is None for w in ws) >= 2
for nome in FALTAM:
    reg = json.load(open(REG))
    if nome not in reg:
        while not livre(): time.sleep(120)
        subprocess.run(["python", "frente_ocsvm/d1_disputa.py", "envia", "--so", nome], cwd=RAIZ, check=True)
        reg = json.load(open(REG))
        anota(nome, reg[nome], "enfileirada")
    tid = reg[nome]
    visto = None
    while True:
        t = c.tasks.get_all(id=[tid], only_fields=["status", "status_message"])[0]
        st = str(getattr(t.status, "value", t.status))
        if st != visto:
            visto = st
            if st != "queued": anota(nome, tid, st)
        if st in ("completed", "failed", "stopped", "closed"): break
        time.sleep(60)
    if visto != "completed":
        anota(nome, tid, f"PAROU a fila: {visto} ({t.status_message})"); sys.exit(1)
anota("—", "—", "fila do D1 concluída")
