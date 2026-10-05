#!/usr/bin/env python3
"""Telemetria de uma execução: roda um comando e mede o que ele consome. Só biblioteca padrão, Linux.

PARA QUÊ. A engenharia perguntou como o pacote consome recursos "de fato em produção". Este arquivo roda o PRÓPRIO
comando do pacote (inferência ou retreino) e registra, por tentativa, o que cada pergunta pede. Não altera o pacote.

COMO SE USA (o comando vem depois de `--`):
    python3 telemetria_execucao.py --rotulo retreino --log telemetria.jsonl --saidas /caminho/modelos \\
        -- python3 constroi_bundle.py --historico dados.csv --mes 2026-03
    python3 telemetria_execucao.py --rotulo inferencia --json-estado estado.json --tentativas 3 --ok 0,2,3 \\
        -- python3 tc33003a_exemplo.py --dias 60 --json estado.json

O QUE MEDE, e como (a diferença entre "exato" e "amostrado" é dita, porque muda a leitura):
  duração              relógio, do início ao fim do comando.
  CPU                  tempo de CPU (usuário + sistema) EXATO, do rusage do processo filho; média = tempo de CPU /
                       duração, em núcleos; PICO AMOSTRADO a cada --intervalo s sobre a árvore de processos.
  RAM                  pico do MAIOR processo (EXATO, rusage) e pico da SOMA da árvore (amostrado).
  disco                blocos lidos e escritos na camada de bloco (EXATO, rusage; não conta o que veio do cache de
                       página) e bytes lógicos lidos e escritos (rchar/wchar de /proc/PID/io, amostrado: o final de
                       um processo muito curto pode escapar). Se o cgroup v2 estiver acessível, também os bytes
                       lidos/escritos e o tempo de CPU do cgroup inteiro (exatos, mas do contêiner todo).
  espaço temporário    o TMPDIR do comando é uma pasta nova; mede-se o MAIOR tamanho que ela atingiu (amostrado).
  tamanho das saídas   o tamanho de cada caminho em --saidas, depois da execução (arquivo ou pasta).
  falhas e retries     código de saída por tentativa; --tentativas N repete se o código não estiver em --ok.
  atraso               com --json-estado (o contrato do dashboard): gerado_em - fim da janela (atraso do dado até a
                       execução), e para cada episódio o instante em que ele PODE ser confirmado (início + 120 min)
                       contra gerado_em.
Cada tentativa vira uma linha JSON em --log (e um resumo no terminal).
"""
from __future__ import annotations
import argparse, json, os, platform, resource, shutil, socket, subprocess, sys, tempfile, time
from datetime import datetime, timezone

CLK = os.sysconf("SC_CLK_TCK")
PAGINA = os.sysconf("SC_PAGE_SIZE")


def ler(caminho: str) -> str | None:
    try:
        with open(caminho) as f:
            return f.read()
    except OSError:
        return None


def tabela_processos() -> dict[int, tuple[int, int, int]]:
    """pid -> (ppid, ticks de CPU usuário+sistema, RSS em bytes), para todos os processos legíveis."""
    t = {}
    for nome in os.listdir("/proc"):
        if not nome.isdigit():
            continue
        s = ler(f"/proc/{nome}/stat")
        if not s:
            continue
        r = s[s.rfind(")") + 2:].split()
        try:
            t[int(nome)] = (int(r[1]), int(r[11]) + int(r[12]), int(r[21]) * PAGINA)
        except (IndexError, ValueError):
            continue
    return t


def arvore(raiz: int, t: dict) -> set[int]:
    filhos: dict[int, list[int]] = {}
    for pid, (pp, _, _) in t.items():
        filhos.setdefault(pp, []).append(pid)
    out, pilha = set(), [raiz]
    while pilha:
        p = pilha.pop()
        if p in t and p not in out:
            out.add(p); pilha.extend(filhos.get(p, []))
    return out


def io_processo(pid: int) -> tuple[int, int]:
    s = ler(f"/proc/{pid}/io")
    if not s:
        return 0, 0
    d = dict(l.split(": ") for l in s.strip().splitlines())
    return int(d.get("rchar", 0)), int(d.get("wchar", 0))


def tamanho(caminho: str) -> int:
    if os.path.isfile(caminho):
        return os.path.getsize(caminho)
    n = 0
    for raiz, _, arqs in os.walk(caminho):
        for a in arqs:
            try:
                n += os.path.getsize(os.path.join(raiz, a))
            except OSError:
                pass
    return n


def cgroup() -> dict | None:
    """Contadores do cgroup v2 deste processo (cpu.stat e io.stat), ou None se não houver acesso."""
    rel = (ler("/proc/self/cgroup") or "").strip().split("::")[-1] if "::" in (ler("/proc/self/cgroup") or "") else None
    if rel is None:
        return None
    base = "/sys/fs/cgroup" + rel
    cpu, io = ler(f"{base}/cpu.stat"), ler(f"{base}/io.stat")
    if cpu is None or io is None:
        return None
    r = {"cpu_usec": int(dict(l.split() for l in cpu.splitlines()).get("usage_usec", 0)), "rbytes": 0, "wbytes": 0}
    for linha in io.splitlines():
        for kv in linha.split()[1:]:
            k, _, v = kv.partition("=")
            if k in ("rbytes", "wbytes"):
                r[k] += int(v)
    pico = ler(f"{base}/memory.peak")
    r["memoria_pico_cgroup_mb"] = round(int(pico) / 1e6, 1) if pico and pico.strip().isdigit() else None
    return r


def uma_tentativa(cmd: list[str], intervalo: float, saidas: list[str], rotulo: str, n: int) -> dict:
    tmp = tempfile.mkdtemp(prefix="telemetria_tmp_")
    env = dict(os.environ, TMPDIR=tmp)
    cg0 = cgroup()
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, env=env)
    rss_pico = cpu_pico = 0.0
    tmp_pico = 0
    ultimo_t, ultimo_cpu = time.perf_counter(), {}
    io_vivo: dict[int, tuple[int, int]] = {}
    io_final = [0, 0]
    while True:
        pid_fim, status, ru = os.wait4(p.pid, os.WNOHANG)
        tab = tabela_processos()
        ar = arvore(p.pid, tab)
        agora = time.perf_counter()
        if ar:
            rss_pico = max(rss_pico, sum(tab[x][2] for x in ar))
            cpu_atual = {x: tab[x][1] for x in ar}
            comum = [x for x in cpu_atual if x in ultimo_cpu]
            if comum and agora > ultimo_t:
                cpu_pico = max(cpu_pico, sum(cpu_atual[x] - ultimo_cpu[x] for x in comum) / CLK / (agora - ultimo_t))
            ultimo_cpu, ultimo_t = cpu_atual, agora
            novos = {x: io_processo(x) for x in ar}
            for x in set(io_vivo) - set(novos):                # o processo acabou: guarda o último valor visto
                io_final[0] += io_vivo[x][0]; io_final[1] += io_vivo[x][1]
            io_vivo = novos
        tmp_pico = max(tmp_pico, tamanho(tmp))
        if pid_fim:
            break
        time.sleep(intervalo)
    wall = time.perf_counter() - t0
    for x in io_vivo:
        io_final[0] += io_vivo[x][0]; io_final[1] += io_vivo[x][1]
    cg1 = cgroup()
    codigo = os.waitstatus_to_exitcode(status)
    cpu_s = ru.ru_utime + ru.ru_stime
    r = dict(rotulo=rotulo, tentativa=n, comando=" ".join(cmd), codigo_saida=codigo, duracao_s=round(wall, 3),
             cpu_usuario_s=round(ru.ru_utime, 3), cpu_sistema_s=round(ru.ru_stime, 3), cpu_media_nucleos=round(cpu_s / wall, 3) if wall else None,
             cpu_pico_nucleos_amostrado=round(cpu_pico, 3), ram_pico_maior_processo_mb=round(ru.ru_maxrss / 1024, 1),
             ram_pico_arvore_mb_amostrado=round(rss_pico / 1e6, 1), disco_blocos_lidos_mb=round(ru.ru_inblock * 512 / 1e6, 2),
             disco_blocos_escritos_mb=round(ru.ru_oublock * 512 / 1e6, 2), bytes_logicos_lidos_mb_amostrado=round(io_final[0] / 1e6, 1),
             bytes_logicos_escritos_mb_amostrado=round(io_final[1] / 1e6, 1), temporario_max_mb_amostrado=round(tmp_pico / 1e6, 2),
             falhas_de_pagina_maiores=ru.ru_majflt, intervalo_s=intervalo)
    if cg0 and cg1:
        r.update(cgroup_cpu_s=round((cg1["cpu_usec"] - cg0["cpu_usec"]) / 1e6, 3), cgroup_lidos_mb=round((cg1["rbytes"] - cg0["rbytes"]) / 1e6, 2),
                 cgroup_escritos_mb=round((cg1["wbytes"] - cg0["wbytes"]) / 1e6, 2), cgroup_memoria_pico_mb=cg1["memoria_pico_cgroup_mb"])
    r["saidas_mb"] = {s: round(tamanho(s) / 1e6, 4) for s in saidas if os.path.exists(s)}
    shutil.rmtree(tmp, ignore_errors=True)
    return r


def atraso(json_estado: str) -> dict:
    j = json.load(open(json_estado))
    ger = datetime.fromisoformat(j["gerado_em"].replace("Z", "+00:00")); fim = datetime.fromisoformat(j["janela"]["fim"].replace("Z", "+00:00"))
    out = {"gerado_em": j["gerado_em"], "dado_ate": j["janela"]["fim"], "atraso_do_dado_ate_a_execucao_min": round((ger - fim).total_seconds() / 60, 1), "episodios": []}
    for e in j.get("episodios", []):
        ini = datetime.fromisoformat(e["inicio"].replace("Z", "+00:00"))
        out["episodios"].append({"id": e["id"], "inicio": e["inicio"], "confirmavel_em": (ini.replace(tzinfo=timezone.utc) + __import__("datetime").timedelta(minutes=120)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 "em_curso": e["em_curso"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rotulo", default="execucao"); ap.add_argument("--log", default=None); ap.add_argument("--intervalo", type=float, default=0.1)
    ap.add_argument("--tentativas", type=int, default=1); ap.add_argument("--ok", default="0", help="códigos de saída que NÃO pedem nova tentativa (padrão: 0)")
    ap.add_argument("--saidas", nargs="*", default=[]); ap.add_argument("--json-estado", default=None)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("falta o comando depois de --")
    ok = {int(x) for x in a.ok.split(",")}
    sistema = dict(host=socket.gethostname(), so=platform.platform(), nucleos=os.cpu_count(), python=platform.python_version(),
                   cpu=next((l.split(":", 1)[1].strip() for l in (ler("/proc/cpuinfo") or "").splitlines() if l.startswith("model name")), ""),
                   ram_total_gb=round(int(next((l.split()[1] for l in (ler("/proc/meminfo") or "").splitlines() if l.startswith("MemTotal")), 0)) / 1e6, 1))
    resultados = []
    for n in range(1, a.tentativas + 1):
        r = uma_tentativa(cmd, a.intervalo, a.saidas, a.rotulo, n)
        r["em"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"); r["sistema"] = sistema
        if a.json_estado and os.path.exists(a.json_estado) and r["codigo_saida"] in ok:
            r["atraso"] = atraso(a.json_estado)
        resultados.append(r)
        if a.log:
            with open(a.log, "a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[telemetria] {a.rotulo} #{n}: saída {r['codigo_saida']} | {r['duracao_s']} s | CPU média {r['cpu_media_nucleos']} núcleos (pico {r['cpu_pico_nucleos_amostrado']}) | "
              f"RAM pico {r['ram_pico_maior_processo_mb']} MB | disco lido/escrito {r['disco_blocos_lidos_mb']}/{r['disco_blocos_escritos_mb']} MB | temp máx {r['temporario_max_mb_amostrado']} MB",
              file=sys.stderr, flush=True)
        if r["codigo_saida"] in ok:
            break
        time.sleep(1.0)
    return resultados[-1]["codigo_saida"]


if __name__ == "__main__":
    raise SystemExit(main())
