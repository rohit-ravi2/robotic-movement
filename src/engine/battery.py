"""Scenario battery for the construction procedure: run construct.py over scenarios x seeds in parallel, log one line per run.
usage: battery.py <iters> <seeds> <workers> [scenario,...]   -> results/construct/*.json + results/construct/battery.log"""
import sys, os, subprocess, itertools, time; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import core, construct as C
iters = int(sys.argv[1]); seeds = int(sys.argv[2]); workers = int(sys.argv[3]); scen = sys.argv[4].split(",") if len(sys.argv) > 4 else list(C.SCENARIOS)
jobs = [(s, k) for s in scen for k in range(seeds)]; od = f"{core.RES}/construct"; os.makedirs(od, exist_ok=True); log = open(f"{od}/battery.log", "a"); py = sys.executable; here = os.path.dirname(os.path.abspath(__file__))
running = []
while jobs or running:
    while jobs and len(running) < workers:
        s, k = jobs.pop(0)
        if os.path.exists(f"{od}/{s}_s{k}.json"): continue
        p = subprocess.Popen([py, f"{here}/construct.py", s, str(k), str(iters)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=os.path.dirname(here)); running.append((s, k, p, time.time()))
    for r in list(running):
        s, k, p, t0 = r
        if p.poll() is not None:
            out = [l for l in p.stdout.read().splitlines() if l.startswith("[")]; line = out[-1] if out else f"[{s} s{k}] FAILED rc {p.returncode}"
            print(line, flush=True); log.write(line + "\n"); log.flush(); running.remove(r)
    time.sleep(2)
