"""Q3 robustness (2026-10-17): sprint.plant seeds the start state with q0 + 1e-9*(1 + len(_PL) + 7 uid), so a rung's plant inside a
ladder differs from a fresh process's by ~1e-9 rad. Replay each stored z after pre-building n_pre dummy plants (n_pre in 2, 4, 6),
i.e. start perturbations 3e-9..7e-9, at 1 ms and 0.5 ms, fresh process per job. Output results/overnight/Q3/perturb_q3.json."""
import os, sys, json, subprocess
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CHILD = r'''
import sys, json; sys.path.insert(0, "src")
import numpy as np
from engine import sprint as SP
pth, ix, dt, npre = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]); R = json.load(open(pth))["rungs"][ix]
for k in range(npre): SP._PL[("dummy", k)] = None
P = SP.Sprint("G1", R["v"], stride=R["stride"] if R.get("stride_extrapolated") else None)
r = P.run(np.asarray(R["z"]), dt); s = SP.summarize(r); print("JSON" + json.dumps(s, default=float))
'''
def job(p, i, lag, rt, dt, npre):
    env = dict(os.environ, SPRINT_LAG=str(lag), SPRINT_RATINGS=rt, OMP_NUM_THREADS="1")
    o = subprocess.run(["nice", "-n", "10", sys.executable, "-c", CHILD, p, str(i), str(dt), str(npre)], cwd=ROOT, env=env, capture_output=True, text=True)
    L = [l for l in o.stdout.splitlines() if l.startswith("JSON")]
    return {**(json.loads(L[0][4:]) if L else dict(error=o.stderr[-500:])), **dict(src=p, rung=i, lag=lag, ratings_job=rt, dt=dt, npre=npre)}
specs = json.loads(sys.argv[1])  # [[path, idx, lag, ratings], ...]
jobs = [(p, i, l, rt, dt, n) for p, i, l, rt in specs for dt in (0.001, 0.0005) for n in (2, 4, 6)]
with ThreadPoolExecutor(3) as ex: res = list(ex.map(lambda a: job(*a), jobs))
json.dump(res, open(f"{ROOT}/results/overnight/Q3/perturb_q3.json", "w"), indent=1)
for r in res: print(os.path.basename(r["src"]), r["rung"], r["lag"], r["ratings_job"], r["dt"], r["npre"], r.get("value"), r.get("valid"), r.get("reasons"), r.get("slip_max"), r.get("error", "")[:200])
