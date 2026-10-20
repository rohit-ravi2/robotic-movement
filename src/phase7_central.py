"""Phase 7 — re-run the central test at a raised solver budget.

F-011 recorded a NULL: participation ratio flat at 3.44-3.95 across a five-fold demand range
while required_depth predicted 1 throughout. Its stated strongest next test was to raise the
budget until the full-body seed spread falls below 0.2 m/s (it was 0.805 m/s at 150 iters) and
repeat, because the null rested on a solver whose own spread on that quantity exceeded the
effect being sought.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, branch_i as BI
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ITERS, POPSIZE = 500, 20
SEEDS = [1, 2, 3, 4, 5, 6]
TARGETS = [0.5, 0.9, 1.3, 1.7, 2.1, 2.4]

def job(a):
    kind, payload = a
    S = BI.setup()
    if kind == "max":
        seed = payload
        t0 = time.time()
        _, r = BI.solve(set(S["types"]), seed, iters=ITERS, popsize=POPSIZE)
        return dict(kind="max", seed=seed, v=r["v"], height=r["height"],
                    participation=r["participation"], valid=bool(r["valid"]), secs=time.time()-t0)
    tgt, seed = payload
    t0 = time.time()
    _, r = BI.solve(set(S["types"]), seed, target=tgt, iters=ITERS, popsize=POPSIZE)
    return dict(kind="demand", target=tgt, seed=seed, v=r["v"], height=r["height"],
                participation=r["participation"], valid=bool(r["valid"]),
                work=r["work"], secs=time.time()-t0)

if __name__ == "__main__":
    jobs = [("max", s) for s in SEEDS] + [("demand", (t, s)) for t in TARGETS for s in SEEDS]
    print(f"{len(jobs)} solves at {ITERS} iters popsize {POPSIZE}", flush=True)
    t0 = time.time()
    with Pool(14) as p: res = p.map(job, jobs)
    json.dump(dict(iters=ITERS, popsize=POPSIZE, runs=res),
              open(f"{ROOT}/results/central_hi.json", "w"), indent=1, default=float)
    mx = [r for r in res if r["kind"] == "max" and r["valid"]]
    if mx:
        v = np.array([r["v"] for r in mx])
        print(f"full-body max: best {v.max():.3f} mean {v.mean():.3f} SEED SPREAD {v.max()-v.min():.3f} "
              f"(was 0.805 at 150 iters; target <0.2)", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)
