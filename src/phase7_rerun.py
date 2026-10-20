"""Re-run the key dynamic results on the CORRECTED G1 model (torque motors, locked DOF,
true ground placement). Supersedes F-011, F-012, F-013, F-014, F-015."""
import sys, os, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, branch_i as BI, bank as BK
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ITERS, POPSIZE = 250, 20
SEEDS = [1, 2, 3, 4, 5]
TARGETS = [0.6, 1.0, 1.4, 1.8, 2.2, 2.5]
CHAIN = ["ankle_pitch_joint", "knee_joint", "hip_pitch_joint",
         "waist_pitch_joint", "shoulder_pitch_joint", "elbow_joint"]
MULTISET = [2.2, 1.7, 1.3, 0.9, 0.6, 0.45]

def job(a):
    kind, payload = a
    S = BI.setup()
    if kind == "max":
        seed = payload
        _, r = BI.solve(set(S["types"]), seed, iters=ITERS, popsize=POPSIZE)
    elif kind == "demand":
        tgt, seed = payload
        _, r = BI.solve(set(S["types"]), seed, target=tgt, iters=ITERS, popsize=POPSIZE)
    else:
        cond, seed = payload
        order = list(MULTISET) if cond == "PROXIMAL_STRONG" else list(reversed(MULTISET))
        sc = {t: (s, s) for t, s in zip(CHAIN, order)}
        _, r = BI.solve(set(S["types"]), seed, caps_scale=sc, iters=ITERS, popsize=POPSIZE)
    return dict(kind=kind, payload=str(payload), v=r["v"], height=r["height"],
                participation=r["participation"], valid=bool(r["valid"]),
                takeoff_t=r["takeoff_t"], work=r["work"], grf_bw=r["grf_bw"],
                max_penetration=r["max_penetration"])

if __name__ == "__main__":
    jobs = ([("max", s) for s in SEEDS]
            + [("demand", (t, s)) for t in TARGETS for s in SEEDS]
            + [("asym", (c, s)) for c in ("PROXIMAL_STRONG", "DISTAL_STRONG") for s in SEEDS])
    print(f"{len(jobs)} solves on the corrected model", flush=True)
    t0 = time.time()
    with Pool(14) as p: res = p.map(job, jobs)
    json.dump(res, open(f"{ROOT}/results/rerun.json", "w"), indent=1, default=float)
    print(f"done in {time.time()-t0:.0f}s", flush=True)
