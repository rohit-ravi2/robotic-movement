"""Phase 11 step 3b: recruitment vs demand WITHOUT the search confound. At each target the FULL chain is
warm-started from that target's converged ARM-ONLY solution (legs, trunk, counter-arm at hold), so legs and trunk
are recruited only if the least-work objective pays for them. Also: arm-only ceiling at 600 and 2400 iterations."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, throw as T, branch_i as BI
from multiprocessing import Pool
OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase11")
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_pitch_joint"]
TARGETS = [3.0, 6.0, 9.0, 12.0, 15.0]

def embed(x_arm, full_types):
    nk = T.NK; xa = np.asarray(x_arm); X = np.zeros(len(full_types)*nk + 1); X[-1] = xa[-1]
    for i, jn in enumerate(ARM):
        j = full_types.index(jn); X[j*nk:(j+1)*nk] = xa[i*nk:(i+1)*nk]
    return X

def job(a):
    t0 = time.time()
    SA = T.setup("G1", driven=ARM); BI.apply_dt(SA, T.DT_OPT)
    xa = T.solve(SA, a["seed"], iters=a["iters_arm"], pop=16, target=a["target"])
    ra = T.measure(SA, xa); ra.pop("trace")
    out = dict(a, arm=ra)
    if a.get("full", True):
        S = T.setup("G1"); BI.apply_dt(S, T.DT_OPT)
        x0 = embed(xa, S["types"])
        xf = T.solve(S, a["seed"], iters=a["iters_full"], pop=16, target=a["target"], x0=x0, sigma0=0.05)
        rf = T.measure(S, xf); rf.pop("trace"); out["full"] = rf; out["x_full"] = xf.tolist()
    out["x_arm"] = np.asarray(xa).tolist(); out["secs"] = time.time() - t0
    return out

if __name__ == "__main__":
    jobs = [dict(target=tg, seed=s, iters_arm=150, iters_full=300) for tg in TARGETS for s in range(1, 7)]
    jobs += [dict(target=None, seed=s, iters_arm=it, full=False) for it in (600, 2400) for s in range(1, 5)]
    jobs.sort(key=lambda j: -(j["iters_arm"] + j.get("iters_full", 0)))
    with Pool(int(os.environ.get("NPROC", "14"))) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(f"{OUT}/throw_demand2.json", "w"), default=float)
    K = ("legs", "trunk", "shoulder", "elbow", "wrist", "counter_arm")
    for tg in TARGETS:
        g = [r for r in res if r["target"] == tg]
        for cond in ("arm", "full"):
            v = [r[cond] for r in g if cond in r and r[cond]["valid_all_dt"]]
            if not v: print(tg, cond, "no valid"); continue
            sh = {k: np.median([x["work"].get(k, 0)/max(sum(x["work"].values()), 1e-9) for x in v]) for k in K}
            print(f"target {tg:4.1f} {cond:4s}: n {len(v)} speed {np.median([x['speed'] for x in v]):.2f} work {np.median([sum(x['work'].values()) for x in v]):.0f} J PR {np.median([x['participation'] for x in v]):.2f} " + " ".join(f"{k}:{sh[k]:.2f}" for k in K), flush=True)
    for it in (600, 2400):
        v = [r["arm"]["speed"] for r in res if r["target"] is None and r["iters_arm"] == it and r["arm"]["valid_all_dt"]]
        print(f"arm-only max @ {it} iters: {sorted(round(x, 2) for x in v)}", flush=True)
