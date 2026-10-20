"""Phase 12: run PREDICTION_P12.md (committed before this file). Counterfactual G1 with throwing-arm torque x k.
PRIMARY: arm-only maximum release speed, 6 seeds x 600 iters, per k. SECONDARY: leg-recruitment switch from the
arm-first warm-started demand sweep (targets 4..20 m/s, 4 seeds), per k. 1 ms optimisation, Phase 11 protocol."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, throw as T, branch_i as BI
from throw_demand2 import ARM, embed
from multiprocessing import Pool
OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase12")
KS = (0.5, 1.0, 2.0); TARGETS = [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]

def job(a):
    t0 = time.time(); k = a["k"]
    SA = T.setup("G1", driven=ARM); BI.apply_dt(SA, T.DT_OPT); caps_a = np.full(len(SA["types"]), k)
    xa = T.solve(SA, a["seed"], iters=a["iters_arm"], pop=16, target=a["target"], caps=caps_a)
    ra = T.measure(SA, xa, caps=caps_a); ra.pop("trace"); out = dict(a, arm=ra, x_arm=np.asarray(xa).tolist())
    if a["target"] is not None:
        S = T.setup("G1"); BI.apply_dt(S, T.DT_OPT)
        caps_f = np.array([k if t in ARM else 1.0 for t in S["types"]])
        xf = T.solve(S, a["seed"], iters=a["iters_full"], pop=16, target=a["target"], x0=embed(xa, S["types"]), sigma0=0.05, caps=caps_f)
        rf = T.measure(S, xf, caps=caps_f); rf.pop("trace"); out.update(full=rf, x_full=xf.tolist())
    out["secs"] = time.time() - t0
    return out

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1:2] == ["smoke"]:
        r = job(dict(k=2.0, target=8.0, seed=1, iters_arm=3, iters_full=3)); print("smoke", round(r["secs"]), "s", round(r["arm"]["speed"], 2), round(r["full"]["speed"], 2)); sys.exit()
    jobs = [dict(k=k, target=None, seed=s, iters_arm=600) for k in KS for s in range(1, 7)]
    jobs += [dict(k=k, target=tg, seed=s, iters_arm=150, iters_full=300) for k in KS for tg in TARGETS for s in range(1, 5)]
    jobs.sort(key=lambda j: -(j["iters_arm"] + j.get("iters_full", 0)))
    with Pool(int(os.environ.get("NPROC", "14"))) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(f"{OUT}/counterfactual.json", "w"), default=float)
    print("done", len(res), flush=True)
