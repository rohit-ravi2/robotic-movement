"""Phase 11 step 2 (CONFIRMATORY): NATIVE vs APRIOR (analytic wind-up prior) on the G1 throw, equal budget.
8 seeds at 150x16 and 4 seeds at 600x16 (search-limit check). Optimised at 4 ms, measured faithfully at 1 ms."""
import sys, os, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, throw as T, branch_i as BI
from multiprocessing import Pool
OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase11")

def job(a):
    S = T.setup(a.get("robot", "G1")); BI.apply_dt(S, T.DT_OPT); t0 = time.time()
    if a["method"] == "APRIOR":
        ref = pickle.load(open(f"{OUT}/prior_{a.get('robot','G1')}.pkl", "rb"))
        x, om, ts, amp = T.solve_prior(S, ref, a["seed"], iters=a["iters"], pop=16)
        fn = lambda: T.tracker(S, T.scaled_ref(ref, amp), om, ts)
    else:
        x = T.solve(S, a["seed"], iters=a["iters"], pop=16); om, ts, amp, fn = 0.0, 1.0, None, None
    r = T.measure(S, x, ctrl_fn_factory=fn)
    tr = r.pop("trace")
    return dict(a, **{k: r[k] for k in r}, omega=om, tscale=ts, amp=amp, x=np.asarray(x).tolist(), secs=time.time()-t0, seg_w=tr["seg_w"], dt_meas=tr["dt"])

def ladder():
    jobs = [dict(method=m, seed=s, iters=2400) for m in ("APRIOR", "NATIVE") for s in range(21, 25)]
    with Pool(int(os.environ.get("NPROC", "12"))) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(f"{OUT}/throw_g1_ladder2400.json", "w"), default=float)
    for m in ("APRIOR", "NATIVE"):
        g = [r for r in res if r["method"] == m]; v = [r["speed"] for r in g if r["valid_all_dt"]]
        print(f"2400 {m}: valid+cone {len(v)}/{len(g)} speeds {sorted(round(x, 2) for x in v)}", flush=True)

if __name__ == "__main__" and sys.argv[1:2] == ["ladder"]:
    ladder(); sys.exit()
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    S = T.setup("G1"); ref = T.windup_reference(S); pickle.dump(ref, open(f"{OUT}/prior_G1.pkl", "wb"))
    jobs = [dict(method=m, seed=s, iters=150) for m in ("APRIOR", "NATIVE") for s in range(1, 9)]
    jobs += [dict(method=m, seed=s, iters=600) for m in ("APRIOR", "NATIVE") for s in range(11, 17)]
    jobs.sort(key=lambda j: -j["iters"])
    with Pool(int(os.environ.get("NPROC", "12"))) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(f"{OUT}/throw_g1_main.json", "w"), default=float)
    for it in (150, 600):
        for m in ("APRIOR", "NATIVE"):
            g = [r for r in res if r["method"] == m and r["iters"] == it]; v = [r for r in g if r["valid_all_dt"]]
            print(f"iters {it} {m:6s} valid+in-cone {len(v)}/{len(g)}  median speed {np.median([r['speed'] for r in v]) if v else float('nan'):.2f} m/s  "
                  f"all-run median {np.median([r['speed'] for r in g]):.2f}  reasons {[tuple(r['reasons']) for r in g if not r['valid']][:4]}", flush=True)
