"""Does the analytic-crouch strategy prior lift G1's jump from the STANDARD knees_bent start?
Arms at equal budget and equal 1.1 s horizon: NATIVE (no reference) vs APRIOR (tracked analytic
crouch->takeoff reference + free refinement). Optimised at 4 ms, measured faithfully at 1 ms."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, branch_i as BI, mocap_baseline as MB
from multiprocessing import Pool
OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase9")
def clip0():
    S = BI.setup(); BI.settle(S); q = np.array(S["q_settled"])
    c = dict(key="knees_bent", start=q, Q=np.repeat(q[None], 40, 0), fps=30, takeoff_ref=0.8, horizon=1.1, nk=22)
    return c
def job(a):
    c = clip0(); S = BI.setup(); ref = MB.analytic_clip(c) if a["method"] == "APRIOR" else c
    t0 = time.time()
    if a["method"] == "APRIOR":
        om, _ = MB.tune_gains(S, ref, 1.0); r = MB.solve_b1(S, ref, 0.0, a["seed"], om, 100, 16, task=a["task"]); ts = r["tscale"]
    else:
        om = 0.0; r = MB.solve_b1(S, c, 0.0, a["seed"], 0.0, 100, 16, task=a["task"], native=True); ts = 1.0
    x = np.array(r["x"]); BI.apply_dt(S, 0.001); o = MB.run(S, ref, x, om, ts); BI.apply_dt(S, 0.004)
    ok = not [z for z in o["validity"]["reasons"] if z != "CONTACT_ENERGY"] and not o.get("bound_violation")
    return dict(a, v=o["v"], valid=bool(ok), reasons=o["validity"]["reasons"], rot=o["mech"]["flight_rotation"],
                crouch=o["mech"]["crouch_depth"], T=o["mech"]["T_push"], grf=o["grf_bw"], PR=o["participation"],
                vbound=o["mech"]["v_bound_rigorous"], x=x.tolist(), secs=time.time()-t0)
if __name__ == "__main__":
    jobs = [dict(method=m, task=t, seed=s) for m in ("APRIOR", "NATIVE") for t in ("T2", "T1") for s in range(1, 9)]
    with Pool(int(os.environ.get("NPROC", "6"))) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(f"{OUT}/aprior_standard.json", "w"), default=float)
    for m in ("APRIOR", "NATIVE"):
        for t in ("T2", "T1"):
            g = [r for r in res if r["method"] == m and r["task"] == t and r["valid"]]; L = [r["v"] for r in g if r["rot"] <= 0.35]
            print(m, t, "valid", len(g), "median v", round(float(np.median([r["v"] for r in g])), 3) if g else None, "landable median", round(float(np.median(L)), 3) if L else None, len(L), flush=True)
