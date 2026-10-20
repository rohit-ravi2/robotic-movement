"""Phase 10 (C2 without mocap): does the analytic-crouch strategy prior (NS-04) generalise to H1?
APRIOR (tracked H1 bank full-set crouch -> takeoff reference + free refinement) vs NATIVE, from H1's
certified sole-flat start, equal budget (100 x 16), T1 and T2H. Optimised at H1's native 2 ms,
measured faithfully at 1 ms."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, branch_i as BI, mocap_baseline as MB, contract_g1 as CG
from multiprocessing import Pool
OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase10")
Q0 = np.load(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase9/h1_start_soleflat.npy"))
ITERS, POP = int(os.environ.get("ITERS", "100")), 16

def clip0():
    S = BI.setup(start=Q0, robot="H1"); BI.settle(S); q = np.array(S["q_settled"])
    return dict(key="h1_soleflat", start=Q0, Q=np.repeat(q[None], 40, 0), fps=30, takeoff_ref=0.8, horizon=1.1, nk=22)

def job(a):
    c = clip0(); S = BI.setup(start=Q0, robot="H1"); BI.apply_dt(S, 0.002); BI.settle(S)
    ref = MB.analytic_clip(c, robot="H1") if a["method"] == "APRIOR" else c; t0 = time.time()
    if a["method"] == "APRIOR":
        om, _ = MB.tune_gains(S, ref, 1.0); r = MB.solve_b1(S, ref, 0.0, a["seed"], om, ITERS, POP, task=a["task"]); ts = r["tscale"]
    else:
        om = 0.0; r = MB.solve_b1(S, c, 0.0, a["seed"], 0.0, ITERS, POP, task=a["task"], native=True); ts = 1.0
    x = np.array(r["x"]); BI.apply_dt(S, 0.001); o = MB.run(S, ref, x, om, ts); BI.apply_dt(S, 0.002)
    ok = not [z for z in o["validity"]["reasons"] if z != "CONTACT_ENERGY"] and not o.get("bound_violation")
    return dict(a, v=o["v"], valid=bool(ok), reasons=o["validity"]["reasons"], rot=o["mech"]["flight_rotation"],
                landable=bool(ok and o["mech"]["flight_rotation"] <= BI.ROT_MAX), grf=o["grf_bw"], T=o["mech"]["T_push"],
                crouch=o["mech"]["crouch_depth"], closure=o["energy_audit"]["closure"], omega=om, tscale=ts, x=x.tolist(), secs=time.time()-t0)

if __name__ == "__main__":
    if sys.argv[1:2] == ["smoke"]:
        r = job(dict(method="APRIOR", task="T1", seed=1)); print({k: r[k] for k in ("v", "valid", "reasons", "rot", "crouch", "secs")})
        sys.exit()
    CG.verify_or_die(BI.setup(start=Q0, robot="H1"))
    jobs = [dict(method=m, task=t, seed=s) for m in ("APRIOR", "NATIVE") for t in ("T1", "T2H") for s in range(1, 9)]
    with Pool(int(os.environ.get("NPROC", "12"))) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(f"{OUT}/aprior_h1.json", "w"), default=float)
    for m in ("APRIOR", "NATIVE"):
        for t in ("T1", "T2H"):
            g = [r for r in res if r["method"] == m and r["task"] == t and r["valid"]]
            print(m, t, "valid", len(g), "median v", round(float(np.median([r["v"] for r in g])), 3) if g else None,
                  "landable", sum(r["landable"] for r in g), flush=True)
