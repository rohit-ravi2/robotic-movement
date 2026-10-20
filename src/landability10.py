"""Phase 10 B5: landability of the analytic-prior strategy. Same arms/budget as baseline9 (APRIOR and
native), task T2H = flight rotation <= ROT_MAX (0.35 rad) as a HARD constraint, 8 seeds x 4 clips +
the standard knees_bent start. Optimised at 4 ms, measured faithfully at 1 ms. Valid AND rotation <=
0.35 rad at 1 ms = landable."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, branch_i as BI, mocap_baseline as MB, aprior_standard as AS
from multiprocessing import Pool
OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase10"); PKL = "retargeted_g1_all.pkl"
CLIPS = ["obstacles5_subject3@1585", "fightAndSports1_subject1@3757", "obstacles4_subject3@3107", "jumps1_subject1@1166", "knees_bent"]

def job(a):
    c = AS.clip0() if a["clip"] == "knees_bent" else MB.load_clip(a["clip"], PKL)
    S = BI.setup(c["start"]); BI.apply_dt(S, 0.004); BI.settle(S)
    ref = MB.analytic_clip(c) if a["method"] == "APRIOR" else c; t0 = time.time()
    if a["method"] == "APRIOR":
        om, _ = MB.tune_gains(S, ref, 1.0); r = MB.solve_b1(S, ref, 0.0, a["seed"], om, 100, 16, task="T2H"); ts = r["tscale"]
    else:
        om = 0.0; r = MB.solve_b1(S, c, 0.0, a["seed"], 0.0, 100, 16, task="T2H", native=True); ts = 1.0
    x = np.array(r["x"]); S1 = BI.setup(c["start"]); BI.apply_dt(S1, 0.001)
    o = MB.run(S1, ref, x, om, ts); BI.apply_dt(S1, 0.004)
    ok = not [z for z in o["validity"]["reasons"] if z != "CONTACT_ENERGY"] and not o.get("bound_violation")
    return dict(a, v=o["v"], valid=bool(ok), reasons=o["validity"]["reasons"], rot=o["mech"]["flight_rotation"],
                landable=bool(ok and o["mech"]["flight_rotation"] <= BI.ROT_MAX), grf=o["grf_bw"], T=o["mech"]["T_push"],
                crouch=o["mech"]["crouch_depth"], closure=o["energy_audit"]["closure"], omega=om, tscale=ts, x=x.tolist(), secs=time.time()-t0)

if __name__ == "__main__":
    jobs = [dict(clip=cl, method=m, seed=s) for cl in CLIPS for m in ("APRIOR", "NATIVE") for s in range(1, 9)]
    with Pool(int(os.environ.get("NPROC", "12"))) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(f"{OUT}/landability10.json", "w"), default=float)
    for cl in CLIPS:
        for m in ("APRIOR", "NATIVE"):
            g = [r for r in res if r["clip"] == cl and r["method"] == m]; L = [r["v"] for r in g if r["landable"]]
            print(f"{cl[:14]:14s} {m:6s} landable {len(L)}/{len(g)} median landable v {np.median(L) if L else float('nan'):.2f}", flush=True)
