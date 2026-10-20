"""Phase 9 benchmark (Q3): 8 seeds, Phase 9 plant/measurement code (time-based knots, exact lag,
sustained-flight takeoff), optimised at 4 ms, MEASURED by faithful 1 ms replay.
Arms: OURS (native), B1 lambda=0 (mocap-seeded free refinement = primary comparison),
APRIOR (analytic-crouch-prior-seeded free refinement: non-mocap strategy prior), B0 (retarget+track)."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, branch_i as BI, mocap_baseline as MB, contract_g1 as CG, lineage as LN
from multiprocessing import Pool
OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase9"); PKL = "retargeted_g1_all.pkl"
CLIPS = ["obstacles5_subject3@1585", "fightAndSports1_subject1@3757", "obstacles4_subject3@3107", "jumps1_subject1@1166"]
ITERS, POP = 100, 16

def job(a):
    c = MB.load_clip(a["clip"], PKL); S = BI.setup(c["start"])
    if abs(S["m"].opt.timestep - 0.004) > 1e-12: BI.apply_dt(S, 0.004)
    BI.settle(S); ref = MB.analytic_clip(c) if a["method"] == "APRIOR" else c
    t0 = time.time()
    if a["method"] == "B0":
        om, _ = MB.tune_gains(S, c, 1.0); x = np.zeros(len(S["types"])*c["nk"]); ts = 1.0
    elif a["method"] in ("B1", "APRIOR"):
        om, _ = MB.tune_gains(S, ref, 1.0)
        r = MB.solve_b1(S, ref, 0.0, a["seed"], om, ITERS, POP, task=a["task"]); x = np.array(r["x"]); ts = r["tscale"]
    else:
        om = 0.0; r = MB.solve_b1(S, c, 0.0, a["seed"], 0.0, ITERS, POP, task=a["task"], native=True); x = np.array(r["x"]); ts = 1.0
    # faithful 1 ms measurement (schedule is time-based in Phase 9 code: identical control)
    S1 = BI.setup(c["start"]); BI.apply_dt(S1, 0.001)
    o = MB.run(S1, ref, x, om, ts, record=True)
    BI.apply_dt(S1, 0.004)
    tr = o.pop("trace")
    ok = not [z for z in o["validity"]["reasons"] if z != "CONTACT_ENERGY"] and not o.get("bound_violation")
    return dict(a, v=o["v"], valid=bool(ok), reasons=o["validity"]["reasons"], deviation_mocap=None if a["method"] == "APRIOR" else o["deviation"],
                PR=o["participation"], work=o["work"], rot=o["mech"]["flight_rotation"], T_push=o["mech"]["T_push"],
                crouch=o["mech"]["crouch_depth"], omega=om, tscale=ts, x=x.tolist(), trace=tr, secs=time.time()-t0)

if __name__ == "__main__":
    jobs = []
    for cl in CLIPS:
        jobs.append(dict(clip=cl, method="B0", task="T2", seed=0))
        for s in range(1, 9):
            for task in ("T2", "T1"):
                jobs.append(dict(clip=cl, method="OURS", task=task, seed=s))
                jobs.append(dict(clip=cl, method="B1", task=task, seed=s, lam=0.0))
            jobs.append(dict(clip=cl, method="APRIOR", task="T2", seed=s))
    for cl in CLIPS:
        c = MB.load_clip(cl, PKL); CG.verify_or_die(BI.setup(c["start"]))
    LN.manifest(f"{OUT}/baseline9.manifest.json", question="Q3: native vs mocap-seeded (lambda=0) vs analytic-crouch-prior-seeded refinement, 8 seeds, faithful 1 ms measurement",
                n_jobs=len(jobs), budget=[ITERS, POP], acceptance="physics-valid at 1 ms; paired per-clip bootstrap CIs")
    with Pool(int(os.environ.get("NPROC", "12"))) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(f"{OUT}/baseline9.json", "w"), default=float)
    print(f"baseline9: {len(res)} runs, valid {sum(r['valid'] for r in res)}", flush=True)
