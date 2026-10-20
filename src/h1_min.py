"""Q2/III.5: minimum cross-robot set on H1 (torque-native, certified sole-flat start, contact
tc 0.004, certified-torque feedforward settle). Reproduction (T1/T2) and PR demand sweep.
Optimised at H1's native dt 2 ms, measured by faithful 1 ms replay."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, branch_i as BI, contract_g1 as CG, lineage as LN
from multiprocessing import Pool
OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase9")
Q0 = np.load(f"{OUT}/h1_start_soleflat.npy")
def job(a):
    S = BI.setup(start=Q0, robot="H1"); BI.settle(S); t0 = time.time()
    x, r = BI.solve(set(S["types"]), a["seed"], target=a.get("target"), iters=a["iters"], popsize=a["pop"], task=a["task"], S=S)
    dt0 = S["m"].opt.timestep; BI.apply_dt(S, 0.001)
    o = BI.rollout(x, set(S["types"]), record=False, S=S); BI.apply_dt(S, dt0)
    ok = not [z for z in o["validity"]["reasons"] if z != "CONTACT_ENERGY"] and not o.get("bound_violation")
    return dict(a, v_opt=r["v"], v=o["v"], valid=bool(ok), reasons=o["validity"]["reasons"], PR=o["participation"],
                work=o["work"], T_push=o["mech"]["T_push"], rot=o["mech"]["flight_rotation"], x=np.asarray(x).tolist(), secs=time.time()-t0)
if __name__ == "__main__":
    S = BI.setup(start=Q0, robot="H1"); CG.verify_or_die(S)
    jobs = [dict(kind="max", task=t, seed=s, iters=300, pop=20) for t in ("T1", "T2") for s in range(1, 9)]
    jobs += [dict(kind="demand", task="T1", target=tg, seed=s, iters=250, pop=20) for tg in (0.4, 0.8, 1.2, 1.6) for s in range(1, 5)]
    LN.manifest(f"{OUT}/h1_min.manifest.json", question="Does corrected dynamics + PR demand effect generalise to H1?", n_jobs=len(jobs), plant_hash=CG.plant_hash(S))
    with Pool(int(os.environ.get("NPROC", "12"))) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(f"{OUT}/h1_min.json", "w"), default=float)
    print("h1_min done", sum(r["valid"] for r in res), "/", len(res), flush=True)
