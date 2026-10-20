"""Phase 11 step 3 (GENERATIVE map): recruitment vs demand, the tennis-serve question.
Native search, demand = target release speed (objective: hit the target, least positive work; validity and the
direction cone lexicographic as always). FULL chain (17 driven joints) vs ARM-ONLY (right shoulder x3, elbow,
wrist pitch driven; legs, trunk and counter-arm HELD). Plus each condition's maximum speed."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, throw as T, branch_i as BI
from multiprocessing import Pool
OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase11")
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_pitch_joint"]
TARGETS = [float(v) for v in os.environ.get("TARGETS", "2,4,6,8,10").split(",")]

def job(a):
    S = T.setup("G1", driven=ARM if a["cond"] == "arm_only" else None); BI.apply_dt(S, T.DT_OPT); t0 = time.time()
    x = T.solve(S, a["seed"], iters=a["iters"], pop=16, target=a["target"])
    r = T.measure(S, x); tr = r.pop("trace")
    return dict(a, **r, x=np.asarray(x).tolist(), secs=time.time()-t0, seg_w=tr["seg_w"], jv=tr["jv"], types=S["types"], groups=S["groups"])

if __name__ == "__main__":
    jobs = [dict(cond=c, target=tg, seed=s, iters=150) for c in ("full", "arm_only") for tg in TARGETS for s in range(1, 7)]
    jobs += [dict(cond="arm_only", target=None, seed=s, iters=150) for s in range(1, 9)]
    with Pool(int(os.environ.get("NPROC", "12"))) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(f"{OUT}/throw_demand.json", "w"), default=float)
    for c in ("full", "arm_only"):
        for tg in TARGETS + [None]:
            g = [r for r in res if r["cond"] == c and r["target"] == tg and r["valid_all_dt"]]
            if not g: print(c, tg, "no valid"); continue
            sh = {k: np.median([r["work"].get(k, 0)/max(sum(r["work"].values()), 1e-9) for r in g]) for k in ("legs", "trunk", "shoulder", "elbow", "wrist", "counter_arm")}
            print(f"{c:8s} target {tg}: n {len(g)} speed {np.median([r['speed'] for r in g]):.2f} work {np.median([sum(r['work'].values()) for r in g]):.0f} J  PR {np.median([r['participation'] for r in g]):.2f}  shares " + " ".join(f"{k}:{v:.2f}" for k, v in sh.items()), flush=True)
