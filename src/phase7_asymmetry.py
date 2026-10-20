"""Phase 7 Stage 2 — the asymmetry test, run DYNAMICALLY on G1.

The additive bound is sum_i min(rom_i, power_i, conditional_i). It contains torque limits,
speed limits and angular excursions. It contains no moment arm, no Jacobian and no inter-joint
coupling, so it cannot represent reaction support.

The reaction-chain steelman: using a distal joint loads every joint between it and the ground
with reaction moment. A distal joint's USABLE capacity is therefore bounded by the proximal
chain's ability to carry that reaction. That constraint is nested and non-separable.

The discriminator: hold the capacity multiset fixed and only change WHERE in the chain the
strong capacities sit.
  PROXIMAL_STRONG  strong near the ground  (ankles strongest, elbows weakest)
  DISTAL_STRONG    the exact reverse
Identical multiset, identical sum. The additive bound sees almost no difference. A
reaction-chain-aware system should show PROXIMAL_STRONG >> DISTAL_STRONG.
"""
import sys, os, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, branch_i as BI, bank as BK, robot_bound as R
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# ordered by distance from the ground contact: index 0 is nearest the contact patch
CHAIN = ["ankle_pitch_joint", "knee_joint", "hip_pitch_joint",
         "waist_pitch_joint", "shoulder_pitch_joint", "elbow_joint"]
# monotone capacity multiset. sum = 7.15, identical under any permutation.
MULTISET = [2.2, 1.7, 1.3, 0.9, 0.6, 0.45]

def assignment(name):
    if name == "PROXIMAL_STRONG":  order = list(MULTISET)
    elif name == "DISTAL_STRONG":  order = list(reversed(MULTISET))
    elif name == "MIDDLE_STRONG":  order = [0.6, 1.3, 2.2, 1.7, 0.9, 0.45]
    elif name == "UNIFORM":        order = [np.mean(MULTISET)]*6
    else: raise ValueError(name)
    return {t: (s, s) for t, s in zip(CHAIN, order)}

CONDITIONS = ["PROXIMAL_STRONG", "DISTAL_STRONG", "MIDDLE_STRONG", "UNIFORM"]
SEEDS = [1, 2, 3, 4, 5, 6]
ITERS, POPSIZE = 500, 20

def job(a):
    cond, seed = a
    S = BI.setup()
    sc = assignment(cond)
    t0 = time.time()
    _, r = BI.solve(set(S["types"]), seed, caps_scale=sc, iters=ITERS, popsize=POPSIZE)
    return dict(cond=cond, seed=seed, v=r["v"], height=r["height"],
                participation=r["participation"], valid=bool(r["valid"]),
                work=r["work"], grf_bw=r["grf_bw"],
                max_penetration=r["max_penetration"], secs=time.time()-t0)

def analytic(cond):
    """What the additive bound predicts for the same assignment."""
    D = pickle.load(open(f"{ROOT}/results/pose_bank.pkl", "rb"))["G1"]
    gi = D["group_index"]
    t2g = {"ankle_pitch_joint": "ankles", "knee_joint": "knees", "hip_pitch_joint": "hips",
           "waist_pitch_joint": "waist", "shoulder_pitch_joint": "shoulders",
           "elbow_joint": "elbows"}
    sc = {t2g[t]: v for t, v in assignment(cond).items()}
    caps = BK.caps_from_scaled(D, sc)
    rows = BK.frontier_from_bank(D["bank"], D["groups"], caps, np.array(D["mass"]), 0.25)
    return max(r["v_takeoff"] for r in rows)

if __name__ == "__main__":
    print(f"multiset {MULTISET} sum {sum(MULTISET):.2f}, identical under permutation", flush=True)
    print("ANALYTIC (additive bound) prediction per assignment:", flush=True)
    ana = {}
    for c in CONDITIONS:
        ana[c] = analytic(c)
        print(f"   {c:<16} {ana[c]:.3f} m/s", flush=True)
    sp = max(ana.values())-min(ana.values())
    print(f"   analytic spread across assignments: {sp:.3f} m/s", flush=True)
    jobs = [(c, s) for c in CONDITIONS for s in SEEDS]
    print(f"\n{len(jobs)} dynamic solves at {ITERS} iters / popsize {POPSIZE}", flush=True)
    t0 = time.time()
    with Pool(14) as p: res = p.map(job, jobs)
    json.dump(dict(analytic=ana, multiset=MULTISET, chain=CHAIN, iters=ITERS,
                   popsize=POPSIZE, runs=res),
              open(f"{ROOT}/results/asymmetry.json", "w"), indent=1, default=float)
    print(f"done in {time.time()-t0:.0f}s", flush=True)
