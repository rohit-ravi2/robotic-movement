import sys, json, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "src"))
from multiprocessing import get_context
Q4 = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/overnight/Q4")
def job(a):
    mass, dt, k, z = a
    from engine import dynpred_headcarry_v2 as D2; import t2_gate as G
    D2.register_objects(); r = G.rollout_gated("medbox", mass, "head", np.asarray(z), 10.0, dt)
    return dict(mass=mass, dt=dt, k=k, t_held=r["t_held"], reason=r["reason"], foot_travel_max=r["foot_travel_max"], pelvis_drift=r["pelvis_drift"], valid_slip=r["valid_slip"])
if __name__ == "__main__":
    rng = np.random.default_rng(20261019); jobs = []
    for mass in (34.7, 26.7, 45.0):
        z0 = np.asarray(json.load(open(f"{Q4}/head_medbox_{mass:g}kg.json"))["best_z"])
        zs = [("exact", z0)] if mass != 45.0 else []
        zs += [(f"pert{i}", np.clip(z0 + (0.01 if mass == 45.0 else 0.03)*rng.standard_normal(z0.size), -1, 1)) for i in range(2)]
        jobs += [(mass, dt, k, z.tolist()) for k, z in zs for dt in (0.001, 0.0005)]
    R = []
    with get_context("spawn").Pool(2, maxtasksperchild=1) as p:
        for r in p.imap_unordered(job, jobs): R.append(r); print(r, flush=True)
    json.dump(R, open(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/overnight/verify_q4/ladder_robust.json"), "w"), indent=1, default=float)
