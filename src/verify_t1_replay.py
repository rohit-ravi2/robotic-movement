"""Adversarial verifier (2026-10-17) for T1: replay the two headline maxima (head pot 40 kg, head box 32 kg) from their stored best_z
on FRESH plants (one process per rollout) at 1 ms and 0.5 ms, plus 4 small z perturbations (sigma 0.03 in the [-1,1] search space)
per case and dt as a robustness probe. Writes results/overnight/T1/verify_replay.json."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, multiprocessing as mp
T1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "overnight", "T1")
CASES = ["head_pot_40kg_t1", "head_box_32kg_t1"]


def run(a):
    tag, k, z, dt = a
    from engine import headcarry as HC
    C = json.load(open(f"{T1}/{tag}.json"))
    r = HC.rollout(C["obj"], C["mass"], C["strategy"], np.asarray(z), 10.0, dt, False, None, law=True); r.pop("trace", None)
    keep = {x: r.get(x) for x in ("t_held", "reason", "head_frac", "tilt_max_deg", "foot_travel_max", "pelvis_drift", "e_max", "dt", "scene")}
    keep["gate"] = HC.gate(r, 10.0, C["strategy"], C["mass"]); return tag, k, dt, keep


if __name__ == "__main__":
    rng = np.random.default_rng(20261017); jobs = []
    for tag in CASES:
        z0 = np.asarray(json.load(open(f"{T1}/{tag}.json"))["best_z"])
        zs = [("exact", z0)] + [(f"pert{i}", np.clip(z0 + 0.03*rng.standard_normal(z0.size), -1, 1)) for i in range(4)]
        jobs += [(tag, k, z.tolist(), dt) for k, z in zs for dt in (0.001, 0.0005)]
    out = {}
    with mp.get_context("spawn").Pool(2, maxtasksperchild=1) as p:
        for tag, k, dt, r in p.imap_unordered(run, jobs):
            out.setdefault(tag, {})[f"{k}_{dt}"] = r; print(tag, k, dt, r["t_held"], r["reason"], round(r["foot_travel_max"] or -1, 4), round(r["pelvis_drift"] or -1, 4), r["gate"], flush=True)
    json.dump(out, open(f"{T1}/verify_replay.json", "w"), indent=1)
