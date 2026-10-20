"""T2: re-score every track A hold (standing and push cases) with the slip-gated validity (t2_gate), both dt, fresh plant per dt
(engine.headcarry.plant builds per (obj, mass, dt)). Output results/overnight/T2/rescore_A.json. usage: python t2_rescore_A.py [workers]"""
import os, sys, json, glob, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, multiprocessing as mp
import t2_gate as G

A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "overnight", "A")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "overnight", "T2", "rescore_A.json")

if __name__ == "__main__":
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    cases = []
    for f in sorted(glob.glob(f"{A}/*_*kg*.json")):
        tag = os.path.basename(f)[:-5]
        if "sway" in tag or "pass1" in tag: continue
        d = json.load(open(f)); dist = d.get("sway")
        cases.append((tag, d["obj"], float(d["mass"]), d["strategy"], d["best_z"], float(d.get("T", 10.0)), dist, bool(d.get("valid"))))
    jobs = [(c[1], c[2], c[3], c[4], c[5], dt, c[6], True) for c in cases for dt in (0.001, 0.0005)]
    with mp.get_context("fork").Pool(W) as p: R = p.map(G.eval_job, jobs, chunksize=1)
    out = {}
    for i, c in enumerate(cases):
        r1, r2 = R[2*i], R[2*i + 1]
        keep = ("t_held", "reason", "foot_slide_max", "foot_slide_t", "com_drift_max", "slip_ok", "gate_hc", "valid_slip", "joint_sat", "head_frac", "e_max", "tilt_max_deg", "util_p95")
        out[c[0]] = dict(obj=c[1], mass=c[2], strategy=c[3], disturbance=c[6], valid_trackA=c[7],
                         dt1ms={k: r1.get(k) for k in keep}, dt05ms={k: r2.get(k) for k in keep},
                         valid_slip_both=bool(r1.get("valid_slip") and r2.get("valid_slip")))
        print(c[0], "trackA", c[7], "| slip-gated both", out[c[0]]["valid_slip_both"], "| slide 1ms %.3f 0.5ms %.3f" % (r1.get("foot_slide_max") or -1, r2.get("foot_slide_max") or -1), r1.get("reason"), r2.get("reason"), flush=True)
    json.dump(out, open(OUT, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
