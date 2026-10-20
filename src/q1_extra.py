"""Q1 follow-ups (2026-10-17): (1) robustness of each 10 ms lag-robust maximum: 2 perturbations of the 15-D vector (sigma 0.03,
rng 20261017) at both dt; (2) lag bracket on the 40 kg pot: the 10 ms solution with its servo cap held at the same ABSOLUTE natural
frequency, lag 12/14/16 ms, both dt; (3) 20 ms stance check: the pot at 2 kg (nearly unloaded), caps c in {1.5, 2, 2.8}, 1 ms.
Gate = t2_gate.valid_slip (includes the T1 support gate). Output results/overnight/Q1/extra.json. usage: python q1_extra.py [workers]"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, multiprocessing as mp
import q1_lag as Q


def job(a):
    key, st, obj, mass, z, dt, lag, c = a
    HC = Q._setlag(lag, c); import t2_gate as G
    r = G.eval_job((obj, mass, st, np.asarray(z[:14]), 10.0, dt, None, True)); r = {k: r.get(k) for k in Q.KEEP}; r["wcap"] = c; return key, r


if __name__ == "__main__":
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    L = json.load(open(f"{Q.OUT}/lag_maxima.json")); rng = np.random.default_rng(20261017); jobs = []
    for tag, st, obj, *_ in Q.CASES:
        v = L.get(f"{tag}_lag10ms", {})
        if v.get("lag_max") is None: continue
        z0 = np.asarray(v["lag_max_z"])
        for i in range(2):
            zp = np.clip(z0 + 0.03*rng.standard_normal(z0.size), -1, 1).tolist()
            jobs += [(f"pert|{tag}|{v['lag_max']:g}|{i}|{dt}", st, obj, v["lag_max"], zp, dt, 0.010, Q.cap_of(zp)) for dt in (0.001, 0.0005)]
    zp = L["head_pot_lag10ms"]["lag_max_z"]; w_abs = Q.cap_of(zp)/0.010
    jobs += [(f"bracket|pot40|{lag*1000:g}ms|{dt}", "head", "pot", 40.0, zp, dt, lag, w_abs*lag) for lag in (0.012, 0.014, 0.016) for dt in (0.001, 0.0005)]
    jobs += [(f"stance20|pot2|c{c}|0.001", "head", "pot", 2.0, zp, 0.001, 0.020, c) for c in (1.5, 2.0, 2.8)]
    out = {}
    with mp.get_context("fork").Pool(W, maxtasksperchild=4) as pool:
        for k, r in pool.imap_unordered(job, jobs):
            out[k] = r; Q.log(f"X {k}: valid {r.get('valid_slip')} held {r.get('t_held')} {r.get('reason')} ft {r.get('foot_travel_max')} pd {r.get('pelvis_drift')}")
            json.dump(Q.J(out), open(f"{Q.OUT}/extra.json", "w"), indent=1)
