"""K2 extras: (1) diagnose the dt = 0.5 ms failures at 0 s (box 36 kg every z; pot 44 kg warm-start ERROR): same z under the K2 stand,
with no lag, and under the Q1 capped stand; (2) unloaded stand with milder pushes (0.05 x body weight) at 20 ms; (3) pot 48 kg at 15/20 ms
warm-started from the 44 kg z (budget 48). Output results/overnight/K2/extra.json."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, multiprocessing as mp
import k2_lag as K, k2_ladder as KL

if __name__ == "__main__":
    P = f"{K.OUT}/extra.json"; out = json.load(open(P)) if os.path.exists(P) else {}; R = json.load(open(f"{K.OUT}/refine.json"))
    zb = R["box_lag20ms"]["hist"][0] and json.load(open(f"{K.OUT}/ladder_box_lag20ms.json"))["max_z"]; zp = json.load(open(f"{K.OUT}/ladder_pot_lag20ms.json"))["max_z"]
    zs = json.load(open(f"{K.ROOT}/results/overnight/Q1/lag_maxima.json"))["head_pot_lag10ms"]["lag_max_z"]
    jobs = [("diag|box36|K2_20ms|0.0005", "head", "box", 36.0, zb, 0.0005, 0.020, K.COMP, True, True, 0.0),
            ("diag|box36|nolag|0.0005", "head", "box", 36.0, zb, 0.0005, 0.0, "none", True, True, 0.0),
            ("diag|box36|Q1cap10ms|0.0005", "head", "box", 36.0, zb, 0.0005, 0.010, "none", True, True, 2.0),
            ("diag|pot44warm|K2_20ms|0.0005", "head", "pot", 44.0, zp, 0.0005, 0.020, K.COMP, True, True, 0.0)]
    jobs += [(f"stand|20ms|{dt}|push0.05", "head", "pot", 2.0, zs, dt, 0.020, K.COMP, False, True, 0.0, ("push", 0.05)) for dt in (0.001, 0.0005)]
    jobs = [j for j in jobs if j[0] not in out]
    with mp.get_context("fork").Pool(6, maxtasksperchild=6) as pool:
        for k, r in pool.imap_unordered(K.job, jobs):
            out[k] = r; K.log(f"E {k}: held {r['t_held']} {r['reason']} valid {r.get('valid_slip')} unl {r.get('unloaded_ok')} err {r.get('error')}", "extra.status")
        json.dump(K.J(out), open(P, "w"), indent=1); t_end = time.time() + 3600*0.6
        for lag in (0.020, 0.015):
            key = f"pot48|{int(lag*1000)}ms"
            if key in out: continue
            r = KL.rung(pool, "pot", 48.0, lag, R[f"pot_lag{int(lag*1000)}ms"]["z"], 48, t_end, pop=6); out[key] = r; json.dump(K.J(out), open(P, "w"), indent=1)
            K.log(f"E {key}: valid {r['valid']} evals {r['evals']}", "extra.status")
