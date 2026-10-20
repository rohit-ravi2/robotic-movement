"""K2 acceptance: UNLOADED stock G1H stand (object parked 2.5 m away) with the lag-aware stand (k2_lag.COMP), lag 10/15/20 ms, FRESH
plants at 1 ms and 0.5 ms, 10 s. Conditions: nominal; lateral torso pushes 0.1 x body weight for 0.2 s at 3/6/9 s; controller lag model
mismatch (model = 0.8x and 1.25x the plant lag); and the Q1 capped stand (c = 2, no compensation) as the baseline.
Gate: 10 s, no fall, T1 support gate (loaded foot travel <= 2 cm, pelvis drift <= 5 cm) and the 3 cm foot-slip check.
Output results/overnight/K2/stand_accept.json. usage: python k2_stand.py [workers]"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, multiprocessing as mp
import k2_lag as K

if __name__ == "__main__":
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 6; os.makedirs(K.OUT, exist_ok=True)
    z = json.load(open(f"{K.ROOT}/results/overnight/Q1/lag_maxima.json"))["head_pot_lag10ms"]["lag_max_z"]   # stand params (k_cp, k_trunk, waist) of the Q1 pot; the object is never placed
    jobs = []
    for lag in (0.010, 0.015, 0.020):
        for dt in (0.001, 0.0005):
            b = ("head", "pot", 2.0, z, dt, lag)
            jobs += [(f"{lag*1000:g}ms|{dt}|nominal",) + b + (K.COMP, False, True, 0.0),
                     (f"{lag*1000:g}ms|{dt}|push0.1",) + b + (K.COMP, False, True, 0.0, ("push", 0.1)),
                     (f"{lag*1000:g}ms|{dt}|model0.8",) + b + (K.COMP, False, True, 0.0, None, 0.8),
                     (f"{lag*1000:g}ms|{dt}|model1.25",) + b + (K.COMP, False, True, 0.0, None, 1.25),
                     (f"{lag*1000:g}ms|{dt}|Q1cap_c2",) + b + ("none", False, True, 2.0)]
    P = f"{K.OUT}/stand_accept.json"; out = json.load(open(P)) if os.path.exists(P) else {}
    jobs = [j for j in jobs if j[0] not in out]
    with mp.get_context("fork").Pool(W, maxtasksperchild=4) as pool:
        for k, r in pool.imap_unordered(K.job, jobs):
            out[k] = r; K.log(f"STAND {k}: ok {r['unloaded_ok']} held {r['t_held']} {r['reason']} ft {r['foot_travel_max']} pd {r['pelvis_drift']} slide {r['foot_slide_max']}")
            json.dump(K.J(out), open(P, "w"), indent=1)
