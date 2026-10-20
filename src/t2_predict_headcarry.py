"""T2: dynamic head-carry max-load predictor (engine.dynpred_headcarry) for the scored objects (box, pot: head and hands) and the
held-out objects (tallcyl, crate: head). Grip posture = the search prior's (head dz = 0.5 of the half height; tallcyl dz = 0, the
only reachable grip of a 0.40 m cylinder: IK residual 0.27 m at dz = 0.5). D2 (static LP) at the track A posture (dz = 0).
-> results/overnight/T2/pred_headcarry.json"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, throw as TH
from engine import dynpred_headcarry as D, headcarry as HC
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "overnight", "T2")
A_STATIC = {("head", "box"): 112.49, ("head", "pot"): 112.18, ("hands", "box"): 7.66, ("hands", "pot"): 6.78}   # results/overnight/A/predictor.json a_static
if __name__ == "__main__":
    D.register_objects(); S = TH.setup("G1H", ball=False); m = S["m"]; q = np.array(S["q_settled"]); res = {}
    for st, ob, gp in (("head", "box", dict(dz=0.5)), ("head", "pot", dict(dz=0.5)), ("hands", "box", {}), ("hands", "pot", {}),
                       ("head", "tallcyl", dict(dz=0.0)), ("head", "crate", dict(dz=0.5))):
        t = time.time(); stat = A_STATIC.get((st, ob))
        if stat is None:      # static LP at the track A posture (dz = 0), largest feasible mass on a 2.5 kg grid (the LP is not monotone near 0)
            c, tg = HC.grip_geometry(m, q, st, ob, {}); qa, err, perr = HC.grip_ik(m, q, tg); H, wb, arm = HC.parts(m); fg = sorted(S["foot_geoms"])
            ok = [mm for mm in np.arange(2.5, 200.1, 2.5) if HC.contact_lp(m, qa, *HC.obj_contacts(m, qa, st, ob), mm, fg, S["robot_bodies"], H["fingers"])["s"] <= 1]
            stat = float(max(ok)) if ok else float("inf")      # infeasible at EVERY mass incl. 2.5 kg = an LP failure at this posture, not a capacity: not used
        r = D.predict(st, ob, S=S, static=stat, gp=gp); r["gp"] = gp
        if not np.isfinite(stat): r["D2_note"] = "static LP infeasible at every mass on the grid (also at 2.5 kg): solver/posture failure, D2 not used"
        r["time_s"] = time.time() - t; res[f"{st}_{ob}"] = r
        print(st, ob, "m_max", r["m_max"], r["binding"], r["terms"], flush=True)
    json.dump(res, open(f"{OUT}/pred_headcarry.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else (None if isinstance(o, float) and not np.isfinite(o) else float(o)))
