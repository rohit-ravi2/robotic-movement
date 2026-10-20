"""Q4: head-carry predictor v2 (engine.dynpred_headcarry_v2) on every scored head-carry case and the Q4 held-out medium box.
Grip postures as T2 (head dz = 0.5 of the half height; tallcyl dz = 0, its only reachable grip); medbox: dz = 0.5 if the IK reaches it
(residual < 1 cm), else dz = 0. D2 as T2 (track A value, or the static LP on a 2.5 kg grid at the dz = 0 posture).
-> results/overnight/Q4/pred_headcarry_v2.json"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, throw as TH
from engine import dynpred_headcarry_v2 as V, headcarry as HC
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "overnight", "Q4")
A_STATIC = {("head", "box"): 112.49, ("head", "pot"): 112.18, ("hands", "box"): 7.66, ("hands", "pot"): 6.78, ("head", "crate"): 110.0}   # A/predictor.json, T2 pred_headcarry.json
if __name__ == "__main__":
    V.register_objects(); S = TH.setup("G1H", ball=False); m = S["m"]; q = np.array(S["q_settled"]); res = {}
    which = sys.argv[1].split(",") if len(sys.argv) > 1 else ["head:box", "head:pot", "head:tallcyl", "head:crate", "head:medbox", "hands:box", "hands:pot"]
    for w in which:
        st, ob = w.split(":"); t = time.time()
        gp = {} if st == "hands" else dict(dz=0.0 if ob == "tallcyl" else 0.5)
        if ob == "medbox":
            errs = {dz: HC.grip_ik(m, q, HC.grip_geometry(m, q, st, ob, dict(dz=dz))[1])[1] for dz in (0.5, 0.0)}
            gp = dict(dz=0.5 if errs[0.5] < 0.01 else 0.0); print("medbox IK residual by dz", errs, "->", gp, flush=True)
        stat = A_STATIC.get((st, ob))
        if stat is None:
            c, tg = HC.grip_geometry(m, q, st, ob, {}); qa, err, perr = HC.grip_ik(m, q, tg); H, wb, arm = HC.parts(m); fg = sorted(S["foot_geoms"])
            ok = [mm for mm in np.arange(2.5, 200.1, 2.5) if HC.contact_lp(m, qa, *HC.obj_contacts(m, qa, st, ob), mm, fg, S["robot_bodies"], H["fingers"])["s"] <= 1]
            stat = float(max(ok)) if ok else float("inf")      # LP infeasible at every mass (also 2.5 kg) = a posture/solver failure, as T2: D2 not used
        r = V.predict(st, ob, S=S, static=stat, gp=gp); r["gp"] = gp; r["time_s"] = time.time() - t; res[f"{st}_{ob}"] = r
        print(st, ob, "v2 m_max", r["m_max"], r["binding"], r["terms"], "| v1", r["v1"]["m_max"], r["v1"]["binding"], flush=True)
        json.dump(res, open(f"{OUT}/pred_headcarry_v2.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else (None if isinstance(o, float) and not np.isfinite(o) else float(o)))
