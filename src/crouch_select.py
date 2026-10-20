"""Phase 10: the Gate A start crouch for a given human spec, by the Phase 8 rule: the statically
certified crouch (directional capacity, P11/P14 geometry) NEAREST the Phase 8 crouch, with torque
margin >= 0.10 and support margin >= 0.030 m. Run once per spec; the result is written into the spec
BEFORE any jump is optimised on it."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, gate_a2 as GA, human_planar as HP, plant as P

def margins(h, qj):
    q0 = h.place(qj); th = np.array([q0[i] for i in h.qadr]); d = mujoco.MjData(h.m)
    lo, hi = h.limits(th, np.zeros(5))
    c = P.static_certificate(h.m, d, q0, coll_geoms=[h.fid], tmax_override=np.maximum(lo, hi))
    if c["status"] != "STATIC_FEASIBLE": return None
    t = np.array(c["tau"]); cap = np.where(t >= 0, hi, lo)
    if np.any(np.abs(t) > cap): return None
    return float(np.min(1 - np.abs(t)/np.maximum(cap, 1e-9))), float(c.get("support_margin", np.nan)), t

if __name__ == "__main__":
    h = HP.Human(spec=sys.argv[1]); C0 = np.array(GA.CROUCH); rng = np.random.default_rng(0)
    lo_r = [h.m.jnt_range[j][0] for j in h.jid]; hi_r = [h.m.jnt_range[j][1] for j in h.jid]
    best = []
    for k in range(6000):
        sc = 0.02 + 0.5*k/6000
        qj = np.clip(C0 + rng.normal(0, sc, 5)*np.array([1, 1, 1, 0.5, 0.5]), lo_r, hi_r)
        r = margins(h, qj)
        if r and r[0] >= 0.10 and r[1] >= 0.030:
            best.append((float(np.linalg.norm(qj - C0)), qj.round(3).tolist(), r[0], r[1], r[2].round(1).tolist()))
        if len(best) >= 40: break
    best.sort()
    print(json.dumps(best[:3], indent=0))
    z = h.place(best[0][1]); print("crouch COM z", round(float(h.com()[2]), 3), "stand COM z", round(float((h.place(None), h.com()[2])[1]), 3))
