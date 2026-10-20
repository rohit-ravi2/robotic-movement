"""HEAD-CARRY PREDICTOR v2 (track Q4, 2026-10-17): engine.dynpred_headcarry (T2) with the load's recovery transient routed through
the body the way it physically travels. Stock G1H spec only; no constant fitted to a found load.

What T2 got wrong (pre-registered at 1c85c8f): the tall cylinder (r 0.08, h 0.40 m) was predicted 11.3 kg (binding D3, fore-aft) and
24 kg was found valid at both dt (34 kg failed). T2's D3 put the WHOLE recovery transient of a load at the patch edge,
m (g e + lambda^2 e z_head), through the sole: a support-polygon (ZMP) check with the head's 1.3 m lever. Two errors in that:
  1. A transient does not have to pass through the centre of pressure. The ground reaction may leave the ZMP for the duration of a
     transient when the body changes its angular momentum (the centroidal moment pivot, the hip strategy): the excess is carried by
     the hip torques (2 x hip pitch / hip roll ratings minus their gravity load at the posture). Only the SUSTAINED part, the
     load's weight at its offset, must sit inside the support polygon / ankle authority.
  2. The load sits on the head, i.e. on the upper body, which reaches the pelvis only through the waist. The full recovery
     transient (static offset plus the recovery acceleration at the load COM's height above the waist joint) must fit inside the
     waist rating (minus the upper body's own gravity load at the posture). This is the joint the failed runs saturate (crate 85 kg:
     waist pitch; tall cylinder 45 kg: waist pitch), and T2's predictor had no waist term at all.
  For a tall load the lambda^2 z term is large, so (1) removes the false ZMP bound, and the waist lever (z_c - z_waist ~ 0.6 m)
  replaces the head-height lever (1.3 m); for a short load lambda^2 is small and both bounds reduce to the static edge offset.
The recovery itself is T2's: the load must be recoverable from the patch edge (the gate's 7 cm, a task definition), which for an
unstable rocking mode (lambda^2 = g (h - R)/rho^2: COM height h above the dome contact, dome radius R per axis, rho^2 = I/m + h^2 from
the object's pitch / roll inertia) means capturing it there with net recovery acceleration lambda^2 e (the arrival state on the
mode's unstable manifold, v = lambda e, captured at the edge with the least force). D1 (arm moment at the palms' height inside each arm
joint's torque-SPEED envelope), D2 (static whole-body LP), D4 (foot friction) are unchanged. Added as a reported check (not a mass
bound, it is mass-independent): head-contact friction, |palm push| = m g (h - R) e / (h h_p) <= mu_head m g.
Terms per axis:  D3s  m e / (M + m) <= min(support half-span, 2 tau_ankle / ((M + m) g))
                 D3t  m g e (1 + lambda^2 z_head / g) <= (M + m) g min(...) + tau_hip_avail          (ZMP + hip strategy)
                 D5   m g e (1 + lambda^2 (z_c - z_waist) / g) <= tau_waist_avail                    (the waist carries the load)
Prediction = min(D1, D2, D3s, D3t, D5, D4). Hands strategy: D5h = the waist pitch must hold the load ahead of it,
m g x_ahead <= tau_waist_avail; otherwise T2's hands terms."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
from engine import dynpred_headcarry as D1M

G = D1M.G; E_MAX = D1M.E_MAX
OBJ_Q4 = {"medbox": dict(kind="box", size=(0.15, 0.12, 0.13))}     # Q4 held-out: a 0.30 x 0.24 x 0.26 m box (crate 0.12 m < 0.26 m < cylinder 0.40 m tall)


def register_objects():
    D1M.register_objects()
    from engine import headcarry as HC
    import leverage_probes as LP
    for k, v in OBJ_Q4.items(): HC.OBJ.setdefault(k, v); LP.OBJ.setdefault(k, v)


def _bisect(ok, hi=400.0):
    if ok(hi): return hi
    if not ok(0.0): return 0.0
    lo = 0.0
    for _ in range(60):
        mid = 0.5*(lo + hi); lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def predict(strategy, obj, S=None, static=None, gp=None):
    import throw as TH
    from engine import headcarry as HC
    register_objects()
    S = S or TH.setup("G1H", ball=False); m = S["m"]; q = np.array(S["q_settled"])
    base = D1M.predict(strategy, obj, S=S, static=static, gp=gp)          # T2 terms (D1, D2, D4, the hands LP) and geometry
    c, tg = HC.grip_geometry(m, q, strategy, obj, gp if gp is not None else (dict(dz=0.5) if strategy == "head" else {})); qa, err, perr = HC.grip_ik(m, q, tg)
    d = mujoco.MjData(m); d.qpos[:] = qa; mujoco.mj_forward(m, d); M = base["M_robot"]; hw, hh = HC.half(obj)
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    avail = lambda n: float(S["rb"].J[n]["tau"]) - abs(float(d.qfrc_bias[m.jnt_dofadr[jid(n)]]))
    waist = {"pitch": avail("waist_pitch_joint"), "roll": avail("waist_roll_joint")}
    z_w = {"pitch": float(d.xanchor[jid("waist_pitch_joint")][2]), "roll": float(d.xanchor[jid("waist_roll_joint")][2])}
    hip = {ax: sum(avail(f"{s}_hip_{ax}_joint") for s in ("left", "right")) for ax in ("pitch", "roll")}
    tau_ank = {ax: float(S["rb"].J[f"left_ankle_{ax}_joint"]["tau"]) for ax in ("pitch", "roll")}
    fg = sorted(S["foot_geoms"]); pts = []
    for g in fg:
        pts.append(d.geom_xpos[g][:2])
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            R = d.geom_xmat[g].reshape(3, 3); s_ = m.geom_size[g]
            pts += [(d.geom_xpos[g] + R @ np.array([i*s_[0], j*s_[1], 0]))[:2] for i in (-1, 1) for j in (-1, 1)]
    pts = np.array(pts); comw = d.subtree_com[1][:2]
    out = dict(v1=dict(m_max=base["m_max"], binding=base["binding"], terms=base["terms"]), strategy=strategy, obj=obj, M_robot=M, ik_err=err,
               waist_avail=waist, z_waist=z_w, hip_avail=hip, tau_ankle=tau_ank)
    terms = dict(D1_topple=(base["terms"].get("D1_topple") if strategy == "head" else base["terms"].get("D1_squeeze")), D2_static=base["terms"].get("D2_static"))
    if strategy == "head":
        z_head = float(base["dome"]["top"][2]); z_c = z_head + hh; h_p = base["h_palm"]; ax_out = {}
        for ax_name, v in base["D1_topple"].items():
            lat = ax_name.startswith("roll")                       # roll about x: lateral tip -> y, waist ROLL, hip ROLL, ankle ROLL
            key = "roll" if lat else "pitch"; iax = 1 if lat else 0
            lam2 = 0.0 if v.get("stable") else float(v["lam"])**2; R_ = v["R"]
            span = (pts[:, iax].min() - comw[iax], pts[:, iax].max() - comw[iax]); poly = min(abs(span[0]), abs(span[1]))
            auth = lambda mm: min(poly, 2*tau_ank[key]/((M + mm)*G))
            d3s = _bisect(lambda mm: mm*E_MAX/(M + mm) <= auth(mm))
            d3t = _bisect(lambda mm: mm*G*E_MAX*(1 + lam2*z_head/G) <= (M + mm)*G*auth(mm) + hip[key])
            d5 = _bisect(lambda mm: mm*G*E_MAX*(1 + lam2*(z_c - z_w[key])/G) <= waist[key])
            fr = max(0.0, (hh - R_))*E_MAX/(hh*max(h_p, 1e-3))      # palm push / load weight needed at the edge = head friction use (mu 1)
            ax_out[ax_name] = dict(lam2=lam2, poly=poly, D3s=d3s, D3t=d3t, D5_waist=d5, head_friction_use=fr, z_c_minus_z_waist=z_c - z_w[key], D1=v["m_max"])
        out["axes"] = ax_out; out["z_head"] = z_head; out["z_load_com"] = z_c
        for t in ("D3s", "D3t", "D5_waist"): terms[t] = min(a[t] for a in ax_out.values())
        terms["D4_friction"] = base["terms"].get("D4_friction")
        out["head_friction_ok"] = all(a["head_friction_use"] <= 1.0 for a in ax_out.values())
    else:
        com = HC.obj_contacts(m, qa, strategy, obj)[1]; x_ahead = float(com[0] - d.xanchor[jid("waist_pitch_joint")][0])
        terms["D3_zmp"] = base["terms"].get("D3_zmp"); terms["D5_waist"] = waist["pitch"]/(G*max(abs(x_ahead), 1e-3)); out["x_ahead_of_waist"] = x_ahead
    terms = {k: (np.inf if v is None else float(v)) for k, v in terms.items()}
    b = min(terms, key=lambda k: terms[k]); out["terms"] = {k: (None if not np.isfinite(v) else round(v, 2)) for k, v in terms.items()}
    out["m_max"] = round(terms[b], 2) if np.isfinite(terms[b]) else None; out["binding"] = b
    return out
