"""Brace preparation as a TRACKED reference: stance -> brace configuration over t_w (priors/brace.brace_prior: COM shifted toward the
contact direction, the limb pre-placed on the contact point, torque / tipping / friction caps), then a SUSTAINED PUSH segment for
the rest of the horizon: the capacity-weighted constant direction field u = c Jc^T d integrated from the brace pose while progress
along d continues (no reversal, no wind-up), feet anchored, joint limits at 95%. Representations:
    "torque_knots+brace_prior"  TorqueKnotsTrackedPrior whose reference is this one (EffectTask / ImpactTask / BaseEffectTask plants)
    "servo_knots+brace_prior"   ServoKnots whose reference is this one (ReachTask-derived plants; legs by the balance controller)
Both also answer a demand key prior="brace" (engine/construct.py-style demands) without editing tasks/*: the reference is PINNED into
the wrapped task's cache under the key its controller reads. Contact point and direction come from the demand (contact_point /
force_direction, else the object's near face along (elev, az), else target_world for servo tasks). A preparation OPTION, never
prescribed: programs list it in strategy_space beside the wind-up and the free knots."""
from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
from .torque_knots import TorqueKnotsTrackedPrior
from .servo_knots import ServoKnots

EFF_R = 0.04      # tasks/impact.py effector collider radius: the palm centre sits one radius short of the contact face


def _yaw0(q): qw, qx, qy, qz = np.asarray(q)[3:7]; return float(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))


def direction_world(demand: dict, q_start) -> np.ndarray:
    """World unit direction of the sustained force: demand["force_direction"] (world xyz) > demand["push"] (servo carry vector) > (elev, az) body frame."""
    if demand.get("force_direction") is not None: v = np.asarray(demand["force_direction"], float)
    elif demand.get("push") is not None and np.linalg.norm(demand["push"]) > 1e-9: v = np.asarray(demand["push"], float)
    else:
        e, a = map(float, demand.get("direction") or (0.0, 0.0)); y = _yaw0(q_start); c, s = np.cos(y), np.sin(y); b = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
        v = np.array([c*b[0] - s*b[1], s*b[0] + c*b[1], b[2]])
    return v/max(np.linalg.norm(v), 1e-12)


def contact_point(task, S, demand: dict, dvec) -> np.ndarray:
    """World contact point: demand["contact_point"]; else the object's near face along dvec minus the collider radius; else target_world;
    else the palm moved 0.15 m along dvec (an empty-hand push against nothing: the brace is still defined)."""
    import mujoco
    if demand.get("contact_point") is not None: return np.asarray(demand["contact_point"], float)
    o = demand.get("object")
    if o is not None and o.get("pos") is not None:
        pos = np.asarray(o["pos"], float)
        if o.get("kind", "ball") == "box":
            # the NEAR FACE along dvec; on it, the point closest to the effector's rest position (clipped inside the face minus the
            # collider radius): a push at the face centre of a 60 cm box demanded a twisted deep squat the plant could not hold
            size = np.asarray(o["size"], float); ax = int(np.argmax(np.abs(dvec))); half = float(size[ax])
            d_ = mujoco.MjData(S["m"]); d_.qpos[:] = S["q_settled"]; mujoco.mj_forward(S["m"], d_); eff = np.asarray(d_.site_xpos[S["site"]], float)
            p = pos - np.sign(dvec[ax])*half*np.eye(3)[ax]; p = p - dvec*EFF_R
            for k in range(3):
                if k != ax: p[k] = float(np.clip(eff[k], pos[k] - size[k] + EFF_R, pos[k] + size[k] - EFF_R))
            return p
        return pos - dvec*(float(o.get("radius", 0.05)) + EFF_R)
    if hasattr(task, "target_world"):
        try: return np.asarray(task.target_world(S, demand), float)
        except Exception: pass
    d = mujoco.MjData(S["m"]); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(S["m"], d); return d.site_xpos[S["site"]] + 0.15*dvec


def push_path(S, q_start, dvec, step: float = 0.02, n_max: int = 200, limit_frac: float = 0.05, capacity_mode: str = "power") -> np.ndarray:
    """Constant direction field from the brace pose: joint steps along u = c Jc^T d while the effector still progresses along d.
    Feet anchored on the base (priors.brace._anchor_feet); every 5th step certified statically."""
    import mujoco, throw as T, plant as P
    from .priors.brace import _anchor_feet, sagittal_projector
    m = S["m"]; J = S["jarr"]; qa = np.asarray(J["qad"]); dofs = np.asarray(J["dof"]); dv = np.asarray(dvec, float)
    jids = [int(m.dof_jntid[int(d_)]) for d_ in dofs]; lo = np.array([m.jnt_range[j][0] for j in jids]); hi = np.array([m.jnt_range[j][1] for j in jids]); lo, hi = lo + limit_frac*(hi - lo), hi - limit_frac*(hi - lo)
    c = J["tau"]*J["om"]/4.0 if capacity_mode == "power" else np.ones(len(J["tau"]))
    sup = S.get("support", (0, 1)); fb = [int(m.geom_bodyid[next(iter(S["feet"][i]))]) for i in sup]
    d0 = mujoco.MjData(m); d0.qpos[:] = q_start; mujoco.mj_forward(m, d0); feet0 = [(d0.xpos[b].copy(), d0.xmat[b].reshape(3, 3).copy()) for b in fb]
    cg = [g for g in P.PlantContractVerifier(m, mujoco.MjData(m), P.PlantContract(S["mjcf"], "torque")).collision_geoms() if g in S.get("robot_geoms", set(range(m.ngeom)))]
    q = np.array(q_start, float); path = [q.copy()]; sag = sagittal_projector(S)
    for k in range(n_max):
        Jc = T.constrained_jacobian(S, q); u = sag(c*(Jc.T @ dv)); nu = float(np.linalg.norm(u))
        if nu < 1e-12: break
        dq = np.clip(q[qa] + step*u/nu, lo, hi) - q[qa]; gain = float(dv @ (Jc @ dq))
        if gain <= 1e-5: break
        v = np.zeros(m.nv); v[dofs] = dq; qn = q.copy(); mujoco.mj_integratePos(m, qn, v, 1.0); qn = _anchor_feet(m, S, qn, feet0, fb)
        if k % 5 == 4 and not T._cert_ok(P.static_certificate(m, mujoco.MjData(m), qn, coll_geoms=cg)): break
        q = qn; path.append(q.copy())
    return np.array(path)


def brace_push_reference(S, p_contact, dvec, horizon: float, t_w: float = 0.40, nk: int = 10, fps: int = 100, iters: int = 40, **kw) -> dict:
    """Tracker-format reference: stance (0.05 s) -> brace over t_w -> sustained push to the horizon. takeoff_ref = horizon (the whole
    rollout is the effect: contact duration and force are measured over it)."""
    from .priors.brace import brace_prior
    br = brace_prior(S, p_contact, dvec, iters=iters, t_ref=t_w, horizon=horizon, fps=fps, nk=nk, **kw)
    q_s = np.array(S["q_settled"]); qb = np.asarray(br["brace"], float); P1 = np.array([q_s, qb]) if br["steps"] == 0 else np.vstack([q_s[None], br["Q"][int(0.05*fps) + 1:int(t_w*fps) + 1], qb[None]])
    P2 = push_path(S, qb, dvec)
    def along(P_, w):
        if len(P_) == 1: return P_[0]
        seg = np.linalg.norm(np.diff(P_, axis=0), axis=1); cs = np.concatenate([[0], np.cumsum(seg)]); L = w*cs[-1]
        i = min(int(np.searchsorted(cs, L, side="right")) - 1, len(seg) - 1); f = (L - cs[i])/max(seg[i], 1e-12); return P_[i] + f*(P_[i+1] - P_[i])
    n = int(horizon*fps) + 1; t = np.arange(n)/fps; Q = np.repeat(q_s[None], n, 0)
    for i, ti in enumerate(t):
        if ti <= 0.05: qq = q_s
        elif ti <= t_w: qq = along(P1, (ti - 0.05)/max(t_w - 0.05, 1e-6))
        else: qq = along(P2, min(1.0, (ti - t_w)/max(horizon - t_w, 1e-6)))
        Q[i] = qq; Q[i, 3:7] /= np.linalg.norm(Q[i, 3:7])
    return dict(key="brace_prior", start=q_s, Q=Q, fps=fps, takeoff_ref=float(horizon), horizon=float(horizon), nk=nk, windup_steps=0, path_steps=len(P2) - 1, prep_qpos=None, t_prep_end=0.05,
                t_w=float(t_w), t_ref=float(horizon), brace=br["brace"], f_max=br["f_max"], limiting=br["limiting"], limiting_joint=br["limiting_joint"], caps=br["caps"], feasible=br["feasible"], reach_err=br["reach_err"], contact_point=list(map(float, p_contact)), direction=list(map(float, dvec)), legs=True)


def hold_check(task, Sb, qb, tol_q: float = 0.15, tol_z: float = 0.08):
    """Does the plant's settle (stiff servos, torque limits, contacts) stay at the brace pose? (pelvis drop, joint drift, residual speed)."""
    import throw as T, branch_i as BI
    from engine import core
    kw = Sb.get("_setup_kw") or dict(site=T.CFG[task.robot]["hand_site"], ball=False, mass=0.0, support="both", driven=None)
    Sp = T.setup(task.robot, start=qb, **kw); BI.apply_dt(Sp, core.DT_OPT); BI.settle(Sp); qs = np.asarray(Sp["q_settled"]); qa = Sp["jarr"]["qad"]
    drift = float(np.max(np.abs(qs[qa] - qb[qa]))); dz = float(qb[2] - qs[2]); ok = drift <= tol_q and abs(dz) <= tol_z and float(Sp.get("settle_qdot_max", 1.0)) < 0.05
    return ok, dict(joint_drift=drift, pelvis_drop=dz, settle_qdot=float(Sp.get("settle_qdot_max", np.nan)))


def torque_brace_ref(task, S, demand: dict, horizon: float, t_w: float = 0.40) -> dict:
    """Reference for an EffectTask-family plant, built on the BARE robot when the task has one (tasks/impact.py: a free object breaks the
    certificate LP) and padded with the world plant's object columns."""
    Sb = task.bare(S, demand) if hasattr(task, "bare") else S; q0 = np.asarray(S["q_settled"]); dv = direction_world(demand, q0); pc = contact_point(task, S, demand, dv)
    t_w = float(demand.get("t_brace", max(t_w, 0.5*horizon)))       # a brace is quasi-static: half the horizon to reach it unless the demand says otherwise (0.35 s for a 22 cm squat toppled the robot)
    r = None
    for iters in (40, 0):                                            # HOLD CHECK: the plant's own settle at the brace pose must stay (the LP certificate passed a twisted squat the settle toppled); fallback = placement only
        r = brace_push_reference(Sb, pc, dv, horizon, t_w=t_w, nk=int(round(10*horizon/0.8)), iters=iters); ok, info = hold_check(task, Sb, np.asarray(r["brace"], float)); r["hold_ok"] = ok; r["hold_info"] = info; r["hold_fallback"] = (iters == 0)
        if ok: break
    nqb = Sb["m"].nq; qb = np.asarray(r["brace"], float)
    if nqb < S["m"].nq: r["Q"] = np.hstack([np.asarray(r["Q"]), np.tile(q0[nqb:], (len(r["Q"]), 1))])
    # PREPARATION-DEPENDENT GRAVITY HOLD: EffectTask.controller schedules the hold torques from the stance's to the ones settled at
    # prep_qpos over [0.05, t_prep_end] (the one-foot mechanism of throw.py, reused): a deep brace tracked from the STANDING hold sagged
    # and toppled at every speed; with its own settled hold the tracker only has to shape the motion
    r["prep_qpos"] = list(qb) + list(q0[nqb:]); r["prep_qpos_bare"] = list(qb); r["t_prep_end"] = float(t_w)
    # STATIC PUSH FEEDFORWARD: a position tracker presses only through its tracking error (the hand sat on the face at ~0 N); the brace's own
    # statics give the joint torques that apply f d at the point, tau_ff = Jc^T (f d) (feet pinned; the same map force_capacity bounds):
    # normalised by the rated torques, ramped in over 0.1 s after t_w, scaled by the searched amplitude. f_push = demand["f_push"] or half
    # the brace capacity (torque / tipping / friction caps), a body-derived default, never a technique.
    import throw as T
    Jc = T.constrained_jacobian(Sb, qb); f_push = float(demand.get("f_push", 0.5*max(float(r["f_max"]), 0.0))); a_ff = (Jc.T @ (f_push*dv))/np.maximum(np.asarray(Sb["jarr"]["tau"], float), 1e-9)
    r["f_push"] = f_push; r["a_ff"] = np.clip(a_ff, -1.0, 1.0).tolist()
    r["start"] = q0.copy(); return r


def servo_brace_ref(task, S, demand: dict, t_w: float = 0.40) -> dict:
    """Reference for a ReachTask-derived (servo) plant: built on the bare robot at the current (handed-off) robot pose, padded."""
    import throw as T, branch_i as BI
    from engine import core
    H = float(demand.get("horizon", 1.5)); q = np.asarray(S["q_settled"]); v = np.asarray(S["v_settled"])
    Sb = T.setup(task.robot, ball=False); BI.apply_dt(Sb, core.DT_OPT); BI.settle(Sb); nqb = Sb["m"].nq
    dv = direction_world(demand, q); pc = contact_point(task, S, demand, dv)
    t_w = float(demand.get("t_brace", max(t_w, 0.4*H)))
    with core.handoff(Sb, q[:nqb], v[:Sb["m"].nv]): r = brace_push_reference(Sb, pc, dv, H, t_w=t_w, nk=10)
    r["Q"] = np.hstack([np.asarray(r["Q"]), np.tile(q[nqb:], (len(r["Q"]), 1))]) if nqb < S["m"].nq else np.asarray(r["Q"]); r["start"] = q.copy(); r["takeoff_ref"] = float(min(H, t_w + 0.6)); return r


def push_feedforward(fn, ref: dict, p: dict, nT: int, ramp: float = 0.1):
    """ctrl_fn = tracker + amp * w(t) * a_ff, w ramping 0 -> 1 over `ramp` s after the brace time t_w (both in the reference's time scale)."""
    a_ff = np.asarray(ref.get("a_ff") or np.zeros(nT), float); ts = float(p.get("tscale", 1.0)); t_w = float(ref.get("t_w", 0.4))*ts; amp = float(p.get("amp", 1.0))
    def fn2(t, d, S_):
        u = fn(t, d, S_) if fn is not None else np.zeros(nT); w = float(np.clip((t - t_w)/max(ramp, 1e-6), 0.0, 1.0)); return u + amp*w*a_ff
    fn2.a_ff = a_ff; return fn2


class TorqueKnotsBracePrior(TorqueKnotsTrackedPrior):
    name = "torque_knots+brace_prior"; label = "torque knots + tracked brace -> sustained push reference (+ static push feedforward)"
    def __init__(self, task, horizon: Optional[float] = None): super().__init__(task, horizon, prior="brace")


class ServoKnotsBracePrior(ServoKnots):
    name = "servo_knots+brace_prior"; label = "servo targets + knot offsets over a brace -> sustained push reference"
    def __init__(self, task, horizon: Optional[float] = None): super().__init__(task, horizon, prior="brace")


def from_construct_scenario(name: str, robot: str = "G1", representation: str = "torque_knots+brace_prior", horizon: Optional[float] = None) -> dict:
    """Adapter for engine/construct.py SCENARIOS (e.g. "push_box_sustained"): the same (task, demand) with the brace representation, ONE
    prior-only rollout; returns dict(value, reasons, t_contact, f_max_obj, dmin_obj, ref). No search."""
    from engine import construct as CS
    from . import get
    task, demand = CS.make_task(robot, CS.SCENARIOS[name]); rep = get(representation)(task, horizon=horizon); S = rep.plant(demand); z0 = rep.z0(S, demand)
    rec = rep.run(S, demand, z0, record=True); ex = rec["extra"]; ref = rep.brace_ref(S, demand) if hasattr(rep, "brace_ref") else None
    return dict(scenario=name, value=float(rec["value"]), reasons=list(rec["reasons"]), violation=float(rec["violation"]), t_contact=ex.get("t_contact"), f_max_obj=ex.get("f_max_obj"), dmin_obj=ex.get("dmin_obj"), t_rel=ex.get("t_rel"),
                ref=(None if ref is None else dict(f_max=ref["f_max"], limiting=ref["limiting"], limiting_joint=ref["limiting_joint"], feasible=ref["feasible"], push_steps=ref["path_steps"], reach_err=ref["reach_err"], contact_point=ref["contact_point"], hold_ok=ref.get("hold_ok"), hold_fallback=ref.get("hold_fallback"))), rec=rec)
