"""SUPPORT TRANSITION (rise / lower) representation: ONE prior family for the non-standing-start regime (P25-P29, P31; P26 lowers
onto a seat). The scaffold is priors/rise.support_path: a path of statically certified poses from the start posture to the goal
posture over EVOLVING SUPPORT SETS (feet, shins, hands, pelvis on a seat; contacts released only when the certificate holds without
them, limbs freed and re-placed, auxiliary hands acquired), derived from the body and the scene, never from a routine. It is
tracked on the true dynamics at a searched time scale and gain with free torque knots on top, every driven joint torque-controlled:
    u(t) = tau_LP(s(t))/tau_rated + PD(q_ref(t) - q) + J_c^T F_com + knots(t)
tau_LP is the certificate LP's own holding torque at the current path point (the support-dependent gravity hold, the one-foot
hold_sched of throw.py generalised to any support set); PD = the Tracker law (omega^2 I_eff, 2 omega I_eff) with I_eff the joint
inertia UNDER THE NODE'S SUPPORT (priors.rise.constrained_ieff; the free-floating diag(M^-1) left ankle rolls limp); F_com =
M (wc^2 (c_ref - c) + 2 wc (v_ref - v)) through the constrained COM Jacobian of the node's support.

    "torque_knots+rise_prior"   z = [knots nT x nk (torque offsets, rated-torque units) | tscale | omega | wc]

THE ROLLOUT (own loop; throw.rollout's actuator model and gates verbatim: 40 ms activation lag, torque-speed envelope, back-EMF brake,
energy accounting + closure, START_NOT_AT_REST, OVERSPEED) with the SUPPORT CONTRACT generalised from "the support feet stay loaded"
to the certified schedule: NONFOOT_CONTACT = a robot geom on the FLOOR (throw.rollout's rule) whose group is in NO support set of the
path and was not touching at the start (start contacts are the initial condition until they first break);
STANCE_LOST = a group the schedule declares load-bearing unloaded for FLIGHT_CONFIRM s (cumulative) inside its declared interval
(0.15 s of grace at each declared change), the final support through to the horizon; FELL = the COM more than 0.25 m below the
PATH's COM (throw.py: below the start's; a planned descent is not a fall). The horizon is the program's (demand["horizon"][1]) or the
representation's. value = goal progress from the GoalSpec (standing: -time to a held standing pose, graded failure; lower /
seated: -error), objective = -value; the PROGRAM's evaluate stays the judge."""
from __future__ import annotations
from typing import Optional
import numpy as np
from .base import Representation

FLIGHT_CONFIRM = 0.020
GRACE = 0.15
SPEED_FRAC = 0.06          # nominal joint speed along the path: 6 % of the rated no-load speed (quasi-static); tscale searches around it
COM_SPEED = 0.25           # m/s nominal COM speed cap along the path
GAINS = "constrained"      # PD inertia: "constrained" (under the node's support) | "free" (the Tracker's 1/diag(M^-1) at the start)
IEFF_CAP = 5.0             # constrained inertia capped at IEFF_CAP x the free-floating value
MASK_FREE = "legs"         # COM feedback withheld from free limbs: "legs" (a free leg only), "all" (free legs and arms), "none"
FF_REFINE = True           # feedforward from the least-total-torque certificate solution (False: the min-max solution)
T_HOLD_END = 0.6           # the path must end this long before the horizon (programs hold 0.4-0.5 s at rest)


SPEED_FRAC_FREE = 0.12     # joints of a limb that carries no load in the segment (a swinging leg, a reaching arm)


def _limbs(S):
    out = []
    for t in S["types"]:
        side = "left_" if t.startswith("left") else ("right_" if t.startswith("right") else None)
        out.append(None if side is None else ([side + "thigh", side + "shin", side + "foot"] if any(k in t for k in ("hip", "knee", "ankle")) else [side + "arm", side + "hand"]))
    return out


def _loaded_mask(S, active) -> np.ndarray:
    """1 for joints of the trunk and of limbs with a group in the support set, 0 for joints of free limbs."""
    a = set(active)
    if MASK_FREE == "none": return np.ones(len(S["types"]))
    return np.array([1.0 if (lb is None or set(lb) & a or (MASK_FREE == "legs" and lb[0].endswith("arm"))) else 0.0 for lb in _limbs(S)])


def path_timing(S, Q, active=None, speed_frac: float = SPEED_FRAC, com_speed: float = COM_SPEED) -> np.ndarray:
    """Nominal node times: each segment takes the longer of max_j |dq_j| / (f_j om_j) and |dCOM| / com_speed, f_j = speed_frac for
    joints of load-bearing limbs, SPEED_FRAC_FREE for joints of a limb none of whose groups is in the segment's support."""
    import mujoco
    m = S["m"]; J = S["jarr"]; qad = np.asarray(J["qad"]); om = np.asarray(J["om"], float); d = mujoco.MjData(m); com = []
    for q in Q: d.qpos[:] = q; mujoco.mj_forward(m, d); com.append(d.subtree_com[1].copy())
    com = np.array(com); dq = np.abs(np.diff(Q[:, qad], axis=0)); f = np.full(dq.shape, speed_frac)
    if active is not None:
        limb = []
        for t in S["types"]:
            side = "left_" if t.startswith("left") else ("right_" if t.startswith("right") else None)
            limb.append(None if side is None else ([side + "thigh", side + "shin", side + "foot"] if any(k in t for k in ("hip", "knee", "ankle")) else [side + "arm", side + "hand"]))
        for i in range(len(dq)):
            a = set(active[i]) & set(active[i + 1])
            for j, lb in enumerate(limb):
                if lb is not None and not (set(lb) & a) and not (set(lb) & (set(active[i]) | set(active[i + 1]))): f[i, j] = SPEED_FRAC_FREE
    dt = np.maximum((dq/(f*om)).max(1), np.linalg.norm(np.diff(com, axis=0), axis=1)/com_speed)
    return np.concatenate([[0.0], np.cumsum(np.maximum(dt, 1e-3))]), com


def build_reference(S, demand) -> dict:
    """priors.rise.support_path -> shortcut nodes -> tracker reference (node times at tscale 1, joint targets, LP holding torques in
    rated units for the driven joints, declared support per node, COM height per node)."""
    from .priors import rise as PR
    goal = PR.goal_from_spec(S, demand.get("goal"))
    if goal["kind"] == "unsupported":
        return dict(ok=False, reason=goal["why"], goal=goal)
    P = PR.support_path(S, goal)
    keep = PR.shortcut(S, P)
    Q = P["Q"][keep]; A = [P["active"][i] for i in keep]; m = S["m"]; J = S["jarr"]; aid = np.asarray(J["aid"]); tau_r = np.asarray(J["tau"], float)
    TAU = []; gr0 = PR.contact_groups(S); su0 = PR.surfaces(S)
    for i in keep:                                                       # feedforward: the refined (least total torque) certificate solution
        c_ = PR.certificate(S, P["Q"][i], gr0, su0, P["active"][i], margin=0.0, refine=FF_REFINE); t_ = c_["tau"] if c_.get("tau") is not None else P["tau"][i]
        TAU.append(np.zeros(len(aid)) if t_ is None else np.clip(np.asarray(t_)[aid]/tau_r, -1, 1))
    tn, com = path_timing(S, Q, A)
    groups = sorted({g for a in A for g in a}); gr = PR.contact_groups(S); su = PR.surfaces(S)
    # COM feedback through load-bearing limbs + trunk only (a free limb is light: J^T F whipped a swinging leg)
    Jc = np.array([PR.constrained_com_jacobian(S, q, a, gr, su)*_loaded_mask(S, a)[None, :] for q, a in zip(Q, A)])
    IE = np.array([PR.constrained_ieff(S, q, a, gr, su, cap=IEFF_CAP) for q, a in zip(Q, A)])
    return dict(ok=True, goal=goal, Q=Q, tau=np.array(TAU), Jcom=Jc, ieff=IE, active=A, t=tn, T=float(tn[-1]), com_z=com[:, 2], com=com, support_groups=groups,
                reached=P["reached"], reason=P["reason"], events=P["events"], n_path=int(P["n"]), n_nodes=len(keep), s_max=float(np.nanmax(P["s"][1:])) if P["n"] > 1 else float("nan"),
                margin_min=float(np.nanmin(P["margin"][1:])) if P["n"] > 1 else float("nan"))


class RiseController:
    """u(t) = tau_LP(t) + PD toward q_ref(t) (engine Tracker gains) + COM feedback through the active support: J_c^T M (wc^2 (c_ref - c)
    + 2 wc (v_ref - v)), J_c the constrained COM Jacobian of the node's support set (the support pushes the COM back onto the path);
    the reference holds its last node past its end."""
    def __init__(self, S, ref, tscale, omega, wc=6.0):
        from scipy.interpolate import PchipInterpolator, interp1d       # shape-preserving: a cubic spline through sparse shortcut nodes overshot (P31 fell with 6 nodes, stood with 5)
        J = S["jarr"]; self.qad = np.asarray(J["qad"]); self.dof = np.asarray(J["dof"]); self.tau = np.asarray(J["tau"], float)
        t = np.asarray(ref["t"])*float(tscale); Qj = np.asarray(ref["Q"])[:, self.qad]
        self.T = float(t[-1]); self.sp = PchipInterpolator(t, Qj, axis=0) if len(t) > 1 else None; self.dsp = self.sp.derivative() if self.sp is not None else None
        self.ff = interp1d(t, np.asarray(ref["tau"]), axis=0, bounds_error=False, fill_value=(ref["tau"][0], ref["tau"][-1])) if len(t) > 1 else (lambda tt: ref["tau"][0])
        self.q_end = Qj[-1]; ieff = np.asarray(ref["ieff"], float)          # per node: inertia under that node's support (constrained_ieff)
        if GAINS == "free": Minv = S["Minv_key"]; ieff = np.tile([1.0/Minv[d_, d_] for d_ in self.dof], (len(t), 1))
        self.kp = omega**2*ieff; self.kd = 2.0*omega*ieff; self.t_nodes = t
        self.Jc = np.asarray(ref["Jcom"]); self.com = np.asarray(ref["com"]); self.M = float(S["M"]); self.wc = float(wc); self.c_prev = None; self.t_prev = None
        self.csp = PchipInterpolator(t, self.com, axis=0) if len(t) > 1 else None; self.cvel = self.csp.derivative() if self.csp is not None else None
    def ref_at(self, tt):
        if self.sp is None or tt >= self.T: return self.q_end, np.zeros_like(self.q_end)
        return self.sp(tt), self.dsp(tt)
    def __call__(self, tt, d, S):
        qr, qdr = self.ref_at(tt); q = d.qpos[self.qad]; qd = d.qvel[self.dof]; i = int(np.clip(np.searchsorted(self.t_nodes, tt), 0, len(self.t_nodes) - 1))
        u = np.asarray(self.ff(min(tt, self.T)), float) + (self.kp[i]*(qr - q) + self.kd[i]*(qdr - qd))/self.tau
        if self.wc > 0:
            c = d.subtree_com[1].copy(); v = np.zeros(3) if self.c_prev is None or tt <= self.t_prev else (c - self.c_prev)/(tt - self.t_prev); self.c_prev, self.t_prev = c, tt
            cr = self.csp(min(tt, self.T)) if self.csp is not None else self.com[-1]; vr = self.cvel(tt) if (self.cvel is not None and tt < self.T) else np.zeros(3)
            F = self.M*(self.wc**2*(cr - c) + 2.0*self.wc*(vr - v)); u = u + (self.Jc[i].T @ F)/self.tau
        return u


def support_rollout(S, ref, p, ctrl_fn=None, horizon: float = 4.0, record: bool = False) -> dict:
    """The rollout (module doc): throw.rollout's actuator model and gates with the support contract of the certified schedule."""
    import mujoco, branch_i as BI, plant as P, throw as T
    from validity import TrajectoryValidityReport
    from .priors import rise as PR
    m = S["m"]; d = mujoco.MjData(m); BI.settle(S); BI._set_capacity_limits(m, S, None)
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    d.qpos[:] = S["q_settled"]; d.qvel[:] = S["v_settled"]; mujoco.mj_forward(m, d)
    E0 = float(d.energy[0] + d.energy[1]); start_qdot = float(np.max(np.abs(d.qvel[6:])))
    J = S["jarr"]; nT = len(S["types"]); tau_v = np.asarray(J["tau"], float); om_v = np.asarray(J["om"], float)
    for (jid, dof, qad, tau, om, sgn, aid) in (S["tmap"][t][0] for t in S["types"]):
        m.actuator_ctrlrange[aid][:] = [-tau, tau]; m.actuator_forcerange[aid][:] = [-tau, tau]; m.jnt_actfrcrange[jid][:] = [-tau, tau]
    kn = np.asarray(p["x"], float).reshape(nT, -1); NKl = kn.shape[1]; dt = m.opt.timestep; nst = int(round(horizon/dt)); knot_dt = horizon/NKl
    act = np.asarray(ref["tau"][0], float).copy(); alpha = 1.0 - np.exp(-dt/0.040)
    for (a_, dof, qad, tau, kp, kd) in S["held"]: d.ctrl[a_] = S["q_settled"][qad]
    # the support contract of the schedule: geom -> group, declared intervals per group (scaled times), the final support to the end
    groups = PR.contact_groups(S); g_of = {g: k for k, gs in groups.items() for g in gs}; surf_g = {s["geom"] for s in PR.surfaces(S)}
    ts = float(p["tscale"]); tn = np.asarray(ref["t"])*ts; A = ref["active"]; declared = set(ref["support_groups"]); intervals = {}
    for k in declared:
        on = np.array([k in a for a in A]); i = 0
        while i < len(on):
            if on[i]:
                j = i
                while j + 1 < len(on) and on[j + 1]: j += 1
                t0 = tn[i] + (GRACE if i > 0 else 0.0); t1 = (tn[j] - GRACE) if j < len(on) - 1 else horizon
                if t1 > t0: intervals.setdefault(k, []).append((t0, t1))
                i = j + 1
            else: i += 1
    com_ref = np.asarray(ref["com_z"], float)
    # contacts present AT THE START are the initial condition: allowed until they first break (then a re-touch is undeclared)
    init = {}
    for ci in range(d.ncon):
        c = d.contact[ci]; g1, g2 = int(c.geom1), int(c.geom2); o = g2 if g1 in surf_g else (g1 if g2 in surf_g else None)
        if o is not None and o in S["robot_geoms"]: init[g_of.get(o, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, o) or str(o))] = 0.0
    touching = set(); unl_t = {}
    W_act = W_con = W_pas = 0.0; nonfoot = []; unl = {k: 0.0 for k in intervals}; stance_lost = []; fell = False; overspeed = 0.0; over_j = np.zeros(nT); peak_grf = 0.0
    seg_w = []; jv = []; tq_tr = []; pw_tr = []; q_tr = []; qd_tr = []; foot_tr = []; vel6 = np.zeros(6); work = np.zeros(nT)
    fb = [S["feet"][0], S["feet"][1]]
    for s in range(nst):
        tt = s*dt; k = min(int(tt/knot_dt + 1e-9), NKl - 1); k2 = min(k + 1, NKl - 1); f = min(1.0, (tt - k*knot_dt)/knot_dt) if k < NKl - 1 else 0.0
        cmd = kn[:, k] + (kn[:, k2] - kn[:, k])*f
        if ctrl_fn is not None: cmd = cmd + ctrl_fn(tt, d, S)
        act += alpha*(cmd - act)
        w = d.qvel[J["dof"]]; frac = np.maximum(0.0, 1.0 - np.abs(w)/om_v); hi = np.where(w > 0, tau_v*frac, tau_v); lo = np.where(w < 0, tau_v*frac, tau_v)
        tq = np.clip(act*tau_v, -lo, hi); rw = np.abs(w)/om_v; overspeed = max(overspeed, float(rw.max())); over_j = np.maximum(over_j, rw)
        d.ctrl[J["aid"]] = tq; BI.backemf_brake(m, d, S)
        qv0 = d.qvel.copy(); mujoco.mj_step(m, d); qm = 0.5*(qv0 + d.qvel)
        W_act += float(np.dot(d.actuator_force, qm[S["act_dof"]]))*dt; W_con += float(np.dot(d.qfrc_constraint, qm))*dt; W_pas += float(np.dot(d.qfrc_passive, qm))*dt + float(np.dot(d.qfrc_applied, qm))*dt
        pw = tq*d.qvel[J["dof"]]; work += np.where(pw > 0, pw, 0.0)*dt
        load = {}; fz = 0.0; ff = [0.0, 0.0]
        for ci in range(d.ncon):
            c = d.contact[ci]; g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in surf_g: o = g2
            elif g2 in surf_g: o = g1
            else: continue
            if o not in S["robot_geoms"]: continue
            fv = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fv); fz += fv[0]; k_ = g_of.get(o)
            if o in fb[0]: ff[0] += fv[0]
            elif o in fb[1]: ff[1] += fv[0]
            nm = k_ or (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, o) or str(o)); touching.add(nm)
            if k_ is not None and k_ in declared: load[k_] = load.get(k_, 0.0) + fv[0]
            elif (g1 == S["floor"] or g2 == S["floor"]) and not (nm in init and init[nm] < FLIGHT_CONFIRM) and nm not in nonfoot:
                nonfoot.append(nm)                                               # judged on the FLOOR, as throw.rollout does (a seat edge brushed by a shin is not a fall)
        if s > 5:
            for k_, iv in intervals.items():
                if any(a <= tt <= b for a, b in iv) and load.get(k_, 0.0) < 1.0:
                    unl[k_] += dt; unl_t.setdefault(k_, [round(tt, 3), round(tt, 3)])[1] = round(tt, 3)
                    if unl[k_] >= FLIGHT_CONFIRM - 1e-12 and k_ not in stance_lost: stance_lost.append(k_)
        for nm in init:
            if nm not in touching: init[nm] += dt
        touching.clear(); peak_grf = max(peak_grf, fz)
        if s % 20 == 0:
            i_ref = int(np.clip(np.searchsorted(tn, tt), 0, len(tn) - 1))
            if float(d.subtree_com[1][2]) < com_ref[i_ref] - 0.25: fell = True
        q_tr.append(d.qpos.copy())
        if record:
            sw = []
            for b in S["seg"]: mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel6, 0); sw.append(float(np.linalg.norm(vel6[:3])))
            seg_w.append(sw); jv.append(d.qvel[J["dof"]].copy()); tq_tr.append(tq.copy()); pw_tr.append(pw.copy()); qd_tr.append(d.qvel.copy()); foot_tr.append(np.asarray(ff, float))
        if not np.all(np.isfinite(d.qpos)): break
    E1 = float(d.energy[0] + d.energy[1]); resid = (E1 - E0) - W_act; closure = resid - W_con - W_pas
    rep = TrajectoryValidityReport(start_qdot_max=start_qdot, energy_residual=resid, energy_tol=max(1.0, 0.02*abs(W_act)), actuator_work=W_act, contact_energy=0.0, ke_takeoff=0.0,
                                   nonfoot_contact=bool(nonfoot), took_off=True, peak_grf_bw=peak_grf/(S["M"]*T.G), energy_closure=closure, closure_tol=max(5.0, 0.02*(abs(W_act) + abs(W_con)))).evaluate()
    reasons = list(rep.reasons) + (["STANCE_LOST"] if stance_lost else []) + (["FELL"] if fell else []) + (["OVERSPEED"] if overspeed > T.OVERSPEED_TOL else [])
    if not np.all(np.isfinite(d.qpos)): reasons.append("NUMERICAL")
    gw = {}
    for g_, wv in zip(S["groups"], work): gw[g_] = gw.get(g_, 0.0) + float(wv)
    out = dict(reasons=reasons, nonfoot=nonfoot, stance_lost=stance_lost, unloaded={k: round(v, 3) for k, v in unl.items()}, unloaded_span=unl_t, overspeed=overspeed, overspeed_by_joint={S["types"][i]: round(float(v), 2) for i, v in enumerate(over_j) if v > 1.0},
               W_act=W_act, closure=closure, work=gw, peak_grf_bw=peak_grf/(S["M"]*T.G), final_state=(d.qpos.copy(), d.qvel.copy()))
    out["q"] = np.asarray(q_tr); out["dt"] = dt
    if record:
        out["trace"] = dict(seg_w=np.asarray(seg_w).tolist(), jv=np.asarray(jv).tolist(), torque=np.asarray(tq_tr).tolist(), power=np.asarray(pw_tr).tolist(), q=np.asarray(q_tr).tolist(), qvel=np.asarray(qd_tr).tolist(),
                            foot_load_bw=(np.asarray(foot_tr)/(S["M"]*T.G)).tolist(), dt=dt, release_step=int(nst), release_time=float(horizon))
    return out


def goal_value(goal: dict, demand: dict, q: np.ndarray, dt: float, horizon: float) -> tuple:
    """(value, measures) from the planner goal and the GoalSpec tolerances: standing -> -t_stand (held 0.4 s within tol) else
    -(H + 1 + graded error); lower -> -(|z_hold - z*| + excess descent speed); seated -> -(|z_end - z_path_end| + |x_end - x*|)."""
    from movement_discovery.tasks.locomotion._local import pelvis_kin
    xyz, yaw, tilt, v = pelvis_kin(q, dt); vz = np.gradient(xyz[:, 2], dt); sp = np.linalg.norm(np.column_stack([v, vz]), axis=1); n4 = max(1, int(round(0.4/dt)))
    gs = demand.get("goal"); tol = dict(getattr(gs, "tolerance", {}) or {})
    if goal["kind"] == "standing":
        tz = float(tol.get("z", 0.06)); tt = float(tol.get("tilt", np.radians(15))); ok = (np.abs(xyz[:, 2] - goal["z"]) < tz) & (tilt < tt) & (sp < 0.15)
        t_ok = None
        run = np.convolve(ok.astype(int), np.ones(n4, int), "valid")
        hit = np.where(run == n4)[0]
        if len(hit): t_ok = float(hit[0]*dt)
        err = float(np.mean(np.abs(xyz[-n4:, 2] - goal["z"])) + 0.3*np.mean(tilt[-n4:]) + 0.3*np.mean(sp[-n4:]))
        return (-t_ok if t_ok is not None else -(horizon + 1.0 + err)), dict(t_stand=t_ok, z_end=float(xyz[-1, 2]), z_max=float(xyz[:, 2].max()), err_end=err)
    if goal["kind"] == "lower":
        e = float(abs(xyz[-n4:, 2].mean() - goal["z"])); drop = float(max(0.0, -vz.min() - float(tol.get("vz_max", 0.6))))
        return -(e + drop + 0.3*float(sp[-n4:].max())), dict(z_end=float(xyz[-1, 2]), err=e, vz_min=float(vz.min()))
    if goal["kind"] == "seated":
        e = float(abs(xyz[-1, 0] - goal["x"])) + float(np.mean(sp[-n4:]))
        return -e, dict(z_end=float(xyz[-1, 2]), x_end=float(xyz[-1, 0]))
    return -1e3, {}


class TorqueKnotsRisePrior(Representation):
    name = "torque_knots+rise_prior"; label = "torque knots + tracked certified support-transition path (rise / lower)"
    TSCALE = (0.5, 1.5); OMEGA = (12.0, 40.0); WC = (2.0, 14.0)
    def __init__(self, task, horizon: Optional[float] = None):
        super().__init__(task, horizon)
    def native(self) -> bool: return hasattr(self.task, "one_foot_hold")
    def _d(self, demand):
        self._t(demand); d = dict(demand, prior=None)
        if d.get("driven") is None:
            from movement_discovery.embodiment import driven_set
            d["driven"] = driven_set(self.robot, "full_body")         # every actuated joint torque-driven: arms may bear load
        return d
    def H(self, demand) -> float:
        if self.horizon is not None: return float(self.horizon)
        h = demand.get("horizon")
        if isinstance(h, (list, tuple)) and len(h) == 2: return float(h[1])
        g = demand.get("goal"); return float(getattr(g, "horizon", None) or 3.0)
    def nkk(self, demand) -> int: return max(4, int(round(self.H(demand)/0.5)))
    def plant(self, demand, dt=None, fresh=False): return super().plant(self._d(demand), dt, fresh)
    def ref(self, S, demand) -> dict:
        key = (S["m"].opt.timestep, np.round(np.asarray(S["q_settled"]), 3).tobytes(), repr(getattr(demand.get("goal"), "params", None)))
        if key not in self._refs: self._refs[key] = build_reference(S, self._d(demand))
        return self._refs[key]
    def n_params(self, S, demand): return len(S["types"])*self.nkk(demand) + 3
    def tscale_range(self, S, demand):
        r = self.ref(S, demand); T = max(r.get("T", 1.0), 1e-3); cap = max(0.05, (self.H(demand) - T_HOLD_END)/T)
        lo, hi = self.TSCALE; hi = min(hi, cap); lo = min(lo, hi); return lo, hi
    def unpack(self, S, demand, z):
        z = np.asarray(z, float); n = len(S["types"])*self.nkk(demand); lo, hi = self.tscale_range(S, demand)
        return dict(x=z[:n], tscale=lo + (hi - lo)*(float(z[n]) + 1)/2, omega=self.OMEGA[0] + (self.OMEGA[1] - self.OMEGA[0])*(float(z[n + 1]) + 1)/2,
                    wc=self.WC[0] + (self.WC[1] - self.WC[0])*(float(z[n + 2]) + 1)/2, nk=self.nkk(demand), horizon=self.H(demand))
    def controller(self, S, demand, p):
        r = self.ref(S, demand)
        if not r.get("ok"): return None
        return RiseController(S, r, p["tscale"], p["omega"], p.get("wc", 6.0))
    def rollout(self, S, demand, p, ctrl_fn=None, record=False):
        d = self._d(demand); r = self.ref(S, d); H = self.H(d)
        if not r.get("ok"):
            return dict(value=-1e3, valid=False, reasons=["UNSUPPORTED_GOAL"], violation=10.0, trace=None, final_state=(S["q_settled"], S["v_settled"]), cost=0.0, extra=dict(reason=r.get("reason")))
        with self.start_ctx(S, d):
            o = support_rollout(S, r, p, ctrl_fn=ctrl_fn, horizon=H, record=record)
        q, dt = o.pop("q"), o.pop("dt")
        value, meas = goal_value(r["goal"], d, q, dt, H)
        viol = 0.0
        if "STANCE_LOST" in o["reasons"]: viol += 1.0 + 10.0*sum(o["unloaded"].values())
        if "NONFOOT_CONTACT" in o["reasons"]: viol += 5.0
        if "FELL" in o["reasons"]: viol += 5.0
        if "ENERGY_CLOSURE" in o["reasons"] or "ENERGY_ACCOUNTING_VIOLATION" in o["reasons"]: viol += 2.0
        if "OVERSPEED" in o["reasons"]: viol += 1.0 + 10.0*max(0.0, o["overspeed"] - 1.10)
        if "START_NOT_AT_REST" in o["reasons"] or "NUMERICAL" in o["reasons"]: viol += 1.0
        ex = dict(meas, tscale=p["tscale"], omega=p["omega"], wc=p.get("wc"), t_path=float(r["T"])*p["tscale"], nonfoot=o["nonfoot"], stance_lost=o["stance_lost"], unloaded=o["unloaded"], unloaded_span=o["unloaded_span"], overspeed=o["overspeed"],
                  overspeed_by_joint=o["overspeed_by_joint"], work=o["work"], W_act=o["W_act"], closure=o["closure"], t_rel=H, final_state=o["final_state"], trace=(o.get("trace") if record else None),
                  path=dict(n_path=r["n_path"], n_nodes=r["n_nodes"], reached=r["reached"], reason=r["reason"], s_max=r["s_max"], margin_min=r["margin_min"], support_groups=r["support_groups"], events=r["events"][:40]))
        return dict(value=float(value), valid=(viol <= 0 and not o["reasons"]), reasons=o["reasons"], violation=float(viol), trace=(o.get("trace") if record else None), final_state=o["final_state"],
                    cost=float(sum(o["work"].values())), extra=ex)
    def objective(self, rec, demand): return -float(rec["value"])
    def floor(self, demand): return None
    def sigma0(self, demand, warm): return 0.10 if warm is None else 0.05
    def z0(self, S, demand):
        """Prior-only: zero knots, the path at its nominal timing (tscale 1, clipped into the range that fits the horizon), omega 25,
        COM gain 10 (the slowest fitting scale stood up and then lost the stance on P25: 1.5x slower is not more static here)."""
        z = np.zeros(self.n_params(S, demand)); lo, hi = self.tscale_range(S, demand); ts = float(np.clip(1.0, lo, hi))
        z[-3] = 2*(ts - lo)/max(hi - lo, 1e-9) - 1 if hi > lo else 0.0; z[-2] = 2*(25.0 - self.OMEGA[0])/(self.OMEGA[1] - self.OMEGA[0]) - 1
        z[-1] = 2*(10.0 - self.WC[0])/(self.WC[1] - self.WC[0]) - 1; return z
    def warm(self, S, demand, stored): return np.asarray(stored["z"] if "z" in stored else stored["x"], float)
    def describe(self):
        return dict(super().describe(), prior="support_transition", params="knots nT x nk (torque offsets, rated units, over the horizon) + tscale + omega + wc (COM feedback, rad/s)", horizon=self.horizon)
