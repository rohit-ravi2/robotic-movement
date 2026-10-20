"""ABSORB: the receiving analogue of the wind-up. For an INCOMING object (interception programs P73-P80) the body-derived
reference is: predict the object's free path from the world itself (the plant steps its own object with the robot pinned and
non-colliding: ballistic flight, floor rolling, table sliding all come out of the same call), choose the interception time
and point inside the body's workspace, PRE-POSITION the effector(s) there with a reach-style whole-body IK (support feet pinned
through the constrained Jacobian, the whole-body COM held over the support shifted AGAINST the incoming momentum, capacity-
weighted joint use), and at contact an ABSORB STROKE: the effector moves ALONG the object's incoming velocity, first gaining
speed before contact (velocity matching within the reach budget), then decelerating object + effector together over

    s = (m_obj + L) v'^2 / (2 F_use),      v' = (m_obj v + L v_c)/(m_obj + L)   (inelastic contact, momentum conserved)

where L is the effector's apparent mass along the incoming direction (operational-space inertia 1/(d^T Jc M^-1 Jc^T d) at the
intercept pose, summed over the receiving effectors), F_use half the limb's force capacity against the object
(priors.brace.force_capacity: torque / tipping / friction caps, gravity subtracted), v_c the pre-contact effector speed the
retreat budget (u_min, u_max: how far the IK can follow the line before leaving the workspace) allows. Hands are placed
symmetrically about the incoming direction (one effector: on the line; two: a 45 deg cradle either side, laterally ordered by
their rest positions). The geometry (IK along the retreat line) is computed once per plant; the TIMING LAW u(t) is rebuilt per
evaluation from the searched stroke scale and arrival shift (cheap), and tracked by throw.tracker, torque knots on top.

    "torque_knots+absorb_prior"   z = [knots nT x nk | release | stroke scale (0.5..2 x s, log) | arrival shift (+-0.08 s)]

Body-derived, object-derived, no per-program choreography: the program's world/demand (object, effector, support, colliders)
is all it reads. A preparation OPTION: programs list it in strategy_space after their first entry. MuJoCo only inside functions."""
from __future__ import annotations
from typing import Optional
import copy
import numpy as np
from .torque_knots import TorqueKnotsTrackedPrior

EFF_R = 0.04          # world.colliders / tasks/impact.py effector sphere radius
G = 9.80665
CRADLE = np.radians(45.0)
SHIFT = 0.08          # arrival-shift search range (s)
SCALE = (0.5, 2.0)    # stroke-scale search range (multiplies the momentum-derived stroke)


# ------------------------------------------------------------------ the world predicts its own object
def object_path(S, horizon: float, dt_out: float = 0.002) -> dict:
    """Free path of the world object 'ball_free' from the plant's launched state (S['v_settled'] carries qvel0): a copy of the model
    with every robot geom non-colliding, the robot's dofs re-pinned at the settled state each step. Returns t, p, v, R (body rotation)."""
    import mujoco
    m = copy.copy(S["m"]); rg = S["robot_geoms"]
    for g in rg: m.geom_contype[g] = 0; m.geom_conaffinity[g] = 0
    d = mujoco.MjData(m); q0 = np.asarray(S["q_settled"], float); v0 = np.asarray(S["v_settled"], float); d.qpos[:] = q0; d.qvel[:] = v0
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"); qa = int(m.jnt_qposadr[j]); da = int(m.jnt_dofadr[j]); bb = int(m.jnt_bodyid[j])
    keep_q = np.ones(m.nq, bool); keep_q[qa:qa + 7] = False; keep_v = np.ones(m.nv, bool); keep_v[da:da + 6] = False
    d.ctrl[:] = 0; mujoco.mj_forward(m, d); every = max(1, int(round(dt_out/m.opt.timestep))); n = int(round(horizon/m.opt.timestep))
    T_, P_, V_, R_ = [], [], [], []
    for s in range(n + 1):
        if s % every == 0:
            T_.append(s*m.opt.timestep); P_.append(d.xpos[bb].copy()); V_.append(d.qvel[da:da + 3].copy()); R_.append(d.xmat[bb].reshape(3, 3).copy())
        mujoco.mj_step(m, d); d.qpos[keep_q] = q0[keep_q]; d.qvel[keep_v] = 0.0
    gid = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bb][0]
    return dict(t=np.array(T_), p=np.array(P_), v=np.array(V_), R=np.array(R_), gtype=int(m.geom_type[gid]), gsize=np.array(m.geom_size[gid], float), mass=float(m.body_mass[bb]))


def object_extent(path: dict, k: int, dvec) -> float:
    """Half-extent of the object along dvec at sample k (sphere: radius; box: support function of the oriented box)."""
    import mujoco
    if path["gtype"] == int(mujoco.mjtGeom.mjGEOM_BOX): return float(np.sum(np.abs(path["R"][k].T @ np.asarray(dvec))*path["gsize"][:3]))
    return float(path["gsize"][0])


# ------------------------------------------------------------------ whole-body kinematics with the support feet pinned
def _support_bodies(S):
    m = S["m"]; return [int(m.geom_bodyid[next(iter(S["feet"][i]))]) for i in S.get("support", (0, 1))]


def _cjac(S, d, site_ids, fb):
    """Stacked constrained Jacobians (3 rows per site, + 2 COM-xy rows) w.r.t. the driven joints, support feet held still
    (floating-base velocity solved by least squares from the foot constraints, as throw.constrained_jacobian), and the base map Bv."""
    import mujoco
    m = S["m"]; dofs = np.asarray(S["jarr"]["dof"]); Jf = []
    for b in fb:
        a = np.zeros((3, m.nv)); r = np.zeros((3, m.nv)); mujoco.mj_jacBody(m, d, a, r, b); Jf += [a, r]
    Jf = np.vstack(Jf); Bv = -np.linalg.lstsq(Jf[:, :6], Jf[:, dofs], rcond=None)[0]
    rows = []
    for sid in site_ids:
        jp = np.zeros((3, m.nv)); mujoco.mj_jacSite(m, d, jp, None, sid); rows.append(jp[:, :6] @ Bv + jp[:, dofs])
    jc = np.zeros((3, m.nv)); mujoco.mj_jacSubtreeCom(m, d, jc, 0); rows.append((jc[:, :6] @ Bv + jc[:, dofs])[:2])
    return np.vstack(rows), Bv


def _constrained_inertia(S, d, Bv):
    import mujoco
    m = S["m"]; dofs = np.asarray(S["jarr"]["dof"]); M = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, M, d.qM)
    N = np.zeros((m.nv, len(dofs))); N[:6] = Bv; N[dofs, np.arange(len(dofs))] += 1.0
    Mc = N.T @ M @ N; return 0.5*(Mc + Mc.T)


def _inv_inertia(S, d, Bv):
    """Mc^-1, Mc = N^T M N: the joint-space inertia of the feet-pinned system (N: driven joint velocities -> full qvel, base through
    Bv, held joints locked). As IK weights it gives the MINIMUM-KINETIC-ENERGY motion (heavy legs / trunk move least)."""
    Mc = _constrained_inertia(S, d, Bv); n = len(Mc); return np.linalg.inv(Mc + 1e-6*np.trace(Mc)/n*np.eye(n))


class WholeBodyIK:
    """Damped, capacity-weighted Gauss-Newton on the driven joints: sites -> targets, whole-body COM xy -> support centre + shift.
    Double support: sagittal mirrored legs (priors.brace.sagittal_projector); single support: every joint free (the stance leg
    carries the COM over its foot). Joint limits at 95 %, feet re-anchored (priors.brace._anchor_feet)."""
    def __init__(self, S, site_ids, capacity_mode: str = "kinetic", limit_frac: float = 0.05, w_com: float = 2.0):
        import mujoco
        from .priors.brace import sagittal_projector
        self.S = S; self.m = S["m"]; self.sites = list(site_ids); J = S["jarr"]; self.qa = np.asarray(J["qad"]); self.dofs = np.asarray(J["dof"])
        jids = [int(self.m.dof_jntid[int(x)]) for x in self.dofs]; lo = np.array([self.m.jnt_range[j][0] for j in jids]); hi = np.array([self.m.jnt_range[j][1] for j in jids])
        self.lo, self.hi = lo + limit_frac*(hi - lo), hi - limit_frac*(hi - lo)
        self.mode = capacity_mode; c = J["tau"]*J["om"]/4.0 if capacity_mode == "power" else np.ones(len(J["tau"])); self.W = np.diag(c/np.mean(c))
        self.fb = _support_bodies(S); self.w_com = w_com; self.d = mujoco.MjData(self.m)
        self.proj = sagittal_projector(S) if len(S.get("support", (0, 1))) == 2 else (lambda dq: dq)
        q0 = np.asarray(S["q_settled"], float); self.q_post = q0.copy(); self.k_post = 0.1; self.fwd(q0); self.feet0 = [(self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()) for b in self.fb]
    def fwd(self, q):
        import mujoco
        self.d.qpos[:] = q; mujoco.mj_forward(self.m, self.d); return self.d
    def support_centre(self, q):
        d = self.fwd(q); return np.array([d.geom_xpos[g][:2] for i in self.S.get("support", (0, 1)) for g in self.S["feet"][i]]).mean(0)
    def state(self, q):
        d = self.fwd(q); return [d.site_xpos[s].copy() for s in self.sites], d.subtree_com[0][:2].copy()
    def solve(self, q, targets, com_xy, iters: int = 30, tol: float = 2e-3, lam: float = 1e-3, step_max: float = 0.15):
        import mujoco
        from .priors.brace import _anchor_feet
        q = np.array(q, float)
        for it in range(iters):
            kp = self.k_post if it < max(1, (2*iters)//3) else 0.0       # posture-regularised first, then a pure-task polish (the term leaks into the task far out)
            P, c = self.state(q); e = np.concatenate([np.asarray(t_) - p for t_, p in zip(targets, P)] + [self.w_com*(np.asarray(com_xy) - c)])
            if np.max(np.abs(e[:-2])) < tol and np.max(np.abs(e[-2:])) < 2*tol*self.w_com: break
            A, Bv = _cjac(self.S, self.d, self.sites, self.fb); A[-2:] *= self.w_com
            Wd = _inv_inertia(self.S, self.d, Bv) if self.mode == "kinetic" else self.W
            G_ = Wd @ A.T @ np.linalg.inv(A @ Wd @ A.T + lam*np.trace(A @ Wd @ A.T)/len(e)*np.eye(len(e))); dq = G_ @ e
            dq = dq + (np.eye(len(dq)) - G_ @ A) @ (kp*(self.q_post[self.qa] - q[self.qa]))       # null space: stay near the settled stance
            dq = self.proj(dq); nrm = float(np.max(np.abs(dq)))
            if nrm > step_max: dq *= step_max/nrm
            dq = np.clip(q[self.qa] + dq, self.lo, self.hi) - q[self.qa]
            v = np.zeros(self.m.nv); v[self.dofs] = dq; v[:6] = Bv @ dq; mujoco.mj_integratePos(self.m, q, v, 1.0); q = _anchor_feet(self.m, self.S, q, self.feet0, self.fb)
        P, c = self.state(q); err = max(float(np.linalg.norm(np.asarray(t_) - p)) for t_, p in zip(targets, P))
        return q, err, float(np.linalg.norm(np.asarray(com_xy) - c))


# ------------------------------------------------------------------ the absorb stroke (pure arithmetic)
def stroke_plan(speed: float, m_obj: float, L: float, caps: dict, n_eff: int, u_min: float, u_max: float) -> dict:
    """Velocity matching + absorb stroke inside the retreat room [u_min, u_max] (u < 0: toward the object, before contact).
    LIMB vs SUPPORT: the effector's own acceleration is torque-limited (internal), F_tau = n_eff * cap_torque / 2; the object's momentum
    also loads the support, bounded by the tipping / friction caps (half of each used: tracking and gravity headroom). Pre-contact the
    effector accelerates to v_c along the object's velocity (a_acc = F_tau / L) over d_pre = v_c^2 / (2 a_acc); the contact is
    inelastic, v' = (m_obj v + L v_c)/(m_obj + L); object + effector then decelerate at a_dec = F_use/(m_obj + L) over
    s = (m_obj + L) v'^2 / (2 F_use). v_c is the largest speed (<= v) whose pre-travel and stroke fit the room."""
    F_tau = max(10.0, 0.5*float(caps["torque"]))*n_eff; tip = float(caps["tipping"]) if float(caps["tipping"]) > 0 else float(caps["torque"])
    F_sup = max(5.0, 0.5*min(tip, float(caps["friction"]))); F_use = min(F_tau, F_sup*(m_obj + L)/max(m_obj, 1e-6)); a_acc = F_tau/L; a_dec = F_use/(m_obj + L)
    v_c = float(min(speed, np.sqrt(2*a_acc*max(-u_min, 0.0))))
    for _ in range(60):                                     # the stroke must fit its room: lower the matched speed until it does
        v1 = (m_obj*speed + L*v_c)/(m_obj + L); s = v1*v1/(2*a_dec)
        if s <= u_max or v_c <= 1e-3: break
        v_c *= 0.9
    v_c = max(v_c, 0.0); v1 = (m_obj*speed + L*v_c)/(m_obj + L); s = min(v1*v1/(2*a_dec), max(u_max, 0.0)); d_pre = v_c*v_c/(2*a_acc); tau = 2*d_pre/v_c if v_c > 1e-3 else 0.0
    return dict(F_tau=F_tau, F_sup=F_sup, F_use=F_use, a_acc=a_acc, a_dec=a_dec, v_c=v_c, v1=v1, s=s, d_pre=d_pre, tau=tau, rel_impact=float(speed - v_c))


# ------------------------------------------------------------------ the absorb geometry (once per plant)
def receivers(task, S, demand):
    """The receiving effectors: the CatchTask's colliders (S['_hand_bodies'] keys, main effector first), else the demand effector.
    Each with the collider-centre offset from its site (foot spheres sit r above the sole site), in world coordinates at rest."""
    import mujoco
    m = S["m"]; names = list(S.get("_hand_bodies") or [demand.get("effector")]); eff = demand.get("effector")
    if eff in names: names = [eff] + [n for n in names if n != eff]
    d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d); out = []
    for n in names:
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n); gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"eff_{n}")
        off = (d.geom_xpos[gid] - d.site_xpos[sid]) if gid >= 0 else np.zeros(3)
        out.append(dict(name=n, rest=d.site_xpos[sid].copy(), offset=off.copy()))
    return out


def _lateral(vhat):
    e = np.cross([0.0, 0.0, 1.0], vhat); n = np.linalg.norm(e)
    return e/n if n > 1e-6 else np.array([0.0, 1.0, 0.0])


def _apparent_mass(S, d, sids, fb, dvec):
    """Effector apparent mass along dvec, 1/(d^T Jc Mc^-1 Jc^T d) per site, summed (Mc: _inv_inertia)."""
    A, Bv = _cjac(S, d, sids, fb); Mi = _inv_inertia(S, d, Bv); L = 0.0
    for k in range(len(sids)):
        Jk = A[3*k:3*k + 3]; L += 1.0/max(float(dvec @ Jk @ Mi @ Jk.T @ dvec), 1e-6)
    return L


def absorb_geometry(task, S, demand: dict, horizon: float, dt_cand: float = 0.03, du: float = 0.01, u_lim: float = 0.45, verbose: bool = False) -> dict:
    """Interception choice + pre-position + retreat line. For each candidate arrival time t (every dt_cand from 0.15 s to H - 0.12 s)
    the receivers' collider centres are put at the object's leading face (cradle offsets), the IK places them (warm-started along t),
    and the retreat line is followed both ways (u < 0 toward the object = pre-contact acceleration room, u > 0 along its velocity =
    stroke room). Score = the predicted post-contact relative speed (object speed minus achievable matched effector speed), ties
    broken by reach residual; a candidate must leave the joints time to reach the ready pose (|dq_j| / (0.35 omega_j) + 0.1 s)."""
    import mujoco
    from .priors.brace import force_capacity
    Sb = task.bare(S, demand) if hasattr(task, "bare") else S; nqb = Sb["m"].nq; q0w = np.asarray(S["q_settled"], float)
    path = object_path(S, horizon); rc = receivers(task, S, demand); sids = [mujoco.mj_name2id(Sb["m"], mujoco.mjtObj.mjOBJ_SITE, r["name"]) for r in rc]
    ik = WholeBodyIK(Sb, sids); q_s = np.array(Sb["q_settled"], float); m_obj = path["mass"]; J = Sb["jarr"]
    sc0 = ik.support_centre(q_s); M = float(Sb["M"]); _, com0 = ik.state(q_s); com_h = float(ik.d.subtree_com[0][2])
    restc = np.mean([r["rest"] for r in rc], axis=0)
    def targets_at(k, u=0.0):
        v = path["v"][k]; sp = float(np.linalg.norm(v)); vh = v/max(sp, 1e-9); el = _lateral(vh); ext = object_extent(path, k, vh) + EFF_R; out = []
        for r in rc:
            sgn = 0.0 if len(rc) == 1 else float(np.sign(np.dot(r["rest"] - restc, el)) or 1.0); th = 0.0 if len(rc) == 1 else CRADLE
            ctr = path["p"][k] + ext*(np.cos(th)*vh + np.sin(th)*sgn*el) + u*vh; out.append(ctr - r["offset"])
        return out, vh, sp
    def t_move(q1): return 0.1 + float(np.max(np.abs(q1[J["qad"]] - q_s[J["qad"]])/(0.35*np.asarray(J["om"], float))))
    ts = path["t"]; cands = []; q_prev = q_s.copy()
    for t in np.arange(0.15, horizon - 0.12 + 1e-9, dt_cand):
        k = int(np.argmin(np.abs(ts - t))); tg, vh, sp = targets_at(k)
        if sp < 0.05: continue
        q_c, err, _ = ik.solve(q_prev, tg, sc0, iters=60); q_prev = q_c
        if err > 0.03: cands.append(dict(t=float(t), k=k, err=err, ok=False)); continue
        cands.append(dict(t=float(t), k=k, err=err, ok=True, q=q_c, vh=vh, sp=sp))
    best = None
    for c in cands:
        if not c["ok"]: continue
        k, vh, sp = c["k"], c["vh"], c["sp"]
        # retreat line both ways (continuation, 1 cm steps)
        line = {0.0: c["q"]}
        for sgn in (-1.0, 1.0):
            q = c["q"]; u = 0.0
            while abs(u) < u_lim - 1e-9:
                u_n = u + sgn*du; tg, _, _ = targets_at(k, u_n); q_n, err, _ = ik.solve(q, tg, sc0, iters=9)
                if err > max(0.012, c["err"] + 0.006): break
                q, u = q_n, u_n; line[round(u, 4)] = q
        us = np.array(sorted(line)); u_min, u_max = float(us[0]), float(us[-1]); d = ik.fwd(c["q"])
        L = _apparent_mass(Sb, d, sids, ik.fb, vh)
        Sb_ = dict(Sb, site=sids[0]); cap = force_capacity(Sb_, c["q"], targets_at(k)[0][0], -vh)
        sp_ = stroke_plan(sp, m_obj, L, cap["caps"], len(rc), u_min, u_max); F_use, a_acc, a_dec, v_c, v1, s, d_pre, tau = (sp_[k_] for k_ in ("F_use", "a_acc", "a_dec", "v_c", "v1", "s", "d_pre", "tau"))
        q_ready = line[round(float(us[np.argmin(np.abs(us + d_pre))]), 4)]; t_need = 0.05 + t_move(q_ready) + tau
        c.update(u=us, Q=np.array([line[round(float(x), 4)] for x in us]), u_min=u_min, u_max=u_max, L=L, F_cap=cap, F_use=F_use, a_acc=a_acc, a_dec=a_dec, v_c=v_c, v1=v1, s=s, d_pre=d_pre, tau=tau, t_need=t_need,
                 rel_impact=float(sp - v_c), timing_ok=bool(c["t"] >= t_need))
        if verbose: print(f"  t_a {c['t']:.2f} err {c['err']*100:.1f}cm u [{u_min:+.2f},{u_max:+.2f}] L {L:.2f} F_use {F_use:.0f} v {sp:.2f} v_c {v_c:.2f} s {s:.3f} t_need {t_need:.2f}", flush=True)
        key = (not c["timing_ok"], round(c["rel_impact"], 1), -round(c["t"] - c["t_need"], 2))     # least impact speed, then the most preparation slack
        if best is None or key < best[0]: best = (key, c)
    if best is None: raise RuntimeError(f"absorb: no reachable interception on the object's path (closest IK residual {min((c['err'] for c in cands), default=np.nan):.3f} m)")
    c = best[1]; k = c["k"]
    # COM shifted AGAINST the incoming momentum (toward the object) by the ZMP displacement of the peak absorb force at the
    # contact height, capped at 4 cm (inside the support), and the line re-solved with that shift
    F_pk = (m_obj + c["L"])*c["v1"]**2/(2*max(c["s"], 1e-3)); h = float(targets_at(k)[0][0][2]); dxy = -c["vh"][:2]; nxy = float(np.linalg.norm(dxy))
    shift = (dxy/nxy if nxy > 1e-6 else np.zeros(2))*min(0.04, F_pk*min(nxy, 1.0)*h/(M*G)); com_t = sc0 + shift
    Qs = []
    for i, u in enumerate(c["u"]):
        tg, _, _ = targets_at(k, float(u)); q, _, _ = ik.solve(c["Q"][i], tg, com_t, iters=8); Qs.append(q)
    Qs = np.array(Qs)
    pad = (lambda Q_: np.hstack([Q_, np.tile(q0w[nqb:], (len(Q_), 1))]) if nqb < len(q0w) else Q_)
    g = dict(key="absorb_prior", t_a=c["t"], k=k, p_obj=path["p"][k].tolist(), v_obj=path["v"][k].tolist(), vhat=c["vh"].tolist(), speed=c["sp"], m_obj=m_obj, L=c["L"], F_use=c["F_use"], F_cap=dict(f_max=c["F_cap"]["f_max"], limiting=c["F_cap"]["limiting"], caps=c["F_cap"]["caps"]),
             a_acc=c["a_acc"], a_dec=c["a_dec"], v_c=c["v_c"], v1=c["v1"], s=c["s"], d_pre=c["d_pre"], tau=c["tau"], t_need=c["t_need"], timing_ok=c["timing_ok"], rel_impact=c["rel_impact"], reach_err=c["err"],
             u=c["u"], Qline=pad(Qs), u_min=c["u_min"], u_max=c["u_max"], com_shift=shift.tolist(), F_peak=F_pk, start=q0w.copy(), start_bare=q_s, receivers=[r["name"] for r in rc], nqb=nqb,
             candidates=[dict(t=x["t"], err=x["err"], ok=x["ok"], v_c=x.get("v_c"), s=x.get("s"), rel_impact=x.get("rel_impact"), timing_ok=x.get("timing_ok")) for x in cands])
    return g


def _q_on_line(g, u):
    us = g["u"]; u = float(np.clip(u, us[0], us[-1])); i = int(np.clip(np.searchsorted(us, u) - 1, 0, len(us) - 2)); f = (u - us[i])/max(us[i + 1] - us[i], 1e-12) if len(us) > 1 else 0.0
    Q = g["Qline"]; q = Q[i] + f*(Q[min(i + 1, len(Q) - 1)] - Q[i]); q = q.copy(); q[3:7] /= np.linalg.norm(q[3:7]); return q


def timing_law(g, scale: float = 1.0, shift: float = 0.0, horizon: float = 0.8, fps: int = 100) -> dict:
    """Tracker-format reference from the geometry: stance (0.05 s) -> ready pose (smoothstep, until t_a - tau) -> constant
    acceleration along the line to v_c at arrival -> continue at v' and decelerate to rest over scale * s -> hold."""
    t_a = g["t_a"] + shift; tau = g["tau"]; d_pre = g["d_pre"]; v_c = g["v_c"]; v1 = g["v1"]; s = max(0.01, min(scale*g["s"], g["u_max"])); a_dec = v1*v1/(2*s) if v1 > 1e-6 else 1.0
    T_s = v1/a_dec if v1 > 1e-6 else 0.0; t_p = max(0.06, t_a - tau); q_s = np.asarray(g["start"], float); q_r = _q_on_line(g, -d_pre)
    n = int(round(horizon*fps)) + 1; Q = np.repeat(q_s[None], n, 0)
    for i in range(n):
        t = i/fps
        if t <= 0.05: Q[i] = q_s; continue
        if t <= t_p:
            w = (t - 0.05)/max(t_p - 0.05, 1e-6); w = w*w*(3 - 2*w); Q[i] = q_s + w*(q_r - q_s); Q[i, 3:7] /= np.linalg.norm(Q[i, 3:7]); continue
        if t <= t_a: u = -d_pre + 0.5*(v_c/max(tau, 1e-6))*(t - t_p)**2 if tau > 1e-6 else 0.0
        else: tt = min(t - t_a, T_s); u = v1*tt - 0.5*a_dec*tt*tt
        Q[i] = _q_on_line(g, u)
    t_end = t_a + T_s
    return dict(key="absorb_prior", start=q_s, Q=Q, fps=fps, horizon=float(horizon), takeoff_ref=float(min(horizon, t_end)), nk=10, t_prep_end=float(t_p), prep_qpos=list(q_r), prep_qpos_bare=list(q_r[:g["nqb"]]),
                t_a=float(t_a), t_end=float(t_end), stroke=s, scale=float(scale), shift=float(shift))


FF_ON = True
class StanceTracker:
    """throw.tracker's PD law (normalised command kp (q_r - q) + kd (qd_r - qd), clamped cubic spline of the reference, held
    past its end) with gains from the FEET-PINNED joint-space inertia diag(Mc) at the start pose instead of the free-floating
    1/Minv_jj: with the feet on the floor a hip / ankle / waist joint moves the whole upper body, and the free-floating gains left
    the legs at ~30 % torque with 0.5 rad error while the arm's reaction pitched the trunk 18 deg."""
    def __init__(self, S, ref, omega=20.0, zeta=1.0):
        import mujoco
        from scipy.interpolate import CubicSpline
        J = S["jarr"]; Qs = np.asarray(ref["Q"]); t = np.arange(len(Qs))/ref["fps"]; sp = CubicSpline(t, Qs[:, J["qad"]], bc_type="clamped"); dsp = sp.derivative()
        dt = S["m"].opt.timestep; tg = np.arange(0, ref["horizon"] + 2*dt, dt); tc = np.minimum(tg, t[-1]); self.QR = sp(tc); self.QDR = dsp(tc); self.QDR[tg >= t[-1]] = 0.0; self.dt = dt; self.J = J
        key = ("Mc_diag", id(S["m"]))
        if key not in S:
            d = mujoco.MjData(S["m"]); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(S["m"], d); _, Bv = _cjac(S, d, [], _support_bodies(S))
            free = np.array([1.0/S["Minv_key"][dof, dof] for dof in J["dof"]]); jn = [mujoco.mj_id2name(S["m"], mujoco.mjtObj.mjOBJ_JOINT, int(S["m"].dof_jntid[int(x)])) or "" for x in J["dof"]]
            trunk = np.array([any(k in n for k in STANCE_JOINTS) for n in jn]); S[key] = np.where(trunk, np.maximum(np.diag(_constrained_inertia(S, d, Bv)), free), free)
        ieff = S[key]; self.kp = omega**2*ieff; self.kd = 2*zeta*omega*ieff; self.tau = J["tau"]
    def __call__(self, t, d, S):
        i = min(int(round(t/self.dt)), len(self.QR) - 1); q = d.qpos[self.J["qad"]]; qd = d.qvel[self.J["dof"]]
        return (self.kp*(self.QR[i] - q) + self.kd*(self.QDR[i] - qd))/self.tau


STANCE_JOINTS = ("hip", "waist")   # the joints that carry the trunk over the stance get the feet-pinned inertia (stiff ankles rocked the feet off the floor)
TRACKER = "stance"   # "stance" (StanceTracker) | "throw" (throw.tracker)
OMEGA_MAX = 40.0     # PD + 40 ms first-order actuator lag: tau s^3 + s^2 + 2 w s + w^2 is stable for w < 2 / tau = 50 rad/s
LEAD = 0.04          # throw.rollout's actuator filter time constant (act += alpha (cmd - act), tau = 40 ms): the command leads by it


def inertial_feedforward(S, r: dict) -> np.ndarray:
    """The reference's own INERTIAL inverse dynamics, tau = ID(q, qd, qdd) - ID(q, 0, 0) (gravity is the hold schedule's), on the
    upper-body driven joints (leg torques depend on the contact forces), as activation fractions per reference frame (nF x nT)."""
    import mujoco
    m = S["m"]; J = S["jarr"]; dofs = np.asarray(J["dof"]); Q = np.asarray(r["Q"], float); fps = float(r["fps"]); n = len(Q); h = 1.0/fps
    V = np.zeros((n, m.nv)); A = np.zeros((n, m.nv))
    for i in range(n):
        a, b = max(i - 1, 0), min(i + 1, n - 1)
        if b > a: mujoco.mj_differentiatePos(m, V[i], (b - a)*h, Q[a], Q[b])
    A[1:-1] = (V[2:] - V[:-2])/(2*h); up = np.array([g != "legs" for g in S["groups"]]); d = mujoco.MjData(m); out = np.zeros((n, len(dofs)))
    for i in range(n):
        d.qpos[:] = Q[i]; d.qvel[:] = V[i]; d.qacc[:] = A[i]; mujoco.mj_inverse(m, d); f1 = d.qfrc_inverse[dofs].copy()
        d.qvel[:] = 0; d.qacc[:] = 0; mujoco.mj_inverse(m, d); out[i] = np.where(up, (f1 - d.qfrc_inverse[dofs])/np.asarray(J["tau"], float), 0.0)
    return np.clip(out, -1.0, 1.0)


def with_feedforward(fn, ff: np.ndarray, fps: float, lead: float = LEAD):
    def fn2(t, d, S_):
        i = min(int(round((t + lead)*fps)), len(ff) - 1); return fn(t, d, S_) + ff[i]
    fn2.ff = ff; return fn2


class TorqueKnotsAbsorbPrior(TorqueKnotsTrackedPrior):
    name = "torque_knots+absorb_prior"; label = "torque knots + tracked absorb reference (intercept pre-position -> velocity-matched stroke along the incoming velocity)"
    def __init__(self, task, horizon: Optional[float] = None): super().__init__(task, horizon, prior="absorb")
    def _H(self):
        import throw as T
        return float(self.horizon or T.HOR)
    def geometry(self, S, demand):
        """One geometry per (object, effector, support, driven) shared by every timestep's plant (the two-timestep check replays the
        SAME reference; only the start pose is the plant's own), cached per plant id; pinned into the task's reference cache."""
        key = ("absorb", id(S["m"]))
        if key not in self._refs:
            if demand.get("object") is None: raise ValueError("torque_knots+absorb_prior needs an incoming world object (demand['object'])")
            gk = ("absorb_geom", str(demand.get("object")), demand.get("effector"), demand.get("support", "both"), tuple(demand.get("driven") or ()))
            if gk not in self._refs: self._refs[gk] = absorb_geometry(self.task, S, demand, self._H())
            g = dict(self._refs[gk], start=np.asarray(S["q_settled"], float).copy()); self._refs[key] = g
            self.task._refs[self._key(S, self._d(demand))] = timing_law(g, horizon=self._H())   # PINNED: runner.run stores this reference, not a wind-up
        return self._refs[key]
    def ref(self, S, demand): self.geometry(S, demand); return self.task._refs[self._key(S, self._d(demand))]
    def unpack(self, S, demand, z):
        if not self.native(): return super().unpack(S, demand, z)
        z = np.asarray(z, float); nk = self.nk(); n = len(S["types"])*nk + 1; p = dict(x=z[:n], nk=nk)
        p["scale"] = float(np.exp(np.log(SCALE[0]) + (np.log(SCALE[1]) - np.log(SCALE[0]))*(np.clip(z[n], -1, 1) + 1)/2)); p["shift"] = float(SHIFT*np.clip(z[n + 1], -1, 1))
        p["amp"] = p["scale"]; p["tscale"] = 1.0; p["omega"] = self.omega(S, demand); return p
    def omega(self, S, demand) -> float:
        """Tracking bandwidth from the stroke itself: omega = 4 / T_stroke (T_stroke = 2 s / v'), at least the demand's, at most OMEGA_MAX
        (a ~0.1 s receiving stroke tracked at the throw's omega 20 lagged the object by 0.1-0.2 m)."""
        g = self.geometry(S, demand); T_s = 2*g["s"]/max(g["v1"], 1e-3) if g["v1"] > 1e-3 else 0.2
        return float(np.clip(4.0/max(T_s, 1e-3), min(float(demand.get("omega", 20.0)), OMEGA_MAX), OMEGA_MAX))
    def controller(self, S, demand, p):
        import throw as T
        if not self.native(): return super().controller(S, demand, p)
        g = self.geometry(S, demand); r = timing_law(g, p.get("scale", 1.0), p.get("shift", 0.0) - LEAD, self._H())   # tracked reference led by the actuator lag
        a0 = np.array([S["a_hold"][t] for t in S["types"]]); a1 = self.task.one_foot_hold(S, self._d(demand), r); t0, t1 = 0.05, r["t_prep_end"]
        S["hold_sched"] = lambda tt, a0=a0, a1=a1, t0=t0, t1=t1: a0 + (a1 - a0)*float(np.clip((tt - t0)/max(t1 - t0, 1e-6), 0.0, 1.0))
        fn = StanceTracker(S, r, p["omega"]) if TRACKER == "stance" else T.tracker(S, r, p["omega"], 1.0)
        return with_feedforward(fn, inertial_feedforward(S, r), r["fps"], 0.0) if FF_ON else fn      # r is already led by LEAD
    def z0(self, S, demand):
        if not self.native(): return super().z0(S, demand)
        import throw as T
        g = self.geometry(S, demand); r = timing_law(g, horizon=self._H()); H = self._H(); n = len(S["types"])*self.nk(); z = np.zeros(self.n_params(S, demand))
        t_end = min(H, r["t_end"] + 0.02); z[n] = float(np.clip(2*(t_end - T.T_REL_MIN)/(H - T.T_REL_MIN) - 1, -1, 1))
        z[n + 1] = 2*(0.0 - np.log(SCALE[0]))/(np.log(SCALE[1]) - np.log(SCALE[0])) - 1; z[n + 2] = 0.0       # stroke scale 1, no shift
        return z
    def warm(self, S, demand, stored):
        if not self.native(): return super().warm(S, demand, stored)
        sc = float(stored.get("scale", stored.get("amp", 1.0))); sh = float(stored.get("shift", 0.0))
        return np.array(list(np.asarray(stored["x"])) + [2*(np.log(sc) - np.log(SCALE[0]))/(np.log(SCALE[1]) - np.log(SCALE[0])) - 1, sh/SHIFT])
    def describe(self):
        return dict(super().describe(), prior="absorb", params=("knots nT x nk (torque offsets on the gravity hold) + release + stroke scale [0.5,2] (log) + arrival shift [-0.08,0.08] s" if self.native() else "delegated to the task"))
