"""brace_prior(S, contact_point, direction): the static configuration from which the body can apply the LARGEST force at a world
point along a direction under its support contract. Force capacity at a pose q, with the support feet pinned (constrained
Jacobian Jc) and the effector at the point: by virtual work a force f d at the effector costs joint torques f (Jc^T d) on top of
the gravity torques g(q); three caps bound f: (1) torque, f <= min_j (tau_j - |g_j|)/|(Jc^T d)_j|; (2) tipping, the ground reaction
point (ZMP) under gravity + the reaction -f d at p must stay inside the support polygon of the support feet; (3) friction,
f |d_xy| <= mu (M g + f d_z). The pose is improved by finite-difference ascent of f_max over the driven joints, projected onto
the null space of the effector position (the hand stays on the point), feet re-anchored on the base (strategy.ascent_prior
style), joint limits at 95%, certified by the static certificate every 5 steps. Returns a tracker-format reference
stance -> brace over t_ref and the capacity breakdown. A preparation CANDIDATE for sustained-force tasks; not prescribed."""
from __future__ import annotations
from typing import Optional, Sequence
import numpy as np

G = 9.80665


def _anchor_feet(m, S, qn, feet0, fb_ids, iters=3):
    import mujoco
    d = mujoco.MjData(m)
    for _ in range(iters):
        d.qpos[:] = qn; mujoco.mj_forward(m, d); err = []; Jf = []
        for b, (p0, R0) in zip(fb_ids, feet0):
            a = np.zeros((3, m.nv)); r = np.zeros((3, m.nv)); mujoco.mj_jacBody(m, d, a, r, b); Jf += [a[:, :6], r[:, :6]]
            R = d.xmat[b].reshape(3, 3); dR = R0 @ R.T; err += [p0 - d.xpos[b], 0.5*np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])]
        err = np.concatenate(err); Jf = np.vstack(Jf)
        if np.linalg.norm(err) < 1e-6: break
        dv = np.linalg.lstsq(Jf, err, rcond=None)[0]; v = np.zeros(m.nv); v[:6] = dv; mujoco.mj_integratePos(m, qn, v, 1.0)
    return qn


def sagittal_projector(S):
    """Leg motion restricted to SAGITTAL, MIRRORED joints (hip_pitch / knee / ankle_pitch equal left-right; hip yaw/roll and ankle roll
    frozen), as engine.strategy.ascent_prior does: with both feet pinned, an asymmetric leg motion has no base pose that keeps both
    feet planted (6 base dofs, 12 foot constraints) and the least-squares anchor tilts the feet into the floor."""
    import mujoco
    m = S["m"]; J = S["jarr"]; jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.dof_jntid[int(dof)])) or "" for dof in J["dof"]]
    legs = np.array([any(k in n for k in ("hip", "knee", "ankle")) for n in jn]); free = legs & np.array([not any(k in n for k in ("yaw", "roll")) for n in jn])
    mirror = {}
    for i, n in enumerate(jn):
        if free[i] and "left" in n:
            j_ = [k for k, n2 in enumerate(jn) if n2 == n.replace("left", "right")]
            if j_: mirror[i] = j_[0]; mirror[j_[0]] = i
    def proj(dq):
        dq = np.where(legs & ~free, 0.0, np.asarray(dq, float)); out = dq.copy()
        for i, j_ in mirror.items(): out[i] = 0.5*(dq[i] + dq[j_])
        return out
    return proj


def force_capacity(S, q, p_world, dvec, mu=0.9, margin=0.02, cg=None) -> dict:
    """f_max at pose q and the binding cap. Uses throw.constrained_jacobian (support feet pinned) and plant._support_margin."""
    import mujoco, throw as T, plant as P
    m = S["m"]; J = S["jarr"]; d = mujoco.MjData(m); d.qpos[:] = q; d.qvel[:] = 0; mujoco.mj_forward(m, d)
    dv = np.asarray(dvec, float); dv = dv/max(np.linalg.norm(dv), 1e-12); Jc = T.constrained_jacobian(S, q); a = Jc.T @ dv
    g = np.array([float(d.qfrc_bias[int(dof)]) for dof in J["dof"]]); tau = np.asarray(J["tau"], float)
    with np.errstate(divide="ignore"): ft = np.where(np.abs(a) > 1e-6, np.maximum(tau - np.abs(g), 0.0)/np.maximum(np.abs(a), 1e-6), np.inf)
    j_t = int(np.argmin(ft)); f_tau = float(ft[j_t])
    # tipping: ZMP of gravity at the COM plus the reaction A = -f d applied at p (floor at z = 0 after placement)
    M = float(S["M"]); c = d.subtree_com[0].copy(); p = np.asarray(p_world, float)
    sup = S.get("support", (0, 1)); cps = P.potential_contacts(m, d, sorted(set().union(*[S["feet"][i] for i in sup])), 0.003); pts = np.array([c_[:2] for c_ in cps]) if cps else np.zeros((0, 2))   # sole-floor contact points (geom centres span 5 cm only)
    def zmp(f):
        N = M*G + f*dv[2]
        return np.array([(M*G*c[0] - f*(p[2]*dv[0] - p[0]*dv[2]))/N, (M*G*c[1] + f*(p[1]*dv[2] - p[2]*dv[1]))/N])
    def inside(f): return P._support_margin(pts, zmp(f)) >= margin if len(pts) >= 3 else False
    f_tip = 0.0
    if inside(0.0):
        lo, hi = 0.0, max(f_tau, 1.0)*2 + 10.0
        if inside(hi): f_tip = hi
        else:
            for _ in range(30):
                mid = 0.5*(lo + hi); lo, hi = (mid, hi) if inside(mid) else (lo, mid)
            f_tip = lo
    hxy = float(np.hypot(dv[0], dv[1])); f_fr = np.inf if hxy <= mu*dv[2] + 1e-9 else mu*M*G/max(hxy - mu*dv[2], 1e-9)   # from f hxy <= mu (M g + f dz)
    caps = dict(torque=f_tau, tipping=f_tip, friction=float(f_fr)); lim = min(caps, key=caps.get); m0 = P._support_margin(pts, zmp(0.0)) if len(pts) >= 3 else float("nan")
    return dict(f_max=float(caps[lim]), limiting=lim, limiting_joint=(S["types"][j_t] if lim == "torque" else None), caps=caps, torque_use=np.abs(g)/tau, zmp0=zmp(0.0).tolist(), margin0=float(m0), feasible=bool(m0 >= margin))


def brace_prior(S, contact_point: Sequence[float], direction: Sequence[float], iters: int = 60, step: float = 0.02, limit_frac: float = 0.05, t_ref: float = 0.6, horizon: Optional[float] = None, fps: int = 100, nk: int = 10, mu: float = 0.9, margin: float = 0.02, balance_gain: float = 1000.0) -> dict:
    import mujoco, throw as T, plant as P
    m = S["m"]; J = S["jarr"]; qa = np.asarray(J["qad"]); dofs = np.asarray(J["dof"]); q_s = np.array(S["q_settled"]); sid = S["site"]
    jids = [int(m.dof_jntid[int(dof)]) for dof in dofs]; lo = np.array([m.jnt_range[j][0] for j in jids]); hi = np.array([m.jnt_range[j][1] for j in jids]); lo, hi = lo + limit_frac*(hi - lo), hi - limit_frac*(hi - lo)
    sup = S.get("support", (0, 1)); fb_ids = [int(m.geom_bodyid[next(iter(S["feet"][i]))]) for i in sup]
    d0 = mujoco.MjData(m); d0.qpos[:] = q_s; mujoco.mj_forward(m, d0); feet0 = [(d0.xpos[b].copy(), d0.xmat[b].reshape(3, 3).copy()) for b in fb_ids]
    cg = [g for g in P.PlantContractVerifier(m, mujoco.MjData(m), P.PlantContract(S["mjcf"], "torque")).collision_geoms() if g in S.get("robot_geoms", set(range(m.ngeom)))]
    p_t = np.asarray(contact_point, float); dv = np.asarray(direction, float); dv = dv/max(np.linalg.norm(dv), 1e-12)
    def site(q): d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_forward(m, d); return d.site_xpos[sid].copy()
    def step_cfg(q, dq):
        v = np.zeros(m.nv); v[dofs] = dq; qn = q.copy(); mujoco.mj_integratePos(m, qn, v, 1.0); return _anchor_feet(m, S, qn, feet0, fb_ids)
    def feet_centre(q):
        d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_forward(m, d); pts = np.array([d.geom_xpos[g_][:2] for i in sup for g_ in S["feet"][i]]); return pts.mean(0), d.subtree_com[0][:2].copy()
    def place(q, n_it=4, w_com=3.0):
        """Gauss-Newton on the driven joints: effector -> contact point AND whole-body COM xy -> the support feet centre, stacked
        (feet pinned through the constrained Jacobians; engine.strategy.constrained_jacobian for the COM). The body reaches the point
        by bending the legs and counter-moving, not by leaning its COM off the feet (a first-order null-space projection drifted)."""
        from engine import strategy as ST
        for _ in range(n_it):
            e = p_t - site(q); fc, cxy = feet_centre(q); ec = fc - cxy
            if np.linalg.norm(e) < 2e-3 and np.linalg.norm(ec) < 5e-3: break
            Jc = T.constrained_jacobian(S, q); A = ST.constrained_jacobian(S, q, ("com",))[:2]
            M_ = np.vstack([Jc, w_com*A]); b_ = np.concatenate([e, w_com*ec])
            dq = sag(np.linalg.lstsq(M_.T @ M_ + 1e-4*np.eye(len(dofs)), M_.T @ b_, rcond=None)[0]*0.8)
            dq = np.clip(q[qa] + dq, lo, hi) - q[qa]; q = step_cfg(q, dq)
        return q
    sag = sagittal_projector(S)
    def cert(q): return T._cert_ok(P.static_certificate(m, mujoco.MjData(m), q, coll_geoms=cg))
    def score(c_): return c_["f_max"] + float(balance_gain)*min(0.0, (c_["margin0"] if np.isfinite(c_["margin0"]) else -0.1) - margin)   # N; negative while the unloaded pose is outside the support margin
    q = place(q_s.copy(), 12); path = [q.copy()]; cap = force_capacity(S, q, p_t, dv, mu, margin); hist = [cap["f_max"]]; reach_err = float(np.linalg.norm(p_t - site(q)))
    for k in range(iters):
        f0 = score(cap); grad = np.zeros(len(dofs))
        for i in range(len(dofs)):
            qp = q.copy(); qp[qa[i]] = np.clip(qp[qa[i]] + 1e-3, lo[i], hi[i]); grad[i] = (score(force_capacity(S, qp, p_t, dv, mu, margin)) - f0)/1e-3
        Jc = T.constrained_jacobian(S, q); A = Jc @ Jc.T + 1e-6*np.eye(3); grad = sag(grad - Jc.T @ np.linalg.solve(A, Jc @ grad))     # null space of the effector position, sagittal legs
        if np.linalg.norm(grad) < 1e-6: break
        dq = step*grad/np.linalg.norm(grad); dq = np.clip(q[qa] + dq, lo, hi) - q[qa]; qn = place(step_cfg(q, dq))
        cn = force_capacity(S, qn, p_t, dv, mu, margin)
        if score(cn) <= f0 + 1e-6 or (k % 5 == 4 and cn["feasible"] and not cert(qn)): break
        q, cap = qn, cn; path.append(q.copy()); hist.append(cap["f_max"])
    H = float(horizon or T.HOR); n = int(H*fps) + 1; t = np.arange(n)/fps; Q = np.repeat(q_s[None], n, 0); P_ = np.array(path)
    seg = np.linalg.norm(np.diff(P_, axis=0), axis=1) if len(P_) > 1 else np.zeros(0); cs = np.concatenate([[0], np.cumsum(seg)])
    for i, ti in enumerate(t):
        w = float(np.clip((ti - 0.05)/max(t_ref - 0.05, 1e-6), 0.0, 1.0))
        if len(P_) == 1: Q[i] = P_[0]; continue
        L = w*cs[-1]; j = min(int(np.searchsorted(cs, L, side="right")) - 1, len(seg) - 1); f = (L - cs[j])/max(seg[j], 1e-12); Q[i] = P_[j] + f*(P_[j+1] - P_[j]); Q[i, 3:7] /= np.linalg.norm(Q[i, 3:7])
    return dict(key="brace_prior", start=q_s, Q=Q, fps=fps, takeoff_ref=float(t_ref), horizon=H, nk=nk, brace=q.tolist(), f_max=float(cap["f_max"]), limiting=cap["limiting"], limiting_joint=cap["limiting_joint"], caps=cap["caps"], f_history=hist, steps=len(path) - 1, feasible=cap["feasible"], margin0=cap["margin0"], reach_err=float(np.linalg.norm(p_t - site(q))), reach_err_start=reach_err, contact_point=p_t.tolist(), direction=dv.tolist())
