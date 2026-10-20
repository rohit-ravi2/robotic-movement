"""J4: ONE strategy generator for every task whose goal is a velocity of some body point (COM for a jump, palm for a
throw, hand for a reach): capacity-weighted ascent of the task velocity in joint space, with both feet held still.
  u(q) = c * (J_c(q)^T d),   c_j = tau_j * omega_j / 4   (peak actuator power),   J_c = constrained Jacobian (feet fixed)
WIND-UP (crouch / cock): integrate -u from the stance while the pose stays statically certifiable;
RELEASE (extend / whip): integrate +u from the stance for the same excursion budget while progress along d continues.
This reproduces the throw wind-up prior (src/throw.py windup_reference) when effector = palm, and generates the jump's
deep-crouch prior from the SAME rule when effector = COM, d = +z (no pose bank). Reference = start -> wind-up -> release."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco, plant as P


def constrained_jacobian(S, q, effector, return_base=False):
    """d(effector velocity)/d(driven joint velocity) with both feet stationary (floating base solved by least squares).
    effector: ("site", id) | ("com",) | ("body", id)."""
    m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_forward(m, d)
    jp = np.zeros((3, m.nv))
    if effector[0] == "site": mujoco.mj_jacSite(m, d, jp, None, effector[1])
    elif effector[0] == "com": mujoco.mj_jacSubtreeCom(m, d, jp, 0)
    else: mujoco.mj_jacBody(m, d, jp, None, effector[1])
    Jf = []
    for fb in S.get("foot_bodies_ids") or _foot_bodies(S):
        a = np.zeros((3, m.nv)); r = np.zeros((3, m.nv)); mujoco.mj_jacBody(m, d, a, r, fb); Jf += [a, r]
    Jf = np.vstack(Jf); dofs = S["jarr"]["dof"]; base = np.arange(6); out = np.zeros((3, len(dofs))); Bv = np.zeros((6, len(dofs)))
    for i, dof in enumerate(dofs):
        vb = -np.linalg.lstsq(Jf[:, base], Jf[:, dof], rcond=None)[0]
        out[:, i] = jp[:, base] @ vb + jp[:, dof]; Bv[:, i] = vb
    if return_base: return out, Bv
    return out


def _foot_bodies(S):
    import branch_i as BI
    m = S["m"]; ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in BI.ROBOTS[S.get("robot", "G1")]["foot_bodies"]]
    S["foot_bodies_ids"] = ids; return ids


def ascent_prior(S, effector, dvec, horizon, nk, t_w=0.40, t_ref=0.55, step=0.02, n_max=800, move_legs_in_windup=True,
                 leg_cap=None, limit_frac=0.05, fps=100, key="ascent_prior", hold_com_xy=False, integrate_base=True, n_wind=None, stop_fn=None, support_margin=0.0, trunk_cap=None, sagittal_legs=False, cap_ref=None):
    """hold_com_xy: project the direction so the whole-body COM does not move horizontally to first order (balance
    kept inside the direction field; needed when the legs move, i.e. a crouch). integrate_base: move the floating base
    with the feet-fixed solution (reference feet stay planted); False reproduces the Phase 11 throw prior exactly."""
    """Generic wind-up -> release reference. move_legs_in_windup: True for a jump (the crouch IS leg motion), False for a
    standing throw (legs stay in the certified stance during the cock, as in Phase 11). leg_cap: optional +- cap (rad) on
    leg excursion in the release path (throw used 0.30)."""
    # ACTUATOR space (one entry per driven actuator; bilateral tasks have 2 actuators per type, per-joint tasks 1)
    m = S["m"]; J = S["jarr"]; q_s = np.array(S["q_settled"]); qa = np.asarray(J["qad"])
    q_c = np.asarray(cap_ref) if cap_ref is not None else q_s      # leg/trunk caps are measured from THIS pose (chained segments: the original stance, else caps accumulate segment by segment)
    jids = [int(m.dof_jntid[int(dof)]) for dof in J["dof"]]; jnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in jids]
    lo = np.array([m.jnt_range[j][0] for j in jids]); hi = np.array([m.jnt_range[j][1] for j in jids])
    lo, hi = lo + limit_frac*(hi - lo), hi - limit_frac*(hi - lo)
    legs = np.array([any(k in n for k in ("hip", "knee", "ankle")) for n in jnames])
    # SAGITTAL SQUAT structure for a leg motion that the balance controller must track: hip yaw/roll and ankle roll stay at
    # the stance (a twisted or asymmetric leg reference turned the pelvis 7 deg and shifted the COM 7 cm on the wipe squat);
    # the pitch joints (hip pitch, knee, ankle pitch) move MIRRORED left/right
    leg_free = np.array([legs[i] and not any(k in n for k in ("yaw", "roll")) for i, n in enumerate(jnames)])
    mirror = {}
    for i, n in enumerate(jnames):
        if leg_free[i] and "left" in n:
            j_ = [k for k, n2 in enumerate(jnames) if n2 == n.replace("left", "right")]
            if j_: mirror[i] = j_[0]; mirror[j_[0]] = i
    def sagittal(dq):
        dq = np.where(legs & ~leg_free, 0.0, dq); out = dq.copy()
        for i, j_ in mirror.items(): out[i] = 0.5*(dq[i] + dq[j_])
        return out
    trunk = np.array([(("waist" in n or n == "torso") and "pitch" not in n) for n in jnames]); trunk_pitch = np.array([("waist" in n and "pitch" in n) for n in jnames])
    c = np.asarray(J["tau"], float)*np.asarray(J["om"], float)/4.0
    dfun = dvec if callable(dvec) else (lambda q, _d=np.asarray(dvec, float)/np.linalg.norm(dvec): _d)   # fixed direction or d(q) (e.g. toward a target)
    cg = P.PlantContractVerifier(m, mujoco.MjData(m), P.PlantContract(S["mjcf"], "torque")).collision_geoms()
    dofs = np.asarray(J["dof"])
    def direction(q):
        Jc, Bv = constrained_jacobian(S, q, effector, return_base=True); dvec = dfun(q)
        g = Jc.T @ dvec
        if hold_com_xy:
            Jcom = constrained_jacobian(S, q, ("com",)); A = Jcom[:2]                 # rows: COM x, y
            W = np.diag(c); AW = A @ W @ A.T
            g = g - A.T @ np.linalg.solve(AW + 1e-9*np.eye(2), A @ W @ g)      # c-weighted projection onto {A dq = 0}
        u = c*g; u = u/max(np.linalg.norm(u), 1e-12); return u, Bv
    fb_ids = S.get("foot_bodies_ids") or _foot_bodies(S)
    d0 = mujoco.MjData(m); d0.qpos[:] = q_s; mujoco.mj_forward(m, d0)
    feet0 = [(d0.xpos[b].copy(), d0.xmat[b].reshape(3, 3).copy()) for b in fb_ids]         # the planted feet: world pose at the start
    def anchor_feet(qn, iters=3):
        """Gauss-Newton on the 6 base DOFs so both feet keep their initial world pose (first-order base integration
        drifts; a tilted foot makes the static certificate CONTACT_INFEASIBLE and truncates the path)."""
        d = mujoco.MjData(m)
        for _ in range(iters):
            d.qpos[:] = qn; mujoco.mj_forward(m, d); err = []; Jf = []
            for b, (p0, R0) in zip(fb_ids, feet0):
                a = np.zeros((3, m.nv)); r = np.zeros((3, m.nv)); mujoco.mj_jacBody(m, d, a, r, b); Jf += [a[:, :6], r[:, :6]]
                R = d.xmat[b].reshape(3, 3); dR = R0 @ R.T; rv = 0.5*np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
                err += [p0 - d.xpos[b], rv]
            err = np.concatenate(err); Jf = np.vstack(Jf)
            if np.linalg.norm(err) < 1e-6: break
            dv = np.linalg.lstsq(Jf, err, rcond=None)[0]; v = np.zeros(m.nv); v[:6] = dv; mujoco.mj_integratePos(m, qn, v, 1.0)
        return qn
    def step_cfg(q, dq):
        """Integrate joints by dq and the floating base by the feet-fixed base velocity (quaternion-correct), then
        re-anchor the feet exactly."""
        v = np.zeros(m.nv); v[dofs] = dq
        if integrate_base:
            _, Bv = constrained_jacobian(S, q, effector, return_base=True); v[:6] = Bv @ dq
        qn = q.copy(); mujoco.mj_integratePos(m, qn, v, 1.0)
        return anchor_feet(qn) if integrate_base else qn
    def certified(q):
        c_ = P.static_certificate(m, mujoco.MjData(m), q, coll_geoms=cg)
        return c_["status"] == "STATIC_FEASIBLE" and float(c_.get("support_margin", 1.0) or 0.0) >= support_margin
    q = q_s.copy(); wind = [q.copy()]
    for k in range(n_max if n_wind is None else n_wind):
        u, _ = direction(q); dq = -step*u
        if not move_legs_in_windup: dq = np.where(legs, 0.0, dq)
        dq = np.clip(q[qa] + dq, lo, hi) - q[qa]
        if np.linalg.norm(dq) < 1e-4: break
        qn = step_cfg(q, dq)
        if k % 5 == 4 and not certified(qn): break
        q = qn; wind.append(q.copy())
    qw = q.copy(); q = q_s.copy(); fwd = [q.copy()]; prog = 0.0
    for k in range(len(wind) - 1 if n_wind is None else n_max):
        u, _ = direction(q); dq = step*u
        if leg_cap is not None:
            if leg_cap > 0 and sagittal_legs: dq = sagittal(dq)
            dq = np.where(legs, np.clip(q[qa] + dq, np.minimum(q_c[qa] - leg_cap, q[qa]), np.maximum(q_c[qa] + leg_cap, q[qa])) - q[qa], dq)
        if trunk_cap is not None:   # yaw/roll capped tightly (lateral hopping otherwise); forward pitch allowed up to 0.35 rad (bending over a counter is natural)
            # a cap FORBIDS moving further out of the band; it never forces a return into it (a handed-off pose already outside
            # the band, e.g. the trunk bent 0.3 by a pick, snapped back on the first frame and threw the held tool: J27)
            dq = np.where(trunk, np.clip(q[qa] + dq, np.minimum(q_c[qa] - trunk_cap, q[qa]), np.maximum(q_c[qa] + trunk_cap, q[qa])) - q[qa], dq)
            dq = np.where(trunk_pitch, np.clip(q[qa] + dq, np.minimum(q_c[qa] - 0.35, q[qa]), np.maximum(q_c[qa] + 0.35, q[qa])) - q[qa], dq)
        dq = np.clip(q[qa] + dq, lo, hi) - q[qa]
        gain = float(dfun(q) @ (constrained_jacobian(S, q, effector) @ dq))
        if gain <= 1e-5 or (stop_fn is not None and stop_fn(q)): break
        qn = step_cfg(q, dq)
        if support_margin > 0 and k % 5 == 4 and not certified(qn): break      # a reach path that leaves the support margin stops short
        prog += gain; q = qn; fwd.append(q.copy())
    wind = np.array(wind); path = np.vstack([wind[::-1], np.array(fwd)[1:]]) if len(fwd) > 1 else wind[::-1]   # FULL configurations
    n = int(horizon*fps) + 1; t = np.arange(n)/fps; Q = np.repeat(q_s[None], n, 0)
    def along(P_, w):
        if len(P_) == 1: return P_[0]
        seg = np.linalg.norm(np.diff(P_, axis=0), axis=1); cs = np.concatenate([[0], np.cumsum(seg)]); L = w*cs[-1]
        i = min(int(np.searchsorted(cs, L, side="right")) - 1, len(seg) - 1); f = (L - cs[i])/max(seg[i], 1e-12)
        return P_[i] + f*(P_[i+1] - P_[i])
    for i, ti in enumerate(t):
        if ti <= 0.05: qq = q_s
        elif ti <= t_w: qq = along(wind, (ti - 0.05)/(t_w - 0.05))
        elif ti <= t_ref: qq = along(path, (ti - t_w)/(t_ref - t_w))
        else: qq = path[-1]
        Q[i] = qq; Q[i, 3:7] /= np.linalg.norm(Q[i, 3:7])
    return dict(key=key, start=q_s, Q=Q, fps=fps, takeoff_ref=t_ref, horizon=horizon, nk=nk, windup=qw[qa].tolist(), release=path[-1][qa].tolist(),
                windup_steps=len(wind) - 1, path_steps=len(path) - 1, progress=prog)


def jump_prior(S, horizon=1.1, nk=22, t_ref=0.8):
    """Deep-crouch prior from the ascent rule: effector = whole-body COM, d = +z, legs move in the wind-up (crouch)."""
    return ascent_prior(S, ("com",), [0, 0, 1], horizon, nk, t_w=max(0.1, t_ref - 0.25), t_ref=t_ref, move_legs_in_windup=True, key="jump_ascent_prior", hold_com_xy=True)


def throw_prior(S, elev_mid, horizon, nk, t_w=0.40, t_ref=0.55, integrate_base=True):
    """The Phase 11 wind-up prior through the same rule: effector = palm site, d = cone centre, legs fixed in the cock."""
    return ascent_prior(S, ("site", S["site"]), [np.cos(elev_mid), 0.0, np.sin(elev_mid)], horizon, nk, t_w=t_w, t_ref=t_ref,
                        move_legs_in_windup=False, leg_cap=0.30, key="throw_ascent_prior", integrate_base=integrate_base)


def reach_prior(S, target_world, horizon=1.2, nk=10, t_ref=0.6, tol=0.02, trunk_cap=None, leg_cap=0.0, squat_if_short=0.0, cap_ref=None):
    """Reach prior from the same rule: effector = palm, d(q) = unit(target - palm(q)), no wind-up, balance kept
    (COM xy projection), stop when the palm is within tol of the target. leg_cap > 0 lets the legs SQUAT (+- cap rad about
    the stance) along the same ascent direction; squat_if_short > 0 = run arm+trunk first and redo with legs (cap =
    squat_if_short) when the palm ends more than 2*tol from the target (a low surface is out of the arm's reach with the
    COM held over the feet: diagnosed on the wipe at 0.55 m, 4 cm short with no joint at a limit)."""
    m = S["m"]; sid = S["site"]
    def palm(q):
        d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_forward(m, d); return d.site_xpos[sid].copy()
    def dfun(q):
        v = np.asarray(target_world) - palm(q); return v/max(np.linalg.norm(v), 1e-9)
    def stop(q): return np.linalg.norm(np.asarray(target_world) - palm(q)) <= tol
    def run(cap):
        # legs are owned by the balance controller in the quasi-static reach; with cap 0 the reference moves ARM + TRUNK only
        r = ascent_prior(S, ("site", sid), dfun, horizon, nk, t_w=0.05, t_ref=t_ref, move_legs_in_windup=cap > 0, leg_cap=float(cap), key="reach_ascent_prior",
                         hold_com_xy=True, n_wind=0, stop_fn=stop, limit_frac=0.05, support_margin=0.0, trunk_cap=trunk_cap, sagittal_legs=True, cap_ref=cap_ref)   # path check off: an intermediate-pose certificate truncated valid paths; the task gate judges the end state
        qf = r["Q"][min(int(t_ref*r["fps"]), len(r["Q"]) - 1)]; r["miss"] = float(np.linalg.norm(np.asarray(target_world) - palm(qf))); r["legs"] = cap > 0
        return r
    r = run(leg_cap)
    if leg_cap == 0 and squat_if_short > 0 and r["miss"] > 2*tol:
        r2 = run(squat_if_short)
        if r2["miss"] < r["miss"]: r = r2
    return r