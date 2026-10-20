"""DYNAMIC LEVERAGE PREDICTOR for push / pull of a sliding load (Track B, 2026-10-17).

The static brace predictor (priors/brace.py) asked: from an UNLOADED-stable pose (COM inside the feet), how much force can the hand
apply before the ground reaction point leaves the feet? For G1 that is 42 N (tipping), so it said a 20 kg sled at mu 0.4 (82 N) is
infeasible; the search then pulled 20 kg 0.35 m. The reason: once the hand is attached to the load, the body may lean its COM OUT of
its own feet and hang on the load; the ground reaction point stays in the feet because the load's reaction closes the moment:

    M g (x_c - x_p) - F_x h + (vertical hand force) * (x_hand - x_p) = M z_c x_ddot_c + dL/dt      (sagittal, about the CoP x_p)

so the sustainable force is F_x ~ M (g (x_c - x_p) - z_c x_ddot_c) / h: a LEAN term (COM ahead of the CoP) and an ACCELERATION
term (decelerating COM). This module computes, from the robot spec only (no rollout):

1. LOADED-EQUILIBRIUM CAPACITY at a pose (an LP, the L0 formulation extended): all actuator torques |tau| <= tau_max (every joint,
   wrist included, through J^T), joints resting on a limit may take a one-sided limit torque (a locked elbow), foot contacts with
   unilateral normals and an 8-facet inner friction pyramid at the MODEL's foot friction, and the hand-sled wrench as variables. The
   sled side is linear in (wrench, sled mass m_s): it slides when s F_x + mu F_z_up >= mu g m_s (lifting the handle unloads the
   sled's friction), stays on the floor (F_z_up <= m_s g) and does not tip (|pitch moment| <= half-length * N, |roll| <=
   half-width * N, |yaw| <= mu N r). The LP maximises m_s. Two interfaces: WELD (6-D wrench, the simulator's abstraction) and
   POINT (force only at the palm, i.e. what a flat palm transmits: this is where the 5 N m wrist binds).
2. STROKE: the sled must move >= 0.2 m with the feet planted, so the hand travels 0.2 m. Poses along the stroke come from a
   generated family (rigid ankle lean phi, waist pitch, symmetric crouch; right-arm IK to put the palm on the moved handle).
   F_drive(s) = max over the family of the LP's drive force at that hand displacement.
3. LEAN-AND-DRIVE (dynamics): the sled moves through the stroke iff the work-energy of the coupled system stays positive,
   E(s) = int_0^s (F_drive - mu g m_s) ds' > 0 (the robot may carry momentum through a weak stretch), and the stroke time
   int ds / v(s), v = sqrt(2 E / M_eff), M_eff = m_s + kappa^2 M (kappa = robot COM travel per hand travel along the best path)
   must be <= the horizon. The predicted maximum mass is the largest m_s meeting both; the predicted lean is the COM-minus-feet-centre
   along the best path.
Also reported: the closed-form ZMP number M g (x_c - x_p)/h at the best pose (x_p at the sole edge), and the binding constraint
(joint name / friction / sole edge / sled tipping).

Labels: hand-sled attachment = WELD (simulator abstraction) for the main number; POINT = palm force only. Simulator only.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
from scipy.optimize import linprog
import plant as P
from engine import leverage as LV

G = 9.80665
ARM_R = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
         "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]


def _jid(m, n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)


def crouch_pose(m, q0, c):
    """Symmetric sagittal crouch: hip pitch -c, knee +2c, ankle pitch -c (foot stays parallel to the pelvis), base lowered/shifted so
    the mean foot site returns to where it was."""
    d = mujoco.MjData(m); d.qpos[:] = q0; mujoco.mj_forward(m, d)
    fs = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in ("left_foot", "right_foot")]; f0 = np.mean([d.site_xpos[s] for s in fs], 0)
    q = np.array(q0, float)
    for s in ("left", "right"):
        for jn, k in (("hip_pitch_joint", -1.0), ("knee_joint", 2.0), ("ankle_pitch_joint", -1.0)):
            q[m.jnt_qposadr[_jid(m, f"{s}_{jn}")]] += k*c
    d.qpos[:] = q; mujoco.mj_forward(m, d); q[:3] += f0 - np.mean([d.site_xpos[s] for s in fs], 0)
    return q


def lean_pose(m, q0, phi):
    """Rigid lean about the ankle-pitch axis midpoint, ankles counter-rotated (soles stay flat); + = top towards +x."""
    d = mujoco.MjData(m); d.qpos[:] = q0; mujoco.mj_forward(m, d); q = np.array(q0)
    js = [_jid(m, f"{s}_ankle_pitch_joint") for s in ("left", "right")]; A = 0.5*(d.xanchor[js[0]] + d.xanchor[js[1]])
    Ry = np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]]); q[:3] = A + Ry @ (np.asarray(q0[:3]) - A)
    qr = np.zeros(4); mujoco.mju_mat2Quat(qr, Ry.flatten()); qn = np.zeros(4); mujoco.mju_mulQuat(qn, qr, np.asarray(q0[3:7])); q[3:7] = qn
    for j in js: q[m.jnt_qposadr[j]] -= phi
    return q


def in_range(m, q, names, tol=1e-6):
    for n in names:
        j = _jid(m, n)
        if m.jnt_limited[j] and not (m.jnt_range[j][0] - tol <= q[m.jnt_qposadr[j]] <= m.jnt_range[j][1] + tol): return False
    return True


def capacity(m, q, hand_body, p_hand, sgn, mu_sled, robot_bodies, foot_geoms, sled_off=(0.5, 0.0), sled_half=(0.3, 0.15),
             interface="weld", m_fixed=None, lim_tol=0.02, tol=0.003, foot_mu_scale=1.0, allow_lift=True):
    """LP at pose q. sgn = +1 push (+x), -1 pull. Variables (scaled by robot weight W): u (nu, tau/tau_max), f (3 nc), l (limit
    torques, one-sided), w = [Fx, Fz, Mx, My, Mz] (the hand's wrench ON THE SLED at p_hand, Fy = 0), and m_s*g/W.
    m_fixed=None: maximise m_s. m_fixed=value: maximise the drive s Fx + mu Fz at that sled mass (F_drive)."""
    d = mujoco.MjData(m); d.qpos[:] = q; d.qvel[:] = 0; mujoco.mj_forward(m, d)
    rd = np.array([i for i in range(m.nv) if int(m.dof_bodyid[i]) in robot_bodies]); nu = m.nu
    W = float(sum(m.body_mass[b] for b in robot_bodies))*G
    cp = P.potential_contacts(m, d, sorted(foot_geoms), tol, with_geom=True); pts = np.array([c[:3] for c in cp], float); nc = len(pts)
    if nc < 3: return dict(status="NO_SUPPORT")
    cbody = [int(m.geom_bodyid[int(c[3])]) for c in cp]
    tm = LV.tau_max(m); names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a][0])) for a in range(nu)]
    # joints at a limit: one-sided limit torque available
    lims = []
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or not m.jnt_limited[j] or int(m.jnt_bodyid[j]) not in robot_bodies: continue
        qv = q[m.jnt_qposadr[j]]; lo, hi = m.jnt_range[j]
        if qv - lo < lim_tol: lims.append((int(m.jnt_dofadr[j]), +1.0, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)))
        elif hi - qv < lim_tol: lims.append((int(m.jnt_dofadr[j]), -1.0, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)))
    nl = len(lims)
    iu, iF, iL, iW, iM = 0, nu, nu + 3*nc, nu + 3*nc + nl, nu + 3*nc + nl + 5; n = iM + 1
    Aeq = np.zeros((m.nv, n)); b = np.array(d.qfrc_bias, float)
    for a in range(nu): Aeq[int(m.jnt_dofadr[int(m.actuator_trnid[a][0])]), a] = float(m.actuator_gear[a][0])*tm[a]
    jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    for i in range(nc):
        mujoco.mj_jac(m, d, jp, jr, pts[i], cbody[i]); Aeq[:, iF+3*i:iF+3*i+3] = jp.T*W        # world-frame force (fx, fy, fz)
    for k, (dof, sg, _) in enumerate(lims): Aeq[dof, iL+k] = sg*W
    mujoco.mj_jac(m, d, jp, jr, p_hand, hand_body)
    # the sled pushes back on the robot with -wrench: columns for (Fx, Fz, Mx, My, Mz)
    Aeq[:, iW+0] = -jp[0]*W; Aeq[:, iW+1] = -jp[2]*W; Aeq[:, iW+2] = -jr[0]*W; Aeq[:, iW+3] = -jr[1]*W; Aeq[:, iW+4] = -jr[2]*W
    Aeq, beq = Aeq[rd]/W, b[rd]/W
    rows, rhs = [], []
    def le(coef, r=0.0):
        v = np.zeros(n)
        for k_, c_ in coef: v[k_] += c_
        rows.append(v); rhs.append(r)
    mf = 0.0; k8 = np.cos(np.pi/8)
    for g in cp: mf = max(mf, float(m.geom_friction[int(g[3])][0]))
    mu_f = mf*k8*foot_mu_scale
    for i in range(nc):
        o = iF + 3*i
        for ang in np.arange(8)*np.pi/4: le([(o, np.cos(ang)), (o+1, np.sin(ang)), (o+2, -mu_f)])
    # sled: N = m_s g - Fz ;   s Fx + mu Fz >= mu m_s g ; Fz <= m_s g ; moments about the sled floor centre c
    s = float(sgn); c = np.array([p_hand[0] + sled_off[0], p_hand[1] + sled_off[1], 0.0]); r = p_hand - c
    # M_c = Mo + r x F with F = (Fx, 0, Fz): (r x F) = (r_y Fz, r_z Fx - r_x Fz, -r_y Fx)
    Mc = {0: [(iW+2, 1.0), (iW+1, r[1])], 1: [(iW+3, 1.0), (iW+0, r[2]), (iW+1, -r[0])], 2: [(iW+4, 1.0), (iW+0, -r[1])]}
    caps = {0: sled_half[1], 1: sled_half[0], 2: mu_sled*0.1}
    for ax in (0, 1, 2):
        for sg in (1, -1): le([(k_, sg*c_) for k_, c_ in Mc[ax]] + [(iM, -caps[ax]), (iW+1, caps[ax])])     # |M| <= cap (m_s g - Fz)
    le([(iW+1, 1.0), (iM, -1.0)])                                                                       # Fz <= m_s g
    if m_fixed is None: le([(iW+0, -s), (iW+1, -mu_sled), (iM, mu_sled)])                               # mu m_s g <= s Fx + mu Fz
    bnds = [(-1, 1) if np.isfinite(tm[a]) else (None, None) for a in range(nu)]
    bnds += [(0, None) if j % 3 == 2 else (None, None) for j in range(3*nc)] + [(0, None)]*nl
    bnds += [(None, None), (None, None) if allow_lift else (None, 0.0)] + ([(None, None)]*3 if interface == "weld" else [(0, 0)]*3)
    bnds += [(0, None)] if m_fixed is None else [(m_fixed*G/W, m_fixed*G/W)]
    cobj = np.zeros(n)
    if m_fixed is None: cobj[iM] = -1.0
    else: cobj[iW+0] = -s; cobj[iW+1] = -mu_sled
    res = linprog(cobj, A_ub=np.array(rows), b_ub=np.array(rhs), A_eq=Aeq, b_eq=beq, bounds=bnds, method="highs")
    if not res.success: return dict(status="INFEASIBLE", detail=str(res.message)[:80])
    x = res.x; u = x[:nu]; f = x[iF:iL].reshape(nc, 3)*W; w = x[iW:iM]*W; ms = x[iM]*W/G
    # binding: duals of the bounds -> which joint torques sit at +-1 and matter
    marg = np.abs(getattr(res, "lower").marginals[:nu]) + np.abs(getattr(res, "upper").marginals[:nu])
    bind_j = [names[a] for a in np.argsort(-marg)[:4] if marg[a] > 1e-9]
    fz = f[:, 2].sum(); cop = (f[:, 2] @ pts[:, :2])/max(fz, 1e-9); ft = np.hypot(f[:, 0], f[:, 1]); fric_use = float(ft.sum()/max(mf*fz, 1e-9))   # vs the model's mu (the LP uses the inner 8-facet pyramid)
    com = d.subtree_com[int(min(robot_bodies))].copy() if 1 in robot_bodies else d.subtree_com[1].copy()
    feet = np.array([d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, nm_)] for nm_ in ("left_foot", "right_foot")]).mean(0)
    xs = pts[:, 0]
    # closed-form sagittal ZMP number at this pose: F = M g (x_c - x_p_edge)/h, the lean term alone (no vertical hand force, no accel)
    F_zmp = float(W*(com[0] - (xs.min() if s > 0 else xs.max()))/max(p_hand[2], 1e-3))*s
    lim_used = [lims[k][2] for k in range(nl) if x[iL+k]*W > 1e-3]
    return dict(status="OK", m_s=float(ms), drive=float(s*w[0] + mu_sled*w[1]), Fx=float(w[0]), Fz_up=float(w[1]), M=w[2:].tolist(),
                max_util=float(np.abs(u).max()), worst=names[int(np.argmax(np.abs(u)))], binding_joints=bind_j,
                util_top={names[a]: round(float(abs(u[a])), 3) for a in np.argsort(-np.abs(u))[:5]},
                cop_x=float(cop[0]), sole_x=(float(xs.min()), float(xs.max())), fric_use=fric_use, mu_foot=mf,
                lean=float(com[0] - feet[0]), com=com.tolist(), hand=np.asarray(p_hand).tolist(), F_zmp_closed_form=F_zmp, limit_torques_used=lim_used)


def family(S, s_disp, sgn, phis, waists, crouches, ik_tol=0.01):
    """Poses with the palm at the handle displaced by s_disp along sgn x. Returns [(params, q, palm)] (IK-reachable, in range)."""
    m = S["m"]; q0 = np.array(S["q_settled0"] if "q_settled0" in S else S["q_settled"]); d = mujoco.MjData(m); d.qpos[:] = q0; mujoco.mj_forward(m, d)
    sid = S["site"]; hb = S["hand_body"]; p0 = d.site_xpos[sid].copy(); lp = d.xmat[hb].reshape(3, 3).T @ (p0 - d.xpos[hb])
    tgt = p0 + np.array([sgn*s_disp, 0, 0]); out = []
    wj = _jid(m, "waist_pitch_joint"); legs = [f"{s_}_{j}" for s_ in ("left", "right") for j in ("hip_pitch_joint", "knee_joint", "ankle_pitch_joint")]
    for c in crouches:
        qc = crouch_pose(m, q0, c)
        for w in waists:
            qw = qc.copy(); qw[m.jnt_qposadr[wj]] = w
            for ph in phis:
                ql = lean_pose(m, qw, np.radians(ph))
                if not in_range(m, ql, legs + ["waist_pitch_joint"]): continue
                qa, err = LV.site_ik(m, ql, [(hb, lp, tgt)], ARM_R, iters=150)
                if err > ik_tol: continue
                out.append(((float(ph), float(w), float(c)), qa, tgt.copy()))
    return out


def predict(S, sgn, mu_sled=0.4, stroke=0.2, horizon=4.0, ds=0.025, phis=np.arange(-30, 30.1, 5.0), waists=(-0.3, 0.0, 0.3, 0.5),
            crouches=(0.0, 0.25, 0.5), interface="weld", verbose=False, **cap_kw):
    m = S["m"]; rb = S["robot_bodies"]; fg = S["foot_geoms"]; M = float(sum(m.body_mass[b] for b in rb))
    svals = np.round(np.arange(0.0, stroke + 1e-9, ds), 4); per_s = []
    for sd in svals:
        rows = []
        for prm, q, tgt in family(S, sd, sgn, phis, waists, crouches):
            r = capacity(m, q, S["hand_body"], tgt, sgn, mu_sled, rb, fg, interface=interface, **cap_kw)
            if r["status"] == "OK": rows.append((prm, q, r))
        rows.sort(key=lambda t: -t[2]["m_s"]); per_s.append((sd, rows))
        if verbose: print("s", sd, "poses", len(rows), "best m_s", rows[0][2]["m_s"] if rows else None, rows[0][0] if rows else None, flush=True)
    if not all(r for _, r in per_s): return dict(status="STROKE_UNREACHABLE", reachable=[float(sd) for sd, r in per_s if r])
    m_qs = min(r[0][2]["m_s"] for _, r in per_s)                         # quasi-static: every point of the stroke sustains sliding
    d0 = mujoco.MjData(m); d0.qpos[:] = S["q_settled0"] if "q_settled0" in S else S["q_settled"]; mujoco.mj_forward(m, d0); zc0 = float(d0.subtree_com[1][2])
    def drive_curve(ms, top=8):
        F = []; best = []
        for sd, rows in per_s:
            bb = None
            for prm, q, r0 in rows[:top]:
                r = capacity(m, q, S["hand_body"], np.array(r0["hand"]), sgn, mu_sled, rb, fg, interface=interface, m_fixed=ms, **cap_kw)
                if r["status"] == "OK" and (bb is None or r["drive"] > bb[1]["drive"]): bb = (prm, r)
            F.append(bb[1]["drive"] if bb else -np.inf); best.append(bb)
        return np.array(F), best
    def feasible(ms, lunge=False):
        """lunge=False: the sled must break away quasi-statically at s=0 (drive > friction), then momentum may carry it through weak
        stretches. lunge=True (UPPER BOUND): the body may first fall from the settled stance into the start pose with the hand fixed and
        deliver that whole potential-energy drop M g (z_c0 - z_c) to the sled, lossless."""
        F, best = drive_curve(ms); ex = F - mu_sled*G*ms
        if not np.isfinite(ex).all(): return False, dict(why="no statically admissible pose at some s for this mass")
        E0 = M*G*max(0.0, zc0 - best[0][1]["com"][2]) if lunge else 0.0
        if ex[0] <= 0 and E0 <= 0: return False, dict(why="cannot break away: drive < friction at s=0", ex=ex.tolist())
        E = E0 + np.concatenate([[0.0], np.cumsum(0.5*(ex[1:] + ex[:-1])*np.diff(svals))])
        if (E[1:] <= 0).any(): return False, dict(why="momentum exhausted inside the stroke", ex=ex.tolist(), E=E.tolist())
        coms = np.array([b[1]["com"][0] for b in best]); kappa = abs(coms[-1] - coms[0])/max(stroke, 1e-9); Meff = ms + kappa**2*M
        v = np.sqrt(np.maximum(2*E/Meff, 0))
        if E0 > 0: t = float(np.sum(np.diff(svals)/np.maximum(0.5*(v[1:] + v[:-1]), 1e-9)))
        else: v[0] = 0.0; t = np.sqrt(2*svals[1]/max(ex[0]/Meff, 1e-9)) + float(np.sum(np.diff(svals[1:])/np.maximum(0.5*(v[1:-1] + v[2:]), 1e-9)))
        return t <= horizon, dict(why=None if t <= horizon else f"too slow: {t:.1f} s", t_stroke=float(t), E0_lunge=float(E0), kappa=float(kappa), M_eff=float(Meff), ex=ex.tolist(), E=E.tolist(),
                                  lean=[b[1]["lean"] for b in best], pose=[b[0] for b in best], binding=[b[1]["binding_joints"][:2] for b in best],
                                  worst=[b[1]["worst"] for b in best], cop=[b[1]["cop_x"] for b in best], Fz_up=[b[1]["Fz_up"] for b in best], Fx=[b[1]["Fx"] for b in best], fric_use=[b[1]["fric_use"] for b in best])
    # feasibility is NOT monotone in mass at the low end (a leaned pose needs the load's support), so scan a grid
    grid = np.arange(1.0, max(2.0*m_qs, 20.0) + 1e-9, 0.5)
    def scan(lunge):
        best = None
        for ms in grid:
            ok, det = feasible(ms, lunge)
            if ok: best = (float(ms), det)
        return best
    main = scan(False); bound = scan(True)
    if main is None: return dict(status="NOTHING_MOVES", m_qs=m_qs, lunge_bound=bound and bound[0])
    lo, det = main; _, det_hi = feasible(lo + 0.5)
    s0 = per_s[0][1][0]; s_end = per_s[-1][1][0]
    return dict(status="OK", interface=interface, M_robot=M, mu_sled=mu_sled, stroke=stroke, horizon=horizon, m_max_total=float(lo), payload_max=float(lo - 1.0),
                m_quasistatic_total=float(m_qs), at_max=det, why_heavier_fails=det_hi and det_hi.get("why"), m_max_total_lunge_bound=bound and bound[0], payload_max_lunge_bound=bound and bound[0] - 1.0, lunge_at_max=bound and bound[1],
                lean_pred_mean=float(np.mean(det["lean"])), lean_pred_start=float(det["lean"][0]), lean_pred_end=float(det["lean"][-1]),
                best_static_start=dict(pose=s0[0], **{k: s0[2][k] for k in ("m_s", "lean", "worst", "binding_joints", "util_top", "cop_x", "sole_x", "Fz_up", "Fx", "F_zmp_closed_form", "fric_use", "limit_torques_used")}),
                best_static_end=dict(pose=s_end[0], **{k: s_end[2][k] for k in ("m_s", "lean", "worst", "binding_joints", "cop_x", "Fz_up", "Fx", "F_zmp_closed_form")}),
                per_s_best=[dict(s=float(sd), n_poses=len(r), m_s=r[0][2]["m_s"], pose=r[0][0], lean=r[0][2]["lean"], worst=r[0][2]["worst"]) for sd, r in per_s])
