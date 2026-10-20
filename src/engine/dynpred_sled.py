"""DYNAMIC SLED PREDICTOR (track T2, 2026-10-17): engine.leverage_dyn (track B) rebuilt with the dynamic realisability it lacked.
Spec only (stock G1, vendor ratings, the model's friction); no constant fitted to a found displacement.

Track B's predictor maximised the sled mass with the centre of pressure ON THE SOLE EDGE (push: heel, pull: toe) and the feet at
0.77-0.84 of their friction: a statically admissible but dynamically unrealisable pose (an inverted pendulum whose CoP sits on the
support edge has no authority left on one side: any perturbation of the COM that way cannot be corrected, so the pose falls or the
foot rolls, which the 3 cm FEET_SLID gate sees). The dynamic version keeps every track B term (joint torques within ratings through
J^T including the 5 N m wrists, foot friction pyramid at the model mu, handle lift unloading the sled, sled tipping, the 0.2 m stroke
with the work-energy / horizon check (COM acceleration over the stroke enters through M_eff and the stroke time)) and adds:
  ZMP REALISABILITY: the CoP must sit at the middle of the support's fore-aft extent (equal stabilising authority in both directions:
  the only choice that does not presuppose the sign of the next perturbation). The force must then come from lean (COM offset),
  posture and joint torques, not from parking the CoP on an edge.
  FOOT SLIP: (a) the friction pyramid (model mu) as before; (b) soft-contact creep, from the model's solref/solimp: under a net
  horizontal load F the stance creeps at v = (1-d)/d * (d_max tau_s / 2) * F / M; over the 4 s horizon this stays below the 3 cm
  gate for F < 0.03 M 2 d / ((1-d) d_max tau_s T)  (computed and reported; it does not bind for these forces).
Interfaces: WELD (the simulator abstraction the search used) and POINT (palm force only).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
from scipy.optimize import linprog
import plant as P
from engine import leverage as LV, leverage_dyn as LD
G = 9.80665

def capacity_dyn(m, q, hand_body, p_hand, sgn, mu_sled, robot_bodies, foot_geoms, sled_off=(0.5, 0.0), sled_half=(0.3, 0.15),
             interface="weld", m_fixed=None, lim_tol=0.02, tol=0.003, foot_mu_scale=1.0, allow_lift=True, cop_center=True, minimize=False):
    """LP at pose q. sgn = +1 push (+x), -1 pull. Variables (scaled by robot weight W): u (nu, tau/tau_max), f (3 nc), l (limit
    torques, one-sided), w = [Fx, Fz, Mx, My, Mz] (the hand's wrench ON THE SLED at p_hand, Fy = 0), and m_s*g/W.
    m_fixed=None: maximise m_s. m_fixed=value: maximise the drive s Fx + mu Fz at that sled mass (F_drive).
    minimize=True (with m_fixed, Q4 2026-10-17): MINIMISE the drive instead -> the least hand force the robot needs from the sled
    at this pose (its braking / holding authority: the sled decelerates at (mu m_s g - drive_min)/m_s, holds if drive_min <= mu m_s g)."""
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
    A_eq2 = []; b_eq2 = []
    if cop_center:      # DYNAMIC REALISABILITY: the CoP at the middle of the support's x-extent (equal stabilising authority both ways)
        xm = 0.5*(pts[:, 0].min() + pts[:, 0].max()); row = np.zeros(n)
        for i in range(nc): row[iF + 3*i + 2] = pts[i, 0] - xm
        A_eq2.append(row); b_eq2.append(0.0)
    if A_eq2: Aeq = np.vstack([Aeq, np.array(A_eq2)]); beq = np.concatenate([beq, b_eq2])
    bnds = [(-1, 1) if np.isfinite(tm[a]) else (None, None) for a in range(nu)]
    bnds += [(0, None) if j % 3 == 2 else (None, None) for j in range(3*nc)] + [(0, None)]*nl
    bnds += [(None, None), (None, None) if allow_lift else (None, 0.0)] + ([(None, None)]*3 if interface == "weld" else [(0, 0)]*3)
    bnds += [(0, None)] if m_fixed is None else [(m_fixed*G/W, m_fixed*G/W)]
    cobj = np.zeros(n)
    if m_fixed is None: cobj[iM] = -1.0
    else: cobj[iW+0] = -s; cobj[iW+1] = -mu_sled
    if minimize and m_fixed is not None: cobj = -cobj
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




def predict(S, sgn, mu_sled=0.4, interface="weld", cop_center=True, **kw):
    """engine.leverage_dyn.predict with capacity -> capacity_dyn (the CoP-centre realisability row)."""
    orig = LD.capacity
    LD.capacity = lambda *a, **k: capacity_dyn(*a, cop_center=cop_center, **k)
    try:
        r = LD.predict(S, sgn, mu_sled=mu_sled, interface=interface, **kw)
    finally:
        LD.capacity = orig
    r["cop_center"] = cop_center; r["creep_force_limit_N"] = creep_limit(S)
    return r


def creep_limit(S, T=4.0, slide=0.03):
    """Net horizontal stance load below which MuJoCo's soft friction creep stays under `slide` over T (model solref/solimp)."""
    m = S["m"]; fg = sorted(S["foot_geoms"]); g = fg[0]
    tau_s = float(m.geom_solref[g][0]) if m.geom_solref[g][0] > 0 else 0.02
    d0, dw = float(m.geom_solimp[g][0]), float(m.geom_solimp[g][1]); d = dw; dmax = dw
    M = float(sum(m.body_mass[b] for b in S["robot_bodies"]))
    return float(slide*M*2*d/((1 - d)*dmax*tau_s*T))
