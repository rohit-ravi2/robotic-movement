"""HEAD-CARRY WITH ARM BALANCE + MAXIMUM LOAD CAPACITY (Track A, 2026-10-17). Robot G1H (G1 + Dex3 hands), simulator only.

The L1 probe showed that open-loop posture knots cannot keep a FREE load on G1H's domed head (it slides off in 0.6-1.7 s).
A person carrying a pot on the head does not use a ring: the head takes the weight and the hands, resting on the pot's sides,
balance it by feedback. This module builds that from the body, not from a script:

1. GRIP GEOMETRY (body IK): each palm's flat face (Dex3 palm plate, wrist_yaw_link) is placed on the object's side by damped
   least-squares IK on the 7 arm joints (palm point + finger direction + palm normal; engine.leverage.site_ik). The arm servo
   targets are that posture.
2. STABILISATION LAW (every 2 ms): object state (COM error e relative to the support (head top) or grip centre, relative
   velocity, tilt rotation vector and angular rate) -> desired corrective wrench on the object
        F = -m (wn^2 e + 2 z wn v)            (hands: + m g z_hat, the palms must also carry the weight)
        M = -I (wn^2 phi + 2 z wn omega)
   allocated to two palm point forces by the CAPACITY-WEIGHTED Jacobian (weighted least squares: wrench error + sum
   (tau_j/tau_max_j)^2; the throw prior's construction family), plus an inward squeeze N_i per palm sized for friction
        N_i = N_min + s_N |f_t,i|/mu,
   and the total law torque is scaled uniformly so no arm joint exceeds 0.9 tau_max (direction kept)
   joint torques tau = J_i^T w_i (+ arm gravity feed-forward) enter the arm servos as target offsets tau/kp (actuator torque,
   clipped to 0.9 tau_max). No weld, no applied external force on the object.
   GAIN RULE (from the object and the arm, before search): z = 1; wn = min(2 lambda, sqrt(F_cap/(m e_max))) where
   lambda = sqrt(g/h_c) is the toppling rate of the object's COM over the head contact point (h_c = COM height above it;
   hands: lambda = 0, wn from capacity only), F_cap = the worst-direction palm force the arm can produce from its torque
   envelope through J^T at the grip posture, e_max = the allowed COM excursion (head patch radius / grip slip tolerance).
   The search only refines scales and offsets on top of this.
   The whole-body balance is the Phase 14 capture-point stand controller, which reads the WORLD COM (robot + load).
3. MAX-LOAD PREDICTOR (spec only, before search): static contact LP (engine.leverage family) with the object as a free body in
   frictional contact (head: one point at the dome top, mu 1; palms: 4-corner patches, mu 1) and joint torques within limits;
   the largest mass whose worst-case utilisation over the object-contact geometry TILTED by theta about the COM (4 directions)
   stays <= 1. theta = 0 is the static capacity (a), theta > 0 adds the corrective-moment requirement (b).
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
from scipy.optimize import linprog

G = 9.80665
MU = 1.0
OBJ = {"box": dict(kind="box", size=(0.15, 0.15, 0.08)), "pot": dict(kind="cylinder", size=(0.13, 0.09))}
# palm plate (Dex3, wrist_yaw_link frame): face at local y=+0.017 (right) / -0.017 (left), spanning x 0.041..0.128, z +-0.044
PALM_C = {"r": np.array([0.085, 0.017, 0.0]), "l": np.array([0.085, -0.017, 0.0])}
PALM_N = {"r": np.array([0.0, 1.0, 0.0]), "l": np.array([0.0, -1.0, 0.0])}     # local outward normal of the palm face
PATCH = [(0.05, 0.035), (0.05, -0.035), (0.12, 0.035), (0.12, -0.035)]           # (x, z) corners of the modelled palm patch
A_PATCH = 0.035
HEAD_PATCH_R = 0.07      # head support patch radius (head mesh half-extent in y is 0.078 m)
GRIP_SLIP = 0.03         # hands strategy: allowed grip excursion used by the gain rule
ARM7 = ["shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint", "elbow_joint", "wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint"]
RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "overnight", "A"))


def half(obj):
    o = OBJ[obj]; return (o["size"][1], o["size"][2]) if o["kind"] == "box" else (o["size"][0], o["size"][1])   # (half width along y, half height)


def parts(m):
    from engine import leverage as LV
    H = LV.g1h_parts(m)
    wb = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{'right' if s == 'r' else 'left'}_wrist_yaw_link") for s in "rl"}
    arm = {s: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{'right' if s == 'r' else 'left'}_{j}") for j in ARM7] for s in "rl"}
    return H, wb, arm


def head_local(m, d, H):
    """Head-top point in the torso frame (fixed: the head is part of torso_link)."""
    from engine import leverage as LV
    tb = H["torso"]; ht = LV.head_top(m, d, H["head_geom"]); R = d.xmat[tb].reshape(3, 3)
    return R.T @ (ht - d.xpos[tb])


def _rot_about(v, axis, ang):
    axis = axis/np.linalg.norm(axis); return v*np.cos(ang) + np.cross(axis, v)*np.sin(ang) + axis*(axis @ v)*(1 - np.cos(ang))


def grip_geometry(m, q, strategy, obj, gp):
    """Object nominal centre (world, at pose q) and the palm targets. gp: dz (contact height on the side, fraction of half
    height), dx (fore-aft shift of the contact, m), fang (finger direction rotated about the palm normal, rad), hx/hz (hands:
    grip centre ahead of the torso / height). Returns centre, {s: (point, inward normal, finger dir)}."""
    H, wb, arm = parts(m); d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_forward(m, d)
    hw, hh = half(obj)
    if strategy == "head":
        from engine import leverage as LV
        c = LV.head_top(m, d, H["head_geom"]) + [0, 0, hh + 0.002]; f0 = np.array([0, 0, 1.0])
    else:
        tb = H["torso"]; c = np.array([d.xpos[tb][0] + gp.get("hx", 0.30), d.xpos[tb][1], gp.get("hz", 1.0)]); f0 = np.array([1.0, 0, 0])
    out = {}
    for s, sg in (("r", -1), ("l", 1)):
        n_in = np.array([0, -sg, 0.0])                      # into the object: right palm (y<0) pushes +y
        p = c + [gp.get("dx", 0.0), sg*hw, gp.get("dz", 0.0)*hh]
        fdir = _rot_about(f0, n_in, sg*gp.get("fang", 0.0))
        out[s] = (p, n_in, fdir)
    return c, out


def clear_ik(m, q, targets, bid, ja, c, gap, rounds=4):
    """Grip IK, then shift each palm target along its normal until the CLOSEST hand geom (palm, wrist links, thumb, fingers) is
    exactly `gap` from the object placed at its nominal centre c (gap > 0: the hands rest next to the sides and only the law's
    torques make contact; gap < 0: a geometric clamp from the posture). Returns q, ik residual, per-side achieved distance."""
    H, wb, arm = parts(m); og = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
    hg = {s: [g for g in range(m.ngeom) if m.geom_contype[g] and _under(m, g, int(m.body_parentid[int(m.body_parentid[wb[s]])]))] for s in "rl"}
    tg = {s: list(v) for s, v in targets.items()}; d = mujoco.MjData(m); fr = np.zeros(6)
    qa = None
    for _ in range(rounds):
        qa, err, perr = grip_ik(m, q if qa is None else qa, {s: tuple(v) for s, v in tg.items()}, n_restart=(4 if qa is None else 1))
        d.qpos[:] = qa; d.qpos[ja:ja+3] = c; d.qpos[ja+3:ja+7] = [1, 0, 0, 0]; mujoco.mj_forward(m, d)
        dist = {s: min(mujoco.mj_geomDistance(m, d, g1, g2, 0.2, fr) for g1 in og for g2 in hg[s]) for s in "rl"}
        if all(abs(dist[s] - gap) < 5e-4 for s in "rl"): break
        for s in "rl": tg[s][0] = np.asarray(tg[s][0]) - (gap - dist[s])*np.asarray(tg[s][1])   # move against the inward normal
    return qa, err, dist


def grip_ik(m, q, targets, n_restart=4):
    """Arm IK (palm centre on the side, finger direction, palm normal into the object). Tries a few starting arm postures and
    keeps the lowest residual. Returns q, residual (m)."""
    from engine import leverage as LV
    H, wb, arm = parts(m); names = [f"{'right' if s == 'r' else 'left'}_{j}" for s in "rl" for j in ARM7]
    tg = []
    for s in "rl":
        p, n_in, fdir = targets[s]
        tg += [(wb[s], PALM_C[s], p), (wb[s], PALM_C[s] + 0.06*np.array([1.0, 0, 0]), p + 0.06*fdir), (wb[s], PALM_C[s] + 0.06*PALM_N[s], p + 0.06*n_in)]
    best = (np.inf, None)
    starts = [dict(), dict(shoulder_pitch_joint=-1.2, elbow_joint=0.5), dict(shoulder_pitch_joint=-2.4, elbow_joint=0.3), dict(shoulder_pitch_joint=-2.8, shoulder_roll_joint=0.6, elbow_joint=-0.5)]
    for st in starts[:n_restart]:
        q0 = np.array(q, float)
        for s, sg in (("r", -1), ("l", 1)):
            for j, v in st.items():
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{'right' if s == 'r' else 'left'}_{j}")
                q0[m.jnt_qposadr[jid]] = v*sg if "roll" in j else v
        qs, err = LV.site_ik(m, q0, tg, names, iters=400, step=0.5, damp=1e-3, tol=1e-4)
        # residual of the palm CENTRES only (the orientation points share the error budget)
        d = mujoco.MjData(m); d.qpos[:] = qs; mujoco.mj_forward(m, d)
        e = max(np.linalg.norm(d.xpos[wb[s]] + d.xmat[wb[s]].reshape(3, 3) @ PALM_C[s] - targets[s][0]) for s in "rl")
        if err < best[0]: best = (err, qs, e)
    return best[1], float(best[0]), float(best[2])


# ------------------------------------------------------------------------------------------------ predictor (spec only)
def contact_lp(m, q, obj_contacts, com, mass, floor_geoms, robot_bodies, exclude_act, tol=0.003, acc=(0.0, 0.0)):
    """Static LP: robot (feet on the floor, joint torques within limits) + a FREE object of `mass` at `com` touching the robot at
    obj_contacts = [dict(body, point (robot side, for the Jacobian), opoint (object side, for the object's moment balance),
    normal (into the object), mu)]. Minimise the max joint utilisation s. Returns s (inf if infeasible), worst joint, util.
    acc: a horizontal SWAY acceleration of the whole system (robot and object), D'Alembert: effective gravity (acc_x, acc_y, -g)."""
    import plant as P; from engine import leverage as LV
    d = mujoco.MjData(m); d.qpos[:] = q; d.qvel[:] = 0; g0 = m.opt.gravity.copy(); m.opt.gravity[:] = [acc[0], acc[1], -G]
    try: mujoco.mj_forward(m, d)
    finally: m.opt.gravity[:] = g0
    b = np.array(d.qfrc_bias, float); mujoco.mj_forward(m, d)
    nv, nu = m.nv, m.nu; rd = np.array([i for i in range(nv) if int(m.dof_bodyid[i]) in robot_bodies])
    cp = P.potential_contacts(m, d, sorted(set(floor_geoms)), tol, with_geom=True)
    fl = [dict(body=int(m.geom_bodyid[int(c[3])]), point=np.array(c[:3], float), normal=np.array([0, 0, 1.0]), mu=0.9) for c in cp]
    W = float(sum(m.body_mass[b] for b in robot_bodies))*G
    tm = LV.tau_max(m); tm_s = np.where(np.isfinite(tm), tm, 1e3)
    nf, no = len(fl), len(obj_contacts); n = nu + 3*nf + 3*no + 1; iS = n - 1; io = nu + 3*nf
    Sel = np.zeros((nv, nu))
    for a in range(nu): Sel[int(m.jnt_dofadr[int(m.actuator_trnid[a][0])]), a] = float(m.actuator_gear[a][0])*tm_s[a]
    jp = np.zeros((3, nv)); Aeq = np.zeros((nv, n)); Aeq[:, :nu] = Sel
    def basis(nrm):
        nrm = nrm/np.linalg.norm(nrm); t1 = np.cross(nrm, [1, 0, 0] if abs(nrm[0]) < 0.9 else [0, 1, 0]); t1 /= np.linalg.norm(t1); return np.stack([nrm, t1, np.cross(nrm, t1)], 1)
    for i, c in enumerate(fl):
        mujoco.mj_jac(m, d, jp, None, c["point"], c["body"]); Aeq[:, nu+3*i:nu+3*i+3] = jp.T @ basis(c["normal"])*W
    Bo = []
    for k, c in enumerate(obj_contacts):     # force ON the object = Bk x_k*W; the robot feels the opposite
        Bk = basis(c["normal"]); Bo.append(Bk)
        mujoco.mj_jac(m, d, jp, None, c["point"], c["body"]); Aeq[:, io+3*k:io+3*k+3] = -jp.T @ Bk*W
    Aeq, beq = Aeq[rd]/W, b[rd]/W
    Ao = np.zeros((6, n)); bo = np.zeros(6); bo[:3] = np.array([-acc[0], -acc[1], G])*mass/W
    for k, c in enumerate(obj_contacts):
        r_ = np.asarray(c["opoint"]) - com; Rx = np.array([[0, -r_[2], r_[1]], [r_[2], 0, -r_[0]], [-r_[1], r_[0], 0]])
        Ao[:3, io+3*k:io+3*k+3] = Bo[k]; Ao[3:, io+3*k:io+3*k+3] = Rx @ Bo[k]
    Aeq = np.vstack([Aeq, Ao]); beq = np.concatenate([beq, bo])
    rows = []
    for i in range(nu):
        for sg in (1, -1):
            r = np.zeros(n); r[i] = sg; r[iS] = -1.0; rows.append(r)
    for i, c in enumerate(fl + list(obj_contacts)):
        o = nu + 3*i; k = c["mu"]/np.sqrt(2.0)
        for tt in (1, 2):
            for sg in (1, -1):
                r = np.zeros(n); r[o+tt] = sg; r[o] = -k; rows.append(r)
    bnds = [(None, None)]*nu + [(0, None) if j % 3 == 0 else (None, None) for j in range(3*(nf + no))] + [(0, None)]
    keep = [a for a in range(nu) if a not in set(exclude_act)]
    # exclude_act (fingers) are left out of s: their rows are dropped
    rows = [r for r in rows if not any(r[a] != 0 for a in exclude_act)]
    c1 = np.zeros(n); c1[iS] = 1.0
    r1 = linprog(c1, A_ub=np.array(rows), b_ub=np.zeros(len(rows)), A_eq=Aeq, b_eq=beq, bounds=bnds, method="highs")
    if not r1.success: return dict(s=np.inf, status="INFEASIBLE")
    u = np.abs(r1.x[:nu]); names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a][0])) for a in range(nu)]
    wj = max(keep, key=lambda a: u[a]); fo = r1.x[io:io+3*no].reshape(no, 3)*W
    return dict(s=float(r1.x[iS]), status="OK", worst=names[wj], util={names[a]: float(u[a]) for a in keep}, obj_normal=fo[:, 0].tolist())


def obj_contacts(m, q, strategy, obj, tilt=None):
    """Object-side contact set at posture q (palms from the actual IK posture). tilt = (axis (3,), angle): the whole object-contact
    geometry rotated about the COM (contact points + normals); the robot-side Jacobian points stay at the posture."""
    H, wb, arm = parts(m); d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_forward(m, d)
    hw, hh = half(obj); cs = []
    pal = {}
    for s in "rl":
        R = d.xmat[wb[s]].reshape(3, 3); pal[s] = R
        for (x, z) in PATCH:
            lp = np.array([x, PALM_C[s][1], z]); p = d.xpos[wb[s]] + R @ lp
            cs.append(dict(body=wb[s], point=p, opoint=p.copy(), normal=R @ PALM_N[s], mu=MU))
    pc = 0.5*(sum(c["point"] for c in cs[:4])/4 + sum(c["point"] for c in cs[4:])/4)
    if strategy == "head":
        from engine import leverage as LV
        ht = LV.head_top(m, d, H["head_geom"]); com = ht + [0, 0, hh]
        cs.append(dict(body=H["torso"], point=ht, opoint=ht.copy(), normal=np.array([0, 0, 1.0]), mu=MU))
    else:
        com = pc
    if tilt is not None:
        ax, ang = tilt
        for c in cs: c["opoint"] = com + _rot_about(c["opoint"] - com, ax, ang); c["normal"] = _rot_about(c["normal"], ax, ang)
    return cs, com


def predict_max_load(strategy, obj, gp=None, theta=10.0, sway_frac=0.5, m_hi=200.0, S=None):
    """MAXIMUM LOAD from the spec (before any search), three requirements, each a bisection on the mass (1e-2 kg):
    (a) STATIC: the contact LP (object weight through its contacts, joint torques within limits) has max utilisation <= 1;
    (b) ARM CORRECTION: the same with the object-contact geometry TILTED by theta about the object's COM, worst of 4 directions
        (the COM leaves the line over the single head contact: a moment only the palms can supply; hands: the grip tilts);
    (c) BALANCE / COM: the largest horizontal sway acceleration the stance can sustain statically (D'Alembert gravity (a_x, a_y,
        -g) on robot AND load; worst of 4 directions; limited by the feet's CoP / ankle-hip torques) must stay >= sway_frac x the
        UNLOADED robot's own value. Loading the head raises the COM and loading the hands moves it forward, which is what this
        measures. m_max = min(a, b, c); the binding requirement and joint are reported."""
    import throw as TH
    S = S or TH.setup("G1H", ball=False); m = S["m"]; q = np.array(S["q_settled"]); gp = gp or {}
    c, tg = grip_geometry(m, q, strategy, obj, gp); qa, err, perr = grip_ik(m, q, tg)
    H, wb, arm = parts(m); fg = sorted(S["foot_geoms"])
    D4 = [np.array(a, float) for a in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))]
    def lp(mass, tilt=None, acc=(0.0, 0.0)):
        cs, com = obj_contacts(m, qa, strategy, obj, tilt)
        if mass <= 0: cs, com = [], com
        return contact_lp(m, qa, cs, com, max(mass, 0.0), fg, S["robot_bodies"], H["fingers"], acc=acc)
    def s_static(mass): return lp(mass)
    def s_tilt(mass): return max((lp(mass, (ax, np.radians(theta))) for ax in D4), key=lambda r: r["s"])
    def a_max(mass, dirs=D4):
        out = []
        for u in dirs:
            lo, hi = 0.0, 6.0
            if lp(mass, None, tuple(hi*u[:2]))["s"] <= 1: out.append(hi); continue
            while hi - lo > 0.01:
                mid = 0.5*(lo + hi)
                if lp(mass, None, tuple(mid*u[:2]))["s"] <= 1: lo = mid
                else: hi = mid
            out.append(lo)
        return min(out), out
    a0, a0_dirs = a_max(0.0); a_req = sway_frac*a0
    def bis(ok, lo=0.5, hi=m_hi):
        if ok(hi): return hi, "ceiling"
        if not ok(lo): return 0.0, "fails at 0.5 kg"
        while hi - lo > 0.01:
            mid = 0.5*(lo + hi)
            if ok(mid): lo = mid
            else: hi = mid
        return round(lo, 2), hi
    ma, ha = bis(lambda mm: s_static(mm)["s"] <= 1)
    mb, hb = bis(lambda mm: s_tilt(mm)["s"] <= 1)
    mc, hc = bis(lambda mm: a_max(mm)[0] >= a_req)
    def lim(r): return r.get("worst") if r["status"] == "OK" else "no static solution (friction / tipping)"
    req = dict(a_static=dict(m_max=ma, limiting=(lim(s_static(hb if isinstance(ha, float) else ma)) if isinstance(ha, float) else ha)),
               b_tilt=dict(m_max=mb, theta_deg=theta, limiting=(lim(s_tilt(hb)) if isinstance(hb, float) else hb)),
               c_balance=dict(m_max=mc, a_unloaded=a0, a_unloaded_dirs=a0_dirs, a_required=a_req, limiting="sway acceleration capacity (CoP / ankle-hip)"))
    req["a_static"]["limiting"] = lim(s_static(ha)) if isinstance(ha, float) else ha
    binding = min(req, key=lambda k: req[k]["m_max"])
    curve = {f"{mm:g}": dict(s_static=s_static(mm)["s"], s_tilt=s_tilt(mm)["s"], tilt_worst=s_tilt(mm).get("worst"), a_max=a_max(mm)[0]) for mm in (2, 5, 8, 12, 16)}
    return dict(strategy=strategy, obj=obj, ik_err=err, palm_err=perr, gp=gp, requirements=req, m_max=req[binding]["m_max"], binding=binding, curve=curve)


# ------------------------------------------------------------------------------------------------ the rollout with the law
def arm_capacity(m, d, wb, arm, s, pt, dirs=None):
    """Worst-direction palm force the arm can produce from its torque envelope through J^T (N), at the current posture."""
    from engine import leverage as LV
    tm = LV.tau_max(m); jp = np.zeros((3, m.nv)); mujoco.mj_jac(m, d, jp, None, pt, wb[s])
    dofs = [int(m.jnt_dofadr[j]) for j in arm[s]]; aid = [[a for a in range(m.nu) if int(m.actuator_trnid[a][0]) == j][0] for j in arm[s]]
    J = jp[:, dofs]; tmax = tm[aid]
    if dirs is None:
        dirs = [np.array(v, float)/np.linalg.norm(v) for v in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0), (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1)]]
    return float(min(1.0/np.max(np.abs(J.T @ u)/tmax) for u in dirs))


def decode(z, strategy):
    """Search vector (all in [-1, 1]) -> parameters. Index: 0 k_cp, 1 k_trunk (stand gains 0.5..1.5x), 2 wn scale (2^(1.5 z)),
    3 zeta scale (2^z), 4 squeeze s_N (1 + 1.5 z, >= 0.1), 5 N_min (m g * 0.15 * 2^z), 6 contact height dz (head: 0.5+-0.5 of the half height, so the wrist clears the bottom edge; hands: +-0.6), 7 contact dx (+-4 cm),
    8 finger angle (+-0.8 rad), 9 waist pitch offset (+-0.3 rad), 10/11 hands: grip centre hx (0.30 +- 0.1), hz (1.0 +- 0.15), 12 hand-object clearance at the grip posture (0 +- 5 mm: hands resting on the sides;
    negative = a geometric clamp), 13 arm servo frequency scale (2^z x the rule w_arm = wn)."""
    z = np.clip(np.asarray(z, float), -1, 1); zz = np.zeros(14); zz[:len(z)] = z
    gp = dict(dz=(0.5 + 0.5*zz[6]) if strategy == "head" else 0.6*zz[6], dx=0.04*zz[7], fang=0.8*zz[8])
    if strategy == "hands": gp.update(hx=0.30 + 0.10*zz[10], hz=1.0 + 0.15*zz[11])
    return dict(gap=0.005*zz[12], k_cp=zz[0], k_trunk=zz[1], wn_s=2**(1.5*zz[2]), zeta_s=2**zz[3], sN=max(0.1, 1 + 1.5*zz[4]), Nmin_s=2**zz[5], waist=0.3*zz[9], warm_s=2**zz[13], gp=gp)


DIM = {"head": 14, "hands": 14}   # head ignores 10/11


def plant(obj, mass, dt, fresh=None):
    """fresh (default: env HC_FRESH, on unless HC_FRESH=0): drop every cached world-scene plant first, so each rollout gets a
    plant BUILT at its own dt (compiled from the MJCF, settled at dt) instead of a cached plant re-timed from another dt and
    carrying servo changes from earlier rollouts in the same worker (T1 support gate, 2026-10-17)."""
    import leverage_probes as LP, throw as TH
    if fresh is None: fresh = os.environ.get("HC_FRESH", "1") != "0"
    if fresh:
        for k in [k for k in TH._CACHE if k[3] is not None]: TH._CACHE.pop(k)
    S, path = LP.l1_plant(obj, mass, "head", dt); return S, path


def rollout(obj, mass, strategy, z, T=10.0, dt=0.001, record=False, sway=None, law=True, t_pre=1.5, push=None, lag=None, wcap=None):
    """Pre-settle t_pre s (object parked away, arms move to the grip posture, capture point on the ROBOT COM), then the object is
    PLACED (head: on the head top, palms at its sides; hands: between the palms) at rest and the hold of T s begins.
    sway = (amplitude m, frequency Hz): lateral capture-point target oscillation (weight shift foot to foot) during the hold.
    push = [(t0, duration, (Fx, Fy, Fz))]: external pushes on the torso (a disturbance test; the only applied force in the scene).
    law=False: the same posture with no feedback law (ablation). Returns hold metrics and, with record, a qpos trace."""
    import balance as B, branch_i as BI; from engine import core, leverage as LV
    S, path = plant(obj, mass, dt); m = S["m"]; P_ = decode(z, strategy); H, wb, arm = parts(m)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "load"); ja = int(m.jnt_qposadr[m.body_jntadr[bid]]); jv = int(m.jnt_dofadr[m.body_jntadr[bid]])
    tb = H["torso"]; pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis"); hw, hh = half(obj)
    q0 = np.array(S["q_settled"]); q0[ja:ja+3] = [0.0, 2.5, hh + 0.01]; q0[ja+3:ja+7] = [1, 0, 0, 0]
    # grip posture (IK at the settled stance) and the object's nominal centre in the torso frame
    c_w, tg = grip_geometry(m, q0, strategy, obj, P_["gp"]); qa, ik_err, palm_err = clear_ik(m, q0, tg, bid, ja, c_w, P_["gap"])
    d = mujoco.MjData(m); d.qpos[:] = q0; mujoco.mj_forward(m, d)
    R0 = d.xmat[tb].reshape(3, 3); c_loc = R0.T @ (c_w - d.xpos[tb]); h_loc = head_local(m, d, H)
    aid = {s: [[a for a in range(m.nu) if int(m.actuator_trnid[a][0]) == j][0] for j in arm[s]] for s in "rl"}
    dofs = {s: [int(m.jnt_dofadr[j]) for j in arm[s]] for s in "rl"}; qadr = {s: [int(m.jnt_qposadr[j]) for j in arm[s]] for s in "rl"}
    tm = LV.tau_max(m)
    # GAIN RULE (stated in the module doc): from the object's mass/inertia and the arm's capacity at the grip posture
    d2 = mujoco.MjData(m); d2.qpos[:] = qa; mujoco.mj_forward(m, d2)
    Fcap = min(arm_capacity(m, d2, wb, arm, s, d2.xpos[wb[s]] + d2.xmat[wb[s]].reshape(3, 3) @ PALM_C[s]) for s in "rl")
    h_c = hh if strategy == "head" else 0.0; e_max = HEAD_PATCH_R if strategy == "head" else GRIP_SLIP
    lam = np.sqrt(G/h_c) if h_c > 0 else 0.0; wn_cap = np.sqrt(2*Fcap/(mass*e_max))   # two palms
    wn0 = min(2*lam, wn_cap) if lam > 0 else min(wn_cap, 20.0); wn = wn0*P_["wn_s"]; zeta = 1.0*P_["zeta_s"]
    Iobj = np.array(m.body_inertia[bid]); Nmin = (0.02 if strategy == "head" else 0.15)*mass*G*P_["Nmin_s"]   # head: a light touch (the head carries the weight)
    ctl, par = core.landing_controller(S, "G1H"); ctl.p["mode"] = "stand"; ctl.t_touch = -1e9
    ctl.p["k_cp"] *= 1.0 + 0.5*P_["k_cp"]; ctl.p["k_trunk"] *= 1.0 + 0.5*P_["k_trunk"]; ctl._kc = None
    wp = [a for a in range(m.nu) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a][0])) == "waist_pitch_joint"][0]
    d.qpos[:] = q0; d.qvel[:] = np.array(S["v_settled"]); d.qvel[jv:jv+6] = 0; mujoco.mj_forward(m, d)
    restore = B.all_servos(S, ctl.p["stiff"])
    # ARM IMPEDANCE: the stand controller's servos are stiff (shoulder kp ~ 7e3 Nm/rad: the palm barely yields, so a torque-level
    # law cannot create a contact force). The arm servos get natural frequency w_arm = wn (the object law's bandwidth, the rule)
    # x a searched scale, critically damped; gravity is fed forward, so the posture holds and tau/kp offsets ARE contact forces.
    w_arm = wn*P_["warm_s"]
    for s in "rl":
        for a in aid[s]:
            ieff = float(m.actuator_gainprm[a, 0])/ctl.p["stiff"]**2; kpa = w_arm**2*ieff
            BI._as_servo(m, int(a), kpa, 2.0*np.sqrt(kpa*ieff), float(m.actuator_forcerange[a][1]))
    LG = _lag_setup(m, S, ACT_LAG if lag is None else lag, LAG_WCAP if wcap is None else wcap)   # opt-in actuator lag (Q1); None at lag 0
    kp = np.array(m.actuator_gainprm[:, 0]); AA = np.array(aid["r"] + aid["l"]); QA = np.array(qadr["r"] + qadr["l"]); DA = np.array(dofs["r"] + dofs["l"]); CR = np.array(m.actuator_ctrlrange[AA])
    if LG is not None: kp[LG["a"]] = LG["kp"]; CR = np.c_[LG["lo_all"][AA], LG["hi_all"][AA]]; _lag_motors(m, LG)
    keep = np.array([np.isfinite(tm[a]) and "hand" not in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a][0])) or "") for a in range(m.nu)])
    head_geoms = set(g for g in range(m.ngeom) if m.geom_bodyid[g] == tb); obj_geoms = set(g for g in range(m.ngeom) if m.geom_bodyid[g] == bid)
    floor = set(g for g in range(m.ngeom) if m.geom_bodyid[g] == 0); palm_geoms = {s: set(g for g in range(m.ngeom) if _under(m, g, wb[s])) for s in "rl"}
    cp0 = np.asarray(getattr(ctl, "cp_offset", 0.0), float); jc = np.zeros((3, m.nv)); jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    n_pre = int(round(t_pre/dt)); n = n_pre + int(round(T/dt)); trace = []; reason = None; t = -t_pre
    st = dict(u2=0.0, k=0, umax=[], head_frac=[], palm_N={"r": [], "l": []}, e=[], tilt=[], arm_u=np.zeros(14), arm_k=0, emax=0.0, tiltmax=0.0, pely=[])
    tau_ff = np.zeros(m.nu); placed = False; cf = np.zeros(6); dpred = None
    # SUPPORT GATE instrumentation (T1): every SUP_DT s of sim time (dt-independent sampling) each foot's horizontal position
    # (ankle roll link origin) and floor normal force; path length is accumulated only between samples where that foot is LOADED
    feet_b = {s_: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{'right' if s_ == 'r' else 'left'}_ankle_roll_link") for s_ in "rl"}
    feet_g = {s_: set(g for g in range(m.ngeom) if _under(m, g, feet_b[s_])) for s_ in "rl"}
    n_sup = max(1, int(round(SUP_DT/dt))); W_tot = (ROBOT_MASS + mass)*G
    sup = dict(prev={s_: None for s_ in "rl"}, travel={s_: 0.0 for s_ in "rl"}, pel0=None, drift=0.0, n_load={s_: 0 for s_ in "rl"}, k=0)
    try:
        for (a_, dof, qad, tau, kp_, kd_) in S["held"]: d.ctrl[a_] = q0[qad]
        tgt = d.ctrl.copy()      # lag mode: servo TARGETS live here; d.ctrl of the lagged actuators carries torques
        for s_ in range(n):
            t = s_*dt - t_pre; k_c = int(round((t + t_pre)/B.CTRL_DT, 6))
            if not placed and s_ >= n_pre and not PLACE:     # UNLOADED stand test: start the hold clock, the object stays parked
                placed = True; ctl.cp_offset = cp0
            if not placed and s_ >= n_pre:     # PLACE the object (initial condition: a helper sets it down at rest)
                Rt = d.xmat[tb].reshape(3, 3); c_now = d.xpos[tb] + Rt @ c_loc
                if strategy == "head":
                    ht = d.xpos[tb] + Rt @ h_loc; c_now = np.array([ht[0], ht[1], ht[2] + hh + 0.0005])
                yaw = np.arctan2(Rt[1, 0], Rt[0, 0]); d.qpos[ja:ja+3] = c_now; d.qpos[ja+3:ja+7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
                d.qvel[jv:jv+6] = 0; mujoco.mj_forward(m, d); placed = True; ctl.cp_offset = cp0
                if os.environ.get("HC_DEBUG"):
                    og = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]; fr = np.zeros(6)
                    print("placed: palm-object distance", {s: round(min(mujoco.mj_geomDistance(m, d, g1, g2, 0.2, fr) for g1 in og for g2 in range(m.ngeom) if m.geom_contype[g2] and _under(m, g2, wb[s])), 4) for s in "rl"},
                          "kp arm", np.round(kp[aid["r"]], 1))
            if k_c != ctl._kc:
                ctl._kc = k_c
                if LG is not None: d.ctrl[:] = tgt
                if not placed:   # capture point on the ROBOT COM while the object is parked away
                    mujoco.mj_jacSubtreeCom(m, d, jc, 0); vw = jc @ d.qvel; mujoco.mj_jacSubtreeCom(m, d, jc, pel); vr = jc @ d.qvel
                    w0 = np.sqrt(G/max(float(d.subtree_com[0][2]), 0.2)); ctl.cp_offset = cp0 + (d.subtree_com[0][:2] - d.subtree_com[pel][:2]) + (vw[:2] - vr[:2])/w0
                elif sway is not None and t > 1.0:
                    ctl.cp_offset = cp0 + np.array([0.0, sway[0]*np.sin(2*np.pi*sway[1]*(t - 1.0))])
                dP = d
                if LG is not None and LG.get("comp") in ("pred", "pp") and PRED_OUTER:   # K2: the balance loop acts on the state predicted one lag ahead
                    if dpred is None: dpred = mujoco.MjData(m)
                    dpred.qpos[:] = d.qpos; dpred.qvel[:] = d.qvel; mujoco.mj_integratePos(m, dpred.qpos, d.qvel, LG["tau_hat"]); mujoco.mj_forward(m, dpred); dP = dpred
                d.ctrl[:] = ctl.targets(k_c*B.CTRL_DT, dP, True); w = min(1.0, (t + t_pre)/1.0)
                d.ctrl[wp] += w*P_["waist"]
                d.ctrl[AA] = (1 - w)*q0[QA] + w*qa[QA]
                tau_ff[:] = 0; tau_ff[AA] = d.qfrc_bias[DA]    # arm gravity feed-forward (the servo would otherwise sag by g/kp)
                if placed and law and PLACE:
                    tau_ff += law_torques(m, d, strategy, bid, tb, c_loc, h_loc, mass, Iobj, wn, zeta, P_["sN"], Nmin, wb, aid, dofs, tm, jp, jr)
                d.ctrl[AA] = np.clip(d.ctrl[AA] + np.clip(tau_ff[AA], -0.9*tm[AA], 0.9*tm[AA])/kp[AA], CR[:, 0], CR[:, 1])
                if LG is not None: tgt = d.ctrl.copy()
            if LG is not None: _lag_step(m, d, LG, tgt)
            if push:
                d.xfrc_applied[tb, :3] = 0
                for (tp, du, Fp) in push:
                    if tp <= t < tp + du: d.xfrc_applied[tb, :3] = Fp
            BI.backemf_brake(m, d, S); mujoco.mj_step(m, d)
            if not np.isfinite(d.qacc).all(): reason = "DIVERGED"; break
            if d.xmat[tb][8] < np.cos(np.radians(35)): reason = "FELL"; break
            if not placed: continue
            if (s_ - n_pre) % n_sup == 0:
                fz = {"r": 0.0, "l": 0.0}
                for ci in range(d.ncon):
                    con = d.contact[ci]; g1, g2 = con.geom1, con.geom2
                    if (g1 in floor) == (g2 in floor): continue
                    other = g2 if g1 in floor else g1
                    for s2 in "rl":
                        if other in feet_g[s2]: mujoco.mj_contactForce(m, d, ci, cf); fz[s2] += abs(cf[0])
                for s2 in "rl":
                    p2 = np.array(d.xpos[feet_b[s2]][:2])
                    if fz[s2] >= LOAD_FRAC*W_tot:
                        if sup["prev"][s2] is not None: sup["travel"][s2] += float(np.linalg.norm(p2 - sup["prev"][s2]))
                        sup["prev"][s2] = p2; sup["n_load"][s2] += 1
                    else: sup["prev"][s2] = None
                pp = np.array(d.xpos[pel][:2])
                if sup["pel0"] is None: sup["pel0"] = pp
                sup["drift"] = max(sup["drift"], float(np.linalg.norm(pp - sup["pel0"]))); sup["k"] += 1
            c = d.xipos[bid]; Rt = d.xmat[tb].reshape(3, 3)
            if not PLACE: en = 0.0
            elif strategy == "head":
                ht = d.xpos[tb] + Rt @ h_loc; e = c[:2] - ht[:2]; en = float(np.linalg.norm(e))
                if en > HEAD_PATCH_R: reason = "OFF_PATCH"; break
                if c[2] < ht[2]: reason = "DROPPED"; break
            else:
                cw = d.xpos[tb] + Rt @ c_loc; en = float(np.linalg.norm(c - cw))
                if en > 0.10: reason = "SLIPPED"; break
            tilt = float(np.degrees(np.arccos(np.clip(d.xmat[bid][8], -1, 1)))); st["emax"] = max(st["emax"], en); st["tiltmax"] = max(st["tiltmax"], tilt)
            if os.environ.get("HC_DEBUG") and s_ % 1000 == 0:
                for ci in range(d.ncon):
                    con = d.contact[ci]
                    if con.geom1 in obj_geoms or con.geom2 in obj_geoms:
                        mujoco.mj_contactForce(m, d, ci, cf); print(round(t, 2), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[con.geom1])), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[con.geom2])), np.round(con.pos, 3), np.round(con.frame[:3], 2), round(cf[0], 1), round(con.dist, 4))
            if s_ % 10 == 0:
                hn = 0.0; pn = {"r": 0.0, "l": 0.0}; fl_hit = False
                for ci in range(d.ncon):
                    con = d.contact[ci]; g1, g2 = con.geom1, con.geom2
                    if g1 in obj_geoms or g2 in obj_geoms:
                        other = g2 if g1 in obj_geoms else g1
                        if other in floor: fl_hit = True
                        mujoco.mj_contactForce(m, d, ci, cf)
                        if other in head_geoms: hn += abs(cf[0])*abs(con.frame[2])
                        for s in "rl":
                            if other in palm_geoms[s]: pn[s] += abs(cf[0])
                if fl_hit and PLACE: reason = "FLOOR"; break
                u = np.abs(d.actuator_force[keep])/tm[keep]; st["u2"] += float(np.mean(u**2)); st["k"] += 1; st["umax"].append(float(u.max()))
                st["head_frac"].append(hn/(mass*G)); st["palm_N"]["r"].append(pn["r"]); st["palm_N"]["l"].append(pn["l"]); st["e"].append(en); st["tilt"].append(tilt)
                st["arm_u"] += np.abs(np.array([d.actuator_force[a]/tm[a] for s in "rl" for a in aid[s]])); st["arm_k"] += 1
                st["pely"].append(float(d.subtree_com[0][1]))
                if record: trace.append(d.qpos.copy())
    finally:
        if LG is not None: _lag_restore(m, LG)
        restore()
    k = max(1, st["k"]); held = float(t if reason else T)
    names = [f"{s}_{j}" for s in "rl" for j in ARM7]
    r = dict(t_held=max(0.0, held), reason=reason, util_rms=float(np.sqrt(st["u2"]/k)), util_p95=(float(np.percentile(st["umax"], 95)) if st["umax"] else None),
             head_frac=(float(np.mean(st["head_frac"])) if st["head_frac"] else 0.0), palm_N_mean={s: (float(np.mean(v)) if v else 0.0) for s, v in st["palm_N"].items()},
             e_mean=(float(np.mean(st["e"])) if st["e"] else None), e_max=st["emax"], com_y_range=(float(np.ptp(st["pely"])) if st["pely"] else None), tilt_max_deg=st["tiltmax"], tilt_mean_deg=(float(np.mean(st["tilt"])) if st["tilt"] else None),
             arm_util_mean=dict(zip(names, (st["arm_u"]/max(1, st["arm_k"])).round(3).tolist())), gains=dict(w_arm=w_arm, wn=wn, wn_rule=wn0, lam=lam, wn_cap=wn_cap, F_cap=Fcap, zeta=zeta, Nmin=Nmin, sN=P_["sN"]),
             ik_err=ik_err, palm_err=palm_err, scene=path, foot_travel=dict(sup["travel"]), foot_travel_max=max(sup["travel"].values()),
             foot_loaded_frac={s2: sup["n_load"][s2]/max(1, sup["k"]) for s2 in "rl"}, pelvis_drift=sup["drift"], dt=dt, trace=(np.asarray(trace) if record else None), trace_dt=10*dt,
             act_lag=(None if LG is None else dict(tau=LG["tau_lag"], comp=LG["comp"], tau_hat=LG["tau_hat"], pred_outer=PRED_OUTER, wcap=LG["wcap"], w_max=LG["w_max"], n_lagged=len(LG["a"]), n_capped=LG["n_capped"], envelope_joints=int(LG["has_om"].sum()))))
    return r


# ------------------------------------------------------------------------------------------------ actuator lag (Q1, opt-in)
ACT_LAG = float(os.environ.get("HC_LAG", 0.0))     # s; 0 = the T1/T2 plant (no lag, no droop), unchanged default
LAG_WCAP = float(os.environ.get("HC_LAG_WCAP", 0.0))   # 0 = servo gains as built; c > 0: every lagged servo's natural frequency capped at c/tau
# LAG-AWARE STAND (K2, 2026-10-18), opt-in: LAG_COMP = "inv" -> every lagged servo runs an internal copy of the known first-order
# lag on its own commands (act_hat, exact: the droop clip acts after the lag) and commands  cmd = act_hat + (tau/tau_c)(u_d - act_hat),
# u_d = kp (tg - q) - kv qdot, clipped to +-tau_max, so the delivered torque follows u_d with time constant tau_c (slew-bounded by the
# rating: |d act/dt| <= 2 tau_max/tau). GAIN RULE (from the plant, no per-load tuning): each servo keeps its as-built natural
# frequency w_i = sqrt(kp_i/I_i) capped at W_EXPL (the explicit-evaluation bound, 400 rad/s, Q1 sanity check), critically damped,
# and tau_c,i = C_INV / w_i (C_INV = 0.5: the third-order loop I tau_c s^3 + I s^2 + kv s + kp has Routh bound w tau_c < 2, so 4x margin);
# Overdrive tau/tau_c >= 1 (no compensation if tau_c >= tau). Discrete gain per step dt w/C_INV <= 0.8 at 1 ms (stable, no overshoot).
# "pred": the sprint's constant-rate state prediction only (PD on q + tau qdot, gains as above, no inversion) for comparison.
LAG_COMP = os.environ.get("HC_LAG_COMP", "none")
W_EXPL = float(os.environ.get("HC_WEXPL", 400.0)); C_INV = float(os.environ.get("HC_CINV", 0.5))
TAU_HAT_SCALE = float(os.environ.get("HC_TAU_HAT", 1.0))   # K2 robustness: the controller's lag model = TAU_HAT_SCALE x the plant lag (mismatch test)
PRED_OUTER = os.environ.get("HC_PRED_OUTER", "1") != "0"   # pred/pp: the stand's balance loop (capture point, trunk) also uses the predicted state
PLACE = os.environ.get("HC_PLACE", "1") != "0"     # 0: UNLOADED stand test (the object stays parked 2.5 m away; support gate still measured)


def _lag_setup(m, S, tau_lag, wcap):
    """ACTUATOR LAG MODE (Q1, 2026-10-17). Every body actuator (all 29 non-finger joints: legs, waist, arms; the Dex3 finger servos
    stay ideal, a listed simulation choice) keeps its position-servo LAW (the same kp, kv, target), but the servo is evaluated
    explicitly every physics step and its output torque goes through the project's first-order actuator lag (branch_i's exact
    discretisation alpha = 1 - exp(-dt/tau)) and then the torque-speed envelope with droop that the torque mode uses (driving
    torque <= tau_max (1 - |w|/om), braking <= tau_max; om = vendor rated speed, known for the 19 'driven' joints; the back-EMF
    brake above om is unchanged). wcap > 0 additionally caps each lagged servo's natural frequency at wcap/tau (critically damped,
    I_eff from the keyframe mass matrix): a CONTROLLER gain choice, recorded; no hardware value changes."""
    if not tau_lag or tau_lag <= 0: return None
    import branch_i as BI
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a][0])) or "" for a in range(m.nu)]
    L = np.array([a for a in range(m.nu) if "hand" not in names[a]])
    dof = np.array([int(m.jnt_dofadr[int(m.actuator_trnid[a][0])]) for a in L]); qad = np.array([int(m.jnt_qposadr[int(m.actuator_trnid[a][0])]) for a in L])
    kp = np.array(m.actuator_gainprm[L, 0], float); kv = -np.array(m.actuator_biasprm[L, 2], float); tq = np.array(m.actuator_forcerange[L, 1], float)
    ieff = np.array([1.0/float(S["Minv_key"][k, k]) for k in dof]); w_max = (wcap/tau_lag) if wcap and wcap > 0 else np.inf
    if LAG_COMP in ("inv", "pred", "pp"): w_max = W_EXPL      # K2 lag-aware stand: the c/tau cap is replaced by compensation
    w = np.sqrt(kp/ieff); cap = w > w_max; n_capped = int(cap.sum())
    if n_capped: kp = np.where(cap, w_max**2*ieff, kp); kv = np.where(cap, 2.0*np.sqrt(kp*ieff), kv)
    if LAG_COMP == "pp":   # K2 equal-decay pole placement on the lagged joint (I tau s^3 + I s^2 + (kv + kp tau) s + kp, PD on q + tau qdot):
        p_ = 1.0/(3.0*tau_lag); kv = 2.0*kp*tau_lag + 2.0*ieff*p_*p_*tau_lag   # all three poles at Re = -1/(3 tau), kp kept
    J = S["jarr"]; om = np.full(len(L), np.inf); jom = dict(zip([int(a) for a in J["aid"]], np.asarray(J["om"], float)))
    for i, a in enumerate(L):
        if int(a) in jom: om[i] = jom[int(a)]
    lo_all = np.array(m.actuator_ctrlrange[:, 0]); hi_all = np.array(m.actuator_ctrlrange[:, 1])
    saved = {k: np.array(getattr(m, k)[L]) for k in ("actuator_gaintype", "actuator_gainprm", "actuator_biastype", "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange", "actuator_ctrllimited", "actuator_forcelimited")}
    dt = m.opt.timestep
    return dict(a=L, dof=dof, qad=qad, kp=kp, kv=kv, tq=tq, om=om, has_om=np.isfinite(om), lo=lo_all[L], hi=hi_all[L], lo_all=lo_all, hi_all=hi_all,
                alpha=1.0 - np.exp(-dt/tau_lag), act=None, saved=saved, comp=LAG_COMP, tau_hat=float(tau_lag)*TAU_HAT_SCALE, odrive=np.maximum(1.0, tau_lag*np.sqrt(kp/ieff)/C_INV), w_srv=np.sqrt(kp/ieff), tau_lag=float(tau_lag), wcap=float(wcap or 0.0), w_max=float(w_max), n_capped=n_capped, BI=BI)


def _lag_motors(m, LG):
    for i, a in enumerate(LG["a"]): LG["BI"]._as_motor(m, int(a), float(LG["tq"][i]))


def _lag_restore(m, LG):
    for k, v in LG["saved"].items(): getattr(m, k)[LG["a"]] = v


def _lag_step(m, d, LG, tgt):
    """One physics step of the lagged servo torque path: PD (explicit) -> first-order lag -> envelope with droop -> motor ctrl."""
    q = d.qpos[LG["qad"]]; w = d.qvel[LG["dof"]]; tg = np.clip(tgt[LG["a"]], LG["lo"], LG["hi"]); tq = LG["tq"]
    comp = LG.get("comp", "none")
    if comp in ("pred", "pp"): q = q + LG["tau_hat"]*w            # sprint-style: feedback on the state predicted one lag ahead
    cmd = np.clip(LG["kp"]*(tg - q) - LG["kv"]*w, -tq, tq)
    if comp == "inv" and LG["act"] is not None:          # internal lag model: drive act_hat toward u_d with time constant tau_c
        cmd = np.clip(LG["act"] + LG["odrive"]*(cmd - LG["act"]), -tq, tq)
    if LG["act"] is None: LG["act"] = cmd.copy()
    else: LG["act"] += LG["alpha"]*(cmd - LG["act"])
    frac = np.where(LG["has_om"], np.clip(1.0 - np.abs(w)/np.where(LG["has_om"], LG["om"], 1.0), 0.0, 1.0), 1.0)
    hi = np.where(w > 0, tq*frac, tq); lo = np.where(w < 0, tq*frac, tq)
    d.ctrl[LG["a"]] = np.clip(LG["act"], -lo, hi)


def _under(m, g, root):
    b = int(m.geom_bodyid[g])
    while b > 0:
        if b == root: return True
        b = int(m.body_parentid[b])
    return False


def law_torques(m, d, strategy, bid, tb, c_loc, h_loc, mass, Iobj, wn, zeta, sN, Nmin, wb, aid, dofs, tm, jp, jr):
    """The stabilisation law (module doc, item 2): object error -> wrench -> capacity-weighted palm allocation + squeeze -> J^T."""
    Rt = d.xmat[tb].reshape(3, 3); c = np.array(d.xipos[bid]); Ro = d.xmat[bid].reshape(3, 3)
    vo = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, vo, 0); wo, vlin = vo[:3], vo[3:]
    vt = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, tb, vt, 0)
    if strategy == "head":
        ref = d.xpos[tb] + Rt @ h_loc; vref = vt[3:] + np.cross(vt[:3], ref - d.xpos[tb])
        e = np.r_[c[:2] - ref[:2], 0.0]; ev = np.r_[vlin[:2] - vref[:2], 0.0]; F = -mass*(wn*wn*e + 2*zeta*wn*ev)
    else:
        ref = d.xpos[tb] + Rt @ c_loc; vref = vt[3:] + np.cross(vt[:3], ref - d.xpos[tb])
        e = c - ref; ev = vlin - vref; F = -mass*(wn*wn*e + 2*zeta*wn*ev) + np.array([0, 0, mass*G])
    # tilt: rotation vector taking the object's axes to the torso-yaw upright frame
    yaw = np.arctan2(Rt[1, 0], Rt[0, 0]); Ry = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    Re = Ro @ Ry.T; phi = 0.5*np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
    Iw = Ro @ np.diag(Iobj) @ Ro.T; M = -Iw @ (wn*wn*phi + 2*zeta*wn*(wo - vt[:3]))
    # capacity-weighted allocation: point FORCES at the two palms (a palm on a side cannot deliver a reliable moment), weighted
    # least squares  min ||A f - (F, M/L)||^2 + rho sum_j (tau_j/tau_max_j)^2,  L = the object's half height (moment units).
    # The moment about the palm line is not reachable by palm forces alone; the LS turns it into a push at the palm height
    # (above the COM) against the head's friction at the bottom: the pot's top is pushed back over the head.
    Bs = []; ps = []; nrm = {}
    for s in "rl":
        R = d.xmat[wb[s]].reshape(3, 3); p = d.xpos[wb[s]] + R @ PALM_C[s]; ps.append(p); nrm[s] = R @ PALM_N[s]
        mujoco.mj_jac(m, d, jp, None, p, wb[s]); Bs.append(jp[:, dofs[s]].T)   # 7x3: tau = B f
    L = max(0.05, float(np.max(m.geom_size[[g for g in range(m.ngeom) if m.geom_bodyid[g] == bid][0]])))
    Wt = np.r_[1.0/tm[aid["r"]]**2, 1.0/tm[aid["l"]]**2]; Bb = np.zeros((14, 6)); Bb[:7, :3] = Bs[0]; Bb[7:, 3:] = Bs[1]
    A = np.zeros((6, 6))
    for i, p in enumerate(ps):
        r_ = p - c; A[:3, 3*i:3*i+3] = np.eye(3); A[3:, 3*i:3*i+3] = np.array([[0, -r_[2], r_[1]], [r_[2], 0, -r_[0]], [-r_[1], r_[0], 0]])/L
    rho = 1e-2*(mass*G)**2*np.max(1.0/Wt)/14.0      # makes the torque cost comparable to a weight-sized force error
    Hm = A.T @ A + rho*(Bb.T @ (Wt[:, None]*Bb)); f_all = np.linalg.solve(Hm + 1e-9*np.eye(6), A.T @ np.r_[F, M/L])
    tau = np.zeros(m.nu); tau_s = []
    for i, s in enumerate("rl"):
        f = f_all[3*i:3*i+3]; nn = nrm[s]; ft = f - (f @ nn)*nn
        N = Nmin + sN*np.linalg.norm(ft)/MU
        tau_s.append(Bs[i] @ (f + N*nn))
        if os.environ.get("HC_DEBUG2") and int(round(d.time*1000)) % 50 == 0: print(f"t={d.time:.3f} {s} e={np.round(e,3)} phi={np.round(phi,3)} F={np.round(F,1)} M={np.round(M,2)} f={np.round(f,1)} N={N:.1f} tau={np.round(tau_s[-1],2)}")
    # saturation: scale the whole law torque uniformly (keeps the wrench direction) so no arm joint exceeds 0.9 tau_max
    ts = np.r_[tau_s[0], tau_s[1]]; k = min(1.0, 0.9/max(1e-9, float(np.max(np.abs(ts)*np.sqrt(Wt)))))
    for i, s in enumerate("rl"):
        for k_, a in enumerate(aid[s]): tau[a] = k*tau_s[i][k_]
    return tau


# ------------------------------------------------------------------------------------------------ search
def disturbance(sway):
    """sway spec -> (sway, push): None; ("sway", amp, f); ("push", F_frac) = lateral torso pushes of F_frac x robot weight for
    0.2 s at t = 3 (+y), 6 (-y), 9 s (+y)."""
    if sway is None: return None, None
    if sway[0] == "push":
        W = 9.80665*ROBOT_MASS; return None, [(3.0, 0.2, (0, sway[1]*W, 0)), (6.0, 0.2, (0, -sway[1]*W, 0)), (9.0, 0.2, (0, sway[1]*W, 0))]
    return (sway[1], sway[2]) if sway[0] == "sway" else tuple(sway), None


SCORE_SUPPORT = os.environ.get("HC_SCORE_SUPPORT", "1") != "0"
ROBOT_MASS = 34.394   # G1H robot bodies (measured from the model: sum of body_mass over S["robot_bodies"])


def evaluate(args):
    obj, mass, strategy, z, T, dt, sway = args
    try:
        sw, pu = disturbance(sway); r = rollout(obj, mass, strategy, z, T, dt, False, sw, push=pu)
    except Exception as ex:
        return dict(score=10.0, t_held=0.0, reason=f"ERROR {type(ex).__name__}: {ex}")
    pen = 0.5*max(0.0, (r.get("tilt_max_deg") or 0.0) - 20.0)/TILT_GATE
    if strategy == "head": pen += max(0.0, 0.5 - r["head_frac"])          # the head must carry at least half the weight
    if SCORE_SUPPORT:   # T1: the support gate inside the search score (0 inside the gate, then linear, capped)
        pen += 0.5*min(4.0, max(0.0, r.get("foot_travel_max", 1.0)/FOOT_TRAVEL_GATE - 1.0)) + 0.5*min(4.0, max(0.0, r.get("pelvis_drift", 1.0)/PELVIS_DRIFT_GATE - 1.0))
    r["score"] = -(r["t_held"]/T) + 0.3*r["util_rms"] + pen + 0.5*(r["e_mean"] or 0.0)/(HEAD_PATCH_R if strategy == "head" else GRIP_SLIP)*0.2
    r.pop("trace", None); return r


SUP_DT = 0.02        # s: support-gate sampling interval (sim time, the same at every dt)
LOAD_FRAC = 0.05     # a foot is LOADED when its floor normal force >= 5% of (robot + load) weight
FOOT_TRAVEL_GATE = 0.02   # m: total horizontal travel of each foot while loaded, over the hold
PELVIS_DRIFT_GATE = 0.05  # m: max horizontal pelvis displacement from its position at placement, over the hold
TILT_GATE = 45.0   # deg: an object spun in the grip / tipped on the head is not "held", even if its COM stayed in place


def gate(r, T, strategy, mass, support=True):
    ok = r["t_held"] >= T - 1e-6 and r["reason"] is None and (r.get("tilt_max_deg") or 0.0) <= TILT_GATE
    if strategy == "head": ok = ok and r["head_frac"] >= 0.5
    if support: ok = ok and support_ok(r)
    return bool(ok)


def support_ok(r):
    """T1 support gate (both strategies): no foot skates while loaded and the pelvis stays put. Results stored before the gate
    existed have no foot_travel field and FAIL it (they must be replayed)."""
    if r.get("foot_travel_max") is None or r.get("pelvis_drift") is None: return False
    return r["foot_travel_max"] <= FOOT_TRAVEL_GATE and r["pelvis_drift"] <= PELVIS_DRIFT_GATE
