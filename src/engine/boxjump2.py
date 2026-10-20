"""M3b BALANCED BOX JUMP (2026-10-20). Stand in front of a rigid box (toes TOE_GAP behind its near edge), jump onto it, land with
both feet on the top and hold balance standing for >= HOLD_T s. Stock G1 / stock H1, published ratings (engine.boxjump.rating_table),
every body actuator through a first-order actuator lag (default 10 ms), validated on fresh plants at 1 ms and 0.5 ms.

Construction (physics-derived; every constant is in SUPPLIED):
  geometry   the scene standard fixes the toe gap; the feet must travel d_f = gap + sole length + EDGE_MARGIN; the flight time T
             follows from the takeoff vz; the COM need only travel d_f - e, where e = vx sqrt(z_td/g) is the capture-point offset the
             feet are placed ahead of the COM in flight, so vx_needed = (d_f - e)/T  (predict()).
  push       whole-body kinematic reference (damped-LS IK: feet fixed, COM path, torso attitude) tracked by stiff lagged servos:
             counter-movement dip, then the COM is driven along the launch direction phi until the legs reach full extension.
  flight     per-leg IK in the world frame on the measured state: soles level, feet lifted to box_top + clear until they are past
             the edge (knees tucked), then placed at the predicted touchdown COM + e (capture point), ballistic COM prediction.
  landing    whole-body IK reference with the feet fixed where they landed: COM x follows the LIP capture trajectory
             x_ref = c + (x0 - c + (v0 + lx (x0 - c)) t) e^{-lx t} toward the sole centre c, plus capture-point feedback on the
             state predicted one lag ahead; COM z absorbs (critically damped) then rises to stance height; torso returns upright.
  servos     every body actuator: explicit PD on the state predicted one lag ahead (q + tau qdot) -> first-order lag tau ->
             torque-speed envelope (driving <= tau_max (1 - |w|/om)) -> rated torque clip; back-EMF brake above om (all joints).
Gates: see gates in rollout(). Robust = nominal valid at both dt and >= 3/4 fresh perturbations valid at both dt."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
import branch_i as BI, plant as P
from engine import boxjump as BJ, core

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); OUT = f"{ROOT}/results/movements/M3b_box_jump"
G = 9.80665
TOE_GAP = 0.075          # m, declared scene standard: toes 5-10 cm behind the near edge (midpoint)
EDGE_MARGIN = 0.02       # m: every sole geom's heel end past the edge
HOLD_T = 3.0             # s: balance hold window (standing)
SETTLE_MAX = 1.6         # s after touchdown by which the hold window opens (absorb + stand-up)
T_PRE = 0.3              # s of quiet stance before the dip
LAG = float(os.environ.get("BJ2_LAG", "0.010"))
W_SRV = float(os.environ.get("BJ2_W", "0"))   # 0 -> servo natural frequency = 2/LAG (critic / Q1 convention), else explicit
STANCE_CAP = 100.0      # stance effective inertia <= 100 x free-floating (G1 ankle roll 600x chattered at 1 ms; discrete stability)
DEBUG = os.environ.get("BJ2_DEBUG", "0") == "1"
PUSH = os.environ.get("BJ2_PUSH", "bang")
YIELD = os.environ.get("BJ2_YIELD", "0") == "1"; YIELD_D = float(os.environ.get("BJ2_YIELD_D", "0.015"))
CTRL_DT = 0.002          # controller tick (500 Hz), targets zero-order held
FOOT_TRAVEL_MAX = 0.02; PELVIS_DRIFT_MAX = 0.05; OVERSPEED_TOL = BI.OVERSPEED_TOL; LIM_TOL = 0.05
SUPPLIED = dict(
    scene="box 0.8 m deep x 1.2 m wide, static rigid, near edge TOE_GAP=0.075 m ahead of the standing toes (human box-jump practice "
          "5-10 cm); G1 box contact pairs = stock floor pairs replicated (engine.boxjump.scene)",
    actuators="all body actuators lagged (first-order tau, default 10 ms), explicit PD on q + tau qdot, natural frequency w = 2/tau "
              "(critically damped on I_eff = 1/Minv_ii at the keyframe), torque-speed envelope with spec omega_max, published ratings",
    hold=f"hold window {HOLD_T} s opening by touchdown + {SETTLE_MAX} s (absorb + stand-up); foot travel <= 0.02 m, pelvis drift <= 0.05 m",
    placement="feet placed at predicted touchdown COM x + k_e vx sqrt(z_td/g), bounded so every heel clears the edge + margin",
    search="CMA-ES seed >= 1 over the 14 controller parameters PAR (bounded), objective = worse of 1 ms and 0.5 ms fresh plants, "
           "+ a takeoff-vx term pulling vx toward the predicted minimum",
)


# ------------------------------------------------------------------ prediction (no simulation)
TAKEOFF_FRAC = 0.85      # predicted fraction of the spec takeoff-speed ceiling reached by the lagged servo push (stated before search)
ZTD_FRAC = 0.90          # COM height above the soles at touchdown / stance COM height (legs slightly bent to absorb)


def geometry(robot):
    g = BJ.stance_geometry(robot); L = g["toe_x"] - g["heel_x"]
    return dict(g, sole_len=L, D=TOE_GAP + g["foot_front"], d_feet=TOE_GAP + L + EDGE_MARGIN)


def predict(robot, h, v_to=None):
    geo = geometry(robot); vc, _ = BJ.v_ceiling(robot); v = v_to or TAKEOFF_FRAC*vc
    z_st = geo["com_z_key"]; z_to = z_st + BJ.RISE_TO[robot][0]; z_td = ZTD_FRAC*z_st; w = np.sqrt(G/z_td)
    dz = h + z_td - z_to; off0 = geo["com_x"] - geo["foot_centre_x"]
    e_max = {"G1": 0.14, "H1": 0.18}[robot]       # half the horizontal leg reach at touchdown height (model geometry, see PREDICTION)
    best = None
    for vx in np.linspace(0.02, v - 0.05, 2000):
        vz = np.sqrt(v*v - vx*vx); disc = vz*vz - 2*G*dz
        if disc < 0: continue
        T = (vz + np.sqrt(disc))/G; e = min(vx/w, e_max)
        need = geo["d_feet"] + off0 - e          # COM travel when the feet are placed e ahead of the COM at touchdown
        if vx*T >= need: best = dict(vx=float(vx), vz=float(vz), T=float(T), e=float(e)); break
    nop = None
    for vx in np.linspace(0.02, v - 0.05, 2000):
        vz = np.sqrt(v*v - vx*vx); disc = vz*vz - 2*G*dz
        if disc < 0: continue
        T = (vz + np.sqrt(disc))/G
        if vx*T >= geo["d_feet"] + off0: nop = dict(vx=float(vx), T=float(T)); break
    half = 0.5*geo["sole_len"] - 0.03
    cap = dict(no_place=float(w*(geo["toe_x"] - geo["com_x"] - 0.03)), placed=float(w*(e_max + half)), cp_tol=float(w*half))
    return dict(robot=robot, h=h, v_to=v, z_to=z_to, z_td=z_td, w_td=float(w), dz=dz, d_feet=geo["d_feet"], D=geo["D"], e_max=e_max,
                with_placement=best, no_placement=nop, landing_capacity=cap,
                feasible=bool(best is not None and best["vx"] <= cap["placed"]))


# ------------------------------------------------------------------ robot model helpers
def _spec_omega(robot):
    import yaml
    sp = yaml.safe_load(open(f"{ROOT}/spec/{robot.lower()}.yaml")); return {k: float(r["omega_max"]["value"]) for k, r in sp["joints"].items() if "omega_max" in r}


class Body:
    """Index sets and kinematics scratch for a plant."""
    def __init__(self, S):
        m = S["m"]; self.m = m; self.S = S; robot = S["robot"]; self.robot = robot; self.kd = mujoco.MjData(m)
        om = _spec_omega(robot); self.nv = m.nv
        self.aid = np.arange(m.nu); jid = m.actuator_trnid[:, 0]
        self.qad = np.array([m.jnt_qposadr[j] for j in jid]); self.dof = np.array([m.jnt_dofadr[j] for j in jid])
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in jid]; self.names = names
        self.tau = np.array(m.actuator_forcerange[:, 1], float)
        self.om = np.array([om.get(n, np.inf) for n in names])
        self.ieff = np.array([1.0/float(S["Minv_key"][k, k]) for k in self.dof])         # free-floating (flight)
        self.fb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in BI.ROBOTS[robot]["foot_bodies"]]
        self.tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.leg = []
        for b in self.fb:
            dl = []; x = b
            while x > 1:
                for j in range(m.body_jntadr[x], m.body_jntadr[x] + m.body_jntnum[x]): dl.append(int(m.jnt_dofadr[j]))
                x = m.body_parentid[x]
            self.leg.append(np.array(sorted(dl)))
        self.legdofs = np.concatenate(self.leg)
        self.upper = [k for k in range(6, m.nv) if k not in set(self.legdofs.tolist())]
        self.armdofs = set(int(self.dof[i]) for i, n in enumerate(self.names) if any(w in n for w in ("shoulder", "elbow", "wrist")))     # waist + arms: posture-regularised in the IK
        self.lo = np.full(m.nq, -np.inf); self.hi = np.full(m.nq, np.inf)
        for j in range(m.njnt):
            if m.jnt_limited[j] and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                self.lo[m.jnt_qposadr[j]] = m.jnt_range[j][0] + 0.01; self.hi[m.jnt_qposadr[j]] = m.jnt_range[j][1] - 0.01
        self.dof2q = {int(m.jnt_dofadr[j]): int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}
        self.fgeoms = {fb: [g for g in S["foot_geoms"] if int(m.geom_bodyid[g]) == fb and (m.geom_contype[g] or m.geom_conaffinity[g] or robot == "G1") and
                            (robot != "G1" or (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith(("foot1_collision", "foot2_collision", "foot3_collision")))] for fb in self.fb}
        S["fbodies"] = self.fb; S["fgeoms"] = self.fgeoms
        # stance references (settled pose)
        d = self.kd; d.qpos[:] = S["q_settled"]; self.fwd(d)
        # STANCE effective inertia: diagonal of the joint-space inverse inertia with both soles welded (contact-constrained),
        # so a stance joint's servo sees the body it actually moves (the ankle carries the whole body, not just the foot)
        dd = mujoco.MjData(m); dd.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, dd); M = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, M, dd.qM); Mi = np.linalg.inv(M)
        Js = []
        for fb in self.fb:
            jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv)); mujoco.mj_jacBody(m, dd, jp, jr, fb); Js += [jp, jr]
        Jc = np.vstack(Js); Mc = Mi - Mi@Jc.T@np.linalg.solve(Jc@Mi@Jc.T, Jc@Mi)
        self.ieff_st = np.array([1.0/max(float(Mc[k, k]), 1e-9) for k in self.dof]); self.ieff_st = np.clip(self.ieff_st, self.ieff, STANCE_CAP*self.ieff)
        self.q0 = np.array(S["q_settled"]); self.foot_q0 = [d.xquat[b].copy() for b in self.fb]; self.torso_q0 = d.xquat[self.tb].copy()
        fi = BJ._foot_info(m, d, S)
        self.sole_dz = [float(d.xpos[b][2] - f[0]) for b, f in zip(self.fb, fi)]           # ankle-body origin above its sole bottom
        self.heel_off = [float(d.xpos[b][0] - f[1]) for b, f in zip(self.fb, fi)]          # ankle x - heel x
        self.toe_off = [float(f[2] - d.xpos[b][0]) for b, f in zip(self.fb, fi)]           # toe x - ankle x
        self.y0 = [float(d.xpos[b][1]) for b in self.fb]
        self.z_stand = float(d.subtree_com[1][2]); self.com0 = d.subtree_com[1].copy()
        self.fc0 = float(np.mean([0.5*(f[1] + f[2]) for f in fi]))

    def fwd(self, d):
        mujoco.mj_kinematics(self.m, d); mujoco.mj_comPos(self.m, d)

    def clamp(self, q):
        return np.clip(q, self.lo, self.hi)


def _rotvec(qt, qc):
    """World-frame rotation vector taking orientation qc to qt."""
    inv = np.zeros(4); mujoco.mju_negQuat(inv, qc); dq = np.zeros(4); mujoco.mju_mulQuat(dq, qt, inv); r = np.zeros(3); mujoco.mju_quat2Vel(r, dq, 1.0); return r


def ik_whole(B, q, feet, com, torso_q, post_w=1.0, n_it=3, q_post=None, waist_w=1.0, w_feet=1e4, w_com=10.0, w_torso=2.0, lam=1e-3):
    """Damped-LS whole-body IK from q (free base): feet = [(pos, quat)] for both feet bodies, com = target, torso_q = target quat."""
    m = B.m; d = B.kd; nv = m.nv; q = q.copy()
    jp = np.zeros((3, nv)); jr = np.zeros((3, nv))
    for _ in range(n_it):
        d.qpos[:] = q; B.fwd(d)
        A = lam*np.eye(nv); b = np.zeros(nv)
        for fb, (pt, qt) in zip(B.fb, feet):
            mujoco.mj_jacBody(m, d, jp, jr, fb)
            e = pt - d.xpos[fb]; A += w_feet*jp.T@jp; b += w_feet*jp.T@e
            e = _rotvec(qt, d.xquat[fb]); A += w_feet*jr.T@jr; b += w_feet*jr.T@e
        mujoco.mj_jacSubtreeCom(m, d, jp, 1); e = com - d.subtree_com[1]; A += w_com*jp.T@jp; b += w_com*jp.T@e
        mujoco.mj_jacBody(m, d, jp, jr, B.tb); e = _rotvec(torso_q, d.xquat[B.tb]); A += w_torso*jr.T@jr; b += w_torso*jr.T@e
        # posture regulariser on all hinge dofs (toward the stance pose)
        qp = B.q0 if q_post is None else q_post
        for k in B.upper: qa = B.dof2q[k]; wk = post_w*(1e3 if k in B.armdofs else waist_w); A[k, k] += wk; b[k] += wk*(qp[qa] - q[qa])
        dq = np.linalg.solve(A, b)
        # active set: hinge dofs at a limit that the step would push further are frozen and the step re-solved (keeps the
        # reference consistent with flat feet when e.g. the ankle reaches its dorsiflexion limit)
        frz = [k for k, qa in B.dof2q.items() if (q[qa] <= B.lo[qa] + 1e-4 and dq[k] < 0) or (q[qa] >= B.hi[qa] - 1e-4 and dq[k] > 0)]
        if frz:
            A[frz, :] = 0; A[:, frz] = 0; A[frz, frz] = 1.0; b[frz] = 0; dq = np.linalg.solve(A, b)
        n = np.linalg.norm(dq)
        if n > 0.3: dq *= 0.3/n
        mujoco.mj_integratePos(m, q, dq, 1.0); q = B.clamp(q)
    return q


def ik_legs(B, q_meas, qref, feet, n_it=3, lam=1e-3, w_rot=0.3):
    """Per-leg damped-LS IK on the measured base pose: foot body pos + level orientation targets (world). Returns qref (legs edited)."""
    m = B.m; d = B.kd; q = q_meas.copy(); q[7:] = qref[7:]; jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    for _ in range(n_it):
        d.qpos[:] = q; B.fwd(d); dq = np.zeros(m.nv)
        for i, (fb, (pt, qt)) in enumerate(zip(B.fb, feet)):
            mujoco.mj_jacBody(m, d, jp, jr, fb); L = B.leg[i]
            Jp = jp[:, L]; Jr = jr[:, L]; ep = pt - d.xpos[fb]; er = _rotvec(qt, d.xquat[fb])
            A = Jp.T@Jp + w_rot*Jr.T@Jr + lam*np.eye(len(L)); bb = Jp.T@ep + w_rot*Jr.T@er
            x = np.linalg.solve(A, bb); n = np.linalg.norm(x)
            if n > 0.3: x *= 0.3/n
            dq[L] = x
        mujoco.mj_integratePos(m, q, dq, 1.0); q = B.clamp(q)
    out = qref.copy(); out[7:] = q[7:]
    return out


def _mj(x, T):
    s = np.clip(x/T, 0.0, 1.0) if T > 0 else 1.0; return s*s*s*(10 - 15*s + 6*s*s)


def _quintic(t, x0, v0, x1, T):
    """Quintic from (x0, v0, a=0) to (x1, 0, 0) over T; constant after."""
    if t >= T: return x1
    D = x1 - x0; c3 = (10*D - 6*v0*T)/T**3; c4 = (-15*D + 8*v0*T)/T**4; c5 = (6*D - 3*v0*T)/T**5
    return x0 + v0*t + c3*t**3 + c4*t**4 + c5*t**5


def _quat_pitch(qbase, pitch):
    """qbase rotated by pitch about world y (positive = forward lean)."""
    qy = np.array([np.cos(pitch/2), 0.0, np.sin(pitch/2), 0.0]); out = np.zeros(4); mujoco.mju_mulQuat(out, qy, qbase); return out


def _pitch_of(qt, q0):
    r = _rotvec(qt, q0); return float(r[1])


# ------------------------------------------------------------------ controller parameters
PAR = [  # name, lo, hi, default
    ("depth", 0.08, 0.26, 0.14),       # dip depth of the COM (m)
    ("t_dip", 0.25, 0.70, 0.29),       # dip duration (s)
    ("lean_dip", -0.04, 0.08, 0.046),   # COM x shift in the dip (m, + forward)
    ("torso_dip", 0.0, 0.8, 0.65),     # torso forward pitch in the dip (rad)
    ("phi", 0.0, 0.8, 0.32),          # launch direction from vertical (rad)
    ("toe", 0.0, 0.5, 0.24),
    ("t_ramp", 0.0, 0.25, 0.0),
    ("t_tuck", 0.0, 0.25, 0.0),        # flight tuck blend-in time (s)        # launch-target ramp time (s; 0 = step)           # ankle plantarflexion added to the launch posture (rad)
    ("torso_push", -0.2, 0.4, -0.07),   # torso pitch of the launch posture (rad)
    ("arm", 0.0, 1.5, 0.86),           # arm swing amplitude (shoulder pitch, rad)
    ("k_e", 0.1, 1.2, 0.4),            # in-flight placement gain on the capture offset
    ("ztd", 0.80, 0.99, 0.88),     # touchdown COM height / stance COM height
    ("clear", 0.02, 0.10, 0.05),       # sole clearance over the box top before the edge (m)
    ("lx", 0.5, 2.0, 1.0),             # landing x-trajectory rate / w_td
    ("kcp", 0.0, 1.5, 0.5),            # capture-point feedback gain
    ("absorb", 0.02, 0.10, 0.05),      # COM absorb depth below touchdown height (m); absorb time = 2 absorb/|vz_td|
    ("t_up", 0.4, 1.2, 0.8),           # stand-up duration (s)
    ("w_land", 0.5, 1.0, 1.0),         # landing servo frequency / w (compliance during the absorb + hold)
    ("waist_fly", -0.4, 0.2, 0.0),
    ("waist_push", -0.5, 0.2, -0.2),   # waist-pitch target offset in the dip and push (pre-positions the weak 35 N m waist away from its forward stop)    # trunk-pitch (waist) target offset in flight (rad; - = trunk extension)
]
PNAMES = [p[0] for p in PAR]


def par_from(u):
    u = np.clip(np.asarray(u, float), -1, 1)
    return {n: lo + (hi - lo)*0.5*(x + 1) for (n, lo, hi, _), x in zip(PAR, u)}


def u_from_par(par):
    return np.array([float(np.clip(2*(par.get(n, d) - lo)/(hi - lo) - 1, -1, 1)) for (n, lo, hi, d) in PAR])


def u_default():
    return np.array([2*(dflt - lo)/(hi - lo) - 1 for (_, lo, hi, dflt) in PAR])


# ------------------------------------------------------------------ rollout
def make_plant(robot, h, dt, box_dx=0.0, mu_scale=1.0, start_noise=None):
    D = geometry(robot)["D"]; S = BJ.plant(robot, h, D, dt, box_dx=box_dx, mu_scale=mu_scale, start_noise=start_noise); return S


def rollout(S, p, record=False, mode="jump", drop_vx=0.0, drop_h=0.15, lag=None, push=None, T_after=None):
    """mode 'jump': full box jump from the settled stance. mode 'drop': landing-capacity probe - the settled body is placed
    drop_h above the box top, centred so its feet land 0.25 m past the edge, with COM velocity (drop_vx, 0, 0), flight controller on."""
    m = S["m"]; d = mujoco.MjData(m); dt = m.opt.timestep; h = S["box_h"]; edge = S["edge_x"]; robot = S["robot"]
    lag = LAG if lag is None else lag
    push_only = mode == "push"; mode = "jump" if push_only else mode
    B = S.get("_B") or Body(S); S["_B"] = B
    nu = m.nu; tq = B.tau; om = B.om; has_om = np.isfinite(om)
    w = (W_SRV or (2.0/lag if lag > 0 else 200.0))*np.ones(nu)
    wl = w*p.get("w_land", 1.0)
    KP = dict(fly=(w*w*B.ieff, 2.0*w*B.ieff), stance=(w*w*B.ieff_st, 2.0*w*B.ieff_st), land=(wl*wl*B.ieff_st, 2.0*wl*B.ieff_st))     # gains follow the contact state
    kp, kv = KP["fly" if mode == "drop" else "stance"]
    saved = {k: np.array(getattr(m, k)) for k in ("actuator_gaintype", "actuator_gainprm", "actuator_biastype", "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange", "actuator_ctrllimited", "actuator_forcelimited")}
    for a in range(nu): BI._as_motor(m, a, float(tq[a]))
    alpha = 1.0 - np.exp(-dt/lag) if lag > 0 else 1.0
    q0 = B.q0.copy(); d.qpos[:] = q0; d.qvel[:] = 0
    z_st = B.z_stand; z_td = p["ztd"]*z_st; w_td = np.sqrt(G/z_td)
    if mode == "drop":
        sh = (edge + 0.25) - B.fc0; d.qpos[0] += sh; d.qpos[2] += h + drop_h; d.qvel[0] = drop_vx
    mujoco.mj_forward(m, d)
    # gravity-holding torque as the lag state's start (the body starts at rest, held)
    act = np.zeros(nu); start_rest = True
    if mode == "jump":
        # PRE-SETTLE (not part of the trial): 0.6 s of quiet stance under this rollout's own lagged servo law (targets = the settled
        # stance), so the lag state holds the stance torques at t = 0; the start gate then requires max |qdot| < 0.05 rad/s
        mujoco.mj_inverse(m, d); act = np.clip(d.qfrc_inverse[B.dof], -tq, tq)
        for _ in range(int(0.6/dt)):
            qm = d.qpos[B.qad]; wq = d.qvel[B.dof]; act += alpha*(np.clip(kp*(q0[B.qad] - (qm + lag*wq)) - kv*wq, -tq, tq) - act)
            d.ctrl[:] = act; mujoco.mj_step(m, d)
        start_rest = float(np.max(np.abs(d.qvel[6:]))) < 0.05; d.time = 0.0
    feet0 = [(d.xpos[b].copy(), B.foot_q0[i].copy()) for i, b in enumerate(B.fb)]
    fl, bx = S["floor"], S["box"]; foot = set(S["foot_geoms"])
    qref = d.qpos.copy(); tgt = qref[B.qad].copy()
    st = dict(phase="pre" if mode == "jump" else "flight", t_phase=0.0, takeoff=None, takeoff_t=None, touch_t=None, touch_on=None, td=None,
              nonfoot_box=False, front_face=False, floor_after=False, overspeed=0.0, lim_exc=0.0, fell=False, hops=0, air_run=0.0,
              cp_margin=None, flight_ctl=None, nonfoot_geoms=set())
    c_dip0 = d.subtree_com[1].copy(); c_push0 = None; v_push = np.zeros(3); land = None; win = None
    shoulders = [i for i, n in enumerate(B.names) if "shoulder_pitch" in n]; ankles = [i for i, n in enumerate(B.names) if "ankle" in n and "roll" not in n]; q_launch = None; q_fly = None; q_to_ref = None
    waist_p = next((i for i, n in enumerate(B.names) if n in ("waist_pitch_joint", "torso")), None)   # trunk pitch joint (H1 'torso' is yaw: excluded below)
    if waist_p is not None and abs(m.jnt_axis[m.actuator_trnid[waist_p, 0]][1]) < 0.9: waist_p = None
    arm0 = q0[B.qad[shoulders]].copy() if shoulders else None
    t_end = None; Tmax = (T_PRE + p["t_dip"] + 0.6 + 1.2 + SETTLE_MAX + HOLD_T + 0.1) if mode == "jump" else (0.8 + SETTLE_MAX + HOLD_T + 0.1)
    if T_after: Tmax = T_after
    fz_last = 0.0; trace = [] if record else None; in_air_t = None; cand = None; kc_last = -1
    try:
        for s in range(int(Tmax/dt)):
            t = s*dt; kc = int(t/CTRL_DT + 1e-9)
            com = d.subtree_com[1]; vcom = P.com_velocity(m, d)
            if kc != kc_last:
                kc_last = kc; ph = st["phase"]
                if ph == "pre":
                    if t >= T_PRE: st["phase"] = ph = "dip"; st["t_phase"] = t
                    qref = q0.copy()
                if ph == "dip":
                    tt = t - st["t_phase"]; s_ = _mj(tt, p["t_dip"])
                    ct = c_dip0 + s_*np.array([p["lean_dip"], 0.0, -p["depth"]])
                    tq_ = _quat_pitch(B.torso_q0, s_*p["torso_dip"])
                    qp = B.q0.copy()
                    if arm0 is not None: qp[B.qad[shoulders]] = arm0 + s_*p["arm"]*0.6; qref[B.qad[shoulders]] = qp[B.qad[shoulders]]   # arms back (+ shoulder pitch)
                    if waist_p is not None: qp[B.qad[waist_p]] = np.clip(q0[B.qad[waist_p]] + s_*p.get("waist_push", 0.0), B.lo[B.qad[waist_p]], B.hi[B.qad[waist_p]]); qref[B.qad[waist_p]] = qp[B.qad[waist_p]]
                    qref = ik_whole(B, qref, feet0, ct, tq_, q_post=qp, waist_w=(1e3 if waist_p is not None and "waist_push" in p else 1.0))
                    if DEBUG and kc % 25 == 0: B.kd.qpos[:] = qref; B.fwd(B.kd); print(f"D t{tt:.3f} x{com[0]:+.3f}/{ct[0]:+.3f}/{B.kd.subtree_com[1][0]:+.3f} z{com[2]:.3f}/{ct[2]:.3f}/{B.kd.subtree_com[1][2]:.3f} vx{vcom[0]:+.2f} pit{_pitch_of(d.xquat[B.tb], B.torso_q0):+.2f}/{s_*p['torso_dip']:.2f} ank{d.qpos[B.qad[4]]:+.2f}/{qref[B.qad[4]]:+.2f} fp{_pitch_of(d.xquat[B.fb[0]], B.foot_q0[0]):+.3f} ankF{d.actuator_force[4]:+.0f}")
                    if tt >= p["t_dip"]: st["phase"] = "push"; st["t_phase"] = t; c_push0 = ct.copy()
                elif ph == "push":
                    tt = t - st["t_phase"]
                    if PUSH == "bang":
                        # BANG push: the servo targets jump to the launch posture (IK: COM at stance height + 3 cm, displaced along the
                        # launch direction phi, torso upright) + ankle plantarflexion 'toe'; the stiff servos saturate (max effort)
                        if q_launch is None:
                            cl = c_push0 + np.array([np.tan(p["phi"])*(B.z_stand + 0.03 - c_push0[2]), 0.0, B.z_stand + 0.03 - c_push0[2]])
                            q_launch = qref.copy()
                            qp = B.q0.copy()
                            if arm0 is not None: qp[B.qad[shoulders]] = arm0 - p["arm"]; q_launch[B.qad[shoulders]] = qp[B.qad[shoulders]]
                            if waist_p is not None: qp[B.qad[waist_p]] = np.clip(q0[B.qad[waist_p]] + p.get("waist_push", 0.0), B.lo[B.qad[waist_p]], B.hi[B.qad[waist_p]]); q_launch[B.qad[waist_p]] = qp[B.qad[waist_p]]
                            for _ in range(5): q_launch = ik_whole(B, q_launch, feet0, cl, _quat_pitch(B.torso_q0, p["torso_push"]), q_post=qp, waist_w=(1e3 if waist_p is not None and "waist_push" in p else 1.0))
                            for i_ in ankles: q_launch[B.qad[i_]] = min(q_launch[B.qad[i_]] + p["toe"], B.hi[B.qad[i_]])
                            q_push0 = qref.copy()
                        qref = q_push0 + _mj(tt, p.get("t_ramp", 0.0))*(q_launch - q_push0); ct = c_push0     # launch targets ramped in over t_ramp (min-jerk)
                        if DEBUG and tt == 0: B.kd.qpos[:] = q_launch; B.fwd(B.kd); fi_ = BJ._foot_info(m, B.kd, S); print("LAUNCH com", B.kd.subtree_com[1].round(3), "cl", cl.round(3), "toe", round(max(f[2] for f in fi_), 3), "torso pitch", round(_pitch_of(B.kd.xquat[B.tb], B.torso_q0), 2), "knee", round(q_launch[B.qad[3]], 2), "hip", round(q_launch[B.qad[0]], 2), "ank", round(q_launch[B.qad[4]], 2))
                    else:
                        u = np.array([np.sin(p["phi"]), 0.0, np.cos(p["phi"])])
                        ct = c_push0 + u*0.5*p["acc"]*tt*tt
                        tq_ = _quat_pitch(B.torso_q0, max(0.0, 1 - tt/0.25)*p["torso_dip"])
                        qref = ik_whole(B, qref, feet0, ct, tq_)
                    if arm0 is not None: qref[B.qad[shoulders]] = arm0 + p["arm"]*(0.6 - 1.6*_mj(tt, 0.25))
                    if DEBUG and kc % 10 == 0: B.kd.qpos[:] = qref; B.fwd(B.kd); print(f"P t{tt:.3f} x{com[0]:+.3f}/{ct[0]:+.3f}/{B.kd.subtree_com[1][0]:+.3f} z{com[2]:.3f}/{ct[2]:.3f}/{B.kd.subtree_com[1][2]:.3f} vx{vcom[0]:+.2f} vz{vcom[2]:+.2f} fz{fz_last:.0f} pit{_pitch_of(d.xquat[B.tb], B.torso_q0):+.2f} knee{d.qpos[B.qad[3]]:.2f}/{qref[B.qad[3]]:.2f}")
                    if tt > 0.6: st["phase"] = "flight_wait"
                elif ph in ("flight", "flight_wait") and st["takeoff_t"] is not None or (mode == "drop" and ph == "flight"):
                    st["phase"] = "flight"; kp, kv = KP["fly"]
                    # ballistic prediction from the state one lag ahead
                    xc, zc = float(com[0] + lag*vcom[0]), float(com[2] + lag*vcom[2]); vx, vz = float(vcom[0]), float(vcom[2] - lag*G)
                    a_ = 0.5*G; zt = h + z_td; disc = vz*vz + 2*G*(zc - zt)
                    T_rem = (vz + np.sqrt(disc))/G if disc > 0 else 0.0
                    x_td = xc + vx*max(T_rem, 0.0); e = p["k_e"]*vx/w_td
                    feet = []
                    for i, fb in enumerate(B.fb):
                        x_fc = x_td + e                                                # sole-centre target (x)
                        x_fc = max(x_fc, edge + EDGE_MARGIN + 0.01 + 0.5*(B.heel_off[i] + B.toe_off[i]))
                        x_ank = x_fc - 0.5*(B.toe_off[i] - B.heel_off[i])
                        sole_z = d.xpos[fb][2] - B.sole_dz[i]; toe_x = d.xpos[fb][0] + B.toe_off[i]; heel_x = d.xpos[fb][0] - B.heel_off[i]
                        past = heel_x > edge + 0.005
                        z_land = zc - z_td + B.sole_dz[i] + (com[2] - zc)            # ankle z if the leg holds the touchdown geometry now
                        z_ank = max(z_land, h + p["clear"] + B.sole_dz[i]) if not past else max(z_land, h + B.sole_dz[i] - 0.01)
                        if (not past) and sole_z < h + 0.5*p["clear"]: x_ank = min(x_ank, edge - 0.01 - B.toe_off[i])   # lift before crossing
                        feet.append((np.array([x_ank, B.y0[i] + (com[1] - B.com0[1]), z_ank]), B.foot_q0[i]))
                    q_ik = ik_legs(B, d.qpos, qref if q_fly is None else q_fly, feet)
                    if q_to_ref is None: q_to_ref = qref.copy()
                    q_fly = q_ik; be = _mj(t - st["takeoff_t"], p.get("t_tuck", 0.0)) if mode == "jump" else 1.0
                    qref = q_to_ref + be*(q_ik - q_to_ref)          # tuck blended in over t_tuck (limits the hip-flexion reaction on the 35 N m waist)
                    for k in B.upper:
                        if k not in B.armdofs: qref[B.dof2q[k]] = q0[B.dof2q[k]]          # waist to its stance (mid-range) posture in flight
                    if waist_p is not None: qref[B.qad[waist_p]] = np.clip(q0[B.qad[waist_p]] + p.get("waist_push", 0.0) + p.get("waist_fly", 0.0), B.lo[B.qad[waist_p]], B.hi[B.qad[waist_p]])   # trunk extension in flight (counter-rotates the legs forward)
                    if arm0 is not None: qref[B.qad[shoulders]] = arm0 - 0.3*p["arm"]
                    st["flight_ctl"] = dict(T_rem=T_rem, e=e, x_td=x_td)
                elif ph == "land":
                    tt = t - land["t0"]; c = land["c"]; lx = land["lx"]
                    x0, v0 = land["x0"], land["v0"]; ex = np.exp(-lx*tt)
                    xr = c[0] + (x0 - c[0] + (v0 + lx*(x0 - c[0]))*tt)*ex; vr = (v0 - lx*(v0 + lx*(x0 - c[0]))*tt)*ex
                    xp = float(com[0] + lag*vcom[0]); cp = xp + float(vcom[0])/land["w"]; cpr = xr + vr/land["w"]
                    xcmd = xr - p["kcp"]*(cp - cpr)
                    yr = c[1] - p["kcp"]*((com[1] + lag*vcom[1] + vcom[1]/land["w"]) - c[1])
                    zr = _quintic(tt, land["z0"], land["vz0"], land["zl"], land["Ta"])           # absorb: matched to the touchdown z, vz
                    if YIELD: zr = max(zr, float(com[2] + lag*vcom[2]) - YIELD_D)   # yield: never pull the COM down faster than it falls (no leg retraction off the top)
                    s_up = _mj(tt - land["Ta"] - 0.1, p["t_up"]); zr = (1 - s_up)*zr + s_up*(h + z_st)
                    tq_ = _quat_pitch(B.torso_q0, (1 - _mj(tt, 0.8))*land["pitch0"])
                    qref = ik_whole(B, qref, land["feet"], np.array([xcmd, yr, zr]), tq_, n_it=2)
                    if DEBUG and kc % 10 == 0: B.kd.qpos[:] = qref; B.fwd(B.kd); print(f"IK z{B.kd.subtree_com[1][2]-h:.3f} x{B.kd.subtree_com[1][0]-c[0]:+.3f} base{qref[2]-h:.3f} act_base{d.qpos[2]-h:.3f} knee_ref{qref[B.qad[3]]:.2f} knee{d.qpos[B.qad[3]]:.2f}", end=" ")
                    if DEBUG and kc % 2 == 0 and tt < 0.25:
                        fi_ = BJ._foot_info(m, d, S); print(f"H t{tt:.3f} soles {fi_[0][0]-h:+.4f} {fi_[1][0]-h:+.4f} fz{fz_last:5.0f} z{com[2]-h:.3f} zr{zr-h:.3f} vz{vcom[2]:+.2f} pit{_pitch_of(d.xquat[B.tb], B.torso_q0):+.2f} wpit{d.qvel[4]:+.2f} kn{d.qpos[B.qad[3]]:.2f}/{qref[B.qad[3]]:.2f} hp{d.qpos[B.qad[0]]:+.2f}/{qref[B.qad[0]]:+.2f} an{d.qpos[B.qad[4]]:+.2f}/{qref[B.qad[4]]:+.2f} fpit{_pitch_of(d.xquat[B.fb[0]], B.foot_q0[0]):+.2f} tqk{d.ctrl[3]:+.0f} tqh{d.ctrl[0]:+.0f} tqa{d.ctrl[4]:+.0f}")
                    if DEBUG and kc % 10 == 0: print(f"fp{_pitch_of(d.xquat[B.fb[0]], B.foot_q0[0]):+.3f} ank{d.actuator_force[4]:+.1f}/{d.ctrl[4]:+.1f} aq{d.qpos[B.qad[4]]:+.3f}/{qref[B.qad[4]]:+.3f}", end=" ")
                    if DEBUG and kc % 10 == 0: print(f"L t{tt:.3f} x{com[0]-c[0]:+.3f} xr{xr-c[0]:+.3f} xc{xcmd-c[0]:+.3f} z{com[2]-h:.3f} zr{zr-h:.3f} vz{vcom[2]:+.2f} vx{vcom[0]:+.2f} fz{fz_last:.0f} pit{_pitch_of(d.xquat[B.tb], B.torso_q0):+.2f}")
                    if arm0 is not None: qref[B.qad[shoulders]] = arm0
                tgt = qref[B.qad].copy()
            # lagged servos
            qm = d.qpos[B.qad]; wq = d.qvel[B.dof]
            u_ = np.clip(kp*(tgt - (qm + lag*wq)) - kv*wq, -tq, tq)
            act += alpha*(u_ - act)
            frac = np.where(has_om, np.clip(1.0 - np.abs(wq)/np.where(has_om, om, 1.0), 0.0, 1.0), 1.0)
            d.ctrl[:] = np.clip(act, -np.where(wq < 0, tq*frac, tq), np.where(wq > 0, tq*frac, tq))
            over = np.where(has_om, np.clip((np.abs(wq) - np.where(has_om, om, 1))/(0.1*np.where(has_om, om, 1)), 0.0, 1.0), 0.0)
            d.qfrc_applied[:] = 0; d.qfrc_applied[B.dof] = -tq*np.sign(wq)*over
            if push is not None and push[0] <= t - (st["touch_t"] or 1e9) < push[0] + push[1]: d.xfrc_applied[B.tb, :3] = push[2]
            else: d.xfrc_applied[B.tb, :3] = 0
            r_ = np.where(has_om, np.abs(wq)/np.where(has_om, om, 1), 0.0); st["overspeed"] = max(st["overspeed"], float(r_.max()))
            mujoco.mj_step(m, d)
            # contacts
            fz = 0.0; on_box = [False, False]; fz_box = 0.0
            for ci in range(d.ncon):
                c_ = d.contact[ci]; g1, g2 = c_.geom1, c_.geom2
                sup = fl if fl in (g1, g2) else (bx if bx in (g1, g2) else None)
                if sup is None: continue
                other = g2 if g1 == sup else g1; fv = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fv); fz += fv[0]
                if sup == bx:
                    fz_box += fv[0]
                    if other not in foot: st["nonfoot_box"] = True; st["nonfoot_geoms"].add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other) or str(other))
                    else:
                        if c_.pos[2] < h - 0.008: st["front_face"] = True
                        on_box[0 if int(m.geom_bodyid[other]) == B.fb[0] else 1] = True
                elif st["takeoff_t"] is not None or mode == "drop":
                    st["floor_after"] = True
            com = d.subtree_com[1]; vcom = P.com_velocity(m, d); fz_last = fz
            ex_ = np.maximum(B.lo[B.qad] - d.qpos[B.qad], d.qpos[B.qad] - B.hi[B.qad]) - 0.01
            if ex_.max() > st["lim_exc"]: st["lim_exc"] = float(ex_.max()); st["lim_where"] = (B.names[int(np.argmax(ex_))], st["phase"], round(t, 3))
            # takeoff detection (sustained 20 ms without support force, rising)
            if mode == "jump" and st["takeoff_t"] is None and st["phase"] in ("push", "flight_wait"):
                if fz < 1.0 and vcom[2] > 0:
                    if in_air_t is None:
                        mujoco.mj_subtreeVel(m, d)
                        in_air_t = t; cand = dict(vx=float(vcom[0]), vz=float(vcom[2]), z=float(com[2]), x=float(com[0]), Ly=float(d.subtree_angmom[1][1]),
                                                  pitch=_pitch_of(d.xquat[B.tb], B.torso_q0), wy=float(d.qvel[4]))
                    elif t - in_air_t >= BI.FLIGHT_CONFIRM:
                        st["takeoff_t"] = in_air_t; st["takeoff"] = cand; st["phase"] = "flight"
                        if push_only: break
                else: in_air_t = None
                if st["phase"] == "flight_wait" and t - st["t_phase"] > 1.2: break      # never left the ground
            # touchdown
            if st["phase"] == "flight" and fz >= 1.0 and (mode == "drop" or st["takeoff_t"] is not None):
                st["touch_t"] = t; st["touch_on"] = "box" if fz_box > 0 else "floor"
                fi = BJ._foot_info(m, d, S); c = np.array([np.mean([0.5*(f[1] + f[2]) for f in fi]), np.mean([f[3][1] for f in fi]), 0.0])
                xcp = float(com[0] + vcom[0]/w_td)
                st["td"] = dict(vx=float(vcom[0]), vz=float(vcom[2]), x_com=float(com[0]), z_com_rel=float(com[2] - h), feet_centre=float(c[0]),
                                cp_minus_centre=float(xcp - c[0]), cp_to_toe=float(max(f[2] for f in fi) - xcp), heel_margin=float(min(f[1] for f in fi) - edge),
                                torso_pitch=_pitch_of(d.xquat[B.tb], B.torso_q0))
                feet_l = []
                for i, fb in enumerate(B.fb):
                    pz = h + B.sole_dz[i]; feet_l.append((np.array([d.xpos[fb][0], d.xpos[fb][1], pz]), B.foot_q0[i]))
                zl = float(com[2]) - p["absorb"]; Ta = max(0.08, 2.0*p["absorb"]/max(0.3, abs(float(vcom[2]))))
                land = dict(zl=zl, Ta=Ta, t0=t, c=c, x0=float(com[0]), v0=float(vcom[0]), z0=float(com[2]), vz0=float(vcom[2]), w=w_td, lx=p["lx"]*w_td,
                            feet=feet_l, pitch0=_pitch_of(d.xquat[B.tb], B.torso_q0))
                qref = d.qpos.copy(); st["phase"] = "land"; kp, kv = KP["land"]
                t_end = t + SETTLE_MAX + HOLD_T if st["touch_on"] == "box" else t + 0.05
            if st["phase"] == "land":
                if fz < 1.0: st["air_run"] += dt
                else:
                    if st["air_run"] >= BI.FLIGHT_CONFIRM: st["hops"] += 1; st.setdefault("hop_at", []).append((round(t - st["touch_t"], 3), round(st["air_run"], 3)))
                    st["air_run"] = 0.0
                if win is None and t >= st["touch_t"] + SETTLE_MAX:
                    fi = BJ._foot_info(m, d, S); win = dict(t0=t, feet0=[f[3] for f in fi], pel0=d.xpos[1].copy(), ft=0.0, pd=0.0, up_min=1.0, zmin=9.0)
                if win is not None:
                    fi = BJ._foot_info(m, d, S)
                    win["ft"] = max(win["ft"], max(float(np.linalg.norm(f[3] - p0)) for f, p0 in zip(fi, win["feet0"])))
                    win["pd"] = max(win["pd"], float(np.linalg.norm(d.xpos[1][:2] - win["pel0"][:2])))
                    win["up_min"] = min(win["up_min"], float(d.xmat[B.tb][8])); win["zmin"] = min(win["zmin"], float(com[2]))
            if d.xmat[B.tb][8] < 0.3 or com[2] < 0.25: st["fell"] = True; break
            if record and s % max(1, int(0.01/dt)) == 0: trace.append(d.qpos.copy())
            if t_end is not None and t >= t_end: break
    finally:
        for k, v in saved.items(): getattr(m, k)[:] = v
    fi = BJ._foot_info(m, d, S); heel_margin = min(f[1] for f in fi) - edge; up = float(d.xmat[B.tb][8])
    com = d.subtree_com[1]; vcom = P.com_velocity(m, d)
    gates = dict(start_rest=start_rest, took_off=(mode == "drop" or st["takeoff_t"] is not None), touched_box=st["touch_on"] == "box",
                 no_floor_after=not st["floor_after"], no_nonfoot_box=not st["nonfoot_box"], no_front_face=not st["front_face"],
                 not_fell=not st["fell"], held=win is not None and (t - win["t0"]) >= HOLD_T - 2*dt,
                 both_feet_on_top=all(on_box), feet_past_edge=heel_margin >= 0.0,
                 com_over_box=S["edge_x"] + 0.02 <= com[0] <= S["box_far_x"] - 0.02,
                 upright=up >= 0.85 and (win is None or win["up_min"] >= 0.8),
                 standing=win is not None and win["zmin"] >= h + 0.85*B.z_stand, no_hop=st["hops"] == 0,
                 settled=float(np.linalg.norm(vcom)) < 0.1,
                 support=win is not None and win["ft"] <= FOOT_TRAVEL_MAX and win["pd"] <= PELVIS_DRIFT_MAX,
                 envelope=st["overspeed"] <= OVERSPEED_TOL, limits=st["lim_exc"] <= LIM_TOL)
    if mode == "drop": gates.pop("start_rest"); gates.pop("took_off")
    if push_only: return dict(takeoff=st["takeoff"], takeoff_t=st["takeoff_t"], overspeed=st["overspeed"], lim_exc=st["lim_exc"], fell=st["fell"], nonfoot=bool(st["nonfoot_box"]))
    out = dict(valid=bool(all(gates.values())), gates=gates, failed=[k for k, v in gates.items() if not v], takeoff=st["takeoff"], takeoff_t=st["takeoff_t"],
               touch_t=st["touch_t"], touch_on=st["touch_on"], td=st["td"], heel_margin=float(heel_margin), hops=st["hops"],
               overspeed=st["overspeed"], lim_exc=st["lim_exc"], up_end=up, com_end=[float(x) for x in com], v_end=float(np.linalg.norm(vcom)),
               window=(None if win is None else dict(foot_travel=win["ft"], pelvis_drift=win["pd"], up_min=win["up_min"], com_z_min_rel=win["zmin"] - h, T=t - win["t0"])),
               nonfoot_geoms=sorted(st["nonfoot_geoms"]), lim_where=st.get("lim_where"), hop_at=st.get("hop_at"), flight_ctl=st["flight_ctl"], t_end=t, phase_end=st["phase"], lag=lag, dt=dt)
    if record: out["trace"] = np.array(trace)
    return out


ROBUST = os.environ.get("BJ2_ROBUST", "0") == "1"
TRAIN = [(0.0005, dict(box_dx=0.007, mu_scale=0.9)), (0.001, dict(start_sigma=0.002, seed=99, box_dx=-0.007)),     # search-time training perturbations
         (0.0005, dict(start_sigma=0.002, seed=98))]                                                              # (disjoint from the validation set)
if os.environ.get("BJ2_TRAIN2", "0") == "1":     # v2 training set: + controller-parameter noise (seeds 201/202; validation uses 11/12)
    TRAIN = TRAIN + [(0.0005, dict(u_sigma=0.03, useed=201)), (0.0005, dict(u_sigma=0.03, useed=202)), (0.001, dict(mu_scale=0.85, box_dx=0.01))]
PERTURB = [dict(name="P1_param_noise_s11", u_sigma=0.03, seed=11), dict(name="P2_param_noise_s12", u_sigma=0.03, seed=12),
           dict(name="P3_start_noise_2mrad_box+1cm", start_sigma=0.002, seed=13, box_dx=0.01),
           dict(name="P4_mu0.8_box-1cm", mu_scale=0.8, box_dx=-0.01)]


def score(o, robot, h):
    """Lower is better. Valid < 0. Invalid ranked: takeoff, reach (heel margin), contact cleanliness, fall, hold, details."""
    g = o["gates"]; to = o["takeoff"]
    vxt = 0.0
    if to is not None:
        pr = predict(robot, h, v_to=float(np.hypot(to["vx"], to["vz"])))["with_placement"]
        vxt = abs(to["vx"] - pr["vx"]) if pr else 0.5
    if o["valid"]:
        w = o["window"]; return -10.0 + 0.5*vxt + 5*w["foot_travel"] + 2*w["pelvis_drift"] + 20.0*max(0.0, o["lim_exc"] - 0.03) + 2.0*max(0.0, 0.08 - o["td"]["cp_to_toe"])
    if not g.get("took_off", True): return 200.0
    p = 100.0 + 0.5*vxt
    p += 100.0*max(0.0, EDGE_MARGIN - o["heel_margin"]) if o["touch_t"] is not None else 30.0
    p += 5.0*(not g["touched_box"]) + 5.0*(not g["no_nonfoot_box"]) + 3.0*(not g["no_front_face"]) + 3.0*(not g["no_floor_after"])
    if o["td"] is not None: p += 10.0*abs(o["td"]["cp_minus_centre"])        # graded: capture point at touchdown vs the sole centre
    if not g["not_fell"] and o["touch_t"] is not None: p += 8.0*(1.0 - min(1.0, (o["t_end"] - o["touch_t"])/(SETTLE_MAX + HOLD_T)))   # graded: survival time
    p += 4.0*(not g["not_fell"]) + 4.0*(not g["held"]) + 1.0*sum(not g[k] for k in ("both_feet_on_top", "feet_past_edge", "com_over_box", "upright", "standing", "settled", "support", "start_rest"))
    p += (2.0 + 50.0*sum(x[1] for x in (o["hop_at"] or []))) if not g["no_hop"] else 0.0
    p += (2.0 + 200.0*max(0.0, o["lim_exc"] - LIM_TOL)) if not g["limits"] else 0.0
    p += (2.0 + 10.0*max(0.0, o["overspeed"] - OVERSPEED_TOL)) if not g["envelope"] else 0.0
    if o["window"]: p += 10.0*max(0.0, o["window"]["foot_travel"] - FOOT_TRAVEL_MAX) + 10.0*max(0.0, o["window"]["pelvis_drift"] - PELVIS_DRIFT_MAX)
    return p


_CTX = {}
def _eval(a):
    robot, h, u, dt, kw = a
    key = (robot, h, dt, json.dumps(kw, sort_keys=True))
    if key not in _CTX:
        sn = None
        if kw.get("start_sigma"):
            nq = len(core.standard_jump_clip(robot)["start"]); sn = np.zeros(nq); sn[7:] = np.random.default_rng(kw["seed"]).normal(0, kw["start_sigma"], nq - 7)
        _CTX[key] = make_plant(robot, h, dt, box_dx=kw.get("box_dx", 0.0), mu_scale=kw.get("mu_scale", 1.0), start_noise=sn)
    o = rollout(_CTX[key], par_from(u)); return _clean(o)


def search(robot, h, seed, iters, wall, tag, x0=None, sigma=float(os.environ.get("BJ2_SIGMA", "0.08")), pop=int(os.environ.get("BJ2_POP", "12")), workers=12):
    import cma
    from multiprocessing import Pool
    od = f"{OUT}/{tag}"; os.makedirs(od, exist_ok=True); t0 = time.time()
    x0 = u_default() if x0 is None else np.asarray(x0, float)
    es = cma.CMAEvolutionStrategy(x0, sigma, dict(bounds=[-1, 1], popsize=pop, seed=seed, verbose=-9))
    best = dict(f=1e9); hist = []
    with Pool(workers) as pool:
        for g in range(iters):
            X = es.ask(); conds = [(0.001, {}), (0.0005, {})] + (TRAIN if ROBUST else []); nc = len(conds)
            jobs = [(robot, h, (np.clip(x + np.random.default_rng(kw["useed"]).normal(0, kw["u_sigma"], len(x)), -1, 1) if "u_sigma" in kw else x), dt,
                     {k: v for k, v in kw.items() if k not in ("u_sigma", "useed")}) for x in X for (dt, kw) in conds]
            R = pool.map(_eval, jobs); F = []
            for i, x in enumerate(X):
                a, b = R[nc*i], R[nc*i + 1]; sc = [score(R[nc*i + j], robot, h) for j in range(nc)]
                f = max(sc[:2]) + (0.5*sum(max(0.0, c_ - max(sc[:2])) for c_ in sc[2:]) if nc > 2 else 0.0); F.append(f)   # robust: nominal worst + training-perturbation excess
                if f < best["f"]: best = dict(f=f, u=list(map(float, x)), par=par_from(x), m1=a, m05=b, gen=g, secs=time.time() - t0)
            es.tell(X, F); hist.append(dict(gen=g, fmin=float(min(F)), fmed=float(np.median(F)), best=best["f"], secs=time.time() - t0))
            to = best["m1"]["takeoff"] or {}
            print(f"[{tag}] gen {g} fmin {min(F):.2f} best {best['f']:.2f} valid1 {best['m1']['valid']} valid05 {best['m05']['valid']} "
                  f"failed {sorted(set(best['m1']['failed']) | set(best['m05']['failed']))} vx {to.get('vx', 0):.2f} vz {to.get('vz', 0):.2f} t {time.time() - t0:.0f}s", flush=True)
            json.dump(dict(robot=robot, h=h, seed=seed, tag=tag, best=best, hist=hist, lag=LAG, supplied=SUPPLIED, par_def=PAR), open(f"{od}/search_h{h:.2f}.json", "w"), indent=1, default=float)
            if time.time() - t0 > wall: break
    return best


def validate(robot, h, u, tag, workers=10):
    from multiprocessing import Pool
    u = np.asarray(u, float); jobs = [(robot, h, u, dt, {}) for dt in (0.001, 0.0005)]
    for pz in PERTURB:
        uu = np.clip(u + np.random.default_rng(pz["seed"]).normal(0, pz["u_sigma"], u.shape), -1, 1) if pz.get("u_sigma") else u
        kw = {k: pz[k] for k in ("start_sigma", "seed", "box_dx", "mu_scale") if k in pz and not pz.get("u_sigma")}
        jobs += [(robot, h, uu, dt, kw) for dt in (0.001, 0.0005)]
    with Pool(workers) as pool: R = pool.map(_eval, jobs)
    out = dict(robot=robot, h=h, tag=tag, lag=LAG, u=list(map(float, u)), par=par_from(u),
               nominal=dict(valid_both=bool(R[0]["valid"] and R[1]["valid"]), m1=R[0], m05=R[1]), perturb=[])
    for i, pz in enumerate(PERTURB):
        a, b = R[2 + 2*i], R[3 + 2*i]
        out["perturb"].append(dict(name=pz["name"], valid_both=bool(a["valid"] and b["valid"]), failed_1ms=a["failed"], failed_05ms=b["failed"], m1=a, m05=b))
    out["n_pass"] = sum(r["valid_both"] for r in out["perturb"]); out["robust"] = bool(out["nominal"]["valid_both"] and out["n_pass"] >= 3)
    return out


def _clean(o):
    return json.loads(json.dumps({k: v for k, v in o.items() if k != "trace"}, default=lambda x: x.tolist() if hasattr(x, "tolist") else (sorted(x) if isinstance(x, set) else float(x))))


def _cap_job(a):
    robot, vx, dt, lag, drop_h, pov = a
    S = make_plant(robot, 0.20, dt); p = par_from(u_default()); p.update(pov or {})
    o = rollout(S, p, mode="drop", drop_vx=vx, drop_h=drop_h, lag=lag); return dict(robot=robot, vx=vx, dt=dt, lag=lag, drop_h=drop_h, o=_clean(o))


def landing_capacity(robot, lag=LAG, drop_h=0.15, grid=None, workers=8, pov=None):
    from multiprocessing import Pool
    grid = grid or [round(x, 2) for x in np.arange(0.0, 1.0001, 0.1)]
    jobs = [(robot, v, dt, lag, drop_h, pov) for v in grid for dt in (0.001, 0.0005)]
    with Pool(workers) as pool: res = pool.map(_cap_job, jobs)
    return res


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "search":       # search ROBOT H SEED ITERS WALL TAG [WARM_JSON]
        robot, h, seed, iters, wall, tag = sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), sys.argv[7]
        x0 = None
        if len(sys.argv) > 8:
            w_ = json.load(open(sys.argv[8])); x0 = u_from_par(w_["best"]["par"] if "best" in w_ else w_["par"])
        b = search(robot, h, seed, iters, wall, tag, x0=x0, workers=int(os.environ.get("BJ2_WORKERS", "12")))
        v = validate(robot, h, b["u"], tag); json.dump(v, open(f"{OUT}/{tag}/validation_h{h:.2f}.json", "w"), indent=1, default=float)
        print(f"[{tag}] VALIDATION nominal both={v['nominal']['valid_both']} perturb {v['n_pass']}/4 robust={v['robust']}", flush=True)
    elif cmd == "validate":   # validate ROBOT H SEARCHJSON TAG
        robot, h, f_, tag = sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5]; w_ = json.load(open(f_)); u = w_["best"]["u"] if "best" in w_ else w_["u"]
        os.makedirs(f"{OUT}/{tag}", exist_ok=True); v = validate(robot, h, u, tag); json.dump(v, open(f"{OUT}/{tag}/validation_h{h:.2f}.json", "w"), indent=1, default=float)
        print(f"[{tag}] VALIDATION nominal both={v['nominal']['valid_both']} {v['nominal']['m1']['failed']} {v['nominal']['m05']['failed']} perturb {v['n_pass']}/4 robust={v['robust']}", flush=True)
        for r in v["perturb"]: print("  ", r["name"], r["valid_both"], r["failed_1ms"], r["failed_05ms"])
    elif cmd == "capacity":     # capacity ROBOT LAG DROP_H OUTFILE
        robot, lag, dh, outf = sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
        t0 = time.time(); res = landing_capacity(robot, lag, dh, workers=int(os.environ.get("BJ2_WORKERS", "8")))
        for r in res: print(f"{robot} lag {lag*1000:.0f} ms vx {r['vx']:.2f} dt {r['dt']}: valid={r['o']['valid']} {r['o']['failed']} hops={r['o']['hop_at']}", flush=True)
        json.dump(dict(robot=robot, lag=lag, drop_h=dh, par=par_from(u_default()), secs=time.time() - t0, rows=res), open(outf, "w"), indent=1)
    if cmd == "predict":
        rows = [predict(r, h) for r in ("G1", "H1") for h in (0.20, 0.30, 0.40)]
        print(json.dumps(rows, indent=1, default=float))
