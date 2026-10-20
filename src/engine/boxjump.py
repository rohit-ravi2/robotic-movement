"""M3 BOX JUMP (2026-10-18): jump onto a rigid box of height h whose near edge is D ahead of the stance foot centre, land with
both feet on the box top and hold balance for 2 s. Stock G1 and stock H1, published vendor ratings.

Construction (body + objective + support contract -> scaffold -> search -> gates), building on the jump of Phases 9-14:
  objective  = box (h, D): COM must travel D_x = D + FOOT_BACK + EDGE_MARGIN forward and the soles must be above h there.
  scaffold   = the J4 ascent prior (engine.strategy.ascent_prior) with effector = whole-body COM and d = the PREDICTED takeoff
               direction (the optimal launch angle of the safety parabola for (h, D), computed from the spec in predict());
               d = +z reproduces the vertical jump prior. Tracker = mocap_baseline.Tracker with the reference-tuned omega.
  search     = CMA-ES over the jump's torque knots (types x nk, the JumpTask parameterisation) + time scale + ONE flight-bend
               scalar of the Phase-14 landing controller (pre_bend; blended out over t_down after touchdown, BoxLanding).
  landing    = the Phase-14 balance/landing controller (balance.BalanceController, tuned parameters per robot) with the support
               signal = floor OR box contact (terrain-aware: the capture point is referenced to the feet wherever they are).
  gates      = landed (both feet in contact with the box TOP, no floor contact, COM over the box top, upright, held 2.0 s with
               foot travel <= 0.02 m and pelvis drift <= 0.05 m), no non-foot contact with the box (shin/knee/thigh/hands/torso),
               no foot contact on the box's front face, actuator torque-speed envelope (push-off) and force range (servo landing),
               overspeed <= 1.1 x rated in every phase, joint-limit excursion <= 0.02 rad, start at rest; at 1 ms AND 0.5 ms on
               FRESH plants; robustness = >= 3/4 perturbations valid at both dt.
Ratings ('published'): min(MJCF actuatorfrcrange, Unitree URDF effort, the vendor's published joint maximum). G1: hip pitch/yaw 88,
hip roll 88 (MJCF 139), knee 120 (MJCF/URDF 139; unitree.com/g1 EDU max 120), ankle pitch/roll 35 (MJCF 50), waist roll/pitch 35
(MJCF 50), waist yaw 88, shoulders/elbow/wrist roll 25, wrist pitch/yaw 5. H1: URDF effort = MJCF (hips/torso 200, knee 300,
ankle 40, shoulder pitch/roll 40, shoulder yaw/elbow 18); no lower published H1 figure is used.
No geometry, mass, joint range or rated speed is changed. Every choice is in SUPPLIED."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
import branch_i as BI, mocap_baseline as MB, balance as B, plant as P
from engine import core

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); OUT = f"{ROOT}/results/movements/M3_box_jump"; MODELS = f"{OUT}/models"
G = 9.80665
EDGE_MARGIN = 0.02        # m: the heel-side end of each sole must be this far past the near edge
HOLD_T = 2.0              # s: the hold window
SETTLE_T = 0.6            # s after touchdown before the hold window opens (absorb transient)
H_MAX = 4.6               # s: rollout cap
KNOT_DT = 0.05            # the jump's knot spacing (jump13.KNOT_DT)
ACT_TAU = 0.040           # push-off torque path: the project's first-order actuator lag (branch_i), unchanged
FOOT_TRAVEL_MAX = 0.02; PELVIS_DRIFT_MAX = 0.05; LIM_TOL = 0.05; OVERSPEED_TOL = BI.OVERSPEED_TOL
VENDOR_MAX = {"G1": 120.0}
SUPPLIED = dict(
    task="box near edge at D ahead of the stance foot centre (midpoint of the two sole centres); box 0.8 m deep x 1.2 m wide, static, rigid",
    box_contact="G1 (contype 0 everywhere, explicit pairs): every stock scene_mjx floor pair replicated to the box: feet solref 0.008 friction 1, and the stock 17 non-foot floor pairs (hands, wrists, elbows, shoulders, hips, thighs, shins, pelvis, torso, head; condim 3, default solref 0.02); the two linkage braces (no floor pair) get the foot-pair parameters (solref 0.008). Knee/shin/hand contact with the edge is physical and detected. H1: box geom contype 1 (collides with all 33 collision geoms); H1 contact solref = the project's H1 override 0.004 on every geom (MuJoCo default 0.02 checked separately, labelled)",
    ratings="published: min(MJCF, Unitree URDF effort, vendor published joint max 120 N m on G1 EDU); applied to driven torque motors, landing servos and held servos",
    lag="push-off torque path: first-order 40 ms actuator lag (the project lag, as every jump since Phase 8) + torque-speed envelope + back-EMF brake; landing: implicit position servos at the Phase-14 tuned stiffness, 500 Hz zero-order-hold targets, NO actuator lag (a 10 ms lag replay of the landing is a separate labelled check)",
    prior="the jump's J4 ascent prior unchanged (effector = COM, d = +z, COM xy held in the wind-up); the horizontal displacement is produced by the search (torque knots). A tilted-d prior (d along the predicted launch angle) was built and rejected before any search: its wind-up is truncated by the static certificate at 9 steps (9 mm crouch) because -d drives the COM backward out of the support",
    search="CMA-ES (seed >= 1), popsize 16, torque knots (types x 22) + tscale + flight pre_bend in [0, 0.6]; objective = worse of the 1 ms and 0.5 ms fresh plants (0.5 ms evaluated only when 1 ms is valid)",
    landing="Phase-14 BalanceController, tuned parameters per robot (results/phase14/balance_tuning*.json) except pre_bend (searched); flight bend blended to 0 over t_down after touchdown; support signal = floor or box contact",
    gates_v7="envelope gate = push-off overspeed (project OVERSPEED convention); all-phase overspeed and joint-limit penetration <= 0.05 rad reported as STRICT gates (searched with a small penalty)",
    gates=f"both feet on box top at window end, no floor contact after takeoff, COM xy over the box top, upright cos >= 0.85, COM >= box + 0.6 x stance COM height, |v_COM| < 0.3 m/s at end; hold {HOLD_T} s from touchdown + {SETTLE_T} s with foot travel <= {FOOT_TRAVEL_MAX} m and pelvis drift <= {PELVIS_DRIFT_MAX} m; no non-foot box contact; no foot contact below the top face; overspeed <= {OVERSPEED_TOL}; joint-limit excursion <= {LIM_TOL} rad; start qdot < 0.05",
    dt="optimised and validated at 1 ms and 0.5 ms on FRESH plants (no cache), 50 solver iterations",
    step_r2="R2 capture-step landing (BJ_STEP=1, opt-in): after touchdown, if cp = x_com + vx/w0 passes the sole front + step_margin within 0.3 s, the less-loaded foot swings (min-jerk path, sine lift step_lift over step_T, damped-LS IK on its own leg chain at 100 Hz, Newton step <= 0.2 rad, reach <= 0.95 x hip-to-sole length) to the LIP-predicted cp at swing touchdown (CoP reference = stance foot centre + step_cop x toe) + step_beyond; stance ankle error clipped to the sole; at swing touchdown (load > 1 N after step_T/2) the leg holds its landed angles; another capture step with the trailing foot if the cp is still past the new toe (<= 3 steps), else a STAGGERED HOLD: both legs at their landed angles, crouch schedule frozen, trunk reference = its pitch at hold start, cp regulated to its hold-start position clipped between the feet with k_cp x hold_kcp. step_T, step_lift, step_margin, step_ktr (trunk gain factor while stepping), step_beyond, step_cop, hold_kcp and the stiff/k_cp/k_trunk/crouch factors are tuned on the generic flat-floor forward-drop contract (both dt, no box): results/movements/M3_box_jump/r2/tune_step*",
)


# ------------------------------------------------------------------ ratings
_RT = {}
def rating_table(robot):
    if robot not in _RT:
        import yaml
        sp = yaml.safe_load(open(f"{ROOT}/spec/{robot.lower()}.yaml"))
        t = {}
        for jn, rec in sp["joints"].items():
            v = [float(rec[k]["value"]) for k in ("tau_max_peak", "tau_max_continuous") if k in rec]
            if v: t[jn] = min(v + ([VENDOR_MAX[robot]] if robot in VENDOR_MAX else []))
        _RT[robot] = t
    return _RT[robot]


def apply_ratings(S, robot):
    """Torque ceiling per actuator = the published rating (documented-rating choice, no hardware change)."""
    m = S["m"]; rt = rating_table(robot)
    def r_of(aid, cur):
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[aid][0])); return min(cur, rt.get(jn, cur))
    J = S["jarr"]; J["tau"] = np.array([r_of(a, t) for a, t in zip(J["aid"], J["tau"])], float)
    S["tmap"] = {t: [(jid, dof, qad, r_of(aid, tau), om, sgn, aid) for (jid, dof, qad, tau, om, sgn, aid) in v] for t, v in S["tmap"].items()}
    S["held"] = [(a, dof, qad, r_of(a, tau), kp, kv) for (a, dof, qad, tau, kp, kv) in S["held"]]
    for a in range(m.nu):                       # the model's own force ranges (used by every servo mode)
        jid = int(m.actuator_trnid[a][0]); jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jn in rt:
            lim = min(rt[jn], float(m.jnt_actfrcrange[jid][1]) or rt[jn]); m.jnt_actfrcrange[jid] = [-lim, lim]
            m.actuator_forcerange[a] = [-lim, lim]


# ------------------------------------------------------------------ scene
def stance_geometry(robot):
    """Foot centre x and heel/toe extents at the standard start (bare model)."""
    c = core.standard_jump_clip(robot); cfg = BI.ROBOTS[robot]; m = mujoco.MjModel.from_xml_path(cfg["mjcf"]); d = mujoco.MjData(m)
    d.qpos[:] = c["start"]; mujoco.mj_forward(m, d)
    fb = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in cfg["foot_bodies"]}
    xs = []
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) not in fb: continue
        if robot == "G1" and not (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith(("foot1_collision", "foot2_collision", "foot3_collision", "foot_box_collision")): continue
        if robot != "G1" and m.geom_contype[g] == 0: continue
        r = geom_extent(m, d.geom_xmat[g], g)[0]
        xs += [d.geom_xpos[g][0] - r, d.geom_xpos[g][0] + r]
    heel, toe = min(xs), max(xs); fc = 0.5*(heel + toe)
    return dict(foot_centre_x=fc, heel_x=heel, toe_x=toe, foot_back=fc - heel, foot_front=toe - fc,
                com_x=float(d.subtree_com[0][0]), com_z_key=float(d.subtree_com[0][2]))


def scene(robot, h, D, box_dx=0.0):
    """Base scene + a static rigid box; returns the MJCF path (cached in results/movements/M3_box_jump/models)."""
    os.makedirs(MODELS, exist_ok=True)
    geo = stance_geometry(robot); x_e = geo["foot_centre_x"] + D + box_dx
    path = f"{MODELS}/scene_{robot}_h{h:.3f}_D{D:.3f}_dx{box_dx:+.3f}.xml"
    if os.path.exists(path): return path, x_e
    base = BI.ROBOTS[robot]["mjcf"]; ms = mujoco.MjSpec.from_file(base)
    ms.meshdir = os.path.join(os.path.dirname(base), ms.meshdir or "assets") if not os.path.isabs(ms.meshdir or "") else ms.meshdir
    ms.texturedir = os.path.join(os.path.dirname(base), ms.texturedir) if (ms.texturedir and not os.path.isabs(ms.texturedir)) else ms.texturedir
    DX, DY = 0.40, 0.60
    g = ms.worldbody.add_geom(name="box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[DX, DY, h/2], pos=[x_e + DX, 0.0, h/2], rgba=[0.55, 0.45, 0.35, 1])
    if robot == "G1":
        g.contype = 0; g.conaffinity = 0
        fp = [p for p in ms.pairs if "floor" in (p.geomname1, p.geomname2)]
        ref = fp[0]
        for p in fp:
            rg = p.geomname1 if p.geomname2 == "floor" else p.geomname2
            q = ms.add_pair(name=f"{rg}_box", geomname1=rg, geomname2="box"); q.solref = list(p.solref); q.condim = int(p.condim); q.friction = list(p.friction)
        foot_names = {p.geomname1 if p.geomname2 == "floor" else p.geomname2 for p in fp}
        for gg in ms.geoms:
            n = gg.name or ""
            if n.endswith("_collision") and n not in foot_names and "foot_box" not in n:
                q = ms.add_pair(name=f"{n}_box", geomname1=n, geomname2="box"); q.solref = list(ref.solref); q.condim = int(ref.condim); q.friction = list(ref.friction)
    else:
        g.contype = 1; g.conaffinity = 1
    ms.compile(); open(path, "w").write(ms.to_xml())
    return path, x_e


def plant(robot, h, D, dt, box_dx=0.0, mu_scale=1.0, start_noise=None):
    """FRESH plant (no cache) with the box, published ratings, at dt, settled."""
    path, x_e = scene(robot, h, D, box_dx)
    c = core.standard_jump_clip(robot); start = np.array(c["start"], float)
    if start_noise is not None: start = start + start_noise
    cfg = BI.ROBOTS[robot]; old = cfg["mjcf"]; cfg["mjcf"] = path
    try: S = BI._build(start, {}, robot)
    finally: cfg["mjcf"] = old
    m = S["m"]; S["box"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box"); S["box_h"] = h; S["edge_x"] = x_e; S["box_far_x"] = x_e + 0.8
    if mu_scale != 1.0:
        m.geom_friction[:, 0] *= mu_scale; m.pair_friction[:, :2] *= mu_scale
    apply_ratings(S, robot)
    BI.apply_dt(S, dt); BI.settle(S)
    S["support"] = {S["floor"], S["box"]}
    geo = stance_geometry(robot); S["X_need"] = D + box_dx + geo["foot_back"] + EDGE_MARGIN
    return S


# ------------------------------------------------------------------ prediction
# Takeoff-speed ceiling inputs (measured in earlier phases, MJCF peak ratings) and the spec-derived rating correction.
V14 = {"G1": (3.25, "median takeoff of the 5 Phase-14 robust4 deep jumps landed at both dt (2.54-3.56 m/s), MJCF peak ratings"),
       "H1": (2.20, "median of the 5 Phase-14 H1 robust4 deep jumps landed at both dt (2.08-2.34 m/s); H1 published = MJCF")}
RISE_TO = {"G1": (0.066, "Phase-14 G1 robust4 seed 2 replay: COM 0.732 m at takeoff vs 0.666 stance"),
           "H1": (-0.040, "Phase-14 H1 robust4 seed 1 replay: COM 0.910 m at takeoff vs 0.950 stance")}
LEG_TYPES = ("hip_pitch", "knee", "ankle")


def v_ceiling(robot):
    """Takeoff-speed ceiling under the published ratings: the Phase-14 ceiling scaled by sqrt of the leg pitch-chain peak-power
    capacity ratio sum(tau*omega/4) published / MJCF peak (spec only; the rating change is the only input)."""
    import yaml
    sp = yaml.safe_load(open(f"{ROOT}/spec/{robot.lower()}.yaml")); rt = rating_table(robot); P0 = P1 = 0.0
    for jn, rec in sp["joints"].items():
        if not jn.startswith("left") or not any(k in jn for k in LEG_TYPES) or "roll" in jn or "yaw" in jn: continue
        om = float(rec["omega_max"]["value"]); tp = float(rec["tau_max_peak"]["value"]); P0 += tp*om/4; P1 += min(tp, rt.get(jn, tp))*om/4
    return V14[robot][0]*np.sqrt(P1/P0), P1/P0


def predict(robot, h, D, geo=None):
    """Spec prediction. Ballistic COM from takeoff (height z_stand + rise_to) with speed v: the highest point reachable at horizontal
    distance X is the safety parabola  y(X) = v^2/2g - g X^2 / (2 v^2), at launch angle tan th = v^2/(g X).
    Landing needs the COM above the box top by its stance height (legs at stance length at touchdown) after the feet (and the COM
    with them, offset preserved) travel X = D + foot_back + EDGE_MARGIN. h_max = rise_to + y(X); success predicted iff h <= h_max."""
    geo = geo or stance_geometry(robot); X = D + geo["foot_back"] + EDGE_MARGIN
    v, ratio = v_ceiling(robot); y = v*v/(2*G) - G*X*X/(2*v*v); th = float(np.arctan2(v*v, G*X)); hm = RISE_TO[robot][0] + y
    return dict(robot=robot, h=h, D=D, X=X, v=v, power_ratio=ratio, rise_to=RISE_TO[robot][0], h_max=hm, theta_deg=float(np.degrees(th)),
                vx=v*np.cos(th), vz=v*np.sin(th), success=bool(h <= hm))


# ------------------------------------------------------------------ scaffold
def prior(S, theta=None, horizon=1.1, nk=22, t_ref=0.8, tilted=False):
    """Default: the jump's own ascent prior UNCHANGED (engine.strategy.jump_prior: effector = COM, d = +z, COM xy held in the
    wind-up). tilted=True: d = (cos th, 0, sin th) with th the predicted launch angle (REJECTED in the smoke: the static
    certificate truncates the wind-up at 9 steps / 9 mm of crouch because -d drives the COM backward out of the support)."""
    from engine import strategy as ST
    if not tilted: return dict(ST.jump_prior(S, horizon, nk, t_ref), key="jump_ascent_prior")
    d = [float(np.cos(theta)), 0.0, float(np.sin(theta))]
    return ST.ascent_prior(S, ("com",), d, horizon, nk, t_w=max(0.1, t_ref - 0.25), t_ref=t_ref, move_legs_in_windup=True, key="boxjump_tilted_prior", hold_com_xy=False)


CP_PLACE = os.environ.get("BJ_CP", "1") == "1"
CP_FOLLOW = os.environ.get("BJ_CPFOLLOW", "0") == "1"
HIP_STRAT = os.environ.get("BJ_HIPSTRAT", "0") == "1"
FLIGHT_FREE = os.environ.get("BJ_FLIGHTFREE", "0") == "1"
# R1: CP_FOLLOW = orbital-energy release (leg inclination follows the COM after touchdown); R1b: HIP_STRAT = trunk pitch reference
# from the capture-point error; R1c: FLIGHT_FREE = trunk-attitude loop off in flight (the placement is not cancelled).
RATE_LIMIT = os.environ.get("BJ_RATE", "0") == "1"; RATE_FRAC = 0.7
LANDPAR_ON = os.environ.get("BJ_LANDPAR", "0") == "1"
STEP_ON = os.environ.get("BJ_STEP", "0") == "1"          # R2: capture-step landing (see BoxLanding._step_pre)
STEP_T = float(os.environ.get("BJ_STEP_T", "0.25")); STEP_LIFT = float(os.environ.get("BJ_STEP_LIFT", "0.04")); STEP_MARGIN = float(os.environ.get("BJ_STEP_MARGIN", "0.0")); STEP_WIN = 0.3; STEP_NMAX = int(os.environ.get("BJ_STEP_NMAX", "3")); STEP_CLOSE = os.environ.get("BJ_STEP_CLOSE", "0") == "1"
BOTH_DT = os.environ.get("BJ_BOTHDT", "0") == "1"      # R1: objective = worse of 1 ms and 0.5 ms, both always evaluated
# v6: searched landing parameters, each a log-range around the robot's Phase-14 tuned value (u = 0 -> tuned value, u = +-1 -> x/÷ factor)
LANDPAR = (("crouch", 2.0), ("t_down", 3.0), ("t_hold", 3.0), ("t_up", 3.0), ("k_cp", 2.0))


def landpar_from(u, par):
    return {k: float(par[k]*f**float(np.clip(x, -1, 1))) for (k, f), x in zip(LANDPAR, u)}


class BoxLanding(B.BalanceController):
    """Phase-14 controller with (a) a searched flight bend blended out over t_down after touchdown (instead of the step back to the
    stand target the tuned controller makes; at its tuned pre_bend 0.05 the two are the same) and (b) v4: CAPTURE-POINT FOOT
    PLACEMENT in flight (cp=True): the feet are swung ahead of the COM by the capture offset dx = vx/w0 (w0 = sqrt(g/z_stand), the
    same LIP model the controller's stance loop uses), via hip pitch -asin(dx/L) and ankle pitch +asin(dx/L) (sole attitude kept),
    L = hip-to-sole height at the stance; the last flight offset is blended out over t_down after touchdown like the bend.
    The Phase-14 controller only regulates the capture point once in contact, so a forward-moving touchdown (vx ~1 m/s) lands with
    the capture point 0.25-0.3 m ahead of the feet and falls; (b) is the body-derived fix, not a task-fitted constant."""
    def __init__(self, S, q_stand, params, flight_bend, cp=None):
        super().__init__(S, q_stand, params=dict(params, pre_bend=flight_bend)); self.q_stand0 = self.q_stand.copy(); self.q_land0 = self.q_land.copy(); self.fb = flight_bend
        self.dq = np.zeros_like(self.q_stand)
        for k, s in self.dq_crouch.items():
            for a in self.qa[k]: self.dq[a] += s
        self.cp = CP_PLACE if cp is None else cp; self.dcp = np.zeros_like(self.q_stand)
        for a in self.qa["hp"]: self.dcp[a] -= 1.0
        for a in self.qa["ap"]: self.dcp[a] += 1.0
        m = self.m; d0 = mujoco.MjData(m); d0.qpos[:] = self.q_stand; mujoco.mj_forward(m, d0)
        hj = self.j["hp"][0]; self.L = float(d0.xanchor[hj][2] - min(d0.xpos[b][2] for b in self.fbodies)) + 0.05   # + ankle-to-sole
        self.w0 = np.sqrt(G/self.z_stand); self.delta = 0.0
    def targets(self, t, d, in_contact):
        if STEP_ON: self._step_pre(t, d, in_contact)
        tg = self._targets(t, d, in_contact)
        if STEP_ON: tg = self._step_post(t, d, tg)
        if RATE_LIMIT and self.rate is not None:      # v7: driven-joint targets slew at <= RATE_FRAC x rated speed (envelope by construction)
            if self.prev is None: self.prev = np.array([d.qpos[self.m.jnt_qposadr[int(self.m.actuator_trnid[a][0])]] for a in range(self.m.nu)])
            step = self.rate*0.002; tg = np.where(self.rate > 0, self.prev + np.clip(tg - self.prev, -step, step), tg); self.prev = tg.copy()
        return tg

    def set_rate(self, S):
        self.rate = np.zeros(self.m.nu); self.prev = None
        for aid, om in zip(S["jarr"]["aid"], S["jarr"]["om"]): self.rate[int(aid)] = RATE_FRAC*float(om)

    rate = None; prev = None

    def _targets(self, t, d, in_contact):
        if self.t_touch is None and not in_contact and self.cp and FLIGHT_FREE:
            # R1c: no trunk-attitude loop in FLIGHT (nothing to react against: hip flexion for the placement pitches the trunk forward by
            # momentum conservation, and the Phase-14 trunk-vertical term then extends the hips and cancels the placement; R1/R1b probe
            # traces touched down with the COM 2-3 cm AHEAD of the feet despite a 0.13 m commanded placement). Trunk loop resumes at touchdown.
            self.tilt_ref = np.array(self._tilt(d, self.tb))
        elif self.t_touch is not None and FLIGHT_FREE and not (HIP_STRAT and self.p.get("k_hip", 0.0) > 0):
            self.tilt_ref = 0.0
        if self.t_touch is None and not in_contact and self.cp:
            vx = float(P.com_velocity(self.m, d)[0]); dx = vx/self.w0
            self.delta = float(np.arcsin(np.clip(dx/self.L, -0.6, 0.6))); self.q_land = self.q_land0 + self.delta*self.dcp
        if self.t_touch is not None and HIP_STRAT and self.p.get("k_hip", 0.0) > 0:
            # R1b: HIP STRATEGY. The Phase-14 stance loop keeps the trunk vertical and regulates the capture point with the ankles
            # only (35 N m on G1, CoP bounded by the sole). Centroidal momentum: F_x = -(dL_y/dt + (x_cop - x_com) F_z)/z, so a
            # forward trunk-pitch acceleration (dL_y/dt > 0) decelerates the COM beyond the sole-bounded ankle authority. The trunk
            # reference pitches forward in proportion to the capture-point error: tilt_ref = k_hip * (cp_x - feet_x), clipped to the
            # same 0.6 rad bound as the placement. k_hip is a landing gain tuned on the generic flat-floor contract (no box).
            com = d.subtree_com[0]; vcom = P.com_velocity(self.m, d); w0 = np.sqrt(G/max(com[2], 0.2))
            fx = 0.5*(B.foot_xy(d, self.feet[0])[0] + B.foot_xy(d, self.feet[1])[0])
            self.tilt_ref = np.array([float(np.clip(self.p["k_hip"]*(com[0] + vcom[0]/w0 - fx), -0.6, 0.6)), 0.0])
        if self.t_touch is not None:
            tau = t - self.t_touch; r = max(0.0, 1.0 - tau/max(self.p["t_down"], 1e-3))
            if self.cp and CP_FOLLOW:
                # R1 (repair round 1): ORBITAL-ENERGY LANDING. The leg-inclination target is no longer blended to vertical on a timer
                # (which, with the feet planted ahead, servo-drives the COM forward and injects the very momentum the placement was
                # meant to cancel); it FOLLOWS the measured COM-over-feet inclination asin((x_feet - x_com)/L), so the LIP passively
                # trades the forward speed for height-neutral rotation and the leg reaches vertical when the COM stops over the feet.
                # Same LIP model, same L, no new constant; the ankle CP loop of the stance controller acts on the residual.
                fx = float(np.mean([d.xpos[b][0] for b in self.fbodies])); cx = float(d.subtree_com[0][0])
                dm = float(np.arcsin(np.clip((fx - cx)/self.L, -0.6, 0.6)))
                # monotone RELEASE only: the target moves from the touchdown placement toward vertical as the COM comes over the feet,
                # never beyond the placement and never past vertical (following without this bound is positive feedback: v0 fell at vx 0)
                self.rel = min(getattr(self, "rel", abs(self.delta)), max(0.0, dm*np.sign(self.delta))) if self.delta != 0 else 0.0
                self.q_stand = self.q_stand0 + self.fb*r*self.dq + np.sign(self.delta)*self.rel*self.dcp
                # (a world-frame flight placement on the hips was tried and dropped: hip flexion in flight pitches the trunk forward by
                # momentum conservation, so compensating the pelvis pitch is positive feedback; probe touched down with feet 4 cm ahead only)
            else:
                self.q_stand = self.q_stand0 + self.fb*r*self.dq + self.delta*r*self.dcp
        return super().targets(t, d, in_contact)


    # ---------------------------------------------------------- R2: CAPTURE STEP (BJ_STEP=1; general, LIP-derived, see PREDICTION_R2.md)
    #   After touchdown, if the capture point cp = x_com + v_x/w0 lies beyond the front of the soles by more than STEP_MARGIN (the
    #   ankle strategy cannot capture it: R1 bound w0 x (toe - COM)), the less-loaded foot is lifted and swung to the capture point
    #   (x = the cp predicted at swing touchdown, y = its own settled lateral offset from the stance foot, sole at its settled attitude, reach <= the stance leg
    #   length) by damped-least-squares IK on its own leg chain (re-solved at 100 Hz), sine lift STEP_LIFT over STEP_T. The stance
    #   ankle loop regulates cp over the stance foot during the swing. At swing touchdown (load > 1 N after STEP_T/2) the leg freezes at
    #   its landed angles and the two-foot balance continues on the staggered base. One step at most, triggered within STEP_WIN s of
    #   touchdown. Nothing here refers to a box: the support surface height is the stance foot's own sole height.
    def _step_init(self, d):
        m = self.m; S = self.S; self.st_ = dict(phase="none", k=0)
        d0 = mujoco.MjData(m); d0.qpos[:] = self.q_stand0; mujoco.mj_forward(m, d0)
        self.legs = []
        for i, fb in enumerate(self.fbodies):
            js = []; b = fb
            while b > 0 and m.body_jntnum[b] >= 0 and not (m.body_jntnum[b] and m.jnt_type[m.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE):
                for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
                    if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE: js.append(j)
                b = int(m.body_parentid[b])
            js = js[::-1]; hipb = int(m.jnt_bodyid[js[0]])
            gs = [g for g in S["foot_geoms"] if int(m.geom_bodyid[g]) == fb]
            fx = B.foot_xy(d0, self.feet[i]); toe = max(d0.geom_xpos[g][0] + geom_extent(m, d0.geom_xmat[g], g)[0] for g in gs) - fx[0]
            fp3 = self._fp3(d0, i)
            self.legs.append(dict(qa=np.array([m.jnt_qposadr[j] for j in js]), dof=np.array([m.jnt_dofadr[j] for j in js]),
                                  lo=np.array([m.jnt_range[j][0] for j in js]), hi=np.array([m.jnt_range[j][1] for j in js]), hip=hipb, gs=set(gs),
                                  toe=float(toe), R0=d0.xmat[fb].reshape(3, 3).copy(), p0=fp3, reach=0.95*float(np.linalg.norm(d0.xpos[hipb] - fp3))))   # 0.95: keeps the knee off the straight-leg singularity (an unreachable target diverged the IK: hip roll 3 rad)
        self.a_of_q = {int(m.jnt_qposadr[int(m.actuator_trnid[a][0])]): a for a in range(m.nu)}
        self.dk = mujoco.MjData(m)

    def _fp3(self, d, i):
        b, sid, loc = self.feet[i]
        return (d.site_xpos[sid] if sid is not None else d.xpos[b] + d.xmat[b].reshape(3, 3) @ loc).copy()

    def _foot_load(self, d, i):
        m = self.m; gs = self.legs[i]["gs"]; f = 0.0
        for ci in range(d.ncon):
            c = d.contact[ci]
            if c.geom1 in gs or c.geom2 in gs:
                o = c.geom2 if c.geom1 in gs else c.geom1
                if int(m.body_rootid[int(m.geom_bodyid[o])]) == 0:
                    fv = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fv); f += fv[0]
        return f

    def _ik(self, d, i, pt, n_it=8):
        m = self.m; dk = self.dk; dk.qpos[:] = d.qpos; L = self.legs[i]; b, sid, loc = self.feet[i]
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv)); Rt = L["R0"]
        for _ in range(n_it):
            mujoco.mj_kinematics(m, dk); mujoco.mj_comPos(m, dk)
            pc = self._fp3(dk, i); mujoco.mj_jac(m, dk, jp, jr, pc, b); Rc = dk.xmat[b].reshape(3, 3)
            er = 0.5*sum(np.cross(Rc[:, k], Rt[:, k]) for k in range(3)); e = np.concatenate([pt - pc, er])
            Jm = np.vstack([jp[:, L["dof"]], jr[:, L["dof"]]]); dq = Jm.T @ np.linalg.solve(Jm @ Jm.T + 1e-3*np.eye(6), e)
            nq = float(np.linalg.norm(dq)); dq = dq*min(1.0, 0.2/max(nq, 1e-9))         # bounded Newton step
            dk.qpos[L["qa"]] = np.clip(dk.qpos[L["qa"]] + dq, L["lo"], L["hi"])
        return dk.qpos[L["qa"]].copy()

    def _crouch_w(self, t):
        p = self.p; tau = t - self.t_touch
        if tau < p["t_down"]: w = tau/p["t_down"]
        elif tau < p["t_down"] + p["t_hold"]: w = 1.0
        else: w = max(0.0, 1.0 - (tau - p["t_down"] - p["t_hold"])/p["t_up"])
        return w*p["crouch"]

    def _crouch_vec(self):
        v = np.zeros_like(self.q_stand)
        for k, s_ in self.dq_crouch.items():
            for a in self.qa[k]: v[a] += s_
        return v

    def _step_pre(self, t, d, in_contact):
        if not hasattr(self, "st_"): self._step_init(d); self.q_stand00 = self.q_stand0.copy()
        st = self.st_; Tsw = self.p.get("step_T", STEP_T)
        if self.t_touch is None or st["phase"] == "done": return
        if st["phase"] == "hold":
            feet = 0.5*(B.foot_xy(d, self.feet[0]) + B.foot_xy(d, self.feet[1])); self.cp_offset = np.array([st["hold_x"] - feet[0], 0.0]); return
        com = d.subtree_com[0]; vcom = P.com_velocity(self.m, d); w0 = np.sqrt(G/max(com[2], 0.2)); cp = com[:2] + vcom[:2]/w0
        if st["phase"] == "none":
            tau = t - self.t_touch
            if tau > STEP_WIN: st["phase"] = "done"; return
            front = max(B.foot_xy(d, self.feet[i])[0] + self.legs[i]["toe"] for i in (0, 1))
            if cp[0] > front + self.p.get("step_margin", STEP_MARGIN):
                ld = [self._foot_load(d, i) for i in (0, 1)]; sw = 1 if ld[1] < ld[0] else 0
                self.p["k_trunk"] = self.p["k_trunk"]*self.p.get("step_ktr", 1.0)   # trunk loop gain while stepping (one hip carries it)
                st.update(phase="swing", n=1, sw=sw, su=1 - sw, t0=t, k=0, cp_trig=float(cp[0] - front), tau_trig=float(tau),
                          off=(self.legs[sw]["p0"] - self.legs[1 - sw]["p0"]), x_max=-9.0, steps=[], q_land={}, close=False)
        if st["phase"] == "closing":
            # both feet down after the closing step: blend both legs from their landed angles back to the stand posture (feet side by
            # side again, so the stand posture is consistent with the base); the Phase-14 two-foot balance acts throughout
            a = min(1.0, (t - st["t_land"])/Tsw)
            for i in (0, 1): qa = self.legs[i]["qa"]; self.q_stand0[qa] = st["q_fr"][i] + a*(self.q_stand00[qa] - st["q_fr"][i])
            if a >= 1.0: st["phase"] = "done"
            return
        if st["phase"] == "swing":
            sw, su = st["sw"], st["su"]; s_ = t - st["t0"]
            if s_ > 0.5*Tsw and self._foot_load(d, sw) > 1.0:
                st["steps"].append(dict(t_land=float(t), foot=int(sw), dx=float(B.foot_xy(d, self.feet[sw])[0] - B.foot_xy(d, self.feet[su])[0])))
                self.cp_offset = 0.0
                # next action, by the same capture rule: cp still beyond the new front toe -> another capture step with the trailing
                # foot (at most STEP_NMAX steps); else feet still staggered -> a closing step (trailing foot beside the front one);
                # else both legs blend back to the stand posture. The landed leg holds its landed angles meanwhile.
                st["q_land"][sw] = d.qpos[self.legs[sw]["qa"]].copy(); self.q_stand0[self.legs[sw]["qa"]] = st["q_land"][sw] - self._crouch_w(t)*self._crouch_vec()[self.legs[sw]["qa"]]
                self.dq[self.legs[sw]["qa"]] = 0.0; self.dcp[self.legs[sw]["qa"]] = 0.0
                ft = B.foot_xy(d, self.feet[sw])[0] + self.legs[sw]["toe"]; gap = float(B.foot_xy(d, self.feet[sw])[0] - B.foot_xy(d, self.feet[su])[0])
                if cp[0] > ft + self.p.get("step_margin", STEP_MARGIN) and st["n"] < STEP_NMAX:
                    st.update(n=st["n"] + 1, sw=su, su=sw, t0=t, k=0, off=(self.legs[su]["p0"] - self.legs[sw]["p0"]), x_max=-9.0, q_sw=None, close=False)
                elif STEP_CLOSE and abs(gap) > 0.05 and st["n"] < STEP_NMAX + 1:
                    st.update(n=st["n"] + 1, sw=su, su=sw, t0=t, k=0, off=(self.legs[su]["p0"] - self.legs[sw]["p0"]), x_max=-9.0, q_sw=None, close=True)
                elif STEP_CLOSE:
                    if su not in st["q_land"]: st["q_land"][su] = d.qpos[self.legs[su]["qa"]].copy()
                    for i in (0, 1): self.dq[self.legs[i]["qa"]] = 0.0; self.dcp[self.legs[i]["qa"]] = 0.0
                    st.update(phase="closing", t_land=float(t), q_fr=dict(st["q_land"]))
                else:
                    # STAGGERED HOLD (default): the cp is inside the base, so stay in double support; both legs keep their landed angles
                    # as the stand base and the Phase-14 two-foot balance (cp over the feet centre, trunk vertical) acts on them
                    q = d.qpos[self.legs[su]["qa"]].copy(); self.q_stand0[self.legs[su]["qa"]] = q; self.q_stand0[self.legs[sw]["qa"]] = st["q_land"][sw]
                    for i in (0, 1): self.dq[self.legs[i]["qa"]] = 0.0; self.dcp[self.legs[i]["qa"]] = 0.0
                    # the absorb schedule's crouch would keep re-shaping the landed legs (its slow rise extended the front leg and pushed
                    # the body over backward: R2 probe); both legs hold their landed angles, the balance loops act on them, and the cp is
                    # regulated to where it is now (clipped between the feet) instead of re-centred between staggered feet
                    self.dq_crouch = {}
                    xs = sorted(float(B.foot_xy(d, self.feet[i])[0]) for i in (0, 1))
                    st.update(phase="hold", t_land=float(t), hold_x=float(np.clip(cp[0], xs[0], xs[1])))
                    # posture held as landed: the trunk reference is its pitch now (re-erecting it on the frozen staggered legs shifted the
                    # COM back past the rear heel: R2 probe), and the cp loop gain is scaled by hold_kcp (tuned on the flat contract)
                    self.tilt_ref = np.array([float(self._tilt(d, self.tb)[0]), 0.0]); self.p["k_cp"] = self.p["k_cp"]*self.p.get("hold_kcp", 1.0)
                return
            psu = self._fp3(d, su)
            if st["k"] % 5 == 0:
                # LIP with the stance CoP saturated at the stance toe p: cp(t) - p grows as exp(w0 t); aim at the cp predicted for the
                # swing's remaining time (the instantaneous cp lags the step and the step comes up short; R2 first probe: 0.18 m step for 0.39 m needed)
                p_ = B.foot_xy(d, self.feet[su])[0] + self.p.get("step_cop", 0.0)*self.legs[su]["toe"]; trem = max(0.0, Tsw - s_)
                x = (p_ + max(0.0, cp[0] - p_)*np.exp(w0*trem) if cp[0] > p_ else cp[0]) + self.p.get("step_beyond", 0.0); y = psu[1] + st["off"][1]   # step_beyond: land the foot past the predicted cp so the cp ends inside the base (braking margin)
                if st.get("close"): x = float(psu[0])                          # closing step: beside the front foot
                L = self.legs[sw]; hx = d.xpos[L["hip"]]; zt = psu[2] + st["off"][2] + self.p.get("step_lift", STEP_LIFT)*np.sin(np.pi*min(1.0, s_/Tsw)) - 0.03*min(1.0, max(0.0, s_ - Tsw)/0.1)
                rh = np.sqrt(max(0.0, L["reach"]**2 - (hx[2] - zt)**2)); hv = np.array([x, y]) - hx[:2]; hn = np.linalg.norm(hv)
                if hn > rh: x, y = hx[:2] + hv/max(hn, 1e-9)*rh
                x = max(x, st["x_max"]); st["x_max"] = x       # never pull the swing foot back (cp behind the new toe is inside the base)
                # min-jerk path from the lift-off point to the target over the swing time (a target jump through the stiff landing servos
                # saturates the swing hip and its reaction pitches the trunk and kicks the stance leg off the floor: R2 probe, cp off)
                if "p_start" not in st or st.get("p_start_n") != st["n"]: st["p_start"] = self._fp3(d, sw)[:2].copy(); st["p_start_n"] = st["n"]
                u = min(1.0, (s_ + 0.01)/Tsw); mj = 10*u**3 - 15*u**4 + 6*u**5
                xy = st["p_start"] + (np.array([x, y]) - st["p_start"])*mj
                st["q_sw"] = self._ik(d, sw, np.array([xy[0], xy[1], zt])); st["tgt"] = (float(x), float(y), float(zt))
            st["k"] += 1
            # stance ankle: cp over the stance foot, with the error clipped to the sole (heel..toe): the CoP cannot leave it, and an
            # unclipped error drives the stance ankle into full plantarflexion, which lifts the heel and launches the body (first probe)
            fs = B.foot_xy(d, self.feet[su]); feet = 0.5*(B.foot_xy(d, self.feet[0]) + B.foot_xy(d, self.feet[1]))
            e = cp - fs; e[0] = float(np.clip(e[0], -self.legs[su]["toe"], self.legs[su]["toe"])); self.cp_offset = cp - feet - e

    def _step_post(self, t, d, tg):
        st = getattr(self, "st_", {})
        if st.get("phase") == "swing" and st.get("q_sw") is not None:
            for qa, v in zip(self.legs[st["sw"]]["qa"], st["q_sw"]):
                a = self.a_of_q.get(int(qa))
                if a is not None: tg[a] = v
        return tg


def _step_info(ctl):
    st = getattr(ctl, "st_", None)
    if not st: return None
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in st.items() if k in ("phase", "n", "cp_trig", "tau_trig", "t_land", "steps", "tgt")}


# ------------------------------------------------------------------ rollout
def geom_extent(m, xmat, g):
    """Half-extent of geom g along world x and z (exact for sphere, capsule, box)."""
    R = np.asarray(xmat).reshape(3, 3); sz = m.geom_size[g]; t = m.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_SPHERE: return float(sz[0]), float(sz[0])
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE: return float(abs(R[0, 2])*sz[1] + sz[0]), float(abs(R[2, 2])*sz[1] + sz[0])
    if t == mujoco.mjtGeom.mjGEOM_BOX: return float(np.abs(R[0]) @ sz), float(np.abs(R[2]) @ sz)
    return float(m.geom_rbound[g]), float(m.geom_rbound[g])


def _foot_info(m, d, S):
    """Per foot: lowest point of its sole collision geoms, sole x-extent (heel, toe), foot body position."""
    out = []
    for fb in S["fbodies"]:
        zs, lo, hi = [], [], []
        for g in S["fgeoms"][fb]:
            ex, ez = geom_extent(m, d.geom_xmat[g], g); p = d.geom_xpos[g]
            zs.append(p[2] - ez); lo.append(p[0] - ex); hi.append(p[0] + ex)
        out.append((min(zs), min(lo), max(hi), d.xpos[fb].copy()))
    return out


def rollout(S, ref, om, z, par, record=False, push=None, land_lag=None, lag_c=2.0, lag_pred=True):
    """z = [knots (nT*nk), tscale_u, bend_u] in [-1, 1]. Returns metrics + gate verdict."""
    m = S["m"]; d = mujoco.MjData(m); BI.settle(S); dt = m.opt.timestep
    nT = len(S["types"]); nk = ref["nk"]; z = np.clip(np.asarray(z, float), -1, 1)
    kn = z[:nT*nk].reshape(nT, nk); ts = 0.87 + 0.3*z[nT*nk]; bend = 0.3*(1.0 + z[nT*nk + 1])
    trk = MB.Tracker(S, ref, omega=om, tscale=ts)
    if len(z) > nT*nk + 2:           # v6 (BJ_LANDPAR): the landing controller's absorb schedule and capture gain searched too
        par = dict(par, **landpar_from(z[nT*nk + 2:], par))
    ctl = BoxLanding(S, np.array(S["q_settled"]), par, bend); ctl.set_rate(S)
    d.qpos[:] = S["q_settled"]; d.qvel[:] = S["v_settled"]; mujoco.mj_forward(m, d)
    J = S["jarr"]; ti_ = J["ti"]; tau_v = J["tau"]; om_v = J["om"]
    act_j = np.array([S["a_hold"][t] for t in S["types"]])[ti_]*J["sgn"]
    alpha = 1.0 - np.exp(-dt/ACT_TAU)
    for (a_, dof, qad, tau, kp, kd) in S["held"]: d.ctrl[a_] = S["q_settled"][qad]
    if "fbodies" not in S:
        S["fbodies"] = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in BI.ROBOTS[S["robot"]]["foot_bodies"]]
        S["fgeoms"] = {fb: [g for g in S["foot_geoms"] if int(m.geom_bodyid[g]) == fb and (m.geom_contype[g] or m.geom_conaffinity[g] or S["robot"] == "G1") and
                            (S["robot"] != "G1" or (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith(("foot1_collision", "foot2_collision", "foot3_collision")))] for fb in S["fbodies"]}
        S["pelvis"] = 1
    fl, bx, h = S["floor"], S["box"], S["box_h"]; foot = S["foot_geoms"]; tb = ctl.tb
    lq, llo, lhi = S["lim_q"], S["lim_lo"], S["lim_hi"]
    start_qdot = float(np.max(np.abs(d.qvel[6:])))
    z_stand = float(d.subtree_com[0][2]); x0 = float(d.subtree_com[0][0])
    st = dict(phase="push", takeoff_t=None, v_to=None, touch_t=None, touch_on=None, nonfoot_box=False, front_face=False, floor_after=False,
              overspeed=0.0, lim_exc=0.0, zmin=z_stand, apex=z_stand, foot_clear=-1.0, fell=False, peak_util=0.0, max_util_land=0.0,
              arm_exc=0.0, x_touch=None)
    cand_t = None; in_flight = False; servo_on = False; prev_fz = 0.0; win = None; trace = [] if record else None
    q_arm = [i for i, t in enumerate(S["types"]) if "shoulder" in t]; arm_idx = np.where(np.isin(ti_, q_arm))[0]
    q_arm0 = d.qpos[J["qad"][arm_idx]].copy() if len(arm_idx) else None
    nmax = int(H_MAX/dt); end_t = None
    try:
        for s in range(nmax):
            t = s*dt
            if st["phase"] == "push" or (st["phase"] == "confirm"):
                k = min(int(t/KNOT_DT + 1e-9), nk - 1); k2 = min(k + 1, nk - 1); f = min(1.0, (t - k*KNOT_DT)/KNOT_DT) if k < nk - 1 else 0.0
                cmd_j = (kn[:, k] + (kn[:, k2] - kn[:, k])*f)[ti_]*J["sgn"] + trk(t, d, S)
                w = d.qvel[J["dof"]]; act_j += alpha*(cmd_j - act_j)
                frac = np.maximum(0.0, 1.0 - np.abs(w)/om_v)
                hi = np.where(w > 0, tau_v*frac, tau_v); lo_ = np.where(w < 0, tau_v*frac, tau_v)
                tq = np.clip(act_j*tau_v, -lo_, hi); d.ctrl[J["aid"]] = tq
                st["peak_util"] = max(st["peak_util"], float(np.max(np.abs(tq)/tau_v)))
                if q_arm0 is not None: st["arm_exc"] = max(st["arm_exc"], float(np.max(np.abs(d.qpos[J["qad"][arm_idx]] - q_arm0))))
            else:
                if not servo_on and land_lag:
                    # LABELLED SIDE CHECK: landing servos of the driven joints as explicit PD -> first-order lag land_lag -> torque-speed
                    # envelope (held joints stay implicit servos, unlagged). Gain capped at lag_c/land_lag (Q1 convention); lag_pred: PD on
                    # the state predicted one lag ahead (the K2 lag-aware stand's compensation).
                    ieffL = np.array([1.0/float(S["Minv_key"][dof, dof]) for dof in J["dof"]]); wL = min(ctl.p["stiff"], lag_c/land_lag)
                    kpL = wL*wL*ieffL; kdL = 2.0*wL*ieffL; actL = np.array(d.actuator_force[J["aid"]]); alphaL = 1.0 - np.exp(-dt/land_lag)
                    servo_on = "lag"; ctl._kc = None; ctl.t_touch = None; tgL = None
                if not servo_on:
                    for i_a, (aid, dof) in enumerate(zip(J["aid"], J["dof"])):
                        ieff = 1.0/float(S["Minv_key"][dof, dof]); kp = ctl.p["stiff"]**2*ieff
                        BI._as_servo(m, int(aid), kp, 2.0*np.sqrt(kp*ieff), float(J["tau"][i_a]))
                    servo_on = True; ctl._kc = None; ctl.t_touch = None
                k_c = int(round(t/0.002, 6))
                if k_c != ctl._kc: ctl._kc = k_c; d.ctrl[:] = ctl.targets(k_c*0.002, d, prev_fz > 1.0); tgL = d.ctrl[J["aid"]].copy() if servo_on == "lag" else None
                if servo_on == "lag":
                    qL = d.qpos[J["qad"]]; wq = d.qvel[J["dof"]]
                    if lag_pred: qL = qL + land_lag*wq
                    actL += alphaL*(kpL*(tgL - qL) - kdL*wq - actL)
                    frac = np.maximum(0.0, 1.0 - np.abs(wq)/om_v)
                    d.ctrl[J["aid"]] = np.clip(actL, -np.where(wq < 0, tau_v*frac, tau_v), np.where(wq > 0, tau_v*frac, tau_v))
            w = d.qvel[J["dof"]]; r_ = np.abs(w)/om_v; i_ = int(np.argmax(r_))
            if r_[i_] > st["overspeed"]: st["overspeed"] = float(r_[i_]); st["os_where"] = (S["types"][int(ti_[i_])], st["phase"], round(t, 3))
            if st["phase"] in ("push", "confirm"): st["overspeed_push"] = max(st.get("overspeed_push", 0.0), float(r_[i_]))
            d.xfrc_applied[:] = 0
            if push is not None and push[0] <= t < push[0] + push[1]: d.xfrc_applied[tb, :3] = push[2]
            BI.backemf_brake(m, d, S); mujoco.mj_step(m, d)
            if servo_on: st["max_util_land"] = max(st["max_util_land"], float(np.max(np.abs(d.actuator_force[J["aid"]])/tau_v)))
            # contacts
            fz = 0.0; on_box = [False, False]; on_floor_foot = False
            for ci in range(d.ncon):
                c = d.contact[ci]; g1, g2 = c.geom1, c.geom2
                sup = fl if fl in (g1, g2) else (bx if bx in (g1, g2) else None)
                if sup is None: continue
                other = g2 if g1 == sup else g1
                fv = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fv); fz += fv[0]
                if sup == bx:
                    if other not in foot and os.environ.get("BJ_DEBUG") and not st["nonfoot_box"]: print("NONFOOT", t, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g1), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g2), c.pos, c.dist)
                    if other not in foot: st["nonfoot_box"] = True; st.setdefault("nonfoot_geoms", set()).add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other) or str(other))
                    else:
                        if c.pos[2] < h - 0.008 and os.environ.get("BJ_DEBUG") and not st["front_face"]: print("FRONT", t, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other), other, c.pos, c.dist)
                        if c.pos[2] < h - 0.008: st["front_face"] = True
                        on_box[0 if int(m.geom_bodyid[other]) == S["fbodies"][0] else 1] = True
                elif st["takeoff_t"] is not None and st["phase"] == "land":
                    st["floor_after"] = True
                    if other in foot: on_floor_foot = True
            com = d.subtree_com[0]; vcom = P.com_velocity(m, d); vz = float(vcom[2])
            st["zmin"] = min(st["zmin"], float(com[2])) if st["takeoff_t"] is None else st["zmin"]
            ex = np.maximum(llo - d.qpos[lq], d.qpos[lq] - lhi); lim = float(np.max(ex))
            if lim > st["lim_exc"]: st["lim_exc"] = lim; st["lim_where"] = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(S["lim_joints"][int(np.argmax(ex))])), st["phase"], round(t, 3))
            # takeoff (sustained FLIGHT_CONFIRM, as branch_i)
            if st["phase"] in ("push", "confirm"):
                if in_flight:
                    if fz >= 1.0: in_flight = False; st["phase"] = "push"; cand = None
                    elif t - cand_t >= BI.FLIGHT_CONFIRM - 1e-12:
                        st["phase"] = "land"; st["takeoff_t"] = cand_t; st["v_to"] = cand; st["arm_at_to"] = st["arm_exc"]
                elif fz < 1.0 and vz > 0 and s > 0:
                    in_flight = True; cand_t = t; st["phase"] = "confirm"
                    cand = dict(vx=float(vcom[0]), vz=vz, v=float(np.hypot(vcom[0], vz)), z=float(com[2]), crouch=z_stand - st["zmin"])
            else:
                st["apex"] = max(st["apex"], float(com[2]))
                fi = _foot_info(m, d, S)
                if st["touch_t"] is None:
                    st["foot_clear"] = max(st["foot_clear"], min(f_[0] for f_ in fi) - h)
                    if fz >= 1.0:
                        st["touch_t"] = t; st["touch_on"] = "box" if any(on_box) else "floor"
                        st["x_touch"] = min(f_[1] for f_ in fi) - S["edge_x"]
                        if st["touch_on"] == "floor": end_t = t + 0.05      # missed the box: stop shortly
                        else: end_t = t + SETTLE_T + HOLD_T
                if st["touch_t"] is not None and st["touch_on"] == "box" and win is None and t >= st["touch_t"] + SETTLE_T:
                    win = dict(t0=t, feet0=[f_[3] for f_ in fi], pel0=d.xpos[1].copy(), ft=0.0, pd=0.0, up_min=1.0, both_min=1)
                if win is not None:
                    win["ft"] = max(win["ft"], max(float(np.linalg.norm(f_[3] - p0)) for f_, p0 in zip(fi, win["feet0"])))
                    win["pd"] = max(win["pd"], float(np.linalg.norm(d.xpos[1][:2] - win["pel0"][:2])))
                    win["up_min"] = min(win["up_min"], float(d.xmat[tb][8]))
            if d.xmat[tb][8] < 0.3 or com[2] < 0.2: st["fell"] = True; break
            if st["takeoff_t"] is None and t > 2.0: break                    # never took off
            if record and s % max(1, int(0.01/dt)) == 0: trace.append(d.qpos.copy())
            prev_fz = fz
            if end_t is not None and t >= end_t: break
    finally:
        if servo_on is True:
            for i_a, aid in enumerate(J["aid"]): BI._as_motor(m, int(aid), float(J["tau"][i_a]))
    com = d.subtree_com[0]; vcom = P.com_velocity(m, d); fi = _foot_info(m, d, S)
    heel_margin = min(f_[1] for f_ in fi) - S["edge_x"]           # min x over sole contact geoms of both feet - edge
    up = float(d.xmat[tb][8])
    gates = dict(start_rest=start_qdot < 0.05, took_off=st["takeoff_t"] is not None, touched_box=st["touch_on"] == "box",
                 no_floor_after=not st["floor_after"], no_nonfoot_box=not st["nonfoot_box"], no_front_face=not st["front_face"],
                 not_fell=not st["fell"], held=win is not None and (t - win["t0"]) >= HOLD_T - 2*dt,
                 both_feet_on_top=all(on_box), feet_past_edge=heel_margin >= 0.0,      # the heel-most sole geom of BOTH feet is past the near edge
                 com_over_box=S["edge_x"] + 0.02 <= com[0] <= S["box_far_x"] - 0.02,
                 upright=up >= 0.85 and (win is None or win["up_min"] >= 0.8), standing=com[2] >= h + 0.6*z_stand,
                 settled=float(np.linalg.norm(vcom)) < 0.3,
                 support=win is not None and win["ft"] <= FOOT_TRAVEL_MAX and win["pd"] <= PELVIS_DRIFT_MAX,
                 envelope=st.get("overspeed_push", 0.0) <= OVERSPEED_TOL)
    # v7 GATE ALIGNMENT (disclosed): the task's envelope gate follows the project's OVERSPEED convention (branch_i: torque-driven
    # phases, i.e. push-off up to the confirmed takeoff, where the torque-speed envelope model applies). v0-v6 also gated flight and
    # landing speed and joint-limit penetration; those stay as STRICT gates, reported per result and ranked inside the objective.
    strict = dict(envelope_all_phases=st["overspeed"] <= OVERSPEED_TOL, limits=st["lim_exc"] <= LIM_TOL)
    valid = all(gates.values())
    bd = 10.0
    if st["v_to"] is not None:
        X = S["X_need"]; vx = max(st["v_to"]["vx"], 0.05); tt_ = X/vx
        h_reach = (st["v_to"]["z"] - z_stand) + st["v_to"]["vz"]*tt_ - 0.5*G*tt_*tt_
        bd = float(np.log1p(max(0.0, h - h_reach)))      # log-compressed so a slow hop and a backward jump stay ranked (no cap)
    out = dict(valid=bool(valid), gates=gates, failed=[k for k, v in gates.items() if not v], strict=strict, strict_failed=[k for k, v in strict.items() if not v],
               valid_strict=bool(valid and all(strict.values())), overspeed_push=st.get("overspeed_push", 0.0), takeoff=st["v_to"], takeoff_t=st["takeoff_t"],
               touch_t=st["touch_t"], touch_on=st["touch_on"], x_touch=st["x_touch"], heel_margin=float(heel_margin), foot_clear=st["foot_clear"],
               apex=st["apex"], crouch=(st["v_to"] or {}).get("crouch", z_stand - st["zmin"]), arm_exc=st.get("arm_at_to", st["arm_exc"]),
               overspeed=st["overspeed"], lim_exc=st["lim_exc"], peak_util_push=st["peak_util"], max_util_land=st["max_util_land"],
               up_end=up, com_end=com.tolist(), v_end=float(np.linalg.norm(vcom)), z_stand=z_stand,
               window=(None if win is None else dict(foot_travel=win["ft"], pelvis_drift=win["pd"], up_min=win["up_min"], T=t - win["t0"])),
               tscale=ts, flight_bend=bend, t_end=t, ballistic_deficit=bd, step=_step_info(ctl), nonfoot_geoms=sorted(st.get("nonfoot_geoms", [])), lim_where=st.get("lim_where"), os_where=st.get("os_where"))
    if record: out["trace"] = np.array(trace)
    return out


def penalty(o, h):
    """Rank score for an invalid rollout (>= 100); valid rollouts score < 0 in objective()."""
    g = o["gates"]; p = 100.0
    if not g["took_off"]: return p + 300.0     # above every takeoff (v2 had 60: H1 D 0.25 ranked standing still above backward jumps)
    # REACH FIRST, THEN LAND (lexicographic by weight): 1 cm of horizontal or vertical shortfall costs as much as one failed landing
    # flag, so a short hop that stands still ranks below a jump that reaches the box top and falls (v0 weights 20/m left the search
    # in a hop-and-stand basin: 0.49 m/s hop at 0.22 s, toes on the front face, heel 0.21 m short)
    # v2: + the BALLISTIC deficit of the takeoff state (the predictor's own model applied to the measured takeoff): with the COM
    # leaving at (vx, vz) from z_to, the height it still has above stance height after travelling X is h_reach; deficit = h - h_reach
    # (log1p-compressed). v1 (actual shortfalls only, 100/m) still preferred 0.03 m-crouch hops at 0.2 s (0.78 m/s) to the prior's
    # 1.7 m/s backward jump; the ballistic term ranks every takeoff by how close it comes to reaching the box.
    p += 50.0*max(0.0, EDGE_MARGIN - o["heel_margin"]) + 50.0*max(0.0, 0.02 - o["foot_clear"]) + 10.0*o.get("ballistic_deficit", 10.0)
    p += 5.0*(not g["touched_box"]) + 5.0*(not g["no_nonfoot_box"]) + 3.0*(not g["no_front_face"]) + 5.0*(not g["not_fell"])
    p += 3.0*(not g["no_floor_after"]) + 2.0*(not g["held"]) + 1.0*sum(not g[k] for k in ("both_feet_on_top", "com_over_box", "upright", "standing", "settled", "support"))
    p += 2.0*(not g["envelope"]) + 10.0*max(0.0, o["overspeed_push"] - OVERSPEED_TOL) + 0.5*len(o["strict_failed"])
    if o["window"]: p += 5.0*max(0.0, o["window"]["foot_travel"] - FOOT_TRAVEL_MAX) + 5.0*max(0.0, o["window"]["pelvis_drift"] - PELVIS_DRIFT_MAX)
    return p


def quality(o):
    """Valid-solution score (lower = better): margin past the edge (capped), small hold drift."""
    w = o["window"]; return -(1.0 + min(o["heel_margin"], 0.10) - 0.5*w["pelvis_drift"] - 1.0*w["foot_travel"]) + 0.2*len(o["strict_failed"])


def landing_params(robot):
    """Phase-14 tuned landing parameters; BJ_LANDFILE_<ROBOT> = a tune_forward_landing() output (v8) overrides them."""
    f = os.environ.get(f"BJ_LANDFILE_{robot}")
    if f: return dict(json.load(open(core.LANDING_PAR[robot]))["best"], **json.load(open(f))["best_par"])
    return json.load(open(core.LANDING_PAR[robot]))["best"]


class Ctx:
    """A (robot, h, D) problem on FRESH plants (built uncached for this context) at 1 ms and 0.5 ms with its prior and tracker gain.
    ref/om may be passed in (perturbation replays reuse the nominal scaffold so only the perturbed quantity changes)."""
    def __init__(self, robot, h, D, dts=(0.001, 0.0005), ref=None, om=None, **kw):
        self.robot, self.h, self.D = robot, h, D; self.par = landing_params(robot)
        self.S = {dt: plant(robot, h, D, dt, **kw) for dt in dts}
        S1 = self.S[dts[0]]; self.ref = ref if ref is not None else prior(S1)
        self.om = float(om) if om is not None else float(MB.tune_gains(S1, self.ref, 1.0)[0])
    def n(self): return len(self.S[0.001]["types"])*self.ref["nk"] + 2 + (len(LANDPAR) if LANDPAR_ON else 0)
    def run(self, z, dt, **kw): return rollout(self.S[dt], self.ref, self.om, z, self.par, land_lag=self.land_lag, **kw)
    land_lag = None
    def objective(self, z):
        if BOTH_DT:     # R1: both dt ALWAYS evaluated (v0-v8 evaluated 0.5 ms only once 1 ms was valid, so near misses were dt-fragile)
            o1, o2 = self.run(z, 0.001), self.run(z, 0.0005)
            f = [quality(o) if o["valid"] else penalty(o, self.h) + 1.0 for o in (o1, o2)]
            return max(f), o1, o2
        o1 = self.run(z, 0.001)
        if not o1["valid"]: return penalty(o1, self.h) + 1.0, o1, None
        o2 = self.run(z, 0.0005)
        if not o2["valid"]: return 100.0 + 0.5*penalty(o2, self.h)/100.0, o1, o2
        return max(quality(o1), quality(o2)), o1, o2


def _clean(o):
    if o is None: return None
    return json.loads(json.dumps({k: v for k, v in o.items() if k != "trace"}, default=lambda x: x.tolist() if hasattr(x, "tolist") else (bool(x) if isinstance(x, np.bool_) else float(x))))


WARM14 = os.environ.get("BJ_WARM14", "0") == "1"
WARM14_SEED = {"G1": 4, "H1": 1}     # the Phase-14 robust4 jump with the highest takeoff that landed at both dt (G1 3.32/3.39, H1 2.32/2.35 m/s)


def phase14_warm(robot):
    """The Phase-14 robust4 deep jump: analytic pose-bank prior (JumpTask's default prior), its tracker omega, its knots + tscale,
    and the flight bend at the tuned pre_bend 0.05. Returns (ref, omega, z)."""
    f = "jump14_robust4" if robot == "G1" else "jump14_h1_robust4"
    W = [r for r in json.load(open(f"{core.RES}/phase14/{f}.json")) if r["seed"] == WARM14_SEED[robot]][0]
    ref = MB.analytic_clip(core.standard_jump_clip(robot), robot=robot)
    z = np.concatenate([np.clip(W["x"], -1, 1), [(W["tscale"] - 0.87)/0.3], [0.05/0.3 - 1.0]])
    return ref, float(W["omega"]), z


def search_chain(robot, D, heights, seed, iters, wall_rung, tag, pop=16):
    """Warm-started ladder over box heights. Writes results/movements/M3_box_jump/<tag>/rung_h*.json and progress.json."""
    import cma
    od = f"{OUT}/{tag}"; os.makedirs(od, exist_ok=True); warm = None; chain = []
    for h in heights:
        t0 = time.time()
        if WARM14:                       # v5: the Phase-14 robust vertical jump (bank prior, its tracker gain and knots) as the scaffold
            ref14, om14, z14 = phase14_warm(robot); ctx = Ctx(robot, h, D, ref=ref14, om=om14)
            x0 = z14 if warm is None else np.asarray(warm); sig = 0.12 if warm is None else 0.10
            if warm is None and os.environ.get("BJ_WARM_FROM"):     # v6: continue from a stored progress/rung file (its best z)
                wf = json.load(open(os.environ["BJ_WARM_FROM"])); x0 = np.asarray(wf.get("best_z") or wf["z"]); sig = 0.08
            if LANDPAR_ON and len(x0) < ctx.n(): x0 = np.concatenate([x0, np.zeros(ctx.n() - len(x0))])
        else:
            ctx = Ctx(robot, h, D); x0 = np.zeros(ctx.n()) if warm is None else np.asarray(warm); sig = 0.35 if warm is None else 0.15
        n = ctx.n()
        es = cma.CMAEvolutionStrategy(x0, sig, {"bounds": [-1, 1], "seed": seed, "verbose": -9, "maxiter": iters, "popsize": pop})
        best = (np.inf, None); hist = []; first_valid = None; it = 0
        while not es.stop() and time.time() - t0 < wall_rung:
            X = es.ask(); F = []
            for zz in X:
                f, o1, o2 = ctx.objective(zz); F.append(f)
                if f < best[0]: best = (f, np.asarray(zz).copy())
                if f < 0 and first_valid is None: first_valid = it
            es.tell(X, F); it += 1
            hist.append((it, float(min(F)), float(best[0]), int(sum(f < 0 for f in F)), round(time.time() - t0, 1)))
            if it % 5 == 0:
                json.dump(dict(robot=robot, D=D, h=h, seed=seed, it=it, best_f=float(best[0]), best_z=best[1].tolist(), hist=hist, first_valid_iter=first_valid),
                          open(f"{od}/progress_h{h:.2f}.json", "w"))
        f, o1, o2 = ctx.objective(best[1])
        rec = dict(robot=robot, D=D, h=h, seed=seed, iters=it, secs=time.time() - t0, best_f=float(best[0]), valid_both=bool(f < 0), first_valid_iter=first_valid,
                   z=best[1].tolist(), om=ctx.om, prior=ctx.ref["key"], meas_1ms=_clean(o1), meas_05ms=_clean(o2), hist=hist,
                   prediction=predict(robot, h, D), warm_from=(None if warm is None else chain[-1]["h"]))
        json.dump(json.loads(json.dumps(rec, default=float)), open(f"{od}/rung_h{h:.2f}.json", "w"), indent=1)
        chain.append(rec); warm = best[1]
        print(f"[{tag}] h={h} valid_both={rec['valid_both']} f={best[0]:.3f} iters={it} secs={rec['secs']:.0f}", flush=True)
    return chain


PERTURB = [dict(name="P1_z_noise_s11", z_sigma=0.03, z_seed=11), dict(name="P2_z_noise_s12", z_sigma=0.03, z_seed=12),
           dict(name="P3_start_noise_2mrad_box+1cm", start_sigma=0.002, start_seed=13, box_dx=0.01),
           dict(name="P4_mu0.8_box-1cm", mu_scale=0.8, box_dx=-0.01)]


def ref_for(rec):
    """The scaffold a stored rung was searched with (v5: the Phase-14 bank prior; earlier: the ascent prior)."""
    return phase14_warm(rec["robot"])[0] if rec.get("prior") != "jump_ascent_prior" else None


def validate(robot, h, D, z, om, label="", ref=None):
    """Nominal on FRESH plants at 1 and 0.5 ms + the 4 perturbations (each at both dt). Robust iff >= 3/4 perturbations valid at both."""
    z = np.asarray(z, float); nom = Ctx(robot, h, D, om=om, ref=ref); out = dict(robot=robot, h=h, D=D, label=label, om=om)
    o1, o2 = nom.run(z, 0.001), nom.run(z, 0.0005); out["nominal"] = dict(valid_both=bool(o1["valid"] and o2["valid"]), strict_both=bool(o1["valid_strict"] and o2["valid_strict"]), m1=_clean(o1), m05=_clean(o2))
    res = []
    for p in PERTURB:
        zz = z.copy(); kw = {}
        if p.get("z_sigma"): zz = np.clip(z + np.random.default_rng(p["z_seed"]).normal(0, p["z_sigma"], z.shape), -1, 1)
        if p.get("start_sigma"):
            nq = len(core.standard_jump_clip(robot)["start"]); sn = np.zeros(nq); sn[7:] = np.random.default_rng(p["start_seed"]).normal(0, p["start_sigma"], nq - 7); kw["start_noise"] = sn
        if p.get("box_dx"): kw["box_dx"] = p["box_dx"]
        if p.get("mu_scale"): kw["mu_scale"] = p["mu_scale"]
        c = Ctx(robot, h, D, ref=nom.ref, om=om, **kw) if kw else nom
        a, b = c.run(zz, 0.001), c.run(zz, 0.0005)
        res.append(dict(name=p["name"], valid_both=bool(a["valid"] and b["valid"]), strict_both=bool(a["valid_strict"] and b["valid_strict"]), failed_1ms=a["failed"], failed_05ms=b["failed"], m1=_clean(a), m05=_clean(b)))
    out["perturb"] = res; out["n_pass"] = sum(r["valid_both"] for r in res); out["robust"] = bool(out["nominal"]["valid_both"] and out["n_pass"] >= 3)
    out["n_pass_strict"] = sum(r["strict_both"] for r in res); out["robust_strict"] = bool(out["nominal"]["strict_both"] and out["n_pass_strict"] >= 3)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "chain":          # chain ROBOT D SEED ITERS WALL_RUNG TAG
        robot, D, seed, iters, wall, tag = sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), sys.argv[7]
        hs = [float(x) for x in os.environ.get("BJ_HEIGHTS", "0.20,0.30,0.40").split(",")]
        search_chain(robot, D, hs, seed, iters, wall, tag)
    elif cmd == "validate":     # validate TAG  (every rung file in results/movements/M3_box_jump/<TAG>)
        import glob
        tag = sys.argv[2]; od = f"{OUT}/{tag}"; allv = []
        for f in sorted(glob.glob(f"{od}/rung_h*.json")):
            r = json.load(open(f)); t0 = time.time(); v = validate(r["robot"], r["h"], r["D"], r["z"], r["om"], label=tag, ref=ref_for(r)); v["secs"] = time.time() - t0
            allv.append(v); print(f"[{tag}] h={r['h']} nominal={v['nominal']['valid_both']} perturb {v['n_pass']}/4 robust={v['robust']}", flush=True)
            json.dump(json.loads(json.dumps(allv, default=float)), open(f"{od}/validation.json", "w"), indent=1)
    elif cmd == "lagcheck":     # lagcheck TAG [lag_s]: LABELLED side check, landing servos through a first-order actuator lag
        import glob
        tag = sys.argv[2]; lag = float(sys.argv[3]) if len(sys.argv) > 3 else 0.010; od = f"{OUT}/{tag}"; allv = []
        for f in sorted(glob.glob(f"{od}/rung_h*.json")):
            r = json.load(open(f)); c = Ctx(r["robot"], r["h"], r["D"], om=r["om"], ref=ref_for(r)); c.land_lag = lag
            a, b = c.run(r["z"], 0.001), c.run(r["z"], 0.0005)
            allv.append(dict(h=r["h"], lag=lag, valid_1ms=a["valid"], valid_05ms=b["valid"], failed_1ms=a["failed"], failed_05ms=b["failed"], m1=_clean(a), m05=_clean(b)))
            print(f"[{tag}] lag {lag*1000:.0f} ms h={r['h']} 1ms={a['valid']} {a['failed']} 0.5ms={b['valid']} {b['failed']}", flush=True)
        json.dump(json.loads(json.dumps(allv, default=float)), open(f"{od}/lagcheck_{lag*1000:.0f}ms.json", "w"), indent=1)
    elif cmd == "render":       # render TAG H: 1 ms replay of the stored rung, mp4 + contact sheet via engine.render_any
        from engine import render_any as RA
        tag, h = sys.argv[2], float(sys.argv[3]); r = json.load(open(f"{OUT}/{tag}/rung_h{h:.2f}.json"))
        c = Ctx(r["robot"], h, r["D"], om=r["om"], ref=ref_for(r), dts=(0.001,)); o = c.run(r["z"], 0.001, record=True)
        path = scene(r["robot"], h, r["D"])[0]; to = o["takeoff"] or {}
        RA.render(path, o["trace"], 0.01, f"{OUT}/{tag}/boxjump_{r['robot']}_h{h:.2f}_D{r['D']:.2f}.mp4", title=f"{r['robot']} box jump: {h:.2f} m box, edge {r['D']:.2f} m ahead",
                  sub=f"stock body, published ratings; takeoff {to.get('v', 0):.2f} m/s (vx {to.get('vx', 0):.2f}, vz {to.get('vz', 0):.2f}); valid={o['valid']} {'' if o['valid'] else o['failed']}", azimuth=110, distance=3.2)


# ------------------------------------------------------------------ landing tolerance (mechanism probe)
def landing_tolerance(robot, vx, drop=0.20, dt=0.001, T=3.2, cp=True, landpar=None):
    """MECHANISM PROBE (no search): the robot in its flight posture is released from `drop` m above the flat floor with horizontal
    COM velocity vx (and the vertical speed of that fall), and the box-jump landing controller (Phase-14 tuned + CP placement)
    takes over in flight exactly as in rollout(). Same hold-window gates (from touchdown + SETTLE_T, HOLD_T s). Answers: what
    forward touchdown speed can this landing controller hold without stepping or hopping?"""
    S = plant(robot, 0.2, 3.0, dt); m = S["m"]; d = mujoco.MjData(m); par = landing_params(robot)
    if landpar is not None: par = dict(par, **landpar)
    ctl = BoxLanding(S, np.array(S["q_settled"]), par, 0.05/0.3*0 + par.get("pre_bend", 0.05), cp=cp); ctl.set_rate(S)
    J = S["jarr"]; d.qpos[:] = S["q_settled"]; d.qpos[2] += drop; d.qvel[:] = 0; d.qvel[0] = vx
    for (a_, dof, qad, tau, kp, kd) in S["held"]: d.ctrl[a_] = S["q_settled"][qad]
    for i_a, (aid, dof) in enumerate(zip(J["aid"], J["dof"])):
        ieff = 1.0/float(S["Minv_key"][dof, dof]); kp = ctl.p["stiff"]**2*ieff; BI._as_servo(m, int(aid), kp, 2.0*np.sqrt(kp*ieff), float(J["tau"][i_a]))
    mujoco.mj_forward(m, d); ctl._kc = None; fl = S["floor"]; fbs = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in BI.ROBOTS[robot]["foot_bodies"]]
    touch = None; win = None; prev_fz = 0.0; fell = False; hops = 0; airborne = True
    try:
        for s in range(int(T/dt)):
            t = s*dt; k_c = int(round(t/0.002, 6))
            if k_c != ctl._kc: ctl._kc = k_c; d.ctrl[:] = ctl.targets(k_c*0.002, d, prev_fz > 1.0)
            BI.backemf_brake(m, d, S); mujoco.mj_step(m, d)
            fz = 0.0
            for ci in range(d.ncon):
                c = d.contact[ci]
                if fl in (c.geom1, c.geom2): fv = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fv); fz += fv[0]
            if touch is None and fz >= 1.0: touch = t; airborne = False
            elif touch is not None:
                if fz < 1.0 and not airborne: airborne = True; hops += 1
                elif fz >= 1.0: airborne = False
            if touch is not None and win is None and t >= touch + SETTLE_T: win = dict(t0=t, f0=[d.xpos[b].copy() for b in fbs], p0=d.xpos[1].copy(), ft=0.0, pd=0.0)
            if win is not None:
                win["ft"] = max(win["ft"], max(float(np.linalg.norm(d.xpos[b] - p0)) for b, p0 in zip(fbs, win["f0"]))); win["pd"] = max(win["pd"], float(np.linalg.norm(d.xpos[1][:2] - win["p0"][:2])))
            if d.xmat[ctl.tb][8] < 0.3: fell = True; break
            prev_fz = fz
            if touch is not None and t >= touch + SETTLE_T + HOLD_T: break
    finally:
        for i_a, aid in enumerate(J["aid"]): BI._as_motor(m, int(aid), float(J["tau"][i_a]))
    ok = (not fell) and win is not None and win["ft"] <= FOOT_TRAVEL_MAX and win["pd"] <= PELVIS_DRIFT_MAX and d.xmat[ctl.tb][8] > 0.85
    return dict(robot=robot, vx=vx, drop=drop, cp=cp, dt=dt, ok=bool(ok), fell=fell, flight_phases_after_touch=hops, foot_travel=None if win is None else win["ft"],
                pelvis_drift=None if win is None else win["pd"], step=_step_info(ctl), t_end=float(t), slide=float(np.mean([d.xpos[b][0] for b in fbs]) - (S["q_settled"][0] + vx*0)), com_x_end=float(d.subtree_com[0][0]))


# ------------------------------------------------------------------ v8: landing controller re-tuned on FORWARD drops (generic contract)
TUNE_KEYS = (("stiff", 3.0), ("k_cp", 4.0), ("k_trunk", 3.0), ("crouch", 2.0), ("t_down", 3.0), ("t_hold", 3.0), ("t_up", 3.0), ("k_foot", 4.0))
TUNE_VX = (0.0, 0.25, 0.5, 0.75, 1.0)


def tune_forward_landing(robot, iters=20, pop=10, seed=1, out=None):
    """The Phase-14 tuning protocol extended to forward touchdowns: CMA over the landing controller's 8 gains/timings (log-factors
    around the Phase-14 tuned values), objective = drops from 0.20 m at vx in TUNE_VX, each scored 0 if held (window gates) else
    1 + min(foot travel, 1). Generic landing contract (flat floor, no box), so the result is not fitted to a box geometry."""
    import cma
    base = landing_params(robot); log = []
    def lp(u): return {k: float(base[k]*f**float(np.clip(x, -1, 1))) for (k, f), x in zip(TUNE_KEYS, u)}
    def obj(u):
        L = lp(u); sc = 0.0; det = []
        for vx in TUNE_VX:
            r = landing_tolerance(robot, vx, cp=True, landpar=L); det.append(r["ok"])
            sc += 0.0 if r["ok"] else 1.0 + min(1.0, r["foot_travel"] if r["foot_travel"] is not None else 1.0) + (2.0 if vx == 0 else 0.0)
        return sc, det
    es = cma.CMAEvolutionStrategy(np.zeros(len(TUNE_KEYS)), 0.4, {"bounds": [-1, 1], "seed": seed, "verbose": -9, "maxiter": iters, "popsize": pop})
    best = (np.inf, None, None); t0 = time.time()
    while not es.stop():
        X = es.ask(); F = []
        for u in X:
            f, det = obj(u); F.append(f)
            if f < best[0]: best = (f, np.asarray(u).tolist(), det)
        es.tell(X, F); log.append((len(log) + 1, float(min(F)), float(best[0]), round(time.time() - t0)))
        if out: json.dump(dict(robot=robot, best_f=best[0], best_u=best[1], best_par=lp(best[1]), held=dict(zip(TUNE_VX, best[2])), base=base, log=log), open(out, "w"), indent=1)
    return lp(best[1]), best
