"""SPRINT construction (2026-10-14): a running limit cycle DERIVED FROM THE BODY, tracked and refined on the true dynamics.

  stride   = the spec-derived ceiling model (engine.sprint_ceiling) gives the feasible strides at the target speed v (contact time t_c,
             flight t_f, stance hip height h0); compression dz = g t_f t_c/(2 pi) (spring-mass consistency); t_c >= pi x 40 ms (the
             project actuator lag's band), t_f >= 50 ms and >= the knee's fold time; the stride with the largest actuator margin is used.
  scaffold = one gait CYCLE (two steps, mirror-symmetric legs). STANCE: foot pinned, hip sweeps -Lc/2 -> +Lc/2 at v, height h0 - dz sin;
             SWING (Cartesian, relative to the hip): quintic fore-aft path continuous with the stance at liftoff and meeting the ground at
             -beta v (beta: the most ground-speed matching the swing's inverse-dynamics torques allow), sole lifted by a 3 cm bump over the
             floor as the scaffold's own hip height dictates; soles level; numerical IK on the MJCF (no authored joint curves). ARMS: an
             antiphase shoulder swing cancelling the fundamental of the legs' yaw angular momentum (angular-momentum balance). FEEDFORWARD:
             inverse dynamics of the swing/upper-body reference on the model.
  control  = event-synchronised cycle clock (held at a scheduled touchdown until the foot lands); FORCE-CONTROLLED STANCE (half-sine GRF
             with impulse m g T from the ceiling + COM height/velocity PD, directed at the COM and tilted for the attitude moment, residual
             to the CoP within the sole; tau = -J^T [F; M] on the model); PD + inverse-dynamics swing with Raibert foot placement about the
             COM's neutral point (gain t_c/2 + sqrt(h/g)); PD upper body; every gain from body quantities (inertias, t_c, the lag).
  search   = PERIODIC torque knots (8 per cycle; left = right half a cycle later, mirrored) + stride amplitude + time scale + pelvis lean,
             CMA-ES on the true dynamics, lexicographic score, worse of 1 ms and 0.5 ms (engine.task conventions).
  start    = RUNNING START at the flight apex before the right touchdown, COM at (v, sway, 0), zero angular momentum: a limit cycle.
  gates    = actuator torque-speed envelope + back-EMF brake + OVERSPEED > 1.1 x rated (branch_i), first-order actuator lag ACT_TAU
             (project convention 40 ms; SPRINT_LAG overrides, recorded in every result), energy closure, no non-foot floor contact (G1:
             geometric), foot slip <= 3 cm per stance (MuJoCo pyramidal friction, mu 1), no fall, heading within 0.35 rad, periodic
             cycle-boundary state (right touchdowns) after 2 transient cycles, all at BOTH 1 ms and 0.5 ms.
usage: python engine/sprint.py prior  G1 1.0                                          acceptance A (prior only, both dt)
       python engine/sprint.py ladder G1 1.0,1.5,2.0 [iters] [pop] [workers] [tag]    (SPRINT_LAG, SPRINT_WARM=<json>:<rung>, SCALE)
"""
import sys, os, json, time, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
import throw as T, branch_i as BI, plant as P
from engine import core, sprint_ceiling as SC

G = 9.81
RES = f"{core.RES}/sprint"
LEG = {"G1": dict(hp="{s}_hip_pitch_joint", hr="{s}_hip_roll_joint", hy="{s}_hip_yaw_joint", kn="{s}_knee_joint", ap="{s}_ankle_pitch_joint", ar="{s}_ankle_roll_joint"),
       "H1": dict(hp="{s}_hip_pitch", hr="{s}_hip_roll", hy="{s}_hip_yaw", kn="{s}_knee", ap="{s}_ankle", ar=None)}
CENTRE = {"G1": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"], "H1": ["torso"]}
ARM = {"G1": ["{s}_shoulder_pitch_joint", "{s}_shoulder_roll_joint", "{s}_elbow_joint"], "H1": ["{s}_shoulder_pitch", "{s}_shoulder_roll", "{s}_elbow"]}
# HELD-OUT EMBODIMENTS (2026-10-16, registration only): the derived models use the G1 joint names (src/heldout_onboard.py)
LEG["Apollo"] = LEG["T1"] = LEG["G1"]
CENTRE["Apollo"] = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]; CENTRE["T1"] = ["waist_yaw_joint"]
ARM["Apollo"] = ARM["T1"] = ARM["G1"]
LEGK = ("hp", "hr", "hy", "kn", "ap", "ar")
OM_LEG, OM_UP = float(os.environ.get("SPRINT_OML", 25.0)), 20.0        # tracker bandwidths (rad/s): legs 1/0.040 (the project actuator lag's corner), upper body the throw's 20
SWING_MODE = os.environ.get("SPRINT_SWING", "cart")
REF_ADV = os.environ.get("SPRINT_REFADV", "1") == "1"      # the tracked reference is advanced by the actuator lag (as the feedforward is)
KY_DIAG = float(os.environ.get("SPRINT_KY", 1.0))
PREDICT = os.environ.get("SPRINT_PREDICT", "1") == "1"
DAMP_L = os.environ.get("SPRINT_DAMPL", "0") == "1"
V_INST = os.environ.get("SPRINT_VINST", "1") == "1"
RESYNC = os.environ.get("SPRINT_RESYNC", "1") == "1"
STANCE_MODE = "force"
STANCE_G = float(os.environ.get("SPRINT_STG", 0.1 if STANCE_MODE == "force" else 1.0))
ID_FF = os.environ.get("SPRINT_IDFF", "1") == "1"
ARM_MODE = os.environ.get("SPRINT_ARM", "counter")
IEFF_MODE = os.environ.get("SPRINT_IEFF", "locked")
OMATT_MODE = os.environ.get("SPRINT_OMATT", "lag")
SCALED_JOINTS = tuple(os.environ["SPRINT_SCALE_JOINTS"].split(",")) if os.environ.get("SPRINT_SCALE_JOINTS") else ("hip_pitch", "knee") + (("ankle",) if os.environ.get("SPRINT_SCALE_ANKLE") else ())   # counterfactual body: joint-name substrings whose tau AND omega SCALE multiplies (SPRINT_SCALE_JOINTS overrides, 2026-10-16)
ACT_TAU = float(os.environ.get("SPRINT_LAG", 0.040))                   # the project's actuator lag (throw.rollout / branch_i): first-order, 40 ms
RATINGS = os.environ.get("SPRINT_RATINGS", "peak")                     # "peak" (MJCF) or "continuous" (vendor URDF effort) or "published" (URDF effort AND the vendor's published joint maximum); see rating_table
# VENDOR PUBLISHED JOINT MAXIMUM (2026-10-18, track K1): unitree.com/g1 lists "Maximum Torque of Knee Joint [1]: G1 90 N.m, G1 EDU 120 N.m",
# footnote [1]: "The maximum torque of the joint motors of the whole machine is different. This is the maximum torque of the largest joint
# motor among them." -> 120 N.m bounds EVERY joint of the EDU (the 29-DoF model Menagerie ships); it binds only the knee (MJCF/URDF 139).
VENDOR_MAX = {"G1": (120.0, "unitree.com/g1 spec table, knee max torque G1 EDU 120 N.m; footnote [1]: largest joint motor's maximum")}
FLIGHT_GATE = float(os.environ.get("SPRINT_FLIGHTGATE", 0.0))          # opt-in RUNNING gate (s): every step after the transient needs a both-feet-unloaded interval >= this (K1: 0.040)
N_CYC, N_TRANS = 12, 2            # cycles simulated; transient cycles excluded from periodicity and speed
TF_MIN = 0.05                     # a flight the project's 20 ms flight rule can see with margin (2.5 x FLIGHT_CONFIRM)
PATIENCE = 15                     # CMA stops early once valid at both dt and not improving for 15 generations
TD_WIN = 0.25                     # the expected foot may land up to (t_f + TD_WIN t_c) before its scheduled touchdown (late flight); earlier = scuff
CLEAR = 0.03                      # swing-foot clearance the scaffold must give (the ceiling's 3 cm)
SLIP_MAX = 0.03                   # m of foot travel while loaded, per stance
PER_TOL = dict(z=0.02, pitch=0.05, roll=0.05, q=0.08, qd=1.5, vx=0.10)   # cycle-to-cycle periodicity tolerances (RMS for q, qd)
SUPPLIED = dict(
    cycle="two-step running cycle, mirror-symmetric legs (left = right shifted half a cycle, roll/yaw sign-flipped), flight phase required (t_f >= %.2f s)" % TF_MIN,
    stride_choice="among the ceiling model's feasible strides at v (its grid of t_c, t_f, h0; dz from spring-mass consistency), with t_c >= pi x 0.040 s (the stance force's half-sine inside the project actuator lag's -3 dB band; kept at 40 ms whatever the plant lag), t_f >= TF_MIN and t_f >= the knee's fold/unfold time (quintic peak rate and torque within KNEE_U of the rating): the stride with the largest actuator margin (min of the max over hip/knee of stance torque utilisation, stance speed/1.1, swing time/available)",
    swing_cart="(default, SWING_MODE=cart) ankle x relative to the hip: quintic from -Lc/2 (rate -v, continuous with the stance) to +Lc/2 with touchdown rate -beta v (beta = the largest of 1, .75, .5, .25, 0 whose inverse-dynamics hip/knee torques stay within KNEE_U of the rating; beta 1 = foot at rest on the ground at liftoff/touchdown); sole height above the floor: CLEAR x 64 s^3 (1-s)^3, the hip's height from the scaffold (other stance compression / ballistic flight), the leg never longer than at touchdown; soles level; numerical IK on the model",
    swing="(legacy joint-space modes) quintic thigh move liftoff->touchdown with the STANCE joint rates as boundary velocities (ground-speed matching: foot at rest on the ground at touchdown; thigh rate continuous at liftoff); knee folds to the ceiling's 3 cm floor-clearance angle during the first flight, stays folded while the other foot is down, unfolds during the last flight; soles level; foot under the hip laterally",
    compression="dz = g tf tc / (2 pi): the stance descent rate at touchdown equals the flight landing speed (spring-mass consistency; the ceiling's half-sine force impulse then equals m g T exactly)",
    feedforward="inverse dynamics of the swing and upper-body reference on the model (base at v with the scaffold's vertical acceleration, no contact; zero for a stance leg and while waiting for a touchdown), advanced by ACT_TAU",
    stance="FORCE CONTROL of a loaded stance leg (load-weighted): F_z = m (g + dz (pi/t_c)^2 sin) + m (K_z (z_ref - z) + D_z (vz_ref - vz)), K_z = (pi/t_c)^2, D_z = 2 pi/t_c; F_xy directed from the foot at the COM and tilted by the attitude moment (M_x = h dF_y, M_y = -h dF_x); residual moment to the CoP clipped to the sole polygon and |M_z| <= 0.05 F_z; |F_xy| <= 0.8 F_z; tau = -J^T [F; M] through the model's foot Jacobian; the leg's joint PD scaled by 0.1",
    trunk_arms="trunk held at the keyframe; ARMS: antiphase shoulder-pitch swing that cancels the FUNDAMENTAL (cycle-frequency harmonic) of the legs' yaw angular momentum about the COM over the reference cycle (w_sp = -L_z1 / dL_z/dw_sp); the search refines both through knots",
    feedback="Raibert: placement (about the COM's neutral point: + COM-ahead-of-hips offset) dx=(vx-v)(tc/2+sqrt(h/g)) with the instantaneous COM vx (constant in flight), dy=vy(tc/2+sqrt(h/g)) with vy averaged over one step (the sway), latched at touchdown and released through the stance; attitude servo in the stance force: moment about the COM M = -I_c (om^2 (attitude - (lean,0,0)) + 2 om pelvis rate), I_c = composite inertia about the COM, om = min(pi/t_c, 0.5/ACT_TAU); swing-leg hip targets corrected for the actual pelvis attitude (world-frame placement)",
    lag_comp="the tracked reference and the feedforward are advanced by ACT_TAU (the known actuator lag); the feedback terms act on the state predicted ACT_TAU ahead (q + qdot tau, COM + v tau, attitude + rate tau)",
    tracker=f"PD per joint, omega {OM_LEG} legs / {OM_UP} upper, zeta 1, I_eff = the joint's diagonal of the model's mass matrix (locked-chain inertia)",
    act_tau=ACT_TAU,
    ratings=RATINGS + {"peak": " (MJCF actuatorfrcrange)", "continuous": " (vendor URDF effort, min with the MJCF peak; a documented-rating choice, no hardware change)"}.get(RATINGS, " (min of MJCF peak, Unitree URDF effort and the vendor's published joint maximum %s; a documented-rating choice, no hardware change)" % (VENDOR_MAX.get("G1"),)),
    flight_gate=(f"ON: every step after the {N_TRANS} transient cycles must contain a both-feet-unloaded interval (each foot's normal force < 1 N) of >= {FLIGHT_GATE*1000:.0f} ms, else NO_FLIGHT" if FLIGHT_GATE > 0 else "off (flight reported, not gated)"),
    clock="event-synchronised cycle phase: advances with time, held at each scheduled touchdown until the expected foot lands (after >= 20 ms unloaded), jumps to the touchdown phase if that foot lands up to t_f + 0.25 t_c early (earlier contacts are scuffs); both legs' stance/swing follow it",
    gates=f"envelope+backemf+OVERSPEED 1.1, actuator lag ACT_TAU (see act_tau), energy closure, no non-foot floor contact, slip <= {SLIP_MAX} m/stance, no fall, periodic over cycles {N_TRANS}..{N_CYC} at 1 ms and 0.5 ms; tolerances {PER_TOL}",
    start_momentum="base velocity solved so the COM moves at (v, -dvy/2, 0) and the angular momentum about the COM is zero",
    start_sway="initial lateral COM velocity -dvy/2, dvy = g T y_off / h (the scaffold's periodic sway with feet under the hips)",
    friction="MuJoCo pyramidal cone, condim 3, mu = 1.0 (model values)",
    stride_def="stride = one gait cycle = two steps",
    contact="STOCK_CONTACT=1: every robot geom keeps the MJCF contact (H1 solref 0.02)" if os.environ.get("STOCK_CONTACT") == "1" else "branch_i.ROBOTS geom_solref override where declared (H1: 0.004 on every geom; stock 0.02)")


def driven(robot):
    Lg = LEG[robot]; legs = [Lg[k].format(s=s) for s in ("right", "left") for k in LEGK if Lg[k]]
    arms = [a.format(s=s) for s in ("right", "left") for a in ARM[robot]]
    return legs + CENTRE[robot] + arms


def mirror_sign(jn): return -1.0 if ("roll" in jn or "yaw" in jn or jn == "torso") else 1.0


def mirror_name(jn):
    if jn.startswith("right_"): return "left_" + jn[6:]
    if jn.startswith("left_"): return "right_" + jn[5:]
    return jn


# ------------------------------------------------------------------ plant
# RATING SET (2026-10-17, track Q3). The plant's torque ceiling per driven joint is a CHOICE BETWEEN DOCUMENTED VENDOR RATINGS, not a
# change of hardware: "peak" (default) = the Menagerie MJCF actuatorfrcrange; "continuous" = Unitree's URDF effort (spec/<robot>.yaml
# tau_max_continuous), applied as min(peak, continuous). Geometry, masses, joint ranges and omega are untouched. Recorded in SUPPLIED.
_RT = {}
def rating_table(robot):
    if robot not in _RT:
        import yaml
        sp = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "spec", f"{robot.lower()}.yaml")))
        col = "continuous" if RATINGS == "published" else RATINGS
        _RT[robot] = {jn: float(rec[f"tau_max_{col}"]["value"]) for jn, rec in sp["joints"].items() if f"tau_max_{col}" in rec}
        if RATINGS == "published" and robot in VENDOR_MAX: _RT[robot] = {jn: min(v, VENDOR_MAX[robot][0]) for jn, v in _RT[robot].items()}
    return _RT[robot]
_PL = {}
def plant(robot, dt, scale=1.0, uid=0):
    """Throw plant (torque motors, held servos for the undriven joints) with the sprint driven set; scale multiplies hip-pitch and
    knee tau AND omega (the ceiling's counterfactual). A distinct copy per (dt, scale)."""
    key = (robot, dt, round(scale, 4), uid)
    if key in _PL: return _PL[key]
    S0 = T.setup(robot, ball=False, mass=0.0, driven=driven(robot))
    S = T.setup(robot, start=np.array(S0["q0"]) + 1e-9*(1 + len(_PL) + 7*uid), ball=False, mass=0.0, driven=driven(robot))
    S["jarr"] = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in S["jarr"].items()}
    if scale != 1.0:
        for i, t in enumerate(S["types"]):
            if any(k in t for k in SCALED_JOINTS): S["jarr"]["tau"][i] *= scale; S["jarr"]["om"][i] *= scale
    if RATINGS != "peak":                       # documented-rating override (NOT a hardware change): see RATINGS
        for i, t in enumerate(S["types"]):
            c = rating_table(robot).get(t)
            if c is not None: S["jarr"]["tau"][i] = min(S["jarr"]["tau"][i], c)
    if os.environ.get("SPRINT_IDEAL"):          # DIAGNOSTIC ONLY: every driven joint x IDEAL in torque and speed
        f_ = float(os.environ["SPRINT_IDEAL"]); S["jarr"]["tau"] *= f_; S["jarr"]["om"] *= f_
    BI.apply_dt(S, dt); S["scale"] = scale; _PL[key] = S; return S


def body(robot, scale=1.0):
    B = SC.body_spec(robot)
    if scale != 1.0:
        for j in ("hip", "knee"): B[j] = dict(B[j], tau=B[j]["tau"]*scale, omega=B[j]["omega"]*scale)
    if RATINGS != "peak":                       # the stride planner sees the same documented rating as the plant (no-op for "continuous")
        rt = rating_table(robot)
        for j, jn in (("hip", "left_hip_pitch_joint"), ("knee", "left_knee_joint")):
            if jn in rt: B[j] = dict(B[j], tau=min(B[j]["tau"], rt[jn]*scale))
    return B


U_RES = 0.60                      # (legacy) utilisation reserve
HR_MAX = float(os.environ.get("SPRINT_HRMAX", 0.98))
KNEE_U = 0.80                     # the swing knee's quintic fold/unfold may use 80% of the rated knee speed
TC_LAG = float(os.environ.get("SPRINT_TCLAG", 0.040))      # the stride's contact-time floor uses the PROJECT lag (40 ms) whatever the plant's lag
def select_stride(B, v, scale=1.0, tc_min=np.pi*TC_LAG, tf_min=TF_MIN, u_res=U_RES):
    """The ceiling model's feasible strides at v (its own grid) -> the LONGEST FLIGHT among those using <= u_res of the envelope (ties:
    the lowest utilisation); if none fits the reserve, the stride with the largest actuator margin (see SUPPLIED)."""
    L = B["L1"] + B["L2"]; best = None; cands = []
    for tc, tf, hr in itertools.product(np.linspace(0.06, 0.40, 18), np.linspace(0.0, 0.20, 9), np.linspace(0.70, 0.98, 8)):
        if tc < tc_min - 1e-9 or tf < tf_min - 1e-9 or hr > HR_MAX + 1e-9: continue
        dr = G*tf*tc/(2*np.pi)/L          # SPRING-MASS CONSISTENCY: stance descent rate at touchdown = flight landing speed g tf/2
        ok, why, st = SC.stance_ok(B, v, tc, tf, hr*L, dr*L, scale=scale)
        if not ok: continue
        ok, why, sw = SC.swing_ok(B, tc, tf, hr*L, st, scale=scale)
        if not ok: continue
        mg = max(st["hip"][0], st["knee"][0], st["hip"][1]/1.1, st["knee"][1]/1.1, sw["t_hip"]/sw["t_sw"], sw["t_knee"]/sw["t_sw"])
        Lc = v*tc; ka_td = SC.ik(Lc/2, -hr*L, B["L1"], B["L2"])[1]; ka_lo = SC.ik(-Lc/2, -hr*L, B["L1"], B["L2"])[1]; kc = max(sw["k_clear"], ka_td, ka_lo)
        A_ = max(kc - ka_td, kc - ka_lo); I_k = B["m_shank"]*(0.4*B["L2"])**2 + B["m_shank"]*B["L2"]**2/12        # the ceiling's lumped knee inertia
        t_unf = max(1.875*A_/(KNEE_U*B["knee"]["omega"]), np.sqrt(5.77*A_*I_k/(KNEE_U*B["knee"]["tau"])))      # quintic fold/unfold: peak rate and peak torque within KNEE_U of the rating
        if tf < t_unf - 1e-9: continue                                            # the knee must fold and unfold inside a flight (clearance while the other foot is down)
        cands.append(dict(util=float(mg), t_unfold=float(t_unf), tc=float(tc), tf=float(tf), h0=float(hr*L), dz=float(dr*L), step=float(v*(tc + tf)), step_hz=float(1/(tc + tf)), stance=st, swing=sw))
    if not cands: return None
    best = min(cands, key=lambda c: c["util"]); best["rule"] = "max actuator margin among strides whose flight fits the knee fold/unfold"
    return best


# ------------------------------------------------------------------ scaffold (numerical IK on the model)
class Leg:
    """Numerical IK of one leg on the model: ankle-body origin relative to the hip-pitch body (pelvis upright at the origin), and
    sole orientation (pitch, roll) equal to the settled stance's (flat)."""
    def __init__(self, S, side):
        m = S["m"]; r = S["robot"]; Lg = LEG[r]; c = SC.CHAINS[r]; self.m = m; self.side = side
        self.names = [Lg[k].format(s=side) for k in ("hp", "hr", "kn", "ap", "ar") if Lg[k]]
        self.jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.names]; self.qa = [int(m.jnt_qposadr[j]) for j in self.jid]
        self.dofs = [int(m.jnt_dofadr[j]) for j in self.jid]
        self.hipb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, c["hipb"].replace("left", side)); self.ankb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, c["ankleb"].replace("left", side))
        self.d = mujoco.MjData(m); self.q_nom = np.array(S["q_settled"]); self.q_nom[:3] = 0; self.q_nom[3:7] = [1, 0, 0, 0]
        d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; mujoco.mj_kinematics(m, d)
        R = d.xmat[self.ankb].reshape(3, 3); self.ori0 = np.array([np.arctan2(-R[2, 0], R[2, 2]), np.arctan2(R[2, 1], R[2, 2])])
        self.y0 = float((d.xpos[self.ankb] - d.xpos[self.hipb])[1])
    def fk(self, qj):
        d = self.d; d.qpos[:] = self.q_nom
        for a, v in zip(self.qa, qj): d.qpos[a] = v
        mujoco.mj_kinematics(self.m, d); p = d.xpos[self.ankb] - d.xpos[self.hipb]; R = d.xmat[self.ankb].reshape(3, 3)
        e = [p[0], p[1], p[2], np.arctan2(-R[2, 0], R[2, 2])] + ([np.arctan2(R[2, 1], R[2, 2])] if len(self.qa) == 5 else [])
        return np.array(e)
    def target(self, x, z, y=None):
        t = [x, self.y0 if y is None else y, z, self.ori0[0]] + ([self.ori0[1]] if len(self.qa) == 5 else []); return np.array(t)
    def jac(self, qj):
        e0 = self.fk(qj); Jm = np.zeros((len(e0), len(qj)))
        for i in range(len(qj)):
            qp = qj.copy(); qp[i] += 1e-5; Jm[:, i] = (self.fk(qp) - e0)/1e-5
        return e0, Jm
    def solve(self, tgt, q_init):
        qj = np.array(q_init, float); n = len(tgt); W = np.array([1.0, 1.0, 1.0] + [0.3]*(n - 3))      # ankle position first, sole orientation second
        lo = np.array([self.m.jnt_range[j][0] for j in self.jid]); hi = np.array([self.m.jnt_range[j][1] for j in self.jid]); lo, hi = lo + 0.02*(hi - lo), hi - 0.02*(hi - lo)
        for _ in range(40):
            e0, Jm = self.jac(qj); err = tgt - e0
            if np.linalg.norm(err) < 1e-6: break
            dq = np.linalg.lstsq(W[:, None]*Jm, W*err, rcond=None)[0]
            qn_ = np.clip(qj + np.clip(dq, -0.2, 0.2), lo, hi)                   # JOINT RANGES respected (the sole tilts where a level sole is out of range)
            if np.max(np.abs(qn_ - qj)) < 1e-7: qj = qn_; break
            qj = qn_
        e0, Jm = self.jac(qj); return qj, float(np.linalg.norm(tgt - e0)), Jm


def mj_(s): s = np.clip(s, 0, 1); return 10*s**3 - 15*s**4 + 6*s**5
def dmj_(s): s = np.clip(s, 0, 1); return 30*s**2 - 60*s**3 + 30*s**4


def scaffold(S, B, st, v, res=0.001, k_extra=0.0, beta=1.0):
    """One gait cycle sampled at res (s): joint reference Q (driven order), QD, stance feedforward FF (N m), swing weight per leg,
    placement maps (dq per dx, dy) per leg, and the attitude-compensation signs. Right leg: stance [0, tc), left: [Tc/2, Tc/2 + tc)."""
    m = S["m"]; types = S["types"]; nT = len(types); idx = {t: i for i, t in enumerate(types)}
    tc, tf, h0, dz = st["tc"], st["tf"], st["h0"], st["dz"]; Tst = tc + tf; Tc = 2*Tst; tsw = Tc - tc; Lc = v*tc; L1, L2 = B["L1"], B["L2"]
    N = int(round(Tc/res)); ph = np.arange(N)*res
    q_up = np.array([S["q_settled"][S["jarr"]["qad"][i]] for i in range(nT)])
    th_td, ka_td = SC.ik(Lc/2, -h0, L1, L2); th_lo, ka_lo = SC.ik(-Lc/2, -h0, L1, L2)
    kc = min(max(float(st["swing"].get("k_clear", 0.0)), ka_td, ka_lo) + k_extra, 0.95*B["knee"]["rng"][1])
    def st_ang(tau):
        xh = -Lc/2 + v*tau; hh = h0 - dz*np.sin(np.pi*tau/tc); return np.array(SC.ik(-xh, -hh, L1, L2))
    t_fold = tf                                                  # the knee folds in the first flight and unfolds in the last (the stride rule guarantees the rate)
    e_ = 1e-5; w_td = (st_ang(e_) - st_ang(0.0))/e_; w_lo = (st_ang(tc) - st_ang(tc - e_))/e_        # stance joint rates at touchdown / liftoff
    def quintic(p0, v0, p1, v1, T_, t_):
        """position at t_ of the quintic with (p, v, a=0) boundary conditions over [0, T_]"""
        s_ = np.clip(t_/T_, 0, 1); V0, V1 = v0*T_, v1*T_; dp = p1 - p0
        h = [1 - 10*s_**3 + 15*s_**4 - 6*s_**5, s_ - 6*s_**3 + 8*s_**4 - 3*s_**5, 10*s_**3 - 15*s_**4 + 6*s_**5, -4*s_**3 + 7*s_**4 - 3*s_**5]
        return p0*h[0] + V0*h[1] + p1*h[2] + V1*h[3]
    def planar(tau):
        """ankle relative to the hip (x forward, z up) at leg-local time tau in [0, Tc)."""
        if tau < tc:
            xh = -Lc/2 + v*tau; hh = h0 - dz*np.sin(np.pi*tau/tc); return -xh, -hh, True
        s = (tau - tc)/tsw; u = tau - tc
        if SWING_MODE == "cart":
            # CARTESIAN SWING: ankle x relative to the hip by a quintic that leaves and meets the ground at rest (rate -v relative to the hip);
            # the SOLE's height above the floor is a smooth bump (64 s^3 (1-s)^3) of height CLEAR; the hip's height over the floor is the
            # scaffold's (other leg's stance compression, or ballistic flight from the last liftoff)
            xa = quintic(-Lc/2, -v, Lc/2, -beta*v, tsw, u)             # continuous with the stance at liftoff; touchdown rate -beta v
            if Tst <= tau < Tst + tc: Hh = h0 - dz*np.sin(np.pi*(tau - Tst)/tc)
            else: w_ = tau - tc if tau < Tst else tau - (Tst + tc); Hh = h0 + G*tf*w_/2 - G*w_*w_/2
            za = min(Hh - CLEAR*64*s**3*(1 - s)**3, h0, np.sqrt(max((0.985*(L1 + L2))**2 - xa*xa, 0.0)))   # never longer than the touchdown leg nor than the leg itself (98.5% of L1+L2)
            return xa, -za, False
        if SWING_MODE == "matched":  # GROUND-SPEED MATCHING: the swing leaves and meets the ground with the stance's joint rates (foot at rest on the ground at liftoff and touchdown)
            th = quintic(th_lo, w_lo[0], th_td, w_td[0], tsw, u)
            if u < t_fold: ka = quintic(ka_lo, 0.0, kc, 0.0, t_fold, u)                                  # fold in the first flight (from liftoff)
            elif u > tsw - t_fold: ka = quintic(kc, 0.0, ka_td, min(w_td[1], 0.0), t_fold, u - (tsw - t_fold))     # unfold in the last flight (no overshoot past the touchdown angle: a flexing end rate would dip the foot below the floor)
            else: ka = kc                                                                                 # folded while the other foot is down
        else:
            if SWING_MODE == "ground": th = th_lo + (th_td - th_lo)*mj_((tau - Tst)/tc)   # thigh held during the flights, moved while the OTHER foot is down
            else: th = th_lo + (th_td - th_lo)*mj_(s)
            ka = ka_lo + (kc - ka_lo)*mj_(2*s) if s < 0.5 else kc + (ka_td - kc)*mj_(2*s - 1)
        return L1*np.sin(th) + L2*np.sin(th - ka), -(L1*np.cos(th) + L2*np.cos(th - ka)), False
    Q = np.repeat(q_up[None], N, 0); FF = np.zeros((N, nT)); SW = np.zeros((N, 2)); PL = np.zeros((N, 2, nT, 2)); ik_err = 0.0
    Mtot = float(sum(m.body_mass[b] for b in S["robot_bodies"])); legs = {}
    d0 = mujoco.MjData(m); d0.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d0)
    hb0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, SC.CHAINS[S["robot"]]["hipb"].replace("left", "right"))
    dcx, dcz = float(d0.subtree_com[1][0] - d0.xpos[hb0][0]), float(d0.subtree_com[1][2] - d0.xpos[hb0][2])
    for li, side in enumerate(("right", "left")):
        lg = Leg(S, side); legs[side] = lg; off = 0.0 if side == "right" else Tst
        qj = np.array([S["q_settled"][a] for a in lg.qa]); cols = [idx[n] for n in lg.names]
        for k, p_ in enumerate(ph):
            tau = (p_ - off) % Tc; x, z, stance = planar(tau)
            th_, ka_ = SC.ik(x, z, L1, L2); g_ = {"hp": -th_, "hr": 0.0, "kn": ka_, "ap": th_ - ka_, "ar": 0.0}       # planar guess (G1/H1 conventions)
            guess = np.array([g_[k_] for k_ in ("hp", "hr", "kn", "ap", "ar") if LEG[S["robot"]][k_]])
            qa_s, ea, Ja = lg.solve(lg.target(x, z), qj)                                            # warm start from the previous sample ...
            if ea > 1e-4 or abs(qa_s[0]) > 1.6:
                qb_s, eb, Jb = lg.solve(lg.target(x, z), guess)                                     # ... or the planar guess if that branch failed
                if eb < ea: qa_s, ea, Ja = qb_s, eb, Jb
            qj, e, Jm = qa_s, ea, Ja; ik_err = max(ik_err, e)
            Q[k, cols] = qj; Ji = np.linalg.pinv(Jm); PL[k, li][cols] = Ji[:, :2]     # dq per (dx, dy) of the ankle, sole orientation held
            SW[k, li] = 0.0 if stance else 1.0
            if stance:                                    # half-sine GRF along the hip-foot line, through the leg Jacobian at the sole point
                Fz = np.pi*Mtot*G*Tst/(2*tc)*np.sin(np.pi*tau/tc); xh = -Lc/2 + v*tau; hh = h0 - dz*np.sin(np.pi*tau/tc)
                F = np.array([Fz*(xh + dcx)/(hh + dcz), 0.0, Fz]); d = lg.d; d.qpos[:] = lg.q_nom     # directed at the COM (dcx, dcz: COM relative to the hip, from the model)
                for a, vq in zip(lg.qa, qj): d.qpos[a] = vq
                mujoco.mj_forward(m, d); jp = np.zeros((3, m.nv)); pt = d.xpos[lg.ankb] - np.array([0, 0, B["hf"]])
                mujoco.mj_jac(m, d, jp, None, pt, lg.ankb)
                FF[k, cols] = -(jp[:, lg.dofs].T @ F)
    # smooth the swing weight over 20 ms ramps (stance/swing switching of the attitude compensation)
    w = max(1, int(round(0.02/res))); ker = np.ones(w)/w
    SW = np.stack([np.convolve(np.concatenate([SW[-w:, i], SW[:, i], SW[:w, i]]), ker, "same")[w:-w] for i in range(2)], 1)
    QD = (np.roll(Q, -1, 0) - np.roll(Q, 1, 0))/(2*res)
    arm_info = None
    if ARM_MODE == "counter":
        # ARM COUNTER-ROTATION from angular-momentum balance: the legs' yaw angular momentum about the COM over the cycle (reference
        # kinematics, base moving at v) is cancelled by antiphase shoulder-pitch rates, w_sp(phi) = -L_z,legs(phi) / (dL_z/dw_sp)
        sp = [idx[n] for n in types if "shoulder_pitch" in n]; sgn = np.array([1.0 if types[i].startswith("right") else -1.0 for i in sp])
        dd = mujoco.MjData(m); Lz = np.zeros(N)
        def angmom(qj, qdj, extra=None):
            dd.qpos[:] = S["q_settled"]; dd.qpos[:3] = 0; dd.qpos[3:7] = [1, 0, 0, 0]; dd.qpos[S["jarr"]["qad"]] = qj
            vv = np.zeros(m.nv); vv[0] = v; vv[S["jarr"]["dof"]] = qdj
            if extra is not None: vv[S["jarr"]["dof"][sp]] += extra
            dd.qvel[:] = vv; mujoco.mj_forward(m, dd); mujoco.mj_subtreeVel(m, dd); return dd.subtree_angmom[1].copy()
        for k in range(N): Lz[k] = angmom(Q[k], QD[k])[2]
        b = np.array([angmom(Q[k], QD[k], sgn)[2] - Lz[k] for k in range(0, N, max(1, N//20))]).mean()
        F_ = np.fft.rfft(Lz); F_[0] = 0; F_[2:] = 0; Lz1 = np.fft.irfft(F_, N)     # the cycle's FUNDAMENTAL (the arms cannot follow the impact transients; the stance foot takes those)
        w_sp = -Lz1/b; ang = np.cumsum(w_sp)*res; ang -= ang.mean()
        for j, i in enumerate(sp): Q[:, i] = q_up[i] + sgn[j]*ang
        QD = (np.roll(Q, -1, 0) - np.roll(Q, 1, 0))/(2*res)
        arm_info = dict(Lz_ptp=float(np.ptp(Lz)), Lz1_ptp=float(np.ptp(Lz1)), dLz_dw=float(b), sp_ptp=float(np.ptp(ang)), sp_rate_max=float(np.abs(w_sp).max()))
    # attitude compensation: the joint (per leg) whose rotation reproduces a pelvis pitch / roll / yaw, with its sign
    # INVERSE-DYNAMICS FEEDFORWARD of the reference motion on the model (base moving at v, vertical acceleration of the scaffold's
    # stance compression / ballistic flight, no contact): the torques that move the limbs as planned; the stance support is the GRF term
    wq = max(1, int(round(0.5*TC_LAG/res))); kq = np.ones(wq)/wq         # accelerations smoothed over 20 ms (half the project actuator lag), the same scaffold for every plant
    QDs = np.stack([np.convolve(np.concatenate([QD[-wq:, i], QD[:, i], QD[:wq, i]]), kq, "same")[wq:-wq] for i in range(nT)], 1)
    QDD = (np.roll(QDs, -1, 0) - np.roll(QDs, 1, 0))/(2*res); ID = np.zeros((N, nT))
    if ID_FF:
        dd = mujoco.MjData(m); qa_, dof_ = S["jarr"]["qad"], S["jarr"]["dof"]
        for k in range(N):
            tr_ = (k*res) % Tst; az = dz*(np.pi/tc)**2*np.sin(np.pi*tr_/tc) if tr_ < tc else -G
            dd.qpos[:] = S["q_settled"]; dd.qpos[:3] = [0, 0, 5.0]; dd.qpos[3:7] = [1, 0, 0, 0]; dd.qpos[qa_] = Q[k]     # in the air: no contact in the inverse
            dd.qvel[:] = 0; dd.qvel[0] = v; dd.qvel[dof_] = QD[k]; dd.qacc[:] = 0; dd.qacc[2] = az; dd.qacc[dof_] = QDD[k]
            mujoco.mj_inverse(m, dd); ID[k] = dd.qfrc_inverse[dof_]
            for li_, sd_ in enumerate(("right", "left")):      # a STANCE leg's joints get no free-base inverse dynamics (its motion is the body moving over the pinned foot; the GRF term carries it)
                if ((k*res - (0.0 if li_ == 0 else Tst)) % Tc) < tc: ID[k, [i_ for i_, t_ in enumerate(types) if t_.startswith(sd_) and any(x_ in t_ for x_ in ("hip", "knee", "ankle"))]] = 0.0
    FFg = FF.copy(); FF = FF + ID
    comp = []
    for side in ("right", "left"):
        Lg = LEG[S["robot"]]
        for ax, k in ((1, "hp"), (0, "hr"), (2, "hy")):
            if not Lg[k]: continue
            comp.append((idx[Lg[k].format(s=side)], ax, side))
    # ATTITUDE SERVO gains from the body: composite inertia of everything except the stance leg about the stance hip (roll = x, pitch = y,
    # yaw = z through the hip), at the keyframe; omega_att = pi/t_c (an attitude error is corrected within one contact)
    d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d); lg = legs["right"]
    leg_b = set(); b0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, SC.CHAINS[S["robot"]]["hipb"].replace("left", "right"))
    for b in S["robot_bodies"]:
        x_ = b
        while x_ > 0 and x_ != b0: x_ = int(m.body_parentid[x_])
        if x_ == b0: leg_b.add(b)
    hip = d.xpos[lg.hipb].copy(); I_att = np.zeros(3)
    for b in S["robot_bodies"]:
        if b in leg_b or b == 0: continue
        r = d.xipos[b] - hip; Rb = d.ximat[b].reshape(3, 3); Ic = Rb @ np.diag(m.body_inertia[b]) @ Rb.T
        for ax in range(3): I_att[ax] += float(Ic[ax, ax] + m.body_mass[b]*(r @ r - r[ax]**2))
    Mf = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, Mf, d.qM)
    ieff = np.array([Mf[dof, dof] for dof in S["jarr"]["dof"]]) if IEFF_MODE == "locked" else np.array([1.0/S["Minv_key"][dof, dof] for dof in S["jarr"]["dof"]])
    # CENTROIDAL quantities for the force-controlled stance: COM height at touchdown (right sole on the floor at phase 0), composite
    # inertia about the COM, sole polygon about the ankle (CoP limits)
    dq_ = mujoco.MjData(m); dq_.qpos[:] = S["q_settled"]; dq_.qpos[:3] = 0; dq_.qpos[3:7] = [1, 0, 0, 0]; dq_.qpos[S["jarr"]["qad"]] = Q[0]; mujoco.mj_forward(m, dq_)
    zc_td = float(dq_.subtree_com[1][2] - min(P.geom_low(m, dq_, g) for g in S["feet"][1]))
    cc = d.subtree_com[1].copy(); I_c = np.zeros(3)
    for b in S["robot_bodies"]:
        if b == 0: continue
        r = d.xipos[b] - cc; Rb = d.ximat[b].reshape(3, 3); Ic = Rb @ np.diag(m.body_inertia[b]) @ Rb.T
        for ax in range(3): I_c[ax] += float(Ic[ax, ax] + m.body_mass[b]*(r @ r - r[ax]**2))
    ab = lg.ankb; pts = []
    for g in S["feet"][1]:
        c_ = d.geom_xpos[g] - d.xpos[ab]; Rg = d.geom_xmat[g].reshape(3, 3); sz = m.geom_size[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            for sx in (-1, 1):
                for sy in (-1, 1): pts.append(c_ + Rg @ np.array([sx*sz[0], sy*sz[1], 0.0]))
        else: pts.append(c_)
    pts = np.array(pts); sole = dict(heel=float(pts[:, 0].min()), toe=float(pts[:, 0].max()), half_w=float(0.5*np.ptp(pts[:, 1])), hf=float(d.xpos[ab][2] - min(P.geom_low(m, d, g) for g in S["feet"][1])))
    om_att = np.pi/tc if OMATT_MODE == "tc" else min(np.pi/tc, 0.5/ACT_TAU)
    # PERIODIC LATERAL SWAY of the scaffold (feet under the hips): each stance's force, along the foot->COM line, gives the COM a lateral
    # velocity change dvy = g T y_off / h (impulse m g T per step); the running start takes the sway's value before a right touchdown, -dvy/2
    y_off = abs(float(d.xpos[lg.ankb][1] - d.subtree_com[1][1])); h_c = float(d.subtree_com[1][2] - d.xpos[lg.ankb][2]); dvy = G*Tst*y_off/h_c
    return dict(Q=Q, QD=QD, FF=FF, SW=SW, PL=PL, N=N, I_att=I_att, om_att=om_att, Kp_att=om_att**2*I_att, Kd_att=2*om_att*I_att, ieff=ieff, ID=ID, FFg=FFg, zc_td=zc_td, I_c=I_c, sole=sole, arm=arm_info, y_off=y_off, dvy=dvy, res=res, Tc=Tc, Tst=Tst, tc=tc, tf=tf, h0=h0, dz=dz, v=v, Lc=Lc, ik_err=ik_err, comp=comp, q_up=q_up, M=Mtot)


def clearance(S, R):
    """Kinematic check of the scaffold on the model: lowest point of each SWING foot above the floor over the cycle, with the hip
    height given by the stance foot (sole on the floor) or by the ballistic flight from the last liftoff. Returns per-leg minima
    (excluding 15 ms around that leg's own touchdown/liftoff) and the profile."""
    m = S["m"]; d = mujoco.MjData(m); qa = S["jarr"]["qad"]; N = R["N"]; res = R["res"]; tc, tf, Tst = R["tc"], R["tf"], R["Tst"]
    hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, SC.CHAINS[S["robot"]]["hipb"].replace("left", "right"))
    low = np.zeros((N, 2)); zh = np.zeros(N)
    for k in range(N):
        d.qpos[:] = S["q_settled"]; d.qpos[:3] = 0; d.qpos[3:7] = [1, 0, 0, 0]; d.qpos[qa] = R["Q"][k]; mujoco.mj_kinematics(m, d)
        zh[k] = d.xpos[hb][2]; low[k] = [min(P.geom_low(m, d, g) for g in S["feet"][1]), min(P.geom_low(m, d, g) for g in S["feet"][0])]   # (right, left)
    H = np.zeros(N)
    for k in range(N):
        ph = k*res; tr_ = ph % Tst; leg = 0 if (ph % (2*Tst)) < Tst else 1      # leg 0 = right stance in [0, Tst)
        if tr_ < tc: H[k] = -low[k, leg]                                         # pelvis height so that the stance sole is on the floor
    for k in range(N):
        ph = k*res; tr_ = ph % Tst
        if tr_ >= tc:
            k_lo = int(round((ph - tr_ + tc)/res)) - 1; u = tr_ - tc; H[k] = H[k_lo % N] + G*tf*u/2 - G*u*u/2
    cl = low + H[:, None]; mins = []
    for li in (0, 1):
        off = 0.0 if li == 0 else Tst; mask = np.array([((k*res - off) % (2*Tst)) > tc + 0.015 and ((k*res - off) % (2*Tst)) < 2*Tst - 0.015 for k in range(N)])
        mins.append(float(cl[mask, li].min()) if mask.any() else 0.0)
    return dict(min=mins, profile=cl)


# ------------------------------------------------------------------ parameters
NK = 8
def layout(S):
    """Base joints carrying free knots: right leg + right arm over the full cycle (nk), centre joints over half a cycle (nk/2)."""
    types = S["types"]; base = [i for i, t in enumerate(types) if t.startswith("right_")]; centre = [i for i, t in enumerate(types) if not t.startswith(("right_", "left_"))]
    return base, centre


def n_params(S): base, centre = layout(S); return len(base)*NK + len(centre)*(NK//2) + 3


def knot_table(S, z, N):
    """z knots -> per-phase torque offsets (activation fractions) on the N-sample cycle grid, mirror-symmetric."""
    types = S["types"]; nT = len(types); base, centre = layout(S); idx = {t: i for i, t in enumerate(types)}
    K = np.zeros((nT, NK)); o = 0
    for i in base:
        K[i] = z[o:o + NK]; o += NK
        j = idx[mirror_name(types[i])]; K[j] = mirror_sign(types[i])*np.roll(K[i], NK//2)
    for i in centre:
        h = z[o:o + NK//2]; o += NK//2; K[i] = np.concatenate([h, mirror_sign(types[i])*h])
    g = np.arange(N)/N*NK; k0 = np.floor(g).astype(int) % NK; k1 = (k0 + 1) % NK; f = g - np.floor(g)
    return (K[:, k0]*(1 - f) + K[:, k1]*f).T                      # (N, nT)


def rescale_knots(S, z, factor):
    """Knots are activation fractions of each joint's rated torque: a warm start taken from a body whose hip-pitch/knee torque was
    x1 onto a body with x factor keeps the same TORQUES by dividing those joints' knots by factor."""
    z = np.array(z, float); base, centre = layout(S); o = 0
    for i in base:
        if any(k in S["types"][i] for k in ("hip_pitch", "knee")): z[o:o + NK] /= factor
        o += NK
    return z


def unpack(S, z):
    z = np.asarray(z, float); n = len(z) - 3
    return dict(x=z[:n], amp=1.0 + 0.2*float(z[n]), tscale=1.0 + 0.15*float(z[n + 1]), lean=0.05 + 0.25*float(z[n + 2]))   # amp 0.8..1.2, tscale 0.85..1.15, lean -0.2..0.3 rad


def z_prior(S): return np.zeros(n_params(S))


# ------------------------------------------------------------------ rollout
def quat_rpy(q):
    w, x, y, z = q
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)); pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1)); yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


def rpy_quat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r/2), np.sin(r/2), np.cos(p/2), np.sin(p/2), np.cos(y/2), np.sin(y/2)
    return np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])


def initial_state(S, R, p):
    """Running start at the flight apex before the right touchdown: phase Tc - tf/2, base at v, lowest right-foot point g tf^2/8 above
    the floor, joint velocities from the scaffold."""
    m = S["m"]; J = S["jarr"]; Q, QD = cycle_ref(R, p); N = R["N"]; ph0 = R["Tc"] - R["tf"]/2; k = int(round(ph0/R["res"])) % N
    q = np.array(S["q_settled"]); q[:3] = [0, 0, 1.0]; q[3:7] = rpy_quat(0.0, p["lean"], 0.0); q[J["qad"]] = Q[k]
    d = mujoco.MjData(m); qt_ = q.copy(); qt_[J["qad"]] = Q[0]; d.qpos[:] = qt_; mujoco.mj_kinematics(m, d)   # the RIGHT TOUCHDOWN configuration (phase 0)
    low = min(P.geom_low(m, d, g) for g in S["feet"][1]); q[2] += -low + G*R["tf"]**2/8      # its sole reaches the floor after the ballistic drop of half a flight
    v = np.zeros(m.nv); v[J["dof"]] = QD[k]/p["tscale"]
    # base velocity (6) solved so that the COM moves at (v, -dvy/2, 0) and the whole-body angular momentum about the COM is zero
    tgt = np.array([R["v"], -R["dvy"]/2, 0.0, 0.0, 0.0, 0.0])
    def mom(vv):
        d.qpos[:] = q; d.qvel[:] = vv; mujoco.mj_forward(m, d); mujoco.mj_subtreeVel(m, d); return np.concatenate([d.subtree_linvel[1], d.subtree_angmom[1]])
    f0 = mom(v); A = np.zeros((6, 6))
    for i in range(6):
        vp = v.copy(); vp[i] += 1.0; A[:, i] = mom(vp) - f0
    v[:6] = np.linalg.solve(A, tgt - f0)
    return q, v, ph0


def cycle_ref(R, p):
    """Scaffold scaled by amp about the cycle mean (stride amplitude), per joint."""
    if p["amp"] == 1.0: return R["Q"], R["QD"]
    mu = R["Q"].mean(0)[None]; return mu + p["amp"]*(R["Q"] - mu), p["amp"]*R["QD"]


def rollout(S, R, p, n_cyc=N_CYC, record=False):
    """EVENT-DRIVEN per-leg clock (each leg's scaffold time restarts at its own touchdown and liftoff), force-controlled stance
    (centroidal GRF + attitude moment through the stance leg's Jacobian), PD + inverse-dynamics swing with Raibert placement, upper body
    PD on the scaffold; torque knots on the cycle phase; the throw's actuator model (40 ms lag, torque-speed envelope, back-EMF brake)."""
    m = S["m"]; d = mujoco.MjData(m); J = S["jarr"]; nT = len(S["types"]); dt = m.opt.timestep; ar = np.arange(nT)
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    tau_v = J["tau"]; om_v = J["om"]
    for i in range(nT):
        aid = J["aid"][i]; m.actuator_ctrlrange[aid][:] = [-tau_v[i], tau_v[i]]; m.actuator_forcerange[aid][:] = m.actuator_ctrlrange[aid]
        m.jnt_actfrcrange[S["tmap"][S["types"][i]][0][0]][:] = m.actuator_ctrlrange[aid]
    q0, v0, ph0 = initial_state(S, R, p); d.qpos[:] = q0; d.qvel[:] = v0; mujoco.mj_forward(m, d)
    for (a_, dof, qad, tau, kp_, kd_) in S["held"]: d.ctrl[a_] = q0[qad]
    Q, QD = cycle_ref(R, p); KN = knot_table(S, p["x"], R["N"]); ID = R["ID"]; PL = R["PL"]; N = R["N"]; res = R["res"]; ts = p["tscale"]
    Tc, Tst, tc, tf = R["Tc"], R["Tst"], R["tc"], R["tf"]; tsw = Tc - tc; nadv = int(round(ACT_TAU/(res*ts)))
    ieff = R["ieff"]; om = np.array([OM_LEG if any(k_ in t for k_ in ("hip", "knee", "ankle")) else OM_UP for t in S["types"]]); kp = om**2*ieff; kd = 2*om*ieff
    legcols = [np.array([i for i, t in enumerate(S["types"]) if t.startswith(sd) and any(k_ in t for k_ in ("hip", "knee", "ankle"))]) for sd in ("right", "left")]
    is_leg = [np.isin(ar, legcols[0]), np.isin(ar, legcols[1])]; offs = (0.0, Tst)
    h_com = float(d.subtree_com[1][2]); Kc = tc/2 + np.sqrt(h_com/G); a_lp = 1.0 - np.exp(-dt/(Tst*ts))
    alpha = 1.0 - np.exp(-dt/ACT_TAU); act = (ID[int(ph0/res) % N])/tau_v
    E0 = float(d.energy[0] + d.energy[1]); W_act = W_con = W_pas = 0.0; work = np.zeros(nT); overj = np.zeros(nT)
    T_end = n_cyc*Tc*ts; nst = int(round(T_end/dt)); fl = S["floor"]; feet = S["feet"]; Mg = R["M"]*G
    mujoco.mj_subtreeVel(m, d); vf = np.array([d.subtree_linvel[1][0], 0.0])
    z_nom = float(q0[2]); fell = False; t_fall = None; nonfoot = False
    cg = [g for g in range(m.ngeom) if m.geom_bodyid[g] in S["robot_bodies"] and g not in S["foot_geoms"] and m.geom_group[g] == 3] if S["robot"] == "G1" else []
    ankb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in BI.ROBOTS[S["robot"]]["foot_bodies"]]      # [left, right]
    hipb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, SC.CHAINS[S["robot"]]["hipb"].replace("left", sd)) for sd in ("left", "right")]
    # per-leg clocks (li 0 = right = feet[1]); start at the flight apex before the right touchdown
    phi = ph0 - dt/ts; phi_hold = Tc - 1e-9; prev_mode = ["swing", "swing"]; t_ev = [0.0, 0.0]          # unwrapped cycle phase; next touchdown = right at Tc
    loaded = [False, False]; slip0 = [None, None]; slip_max = 0.0; tds = [[], []]; los = [[], []]; flights = []; fl_end = []; both_air = 0.0; t_air = [1.0, 1.0]
    dlatch = [np.zeros(2), np.zeros(2)]; bounds = []; ffl = [0.0, 0.0]; n_early = 0
    omz = np.pi/(tc*ts); Kz, Dz = omz**2, 2*omz; Ka, Da = R["om_att"]**2, 2*R["om_att"]; lean = p["lean"]; comp = R["comp"]
    trace = dict(qpos=[], t=[], com=[], vcom=[], rpy=[], q=[], qd=[], tq=[], load=[], qt=[], L=[], mode=[], cmd=[], act=[], pd=[], idf=[], tst=[], gsc=[]) if record else None
    def bound(t):
        r_, p_, y_ = quat_rpy(d.qpos[3:7]); mujoco.mj_subtreeVel(m, d)
        bounds.append(dict(t=t, z=float(d.qpos[2]), roll=float(r_), pitch=float(p_), yaw=float(y_), q=d.qpos[J["qad"]].copy(), qd=d.qvel[J["dof"]].copy(),
                           vx=float(d.subtree_linvel[1][0]), comx=float(d.subtree_com[1][0]), y=float(d.subtree_com[1][1])))
    for s in range(nst):
        t = s*dt
        # ---- EVENT-SYNCHRONISED CYCLE CLOCK phi: advances with time, is HELD at a scheduled touchdown until that foot lands, and is
        # set to the touchdown phase when the expected foot lands early (late flight); stance/swing of both legs follow phi
        phi = min(phi + dt/ts, phi_hold) if phi < phi_hold else phi_hold
        tl = [phi % Tc, (phi - Tst) % Tc]; mode = ["stance" if tl[li] < tc else "swing" for li in (0, 1)]
        wsw = [max(0.0, min(1.0, (tl[li] - tc)/0.02)) if tl[li] >= tc else max(0.0, 1.0 - tl[li]/0.02) for li in (0, 1)]
        for li in (0, 1):
            if mode[li] == "swing" and prev_mode[li] == "stance": dlatch[li] = np.zeros(2)
        prev_mode = mode
        k0 = int((phi % Tc)/res) % N; g = [k0, k0]; kk = np.full(nT, k0); kka = (kk + (0 if phi >= phi_hold else nadv)) % N      # no look-ahead into a stance that has not begun
        r_, p_, y_ = quat_rpy(d.qpos[3:7]); mujoco.mj_subtreeVel(m, d); vc = d.subtree_linvel[1]; vf += a_lp*(vc[:2] - vf)
        vs_ = vc if V_INST else vf
        cy_, sy_ = np.cos(y_), np.sin(y_); vh = (cy_*vs_[0] + sy_*vs_[1], -sy_*vf[0] + cy_*vf[1])      # sagittal: instantaneous COM velocity (constant in flight, where the touchdown is decided); lateral: averaged over a step (the sway)
        rc = d.subtree_com[1][:2] - 0.5*(d.xpos[hipb[0]][:2] + d.xpos[hipb[1]][:2]); xch = cy_*rc[0] + sy_*rc[1]       # COM ahead of the hips (heading frame)
        dxy = np.array([xch + (vh[0] - R["v"])*Kc, (vh[1] + R["v"]*sy_)*Kc])                  # Raibert placement about the COM's neutral point (heading frame)
        qt = Q[kka, ar].copy(); qdt = QD[kka, ar]/ts
        for li in (0, 1):
            if mode[li] == "swing": dlatch[li] = dxy.copy(); qt += wsw[li]*(PL[(g[li] + nadv) % N, li] @ dxy)
            else: qt += max(0.0, 1.0 - tl[li]/tc)*(PL[(g[li] + nadv) % N, li] @ dlatch[li])      # latched at touchdown, released through the stance
        att_act = (r_, p_, y_)
        for (i, ax, side) in comp:
            li = 0 if side == "right" else 1; qt[i] -= wsw[li]*att_act[ax] + (1 - wsw[li])*(lean if ax == 1 else 0.0)
        qn = d.qpos[J["qad"]]; wv = d.qvel[J["dof"]]
        tst = np.zeros(nT); gsc = np.ones(nT); cvec = d.subtree_com[1]; Rp = d.xmat[1].reshape(3, 3); wW = Rp @ d.qvel[3:6]
        if PREDICT:                                # LAG COMPENSATION: feedback acts on the state predicted one actuator lag ahead (constant-rate prediction)
            qn = qn + wv*ACT_TAU; cvec = cvec + np.asarray(vc)*ACT_TAU; r_, p_, y_ = r_ + wW[0]*ACT_TAU, p_ + wW[1]*ACT_TAU, y_ + wW[2]*ACT_TAU
        for li in (0, 1):
            fi = 1 - li
            if mode[li] != "stance": continue
            w = min(1.0, ffl[fi]/(0.15*Mg))
            if w <= 0: continue
            tcs = tc*ts; tau_s = min(tl[li]*ts + (ACT_TAU if PREDICT else 0.0), tcs); sn = np.sin(np.pi*min(tau_s, tcs)/tcs); cs = np.cos(np.pi*min(tau_s, tcs)/tcs)
            pf = d.xpos[ankb[fi]] - np.array([0, 0, R["sole"]["hf"]])
            z_ref = R["zc_td"] - R["dz"]*sn; vz_ref = -R["dz"]*np.pi/tcs*cs
            Fz = max(0.0, R["M"]*(G + R["dz"]*(np.pi/tcs)**2*sn) + R["M"]*(Kz*(z_ref - cvec[2]) + Dz*(vz_ref - vc[2]))); hz = max(cvec[2] - pf[2], 0.2)
            Mc = R["I_c"]*(-Ka*np.array([r_, p_ - lean, y_])) - Da*(np.array(d.subtree_angmom[1]) if DAMP_L else R["I_c"]*wW)     # attitude PD; damping on the WHOLE-BODY angular momentum (what the flight conserves)
            F = np.array([Fz*(cvec[0] - pf[0])/hz - Mc[1]/hz, Fz*(cvec[1] - pf[1])/hz + Mc[0]/hz, Fz]); fxy = np.hypot(F[0], F[1])
            if fxy > 0.8*Fz: F[:2] *= 0.8*Fz/max(fxy, 1e-9)
            mp = Mc - np.cross(pf - cvec, F)
            mp[1] = np.clip(mp[1], -R["sole"]["toe"]*Fz, -R["sole"]["heel"]*Fz); mp[0] = np.clip(mp[0], -R["sole"]["half_w"]*Fz, R["sole"]["half_w"]*Fz); mp[2] = np.clip(mp[2], -0.05*Fz, 0.05*Fz)
            jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv)); mujoco.mj_jac(m, d, jp, jr, pf, ankb[fi]); cols = legcols[li]
            tst[cols] += w*(-(jp[:, J["dof"][cols]].T @ F + jr[:, J["dof"][cols]].T @ mp)); gsc[cols] = 1.0 - (1.0 - STANCE_G)*w
        idr = ID[kka, ar]
        if phi >= phi_hold: idr = np.where(is_leg[0] | is_leg[1], 0.0, idr)        # waiting for the touchdown: the legs hold the touchdown pose (no swing dynamics)
        cmd = (idr + tst + gsc*(kp*(qt - qn) + kd*(qdt - wv)))/tau_v + KN[kk, ar]
        act += alpha*(cmd - act)
        frac = np.maximum(0.0, 1.0 - np.abs(wv)/om_v); hi = np.where(wv > 0, tau_v*frac, tau_v); lo = np.where(wv < 0, tau_v*frac, tau_v)
        tq = np.clip(act*tau_v, -lo, hi); overj = np.maximum(overj, np.abs(wv)/om_v)
        d.ctrl[J["aid"]] = tq; BI.backemf_brake(m, d, S)
        qv0 = d.qvel.copy(); mujoco.mj_step(m, d); qm = 0.5*(qv0 + d.qvel)
        W_act += float(np.dot(d.actuator_force, qm[S["act_dof"]]))*dt; W_con += float(np.dot(d.qfrc_constraint, qm))*dt
        W_pas += (float(np.dot(d.qfrc_passive, qm)) + float(np.dot(d.qfrc_applied, qm)))*dt
        pw = tq*d.qvel[J["dof"]]; work += np.where(pw > 0, pw, 0.0)*dt
        ff = [0.0, 0.0]
        for ci in range(d.ncon):
            c = d.contact[ci]
            if fl not in (c.geom1, c.geom2): continue
            fv = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fv); o = c.geom2 if c.geom1 == fl else c.geom1
            if o in feet[0]: ff[0] += fv[0]
            elif o in feet[1]: ff[1] += fv[0]
            elif o in S["robot_geoms"]: nonfoot = True
        ffl = ff
        if cg and s % 5 == 0 and min(P.geom_low(m, d, g_) for g_ in cg) < -0.002: nonfoot = True
        for i in (0, 1):                                   # i = foot index (0 left, 1 right); leg li = 1 - i
            li = 1 - i; on = ff[i] > 0.05*Mg; px = d.xpos[ankb[i]][:2].copy()
            if not on: t_air[i] += dt
            if on and not loaded[i]:
                fresh = t_air[i] >= BI.FLIGHT_CONFIRM - 1e-12; t_air[i] = 0.0
                if fresh:
                    tds[i].append(t); slip0[i] = px
                    td_ph = phi_hold + 1e-9                                                     # the expected touchdown's phase (unwrapped)
                    exp_leg = 0 if int(round(td_ph/Tst)) % 2 == 0 else 1
                    if li == exp_leg and phi >= td_ph - (tf + TD_WIN*tc):                       # TOUCHDOWN EVENT of the expected leg: phi jumps to the touchdown phase
                        phi = td_ph; phi_hold = td_ph + Tst - 1e-9; t_ev[li] = t
                        if li == 0: bound(t)                                                    # cycle boundary = right touchdown
                    else: n_early += 1                                                          # a scuff (wrong leg or too early)
            if not on and loaded[i]: los[i].append(t)
            if on: t_air[i] = 0.0
            if on and slip0[i] is not None and mode[li] == "stance": slip_max = max(slip_max, float(np.linalg.norm(px - slip0[i])))
            loaded[i] = on
        if ff[0] < 1.0 and ff[1] < 1.0: both_air += dt
        elif both_air > 0: flights.append(both_air); fl_end.append(t); both_air = 0.0
        if record and s % 2 == 0:
            trace["qpos"].append(d.qpos.copy()); trace["t"].append(t); trace["com"].append(d.subtree_com[1].copy()); trace["vcom"].append(np.array(vc)); trace["rpy"].append((r_, p_, y_))
            trace["q"].append(qn.copy()); trace["qd"].append(wv.copy()); trace["tq"].append(tq.copy()); trace["load"].append(np.array(ff)/Mg); trace["qt"].append(qt.copy())
            trace["L"].append(d.subtree_angmom[1].copy()); trace["mode"].append([1 if mo == "stance" else 0 for mo in mode]); trace["cmd"].append(cmd*tau_v); trace["act"].append(act*tau_v); trace["pd"].append(kp*(qt - qn) + kd*(qdt - wv)); trace["idf"].append(idr.copy()); trace["tst"].append(tst.copy()); trace["gsc"].append(np.array(gsc, float)*np.ones(nT))
        if d.qpos[2] < 0.6*z_nom or abs(r_) > 0.7 or abs(p_ - lean) > 0.7 or not np.isfinite(d.qpos[2]):
            fell = True; t_fall = t; break
    E1 = float(d.energy[0] + d.energy[1]); closure = (E1 - E0) - W_act - W_con - W_pas; ctol = max(5.0, 0.02*(abs(W_act) + abs(W_con)))
    t_run = (t_fall if fell else nst*dt)
    nb = len(bounds); per = None; per_n = 0; v_avg = 0.0
    if nb >= N_TRANS + 2:
        errs = []
        for a, b in zip(bounds[N_TRANS:-1], bounds[N_TRANS + 1:]):
            errs.append(dict(z=abs(b["z"] - a["z"])/PER_TOL["z"], pitch=abs(b["pitch"] - a["pitch"])/PER_TOL["pitch"], roll=abs(b["roll"] - a["roll"])/PER_TOL["roll"],
                             q=float(np.sqrt(np.mean((b["q"] - a["q"])**2)))/PER_TOL["q"], qd=float(np.sqrt(np.mean((b["qd"] - a["qd"])**2)))/PER_TOL["qd"], vx=abs(b["vx"] - a["vx"])/PER_TOL["vx"]))
        per = max(max(e.values()) for e in errs)
        for e in reversed(errs):
            if max(e.values()) <= 1.0: per_n += 1
            else: break
        v_avg = (bounds[-1]["comx"] - bounds[N_TRANS]["comx"])/(bounds[-1]["t"] - bounds[N_TRANS]["t"])
    elif nb >= 2: v_avg = (bounds[-1]["comx"] - bounds[0]["comx"])/max(bounds[-1]["t"] - bounds[0]["t"], 1e-9)
    overspeed = float(overj.max()); reasons = []
    if fell: reasons.append("FELL")
    if overspeed > BI.OVERSPEED_TOL: reasons.append("OVERSPEED")
    if nonfoot: reasons.append("NONFOOT_CONTACT")
    if abs(closure) > ctol: reasons.append("ENERGY_CLOSURE")
    if slip_max > SLIP_MAX: reasons.append("SLIP")
    yaw_end = bounds[-1]["yaw"] if bounds else 0.0
    # PER-STEP FLIGHT: every fresh touchdown (either foot) after the transient must be preceded, since the previous touchdown, by a
    # both-feet-unloaded interval; the longest such interval per step is that step's flight
    t_tr = bounds[N_TRANS]["t"] if len(bounds) > N_TRANS else None; step_fl = []
    if t_tr is not None:
        tdall = sorted(tds[0] + tds[1])
        for a_, b_ in zip(tdall[:-1], tdall[1:]):
            if b_ <= t_tr + 1e-9: continue
            step_fl.append(max([f_ for f_, e_ in zip(flights, fl_end) if a_ < e_ <= b_ + 1e-9], default=0.0))
    flight_min_step = float(min(step_fl)) if step_fl else 0.0
    if FLIGHT_GATE > 0 and not fell and flight_min_step < FLIGHT_GATE: reasons.append("NO_FLIGHT")
    if abs(yaw_end) > 0.35: reasons.append("HEADING")
    viol = 0.0
    if fell: viol += 5.0 + 10.0*(1 - t_run/T_end)
    if "OVERSPEED" in reasons: viol += 1.0 + 5.0*(overspeed - BI.OVERSPEED_TOL)
    if nonfoot: viol += 3.0
    if "ENERGY_CLOSURE" in reasons: viol += 1.0 + min(abs(closure)/ctol, 5.0)
    if "SLIP" in reasons: viol += 1.0 + 20.0*(slip_max - SLIP_MAX)
    if "HEADING" in reasons: viol += 1.0 + 5.0*(abs(yaw_end) - 0.35)
    if "NO_FLIGHT" in reasons: viol += 1.0 + 4.0*(FLIGHT_GATE - flight_min_step)/FLIGHT_GATE + 2.0*float(np.mean(np.array(step_fl) < FLIGHT_GATE) if step_fl else 1.0)
    n_req = n_cyc - N_TRANS - 1
    periodic = (per is not None and per_n >= n_req and per <= 1.0)
    bt = [b["t"] for b in bounds[N_TRANS:]]; step_T = float(np.median(np.diff(bt)))/2 if len(bt) > 2 else None      # cycle = right touchdown to right touchdown (events)
    fl_mid = flights[2*N_TRANS:] if len(flights) > 2*N_TRANS else flights
    out = dict(value=float(v_avg), valid=(not reasons) and periodic, reasons=reasons, violation=viol, periodic=bool(periodic), per_err=None if per is None else float(per), per_n=int(per_n),
               t_run=float(t_run), fell=bool(fell), overspeed=overspeed, closure=float(closure), closure_tol=float(ctol), slip_max=float(slip_max), yaw_end=float(yaw_end),
               n_td=[len(tds[0]), len(tds[1])], n_early=n_early, n_cycles=len(bounds), step_T=step_T, cadence_hz=(1/step_T if step_T else None), flight_mean=(float(np.mean(fl_mid)) if fl_mid else 0.0), flight_n=len(flights),
               flight_min_step=flight_min_step, flight_med_step=(float(np.median(step_fl)) if step_fl else 0.0), n_steps_gated=len(step_fl),
               stride_len=(float(v_avg*2*step_T) if step_T else None), W_pos=float(work.sum()), W_act=float(W_act), speed_ratio_by_joint={S["types"][i]: round(float(overj[i]), 3) for i in range(nT)},
               cot=(float(work.sum())/(R["M"]*G*max(bounds[-1]["comx"] - bounds[0]["comx"], 1e-3)) if len(bounds) > 1 else None), n_bounds=nb, dt=dt)
    if record:
        out["trace"] = {k: np.asarray(v).tolist() for k, v in trace.items()}; out["bounds"] = [dict({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in b.items()}) for b in bounds]
        out["tds"] = tds; out["los"] = los; out["flights"] = flights
    return out


def score(rec, v_des):
    if rec["reasons"]: return 100.0 + rec["violation"] - 0.01*rec["value"]
    short = max(0.0, 0.97*v_des - rec["value"])/v_des
    if not rec["periodic"] or short > 0: return 10.0 + min(rec["per_err"] if rec["per_err"] is not None else 50.0, 50.0)/10.0 + 20.0*short + 0.5*(1 - rec["per_n"]/(N_CYC - N_TRANS - 1))
    return -rec["value"]/v_des


# ------------------------------------------------------------------ problem object (both dt)
class Sprint:
    def __init__(self, robot, v, scale=1.0, stride=None):
        self.robot, self.v, self.scale = robot, float(v), float(scale); self.B = body(robot, scale)
        if stride is None and os.environ.get("SPRINT_STRIDE"):      # diagnostic override: "tc,tf,hr"
            tc_, tf_, hr_ = map(float, os.environ["SPRINT_STRIDE"].split(",")); L_ = self.B["L1"] + self.B["L2"]; dz_ = G*tf_*tc_/(2*np.pi)
            ok1, _, st_ = SC.stance_ok(self.B, self.v, tc_, tf_, hr_*L_, dz_); ok2, _, sw_ = SC.swing_ok(self.B, tc_, tf_, hr_*L_, st_) if ok1 else (False, "", None)
            stride = dict(tc=tc_, tf=tf_, h0=hr_*L_, dz=dz_, stance=st_, swing=sw_, util=None, rule="override", step=self.v*(tc_ + tf_), step_hz=1/(tc_ + tf_))
        self.st = stride or select_stride(self.B, self.v, scale)
        if self.st is None: raise ValueError(f"no feasible stride at {v} m/s (scale {scale})")
        self.S = {dt: plant(robot, dt, scale) for dt in (core.DT_OPT, core.DT_CHECK)}
        S1 = self.S[core.DT_OPT]; hp_i = [i for i, t in enumerate(S1["types"]) if "hip_pitch" in t]; kn_i = [i for i, t in enumerate(S1["types"]) if "knee" in t]
        # GROUND-SPEED MATCHING as far as the swing can afford: the largest beta (foot speed at liftoff/touchdown = beta x ground speed, relative
        # to the hip) whose inverse-dynamics hip and knee torques stay within KNEE_U of the rating (beta 1 = foot at rest on the ground)
        for beta in (1.0, 0.75, 0.5, 0.25, 0.0):
            self.R = scaffold(S1, self.B, self.st, self.v, beta=beta)
            if SWING_MODE != "cart": break
            if np.abs(self.R["ID"][:, hp_i]).max() <= KNEE_U*S1["jarr"]["tau"][hp_i[0]] and np.abs(self.R["ID"][:, kn_i]).max() <= KNEE_U*S1["jarr"]["tau"][kn_i[0]]: break
        self.R["beta"] = beta; self.k_extra = 0.0
        # SWING CLEARANCE on the model: the ceiling's fold angle assumes the hip at h0 over a vertical leg; the knee fold is increased
        # (0.1 rad steps) until every swing foot clears the floor by CLEAR over the cycle (kinematic check, model geometry)
        while SWING_MODE != "cart" and min(clearance(self.S[core.DT_OPT], self.R)["min"]) < CLEAR and self.k_extra < 1.0:
            self.k_extra += 0.1; self.R = scaffold(self.S[core.DT_OPT], self.B, self.st, self.v, k_extra=self.k_extra)
        self.R["k_extra"] = self.k_extra; self.R["clearance"] = clearance(self.S[core.DT_OPT], self.R)["min"]
    def run(self, z, dt=core.DT_OPT, record=False, n_cyc=N_CYC):
        S = self.S[dt]; return rollout(S, self.R, unpack(S, z), n_cyc=n_cyc, record=record)
    def robust(self, z):
        r1 = self.run(z, core.DT_OPT); r2 = self.run(z, core.DT_CHECK); return max(score(r1, self.v), score(r2, self.v)), r1, r2
    def n(self): return n_params(self.S[core.DT_OPT])


def summarize(r):
    keys = ("value", "valid", "reasons", "periodic", "per_err", "per_n", "t_run", "overspeed", "closure", "closure_tol", "slip_max", "yaw_end", "n_td", "cadence_hz", "flight_mean", "flight_min_step", "flight_med_step", "n_steps_gated", "stride_len", "W_pos", "cot", "dt")
    return {k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in keys}


# ------------------------------------------------------------------ CMA with a worker pool
_W = {}
def _init(robot, v, scale, stride):
    _W["P"] = Sprint(robot, v, scale, stride)
def _eval(z): s, r1, r2 = _W["P"].robust(z); return s, summarize(r1), summarize(r2)


def search(robot, v, z0, iters, pop, workers, seed=0, scale=1.0, sigma=0.1, log=None, stride=None):
    import cma, multiprocessing as mp
    Pb = Sprint(robot, v, scale, stride); st = Pb.st
    ctx = mp.get_context("fork"); pool = ctx.Pool(workers, initializer=_init, initargs=(robot, v, scale, st))
    s0, r1, r2 = Pb.robust(z0); best = (s0, np.array(z0), summarize(r1), summarize(r2)); t0 = time.time(); hist = [(0, float(s0))]; last_imp = 0
    es = cma.CMAEvolutionStrategy(np.asarray(z0, float), sigma, {"bounds": [-1, 1], "seed": seed, "verbose": -9, "maxiter": iters, "popsize": pop})
    try:
        while not es.stop():
            X = es.ask(); out = pool.map(_eval, X); F = [o[0] for o in out]; es.tell(X, F); k = int(np.argmin(F))
            if F[k] < best[0]: best = (F[k], np.array(X[k]), out[k][1], out[k][2]); last_imp = es.countiter
            hist.append((es.countiter, float(best[0])))
            if log: log(es.countiter, best, time.time() - t0)
            if best[0] < 0 and es.countiter - last_imp >= PATIENCE: break        # valid at both dt and no improvement for PATIENCE generations
    finally:
        pool.close(); pool.join()
    return dict(score=float(best[0]), z=best[1].tolist(), r1=best[2], r2=best[3], secs=time.time() - t0, hist=hist, stride=st, evals=int(es.countevals))


def stride_record(st): return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items() if k not in ("stance", "swing")} | dict(stance={k: (np.round(v, 3).tolist() if not isinstance(v, float) else round(v, 3)) for k, v in st["stance"].items()}, swing={k: round(float(v), 3) for k, v in st["swing"].items()})


def utilisation(Pb, z):
    """Envelope utilisation per joint over the recorded 1 ms rollout: max |tau| / available (driving line or rated), max |w| / rated."""
    S = Pb.S[core.DT_OPT]; r = Pb.run(z, record=True); tq = np.asarray(r["trace"]["tq"]); w = np.asarray(r["trace"]["qd"]); J = S["jarr"]
    frac = np.maximum(1e-6, 1 - np.abs(w)/J["om"]); avail = np.where(tq*w > 0, J["tau"]*frac, J["tau"]); u = np.abs(tq)/avail
    return r, {t: dict(torque=round(float(u[:, i].max()), 3), torque_p95=round(float(np.percentile(u[:, i], 95)), 3), speed=round(float(np.abs(w[:, i]).max()/J["om"][i]), 3)) for i, t in enumerate(S["types"])}


def technique(Pb, r):
    """Emergence-style descriptors of a recorded rollout: cadence, stride, flight fraction, trunk lean, arm swing amplitude, duty factor."""
    tr = r["trace"]; rpy = np.asarray(tr["rpy"]); q = np.asarray(tr["q"]); S = Pb.S[core.DT_OPT]; types = S["types"]
    t = np.asarray(tr["t"]); keep = t >= N_TRANS*Pb.R["Tc"]
    if not keep.any(): return dict(speed=r["value"], note="fell before the transient cycles ended")
    arm = {tn: round(float(np.ptp(q[keep, i])), 3) for i, tn in enumerate(types) if "shoulder_pitch" in tn or "elbow" in tn}
    load = np.asarray(tr["load"])[keep]; air = float(np.mean((load[:, 0] < 0.01) & (load[:, 1] < 0.01))) if len(load) else None
    duty = [float(np.mean(load[:, i] > 0.05)) for i in (0, 1)] if len(load) else None
    trunk = {tn: round(float(np.mean(q[keep, i])), 3) for i, tn in enumerate(types) if tn in CENTRE[Pb.robot]}
    return dict(cadence_hz=r["cadence_hz"], stride_len=r["stride_len"], flight_fraction=air, flight_mean=r["flight_mean"], duty=duty, pelvis_pitch_mean=round(float(np.mean(rpy[keep, 1])), 3) if keep.any() else None,
                trunk_mean=trunk, arm_ptp=arm, cot=r["cot"], speed=r["value"])


def write(path, obj): os.makedirs(os.path.dirname(path), exist_ok=True); json.dump(obj, open(path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def prior_only(robot, v, scale=1.0, tag=""):
    Pb = Sprint(robot, v, scale); z = z_prior(Pb.S[core.DT_OPT]); out = {}
    for dt in (core.DT_OPT, core.DT_CHECK):
        t0 = time.time(); r = Pb.run(z, dt, record=True); secs = time.time() - t0
        fl = r["flights"]; out[str(dt)] = dict(summarize(r), secs=round(secs, 2), n_cycles=r.get("n_cycles"), first_steps=dict(td=[np.round(x[:6], 3).tolist() for x in r["tds"]], flights=np.round(fl[:8], 3).tolist()))
    res = dict(robot=robot, v=v, scale=scale, kind="prior_only", act_tau=ACT_TAU, stride=stride_record(Pb.st), beta=Pb.R.get("beta"), ik_err=Pb.R["ik_err"], supplied=SUPPLIED, n_params=Pb.n(), meas=out)
    write(f"{RES}/prior_{robot}_v{v}{tag}.json", res); return res


def ladder(robot, speeds, iters, pop, workers, scale=1.0, tag="", seed=0, z_init=None, st_init=None, z_from_scale=1.0):
    z = z_init; rungs = []; path = f"{RES}/ladder_{robot}{tag}.json"; last_st, last_v = (st_init if st_init else (None, None))
    for v in speeds:
        extrap = False
        try:
            if os.environ.get("SPRINT_FIXSTRIDE") and last_st is not None:     # keep the warm-start rung's stride timing at every rung (flagged as extrapolated)
                Pb = Sprint(robot, v, scale, stride=dict(last_st, extrapolated_from=last_v)); extrap = True
            else: Pb = Sprint(robot, v, scale)
        except ValueError as e:                    # no ceiling stride satisfies the scaffold's constraints at this speed
            if os.environ.get("SPRINT_EXTRAP") and last_st is not None:      # BEYOND THE SCAFFOLD CEILING: keep the last feasible stride's timing, raise v (flagged)
                Pb = Sprint(robot, v, scale, stride=dict(last_st, extrapolated_from=last_v)); extrap = True
            else:
                rungs.append(dict(v=v, ok_both_dt=False, no_scaffold=str(e))); write(path, dict(robot=robot, scale=scale, supplied=SUPPLIED, budget=dict(iters=iters, pop=pop, workers=workers), rungs=rungs))
                print(f"sprint {robot}{tag} v={v}: no scaffold stride ({e})", flush=True); break
        last_st, last_v = (Pb.st, v) if not extrap else (last_st, last_v)
        z0 = z_prior(Pb.S[core.DT_OPT]) if z is None else np.asarray(z)
        if z is not None and z_from_scale is not None and z_from_scale != scale: z0 = rescale_knots(Pb.S[core.DT_OPT], z0, scale/z_from_scale); z_from_scale = scale
        lg = open(f"{RES}/ladder_{robot}{tag}_v{v}.log", "a")
        def log(it, best, el): lg.write(f"it {it} best {best[0]:.4f} v1 {best[2]['value']:.3f} {best[2]['reasons']} per {best[2]['per_err']} n {best[2]['per_n']} | v2 {best[3]['value']:.3f} {best[3]['reasons']} per {best[3]['per_err']} | {el:.0f}s\n"); lg.flush()
        res = search(robot, v, z0, iters, pop, workers, seed=int(os.environ.get("SPRINT_SEED", seed)), scale=scale, sigma=float(os.environ.get("SPRINT_SIGMA", 0.1)), log=log, stride=Pb.st)   # SPRINT_SEED / SPRINT_SIGMA recorded via the tag
        Pb2 = Pb; zb = np.asarray(res["z"]); r1 = Pb2.run(zb, core.DT_OPT); r2 = Pb2.run(zb, core.DT_CHECK)
        ok = bool(r1["valid"] and r2["valid"] and min(r1["value"], r2["value"]) >= 0.97*v)
        rec, util = utilisation(Pb2, zb); tech = technique(Pb2, rec)
        rung = dict(v=v, ok_both_dt=ok, stride_extrapolated=extrap, meas=[summarize(r1), summarize(r2)], stride=stride_record(Pb.st), secs=round(res["secs"]), evals=res["evals"], s_per_eval=round(res["secs"]/max(res["evals"], 1), 3),
                    hist=res["hist"], z=res["z"], params={k: v_ for k, v_ in unpack(Pb.S[core.DT_OPT], zb).items() if k != "x"}, utilisation=util, technique=tech)
        rungs.append(rung); write(path, dict(robot=robot, scale=scale, supplied=SUPPLIED, budget=dict(iters=iters, pop=pop, workers=workers), rungs=rungs))
        msg = f"sprint {robot}{tag} v={v}: {'VALID' if ok else 'not valid'} v1 {r1['value']:.2f} {r1['reasons']} per {r1['per_n']} | v2 {r2['value']:.2f} {r2['reasons']} | {res['secs']/60:.0f} min"
        print(msg, flush=True); os.system(f'~/bin/notify "{msg[:150]}" >/dev/null 2>&1')
        if ok: z = res["z"]
        else: break
    return rungs


DOCUMENTED = {"G1": {"spec": 2.0}, "H1": {"stock_2024": 3.3, "modified_high_torque_2026_04_peak": 10.0, "superman_2026_08_unverified": 12.66}}


def summary(tags=("_lag2ms", "_lag40ms")):
    """Collect the ladders, the spec ceiling and the documented speeds into results/sprint/summary.json."""
    ceil = json.load(open(f"{RES}/ceiling.json")); out = {}
    for robot in ("G1", "H1"):
        out[robot] = dict(ceiling_any_cadence=ceil[robot]["v_max"], documented=DOCUMENTED[robot], ladders={})
        for tag in tags:
            pth = f"{RES}/ladder_{robot}{tag}.json"
            if not os.path.exists(pth): continue
            L = json.load(open(pth)); ok = [r for r in L["rungs"] if r["ok_both_dt"]]
            top = max(ok, key=lambda r: r["v"]) if ok else None
            out[robot]["ladders"][tag] = dict(act_tau=L["supplied"].get("act_tau"), rungs=[dict(v=r["v"], ok=r["ok_both_dt"], v_meas=[m["value"] for m in r["meas"]], reasons=[m["reasons"] for m in r["meas"]], s_per_eval=r["s_per_eval"], secs=r["secs"]) if "meas" in r else dict(v=r["v"], ok=False, note=r.get("no_scaffold")) for r in L["rungs"]],
                                              v_max_rung=(top["v"] if top else None), v_max_measured=(min(m["value"] for m in top["meas"]) if top else None),
                                              binding_at_top=(sorted(((k, v["torque_p95"]) for k, v in top["utilisation"].items()), key=lambda kv: -kv[1])[:4] if top else None), technique_at_top=(top["technique"] if top else None))
    write(f"{RES}/summary.json", out); return out


if __name__ == "__main__":
    cmd = sys.argv[1]; robot = sys.argv[2] if len(sys.argv) > 2 else None
    if cmd == "summary": print(json.dumps(summary(), indent=1, default=str)); sys.exit()
    if cmd == "prior":
        r = prior_only(robot, float(sys.argv[3]), float(os.environ.get("SCALE", 1.0)), tag=os.environ.get("SPRINT_TAG", ""))
        print(json.dumps(dict(stride={k: r["stride"][k] for k in ("tc", "tf", "h0", "dz", "step", "step_hz", "util")}, ik_err=r["ik_err"], meas={k: {kk: v[kk] for kk in ("value", "reasons", "periodic", "per_err", "per_n", "t_run", "n_td", "cadence_hz", "flight_mean", "stride_len", "secs", "first_steps")} for k, v in r["meas"].items()}), default=str))
    elif cmd == "ladder":
        sp = [float(x) for x in sys.argv[3].split(",")]; it = int(sys.argv[4]) if len(sys.argv) > 4 else 60; pop = int(sys.argv[5]) if len(sys.argv) > 5 else 12
        wk = int(sys.argv[6]) if len(sys.argv) > 6 else 6; tag = sys.argv[7] if len(sys.argv) > 7 else ""
        zi = None
        if os.environ.get("SPRINT_WARM"):          # warm start: <ladder json>:<rung index>
            pth, ix = os.environ["SPRINT_WARM"].rsplit(":", 1); rw = json.load(open(pth))["rungs"][int(ix)]; zi = rw["z"]; sti = (rw["stride"], rw["v"])
        else: sti = None
        ladder(robot, sp, it, pop, wk, scale=float(os.environ.get("SCALE", 1.0)), tag=tag, z_init=zi, st_init=sti, z_from_scale=float(os.environ.get("SPRINT_WARM_SCALE", 1.0)))   # the warm json's body scale (continuation in body space)
