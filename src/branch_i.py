"""Branch I — dynamic solves on G1. Central test: does actual recruitment track the
analytic required_depth(D)? Plus I.3, the dynamic assignment test."""
import sys, os, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, cma, robot_bound as R, bank as BK
from validity import TrajectoryValidityReport
import plant as P
from multiprocessing import Pool

MEN = os.environ.get("MENAGERIE_DIR", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "models", "mujoco_menagerie"))
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
G = 9.80665; HOR = 0.50; NK = 10
SPEC = f"{ROOT}/spec/g1.yaml"; MJCF = f"{MEN}/unitree_g1/scene_mjx.xml"
# Declared robot configurations (declared-source rule: feet are named by BODY, never inferred
# from geom-name substrings; H1's collision geoms are unnamed).
ROBOTS = {
    "G1": dict(mjcf=MJCF, spec=SPEC, foot_bodies=("left_ankle_roll_link", "right_ankle_roll_link"),
               keyframes=("knees_bent", "home")),
    # G1H (J20, 2026-10-11): the G1 with Dex3 hands (menagerie g1_with_hands + a palm site + the knees_bent keyframe mapped by
    # joint name; spec/g1_hands_palm/). Same body spec as the G1; the 14 finger joints are held (position servos) unless a task drives them.
    "G1H": dict(mjcf=f"{MEN}/unitree_g1/scene_hands_palm.xml", spec=SPEC, foot_bodies=("left_ankle_roll_link", "right_ankle_roll_link"),
                keyframes=("knees_bent", "pregrasp", "ready", "stand")),   # "knees_bent" (arms at the hips, hand front x 0.15): the 2026-10-11 J27 wrap scene has a 0.76 m counter from x 0.25 and the "pregrasp"/"ready" hands (bottom 0.70 m, x to 0.33/0.56) started inside it; the side-approach reach lifts the hand itself
    "H1": dict(mjcf=f"{MEN}/unitree_h1/scene.xml", spec=f"{ROOT}/spec/h1.yaml",
               foot_bodies=("left_ankle_link", "right_ankle_link"), keyframes=("home",),
               # hold stiffness cap: saturate only at 0.05 rad deflection. Uncapped I_eff/dt^2 at
               # H1's dt=2 ms gave 26k-133k N m/rad and bang-bang chatter at the torque limits.
               hold_sat_rad=0.05,
               # contact time constant = 2 x dt (MuJoCo's stability floor; G1's MJX model uses the
               # same rule, 0.008 s at dt 4 ms). H1's default 0.02 s let a rigid statue sink ~3.4 mm
               # unevenly and roll over a 2.4 cm heel margin (Phase 9 H1 diagnosis).
               geom_solref=(0.004, 1.0),
               # settle servos pre-loaded with the static-certificate torques (target offset tau/kp),
               # so the certified holding torque is delivered at ZERO deflection; without it the
               # capped holds let knees/ankles sag ~0.01 rad and the COM crosses the 2.4 cm heel margin
               settle_feedforward=True),
    # HELD-OUT EMBODIMENTS (2026-10-16, src/heldout_onboard.py): derived copies of the menagerie Apollo / Booster T1 with joint and
    # body names mapped to the project convention (registration only); ratings and their sources in spec/apollo.yaml, spec/t1.yaml.
    "Apollo": dict(mjcf=f"{ROOT}/results/heldout/models/scene_apollo_heldout.xml", spec=f"{ROOT}/spec/apollo.yaml",
                   foot_bodies=("left_foot_link", "right_foot_link"), keyframes=("stand_level", "stand")),
    "T1": dict(mjcf=f"{ROOT}/results/heldout/models/scene_t1_heldout.xml", spec=f"{ROOT}/spec/t1.yaml",
               foot_bodies=("left_ankle_roll_link", "right_ankle_roll_link"), keyframes=("home",)),
}

def geom_solref(robot):
    """The robot's declared contact override (H1: solref 0.004 on every geom), or None to keep the MJCF's own contact params.
    STOCK_CONTACT=1 (opt-in, 2026-10-17): no override, every robot keeps its stock menagerie contact (H1: MuJoCo default 0.02, 1)."""
    if os.environ.get("STOCK_CONTACT") == "1": return None
    return ROBOTS[robot].get("geom_solref")


_S = {}
def to_torque_motors(rb):
    """Convert the model's POSITION actuators into torque motors.

    The Menagerie G1 ships 29 position servos: force = kp*(ctrl - q) - kv*qdot with kp=75,
    kv=2 and ctrlrange inherited from the joint range. So d.ctrl is a TARGET ANGLE, not a
    torque. Writing a torque value such as 139 into it commanded a target of 139 rad, clipped
    to the joint limit, i.e. "go to full flexion" - and the torque-speed envelope this project
    is built on was bypassed entirely.

    This project's whole subject is torque and speed limits, so the actuators must be torque
    sources. Each is converted in place to a unit-gain motor with ctrlrange and forcerange set
    to that joint's peak torque, after which d.ctrl IS the commanded torque in N*m.
    """
    m = rb.m
    for a in range(m.nu):
        jid = int(m.actuator_trnid[a][0])
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        tau = rb.J[jn]["tau"] if jn in rb.J else float(m.jnt_actfrcrange[jid][1]) or 25.0
        m.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_gainprm[a][:] = 0.0; m.actuator_gainprm[a][0] = 1.0
        m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_NONE
        m.actuator_biasprm[a][:] = 0.0
        m.actuator_ctrlrange[a][:] = [-tau, tau]
        m.actuator_forcerange[a][:] = [-tau, tau]
        m.actuator_ctrllimited[a] = 1; m.actuator_forcelimited[a] = 1
    return m


def _as_servo(m, a, kp, kv, tau):
    """Implicit position servo: force = kp*(ctrl - q) - kv*qdot, clipped to +-tau."""
    jid = int(m.actuator_trnid[a][0])
    m.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
    m.actuator_gainprm[a][:] = 0.0; m.actuator_gainprm[a][0] = kp
    m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
    m.actuator_biasprm[a][:] = 0.0; m.actuator_biasprm[a][1] = -kp; m.actuator_biasprm[a][2] = -kv
    m.actuator_ctrlrange[a][:] = m.jnt_range[jid] if m.jnt_limited[jid] else [-10, 10]
    m.actuator_forcerange[a][:] = [-tau, tau]
    m.actuator_ctrllimited[a] = 1; m.actuator_forcelimited[a] = 1


def _as_motor(m, a, tau):
    m.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
    m.actuator_gainprm[a][:] = 0.0; m.actuator_gainprm[a][0] = 1.0
    m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_NONE
    m.actuator_biasprm[a][:] = 0.0
    m.actuator_ctrlrange[a][:] = [-tau, tau]; m.actuator_forcerange[a][:] = [-tau, tau]
    m.actuator_ctrllimited[a] = 1; m.actuator_forcelimited[a] = 1


_SS = {}
def setup(start=None, robot="G1"):
    """start=None: the knees_bent keyframe (the protocol of Phases 6-8). start=<qpos array>:
    an alternative start pose (e.g. a retargeted mocap frame); cached per pose."""
    if start is None and robot == "G1":
        if "rb" in _S: return _S
        return _build(None, _S)
    key = (robot, None if start is None else tuple(np.round(np.asarray(start), 6)))
    if key not in _SS: _SS[key] = _build(None if start is None else np.asarray(start, float), {}, robot)
    return _SS[key]


def _build(start, _S, robot="G1"):
    cfg = ROBOTS[robot]
    rb = R.RobotBody(cfg["mjcf"], cfg["spec"], capacity="peak")
    if geom_solref(robot): rb.m.geom_solref[:] = geom_solref(robot)
    m = to_torque_motors(rb)
    types = []                      # one control parameter per sagittal joint type
    for g in rb.groups:
        for t in rb.group_types(g):
            if t not in types: types.append(t)
    tmap = {}
    for t in types:
        tmap[t] = [(rb.J[n]["id"], m.jnt_dofadr[rb.J[n]["id"]], m.jnt_qposadr[rb.J[n]["id"]],
                    rb.J[n]["tau"], rb.J[n]["omega"],
                    -1.0 if (rb.mirror[t] and rb.J[n]["side_geom"] == "right") else 1.0,
                    [a for a in range(m.nu) if m.actuator_trnid[a][0] == rb.J[n]["id"]][0])
                   for n in rb.groups_of_type(t)]
    driven = set()
    for t in types:
        for (jid, dof, qad, tau, om, sgn, aid) in tmap[t]: driven.add(aid)
    # Every actuator the solver does NOT drive must be HELD at its start value ("locked" =
    # held rigid). Phase 6-7 held them with a Python PD written into torque motors. That PD
    # is integrated EXPLICITLY and was numerically unstable on the light joints: defect P7
    # (found 2026-09-21), the "settled" start had wrist joints at 13 rad/s and the base
    # rotating at 6.6 rad/s. Held joints are now MuJoCo position servos, whose velocity term
    # implicitfast integrates implicitly. Declared in the plant contract as
    # "implicit_position_hold"; the ctrl of a held actuator is its TARGET ANGLE.
    kid = -1
    for kname in cfg["keyframes"]:
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, kname)
        if kid >= 0: break
    dk = mujoco.MjData(m); dk.qpos[:] = m.key_qpos[kid] if start is None else start
    mujoco.mj_forward(m, dk)
    Mf = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, Mf, dk.qM)
    Minv = np.linalg.inv(Mf)
    held = []
    for a in range(m.nu):
        if a in driven: continue
        jid = int(m.actuator_trnid[a][0]); dof = int(m.jnt_dofadr[jid])
        tau = float(m.jnt_actfrcrange[jid][1]) or 25.0
        ieff = 1.0/float(Minv[dof, dof])
        kp = 1.0*ieff/(m.opt.timestep**2)
        if cfg.get("hold_sat_rad"): kp = min(kp, tau/cfg["hold_sat_rad"])
        kv = 2.0*np.sqrt(kp*ieff)
        _as_servo(m, a, kp, kv, tau)
        held.append((a, dof, int(m.jnt_qposadr[jid]), tau, float(kp), float(kv)))
    fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    fb = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in cfg["foot_bodies"]}
    assert all(b >= 0 for b in fb), f"declared foot bodies missing in {robot}"
    foot = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in fb}
    act_dof = [int(m.jnt_dofadr[int(m.actuator_trnid[a][0])]) for a in range(m.nu)]
    lim_joints = [j for j in range(m.njnt) if m.jnt_limited[j] and m.jnt_type[j] in (2, 3)]
    ti, aid_, dof_, tau_, om_, sg_ = [], [], [], [], [], []
    for i, t in enumerate(types):
        for (jid, dof, qad, tau, om, sgn, aid) in tmap[t]:
            ti.append(i); aid_.append(aid); dof_.append(dof); tau_.append(tau); om_.append(om); sg_.append(sgn)
    qad_ = [q for t in types for (jid, dof, q, tau, om, sgn, aid) in tmap[t]]
    jarr = dict(qad=np.array(qad_), ti=np.array(ti), aid=np.array(aid_), dof=np.array(dof_), tau=np.array(tau_, float),
                om=np.array(om_, float), sgn=np.array(sg_, float))
    lim_q = np.array([m.jnt_qposadr[j] for j in lim_joints]); lim_lo = np.array([m.jnt_range[j][0] for j in lim_joints])
    lim_hi = np.array([m.jnt_range[j][1] for j in lim_joints])
    _S.update(jarr=jarr, lim_q=lim_q, lim_lo=lim_lo, lim_hi=lim_hi, Minv_key=Minv, rb=rb, m=m, types=types, tmap=tmap,
              q0=np.array(m.key_qpos[kid]) if start is None else np.array(start),
              M=float(sum(m.body_mass)), groups=list(rb.groups),
              driven=sorted(driven), held=held, floor=fl, foot_geoms=foot,
              act_dof=act_dof, lim_joints=lim_joints, robot=robot, mjcf=cfg["mjcf"])
    return _S

OVERSPEED_TOL = 1.10   # validity: max |joint speed| / rated speed over the rollout (2026-10-11 audit)
def _with_overspeed(vd, overspeed):
    if overspeed > OVERSPEED_TOL: vd = dict(vd, reasons=list(vd.get("reasons", [])) + ["OVERSPEED"], valid=False)
    return vd


def backemf_brake(m, d, S, ramp=0.1):
    """Universal actuator speed limit (2026-10-11 audit). A geared BLDC actuator back-driven past its rated (no-load) speed is
    braked by its own back-EMF through the inverter: here a braking torque that ramps from 0 at the rated speed to the rated
    torque at (1+ramp) x rated, applied at the joint as qfrc_applied so it acts in every control mode (torque rollouts,
    position-servo rollouts, the landing tuner). S["motor_envelope"] = False disables it (pre-audit plant)."""
    if not S.get("motor_envelope", True): return
    J = S["jarr"]; dofs = J["dof"]; w = d.qvel[dofs]; om = J["om"]; tau = J["tau"]
    if len(om) != len(dofs): om = np.asarray(om)[J["ti"]]; tau = np.asarray(tau)[J["ti"]]
    over = np.clip((np.abs(w) - om)/(ramp*om), 0.0, 1.0)
    d.qfrc_applied[dofs] = -tau*np.sign(w)*over


def motor_envelope(tq, w, tau, om, on=True):
    """ACTUATOR TORQUE-SPEED ENVELOPE (added 2026-10-11 after the throw audit: the wrist was whipped to 69 rad/s against a
    22 rad/s rated speed, the elbow to 44 against 37). The DRIVING envelope tau_drive(w) = tau * max(0, 1 - |w|/om) and
    braking up to tau were already enforced in both torque rollouts; what was missing is that a joint could be whipped
    PASSIVELY past its rated speed with zero commanded torque. Above the rated speed a geared BLDC actuator is back-driven
    past its no-load speed and its back-EMF brakes it (regenerative, inverter-limited): the rollouts now apply
    -tau * sign(w) there regardless of the command. tq: commanded torques (per driven
    actuator), w: joint velocities, tau/om: rated torque and speed. Conservative and physical; on=False reproduces the
    pre-audit plant."""
    if not on: return tq
    tq = np.asarray(tq, float).copy(); w = np.asarray(w, float); aw = np.abs(w)
    drive = np.sign(tq) == np.sign(w)
    cap = tau*np.clip(1.0 - aw/np.maximum(om, 1e-9), 0.0, 1.0)
    tq = np.where(drive, np.clip(tq, -cap, cap), tq)
    over = aw > om
    tq = np.where(over, -tau*np.sign(w), tq)
    return tq


def _set_capacity_limits(m, S, caps_scale):
    """Make the plant REALISE the declared capacities.

    Defect P6 (found 2026-09-21): to_torque_motors() set ctrlrange/forcerange to the BASELINE
    peak torque, and rollout() scaled torque only in Python. MuJoCo then clamped every scaled-up
    command back to baseline (commanded 110 N*m on the ankle -> actuator_force 50). Any
    caps_scale > 1 was silently capped at 1.0 in torque while its speed scaling still applied.
    Limits are now set from the declared (scaled) capacity on every rollout, so no stale state
    survives between solves, and PCTV-9 verifies the realised limit."""
    for t in S["types"]:
        st = caps_scale[t][0] if (caps_scale and t in caps_scale) else 1.0
        for (jid, dof, qad, tau, om, sgn, aid) in S["tmap"][t]:
            m.actuator_ctrlrange[aid][:] = [-tau*st, tau*st]
            m.actuator_forcerange[aid][:] = [-tau*st, tau*st]
            # second clamp layer: the MJX model's per-JOINT actuatorfrcrange limits the total
            # actuator force at the joint (found by PCTV-6 on its first production run)
            m.jnt_actfrcrange[jid][:] = [-tau*st, tau*st]


def apply_dt(S, dt, iterations=50):
    """Change the simulation timestep CONSISTENTLY (Phase 9, defect A4): held-joint servo gains
    are kp = I_eff/dt^2 (capped per robot) and must be rebuilt, and the start re-settled. Phase 8
    replays changed m.opt.timestep only, leaving 4 ms-derived holds 16x too soft at 1 ms."""
    m = S["m"]; m.opt.timestep = dt; m.opt.iterations = iterations
    cfg = ROBOTS[S.get("robot", "G1")]; Minv = S["Minv_key"]; held = []
    for (a, dof, qad, tau, kp_old, kv_old) in S["held"]:
        ieff = 1.0/float(Minv[dof, dof]); kp = ieff/dt**2
        if cfg.get("hold_sat_rad"): kp = min(kp, tau/cfg["hold_sat_rad"])
        kv = 2.0*np.sqrt(kp*ieff); _as_servo(m, a, kp, kv, tau)
        held.append((a, dof, qad, tau, float(kp), float(kv)))
    S["held"] = held
    for k in ("q_settled", "v_settled", "a_hold"): S.pop(k, None)
    settle(S)


def settle(S, T_settle=4.0):
    """Bring the robot to REST in the knees_bent keyframe, standing on the ground.

    Placement uses true per-geom extent (never geom_rbound). During the settle the study
    joints are ALSO implicit position servos; afterwards their realised holding torques are
    recorded as tau_hold, the actuators are restored to torque motors, and every rollout
    starts with those holding torques applied (a jump released from a held crouch, not from
    limp joints). The start state is gated: max |qdot| must be < 0.05 rad/s."""
    m = S["m"]
    if "q_settled" in S: return
    rbb = S["rb"]
    rbb.d.qpos[:] = S["q0"]; rbb.d.qvel[:] = 0
    mujoco.mj_forward(m, rbb.d)
    rbb.d.qpos[2] -= rbb.lowest_z()
    mujoco.mj_forward(m, rbb.d)
    dd = mujoco.MjData(m); dd.qpos[:] = rbb.d.qpos; dd.qvel[:] = 0
    Minv = S["Minv_key"]
    for t in S["types"]:
        for (jid, dof, qad, tau, om, sgn, aid) in S["tmap"][t]:
            ieff = 1.0/float(Minv[dof, dof]); kp = 1.0*ieff/(m.opt.timestep**2)
            if ROBOTS[S.get("robot", "G1")].get("hold_sat_rad"): kp = min(kp, tau/ROBOTS[S["robot"]]["hold_sat_rad"])
            _as_servo(m, aid, kp, 2.0*np.sqrt(kp*ieff), tau)
    for a in range(m.nu):
        dd.ctrl[a] = dd.qpos[m.jnt_qposadr[int(m.actuator_trnid[a][0])]]
    mujoco.mj_forward(m, dd)
    if ROBOTS[S.get("robot", "G1")].get("settle_feedforward"):
        import plant as _P
        tau_ff = S.get("settle_tau_ff")
        if tau_ff is None:
            cg = _P.PlantContractVerifier(m, dd, _P.PlantContract(S["mjcf"], "torque")).collision_geoms()
            cert = _P.static_certificate(m, mujoco.MjData(m), dd.qpos.copy(), coll_geoms=cg)
            S["settle_certificate"] = cert["status"]
            if cert["status"] == "STATIC_FEASIBLE": tau_ff = np.asarray(cert["tau"], float); S["settle_cert_tau"] = tau_ff
        else: S["settle_certificate"] = "FEEDFORWARD_FROM_BARE_ROBOT"
        # WORLD SCENES: the whole-model static certificate is infeasible when a free object rests on furniture (its gravity
        # has no ground contact in the solve), so the preload was skipped and H1's heel-margin statue tipped over during the
        # settle. The task passes the BARE robot's certified torques (same actuator order) as S["settle_tau_ff"] instead.
        if tau_ff is not None:
            for a in range(m.nu):
                kp = float(m.actuator_gainprm[a][0])
                if kp > 0: dd.ctrl[a] += float(tau_ff[a])/kp
    for _ in range(int(T_settle/m.opt.timestep)):
        mujoco.mj_step(m, dd)
    tau_hold = {}
    for t in S["types"]:
        vals = []
        for (jid, dof, qad, tau, om, sgn, aid) in S["tmap"][t]:
            vals.append(sgn*float(dd.actuator_force[aid])/tau)
            _as_motor(m, aid, tau)
        tau_hold[t] = float(np.mean(vals))
    S["q_settled"] = dd.qpos.copy(); S["v_settled"] = dd.qvel.copy()
    S["ctrl_settled"] = dd.ctrl.copy(); S["a_hold"] = tau_hold
    S["settle_qdot_max"] = float(np.max(np.abs(dd.qvel)))


def flight_rotation(m, d, vz):
    """Whole-body rotation during ballistic flight. Angular momentum about the COM is conserved
    in flight; composite-body angular velocity = I_com^-1 L; flight time = 2 vz / g. Posture
    changes in flight redistribute but cannot remove L, so this is the rotation a rigid landing
    posture would have to absorb."""
    mujoco.mj_subtreeVel(m, d)
    L = np.array(d.subtree_angmom[1]); c = np.array(d.subtree_com[1])
    I = np.zeros((3, 3))
    for b in range(1, m.nbody):
        R = d.ximat[b].reshape(3, 3); r = d.xipos[b] - c
        I += R @ np.diag(m.body_inertia[b]) @ R.T + m.body_mass[b]*(r @ r*np.eye(3) - np.outer(r, r))
    w = np.linalg.solve(I, L); tf = max(0.0, 2.0*vz/G)
    return dict(L=L.tolist(), omega_body=float(np.linalg.norm(w)), flight_time=tf,
                flight_rotation=float(np.linalg.norm(w)*tf))


def hold_params(S, caps_scale=None):
    """Control parameters that simply keep applying the holding torques (PCTV-5)."""
    a = [S["a_hold"][t]/((caps_scale or {}).get(t, (1.0, 1.0))[0]) for t in S["types"]]
    return np.repeat(np.array(a)[:, None], NK, axis=1).ravel()


FLIGHT_CONFIRM = 0.020   # s of sustained zero foot force that defines a flight (dt-independent)


def phase8_schedule(horizon, dt_opt, nk):
    """knot spacing (s) that a Phase 8 solution optimised at timestep dt_opt actually used."""
    return (int(horizon/dt_opt)//nk)*dt_opt


def rollout(params, active_types, caps_scale=None, record=False, S=None, ctrl_fn=None, horizon=None,
            score="first", knot_dt=None, lag="exact", stop_margin=None, landing=None, disturb=None, slip_cap=None):
    """ctrl_fn(t, d, S) -> per-driven-actuator normalised torque command (array over
    S['jarr']['aid'], in actuator sign, before lag and envelope), ADDED to the knot command.
    Used by the mocap-tracking baseline; the plant, lag, envelope and all measurements are
    identical for every controller.
    score="first": the jump is the first upward takeoff (Phase 8 protocol, stops there).
    score="best":  every flight phase in the horizon is scored with cumulative bookkeeping up to
                   its own takeoff, and the best is returned (north-star benchmark: a tracker's
                   incidental early hop must not truncate the jump it is tracking)."""
    S = S or setup(); m = S["m"]; d = mujoco.MjData(m)
    settle(S)
    _set_capacity_limits(m, S, caps_scale)
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    d.qpos[:] = S["q_settled"]; d.qvel[:] = S["v_settled"]
    mujoco.mj_forward(m, d)
    E0 = float(d.energy[0] + d.energy[1])
    start_qdot = float(np.max(np.abs(d.qvel[6:]))) if m.nv > 6 else 0.0
    Hh = horizon or HOR
    nst = int(Hh/m.opt.timestep); dt = m.opt.timestep
    kn = np.asarray(params).reshape(len(S["types"]), -1); NKl = kn.shape[1]
    # start from the HOLDING torque, expressed relative to this morphology's torque limit
    act = np.array([S["a_hold"][t]/((caps_scale or {}).get(t, (1.0, 1.0))[0]) for t in S["types"]])
    # exact first-order lag discretisation (dt-consistent; Phase 8 used forward Euler dt/0.040)
    alpha = 1.0 - np.exp(-dt/0.040) if lag == "exact" else dt/0.040
    if knot_dt is None: knot_dt = Hh/NKl
    for (a_, dof, qad, tau, kp, kd) in S["held"]:
        d.ctrl[a_] = S["q_settled"][qad]
    peak_v = -1e9; peak_grf = 0.0
    v_takeoff = None; takeoff_t = None; max_pen = 0.0; grf_spike = 0.0
    prev_fz = 0.0; W_act = 0.0; E_to = None; W_to = None; ke_to = 0.0
    ACC = {"con": 0.0, "pas": 0.0}
    nonfoot = False; nsat = 0; ncmd = 0; lim_exc = 0.0; up_to = 1.0; overspeed = 0.0; over_j = np.zeros(len(S["jarr"]["aid"]))
    foot = S["foot_geoms"]; floor = S["floor"]
    if "torso_body" not in S: S["torso_body"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    rec = dict(t=[], vz=[], z=[], fz=[], qvel={t: [] for t in S["types"]},
               tq={t: [] for t in S["types"]}) if record else None
    # vectorised per-joint arrays (one entry per driven actuator)
    J = S["jarr"]; ti_ = J["ti"]
    tau_v = J["tau"]*np.array([((caps_scale or {}).get(t, (1.0, 1.0)))[0] for t in S["types"]])[ti_]
    om_v = J["om"]*np.array([((caps_scale or {}).get(t, (1.0, 1.0)))[1] for t in S["types"]])[ti_]
    amask = np.array([1.0 if t in active_types else 0.0 for t in S["types"]])[ti_]
    lq = S["lim_q"]; llo = S["lim_lo"]; lhi = S["lim_hi"]
    act_j = act[ti_]*amask*J["sgn"]
    work_v = np.zeros(len(S["types"])); work_pre = None
    unload_t = 0.0; pk_util = 0.0
    ppow = 0.0; util_sum = 0.0; util_n = 0; impulse = 0.0; W_held = 0.0
    qmin = np.full(len(J["aid"]), np.inf); qmax = np.full(len(J["aid"]), -np.inf)
    z_start = float(d.subtree_com[0][2]); zmin = z_start
    held_a = np.array([h[0] for h in S["held"]]); held_d = np.array([h[1] for h in S["held"]])
    in_flight = False; snaps = []; cand = None; cand_t = 0.0
    # Phase 13 (opt-in): hardware stop margin during push-off, and a flight/landing controller after takeoff
    margin_viol = 0.0; land_mode = False; land = dict(touchdown_t=None, nonfoot_after=False, max_tilt=0.0) if landing else None
    slip_viol = 0.0; slip_steps = 0      # Phase 14c (opt-in): no-slip push-off, |f_t| <= slip_cap * mu * f_n on every loaded foot contact
    def _finalize(v_takeoff, takeoff_t, E_to, W_to, ke_to, up_to, work_pre, z_to, vx_to, wbase_to, rot_to):
        def _pr(wv):
            tot_ = float(np.sum(wv)); return (float(1.0/np.sum((wv/tot_)**2)) if tot_ > 0 else 0.0), tot_
        if work_pre is None:
            work_pre = work_v.copy(); z_to = float(d.subtree_com[0][2]); vx_to = 0.0; wbase_to = 0.0
            rot_to = dict(L=[0, 0, 0], omega_body=0.0, flight_time=0.0, flight_rotation=0.0)
        if E_to is None: E_to = float(d.energy[0] + d.energy[1]); W_to = W_act
        T_push = takeoff_t if takeoff_t is not None else HOR
        # PCTV-8 rigorous per-trajectory bound, conditional only on the MEASURED push-off duration:
        # each study joint's work <= tau0*omega0/4 * T (peak of the linear torque-speed envelope);
        # held-joint work is measured; potential-energy change is measured.
        Pmax = float(np.sum(tau_v*om_v/4.0*amask))
        E_avail = Pmax*T_push + max(W_held, 0.0) - S["M"]*G*(z_to - z_start)
        v_bound = float(np.sqrt(max(0.0, 2.0*E_avail/S["M"])))
        mech = dict(unload_time=unload_t, impulse=impulse, impulse_net=impulse - S["M"]*G*T_push, peak_power=ppow,
                    mean_util=util_sum/max(util_n, 1), peak_util=pk_util, T_push=T_push, z_start=z_start, z_min=zmin,
                    crouch_depth=z_start - zmin, z_takeoff=z_to, vx_takeoff=vx_to,
                    base_angvel_takeoff=wbase_to, **rot_to, W_held=W_held, v_bound_rigorous=v_bound,
                    excursion={t: float(np.max((qmax-qmin)[ti_ == i])) if np.isfinite(qmax[ti_ == i]).all() else 0.0
                               for i, t in enumerate(S["types"])})
        # PRIMARY: participation over the PUSH-OFF (start -> takeoff). Work done in flight does not
        # contribute to the jump. The full-horizon variant is kept for the window-sensitivity check.
        pr, tot = _pr(work_pre)
        pr_full, _ = _pr(work_v)
        work = {t: float(work_pre[i]) for i, t in enumerate(S["types"])}
        vt = v_takeoff if v_takeoff is not None else 0.0
        resid = (E_to - E0) - W_to
        # closure: dE = W_actuator + W_constraint + W_passive (+ integration error), read at the same instant
        closure = resid - ACC["con"] - ACC["pas"]
        rep = TrajectoryValidityReport(
            start_qdot_max=start_qdot, energy_residual=resid,
            energy_tol=max(1.0, 0.02*abs(W_to)), actuator_work=W_to,
            contact_energy=0.5*peak_grf*max_pen, ke_takeoff=ke_to,
            nonfoot_contact=nonfoot, took_off=(v_takeoff is not None and vt > 0),
            peak_grf_bw=peak_grf/(S["M"]*G), grf_spike_bw=grf_spike/(S["M"]*G),
            max_penetration=max_pen, saturation_frac=nsat/max(ncmd, 1),
            joint_limit_excursion=lim_exc, upright_at_takeoff=up_to,
            # impulse is summed over steps 0..s inclusive, i.e. (s+1)*dt of gravity
            momentum_error=(vt - (impulse/S["M"] - G*((T_push + dt) if takeoff_t is not None else HOR))),
            energy_closure=closure, closure_tol=max(5.0, 0.02*(abs(W_to) + abs(ACC["con"])))).evaluate()
        out = dict(v=vt, v_peak_transient=peak_v, takeoff_t=takeoff_t,
                   height=max(0.0, vt)**2/(2*G), work={t: work[t] for t in S["types"]},
                   participation=pr, participation_full=pr_full,
                   work_full={t: float(work_v[i]) for i, t in enumerate(S["types"])}, grf_bw=peak_grf/(S["M"]*G), total_work=float(tot),
                   max_penetration=max_pen, grf_spike_bw=grf_spike/(S["M"]*G),
                   valid=rep.valid and (vt <= v_bound + 1e-3) and (overspeed <= OVERSPEED_TOL), validity=_with_overspeed(rep.to_dict(), overspeed), overspeed=overspeed, overspeed_by_joint={S["types"][int(i)]: round(float(v_), 2) for i, v_ in zip(S["jarr"]["ti"], over_j) if v_ > 1.0}, mech=mech,
                   bound_violation=bool(vt > v_bound + 1e-3),
                   energy_audit=dict(residual=resid, W_constraint=ACC["con"], W_passive=ACC["pas"], closure=closure))
        return out
    for s in range(nst):
        # explicit time-based knot schedule (Phase 9, defect A1): knots are knot_dt SECONDS apart,
        # independent of the simulation timestep; Phase 8 used per = nst//NK steps, so the
        # schedule changed with dt (48 ms at 4 ms, 50 ms at 1 ms) and every "1 ms replay" ran
        # a time-warped control.
        tt = s*dt; k = min(int(tt/knot_dt + 1e-9), NKl-1); k2 = min(k+1, NKl-1)
        f = min(1.0, (tt - k*knot_dt)/knot_dt) if k < NKl-1 else 0.0
        cmd = kn[:, k] + (kn[:, k2]-kn[:, k])*f
        cmd_j = cmd[ti_]*amask*J["sgn"]
        if land_mode and landing.get("controller") is not None:
            # flight + landing by the Phase 14 balance controller (balance.py): implicit servos with physical stiffness,
            # targets updated at a fixed 500 Hz
            ctl = landing["controller"]
            if not land.get("servo_on"):
                for i_a, (aid, dof) in enumerate(zip(J["aid"], J["dof"])):
                    ieff = 1.0/float(S["Minv_key"][dof, dof]); kp = ctl.p["stiff"]**2*ieff
                    _as_servo(m, int(aid), kp, 2.0*np.sqrt(kp*ieff), float(J["tau"][i_a]))
                land["servo_on"] = True; ctl._kc = None; ctl.t_touch = None
            k_c = int(round(s*dt/0.002, 6))
            if k_c != ctl._kc:
                ctl._kc = k_c; d.ctrl[:] = ctl.targets(k_c*0.002, d, prev_fz > 1.0)
            cmd_j = None
        elif land_mode and landing.get("servo"):
            # flight + landing, SERVO variant: driven joints become the same implicit stiff position servos that hold
            # the robot at rest in settle(), targeting the landing posture (torque limits still enforced)
            if not land.get("servo_on"):
                for i_a, (aid, dof) in enumerate(zip(J["aid"], J["dof"])):
                    ieff = 1.0/float(S["Minv_key"][dof, dof]); kp = landing.get("stiff", 1.0)*ieff/(dt**2)
                    _as_servo(m, int(aid), kp, 2.0*np.sqrt(kp*ieff), float(J["tau"][i_a]))
                land["servo_on"] = True
            d.ctrl[J["aid"]] = landing["q"]
            cmd_j = None
        elif land_mode:
            # flight + landing: PD hold toward the landing posture (normalised torque), replaces knots/tracker
            q_ = d.qpos[J["qad"]]; w_ = d.qvel[J["dof"]]
            cmd_j = (landing["kp"]*(landing["q"] - q_) - landing["kd"]*w_)/J["tau"]
        elif ctrl_fn is not None: cmd_j = cmd_j + ctrl_fn(s*dt, d, S)
        w = d.qvel[J["dof"]]
        if cmd_j is None:                      # servo landing mode: MuJoCo applies the servo force
            tq = np.array(d.actuator_force[J["aid"]])
        else:
            act_j += alpha*(cmd_j - act_j)
            frac = np.maximum(0.0, 1.0 - np.abs(w)/om_v)
            hi = np.where(w > 0, tau_v*frac, tau_v); lo_ = np.where(w < 0, tau_v*frac, tau_v)
            tq = np.clip(act_j*tau_v, -lo_, hi); ratio_w = np.abs(w)/om_v; overspeed = max(overspeed, float(np.max(ratio_w))); over_j = np.maximum(over_j, ratio_w)
            d.ctrl[J["aid"]] = tq
        ncmd += len(tq); nsat += int(np.sum(np.abs(tq) >= 0.98*tau_v))
        pw = tq*w; np.add.at(work_v, ti_, np.where(pw > 0, pw, 0.0)*dt)
        if v_takeoff is None:
            ppow = max(ppow, float(np.sum(np.where(pw > 0, pw, 0.0))))
            util_sum += float(np.mean(np.abs(tq)/tau_v)); util_n += 1
            pk_util = max(pk_util, float(np.max(np.abs(tq)/tau_v*amask)))
            qn = d.qpos[J["qad"]]*J["sgn"]
            qmin = np.minimum(qmin, qn); qmax = np.maximum(qmax, qn)
        # actuator power, all actuators, evaluated with the force actually applied this step
        if disturb is not None:                 # Phase 14c: external pushes on the torso, list of (t0, duration, F_xyz)
            d.xfrc_applied[:] = 0.0
            for (t0_, dur_, F_) in disturb:
                if t0_ <= tt < t0_ + dur_: d.xfrc_applied[S["torso_body"], :3] += F_
        qv0 = d.qvel.copy()
        backemf_brake(m, d, S)
        mujoco.mj_step(m, d)
        W_act += float(np.dot(d.actuator_force, 0.5*(qv0[S["act_dof"]] + d.qvel[S["act_dof"]])))*dt
        # Phase 10 (AD3): explicit dissipation bookkeeping so the energy balance can CLOSE.
        # Constraint power covers contacts, joint limits and frictionloss; passive covers damping.
        _qm = 0.5*(qv0 + d.qvel)
        ACC["con"] += float(np.dot(d.qfrc_constraint, _qm))*dt; ACC["pas"] += float(np.dot(d.qfrc_passive, _qm))*dt + float(np.dot(d.qfrc_applied, _qm))*dt   # back-EMF brake work (qfrc_applied) counts as passive dissipation
        vcom = P.com_velocity(m, d); vz = float(vcom[2])
        peak_v = max(peak_v, vz)
        fz = 0.0
        for ci in range(d.ncon):
            c = d.contact[ci]
            if floor not in (c.geom1, c.geom2): continue      # self-contacts are internal forces
            fv = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fv); fz += fv[0]
            if slip_cap is not None and v_takeoff is None and fv[0] > 5.0:
                mu_c = float(c.friction[0]); ratio = float(np.hypot(fv[1], fv[2]))/(mu_c*fv[0])
                slip_viol = max(slip_viol, ratio - slip_cap); slip_steps += int(ratio > slip_cap)
            if v_takeoff is None: max_pen = max(max_pen, -float(c.dist))
            if (v_takeoff is None or land_mode) and floor in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == floor else c.geom1
                if other not in foot:
                    if land_mode: land["nonfoot_after"] = True
                    else: nonfoot = True
        qq = d.qpos[lq]
        lim_exc = max(lim_exc, float(np.max(np.maximum(llo - qq, qq - lhi))))
        if stop_margin is not None and not snaps:
            margin_viol = max(margin_viol, float(np.max(np.maximum(llo + stop_margin - qq, qq - (lhi - stop_margin)))))
        if v_takeoff is None:
            peak_grf = max(peak_grf, fz)
            grf_spike = max(grf_spike, abs(fz - prev_fz))
            unload_t += dt if fz < 1.0 else 0.0
            impulse += fz*dt; zmin = min(zmin, float(d.subtree_com[0][2]))
            W_held += float(np.dot(d.actuator_force[held_a], d.qvel[held_d]))*dt
        prev_fz = fz
        if land_mode:
            if land["touchdown_t"] is None and fz >= 1.0 and not in_flight: land["touchdown_t"] = s*dt
            if nonfoot: land["nonfoot_after"] = True
            land["max_tilt"] = max(land["max_tilt"], float(np.arccos(np.clip(d.xmat[1][8], -1, 1))))
        if record:
            rec["t"].append(s*dt); rec["vz"].append(vz); rec["fz"].append(fz)
            rec["z"].append(float((m.body_mass[1:, None]*d.xipos[1:]).sum(0)[2]/S["M"]))
            for t in S["types"]:
                (jid, dof, qad, tau, om, sgn, aid) = S["tmap"][t][0]
                rec["qvel"][t].append(float(sgn*d.qvel[dof]))
                rec.setdefault("q", {x: [] for x in S["types"]})[t].append(float(sgn*d.qpos[qad]))
                rec["tq"][t].append(float(sum(d.actuator_force[x[6]]*x[5] for x in S["tmap"][t])))
        # TAKEOFF: the instant contact is lost. The bound applies here, not to transients.
        # TAKEOFF (Phase 9, adversarial-pass defect A2): a flight is a loss of foot contact with
        # UPWARD COM velocity that is SUSTAINED for FLIGHT_CONFIRM seconds. The Phase 8 rule
        # accepted a single step; 15/32 repro solutions re-touched the floor within 60 ms. The
        # snapshot is taken at the first flight step and kept only if the flight is confirmed.
        if in_flight:
            if fz >= 1.0:
                in_flight = False; cand = None
            elif cand is not None and (s*dt - cand_t) >= FLIGHT_CONFIRM - 1e-12:
                snaps.append(cand); cand = None
                if landing and not land_mode: land_mode = True
                if score == "first" and not record: break
        elif fz < 1.0 and vz > 0.0 and s > 0 and not (score == "first" and snaps):
            in_flight = True; cand_t = s*dt
            cand = _finalize(vz, s*dt, float(d.energy[0] + d.energy[1]), W_act,
                             0.5*S["M"]*vz*vz, float(d.xmat[1][8]), work_v.copy(),
                             float(d.subtree_com[0][2]), float(vcom[0]),
                             float(np.linalg.norm(d.qvel[3:6])), flight_rotation(m, d, vz))
    if land is not None and land.get("servo_on"):     # restore torque motors (the model is shared across rollouts)
        for i_a, aid in enumerate(J["aid"]): _as_motor(m, int(aid), float(J["tau"][i_a]))
        land["servo_on"] = False
    if cand is not None: snaps.append(cand)             # flight still airborne at horizon end
    if score == "best":
        out = max(snaps, key=lambda o: o["v"]) if snaps else _finalize(None, None, None, None, 0.0, 1.0, None, None, None, None, None)
    else:
        out = snaps[0] if snaps else _finalize(None, None, None, None, 0.0, 1.0, None, None, None, None, None)
    if landing or stop_margin is not None:
        out = snaps[0] if (landing and snaps) else out          # with a landing controller the jump is the FIRST flight
    out["n_flights"] = len(snaps); out["knot_dt"] = knot_dt
    if stop_margin is not None: out["margin_viol"] = margin_viol
    if slip_cap is not None: out["slip_viol"] = max(0.0, slip_viol); out["slip_steps"] = slip_steps
    if landing:
        up_end = float(d.xmat[1][8]); z_end = float(d.subtree_com[0][2]); vc_end = float(np.linalg.norm(P.com_velocity(m, d)))
        land.update(up_end=up_end, z_end=z_end, v_end=vc_end, took_off=bool(snaps))
        # standing threshold: 0.40 m on G1 (Phase 14 rule, start COM 0.666 -> unchanged), 0.6 x start COM on a taller robot (H1: 0.57)
        land["ok"] = bool(snaps and land["touchdown_t"] is not None and not land["nonfoot_after"] and up_end > 0.85 and z_end > max(0.40, 0.6*z_start) and vc_end < 0.5)
        out["landing"] = land
    if record: out["trace"] = rec
    return out

ROT_MAX = 0.35   # rad; task T2 "landable" flight rotation limit, INVENTED (20 deg), sensitivity reported

def solve(active_types, seed, target=None, caps_scale=None, iters=90, popsize=16, task="T1",
          x0=None, sigma0=0.35, S=None):
    S = S or setup(); n = len(S["types"])*NK
    def obj(x):
        r = rollout(x, active_types, caps_scale, S=S)
        # graded penalty, not hard rejection: a hard reject flattens the fitness landscape and
        # CMA-ES terminates immediately with no usable direction.
        pen = 0.0
        vr = r["validity"]
        ce = vr["contact_energy"] - vr["energy_tol"]
        if ce > 0: pen += ce*0.05
        er = vr["energy_residual"] - vr["energy_tol"]
        if er > 0: pen += er*0.05
        if vr["nonfoot_contact"]: pen += 0.5
        if task == "T2":
            ex = r["mech"]["flight_rotation"] - ROT_MAX
            if ex > 0: pen += ex*0.5
        if target is None: return -r["height"] + pen
        if isinstance(task, tuple) and task[0] == "duration":
            # demand x duration control: hit the target AND push off for a prescribed time
            Tc = task[1]; Tp = r["mech"]["T_push"]
            return abs(r["v"]-target) + 2.0*abs(Tp - Tc) + pen
        if task == "reserve":
            # demand as a CONSTRAINT (hit the target to 0.03 m/s), torque reserve as the cost:
            # minimise peak normalised torque (Faity et al. 2021's reserve mechanism).
            return 10.0*max(0.0, abs(r["v"]-target) - 0.03) + r["mech"]["peak_util"] + pen
        return abs(r["v"]-target) + pen
    es = cma.CMAEvolutionStrategy(np.zeros(n) if x0 is None else np.asarray(x0), sigma0,
                                  {"bounds": [-1, 1], "seed": seed, "verbose": -9,
                                   "maxiter": iters, "popsize": popsize})
    while not es.stop():
        X = es.ask(); es.tell(X, [obj(x) for x in X])
    return es.result.xbest, rollout(es.result.xbest, active_types, caps_scale, record=True, S=S)

def job(a):
    kind, payload = a
    S = setup()
    if kind == "max":
        act, seed = payload
        _, r = solve(set(act), seed, iters=150)
        return dict(kind="max", active=list(act), seed=seed, **{k: v for k, v in r.items()})
    if kind == "demand":
        tgt, seed = payload
        _, r = solve(set(S["types"]), seed, target=tgt, iters=150)
        return dict(kind="demand", target=tgt, seed=seed, v=r["v"], height=r["height"],
                    participation=r["participation"], work=r["work"], valid=r["valid"],
                    grf_bw=r["grf_bw"], max_penetration=r["max_penetration"])
    if kind == "perm":
        scale, seed, tag = payload
        _, r = solve(set(S["types"]), seed, caps_scale=scale, iters=150)
        return dict(kind="perm", tag=tag, seed=seed, participation=r["participation"],
                    height=r["height"], v=r["v"], work=r["work"])
    return {}

if __name__ == "__main__":
    S = setup()
    T = S["types"]; print("control types:", T, flush=True)
    jobs = []
    for seed in (1, 2, 3):
        jobs.append(("max", (tuple(T), seed)))
        for t in T: jobs.append(("max", ((t,), seed)))
        jobs.append(("max", (tuple(x for x in T if "knee" in x or "hip" in x), seed)))
    for tgt in (0.5, 0.9, 1.3, 1.7, 2.1, 2.4):        # demand sweep for the central test
        for seed in (1, 2, 3): jobs.append(("demand", (tgt, seed)))
    # I.3 assignment test: permute capacity scales across types, multiset fixed
    rng = np.random.default_rng(5)
    base = {t: (1.0, 1.0) for t in T}
    multiset = [(0.4, 0.4), (0.7, 0.7), (1.0, 1.0), (1.3, 1.3), (1.6, 1.6), (2.2, 2.2)][:len(T)]
    for i in range(6):
        p = list(multiset); rng.shuffle(p)
        sc = {t: p[k] for k, t in enumerate(T)}
        for seed in (1, 2):
            jobs.append(("perm", (sc, seed, f"perm{i}")))
    print(f"{len(jobs)} dynamic solves", flush=True)
    t0 = time.time()
    with Pool(14) as pool: res = pool.map(job, jobs)
    json.dump(res, open(f"{ROOT}/results/branch_i.json", "w"), indent=1, default=float)
    print(f"done in {time.time()-t0:.0f}s", flush=True)
