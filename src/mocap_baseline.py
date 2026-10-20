"""North-star baseline: human motion capture -> GMR retarget -> G1, executed in the SAME plant
(torque motors on the sagittal study joints, implicit holds elsewhere, 40 ms actuator lag,
linear torque-speed envelope, identical contact, takeoff definition, validity report).

B0      naive: PD tracking of the retargeted sagittal joint trajectory; tracking gains chosen
        for TRACKING FIDELITY (RMS joint error to the reference up to its takeoff), never for
        jump height. Timing: human (1.0) and Froude-scaled (sqrt(L_robot/L_human) = 0.87).
B1(l)   retarget + the SAME CMA-ES refinement the native method uses: additive normalised
        torque knots (6 types x NK) + a time-scale factor, objective = -height + l*D, where D
        is the mean squared deviation (rad^2) of the executed sagittal joints from the
        reference over push-off. l = inf is B0; l = 0 is unconstrained refinement seeded by
        the mocap tracker (separates template disadvantage from optimizer advantage).
OURS    robot-native: same knots, no reference, from the SAME start pose, same horizon.
"""
import sys, os, json, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, cma, branch_i as BI
from scipy.interpolate import CubicSpline

BASE = os.environ.get("MOCAP_BASELINE_DIR", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "baseline"))
CLIPS = {"obstacles5_subject3@1585": None, "fightAndSports1_subject4@4684": None}
EXCLUDED = {"fightAndSports1_subject1@3739": "retarget quality check failed: knee range "
            "[-0.09,-0.05] rad (no knee flexion) during a 13 cm pelvis dip"}
PRE = 24            # start 0.8 s (24 frames at 30 fps) before the human takeoff frame
HORIZON = 1.0
NKB = 20            # knots over 1.0 s = same 50 ms spacing as the native 10 over 0.5 s


def _foot_pitch(m, d, body):
    R = d.xmat[body].reshape(3, 3)
    return float(np.arctan2(-R[2, 0], R[2, 2]))       # rotation about world y of the foot x-axis


def _flatten_and_certify(Qframe, dpitch, S0, cg):
    import plant as P
    from scipy.spatial.transform import Rotation as Rot
    m = S0["m"]; d = mujoco.MjData(m)
    q = np.array(S0["q0"]); q[7:] = Qframe[7:]
    # bilateral symmetrisation of the START (the native method enforces the same symmetry):
    # sagittal joints -> mean of left and right; hip/ankle roll and hip yaw -> 0
    jid = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
    for n in ("hip_pitch_joint", "knee_joint", "ankle_pitch_joint", "shoulder_pitch_joint", "elbow_joint"):
        mu = 0.5*(Qframe[jid("left_"+n)] + Qframe[jid("right_"+n)]); q[jid("left_"+n)] = mu; q[jid("right_"+n)] = mu
    for n in ("hip_roll_joint", "hip_yaw_joint", "ankle_roll_joint"):
        q[jid("left_"+n)] = 0.0; q[jid("right_"+n)] = 0.0
    q[jid("waist_roll_joint")] = 0.0; q[jid("waist_yaw_joint")] = 0.0
    w, x, y, z = Qframe[3:7]; e = Rot.from_quat([x, y, z, w]).as_euler("ZYX")
    qq = Rot.from_euler("ZYX", [0.0, e[1] + dpitch, 0.0]).as_quat(); q[3:7] = [qq[3], qq[0], qq[1], qq[2]]  # heading and roll removed (bilateral)
    offs = {}
    for side in ("left", "right"):
        ja = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_ankle_pitch_joint")
        jr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_ankle_roll_joint")
        fb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
        q[m.jnt_qposadr[jr]] = 0.0
        for _ in range(20):
            d.qpos[:] = q; mujoco.mj_forward(m, d)
            fp = _foot_pitch(m, d, fb); q[m.jnt_qposadr[ja]] -= fp
            if abs(fp) < 1e-6: break
    offs = {int(k): float(q[k] - Qframe[k]) for k in range(7, m.nq) if abs(q[k] - Qframe[k]) > 1e-9}
    rb = S0["rb"]; rb.d.qpos[:] = q; mujoco.mj_forward(m, rb.d)
    rb.d.qpos[2] -= rb.lowest_z(); mujoco.mj_forward(m, rb.d); q = rb.d.qpos.copy()
    return q, offs, P.static_certificate(m, d, q, coll_geoms=cg)


def project_start(Qframe):
    """Nearest statically-certified start to a retargeted frame:
    retargeted joint angles; heading removed; feet made flat by ankle offsets; then the
    SMALLEST whole-body lean change (pelvis pitch, searched outward from 0 in 0.02 rad steps
    to +-0.6) whose pose passes the static-equilibrium LP. This is the balance-resolution step
    of a competent retargeting pipeline; it is chosen by feasibility, blind to jump outcome."""
    import plant as P
    S0 = BI.setup(); m = S0["m"]
    cg = P.PlantContractVerifier(m, mujoco.MjData(m), P.PlantContract(BI.MJCF, "torque")).collision_geoms()
    # balance resolution: among certified leans in [-0.6, 0.6], choose the one that CENTRES the
    # COM over the support region (maximum support margin). The first version took the smallest
    # certifying lean; those starts sat within 1-27 mm of the toe edge and tipped over during the
    # settle (PCTV-5 caught it).
    best = None; last = None
    for dp in np.round(np.arange(-0.6, 0.601, 0.02), 3):
        q, offs, cert = _flatten_and_certify(Qframe, dp, S0, cg)
        last = (q, offs, cert, dp)
        if cert["status"] == "STATIC_FEASIBLE" and cert["torque_margin"] >= 0.1:
            if best is None or cert["support_margin"] > best[2]["support_margin"]:
                best = (q, offs, cert, dp)
    if best is not None:
        q, offs, cert, dp = best
        return q, offs, dict(cert, lean_change=float(dp))
    q, offs, cert, dp = last
    return q, offs, dict(cert, lean_change=None)


def load_clip(key, pkl="retargeted_g1_all.pkl", start_frame=None):
    D = pickle.load(open(f"{BASE}/{pkl}", "rb"))[key]
    Q = np.array(D["qpos"]); a = D["takeoff_frame"] - D["frame0"]
    s0 = a - PRE if start_frame is None else start_frame - D["frame0"]
    seg = Q[s0:a+15].copy()
    q0, offs, cert = project_start(seg[0])
    for qa, o in offs.items(): seg[:, qa] += o          # same foot-flattening offset throughout
    tref = (a-s0)/D["fps"]; hor = round(tref + 0.3, 2)
    return dict(key=key, start=q0, Q=seg, fps=D["fps"], takeoff_ref=tref, horizon=hor,
                nk=int(round(hor/0.05)),
                start_certificate={k: v for k, v in cert.items() if k not in ("tau", "forces")},
                ankle_offsets=offs)


class Tracker:
    """ctrl_fn for BI.rollout: normalised PD command toward the reference for every driven
    joint (per joint, not shared L/R), plus deviation bookkeeping."""
    def __init__(self, S, clip, omega=20.0, zeta=1.0, tscale=1.0, gain=1.0):
        J = S["jarr"]; Qs = clip["Q"]; t = np.arange(len(Qs))/clip["fps"]
        self.sp = CubicSpline(t*tscale, Qs[:, J["qad"]], bc_type="clamped")
        self.dsp = self.sp.derivative()
        self.tend = t[-1]*tscale; self.J = J
        # precompute the reference on the simulation time grid (spline evaluation per step
        # dominated rollout cost)
        dt = S["m"].opt.timestep; tg = np.arange(0, clip["horizon"] + 2*dt, dt)
        tc = np.minimum(tg, self.tend)
        self.QR = self.sp(tc); self.QDR = self.dsp(tc); self.QDR[tg >= self.tend] = 0.0; self.dt = dt
        Minv = S["Minv_key"]
        ieff = np.array([1.0/Minv[dof, dof] for dof in J["dof"]])
        self.kp = gain*omega**2*ieff; self.kd = gain*2*zeta*omega*ieff
        self.tau = J["tau"]; self.dev = 0.0; self.n = 0; self.ref_to = clip["takeoff_ref"]*tscale

    def __call__(self, t, d, S):
        i = min(int(round(t/self.dt)), len(self.QR) - 1)
        qr = self.QR[i]; qdr = self.QDR[i]
        q = d.qpos[self.J["qad"]]; qd = d.qvel[self.J["dof"]]
        if t <= self.ref_to:
            self.dev += float(np.mean((q - qr)**2)); self.n += 1
        return (self.kp*(qr - q) + self.kd*(qdr - qd))/self.tau

    @property
    def D(self): return self.dev/max(self.n, 1)


def run(S, clip, x, omega, tscale, record=False, caps_scale=None, knot_dt=None):
    tr = Tracker(S, clip, omega=omega, tscale=tscale)
    r = BI.rollout(x, set(S["types"]), caps_scale, record=record, S=S, ctrl_fn=tr, horizon=clip["horizon"],
                   score="best", knot_dt=knot_dt)
    r["deviation"] = tr.D
    return r


def tune_gains(S, clip, tscale):
    """Tracking-fidelity gain selection: kinematic tracking error over the reference push-off
    under a FIXED-BASE-free rollout with zero residual. Chooses omega; blind to jump height."""
    x0 = np.zeros(len(S["types"])*clip["nk"]); best = None; table = []
    for om in (10, 20, 40, 60, 90, 130):
        r = run(S, clip, x0, om, tscale)
        table.append((om, r["deviation"], r["v"], r["valid"]))
        if np.isfinite(r["deviation"]) and (best is None or r["deviation"] < best[1]): best = (om, r["deviation"])
    return best[0], table


def solve_b1(S, clip, lam, seed, omega, iters, popsize, task="T1", native=False):
    n = len(S["types"])*clip["nk"] + (0 if native else 1)
    def unpack(z):
        if native: return z, 1.0
        return z[:-1], 0.87 + 0.3*z[-1]              # time scale in [0.57, 1.17]
    def obj(z):
        x, ts = unpack(z)
        if native:
            r = BI.rollout(x, set(S["types"]), None, S=S, horizon=clip["horizon"], score="best"); r["deviation"] = 0.0
        else:
            r = run(S, clip, x, omega, ts)
        vr = r["validity"]; pen = 0.0
        pen += max(0, vr["contact_energy"]-vr["energy_tol"])*0.05 + max(0, vr["energy_residual"]-vr["energy_tol"])*0.05
        if vr["nonfoot_contact"]: pen += 0.5
        if task == "T2": pen += max(0.0, r["mech"]["flight_rotation"] - BI.ROT_MAX)*0.5
        if task == "T2H":   # Phase 10 B5: the landable limit as a HARD constraint (rejection-scale penalty)
            ex = r["mech"]["flight_rotation"] - BI.ROT_MAX
            if ex > 0: pen += 1.0 + 5.0*ex
        return -r["height"] + pen + (0.0 if native else lam*r["deviation"])
    es = cma.CMAEvolutionStrategy(np.zeros(n), 0.35, {"bounds": [-1, 1], "seed": seed, "verbose": -9,
                                                     "maxiter": iters, "popsize": popsize})
    while not es.stop():
        X = es.ask(); es.tell(X, [obj(z) for z in X])
    x, ts = unpack(es.result.xbest)
    if native:
        # native solution replayed with a zero-gain tracker purely to MEASURE its deviation
        # from the mocap reference (the Pareto x-axis); omega=0 applies no tracking force
        r = run(S, clip, x, 0.0, 1.0, record=True)
    else:
        r = run(S, clip, x, omega, ts, record=True)
    r["x"] = np.asarray(x).tolist(); r["tscale"] = ts
    return r


def analytic_clip(clip, bank=os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/pose_bank_certified_v2.pkl"), robot="G1"):
    """Q3 decisive test: a NON-mocap strategy prior. Same start pose and horizon as the clip, but the
    tracked reference is built from the analytic pose bank's full-set certified crouch and takeoff
    poses (clipped to joint ranges): start -> crouch at (takeoff_ref - 0.25 s) -> takeoff pose at
    takeoff_ref, then hold. Sagittal joints only; all others keep the start values."""
    import pickle
    D = pickle.load(open(bank, "rb"))[robot]
    rec = max(D["bank"], key=lambda r: len(r["groups"]))      # full group set (6 on G1, 5 on H1)
    S0 = BI.setup() if robot == "G1" else BI.setup(start=clip["start"], robot=robot); m = S0["m"]; rb = S0["rb"]
    def pose_q(tv):
        q = np.array(clip["start"])
        for t, v in tv.items():
            for n in rb.groups_of_type(t):
                r = rb.J[n]; s = -1.0 if (rb.mirror[t] and r["side_geom"] == "right") else 1.0
                q[r["qadr"]] = float(np.clip(s*v, r["lo"], r["hi"]))
        return q
    qc, qt = pose_q(rec["t0"]), pose_q(rec["tT"])
    fps = clip["fps"]; tref = clip["takeoff_ref"]; n = clip["Q"].shape[0]
    t = np.arange(n)/fps; Q = np.zeros_like(clip["Q"])
    t1 = max(0.1, tref - 0.25)
    for i, ti in enumerate(t):
        if ti <= t1: w = ti/t1; Q[i] = (1-w)*clip["start"] + w*qc
        elif ti <= tref: w = (ti-t1)/(tref-t1); Q[i] = (1-w)*qc + w*qt
        else: Q[i] = qt
    return dict(clip, Q=Q, key=clip["key"] + "|analytic_prior")
