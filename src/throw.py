"""Phase 11: overarm ball throw on the corrected G1/H1 plant (PHASE11_PLAN.md).

Reuses the Phase 8-10 plant machinery (torque motors, implicit position holds, settle, realised capacity
limits, dt-consistent holds) by building a setup dict in branch_i's format with ONE CONTROL CHANNEL PER
DRIVEN JOINT (no bilateral symmetry). The ball (0.145 kg) is a point mass at the palm site added to the hand
body; release velocity = palm-site velocity at the (optimised) release instant.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, cma
import branch_i as BI, robot_bound as R, plant as P
from validity import TrajectoryValidityReport

G = 9.80665
BALL = 0.145
OVERSPEED_TOL = 1.10   # validity: max |joint speed| / rated speed over the rollout (2026-10-11 audit: the 25.2 m/s throw whipped the wrist to 3.1x)
HOR = 0.8; NK = 10; T_REL_MIN = 0.15
LAG_TAU = 0.040   # first-order actuator lag time constant (s); the construct campaigns used 40 ms; a caller may set it (M1 kick_goal: 10 ms)
ELEV = (0.0, np.radians(45.0)); AZ = np.radians(20.0)
CFG = {
    "G1": dict(hand_site="right_palm",
               driven=["left_hip_pitch_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
                       "right_hip_pitch_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint",
                       "waist_yaw_joint", "waist_pitch_joint",
                       "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                       "right_elbow_joint", "right_wrist_pitch_joint",
                       "left_shoulder_pitch_joint", "left_shoulder_roll_joint"],
               groups={"legs": ["hip", "knee", "ankle"], "trunk": ["waist"], "shoulder": ["right_shoulder"],
                       "elbow": ["right_elbow"], "wrist": ["right_wrist"], "counter_arm": ["left_shoulder"]},
               segments=["pelvis", "torso_link", "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_yaw_link"]),
    # H1 (J11, 2026-10-11): a 4-DOF arm with no wrist and a yaw-only torso (no waist pitch). The palm site is added to a
    # copy of the menagerie model (h1_palm.xml / scene_palm.xml, forearm end at the collision sphere). Capacity-weighted
    # priors and the quasi-static reach then run unchanged on it; "wrist" is an empty group here.
    "G1H": None,   # filled below: the G1 configuration on the hands model
    "H1": dict(hand_site="right_palm", mjcf=os.path.join(os.environ.get("MENAGERIE_DIR", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "models", "mujoco_menagerie")), "unitree_h1/scene_palm.xml"),
               driven=["left_hip_pitch", "left_hip_yaw", "left_knee", "left_ankle",
                       "right_hip_pitch", "right_hip_yaw", "right_knee", "right_ankle",
                       "torso",
                       "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
                       "left_shoulder_pitch", "left_shoulder_roll"],
               groups={"legs": ["hip", "knee", "ankle"], "trunk": ["torso"], "shoulder": ["right_shoulder"],
                       "elbow": ["right_elbow"], "wrist": ["right_wrist"], "counter_arm": ["left_shoulder"]},
               segments=["pelvis", "torso_link", "right_shoulder_yaw_link", "right_elbow_link"]),
}
# HELD-OUT EMBODIMENTS (2026-10-16): registration only (names from src/heldout_onboard.py); driven = the sprint's set
CFG["Apollo"] = dict(hand_site="right_palm", mjcf=None,
                     driven=[f"{s}_{j}" for s in ("left", "right") for j in ("hip_pitch_joint", "hip_roll_joint", "hip_yaw_joint", "knee_joint", "ankle_pitch_joint", "ankle_roll_joint")]
                            + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"] + [f"{s}_{j}" for s in ("right", "left") for j in ("shoulder_pitch_joint", "shoulder_roll_joint", "elbow_joint")],
                     groups={}, segments=["pelvis", "torso_link", "right_shoulder_fe_link", "right_elbow_fe_link", "right_wrist_pitch_link"])
CFG["T1"] = dict(hand_site="right_palm", mjcf=None,
                 driven=[f"{s}_{j}" for s in ("left", "right") for j in ("hip_pitch_joint", "hip_roll_joint", "hip_yaw_joint", "knee_joint", "ankle_pitch_joint", "ankle_roll_joint")]
                        + ["waist_yaw_joint"] + [f"{s}_{j}" for s in ("right", "left") for j in ("shoulder_pitch_joint", "shoulder_roll_joint", "elbow_joint")],
                 groups={}, segments=["pelvis", "waist_link", "right_shoulder_pitch_link", "right_elbow_link", "right_hand_link"])
CFG["G1H"] = dict(CFG["G1"], mjcf=os.path.join(os.environ.get("MENAGERIE_DIR", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "models", "mujoco_menagerie")), "unitree_g1/scene_hands_palm.xml"),
                  driven=CFG["G1"]["driven"] + ["right_wrist_roll_joint", "right_wrist_yaw_joint"])   # manipulation needs the full 3-DOF wrist (orientation control)
_CACHE = {}


def group_of(robot, jn):
    if "left_shoulder" in jn or "left_elbow" in jn: return "counter_arm"
    if "right_shoulder" in jn: return "shoulder"
    if "right_elbow" in jn: return "elbow"
    if "wrist" in jn: return "wrist"
    if "waist" in jn or jn == "torso" or jn.startswith("torso_"): return "trunk"     # H1: the single torso (yaw) joint is the trunk
    if any(k in jn for k in ("hip", "knee", "ankle")): return "legs"
    return "other"


def setup(robot="G1", start=None, driven=None, mjcf=None, ball=True, site=None, mass=None, support="both"):
    """mjcf: an alternative scene (engine.world scenes with objects); ball=False: no merged ball mass (the world carries objects).
    GENERAL EFFECTOR (construct, 2026-10-11): site = effector site name (default the robot's hand site); mass = point mass merged at
    the effector (default BALL when ball=True); support = "both" | "left" | "right": the feet that must stay loaded (a kick's swing foot
    is free and its floor contact is not a stance violation); S["support_mask"] marks the driven joints of the support legs."""
    key = (robot, None if start is None else np.asarray(start).tobytes(), tuple(driven or ()), mjcf, ball, site, mass, support)
    if key in _CACHE: return _CACHE[key]
    cfg = BI.ROBOTS[robot]; tc = CFG[robot]
    rb = R.RobotBody(mjcf or tc.get("mjcf") or cfg["mjcf"], cfg["spec"], capacity="peak")   # a task-config model (e.g. H1 with a palm site) overrides the jump model
    if BI.geom_solref(robot): rb.m.geom_solref[:] = BI.geom_solref(robot)    # STOCK_CONTACT=1 keeps the MJCF contact
    m = BI.to_torque_motors(rb)
    # ball: point mass at the palm site, merged into the hand body (mass + COM; inertia about the new COM is
    # left unchanged: the omitted parallel-axis terms are < 1e-4 kg m^2 for a 0.145 kg mass a few cm away)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site or tc["hand_site"]); hb = int(m.site_bodyid[sid]); assert sid >= 0, site
    m_add = (BALL if ball else 0.0) if mass is None else float(mass)
    if m_add > 0:
        mb = float(m.body_mass[hb]); p = np.array(m.site_pos[sid])
        m.body_ipos[hb] = (mb*m.body_ipos[hb] + m_add*p)/(mb + m_add); m.body_mass[hb] = mb + m_add
        mujoco.mj_setConst(m, mujoco.MjData(m))
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis"); robot_bodies = set()
    for b_ in range(m.nbody):
        x_ = b_
        while x_ > 0 and x_ != pel: x_ = int(m.body_parentid[x_])
        if x_ == pel: robot_bodies.add(b_)
    robot_geoms = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in robot_bodies}
    robot_dofs = np.array([i for i in range(m.nv) if int(m.body_rootid[int(m.jnt_bodyid[int(m.dof_jntid[i])])]) == int(m.body_rootid[pel])])
    driven = list(driven or tc["driven"])
    types = driven
    tmap = {}
    for jn in types:
        J = rb.J[jn]; jid = J["id"]
        tmap[jn] = [(jid, int(m.jnt_dofadr[jid]), int(m.jnt_qposadr[jid]), J["tau"], J["omega"], 1.0,
                     [a for a in range(m.nu) if m.actuator_trnid[a][0] == jid][0])]
    drv = {tmap[t][0][6] for t in types}
    kid = -1
    for kname in cfg["keyframes"]:
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, kname)
        if kid >= 0: break
    dk = mujoco.MjData(m)
    if start is not None and len(start) < m.nq: start = np.concatenate([np.asarray(start, float), m.key_qpos[kid][len(start):]])   # robot-only start in a world scene
    dk.qpos[:] = m.key_qpos[kid] if start is None else start; mujoco.mj_forward(m, dk)
    Mf = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, Mf, dk.qM); Minv = np.linalg.inv(Mf)
    held = []
    for a in range(m.nu):
        if a in drv: continue
        jid = int(m.actuator_trnid[a][0]); dof = int(m.jnt_dofadr[jid])
        tau = float(m.jnt_actfrcrange[jid][1]) or 25.0
        ieff = 1.0/float(Minv[dof, dof]); kp = ieff/(m.opt.timestep**2)
        if cfg.get("hold_sat_rad"): kp = min(kp, tau/cfg["hold_sat_rad"])
        kv = 2.0*np.sqrt(kp*ieff); BI._as_servo(m, a, kp, kv, tau)
        held.append((a, dof, int(m.jnt_qposadr[jid]), tau, float(kp), float(kv)))
    fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    fbodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in cfg["foot_bodies"]]
    feet = [{g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b} for b in fbodies]
    J = dict(qad=np.array([tmap[t][0][2] for t in types]), dof=np.array([tmap[t][0][1] for t in types]),
             aid=np.array([tmap[t][0][6] for t in types]), tau=np.array([tmap[t][0][3] for t in types], float),
             om=np.array([tmap[t][0][4] for t in types], float), ti=np.arange(len(types)), sgn=np.ones(len(types)))
    sup = {"both": [0, 1], "left": [0], "right": [1]}[support]; sides = [cfg["foot_bodies"][i] for i in sup]
    support_mask = np.array([group_of(robot, t) == "legs" and any(t.startswith(("left" if "left" in fb else "right")) for fb in sides) for t in types])
    S = dict(support=sup, support_mask=support_mask, free_foot_geoms=set().union(*[feet[i] for i in (0, 1) if i not in sup]) if len(sup) < 2 else set(),
             jarr=J, Minv_key=Minv, rb=rb, m=m, types=types, tmap=tmap,
             q0=np.array(m.key_qpos[kid]) if start is None else np.array(start), M=float(sum(m.body_mass[b_] for b_ in robot_bodies)), robot_geoms=robot_geoms, robot_bodies=robot_bodies, robot_dofs=robot_dofs,
             driven=sorted(drv), held=held, floor=fl, foot_geoms=set().union(*feet), feet=feet,
             act_dof=[int(m.jnt_dofadr[int(m.actuator_trnid[a][0])]) for a in range(m.nu)], robot=robot,
             mjcf=cfg["mjcf"], site=sid, hand_body=hb,
             seg=[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in tc["segments"]],
             groups=[group_of(robot, t) for t in types])
    BI.apply_dt(S, 0.004)                                 # settles too
    _CACHE[key] = S
    return S


def release_time(r, horizon=HOR):
    return T_REL_MIN + (horizon - T_REL_MIN)*(float(np.clip(r, -1, 1)) + 1)/2


def rollout(x, S, ctrl_fn=None, horizon=HOR, record=False, caps=None):
    """x = knots (len(types) x NK, flattened) + release parameter. Returns release speed, direction,
    validity and mechanics. caps: per-joint torque scale (array) for ablations/sensitivity."""
    m = S["m"]; d = mujoco.MjData(m); BI.settle(S); BI._set_capacity_limits(m, S, None)
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    d.qpos[:] = S["q_settled"]; d.qvel[:] = S["v_settled"]; mujoco.mj_forward(m, d)
    E0 = float(d.energy[0] + d.energy[1]); start_qdot = float(np.max(np.abs(d.qvel[6:])))
    nT = len(S["types"]); kn = np.asarray(x[:-1]).reshape(nT, -1); NKl = kn.shape[1]
    t_rel = release_time(x[-1], horizon); dt = m.opt.timestep; nst = int(round(t_rel/dt))
    J = S["jarr"]; tau_v = J["tau"]*(np.ones(nT) if caps is None else np.asarray(caps)); om_v = J["om"]
    for t, (jid, dof, qad, tau, om, sgn, aid) in zip(S["types"], (S["tmap"][t][0] for t in S["types"])):
        m.actuator_ctrlrange[aid][:] = [-tau_v[S["types"].index(t)], tau_v[S["types"].index(t)]]
        m.actuator_forcerange[aid][:] = m.actuator_ctrlrange[aid]; m.jnt_actfrcrange[jid][:] = m.actuator_ctrlrange[aid]
    a_hold = np.array([S["a_hold"][t] for t in S["types"]])*J["tau"]/tau_v
    act = a_hold.copy()
    alpha = 1.0 - np.exp(-dt/LAG_TAU); knot_dt = horizon/NKl
    for (a_, dof, qad, tau, kp, kd) in S["held"]: d.ctrl[a_] = S["q_settled"][qad]
    W_act = W_con = W_pas = 0.0; work = np.zeros(nT); nonfoot = False; stance_lost = False; peak_grf = 0.0; n_unl = 0; run_unl = [0.0, 0.0]
    seg_w = []; jv = []; tq_trace = []; power_trace = []; q_trace = []; qdot_trace = []; foot_trace = []
    fl = S["floor"]; feet = S["feet"]; vel6 = np.zeros(6); z0 = float(d.subtree_com[0][2]); overspeed = 0.0; over_j = np.zeros(len(J["aid"]))
    for s in range(nst):
        tt = s*dt; k = min(int(tt/knot_dt + 1e-9), NKl-1); k2 = min(k+1, NKl-1)
        f = min(1.0, (tt - k*knot_dt)/knot_dt) if k < NKl-1 else 0.0
        # knots are OFFSETS from the settled holding command: x = 0 holds the stance
        cmd = (S["hold_sched"](tt) if S.get("hold_sched") else a_hold) + kn[:, k] + (kn[:, k2]-kn[:, k])*f    # hold_sched: a support-dependent gravity hold (one-foot support: two-foot hold -> one-foot hold over the preparation)
        if ctrl_fn is not None: cmd = cmd + ctrl_fn(tt, d, S)
        act += alpha*(cmd - act)
        w = d.qvel[J["dof"]]; frac = np.maximum(0.0, 1.0 - np.abs(w)/om_v)
        hi = np.where(w > 0, tau_v*frac, tau_v); lo = np.where(w < 0, tau_v*frac, tau_v)
        tq = np.clip(act*tau_v, -lo, hi); ratio_w = np.abs(w)/om_v; overspeed = max(overspeed, float(np.max(ratio_w))); over_j = np.maximum(over_j, ratio_w)
        d.ctrl[J["aid"]] = tq
        BI.backemf_brake(m, d, S)
        qv0 = d.qvel.copy(); mujoco.mj_step(m, d); qm = 0.5*(qv0 + d.qvel)
        W_act += float(np.dot(d.actuator_force, qm[S["act_dof"]]))*dt
        W_con += float(np.dot(d.qfrc_constraint, qm))*dt; W_pas += float(np.dot(d.qfrc_passive, qm))*dt + float(np.dot(d.qfrc_applied, qm))*dt
        pw = tq*d.qvel[J["dof"]]; work += np.where(pw > 0, pw, 0.0)*dt
        ff = [0.0, 0.0]; fz = 0.0
        for ci in range(d.ncon):
            c = d.contact[ci]
            if fl not in (c.geom1, c.geom2): continue
            fv = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fv); fz += fv[0]
            o = c.geom2 if c.geom1 == fl else c.geom1
            if o in feet[0]: ff[0] += fv[0]
            elif o in feet[1]: ff[1] += fv[0]
            elif o in S.get("robot_geoms", {o}) and o not in S.get("free_foot_geoms", ()): nonfoot = True          # objects resting on the floor are not robot contacts; a swing foot may touch
        # a foot has LEFT the floor only if unloaded for FLIGHT_CONFIRM s (the project-wide flight rule, branch_i /
        # Gate A P17); shorter unloading is contact chatter, counted in unloaded_frac but not a stance loss
        # CUMULATIVE per foot (amended after the v1 campaign: a 20 ms CONTINUOUS rule was exploited by foot
        # "pattering", repeated sub-20 ms unloads totalling 90-180 ms before release)
        for i_f in S.get("support", (0, 1)):
            if s > 5 and ff[i_f] < 1.0: run_unl[i_f] += dt
            if run_unl[i_f] >= BI.FLIGHT_CONFIRM - 1e-12: stance_lost = True
        if s > 5 and min(ff[i_f] for i_f in S.get("support", (0, 1))) < 1.0: n_unl += 1
        peak_grf = max(peak_grf, fz)
        if record:
            sw = []
            for b in S["seg"]:
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel6, 0); sw.append(float(np.linalg.norm(vel6[:3])))
            seg_w.append(sw); jv.append(d.qvel[J["dof"]].copy())
            tq_trace.append(tq.copy()); power_trace.append((tq*d.qvel[J["dof"]]).copy())
            q_trace.append(d.qpos.copy()); qdot_trace.append(d.qvel.copy()); foot_trace.append(np.asarray(ff, dtype=float))
    vcom = P.com_velocity(m, d)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, S["site"], vel6, 0)
    v = np.array(vel6[3:6]); spd = float(np.linalg.norm(v))
    horiz = float(np.hypot(v[0], v[1])); elev = float(np.arctan2(v[2], horiz)); az = float(np.arctan2(v[1], v[0]))
    h_rel = float(d.site_xpos[S["site"]][2])
    E1 = float(d.energy[0] + d.energy[1]); resid = (E1 - E0) - W_act; closure = resid - W_con - W_pas
    fell = float(d.subtree_com[0][2]) < z0 - 0.25
    rep = TrajectoryValidityReport(start_qdot_max=start_qdot, energy_residual=resid, energy_tol=max(1.0, 0.02*abs(W_act)),
                                   actuator_work=W_act, contact_energy=0.0, ke_takeoff=0.5*BALL*spd**2,
                                   nonfoot_contact=nonfoot, took_off=True, peak_grf_bw=peak_grf/(S["M"]*G),
                                   energy_closure=closure, closure_tol=max(5.0, 0.02*(abs(W_act) + abs(W_con)))).evaluate()
    reasons = list(rep.reasons) + (["STANCE_LOST"] if stance_lost else []) + (["FELL"] if fell else []) + (["OVERSPEED"] if overspeed > OVERSPEED_TOL else [])   # a joint past 110% of its rated speed: the motion is not realisable by the declared actuators
    in_cone = (ELEV[0] <= elev <= ELEV[1]) and abs(az) <= AZ
    tot = float(work.sum())
    gw = {}
    for g, wv in zip(S["groups"], work): gw[g] = gw.get(g, 0.0) + float(wv)
    shares = np.array([v_/tot for v_ in gw.values()]) if tot > 0 else np.zeros(1)
    # RELEASE-STATE RECOVERABILITY measures (J22): COM vertical velocity and capture-point offset from the feet centre at release.
    # The speed-only throw dives (COM -2.45 m/s at release) and no controller recovers it; a recoverable throw bounds these.
    com_r = d.subtree_com[0]; w0_ = np.sqrt(G/max(float(com_r[2]), 0.2)); cp_r = com_r[:2] + vcom[:2]/w0_
    fxy = []
    for fg in feet:
        for g_ in fg: fxy.append(d.geom_xpos[g_][:2])
    cp_err_rel = float(np.linalg.norm(cp_r - np.mean(fxy, axis=0))) if fxy else float("nan")
    feet_bw = [float(ff[0]/(S["M"]*G)), float(ff[1]/(S["M"]*G))]          # foot loads at release (a "recoverable" release must still be standing)
    out = dict(unloaded_frac=n_unl/max(nst, 1), overspeed=overspeed, overspeed_by_joint={S["types"][i]: round(float(v_), 2) for i, v_ in enumerate(over_j) if v_ > 1.0}, com_speed=float(np.linalg.norm(vcom)), vcom_z=float(vcom[2]), cp_err_rel=cp_err_rel, feet_bw=feet_bw, speed=spd, v=v.tolist(), elev=elev, az=az, in_cone=bool(in_cone), t_rel=t_rel, h_rel=h_rel,
               range=_range(v, h_rel), valid=not reasons, reasons=reasons, work=gw,
               participation=float(1.0/np.sum(shares**2)) if tot > 0 else 0.0, W_act=W_act, closure=closure,
               peak_grf_bw=peak_grf/(S["M"]*G))
    if record:
        out["trace"] = dict(
            seg_w=np.asarray(seg_w).tolist(), jv=np.asarray(jv).tolist(),
            torque=np.asarray(tq_trace).tolist(), power=np.asarray(power_trace).tolist(),
            q=np.asarray(q_trace).tolist(), qvel=np.asarray(qdot_trace).tolist(),
            foot_load_bw=(np.asarray(foot_trace)/(S["M"]*G)).tolist(), dt=dt,
            release_step=int(nst), release_time=float(t_rel),
        )
    out["final_state"] = (d.qpos.copy(), d.qvel.copy())
    return out


def _range(v, h):
    vx = float(np.hypot(v[0], v[1])); vz = float(v[2])
    tf = (vz + np.sqrt(max(vz*vz + 2*G*max(h, 0.0), 0.0)))/G
    return vx*tf


def penalty(r, target=None, cone=None):
    """Lexicographic: any violation scores >= 100 (worse than every valid throw), GRADED so the search can climb
    toward feasibility; feasible throws score -speed (or the demand-sweep objective).
    (Phase 11 first campaign: a flat 5 m/s penalty let 15-18 m/s stance-breaking flings beat every valid throw.)"""
    viol = 0.0
    elev_range, az_limit = cone if cone is not None else (ELEV, AZ)
    if not r["in_cone"]:
        viol += 20.0*(max(0.0, elev_range[0] - r["elev"], r["elev"] - elev_range[1]) + max(0.0, abs(r["az"]) - az_limit)) + 1.0
    if "STANCE_LOST" in r["reasons"]: viol += 1.0 + 100.0*r.get("unloaded_frac", 1.0)
    viol += 0.0
    if "NONFOOT_CONTACT" in r["reasons"]: viol += 5.0
    if "FELL" in r["reasons"]: viol += 5.0
    if "ENERGY_CLOSURE" in r["reasons"] or "ENERGY_ACCOUNTING_VIOLATION" in r["reasons"]: viol += 2.0
    if viol > 0: return 100.0 + viol - 0.01*r["speed"]
    if target is not None:                     # demand sweep: hit the target speed with least positive work
        return abs(r["speed"] - target)*5.0 + 0.002*sum(r["work"].values())
    return -r["speed"]


def solve(S, seed, iters=150, pop=16, ctrl_fn_factory=None, target=None, caps=None, x0=None, sigma0=0.10):
    nT = len(S["types"]); n = nT*NK + 1
    def obj(z):
        fn = ctrl_fn_factory(z) if ctrl_fn_factory else None
        return penalty(rollout(z, S, ctrl_fn=fn, caps=caps), target)
    es = cma.CMAEvolutionStrategy(np.zeros(n) if x0 is None else x0, sigma0,
                                  {"bounds": [-1, 1], "seed": seed, "verbose": -9, "maxiter": iters, "popsize": pop})
    while not es.stop():
        X = es.ask(); es.tell(X, [obj(z) for z in X])
    return np.asarray(es.result.xbest)


DT_OPT = 0.001   # amended v3: 4 ms optimisation did not transfer (stance/cone failures at 1 ms); 1 ms does (0.5 ms check)


def measure(S, x, ctrl_fn_factory=None, caps=None, dts=(0.001, 0.0005)):
    """Measure a solution at each dt in dts (primary first) and restore the optimisation dt. ctrl_fn_factory
    builds a fresh tracker per dt (it precomputes on the dt grid). Valid only if valid + in-cone at EVERY dt."""
    dt0 = S["m"].opt.timestep; outs = []
    try:
        for dt in dts:
            BI.apply_dt(S, dt); outs.append(rollout(x, S, ctrl_fn=ctrl_fn_factory() if ctrl_fn_factory else None, record=(dt == dts[0]), caps=caps))
    finally:
        BI.apply_dt(S, dt0)
    r = outs[0]; r["dt_check"] = [dict(dt=dt, speed=o["speed"], valid=o["valid"], in_cone=o["in_cone"], elev=o["elev"]) for dt, o in zip(dts, outs)]
    r["valid_all_dt"] = all(o["valid"] and o["in_cone"] for o in outs)
    return r


# ---------------------------------------------------------------- analytic strategy prior (no human data)
def constrained_jacobian(S, q):
    """Palm linear-velocity Jacobian w.r.t. the driven joints with BOTH FEET HELD STATIONARY: the floating-base
    velocity is solved (least squares) from the foot constraints, so leg joints move the palm through the base."""
    m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_forward(m, d)
    jp = np.zeros((3, m.nv)); mujoco.mj_jacSite(m, d, jp, None, S["site"])
    Jf = []
    for nm in [("left_foot", "right_foot")[i] for i in S.get("support", (0, 1))]:      # SUPPORT feet only (a kick's swing foot is free: pinning it zeroed its own direction field)
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, nm)
        a = np.zeros((3, m.nv)); r = np.zeros((3, m.nv)); mujoco.mj_jacSite(m, d, a, r, sid); Jf += [a, r]
    Jf = np.vstack(Jf); dofs = S["jarr"]["dof"]; base = np.arange(6)
    out = np.zeros((3, len(dofs)))
    for i, dof in enumerate(dofs):
        vb = -np.linalg.lstsq(Jf[:, base], Jf[:, dof], rcond=None)[0]    # base motion keeping the feet still
        out[:, i] = jp[:, base] @ vb + jp[:, dof]
    return out


def _cert_ok(c):
    """Static certificate acceptance for reference construction. FRICTION_INFEASIBLE with a clearly positive support margin is a
    solver artefact (2026-10-11: a pose 9 steps from the stance, 5 sole geoms per foot on the floor, COM inside the support, failed
    the friction LP at every mu up to 10 and passed at 1e6); it is accepted when the COM is >= 2 cm inside the support polygon."""
    return c["status"] == "STATIC_FEASIBLE" or (c["status"] == "FRICTION_INFEASIBLE" and float(c.get("support_margin") or 0.0) >= 0.02)


def prepare_support(S, q0_j, placed, cg, iters=300, step=0.005, tol=0.01, lift=0.04):
    """ONE-FOOT SUPPORT PREPARATION (construct, 2026-10-11): move the SUPPORT leg so the COM projection lands over the support
    foot (finite-difference descent, 95% joint limits); after each step the SWING leg is corrected so its foot stays exactly on the
    floor (minimum-norm correction through the foot-height Jacobian), because a lifted swing foot drops out of the static
    certificate's contact set before the weight is over the support foot. Stops at tol or when the certificate fails.
    Derived from the support contract alone: no demonstration of a weight shift."""
    m = S["m"]; J = S["jarr"]; qa = J["qad"]; sup = S["support"]; side = ("left", "right")[sup[0]]
    idx = [i for i, t in enumerate(S["types"]) if S["support_mask"][i]]
    swing = [i for i in (0, 1) if i not in sup]; swing_geoms = set().union(*[S["feet"][i] for i in swing]) if swing else set()
    sw_side = ("left", "right")[swing[0]] if swing else None
    sw_idx = [i for i, t in enumerate(S["types"]) if sw_side and t.startswith(sw_side) and any(k in t for k in ("hip_pitch", "knee", "ankle_pitch", "hip_roll"))]
    lo = np.array([m.jnt_range[S["tmap"][t][0][0]][0] for t in S["types"]]); hi = np.array([m.jnt_range[S["tmap"][t][0][0]][1] for t in S["types"]]); lo, hi = lo + 0.05*(hi - lo), hi - 0.05*(hi - lo)
    fs = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, side + "_foot"); d = mujoco.MjData(m)
    def err(qj):
        d.qpos[:] = placed(qj); mujoco.mj_forward(m, d); return d.subtree_com[0][:2] - d.site_xpos[fs][:2]
    def zsw(qj):
        d.qpos[:] = placed(qj); mujoco.mj_forward(m, d); return (min(P.geom_low(m, d, g) for g in swing_geoms) if swing_geoms else 0.0)
    def keep_swing_down(qj, z_target=0.0):
        for _ in range(6):
            z = zsw(qj) - z_target
            if abs(z) < 1e-4 or not sw_idx: return qj
            Jz = np.zeros(len(sw_idx))
            for k_, i in enumerate(sw_idx):
                qp = qj.copy(); qp[i] += 1e-3; Jz[k_] = (zsw(qp) - z)/1e-3
            if np.linalg.norm(Jz) < 1e-9: return qj
            dq = -Jz*z/float(Jz @ Jz); qj = qj.copy()
            for k_, i in enumerate(sw_idx): qj[i] = float(np.clip(qj[i] + dq[k_], lo[i], hi[i]))
        return qj
    # FLAT FEET: the soles stay parallel to the floor through the ankles (a hip-roll weight shift rolled the support foot onto its
    # edge: 3 sole geoms left within the certificate's 3 mm tolerance, polygon collapsed, LP infeasible)
    fb = {sd: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, sd + "_ankle_roll_link") for sd in ("left", "right")}
    ank = {sd: [i for i, t in enumerate(S["types"]) if t.startswith(sd) and ("ankle_pitch" in t or "ankle_roll" in t)] for sd in ("left", "right")}
    d.qpos[:] = placed(np.array(q0_j, float)); mujoco.mj_forward(m, d); R0 = {sd: d.xmat[fb[sd]].reshape(3, 3)[2, :2].copy() for sd in fb}
    def tilt(qj, sd):
        d.qpos[:] = placed(qj); mujoco.mj_forward(m, d); return d.xmat[fb[sd]].reshape(3, 3)[2, :2] - R0[sd]
    def keep_flat(qj):
        for sd in ("left", "right"):
            ids = ank[sd]
            if len(ids) < 1 or fb[sd] < 0: continue
            for _ in range(4):
                e = tilt(qj, sd)
                if np.linalg.norm(e) < 1e-4: break
                Jt = np.zeros((2, len(ids)))
                for k_, i in enumerate(ids):
                    qp = qj.copy(); qp[i] += 1e-3; Jt[:, k_] = (tilt(qp, sd) - e)/1e-3
                dq = -np.linalg.lstsq(Jt, e, rcond=None)[0]; qj = qj.copy()
                for k_, i in enumerate(ids): qj[i] = float(np.clip(qj[i] + dq[k_], lo[i], hi[i]))
        return qj
    qj = np.array(q0_j, float); last_ok = qj.copy(); path = [qj.copy()]
    for k in range(iters):
        e = err(qj)
        if np.linalg.norm(e) < tol: break
        g = np.zeros(len(qj)); e2 = float(e @ e)
        for i in idx:
            qp = qj.copy(); qp[i] += 1e-3; ep = err(qp); g[i] = (float(ep @ ep) - e2)/1e-3
        if np.linalg.norm(g) < 1e-9: break
        qn = keep_swing_down(keep_flat(np.clip(qj - step*g/np.linalg.norm(g), lo, hi)))
        if k % 5 == 4 and not _cert_ok(P.static_certificate(m, mujoco.MjData(m), placed(qn), coll_geoms=cg)): break
        qj = qn; path.append(qj.copy()); last_ok = qj.copy()
    # LIFT: with the weight over the support foot, the swing foot leaves the floor (lift height, flat sole, certified at each
    # increment): a swing foot still planted and loaded pinned the leg and the wind-up dragged against the floor
    qj = last_ok.copy()
    for z_t in np.linspace(0.005, lift, max(1, int(round(lift/0.005)))):
        qn = keep_flat(keep_swing_down(qj, z_t))
        if not _cert_ok(P.static_certificate(m, mujoco.MjData(m), placed(qn), coll_geoms=cg)): break
        qj = qn; path.append(qj.copy()); last_ok = qj.copy()
    return last_ok, np.array(path), float(np.linalg.norm(err(last_ok)))


def windup_reference(S, step=0.02, n_max=800, leg_cap=0.30, t_w=0.40, t_ref=0.55, fps=100, capacity_mode="power", seed=0, direction=None):
    """Wind-up -> release reference computed from the robot specification alone (no human data).
    Direction field: capacity-weighted constrained-Jacobian ascent of palm displacement along the centre of the
    target cone, u = c * (Jc^T dvec), c_j = tau_max_j * omega_max_j / 4 (peak actuator power).
    WIND-UP: integrate -u over trunk and arm joints from the settled pose (joint limits at 95% of range, legs
    held in the certified stance) for as long as the pose stays statically certifiable. RELEASE PATH: integrate +u from the
    wind-up until palm progress along dvec stops (limits saturate or the gradient vanishes).
    Timing (t_w, t_ref) is nominal; the optimiser may stretch it (tscale)."""
    m = S["m"]; J = S["jarr"]; q_s = np.array(S["q_settled"]); qa = J["qad"]
    lo = np.array([m.jnt_range[S["tmap"][t][0][0]][0] for t in S["types"]]); hi = np.array([m.jnt_range[S["tmap"][t][0][0]][1] for t in S["types"]])
    lo, hi = lo + 0.05*(hi - lo), hi - 0.05*(hi - lo)
    legs = np.array([g == "legs" for g in S["groups"]]) if S.get("support_mask") is None else np.asarray(S["support_mask"])   # support legs are capped; a swing leg (kick) winds up like an arm
    if capacity_mode == "power":
        c = J["tau"]*J["om"]/4.0
    elif capacity_mode in ("geometry", "equal"):
        c = np.ones(len(J["tau"]))
    elif capacity_mode == "random":
        rng = np.random.default_rng(seed); c = rng.uniform(0.5, 1.5, len(J["tau"]))
    else:
        raise ValueError(f"unknown capacity_mode={capacity_mode!r}")
    if direction is None:
        elev, az = 0.5*(ELEV[0] + ELEV[1]), 0.0
    else:
        elev, az = map(float, direction)
    dvec = np.array([np.cos(elev)*np.cos(az), np.cos(elev)*np.sin(az), np.sin(elev)])
    cg = [g for g in P.PlantContractVerifier(m, mujoco.MjData(m), P.PlantContract(S["mjcf"], "torque")).collision_geoms() if g in S.get("robot_geoms", set(range(m.ngeom)))]   # ROBOT geoms only: in a world scene a tee or a ball standing on the floor is not a support contact
    def direction(q):
        u = c*(constrained_jacobian(S, q).T @ dvec); return u/max(np.linalg.norm(u), 1e-12)
    def placed(qj):
        q = q_s.copy(); q[qa] = qj; d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_forward(m, d)
        q[2] -= min(P.geom_low(m, d, g) for g in (set().union(*[S["feet"][i] for i in S.get("support", (0, 1))]) if S.get("support") else S["foot_geoms"])); return q   # placed on the SUPPORT feet (a lifted swing foot must not float the support foot off the floor)
    q0 = q_s[qa].copy(); prep = None
    if len(S.get("support", (0, 1))) == 1:
        q0, prep, prep_err = prepare_support(S, q0, placed, cg)         # weight over the support foot first; the wind-up and the caps are measured from the prepared stance
    qj = q0.copy(); wind = [qj.copy()]
    for k in range(n_max):
        # wind-up moves trunk and arms only (legs stay in the certified stance), so the foot geometry is
        # exact and the static certificate tests the real COM shift of the cocked upper body
        nxt = np.clip(qj - step*direction(placed(qj)), lo, hi)
        nxt = np.where(legs, q0, nxt)        # support legs frozen in the wind-up (two feet: the certified stance; one foot: the PREPARED stance with the weight over the support foot; a moving support leg rolled the foot onto its edge)
        if np.linalg.norm(nxt - qj) < 1e-4: break
        if k % 5 == 4 and not _cert_ok(P.static_certificate(m, mujoco.MjData(m), placed(nxt), coll_geoms=cg)): break
        qj = nxt; wind.append(qj.copy())
    # RELEASE: integrate +u from the STANCE (not from the wind-up) for as many steps as the wind-up took, so the
    # reference sweeps the chain through stance: wind-up -> stance -> release (symmetric excursion budget)
    qw = qj.copy(); qj = q0.copy(); fwd = [qj.copy()]; prog = 0.0
    for k in range(len(wind) - 1):
        u = direction(placed(qj)); nxt = np.clip(qj + step*u, lo, hi)
        cap_ = leg_cap if len(S.get("support", (0, 1))) == 2 else min(leg_cap, 0.15)   # one-foot support: the support leg moves less on the release path (no flat-foot projection there)
        nxt = np.where(legs, np.clip(nxt, q0 - cap_, q0 + cap_), nxt)
        gain = float(dvec @ (constrained_jacobian(S, placed(qj)) @ (nxt - qj)))
        if gain <= 1e-5: break
        prog += gain; qj = nxt; fwd.append(qj.copy())
    wind = np.array(wind); path = np.vstack([wind[::-1], np.array(fwd)[1:]]) if len(fwd) > 1 else wind[::-1]
    # CAPACITY-DERIVED TIMING (construct, 2026-10-11): each segment lasts at least the bang-bang traversal time of its most
    # demanding joint, t = 2 sqrt(dq I_eff / tau_max) (accelerate-decelerate from rest), x 1.2 margin. The arm's segments sit
    # below the Phase 11 constants (0.35 s wind-up, 0.15 s release: unchanged); a leg's do not (the hip saturated at 88 N m
    # chasing a 140 deg swing in 0.15 s and the tracker lagged 80 deg).
    ieff = np.array([1.0/max(float(S["Minv_key"][int(dof), int(dof)]), 1e-9) for dof in J["dof"]])
    def t_min(P_):
        if len(P_) < 2: return 0.0
        dq = np.abs(np.asarray(P_)[-1] - np.asarray(P_)[0]); return float(np.max(2.0*np.sqrt(dq*ieff/np.maximum(J["tau"], 1e-9))))
    t_w_cap = 0.05 + 1.2*t_min(wind); t_rel_cap = 1.2*t_min(path)
    if t_w_cap > t_w or t_rel_cap > (t_ref - t_w):
        t_w = max(t_w, t_w_cap); t_ref = max(t_ref, t_w + t_rel_cap)
        if t_ref > HOR - 0.05: sc_ = (HOR - 0.05)/t_ref; t_w *= sc_; t_ref *= sc_      # fits the rollout horizon; the search's tscale may stretch further
    prep_qpos = None; t_prep_end = 0.05
    if prep is not None and len(prep) > 1:
        prep_qpos = placed(prep[-1]); wind = np.vstack([prep[:-1], wind])   # reference: stance -> weight shift -> wind-up -> release (the release path returns through the PREPARED stance, not the two-foot one)
        L_ = np.linalg.norm(np.diff(wind, axis=0), axis=1); t_prep_end = 0.05 + (t_w - 0.05)*float(L_[:len(prep) - 1].sum()/max(L_.sum(), 1e-12))   # arc fraction of the wind polyline spent on the preparation
    n = int(HOR*fps) + 1; t = np.arange(n)/fps; Q = np.repeat(q_s[None], n, 0)
    def along(P_, w):                          # position along a polyline by arc fraction w
        if len(P_) == 1: return P_[0]
        seg = np.linalg.norm(np.diff(P_, axis=0), axis=1); cs = np.concatenate([[0], np.cumsum(seg)]); L = w*cs[-1]
        i = min(int(np.searchsorted(cs, L, side="right")) - 1, len(seg) - 1); f = (L - cs[i])/max(seg[i], 1e-12)
        return P_[i] + f*(P_[i+1] - P_[i])
    for i, ti in enumerate(t):
        if ti <= 0.05: qq = q0
        elif ti <= t_w: qq = along(wind, (ti - 0.05)/(t_w - 0.05))
        elif ti <= t_ref: qq = along(path, (ti - t_w)/(t_ref - t_w))
        else: qq = path[-1]
        Q[i, qa] = qq
    return dict(key="throw_windup_prior", start=q_s, Q=Q, fps=fps, takeoff_ref=t_ref, horizon=HOR, nk=NK,
                windup=qw.tolist(), release=path[-1].tolist(), windup_steps=len(wind) - 1, path_steps=len(path) - 1,
                palm_progress=prog, prep_steps=(0 if prep is None else len(prep) - 1), prep_err=(None if prep is None else prep_err), t_w=float(t_w), t_ref=float(t_ref), t_min=dict(wind=t_min(wind), release=t_min(path)), prep_qpos=(None if prep_qpos is None else prep_qpos.tolist()), t_prep_end=float(t_prep_end))


def tracker(S, ref, omega, ts):
    import mocap_baseline as MB
    return MB.Tracker(S, ref, omega=omega, tscale=ts)


def scaled_ref(ref, amp):
    """Scale the reference's joint excursions about the stance by amp (strategy kept, magnitude searched)."""
    Q = ref["start"][None] + amp*(ref["Q"] - ref["start"][None])
    return dict(ref, Q=Q)


def solve_prior(S, ref, seed, iters=150, pop=16, omegas=(20.0, 40.0, 60.0), sigma0=0.10):
    """APRIOR arm: track the analytic reference and refine freely. Decision vector = knots + release + tscale +
    amplitude. The prior supplies DIRECTION and COORDINATION; the excursion magnitude that the stance can survive
    dynamically is searched (amp in [0.2, 1.0]). Tracking gain chosen from `omegas` by the prior's own
    zero-refinement outcome over an amplitude grid (extra rollouts counted in the report)."""
    nT = len(S["types"]); n = nT*NK + 3
    def unpack(z): return np.append(z[:nT*NK], z[nT*NK]), 0.87 + 0.3*float(z[nT*NK+1]), 0.6 + 0.4*float(z[-1])
    best = None
    for om in omegas:
        for za in (-1.0, -0.5, 0.0, 0.5, 1.0):
            z0 = np.zeros(n); z0[nT*NK] = 2*(ref["takeoff_ref"] - T_REL_MIN)/(HOR - T_REL_MIN) - 1; z0[-1] = za
            x, ts, amp = unpack(z0); r = rollout(x, S, ctrl_fn=tracker(S, scaled_ref(ref, amp), om, ts))
            sc = penalty(r)
            if best is None or sc < best[0]: best = (sc, om, z0)
    _, om, z0 = best
    def obj(z):
        x, ts, amp = unpack(z); return penalty(rollout(x, S, ctrl_fn=tracker(S, scaled_ref(ref, amp), om, ts)))
    es = cma.CMAEvolutionStrategy(z0, sigma0, {"bounds": [-1, 1], "seed": seed, "verbose": -9, "maxiter": iters, "popsize": pop})
    while not es.stop():
        X = es.ask(); es.tell(X, [obj(z) for z in X])
    x, ts, amp = unpack(np.asarray(es.result.xbest))
    return x, om, ts, amp
