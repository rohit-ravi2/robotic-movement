"""M2_throw_target (2026-10-18): THROW AT A TARGET on the stock G1, with a REAL free ball.

construction = body + objective + support contract -> scaffold -> search -> gates (the construct.py procedure, unchanged in kind):
  body      stock Menagerie G1 MJX (scene_mjx.xml), torque motors at the DOCUMENTED rating table RATING (see ratings()), stock
            joint ranges, masses and speeds; first-order actuator lag LAG (10 ms), torque-speed droop, back-EMF brake.
  objective landing-point error of a 0.145 kg, r 36.5 mm ball on the floor (target circle 1.0 m: hit = error <= 0.5 m).
  support   both feet loaded (stance), no slip, no fall, through release AND a follow-through window FT after release.
  scaffold  the construct wind-up -> release reference (throw.windup_reference) for the release DIRECTION obtained by ballistic
            inversion: target -> minimum-speed (elev, speed) from the reference's own release point; azimuth = bearing to target.
  search    CMA-ES over torque knots (17 joints x 10) + release time + amplitude + time scale (EffectTask parameterisation),
            objective = worse-of-two landing errors at dt 1 ms and 0.5 ms, any gate violation ranks below every valid throw.
  gates     the throw's (envelope, overspeed 1.1x rated, energy closure, stance, no non-foot contact, no fall) + foot slip
            <= SLIP_MAX + follow-through window; fresh plants at 1 ms AND 0.5 ms; >= 4 perturbations (start state + parameters).
BALL: a free body (freejoint) held by a WELD equality to the wrist link at the palm site (labelled attachment: no grip-force
limit, as the merged-mass construction; the weld is stiff, solref 0.002). Release = the weld is deactivated at the searched
release step; the ball then flies as a free body in the full model through the follow-through window, and its exact MuJoCo
state is carried into a ball+floor twin model (same dt, same gravity, vacuum by default) until its first floor contact.
Robot geoms have contype 0 in the MJX model, so the released ball cannot collide with the robot (stated).
usage: throw_target.py search <target_id> <seed> [iters] [pop] | ceiling <seed> [iters] [pop] | robust <ledger.json> | predict
"""
import sys, os, json, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco, yaml, cma
import branch_i as BI, throw as T, plant as P
from validity import TrajectoryValidityReport
from engine import core

ROOT = core.ROOT; OUT = f"{ROOT}/results/movements/M2_throw_target"
MEN = os.environ.get("MENAGERIE_DIR", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "models", "mujoco_menagerie")); BASE_SCENE = f"{MEN}/unitree_g1/scene_mjx.xml"
G = 9.80665
BALL_M, BALL_R = 0.145, 0.0365                      # regulation baseball mass and radius
LAG = float(os.environ.get("M2_LAG", 0.010))          # first-order actuator lag (s); project convention 40 ms, this job 10 ms
FT = 0.30                                             # follow-through window after release in which the support gates still apply
SLIP_MAX = 0.02                                       # m: max horizontal travel of either support foot over the rollout
HIT_R = 0.5                                           # m: target circle of 1.0 m diameter
VENDOR_MAX = 120.0                                    # unitree.com/g1: G1 EDU knee max torque 120 N m (largest joint motor; bounds every joint)
TARGETS = {f"d{d}_{tag}": dict(dist=float(d), az=float(np.radians(a))) for d in (5, 10, 15) for tag, a in (("straight", 0.0), ("left20", 20.0))}
DTS = (0.001, 0.0005)
SELF_GATE = os.environ.get("M2_SELF_GATE", "1") == "1"   # R1 (2026-10-18): robot self-contact gate; set 0 only to replay the round-0 gates
SELF_F = 1.0                                          # N: any robot-robot contact above the 1 N foot-loading threshold is a violation


# ------------------------------------------------------------------ ratings (documented, no hardware change)
def ratings():
    """Per-joint torque ceiling = min(MJCF peak actuatorfrcrange, Unitree URDF effort (spec tau_max_continuous), vendor published
    joint maximum 120 N m). Binds: knee 139 -> 120; ankle pitch/roll, waist roll/pitch 50 -> 35; hip roll 139 -> 88."""
    sp = yaml.safe_load(open(f"{ROOT}/spec/g1.yaml"))
    return {jn: min(float(r["tau_max_peak"]["value"]), float(r["tau_max_continuous"]["value"]), VENDOR_MAX) for jn, r in sp["joints"].items()}


# ------------------------------------------------------------------ scene: G1 + free ball held by an ACTIVE weld at the palm site
def scene_path():
    os.makedirs(OUT, exist_ok=True); p = f"{OUT}/scene_g1_freeball.xml"
    if os.path.exists(p): return p
    spec = mujoco.MjSpec.from_file(BASE_SCENE); spec.meshdir = f"{MEN}/unitree_g1/assets"
    m0 = spec.compile(); d0 = mujoco.MjData(m0)
    kid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_KEY, "knees_bent"); d0.qpos[:] = m0.key_qpos[kid]; mujoco.mj_forward(m0, d0)
    sid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_SITE, "right_palm"); hb = int(m0.site_bodyid[sid]); hand = mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_BODY, hb)
    p_ball = d0.site_xpos[sid].copy(); rel = np.array(m0.site_pos[sid], float)              # ball centre at the palm site (hand frame)
    b = spec.worldbody.add_body(name="ball", pos=p_ball.tolist()); b.add_freejoint(name="ball_free")
    g = b.add_geom(name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_R, 0, 0], mass=BALL_M, rgba=[0.95, 0.95, 0.9, 1])
    g.solref = [0.002, 1.0]; g.solimp = [0.99, 0.999, 0.001, 0.5, 2.0]
    e = spec.add_equality(name="grip_ball", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY, name1=hand, name2="ball")
    e.active = True; e.solref = [0.002, 1.0]; e.solimp = [0.99, 0.999, 0.001, 0.5, 2.0]
    q_hand = d0.xquat[hb]; qi = np.array([q_hand[0], -q_hand[1], -q_hand[2], -q_hand[3]]); qr = np.zeros(4); mujoco.mju_mulQuat(qr, qi, np.array([1.0, 0, 0, 0]))
    e.data = np.concatenate([[0, 0, 0], rel, qr, [1.0]]).tolist()
    for k in spec.keys:
        k.qpos = list(k.qpos) + p_ball.tolist() + [1, 0, 0, 0]
        if len(k.qvel): k.qvel = list(k.qvel) + [0.0]*6
    spec.compile(); open(p, "w").write(spec.to_xml()); return p


def twin_model(dt, density=0.0, viscosity=0.0):
    """Ball + floor plane only, for the remainder of the flight (identical physics for the ball: the robot cannot touch it)."""
    xml = f"""<mujoco><option timestep="{dt}" density="{density}" viscosity="{viscosity}" integrator="Euler"/>
      <worldbody><geom name="floor" type="plane" size="0 0 0.01"/>
      <body name="ball"><freejoint/><geom type="sphere" size="{BALL_R}" mass="{BALL_M}" solref="0.002 1" solimp="0.99 0.999 0.001 0.5 2" {'fluidshape="ellipsoid"' if density > 0 else ''}/></body></worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


_UID = [0]
def plant_ref():
    """SCAFFOLD plant: the bare G1 with the ball's mass merged at the palm (statically equivalent to the welded ball; the static
    certificate that bounds the wind-up does not model the weld, so the free-ball scene cannot be certified), rated torques."""
    S = T.setup("G1", start=np.array(T.setup("G1")["q0"]) + 1e-9*12345)
    _rate(S); BI.apply_dt(S, DTS[0]); BI.settle(S); return S


def _rate(S):
    m = S["m"]; rt = ratings()
    for t in S["types"]:
        jid, dof, qad, tau, om, sgn, aid = S["tmap"][t][0]; S["tmap"][t] = [(jid, dof, qad, min(tau, rt.get(t, tau)), om, sgn, aid)]
    S["jarr"] = dict(S["jarr"]); S["jarr"]["tau"] = np.array([S["tmap"][t][0][3] for t in S["types"]], float)
    held = []
    for (a, dof, qad, tau, kp, kv) in S["held"]:
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a][0])); held.append((a, dof, qad, min(tau, rt.get(jn, tau)), kp, kv))
    S["held"] = held


def plant(dt, pert=None):
    """A FRESH plant (uncached copy) at dt, rated torques applied before the settle; pert = perturbation dict (mass scale per body,
    friction scale, ball mass scale, lag scale, start noise) for the robustness gate."""
    _UID[0] += 1
    S0 = T.setup("G1", mjcf=scene_path(), ball=False, mass=0.0)
    S = T.setup("G1", mjcf=scene_path(), ball=False, mass=0.0, start=np.array(S0["q0"]) + 1e-9*(_UID[0] + 1))
    m = S["m"]; _rate(S)          # rated torques for driven joints (tmap + jarr) and held joints (servo force limits)
    S["ball_body"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball"); S["ball_eq"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "grip_ball")
    S["ball_qadr"] = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]); S["ball_dadr"] = int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")])
    S["lag"] = LAG; S["pert"] = pert
    if pert:
        rng = np.random.default_rng(int(pert["seed"]))
        for bdy in S["robot_bodies"]:
            s = float(np.clip(1.0 + pert["mass_sd"]*rng.standard_normal(), 0.85, 1.15)); m.body_mass[bdy] *= s; m.body_inertia[bdy] *= s
        m.body_mass[S["ball_body"]] *= float(rng.uniform(1 - pert["ball_mass"], 1 + pert["ball_mass"]))
        fs = float(rng.uniform(1 - pert["friction"], 1 + pert["friction"])); m.geom_friction[:, 0] *= fs; m.pair_friction[:, :2] *= fs
        S["lag"] = LAG*float(rng.uniform(1 - pert["lag"], 1 + pert["lag"])); S["pert_draw"] = dict(friction_scale=fs, lag=S["lag"])
        mujoco.mj_setConst(m, mujoco.MjData(m))   # R1 fix: recompute mass-dependent constants after the in-place edits (without it the
                                                  # perturbed ball fell at 9.81 x its mass draw; physics verifier, 94ec882)
    BI.apply_dt(S, dt); BI.settle(S)
    if pert:
        rng2 = np.random.default_rng(1000 + int(pert["seed"])); q = S["q_settled"].copy(); v = S["v_settled"].copy()
        q[S["jarr"]["qad"]] += pert["q_sd"]*rng2.standard_normal(len(S["types"])); v[S["jarr"]["dof"]] += pert["v_sd"]*rng2.standard_normal(len(S["types"]))
        S["q_settled"], S["v_settled"] = q, v
    return S


# ------------------------------------------------------------------ ballistic inversion (vacuum) from a release point
def invert(p_rel, target_xy):
    """Minimum-speed ballistic solution from release point p_rel to the floor point target (z = BALL_R at landing).
    Returns speed, elevation (rad) and azimuth (rad, world). The minimum-speed elevation makes the range first-order insensitive
    to elevation error."""
    dx = np.asarray(target_xy, float) - np.asarray(p_rel[:2], float); R = float(np.linalg.norm(dx)); h = float(p_rel[2] - BALL_R)
    v2 = G*(np.sqrt(R*R + h*h) - h); th = float(np.arctan2(v2, G*R)); return dict(speed=float(np.sqrt(v2)), elev=th, az=float(np.arctan2(dx[1], dx[0])), R=R, h=h)


def target_xy(S, tgt):
    """Target on the floor at distance d and bearing az from the stance centre (mid-feet), body heading = world +x."""
    m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d)
    c = np.mean([d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)][:2] for b in BI.ROBOTS["G1"]["foot_bodies"]], axis=0)
    return c + tgt["dist"]*np.array([np.cos(tgt["az"]), np.sin(tgt["az"])]), c


def release_point(S, ref):
    """Palm-site position at the reference's release pose (the scaffold's own release point; no search)."""
    m = S["m"]; d = mujoco.MjData(m); q = np.array(S["q_settled"]); q[S["jarr"]["qad"]] = ref["release"]; d.qpos[:] = q; mujoco.mj_forward(m, d)
    return d.site_xpos[S["site"]].copy()


# ------------------------------------------------------------------ rollout (throw.rollout + free ball + follow-through + slip)
def rollout(x, S, ctrl_fn=None, record=False, density=0.0, twin_T=4.0):
    m = S["m"]; d = mujoco.MjData(m); BI._set_capacity_limits(m, S, None); m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    d.qpos[:] = S["q_settled"]; d.qvel[:] = S["v_settled"]; d.eq_active[S["ball_eq"]] = 1; mujoco.mj_forward(m, d)
    E0 = float(d.energy[0] + d.energy[1]); start_qdot = float(np.max(np.abs(d.qvel[6:S["ball_dadr"]])))
    nT = len(S["types"]); kn = np.asarray(x[:-1]).reshape(nT, -1); NKl = kn.shape[1]
    t_rel = T.release_time(x[-1], T.HOR); dt = m.opt.timestep; nrel = int(round(t_rel/dt)); nst = nrel + int(round(FT/dt))
    J = S["jarr"]; tau_v = J["tau"]; om_v = J["om"]
    for t, (jid, dof, qad, tau, om, sgn, aid) in zip(S["types"], (S["tmap"][t][0] for t in S["types"])):
        m.actuator_ctrlrange[aid][:] = [-tau, tau]; m.actuator_forcerange[aid][:] = [-tau, tau]; m.jnt_actfrcrange[jid][:] = [-tau, tau]
    a_hold = np.array([S["a_hold"][t] for t in S["types"]]); act = a_hold.copy()
    alpha = 1.0 - np.exp(-dt/S["lag"]); knot_dt = T.HOR/NKl
    for (a_, dof, qad, tau, kp, kd) in S["held"]: d.ctrl[a_] = S["q_settled"][qad]
    fl = S["floor"]; feet = S["feet"]; vel6 = np.zeros(6); z0 = float(d.subtree_com[0][2])
    fsite = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in ("left_foot", "right_foot")]; f0 = [d.site_xpos[s][:2].copy() for s in fsite]; slip = 0.0
    W_act = W_con = W_pas = 0.0; work = np.zeros(nT); nonfoot = stance_lost = False; peak_grf = 0.0; run_unl = [0.0, 0.0]; n_unl = 0
    overspeed = 0.0; self_peak = 0.0; over_j = np.zeros(nT); rel = None; qs = []; bs = []; tqs = []; flb = []
    for s in range(nst):
        tt = s*dt; k = min(int(tt/knot_dt + 1e-9), NKl - 1); k2 = min(k + 1, NKl - 1)
        f = min(1.0, (tt - k*knot_dt)/knot_dt) if k < NKl - 1 else 0.0
        cmd = a_hold + kn[:, k] + (kn[:, k2] - kn[:, k])*f
        if ctrl_fn is not None: cmd = cmd + ctrl_fn(min(tt, T.HOR), d, S)
        act += alpha*(cmd - act)
        w = d.qvel[J["dof"]]; frac = np.maximum(0.0, 1.0 - np.abs(w)/om_v)
        hi = np.where(w > 0, tau_v*frac, tau_v); lo = np.where(w < 0, tau_v*frac, tau_v)
        tq = np.clip(act*tau_v, -lo, hi); rw = np.abs(w)/om_v; overspeed = max(overspeed, float(np.max(rw))); over_j = np.maximum(over_j, rw)
        d.ctrl[J["aid"]] = tq; BI.backemf_brake(m, d, S)
        if s == nrel:                                            # RELEASE: the weld is switched off; the ball is a free body from here
            d.eq_active[S["ball_eq"]] = 0
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, S["site"], vel6, 0); vp = vel6[3:6].copy()
            rel = dict(t=tt, palm_v=vp.tolist(), ball_p=d.qpos[S["ball_qadr"]:S["ball_qadr"] + 3].tolist(), ball_v=d.qvel[S["ball_dadr"]:S["ball_dadr"] + 3].tolist(), palm_p=d.site_xpos[S["site"]].tolist())
        qv0 = d.qvel.copy(); mujoco.mj_step(m, d); qm = 0.5*(qv0 + d.qvel)
        W_act += float(np.dot(d.actuator_force, qm[S["act_dof"]]))*dt
        W_con += float(np.dot(d.qfrc_constraint, qm))*dt; W_pas += float(np.dot(d.qfrc_passive, qm))*dt + float(np.dot(d.qfrc_applied, qm))*dt
        pw = tq*d.qvel[J["dof"]]; work += np.where(pw > 0, pw, 0.0)*dt
        ff = [0.0, 0.0]; fz = 0.0
        for ci in range(d.ncon):
            c = d.contact[ci]
            if fl not in (c.geom1, c.geom2):
                if SELF_GATE and c.geom1 in S["robot_geoms"] and c.geom2 in S["robot_geoms"]:      # R1: robot-robot (self) contact
                    fv = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fv); self_peak = max(self_peak, float(fv[0]))
                continue
            o = c.geom2 if c.geom1 == fl else c.geom1
            fv = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fv)
            if o in feet[0]: ff[0] += fv[0]; fz += fv[0]
            elif o in feet[1]: ff[1] += fv[0]; fz += fv[0]
            elif o in S["robot_geoms"]: nonfoot = True
        for i_f in (0, 1):
            if s > 5 and ff[i_f] < 1.0: run_unl[i_f] += dt
            if run_unl[i_f] >= BI.FLIGHT_CONFIRM - 1e-12: stance_lost = True
        if s > 5 and min(ff) < 1.0: n_unl += 1
        for i_f in (0, 1): slip = max(slip, float(np.linalg.norm(d.site_xpos[fsite[i_f]][:2] - f0[i_f])))
        peak_grf = max(peak_grf, fz)
        if record: qs.append(d.qpos.copy()); tqs.append(tq.copy()); flb.append(np.asarray(ff)/(S["M"]*G))
    E1 = float(d.energy[0] + d.energy[1]); resid = (E1 - E0) - W_act; closure = resid - W_con - W_pas
    fell = float(d.subtree_com[0][2]) < z0 - 0.25
    bq = d.qpos[S["ball_qadr"]:S["ball_qadr"] + 7].copy(); bv = d.qvel[S["ball_dadr"]:S["ball_dadr"] + 6].copy()
    sep = float(np.linalg.norm(bq[:3] - d.site_xpos[S["site"]]))
    # flight remainder in the ball+floor twin (exact MuJoCo state handed over)
    tm = twin_model(dt, density=density, viscosity=(1.8e-5 if density > 0 else 0.0)); td = mujoco.MjData(tm); td.qpos[:] = bq; td.qvel[:] = bv; mujoco.mj_forward(tm, td)
    land = None; traj = [bq[:3].copy()] if record else None; ntw = int(twin_T/dt)
    already = d.ncon > 0 and any(S["ball_body"] in (int(m.geom_bodyid[d.contact[i].geom1]), int(m.geom_bodyid[d.contact[i].geom2])) for i in range(d.ncon))
    if already: land = bq[:3].copy()
    else:
        for i in range(ntw):
            mujoco.mj_step(tm, td)
            if record and i % 10 == 0: traj.append(td.qpos[:3].copy())
            if td.ncon > 0: land = td.qpos[:3].copy(); break
    rep = TrajectoryValidityReport(start_qdot_max=start_qdot, energy_residual=resid, energy_tol=max(1.0, 0.02*abs(W_act)), actuator_work=W_act, contact_energy=0.0,
                                   ke_takeoff=0.0, nonfoot_contact=nonfoot, took_off=True, peak_grf_bw=peak_grf/(S["M"]*G), energy_closure=closure,
                                   closure_tol=max(5.0, 0.02*(abs(W_act) + abs(W_con)))).evaluate()
    reasons = list(rep.reasons) + (["STANCE_LOST"] if stance_lost else []) + (["FELL"] if fell else []) + (["OVERSPEED"] if overspeed > T.OVERSPEED_TOL else []) + (["SLIP"] if slip > SLIP_MAX else []) + ([] if land is not None else ["NO_LANDING"]) + (["SELF_CONTACT"] if self_peak > SELF_F else [])
    viol = 0.0
    if "SELF_CONTACT" in reasons: viol += 5.0 + 0.01*self_peak
    if "STANCE_LOST" in reasons: viol += 1.0 + 100.0*n_unl/max(nst, 1)
    if "NONFOOT_CONTACT" in reasons: viol += 5.0
    if "FELL" in reasons: viol += 5.0
    if "ENERGY_CLOSURE" in reasons or "ENERGY_ACCOUNTING_VIOLATION" in reasons: viol += 2.0
    if "OVERSPEED" in reasons: viol += 1.0 + 10.0*max(0.0, overspeed - T.OVERSPEED_TOL)
    if "SLIP" in reasons: viol += 1.0 + 50.0*(slip - SLIP_MAX)
    if "NO_LANDING" in reasons: viol += 10.0
    if "START_NOT_AT_REST" in reasons: viol += 1.0
    bvr = np.array(rel["ball_v"]) if rel else np.zeros(3); spd = float(np.linalg.norm(bvr))
    out = dict(valid=not reasons, reasons=reasons, violation=float(viol), land=(None if land is None else land.tolist()), release=rel, speed=spd,
               elev=float(np.degrees(np.arctan2(bvr[2], np.hypot(bvr[0], bvr[1])))), az=float(np.degrees(np.arctan2(bvr[1], bvr[0]))),
               t_rel=t_rel, slip=slip, overspeed=overspeed, overspeed_by_joint={S["types"][i]: round(float(v_), 2) for i, v_ in enumerate(over_j) if v_ > 1.0},
               sep_after_ft=sep, work={g: float(sum(w_ for g_, w_ in zip(S["groups"], work) if g_ == g)) for g in set(S["groups"])}, W_act=W_act, closure=closure,
               min_feet_bw_end=float(min(ff)/(S["M"]*G)), unloaded_frac=n_unl/max(nst, 1), self_contact_peak=self_peak)
    if record: out["trace"] = dict(q=np.asarray(qs), torque=np.asarray(tqs), foot_load_bw=np.asarray(flb), dt=dt, release_step=nrel, ball_traj=np.asarray(traj))
    return out


_RS = []
def _REF():
    if not _RS: _RS.append(plant_ref())
    return _RS[0]


# ------------------------------------------------------------------ construction for a target
class Construction:
    """One target: scaffold (reference for the inverted direction), parameterisation, evaluation on a plant."""
    def __init__(self, tgt, S_ball, S_ref=None):
        S_ref = S_ref or _REF()
        self.tgt = tgt; self.xy, self.c = target_xy(S_ball, tgt)
        ref0 = T.windup_reference(S_ref, direction=(np.radians(22.5), tgt["az"]))     # pass 1: the release point of the reference toward the target bearing
        p0 = release_point(S_ball, ref0); inv = invert(p0, self.xy)
        self.ref = T.windup_reference(S_ref, direction=(inv["elev"], inv["az"]))        # pass 2: the scaffold for the inverted direction
        p1 = release_point(S_ball, self.ref); self.inv = invert(p1, self.xy); self.p_rel_ref = p1.tolist(); self.inv_pass1 = inv
        self.nT = len(S_ref["types"]); self.ref_steps = dict(windup=self.ref["windup_steps"], path=self.ref["path_steps"], t_w=self.ref["t_w"], t_ref=self.ref["t_ref"])
    def n(self): return self.nT*T.NK + 3
    def unpack(self, z):
        z = np.asarray(z); n = self.nT*T.NK + 1; amp = 0.2 + 0.8*(float(z[n]) + 1)/2; ts = 0.57 + (1.17 - 0.57)*(float(z[n + 1]) + 1)/2
        return z[:n], amp, ts
    def z0(self):
        z = np.zeros(self.n()); z[self.nT*T.NK] = 2*(self.ref["takeoff_ref"] - T.T_REL_MIN)/(T.HOR - T.T_REL_MIN) - 1; return z
    def run(self, S, z, record=False, density=0.0):
        x, amp, ts = self.unpack(z); fn = T.tracker(S, T.scaled_ref(self.ref, amp), 20.0, ts)
        r = rollout(x, S, ctrl_fn=fn, record=record, density=density)
        r["err"] = (float(np.linalg.norm(np.array(r["land"][:2]) - self.xy)) if r["land"] is not None else 99.0); r["amp"] = amp; r["tscale"] = ts
        if getattr(self, "cone", None):            # CEILING ONLY: the Phase 11 throw cone about the target bearing (elev 0-45 deg, az +-20 deg)
            (e_lo, e_hi), a_tol = self.cone; da = (r["az"] - np.degrees(self.tgt["az"]) + 180.0) % 360.0 - 180.0
            ex = max(0.0, e_lo - r["elev"], r["elev"] - e_hi) + max(0.0, abs(da) - a_tol)
            if ex > 0: r["reasons"] = r["reasons"] + ["OUT_OF_CONE"]; r["valid"] = False; r["violation"] += 1.0 + 20.0*np.radians(ex)
        return r


def score(r, objective="error"):
    if not r["valid"]: return 100.0 + max(r["violation"], 1e-3) + (0.01*min(r["err"], 50.0) if objective == "error" else -0.01*r["speed"])
    return r["err"] if objective == "error" else -r["speed"]


def _soft_agg(rs, plants, objective="error"):
    """R2 SOFT robust aggregation (general; same for every target). Nominal plants are HARD (any gate violation ranks below every
    valid nominal); perturbed in-loop draws are SOFT: a valid draw contributes its landing error, an invalid draw 1.0 + its gate
    violation (capped at 5) + 0.1 x error. f = max(worst nominal error, mean over draws). Robustness is still judged only by the
    disjoint 16-draw rule (check_r1); this changes the search landscape, not any gate."""
    nom = [r for r, S_ in zip(rs, plants) if not S_.get("pert")]; per = [r for r, S_ in zip(rs, plants) if S_.get("pert")]
    sn = [score(r, objective) for r in nom]
    if any(si >= 100 for si in sn): return 100.0 + sum(si - 100.0 for si in sn if si >= 100)
    sp = [(r["err"] if r["valid"] else 1.0 + min(r["violation"], 5.0) + 0.1*min(r["err"], 10.0)) for r in per]
    return max(max(sn), float(np.mean(sp)) if sp else 0.0)


def search(C, seed, iters, pop, objective="error", log=None, sigma=0.10, z0=None, loop_dts=(DTS[0],), n_recheck=8, loop_perts=(), agg="worst"):
    """CMA-ES in the loop at loop_dts (default 1 ms only: halves the cost on a shared machine); every improvement of the best is
    archived and the last n_recheck improvements are re-scored as the worse of 1 ms and 0.5 ms at the end; the best of those is
    returned (so the returned solution was selected on both dt)."""
    Ss = [plant(dt) for dt in loop_dts] + [plant(DTS[0], pert=p_) for p_ in loop_perts]; tally = {}; spd_valid = []
    def obj(z, plants):
        rs = [_v2(C.run(S_, z), S_) for S_ in plants]; sc = [score(r, objective) for r in rs]
        for r_ in rs:
            for why in (r_["reasons"] or ["VALID"]): tally[why] = tally.get(why, 0) + 1
            if r_["valid"]: spd_valid.append(r_["speed"])
        if agg == "soft": return _soft_agg(rs, plants, objective), rs
        if all(si < 100 for si in sc): return max(sc), rs
        return 100.0 + sum(si - 100.0 for si in sc if si >= 100), rs
    z0 = C.z0() if z0 is None else np.clip(np.asarray(z0, float), -1, 1); f0, rr = obj(z0, Ss); best = (f0, z0.copy(), rr); arch = [(f0, z0.copy())]; t0 = time.time(); hist = []
    es = cma.CMAEvolutionStrategy(z0, sigma, {"bounds": [-1, 1], "seed": seed, "verbose": -9, "maxiter": iters, "popsize": pop})
    while not es.stop():
        X = es.ask(); R_ = [obj(z, Ss) for z in X]; F = [r[0] for r in R_]; es.tell(X, F); k = int(np.argmin(F))
        if F[k] < best[0]: best = (F[k], np.array(X[k]), R_[k][1]); arch.append((F[k], np.array(X[k])))
        hist.append(round(best[0], 4))
        if log: log(es.countiter, best, time.time() - t0)
    # re-score the last improvements on BOTH dt (fresh plants) and keep the best worse-of-two
    Sb = [plant(dt) for dt in DTS] + [plant(dt, pert=p_) for p_ in loop_perts for dt in DTS]; cand = []
    for f_, z_ in arch[-n_recheck:]:
        f2, rs = obj(z_, Sb); cand.append((f2, z_, rs))
    cand.sort(key=lambda c: c[0]); b2 = cand[0]
    return (b2[0], b2[1], b2[2], dict(loop_best=float(best[0]), rechecked=[round(c[0], 4) for c in cand], reason_tally=tally, max_valid_speed_seen=(max(spd_valid) if spd_valid else None))), hist, time.time() - t0


PERTS = [dict(seed=k, mass_sd=0.03, ball_mass=0.05, friction=0.10, lag=0.20, q_sd=0.01, v_sd=0.02) for k in range(1, 5)]
# R1 (declared 2026-10-18 before any R1 run): robustness is measured on 16 FRESH draws of the same distribution (seeds 101-116, disjoint from
# the builder's 1-4, the verifier's 31-34/41-48 and the in-loop draws 201-202); robust = nominal hit at both dt AND >= 12/16 draws pass at both dt.
PERTS_R1 = [dict(PERTS[0], seed=k) for k in range(101, 117)]
PERTS_LOOP = [dict(PERTS[0], seed=k) for k in (201, 202)]   # robust-margin search: draws used INSIDE the search only
PERTS_LOOP4 = [dict(PERTS[0], seed=k) for k in (201, 202, 203, 204)]   # R2 soft robust refine: 4 in-loop draws (still disjoint from 101-116)


def _v2(r, S):
    """v2 rule on a perturbed plant: START_NOT_AT_REST is not applied (the start-rate noise is the imposed perturbation)."""
    if S.get("pert") and "START_NOT_AT_REST" in r["reasons"]:
        r["reasons"] = [x for x in r["reasons"] if x != "START_NOT_AT_REST"]; r["valid"] = not r["reasons"]; r["start_gate_exempt"] = True
        if r["valid"]: r["violation"] = 0.0
        else: r["violation"] = max(0.0, r["violation"] - 1.0)
    return r


def check_r1(C, z, perts=PERTS_R1):
    """R1 robustness: nominal at both dt + 16 fresh draws at both dt; counts under v1 (START_NOT_AT_REST applied to perturbed starts:
    solution-independent, it depends only on the draw) and v2 (not applied)."""
    def one(pert):
        out = []
        for dt in DTS:
            S = plant(dt, pert=pert); r = C.run(S, z); v1_valid = r["valid"]; v1_reasons = list(r["reasons"])
            if pert: r = _v2(r, S)
            out.append(dict(dt=dt, valid_v1=v1_valid, reasons_v1=v1_reasons, valid=r["valid"], reasons=r["reasons"], err=round(r["err"], 3), hit_v1=bool(v1_valid and r["err"] <= HIT_R),
                            hit=bool(r["valid"] and r["err"] <= HIT_R), speed=round(r["speed"], 3), elev=round(r["elev"], 2), az=round(r["az"], 2), slip=round(r["slip"], 4),
                            self_peak=round(r["self_contact_peak"], 1), draw=S.get("pert_draw")))
        return out
    nom = one(None); per = [one(p) for p in perts]
    n2 = sum(all(o["hit"] for o in pp) for pp in per); n1 = sum(all(o["hit_v1"] for o in pp) for pp in per)
    nh = all(o["hit"] for o in nom); need = int(np.ceil(0.75*len(per)))
    return dict(nominal=nom, nominal_valid_both=all(o["valid"] for o in nom), nominal_hit_both=nh, perturbations=per, n_pert=len(per), n_pass_v2=n2, n_pass_v1=n1,
                robust_v2=bool(nh and n2 >= need), robust_v1=bool(nh and n1 >= need), need=need, seeds=[p["seed"] for p in perts],
                slip_fail_v2=sum(any("SLIP" in o["reasons"] for o in pp) for pp in per), miss_v2=sum(any(o["valid"] and o["err"] > HIT_R for o in pp) for pp in per))


def check(C, z, perts=PERTS, density=0.0, v2=False):
    """Validity on FRESH plants at both dt (nominal), then each perturbation at both dt. pass = valid AND hit.
    v2 (2026-10-18, after the first checks): on a PERTURBED start the START_NOT_AT_REST gate is not applied, because the start-rate
    noise (N(0, 0.02 rad/s) on 17 joints) is itself the imposed perturbation and exceeds the 0.05 rad/s at-rest threshold by
    construction in some draws; every other gate is unchanged, and the nominal runs keep the gate."""
    def one(pert):
        out = []
        for dt in DTS:
            S = plant(dt, pert=pert); r = C.run(S, z, density=density)
            if v2 and pert and "START_NOT_AT_REST" in r["reasons"]:
                r["reasons"] = [x for x in r["reasons"] if x != "START_NOT_AT_REST"]; r["valid"] = not r["reasons"]; r["start_gate_exempt"] = True
            out.append(dict(dt=dt, valid=r["valid"], reasons=r["reasons"], err=round(r["err"], 3), hit=bool(r["valid"] and r["err"] <= HIT_R), speed=round(r["speed"], 3), elev=round(r["elev"], 2), az=round(r["az"], 2), land=r["land"], slip=round(r["slip"], 4), overspeed=round(r["overspeed"], 3), t_rel=r["t_rel"], draw=S.get("pert_draw")))
        return out
    nom = one(None); per = [one(p) for p in perts]
    n_pass = sum(all(o["hit"] for o in pp) for pp in per)
    return dict(nominal=nom, nominal_valid_both=all(o["valid"] for o in nom), nominal_hit_both=all(o["hit"] for o in nom), perturbations=per, n_pert_pass=n_pass, n_pert=len(per),
                robust=bool(all(o["hit"] for o in nom) and n_pass >= 3), perts_spec=perts)


SUPPLIED = dict(
    body="stock Menagerie G1 MJX (scene_mjx.xml) + a free 0.145 kg r 36.5 mm ball body; masses, geometry, joint ranges, rated speeds untouched",
    ratings="min(MJCF peak, Unitree URDF effort, vendor published max 120 N m): knee 120, hip pitch/yaw/roll 88, ankle pitch/roll 35, waist yaw 88, waist roll/pitch 35, shoulders/elbow/wrist roll 25, wrist pitch/yaw 5 N m; held (undriven) joints capped the same",
    actuator=f"first-order lag {LAG*1000:.0f} ms, torque-speed droop to 0 at rated speed, back-EMF brake above rated, OVERSPEED gate 1.1x rated",
    contact="stock MJX scene contact (foot-floor pairs solref 0.008, friction 1); ball geom solref 0.002 rigid; robot geoms contype 0 (released ball cannot touch the robot); solver iterations 50",
    attachment="ball held by an ACTIVE WELD equality to right_wrist_yaw_link at the palm site (solref 0.002): labelled attachment, no grip-force limit; release = weld switched off at the searched release step",
    flight="free-body flight in the full model during the 0.30 s follow-through, then the exact ball state continued in a ball+floor twin MuJoCo model (same dt); VACUUM (no drag) unless stated; landing = first floor contact of the ball",
    structure="construct wind-up -> release reference (throw.windup_reference: capacity-weighted constrained Jacobian, t_w 0.40, t_ref 0.55, leg cap 0.30, 95% joint range) for the inverted direction; tracker omega 20 rad/s; knots 17x10 + release time + amplitude [0.2,1] + time scale [0.57,1.17] (EffectTask)",
    direction_rule="ballistic inversion (vacuum) of the MINIMUM-SPEED trajectory from the reference's own release point to the target; two passes (bearing reference at 22.5 deg elev -> release point -> inverted direction -> reference -> release point -> inversion reported)",
    objective="landing error (m) at dt 1 ms in the CMA loop; the last 8 improvements re-scored as the worse of 1 ms and 0.5 ms on fresh plants and the best kept; any gate violation ranks below every valid throw",
    search="CMA-ES sigma 0.10, pycma seeds 1-3, 100 iterations x 16 per rung; distance LADDER per bearing (5 -> 10 -> 15 m), each rung warm-started from the previous rung's solution (continuation; the 5 m rung from the scaffold z0)",
    gates=f"stance (both feet, cumulative unload < 20 ms), no fall, no non-foot contact, energy closure, overspeed 1.1x, foot slip <= {SLIP_MAX} m, no robot self-contact > {SELF_F} N (R1), all through release + {FT} s follow-through; fresh plants at 1 ms and 0.5 ms",
    robustness="4 perturbations, each on fresh plants at both dt: per-body mass x N(1, 0.03) clipped +-15%, ball mass +-5%, friction +-10%, actuator lag +-20%, start joint angles N(0, 0.01 rad), start joint rates N(0, 0.02 rad/s); pass = valid AND hit; robust = nominal hit at both dt AND >= 3/4 perturbations pass",
    hit=f"landing error <= {HIT_R} m (target circle 1.0 m diameter)",
)


SUPPLIED_R1 = dict(SUPPLIED,
    search="R1: CMA-ES (pycma seeds 11-16), sigma 0.10, warm-started from stored ledgers chosen by the operator between waves (human-in-the-loop continuation schedule); polish = landing error; ceiling = release speed in the throw cone; directed ceiling = speed in a +-5 deg cone about the target's own inversion; robust_polish = worst of nominal + in-loop draws 201-202",
    robustness="R1: 16 fresh draws (seeds 101-116) of the round-0 perturbation distribution at both dt; robust = nominal hit at both dt AND >= 12/16 draws pass at both dt; counted under v2 (START_NOT_AT_REST not applied to perturbed starts; it depends only on the draw) and v1")
SUPPLIED_R2 = dict(SUPPLIED_R1,
    search="R2: sigma SCHEDULE refine (general, same for every target): a sequence of CMA-ES stages with decreasing sigma (e.g. 0.03 then 0.01), each warm from the previous stage's best; a stage's result is kept only if its worse-of-both-dt score improves on the incumbent's. pycma seeds 21-39. Warm starts: stored ledgers (named in each ledger's warm_from), chosen by the operator from the R1 results before this round's prediction.",
    objective="landing error (refine) or release speed in the target's own +-5 deg inversion cone (directed refine); soft robust refine: nominal hard + mean over 4 in-loop draws 201-204 soft (see _soft_agg); final selection on both dt")


def run_refine(tid, seed, warm, sched, mode="plain", objective="error", cone_tol=None, sub="r2"):
    """R2 GENERAL search improvement: sigma-schedule refinement from a stored ledger. The R1 polishes stalled at their warm starts
    because at sigma 0.10 almost every neighbour of a gate-edge throw is invalid; a smaller step keeps CMA in the valid set.
    sched = "0.03x120,0.01x80"; mode plain | soft (soft robust aggregation with 4 in-loop draws); objective error | speed;
    cone_tol: speed objective gated to +-cone_tol deg about the target's own inversion (the directed ceiling's rule)."""
    od = f"{OUT}/{sub}"; os.makedirs(od, exist_ok=True); tgt = TARGETS[tid]; S_b = plant(DTS[0]); C = Construction(tgt, S_b)
    if cone_tol:
        e = float(np.degrees(C.inv["elev"])); C.cone = ((e - cone_tol, e + cone_tol), cone_tol)
    tag = f"{tid}_{mode}_{objective}_s{seed}"; z = np.clip(np.array(json.load(open(warm))["z"], float), -1, 1)
    lp = PERTS_LOOP4 if mode == "soft" else (); agg = "soft" if mode == "soft" else "worst"
    Sb = [plant(dt) for dt in DTS] + [plant(dt, pert=p_) for p_ in lp for dt in DTS]
    def both(z_):
        rs = [_v2(C.run(S_, z_), S_) for S_ in Sb]
        if agg == "soft": return _soft_agg(rs, Sb, objective)
        sc = [score(r, objective) for r in rs]
        return max(sc) if all(x < 100 for x in sc) else 100.0 + sum(x - 100.0 for x in sc if x >= 100)
    f_inc = both(z); stages = [dict(stage="warm", f_both=f_inc)]; logf = open(f"{od}/{tag}.log", "w"); logf.write(f"warm {warm} f_both {f_inc:.4f}\n"); logf.flush(); t0 = time.time(); hist = []
    for i, st in enumerate(sched.split(",")):
        sg, it = st.split("x"); sg = float(sg); it = int(it)
        def log(k, best, secs, i=i):
            if k % 5 == 0 or k == 1: logf.write(f"stage {i} sigma {sg} it {k} best {best[0]:.4f} | {time.time() - t0:.0f} s\n"); logf.flush()
        best, h, secs = search(C, seed + i, it, 16, objective=objective, log=log, sigma=sg, z0=z, loop_perts=lp, agg=agg); hist += h
        f_new = both(best[1]); kept = f_new < f_inc
        stages.append(dict(stage=i, sigma=sg, iters=it, seed=seed + i, loop_best=best[3]["loop_best"], f_both=f_new, kept=bool(kept), reason_tally=best[3]["reason_tally"], max_valid_speed_seen=best[3]["max_valid_speed_seen"], secs=round(secs)))
        logf.write(f"stage {i} done f_both {f_new:.4f} (incumbent {f_inc:.4f}) kept {kept}\n"); logf.flush()
        if kept: z = np.array(best[1]); f_inc = f_new
    r = C.run(plant(DTS[0]), z); meas = []
    for dt in DTS:
        rr = C.run(plant(dt), z); meas.append(dict(dt=dt, valid=rr["valid"], reasons=rr["reasons"], err=rr["err"], speed=rr["speed"], elev=rr["elev"], az=rr["az"], slip=rr["slip"], overspeed=rr["overspeed"], self_peak=rr["self_contact_peak"]))
    chk = check_r1(C, z) if objective == "error" else None
    led = dict(job=f"M2_throw_target/{sub}", target=tid, spec=tgt, target_xy=C.xy.tolist(), stance_centre=C.c.tolist(), seed=seed, sched=sched, mode=mode, objective=objective,
               cone=getattr(C, "cone", None), loop_perts=[p["seed"] for p in lp], warm_from=warm, inversion=C.inv, secs=round(time.time() - t0), stages=stages, f_both=f_inc, hist=hist, meas=meas,
               valid_both=all(m_["valid"] for m_ in meas), speed_min=min(m_["speed"] for m_ in meas),
               found=dict(err=r["err"], speed=r["speed"], elev=r["elev"], az=r["az"], t_rel=r["t_rel"], release=r["release"], land=r["land"], amp=r["amp"], tscale=r["tscale"], work=r["work"], slip=r["slip"], overspeed=r["overspeed"], overspeed_by_joint=r["overspeed_by_joint"], reasons=r["reasons"], valid=r["valid"], self_contact_peak=r["self_contact_peak"]),
               check_r1=chk, supplied=SUPPLIED_R2, lag=LAG, z=np.asarray(z).tolist())
    json.dump(led, open(f"{od}/{tag}.json", "w"), indent=1, default=float)
    import pickle; pickle.dump(C.ref, open(f"{od}/{tag}_ref.pkl", "wb"))
    msg = f"[refine {tag}] " + " | ".join(f"{m_['dt']}: err {m_['err']:.3f} v {m_['speed']:.2f} elev {m_['elev']:.1f} az {m_['az']:.1f} {m_['reasons']}" for m_ in meas)
    if chk: msg += f" | hit both {chk['nominal_hit_both']} v2 {chk['n_pass_v2']}/16 v1 {chk['n_pass_v1']}/16 robust_v2 {chk['robust_v2']}"
    print(msg, flush=True); logf.write("DONE " + msg + "\n"); logf.close(); return led


def run_search(tid, seed, iters, pop):
    od = f"{OUT}/search"; os.makedirs(od, exist_ok=True); tgt = TARGETS[tid]
    S_ref = plant(DTS[0]); C = Construction(tgt, S_ref)
    logf = open(f"{od}/{tid}_s{seed}.log", "w")
    def log(it, best, secs):
        if it % 5 == 0 or it == 1: logf.write(f"it {it} best {best[0]:.4f} | {secs:.0f} s\n"); logf.flush()
    best, hist, secs = search(C, seed, iters, pop, log=log)
    z = best[1]; chk = check(C, z)
    r = C.run(plant(DTS[0]), z)
    led = dict(job="M2_throw_target", target=tid, spec=tgt, target_xy=C.xy.tolist(), stance_centre=C.c.tolist(), seed=seed, iters=iters, pop=pop, evals=iters*pop*2, secs=round(secs),
               inversion=C.inv, inversion_pass1=C.inv_pass1, p_rel_ref=C.p_rel_ref, ref_takeoff=C.ref["takeoff_ref"], best_score=float(best[0]), hist=hist,
               found=dict(err=r["err"], speed=r["speed"], elev=r["elev"], az=r["az"], t_rel=r["t_rel"], release=r["release"], land=r["land"], amp=r["amp"], tscale=r["tscale"], work=r["work"], slip=r["slip"], overspeed=r["overspeed"], overspeed_by_joint=r["overspeed_by_joint"], reasons=r["reasons"]),
               check=chk, supplied=SUPPLIED, lag=LAG, z=np.asarray(z).tolist())
    json.dump(led, open(f"{od}/{tid}_s{seed}.json", "w"), indent=1, default=float)
    import pickle; pickle.dump(C.ref, open(f"{od}/{tid}_s{seed}_ref.pkl", "wb"))
    print(f"[{tid} s{seed}] err {r['err']:.3f} m speed {r['speed']:.2f} elev {r['elev']:.1f} az {r['az']:.1f} | nominal hit both {chk['nominal_hit_both']} | perts {chk['n_pert_pass']}/4 robust {chk['robust']} | {secs:.0f} s", flush=True)
    logf.write(f"DONE err {r['err']:.3f} robust {chk['robust']}\n"); logf.close()
    return led


def run_chain(bearing, seed, iters, pop):
    """Distance ladder at one bearing: 5 m -> 10 m -> 15 m, each rung warm-started from the previous rung's returned solution
    (continuation in distance, the sprint ladders' convention; the first rung starts at the scaffold's z0)."""
    od = f"{OUT}/search"; os.makedirs(od, exist_ok=True); z_prev = None; out = []
    for d_ in (5, 10, 15):
        tid = f"d{d_}_{bearing}"; tgt = TARGETS[tid]; S_b = plant(DTS[0]); C = Construction(tgt, S_b)
        logf = open(f"{od}/{tid}_s{seed}.log", "w")
        def log(it, best, secs, logf=logf):
            if it % 5 == 0 or it == 1: logf.write(f"it {it} best {best[0]:.4f} | {secs:.0f} s\n"); logf.flush()
        best, hist, secs = search(C, seed, iters, pop, log=log, z0=z_prev)
        z = best[1]; chk = check(C, z); r = C.run(plant(DTS[0]), z)
        led = dict(job="M2_throw_target", target=tid, spec=tgt, target_xy=C.xy.tolist(), stance_centre=C.c.tolist(), seed=seed, iters=iters, pop=pop, evals=iters*pop, secs=round(secs),
                   warm_from=(None if z_prev is None else f"d{dict(((10, 5), (15, 10)))[d_]}_{bearing}_s{seed}"), inversion=C.inv, inversion_pass1=C.inv_pass1, p_rel_ref=C.p_rel_ref, ref=C.ref_steps,
                   best_score=float(best[0]), search_meta=best[3], hist=hist,
                   found=dict(err=r["err"], speed=r["speed"], elev=r["elev"], az=r["az"], t_rel=r["t_rel"], release=r["release"], land=r["land"], amp=r["amp"], tscale=r["tscale"], work=r["work"], slip=r["slip"], overspeed=r["overspeed"], overspeed_by_joint=r["overspeed_by_joint"], reasons=r["reasons"], valid=r["valid"]),
                   check=chk, supplied=SUPPLIED, lag=LAG, z=np.asarray(z).tolist())
        json.dump(led, open(f"{od}/{tid}_s{seed}.json", "w"), indent=1, default=float)
        import pickle; pickle.dump(C.ref, open(f"{od}/{tid}_s{seed}_ref.pkl", "wb"))
        print(f"[{tid} s{seed}] err {r['err']:.3f} m speed {r['speed']:.2f} elev {r['elev']:.1f} az {r['az']:.1f} valid {r['valid']} {r['reasons']} | nominal hit both {chk['nominal_hit_both']} | perts {chk['n_pert_pass']}/4 robust {chk['robust']} | {secs:.0f} s", flush=True)
        logf.write(f"DONE err {r['err']:.3f} robust {chk['robust']}\n"); logf.close(); z_prev = z; out.append(led)
    return out


def run_ceiling(seed, iters, pop, warm=None, tag=""):
    """Speed ceiling of the SAME construction and gates (free ball, 10 ms lag, ratings, both-feet support through the follow-through):
    objective = -release speed, with the Phase 11 throw cone (elev 0-45 deg, az within 20 deg of the bearing) as a gate (added after
    the first warm ceiling run, ceiling_warm_s2, maximised speed by releasing BACKWARD during the wind-up: az 148 deg). Reference = the 15 m straight target's inverted direction (42.5 deg); warm = a ledger whose z starts the search."""
    od = f"{OUT}/ceiling"; os.makedirs(od, exist_ok=True); tgt = TARGETS["d15_straight"]
    S_b = plant(DTS[0]); C = Construction(tgt, S_b); z0 = None; C.cone = ((0.0, 45.0), 20.0)
    if warm: z0 = np.array(json.load(open(warm))["z"])
    logf = open(f"{od}/ceiling{tag}_s{seed}.log", "w")
    def log(it, best, secs):
        if it % 5 == 0 or it == 1: logf.write(f"it {it} best {best[0]:.4f} | {secs:.0f} s\n"); logf.flush()
    best, hist, secs = search(C, seed, iters, pop, objective="speed", log=log, z0=z0)
    z = best[1]; res = []
    for dt in DTS:
        r = C.run(plant(dt), z); res.append(dict(dt=dt, speed=r["speed"], palm_speed=float(np.linalg.norm(r["release"]["palm_v"])) if r["release"] else None, valid=r["valid"], reasons=r["reasons"], elev=r["elev"], az=r["az"], overspeed=r["overspeed"], slip=r["slip"], err_to_15m=r["err"], land=r["land"]))
    led = dict(job="M2_throw_target/ceiling", cone=getattr(C, "cone", None), seed=seed, iters=iters, pop=pop, secs=round(secs), warm=warm, meas=res, valid_both=all(m_["valid"] for m_ in res), speed_min=min(m_["speed"] for m_ in res), search_meta=best[3], hist=hist, supplied=SUPPLIED_R1, lag=LAG, z=np.asarray(z).tolist())
    json.dump(led, open(f"{od}/ceiling{tag}_s{seed}.json", "w"), indent=1, default=float)
    import pickle; pickle.dump(C.ref, open(f"{od}/ceiling{tag}_s{seed}_ref.pkl", "wb"))
    print(f"[ceiling{tag} s{seed}] {[(round(m_['speed'], 2), m_['valid'], m_['reasons']) for m_ in res]} | tally {best[3]['reason_tally']} | {secs:.0f} s", flush=True); logf.write("DONE\n"); logf.close(); return led


def run_polish(tid, seed, iters, pop, warm):
    """Continuation at ONE target from a stored solution (the best valid solution across seeds/rungs), same search and gates."""
    od = f"{OUT}/polish"; os.makedirs(od, exist_ok=True); tgt = TARGETS[tid]; S_b = plant(DTS[0]); C = Construction(tgt, S_b)
    z0 = np.array(json.load(open(warm))["z"]); logf = open(f"{od}/{tid}_s{seed}.log", "w")
    def log(it, best, secs):
        if it % 5 == 0 or it == 1: logf.write(f"it {it} best {best[0]:.4f} | {secs:.0f} s\n"); logf.flush()
    best, hist, secs = search(C, seed, iters, pop, log=log, z0=z0); z = best[1]; chk = check(C, z); r = C.run(plant(DTS[0]), z)
    led = dict(job="M2_throw_target/polish", target=tid, spec=tgt, target_xy=C.xy.tolist(), stance_centre=C.c.tolist(), seed=seed, iters=iters, pop=pop, evals=iters*pop, secs=round(secs), warm_from=warm,
               inversion=C.inv, inversion_pass1=C.inv_pass1, p_rel_ref=C.p_rel_ref, ref=C.ref_steps, best_score=float(best[0]), search_meta=best[3], hist=hist,
               found=dict(err=r["err"], speed=r["speed"], elev=r["elev"], az=r["az"], t_rel=r["t_rel"], release=r["release"], land=r["land"], amp=r["amp"], tscale=r["tscale"], work=r["work"], slip=r["slip"], overspeed=r["overspeed"], overspeed_by_joint=r["overspeed_by_joint"], reasons=r["reasons"], valid=r["valid"]),
               check=chk, supplied=SUPPLIED_R1, lag=LAG, z=np.asarray(z).tolist())
    json.dump(led, open(f"{od}/{tid}_s{seed}.json", "w"), indent=1, default=float)
    import pickle; pickle.dump(C.ref, open(f"{od}/{tid}_s{seed}_ref.pkl", "wb"))
    print(f"[polish {tid} s{seed}] err {r['err']:.3f} speed {r['speed']:.2f} valid {r['valid']} | hit both {chk['nominal_hit_both']} perts {chk['n_pert_pass']}/4 robust {chk['robust']} | {secs:.0f} s", flush=True)
    logf.write("DONE\n"); logf.close(); return led


def run_directed_ceiling(tid, seed, iters, pop, warm, tol=5.0):
    """R1 wave 3, GENERAL staged objective (same rule for every target): maximise release speed with the release direction gated to
    within +-tol deg of the target's own ballistic inversion (elev, az) (the cone is computed from the spec, not fitted). Stage 1 of
    'speed first, then aim'; stage 2 is the landing-error polish from this ledger."""
    od = f"{OUT}/r1_directed"; os.makedirs(od, exist_ok=True); tgt = TARGETS[tid]; S_b = plant(DTS[0]); C = Construction(tgt, S_b)
    e = float(np.degrees(C.inv["elev"])); C.cone = ((e - tol, e + tol), tol)
    z0 = np.array(json.load(open(warm))["z"]); logf = open(f"{od}/{tid}_s{seed}.log", "w")
    def log(it, best, secs):
        if it % 5 == 0 or it == 1: logf.write(f"it {it} best {best[0]:.4f} | {secs:.0f} s\n"); logf.flush()
    best, hist, secs = search(C, seed, iters, pop, objective="speed", log=log, z0=z0); z = best[1]; res = []
    for dt in DTS:
        r = C.run(plant(dt), z); res.append(dict(dt=dt, speed=r["speed"], valid=r["valid"], reasons=r["reasons"], elev=r["elev"], az=r["az"], err=r["err"], slip=r["slip"], overspeed=r["overspeed"], self_peak=r["self_contact_peak"]))
    led = dict(job="M2_throw_target/r1_directed", target=tid, spec=tgt, target_xy=C.xy.tolist(), stance_centre=C.c.tolist(), cone=C.cone, inversion=C.inv, seed=seed, iters=iters, pop=pop, secs=round(secs), warm_from=warm,
               meas=res, valid_both=all(m_["valid"] for m_ in res), speed_min=min(m_["speed"] for m_ in res), search_meta=best[3], hist=hist, supplied=SUPPLIED_R1, lag=LAG, z=np.asarray(z).tolist())
    json.dump(led, open(f"{od}/{tid}_s{seed}.json", "w"), indent=1, default=float)
    import pickle; pickle.dump(C.ref, open(f"{od}/{tid}_s{seed}_ref.pkl", "wb"))
    print(f"[directed {tid} s{seed}] {[(round(m_['speed'], 2), m_['valid'], m_['reasons'], round(m_['err'], 2)) for m_ in res]} need {C.inv['speed']:.2f} | {secs:.0f} s", flush=True)
    logf.write("DONE\n"); logf.close(); return led


def run_robust_polish(tid, seed, iters, pop, warm):
    """R1 GENERAL construction change (same for every target): robust-margin search. The CMA objective is the worst of the nominal
    plant and 2 in-loop perturbation draws (PERTS_LOOP, same distribution as the robustness gate, v2 rule) at 1 ms; the last improvements
    are re-scored as the worst over those 3 plants at BOTH dt. The returned solution is then judged on the 16 disjoint draws (check_r1)."""
    od = f"{OUT}/r1_robust"; os.makedirs(od, exist_ok=True); tgt = TARGETS[tid]; S_b = plant(DTS[0]); C = Construction(tgt, S_b)
    z0 = np.array(json.load(open(warm))["z"]); logf = open(f"{od}/{tid}_s{seed}.log", "w")
    def log(it, best, secs):
        if it % 5 == 0 or it == 1: logf.write(f"it {it} best {best[0]:.4f} | {secs:.0f} s\n"); logf.flush()
    best, hist, secs = search(C, seed, iters, pop, log=log, z0=z0, loop_perts=PERTS_LOOP); z = best[1]
    chk = check_r1(C, z); r = C.run(plant(DTS[0]), z)
    led = dict(job="M2_throw_target/r1_robust", target=tid, spec=tgt, target_xy=C.xy.tolist(), stance_centre=C.c.tolist(), seed=seed, iters=iters, pop=pop, secs=round(secs), warm_from=warm,
               loop_perts=[p["seed"] for p in PERTS_LOOP], inversion=C.inv, best_score=float(best[0]), search_meta=best[3], hist=hist,
               found=dict(err=r["err"], speed=r["speed"], elev=r["elev"], az=r["az"], t_rel=r["t_rel"], release=r["release"], land=r["land"], amp=r["amp"], tscale=r["tscale"], work=r["work"], slip=r["slip"], overspeed=r["overspeed"], reasons=r["reasons"], valid=r["valid"], self_contact_peak=r["self_contact_peak"]),
               check_r1=chk, supplied=SUPPLIED_R1, lag=LAG, z=np.asarray(z).tolist())
    json.dump(led, open(f"{od}/{tid}_s{seed}.json", "w"), indent=1, default=float)
    import pickle; pickle.dump(C.ref, open(f"{od}/{tid}_s{seed}_ref.pkl", "wb"))
    print(f"[robust {tid} s{seed}] err {r['err']:.3f} slip {r['slip']:.4f} valid {r['valid']} | hit both {chk['nominal_hit_both']} v2 {chk['n_pass_v2']}/16 v1 {chk['n_pass_v1']}/16 robust_v2 {chk['robust_v2']} | {secs:.0f} s", flush=True)
    logf.write("DONE\n"); logf.close(); return led


def r1_check_ledger(path, out):
    """R1 re-gate of a stored ledger: nominal (self-contact gate, mass fix) + 16 fresh draws."""
    L, C = load_ledger(path); chk = check_r1(C, np.array(L["z"])); res = dict(path=path, target=L.get("target"), check_r1=chk)
    json.dump(res, open(out, "w"), indent=1, default=float); return res


def r1_regate_all(out):
    """Nominal replay of every stored ledger with the R1 gates (self contact) on fresh plants at both dt."""
    import glob
    rows = []
    for f in sorted(glob.glob(f"{OUT}/search/d*_s*.json")) + sorted(glob.glob(f"{OUT}/polish/d*_s*.json")) + sorted(glob.glob(f"{OUT}/ceiling/*.json")):
        L = json.load(open(f)); import pickle
        C = Construction.__new__(Construction); C.ref = pickle.load(open(f.replace(".json", "_ref.pkl"), "rb")); C.nT = 17
        if "target_xy" in L: C.xy = np.array(L["target_xy"]); C.tgt = L["spec"]
        else:
            C.tgt = TARGETS["d15_straight"]; C.xy, C.c = target_xy(plant(DTS[0]), C.tgt)
            if L.get("cone"): C.cone = L["cone"]
        rr = []
        for dt in DTS:
            r = C.run(plant(dt), np.array(L["z"])); rr.append(dict(dt=dt, valid=r["valid"], reasons=r["reasons"], err=round(r["err"], 3), speed=round(r["speed"], 3), self_peak=round(r["self_contact_peak"], 1), slip=round(r["slip"], 4)))
        rows.append(dict(file=os.path.relpath(f, OUT), meas=rr, valid_both=all(x["valid"] for x in rr), self_contact=any("SELF_CONTACT" in x["reasons"] for x in rr)))
        print(rows[-1]["file"], [(x["valid"], x["reasons"], x["err"], x["speed"], x["self_peak"]) for x in rr], flush=True)
    json.dump(rows, open(out, "w"), indent=1, default=float); return rows


# ------------------------------------------------------------------ summary, drag side study, render
def load_ledger(path):
    import pickle
    L = json.load(open(path)); C = Construction.__new__(Construction); C.ref = pickle.load(open(path.replace(".json", "_ref.pkl"), "rb"))
    C.nT = 17; C.xy = np.array(L["target_xy"]); C.tgt = L["spec"]; return L, C


def summary():
    import glob
    rows = []
    pred = {5: 5.78, 10: 9.03, 15: 11.42}
    for f in sorted(glob.glob(f"{OUT}/search/d*_s*.json")) + sorted(glob.glob(f"{OUT}/polish/d*_s*.json")):
        L = json.load(open(f)); c = L.get("check_v2") or L["check"]; fd = L["found"]; d_ = int(L["spec"]["dist"])
        nom = c["nominal"]
        rows.append(dict(target=L["target"], seed=L["seed"], dist=d_, err_1ms=nom[0]["err"], err_05ms=nom[1]["err"], valid_both=c["nominal_valid_both"], hit_both=c["nominal_hit_both"],
                         reasons=[o["reasons"] for o in nom], speed=round(fd["speed"], 2), palm_speed=round(float(np.linalg.norm(fd["release"]["palm_v"])), 2) if fd.get("release") else None,
                         elev=round(fd["elev"], 1), az=round(fd["az"], 1), elev_inv=round(np.degrees(L["inversion"]["elev"]), 1), az_inv=round(np.degrees(L["inversion"]["az"]), 1), v_inv=round(L["inversion"]["speed"], 2),
                         speed_ratio=round(fd["speed"]/pred[d_], 3), t_rel=round(fd["t_rel"], 3), amp=round(fd["amp"], 3), tscale=round(fd["tscale"], 3), slip=fd["slip"], overspeed=round(fd["overspeed"], 3),
                         perts=c["n_pert_pass"], pert_errs=[[o["err"] for o in pp] for pp in c["perturbations"]], pert_reasons=[[o["reasons"] for o in pp] for pp in c["perturbations"]], robust=c["robust"],
                         work={k: round(v, 1) for k, v in fd["work"].items()}, secs=L["secs"], warm_from=L.get("warm_from"), stage=("polish" if "/polish/" in f else "ladder"), check_rule=("v2" if L.get("check_v2") else "v1"),
                         perts_v1=L["check"]["n_pert_pass"]))
    by_t = {}
    for r in rows: by_t.setdefault(r["target"], []).append(r)
    best = {}
    for t, rs in by_t.items():
        rs_ok = [r for r in rs if r["valid_both"]]
        best[t] = min(rs_ok, key=lambda r: max(r["err_1ms"], r["err_05ms"])) if rs_ok else min(rs, key=lambda r: max(r["err_1ms"], r["err_05ms"]))
    out = dict(rows=rows, best=best, n_targets=len(by_t), targets_hit=sum(any(r["hit_both"] for r in rs) for rs in by_t.values()),
               targets_robust=sum(any(r["robust"] for r in rs) for rs in by_t.values()), seed_hit_rate=f"{sum(r['hit_both'] for r in rows)}/{len(rows)}")
    out["ceiling"] = [dict(file=os.path.basename(f), cone=json.load(open(f)).get("cone"), warm=(os.path.basename(json.load(open(f))["warm"]) if json.load(open(f)).get("warm") else None), seed=json.load(open(f))["seed"],
                           meas=[dict(dt=m_["dt"], speed=round(m_["speed"], 2), valid=m_["valid"], elev=round(m_["elev"], 1), az=round(m_["az"], 1)) for m_ in json.load(open(f))["meas"]],
                           max_valid_speed_seen=json.load(open(f)).get("search_meta", {}).get("max_valid_speed_seen"), tally=json.load(open(f)).get("search_meta", {}).get("reason_tally")) for f in sorted(glob.glob(f"{OUT}/ceiling/*.json"))]
    tal = {}
    for f in glob.glob(f"{OUT}/*/d*_s*.json") + glob.glob(f"{OUT}/ceiling/*.json"):
        for k, v in (json.load(open(f)).get("search_meta", {}).get("reason_tally") or {}).items(): tal[k] = tal.get(k, 0) + v
    out["reason_tally_all"] = tal
    json.dump(out, open(f"{OUT}/summary.json", "w"), indent=1, default=float); return out


def recheck(path):
    """Re-run the robustness check with the v2 rule on a stored ledger; stored beside the original check as check_v2."""
    L, C = load_ledger(path); chk = check(C, np.array(L["z"]), v2=True); L["check_v2"] = chk
    json.dump(L, open(path, "w"), indent=1, default=float); return dict(path=path, nominal_hit_both=chk["nominal_hit_both"], n_pert_pass=chk["n_pert_pass"], robust=chk["robust"])


def drag_side(path):
    """SIDE STUDY: the same solution with air drag on the ball flight (MuJoCo ellipsoid fluid model, air 1.204 kg/m^3, 1.8e-5 Pa s)."""
    L, C = load_ledger(path); S = plant(DTS[0]); r0 = C.run(S, np.array(L["z"])); r1 = C.run(S, np.array(L["z"]), density=1.204)
    return dict(target=L["target"], seed=L["seed"], vacuum_err=r0["err"], drag_err=r1["err"], vacuum_land=r0["land"], drag_land=r1["land"], shortfall=float(np.linalg.norm(np.array(r0["land"][:2]) - np.array(r1["land"][:2]))))


def render_ledger(path, out_mp4):
    """Robot trace through release + follow-through, then the ball's twin flight with the robot frozen at its last pose. A render-only
    copy of the scene adds a visual target disc (no collision)."""
    L, C = load_ledger(path); S = plant(DTS[0]); r = C.run(S, np.array(L["z"]), record=True); tr = r["trace"]
    sp = mujoco.MjSpec.from_file(scene_path()); sp.meshdir = f"{MEN}/unitree_g1/assets"
    tb = sp.worldbody.add_body(name="target", pos=[C.xy[0], C.xy[1], 0.001]); g = tb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[HIT_R, 0.001, 0], rgba=[0.9, 0.2, 0.2, 0.5]); g.contype = 0; g.conaffinity = 0
    sp.compile(); rp = f"{OUT}/render_scene_{L['target']}.xml"; open(rp, "w").write(sp.to_xml())
    Q = list(tr["q"]); qa = S["ball_qadr"]; qlast = Q[-1].copy()
    for p in tr["ball_traj"][1:]:
        for _ in range(10): q = qlast.copy(); q[qa:qa + 3] = p; q[qa + 3:qa + 7] = [1, 0, 0, 0]; Q.append(q)
    for _ in range(300): Q.append(Q[-1])
    title = f"Stock G1 throws at a floor target: {L['spec']['dist']:.0f} m, bearing {np.degrees(L['spec']['az']):.0f} deg left"
    sub = f"free 0.145 kg ball released {r['t_rel']:.2f} s, {r['speed']:.1f} m/s; lands {r['err']:.2f} m from centre (disc r 0.5); 10 ms lag, rated; robot frozen after 0.3 s; sim"
    _render(rp, np.array(Q), tr["dt"], out_mp4, title, sub, lookat=[0.5*C.xy[0] + 0.25, 0.5*C.xy[1], 0.6], distance=0.75*L["spec"]["dist"] + 4.0, slow=0.5)
    return out_mp4


def _render(scene, Q, dt, out, title, sub, lookat, distance, azimuth=125.0, elevation=-14.0, W_=960, H_=540, fps=30, slow=1.0):
    import subprocess, tempfile, shutil
    from PIL import Image, ImageDraw, ImageFont
    m = mujoco.MjModel.from_xml_path(scene); m.vis.global_.offwidth = W_; m.vis.global_.offheight = H_
    rd = mujoco.Renderer(m, H_, W_); d = mujoco.MjData(m); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = distance; cam.azimuth = azimuth; cam.elevation = elevation; cam.lookat[:] = lookat
    try: fb = ImageFont.truetype("DejaVuSans-Bold.ttf", 20); fs = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception: fb = fs = ImageFont.load_default()
    tmp = tempfile.mkdtemp(prefix="m2_render_", dir=os.environ.get("TMPDIR", "/tmp")); Tt = (len(Q) - 1)*dt; frames = []
    for k, t in enumerate(np.arange(0, Tt + 1e-9, slow/fps)):
        i = min(int(round(t/dt)), len(Q) - 1); d.qpos[:] = Q[i]; mujoco.mj_forward(m, d); rd.update_scene(d, camera=cam)
        im = Image.fromarray(rd.render()); dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, W_, 52], fill=(10, 16, 26)); dr.text((12, 5), title, fill=(240, 244, 250), font=fb); dr.text((12, 32), sub, fill=(170, 186, 205), font=fs)
        dr.rectangle([0, H_ - 24, W_, H_], fill=(10, 16, 26)); dr.text((12, H_ - 20), f"t = {t:.2f} s   ({slow:g}x speed)", fill=(255, 214, 140), font=fs)
        p_ = f"{tmp}/f{k:04d}.png"; im.save(p_); frames.append(p_)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{tmp}/f%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", out], check=True)
    pick = [frames[int(round(j))] for j in np.linspace(0, len(frames) - 1, 6)]; ims = [Image.open(p_).resize((W_//2, H_//2)) for p_ in pick]
    cs = Image.new("RGB", (3*W_//2, 2*H_//2))
    for j, im in enumerate(ims): cs.paste(im, ((j % 3)*W_//2, (j//3)*H_//2))
    cs.save(os.path.splitext(out)[0] + "_sheet.png"); shutil.rmtree(tmp)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "chain": run_chain(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 100, int(sys.argv[5]) if len(sys.argv) > 5 else 16)
    elif cmd == "search": run_search(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 80, int(sys.argv[5]) if len(sys.argv) > 5 else 16)
    elif cmd == "ceiling": run_ceiling(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), warm=(sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else None), tag=(sys.argv[6] if len(sys.argv) > 6 else ""))
    elif cmd == "directed_ceiling": run_directed_ceiling(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
    elif cmd == "robust_polish": run_robust_polish(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
    elif cmd == "refine": run_refine(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], mode=sys.argv[6], objective=sys.argv[7], cone_tol=(float(sys.argv[8]) if len(sys.argv) > 8 and sys.argv[8] != "-" else None), sub=(sys.argv[9] if len(sys.argv) > 9 else "r2"))
    elif cmd == "r1check": r1_check_ledger(sys.argv[2], sys.argv[3])
    elif cmd == "r1regate": r1_regate_all(sys.argv[2])
    elif cmd == "polish": run_polish(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
    elif cmd == "summary": print(json.dumps({k: v for k, v in summary().items() if k != "rows"}, indent=1, default=float))
    elif cmd == "recheck": print(recheck(sys.argv[2]))
    elif cmd == "drag": print(drag_side(sys.argv[2]))
    elif cmd == "render": print(render_ledger(sys.argv[2], sys.argv[3]))
    elif cmd == "predict":
        S = plant(DTS[0]); out = {}
        for tid, tgt in TARGETS.items():
            t0 = time.time(); C = Construction(tgt, S); out[tid] = dict(inversion=C.inv, pass1=C.inv_pass1, p_rel=C.p_rel_ref, target_xy=C.xy.tolist(), secs=time.time() - t0); print(tid, C.inv, flush=True)
        json.dump(out, open(f"{OUT}/prediction_inversion.json", "w"), indent=1, default=float)
