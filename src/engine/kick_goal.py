"""M1 KICK_GOAL (2026-10-18): kick a real simulated football into a goal with the stock G1.
construction = body + objective + support contract -> scaffold -> search -> gates (the kick_ball construction, unchanged):
  body      stock Menagerie G1 (MJX model), torque ceiling = min(MJCF peak, Unitree URDF effort, vendor published max 120 N m)
  objective the ball's centre crosses the goal plane inside the mouth (2.0 m x 1.0 m) at distance D, bearing az
  support   left foot (right-foot kick), the ImpactTask stance gates + a support-foot slip gate (<= 2 cm)
  scaffold  body-derived wind-up -> release reference for the effector direction (elev 15 deg = kick_ball's, az = goal bearing)
  search    CMA-ES over torque knots + release time + amplitude + tscale; worse-of (1 ms, 0.5 ms fresh) robust score
  gates     ImpactTask gates (hit, stance, envelope, energy, overspeed), slip gate, BALL RELEASED (no robot contact and
            separating at t_rel), GOAL (simulated ball flight in MuJoCo after t_rel, ball-only model: same floor/ball contact,
            posts + crossbar as geoms, air drag via MuJoCo's fluid model)
usage (from src/):
  python -m engine.kick_goal need                          spec-derived ball speed needed per goal (prediction input)
  python -m engine.kick_goal search <cfg> <seed> [iters] [pop]
  python -m engine.kick_goal robust <ledger.json>          4 perturbations x 2 dt on fresh plants
cfg in CONFIGS. Outputs: results/movements/M1_kick_goal/."""
import sys, os, json, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
import throw as T, branch_i as BI
from engine import core, task as TK, emergence as EM
from engine.construct import SCENARIOS, KICK_DRIVEN
from tasks.impact import ImpactTask

OUT = f"{core.RES}/movements/M1_kick_goal"
LAG = 0.010                     # first-order actuator lag (s), the project's preferred 10 ms (construct campaigns used 40 ms)
R = np.radians
# ---- SUPPLIED (listed in every ledger)
BALL = dict(radius=0.11, mass=0.43, pos=[0.30, -0.17, 0.11])        # FIFA size-5 ball; position = kick_ball's (in front of the right foot)
BALL_SHELL = True               # hollow-shell inertia 2/3 m r^2 (a football is a pressurised shell), not MuJoCo's solid 2/5
ROLL = 0.0055                   # ball rolling friction (m): rolling-resistance coefficient 0.05 (short grass), MuJoCo condim 6
TORSION = 0.005                 # torsional friction (m), MuJoCo default value
AIR = dict(density=1.2, viscosity=1.8e-5)   # ball flight only (MuJoCo ellipsoid fluid model); the kick phase has no air
GOAL = dict(width=2.0, height=1.0, post_r=0.06)
FLIGHT_T = 8.0                  # s cap on the simulated ball flight
SLIP_MAX = 0.02                 # support-foot horizontal travel gate (m)
# ---- REPAIR ROUND 1 physics (2026-10-18, after the physics-lens verification). PHYS=1 replays round 0 exactly.
PHYS = int(os.environ.get("KICK_PHYS", "2"))
#   (a) BALL vs WHOLE BODY: every robot collision-class geom (group 3) collides with the ball (and only with the ball: contype 2,
#       conaffinity 0; ball conaffinity |= 2). Round 0 let the ball pass through the shin/ankle brace (the MJX G1 carries no
#       collision bits on its body geoms; only the effector body's were switched on).
#   (b) CONTINUATION: robot + ball are simulated together for CONT_T after t_rel with the last driven command held (servos keep
#       their targets); any robot-ball contact in that window = BALL_RECONTACT; the flight starts from the continued ball state.
#   (c) GATE-MARGIN objective: among valid solutions maximise min(mouth margin [m], slip margin as a fraction of the 2 cm gate)
#       (+0.01 x ball speed tie-break); round 0 maximised the mouth margin only and parked the support foot at 1.6-2.0 cm.
#   (d) ROBUST-IN-SCORE (opt-in KICK_RDRAWS=k): k perturbation draws (the robustness-check perturbation family, fixed draws =
#       common random numbers, alternating 1 ms / 0.5 ms fresh plants) join the worse-of score.
CONT_T = 0.30
N_ROBUST = int(os.environ.get("KICK_NDRAWS", "16"))   # robustness draws per solution (M2 rule: >= 16); robust = pass >= 3/4 of them at both dt
CONFIGS = {
    "g5_0":   dict(D=5.0,  az=0.0),
    "g5_20":  dict(D=5.0,  az=20.0),         # +20 deg = the robot's left (across the body for a right-foot kick)
    "g10_0":  dict(D=10.0, az=0.0),
    "g10_20": dict(D=10.0, az=20.0),
    "g10_0_power": dict(D=10.0, az=0.0, objective="speed"),   # SIDE STUDY: the same gates, objective = max ball speed (the kick ceiling with a goal)
}
RATED = None
def rating_table():
    """min(MJCF peak, Unitree URDF effort, vendor published 120 N m): knee 139->120, ankle pitch/roll 50->35, waist roll/pitch 50->35,
    hip roll 139->88. Ratings only; geometry, masses, ranges, speeds unchanged."""
    global RATED
    if RATED is None:
        import yaml
        sp = yaml.safe_load(open(os.path.join(core.ROOT, "spec", "g1.yaml")))
        RATED = {jn: min(float(r["tau_max_peak"]["value"]), float(r["tau_max_continuous"]["value"]), 120.0) for jn, r in sp["joints"].items() if "tau_max_continuous" in r}
    return RATED


def demand_for(cfg):
    c = CONFIGS[cfg]; base = SCENARIOS["kick_ball"]
    d = dict(ImpactTask("G1").demand_default(), **{k: v for k, v in base.items() if k != "type"})     # as construct.make_task
    d.update(direction=(R(15), R(c["az"])), object=dict(BALL, tee=False), goal=dict(GOAL, D=c["D"], az=R(c["az"])),
             elev_tol=np.pi, az_tol=np.pi, goal_objective=c.get("objective", "margin"))        # the ball cone is replaced by the GOAL gate
    return d


class GoalKickTask(ImpactTask):
    name = "kick_goal"
    def demand_default(self): return super().demand_default()

    # ---------------- plant: stock body, documented ratings, ball contact
    def _m1(self, S, demand):
        m = S["m"]; bb = m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]
        gb = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bb][0]
        m.geom_condim[gb] = 6; m.geom_friction[gb][1] = TORSION; m.geom_friction[gb][2] = ROLL
        r = float(m.geom_size[gb][0]); mb = float(m.body_mass[bb])
        if BALL_SHELL: m.body_inertia[bb] = [2/3*mb*r*r]*3
        rt = rating_table(); held = []
        for (a, dof, qad, tau, kp, kv) in S["held"]:          # undriven joints held by servos: clip to the rating too (waist roll 35)
            jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a][0])); held.append((a, dof, qad, min(tau, rt.get(jn, tau)), kp, kv))
        S["held"] = held
        S["m1_caps"] = np.array([min(1.0, rt.get(t, 1e9)/float(S["jarr"]["tau"][i])) for i, t in enumerate(S["types"])])
        S["ball_body"] = int(bb); S["ball_geom"] = int(gb); S["ball_qadr"] = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")])
        if PHYS >= 2:
            n_on = 0
            for g in S["robot_geoms"]:
                if int(m.geom_group[g]) == 3 and int(m.geom_contype[g]) == 0 and int(m.geom_conaffinity[g]) == 0: m.geom_contype[g] = 2; m.geom_conaffinity[g] = 0; n_on += 1
            m.geom_conaffinity[gb] = int(m.geom_conaffinity[gb]) | 2; S["ball_body_geoms_on"] = n_on
            # the broad-phase filters BODY pairs by the compile-time OR of their geoms' bits (body_contype/body_conaffinity): refresh them,
            # else bodies whose geoms were all bit-less at compile (shin + ankle brace = knee link, thighs, arms...) are still skipped
            for b in range(m.nbody):
                gs = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b]
                if gs: m.body_contype[b] = int(np.bitwise_or.reduce([int(m.geom_contype[g]) for g in gs])); m.body_conaffinity[b] = int(np.bitwise_or.reduce([int(m.geom_conaffinity[g]) for g in gs]))
            # collision bits switched on AFTER compile are culled by the compile-time body BVH (mid-phase): round 0's "enabled" foot
            # capsules never touched the ball. Disabling the mid-phase makes every enabled geom pair reach narrow-phase.
            m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
        BI.apply_dt(S, m.opt.timestep); BI.settle(S); S["_m1"] = True

    def plant(self, demand, dt=core.DT_OPT, fresh=False):
        S = super().plant(demand, dt, fresh)
        if not S.get("_m1"): self._m1(S, demand)
        return S

    def fresh_plant(self, demand, dt, uid):
        """An uncached plant (perturbation draws): the ImpactTask plant recipe with a unique start offset."""
        kw = dict(site=demand.get("effector"), ball=False, mass=0.0, support=demand.get("support", "both"), driven=demand.get("driven"), mjcf=self.scene(demand))
        S0 = T.setup(self.robot, **kw); S = T.setup(self.robot, start=np.array(S0["q0"]) + 1e-9*(1000 + uid), **kw)
        m = S["m"]; hb = int(m.site_bodyid[S["site"]])
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) == hb and int(m.geom_group[g]) == 3: m.geom_contype[g] = 1; m.geom_conaffinity[g] = 1
        BI.apply_dt(S, dt); BI.settle(S); S["_setup_kw"] = kw; S["ball_dof"] = int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")])
        self._m1(S, demand); return S

    # ---------------- controller in rated-activation units
    def controller(self, S, demand, p):
        fn = super().controller(S, demand, p); inv = 1.0/S["m1_caps"]
        if LIMIT_GUARD and fn is not None:
            # (h) SOFT-STOP GUARD (R2c, general, actuator-firmware style): inside GUARD_DEG of a driven joint's range limit and moving
            # toward it, a braking command ramps from 0 at the margin edge to the joint's full RATED torque at the limit (rated-activation
            # units: 1.0 = rated). Body-derived (ranges, ratings); no task constant. Summed with the tracker + knots, then lagged and clipped.
            J = S["jarr"]; lo = np.array([S["m"].jnt_range[S["m"].dof_jntid[k]][0] for k in J["dof"]]); hi = np.array([S["m"].jnt_range[S["m"].dof_jntid[k]][1] for k in J["dof"]]); mu = np.radians(GUARD_DEG)
            def g(tt, d, S_, fn=fn, inv=inv, lo=lo, hi=hi, mu=mu, qa=J["qad"], dof=J["dof"]):
                q = d.qpos[qa]; w = d.qvel[dof]
                b = np.where(w < 0, np.clip((lo + mu - q)/mu, 0.0, 1.0), 0.0) - np.where(w > 0, np.clip((q - (hi - mu))/mu, 0.0, 1.0), 0.0)
                return fn(tt, d, S_)*inv + b
            if S.get("hold_sched") is not None:
                f0 = S["hold_sched"]; S["hold_sched"] = lambda tt, f0=f0, inv=inv: f0(tt)*inv
            return g
        if S.get("hold_sched") is not None:
            f0 = S["hold_sched"]; S["hold_sched"] = lambda tt, f0=f0, inv=inv: f0(tt)*inv     # holds are fractions of the PEAK torque
        if fn is None: return None
        return lambda tt, d, S_, fn=fn, inv=inv: fn(tt, d, S_)*inv                                # the tracker returns torque / peak

    # ---------------- rollout: kick (robot + ball) -> release check -> ball flight (ball-only MuJoCo) -> goal
    def rollout(self, S, demand, p, ctrl_fn=None, record=False):
        m = S["m"]; fb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, BI.ROBOTS[self.robot]["foot_bodies"][S["support"][0]]); st = dict(x0=None, slip=0.0, ctrl=None)
        k_ = [0]
        def fn(tt, d, S_):
            st["ctrl"] = d.ctrl.copy(); xy = np.array(d.xpos[fb][:2])
            if PHYS >= 3 and k_[0] % 2 == 0:
                ov = lim_overrun(S, d.qpos)
                if ov[0] > st.get("lim", (0.0, -1))[0]: st["lim"] = ov
            k_[0] += 1
            if st["x0"] is None: st["x0"] = xy
            st["slip"] = max(st["slip"], float(np.linalg.norm(xy - st["x0"])))
            return ctrl_fn(tt, d, S_) if ctrl_fn is not None else np.zeros(len(S["types"]))
        dem = dict(demand, _caps=S["m1_caps"].tolist())
        rec = super().rollout(S, dem, p, ctrl_fn=fn, record=record)
        qpos, qvel = rec["final_state"]; reasons = list(rec["reasons"]); viol = float(rec["violation"])
        if st["slip"] > SLIP_MAX: viol += 1.0 + 50.0*(st["slip"] - SLIP_MAX); reasons.append("FOOT_SLID")
        rel = self.released(S, qpos, qvel)
        if not rel["released"]: viol += 1.0 + rel["shortfall"]; reasons.append("BALL_NOT_RELEASED")
        cont = None
        if PHYS >= 3:
            self._last_ctrl = st["ctrl"]; cont = recovery(S, qpos, qvel, st["ctrl"])
            if cont["n_contact"] > 0: viol += 1.0 + 0.05*cont["n_contact"]; reasons.append("BALL_RECONTACT")
            if cont["fell"] and FALLGATE: viol += 1.0 + (1.0 - (REC_T if cont["t_fall"] is None else cont["t_fall"])/REC_T) + max(0.0, 0.85 - cont["up_end"]) + max(0.0, 1.0 - cont["z_com_rel"]); reasons.append("FELL_AFTER")
            lim_ = max(st.get("lim", (0.0, -1))[0], cont["lim_over"] if FALLGATE else 0.0)
            if lim_ > LIM_TOL: viol += 1.0 + 20.0*(lim_ - LIM_TOL); reasons.append("JOINT_LIMIT")
            fl = ball_flight(S, demand, cont["qpos"], cont["qvel"])
        elif PHYS >= 2:
            self._last_ctrl = st["ctrl"]; cont = continuation(S, qpos, qvel, st["ctrl"])
            if cont["n_contact"] > 0: viol += 1.0 + 0.05*cont["n_contact"]; reasons.append("BALL_RECONTACT")
            fl = ball_flight(S, demand, cont["qpos"], cont["qvel"])
        else: fl = ball_flight(S, demand, qpos, qvel)
        if not fl["goal"]: viol += 1.0 + 2.0*fl["miss"]; reasons.append("NO_GOAL")
        r3 = {} if (cont is None or PHYS < 3) else dict(rec_mode=REC_MODE, fallgate=FALLGATE, guard=LIMIT_GUARD, rec_fell=cont["fell"], rec_nonfoot=cont["nonfoot"], rec_t_fall=cont["t_fall"], rec_t_touchdown=cont["t_touchdown"], rec_step_xy=cont["step_xy"], rec_up_end=cont["up_end"], rec_zrel=cont["z_com_rel"], rec_travel=cont["travel"], rec_pelvis_min=cont["pelvis_min"],
                                                       lim_kick_deg=float(np.degrees(st.get("lim", (0.0, -1))[0])), lim_kick_joint=(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(st["lim"][1])) if st.get("lim", (0, -1))[1] >= 0 else None),
                                                       lim_rec_deg=float(np.degrees(cont["lim_over"])), lim_rec_joint=cont["lim_joint"])
        ex = dict(rec["extra"], slip=st["slip"], **r3, **({} if cont is None else dict(cont_contacts=cont["n_contact"], cont_clear=cont["clear"], cont_ball_speed=cont["ball_speed"])), **{f"rel_{k}": v for k, v in rel.items()}, **{f"goal_{k}": v for k, v in fl.items() if k != "path"})
        out = dict(rec, valid=(viol <= 0 and not reasons), reasons=reasons, violation=viol, extra=ex)
        if record: out["flight_path"] = fl["path"]
        return out

    def released(self, S, qpos, qvel):
        m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = qpos; d.qvel[:] = qvel; mujoco.mj_forward(m, d); bb = S["ball_body"]; rg = S["robot_geoms"]
        touch = any(bb in (int(m.geom_bodyid[d.contact[i].geom1]), int(m.geom_bodyid[d.contact[i].geom2])) and (d.contact[i].geom1 in rg or d.contact[i].geom2 in rg) for i in range(d.ncon))
        vel6 = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, S["site"], vel6, 0)
        sep = d.xpos[bb] - d.site_xpos[S["site"]]; u = sep/max(np.linalg.norm(sep), 1e-9); vrel = float(np.dot(np.asarray(qvel[S["ball_dof"]:S["ball_dof"] + 3]) - vel6[3:], u))
        ok = (not touch) and vrel > 0.0
        return dict(released=bool(ok), touching=bool(touch), sep_speed=vrel, shortfall=(1.0 if touch else 0.0) + max(0.0, -vrel))

    def objective(self, rec, demand):
        if demand.get("goal_objective") == "speed": return -rec["value"]                 # power shot: fastest ball that still scores
        if PHYS >= 2: return -(min(rec["extra"]["goal_margin"], (SLIP_MAX - rec["extra"]["slip"])/SLIP_MAX) + 0.01*rec["value"])   # (c) gate margin
        return -(rec["extra"]["goal_margin"] + 0.01*rec["value"])   # most central goal; ball speed a tie-break

    def unpack(self, S, demand, z):
        p = super().unpack(S, demand, z); return p


# ------------------------------------------------------------------ (b) post-release continuation: robot + ball together
def continuation(S, qpos, qvel, ctrl, T_=CONT_T, rec_every=0):
    """Robot + ball from t_rel for T_ s: driven actuators hold their last command (rated ctrlrange, back-EMF brake as in the
    rollout), held servos keep their targets. Counts robot-ball contact steps and the minimum ball-to-robot-geom clearance."""
    m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = qpos; d.qvel[:] = qvel
    if ctrl is not None: d.ctrl[:] = ctrl
    mujoco.mj_forward(m, d); bb = S["ball_body"]; rg = S["robot_geoms"]; nc = 0; clear = 1e9; fr = np.zeros(6)
    gb = S["ball_geom"]; ck = [g for g in rg if int(m.geom_contype[g]) or int(m.geom_conaffinity[g])]; frames = []
    for s in range(int(round(T_/m.opt.timestep))):
        if rec_every and s % rec_every == 0: frames.append(d.qpos.copy())
        BI.backemf_brake(m, d, S); mujoco.mj_step(m, d)
        if any(bb in (int(m.geom_bodyid[d.contact[i].geom1]), int(m.geom_bodyid[d.contact[i].geom2])) and (d.contact[i].geom1 in rg or d.contact[i].geom2 in rg) for i in range(d.ncon)): nc += 1
        if s % 10 == 0: clear = min([clear] + [float(mujoco.mj_geomDistance(m, d, gb, g, 0.5, fr)) for g in ck])
    dv = S["ball_dof"]
    return dict(frames=frames, qpos=d.qpos.copy(), qvel=d.qvel.copy(), n_contact=nc, clear=clear, ball_speed=float(np.linalg.norm(d.qvel[dv:dv + 3])))


# ------------------------------------------------------------------ REPAIR ROUND 2 (PHYS 3): joint-limit gate + post-release recovery
#   (e) JOINT_LIMIT: any limited robot hinge beyond its range by more than LIM_TOL (soft-limit penetration) at any step of the kick
#       or the recovery window is a gate failure (a rigid stop would not allow it; the R1 kicking ankle ran 2.6-4.9 deg past).
#   (f) RECOVERY: after t_rel the robot is no longer frozen on its last command. A general stand-recovery law takes over for
#       REC_T s (= t_rel + 1 s): driven joints track the plant's own settled stance posture through the Phase-14 balance law
#       (capture point of the ROBOT's COM over the feet centre via the ankles, trunk vertical via the hips; gains from the
#       body's constrained Jacobians, balance.BalanceController, unchanged) as an explicit PD (omega = the scaffold tracker's 40
#       rad/s, critically damped, gravity-hold feed-forward), through the same 10 ms first-order lag, rated torque clip and
#       torque-speed droop as the kick; held servos keep their targets; back-EMF brake on. Nothing in it is task-specific.
#       Gates over the window: FELL_AFTER (any non-foot robot geom touches the floor, or at the end torso up < 0.85 or robot COM
#       height < 0.6 x stance: the Phase-14 standing rule without its speed clause), BALL_RECONTACT (any robot-ball contact),
#       JOINT_LIMIT. Reported (not gated): support-foot travel over the window, pelvis minimum height, both feet's final spread.
LIM_TOL = np.radians(1.0)
LIMIT_GUARD = int(os.environ.get("KICK_GUARD", "0")); GUARD_DEG = 3.0
FALLGATE = int(os.environ.get("KICK_FALLGATE", "1"))   # 1: FELL_AFTER gated and JOINT_LIMIT over kick + recovery; 0 (R2b track A): FELL_AFTER reported only, JOINT_LIMIT over the kick phase
REC_T = 1.0
REC_OMEGA = 40.0
def _lim_setup(S):
    if "lim" in S: return S["lim"]
    m = S["m"]; js = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and m.jnt_limited[j] and int(m.jnt_bodyid[j]) != S["ball_body"]]
    S["lim"] = (np.array([m.jnt_qposadr[j] for j in js]), np.array([m.jnt_range[j][0] for j in js]), np.array([m.jnt_range[j][1] for j in js]), js)
    return S["lim"]


def lim_overrun(S, qpos):
    qa, lo, hi, js = _lim_setup(S); q = np.asarray(qpos)[qa]; ov = np.maximum(lo - q, q - hi)
    k = int(np.argmax(ov)); return max(0.0, float(ov[k])), js[k]


def _rec_setup(S):
    if "rec" in S: return S["rec"]
    import balance as B
    m = S["m"]; bc = B.BalanceController(S, S["q_settled"], params=dict(mode="stand"))
    J = S["jarr"]; ieff = np.array([1.0/float(S["Minv_key"][dof, dof]) for dof in J["dof"]])
    kp = REC_OMEGA**2*ieff; kd = 2.0*REC_OMEGA*ieff
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    footg = set().union(*S["feet"]); fbs = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in BI.ROBOTS["G1"]["foot_bodies"]]
    d0 = mujoco.MjData(m); d0.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d0); z_st = float(d0.subtree_com[pel][2])
    S["rec"] = dict(bc=bc, kp=kp, kd=kd, pel=pel, footg=footg, fbs=fbs, z_st=z_st, a_hold=np.array([S["a_hold"][t] for t in S["types"]])*J["tau"])
    return S["rec"]


REC_MODE = os.environ.get("KICK_REC", "step")     # "stand" (R2a: swing leg back to the settled stance) | "step" (R2b: capture step)
#   (g) CAPTURE STEP (R2b, general): until the kicking (swing) foot touches down, its target is the instantaneous capture
#       point of the robot COM (LIPM, omega0 = sqrt(g/z_com)), kept at least the settled stance width from the support foot
#       and within 0.9 x the leg length (hip to sole at the settled pose) of the swing hip: both body-derived. The swing-leg
#       joint targets come from a damped least-squares IK (sole flat, yaw of the support foot, sole height of the settled
#       stance) re-solved at 100 Hz; at touchdown (swing-foot floor load > 1 N) they freeze at the landed angles, and the Phase-14 two-foot balance
#       (capture point over the feet centre, trunk vertical) continues on the new base. R2a pulled the swing leg back to the
#       settled stance, i.e. AWAY from a capture point that lies toward the kick: every R1 final fell under it.
def _leg_setup(S, R_):
    m = S["m"]; sw = 1 - S["support"][0]; side = ("left", "right")[sw]; bc = R_["bc"]
    names = [f"{side}_hip_pitch_joint", f"{side}_hip_roll_joint", f"{side}_hip_yaw_joint", f"{side}_knee_joint", f"{side}_ankle_pitch_joint", f"{side}_ankle_roll_joint"]
    js = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    d0 = mujoco.MjData(m); d0.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d0)
    fsw = bc.feet[sw]; fsu = bc.feet[S["support"][0]]; hip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hip_pitch_link")
    sole = lambda dd, fp: (dd.site_xpos[fp[1]] if fp[1] is not None else dd.xpos[fp[0]])
    R_.update(sw=sw, sw_qa=np.array([m.jnt_qposadr[j] for j in js]), sw_dof=np.array([m.jnt_dofadr[j] for j in js]), sw_lo=np.array([m.jnt_range[j][0] for j in js]), sw_hi=np.array([m.jnt_range[j][1] for j in js]),
              fsw=fsw, fsu=fsu, hip=hip, sole=sole, z_sole=float(sole(d0, fsw)[2]), width=float(np.linalg.norm(sole(d0, fsw)[:2] - sole(d0, fsu)[:2])),
              reach=0.9*float(np.linalg.norm(d0.xpos[hip] - sole(d0, fsw))), dk=mujoco.MjData(m), sw_geoms=set(S["feet"][sw]))


def _swing_ik(S, R_, d, tgt_xy, n_it=8):
    m = S["m"]; dk = R_["dk"]; dk.qpos[:] = d.qpos; fp = R_["fsw"]; qa = R_["sw_qa"]; dof = R_["sw_dof"]
    Ru = d.xmat[R_["fsu"][0]].reshape(3, 3); yaw = np.arctan2(Ru[1, 0], Ru[0, 0]); c, s_ = np.cos(yaw), np.sin(yaw)
    Rt = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]]); pt = np.array([tgt_xy[0], tgt_xy[1], R_["z_sole"]])
    jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    for _ in range(n_it):
        mujoco.mj_kinematics(m, dk); mujoco.mj_comPos(m, dk)
        if fp[1] is not None: pc = dk.site_xpos[fp[1]]; Rc = dk.site_xmat[fp[1]].reshape(3, 3); mujoco.mj_jacSite(m, dk, jp, jr, fp[1])
        else: pc = dk.xpos[fp[0]]; Rc = dk.xmat[fp[0]].reshape(3, 3); mujoco.mj_jacBody(m, dk, jp, jr, fp[0])
        er = 0.5*sum(np.cross(Rc[:, i], Rt[:, i]) for i in range(3)); e = np.concatenate([pt - pc, er])
        Jm = np.vstack([jp[:, dof], jr[:, dof]]); dq = Jm.T @ np.linalg.solve(Jm @ Jm.T + 1e-4*np.eye(6), e)
        dk.qpos[qa] = np.clip(dk.qpos[qa] + dq, R_["sw_lo"], R_["sw_hi"])
    return dk.qpos[qa].copy()


def _rec_targets(S, R_, d, in_contact):
    """Phase-14 stand-mode targets with the ROBOT's COM (pelvis subtree; the world subtree would include the flying ball).
    step mode: the swing leg's base targets come from the capture-step IK; the balance corrections act on it only after touchdown."""
    import balance as B, plant as P
    bc = R_["bc"]; m = S["m"]; q = bc.q_stand.copy(); p = bc.p
    mujoco.mj_subtreeVel(m, d); com = d.subtree_com[R_["pel"]]; vcom = d.subtree_linvel[R_["pel"]]
    w0 = np.sqrt(B.G/max(com[2], 0.2)); cp = com[:2] + vcom[:2]/w0
    swing_free = False
    if REC_MODE == "step":
        if "sw_qa" not in R_: _leg_setup(S, R_)
        if R_.get("sw_frozen") is None:
            swing_free = True; k = R_.get("ik_k", 0); R_["ik_k"] = k + 1
            if k % 5 == 0 or R_.get("sw_q") is None:          # 100 Hz re-solve (targets at 500 Hz)
                su = R_["sole"](d, R_["fsu"])[:2]; tgt = cp.copy(); dvec = tgt - su; dn = np.linalg.norm(dvec)
                if dn < R_["width"]: tgt = su + (dvec/max(dn, 1e-9) if dn > 1e-6 else (R_["sole"](d, R_["fsw"])[:2] - su)/max(np.linalg.norm(R_["sole"](d, R_["fsw"])[:2] - su), 1e-9))*R_["width"]
                hx = d.xpos[R_["hip"]][:2]; hv = tgt - hx; hn = np.linalg.norm(hv)
                rh = np.sqrt(max(0.0, R_["reach"]**2 - (float(d.xpos[R_["hip"]][2]) - R_["z_sole"])**2))   # horizontal reach at the current hip height
                if hn > rh: tgt = hx + hv/max(hn, 1e-9)*rh
                R_["sw_q"] = _swing_ik(S, R_, d, tgt); R_["sw_tgt"] = tgt
        q[R_["sw_qa"]] = R_["sw_frozen"] if R_.get("sw_frozen") is not None else R_["sw_q"]
    swq = set(R_["sw_qa"].tolist()) if (swing_free and "sw_qa" in R_) else set()
    feet = 0.5*(B.foot_xy(d, bc.feet[0]) + B.foot_xy(d, bc.feet[1])); e = cp - feet
    if swing_free: e = cp - B.foot_xy(d, R_["fsu"])          # one foot down: capture point over the support foot
    if in_contact:
        for a in bc.qa["ap"]:
            if a not in swq: q[a] -= p["k_cp"]*bc.g_ap*e[0]
        if bc.g_ar is not None:
            for a in bc.qa["ar"]:
                if a not in swq: q[a] -= p["k_cp"]*bc.g_ar*e[1]
    tilt = bc._tilt(d, bc.tb); vel = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bc.tb, vel, 0)
    for a in bc.qa["hp"]:
        if a not in swq: q[a] -= bc.g_hp*(p["k_trunk"]*tilt[0] + p["k_trunk_d"]*vel[1])
    for a in bc.qa["hr"]:
        if a not in swq: q[a] -= bc.g_hr*(p["k_trunk"]*tilt[1] + p["k_trunk_d"]*vel[0])
    return q


def recovery(S, qpos, qvel, ctrl, T_=REC_T, rec_every=0):
    """Robot + ball from t_rel for T_ s under the general stand-recovery law (see (f)). Returns gates and diagnostics."""
    import balance as B
    m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = qpos; d.qvel[:] = qvel
    if ctrl is not None: d.ctrl[:] = ctrl
    mujoco.mj_forward(m, d); R_ = _rec_setup(S); J = S["jarr"]; aid = J["aid"]; dof = J["dof"]; qad = J["qad"] if "qad" in J else [m.jnt_qposadr[m.dof_jntid[k]] for k in dof]
    tau_v = np.abs(m.actuator_ctrlrange[aid][:, 1]); om_v = J["om"]; dt = m.opt.timestep; alpha = 1.0 - np.exp(-dt/T.LAG_TAU)
    act = np.array(d.ctrl[aid], float); bb = S["ball_body"]; rg = S["robot_geoms"]; fl = S["floor"]; footg = R_["footg"]
    sup_b = R_["fbs"][S["support"][0]]; x0 = d.xpos[sup_b][:2].copy(); travel = 0.0; pel_min = 9.0; nonfoot = False; nc = 0; lim = (0.0, -1); frames = []
    qt = None; kc = None; snap = None; t_fall = None; n_cont = int(round(CONT_T/dt)); R_["sw_frozen"] = None; R_["sw_q"] = None; R_["ik_k"] = 0; t_td = None
    for s in range(int(round(T_/dt))):
        if s == n_cont: snap = (d.qpos.copy(), d.qvel.copy())       # the ball flight starts here (t_rel + CONT_T, as in R1): the robot model has no goal
        if nonfoot and t_fall is None: t_fall = s*dt
        if t_fall is not None and s >= n_cont: break                 # fallen (non-foot floor contact): stop the robot once the flight state is taken
        t = s*dt; fz = 0.0
        fsw_ = 0.0
        for ci in range(d.ncon):
            c = d.contact[ci]; g1, g2 = c.geom1, c.geom2
            if fl in (g1, g2):
                o = g2 if g1 == fl else g1
                if o in footg:
                    fv = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fv); fz += fv[0]
                    if "sw_geoms" in R_ and o in R_["sw_geoms"]: fsw_ += fv[0]
                elif o in rg: nonfoot = True
            if bb in (int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])) and (g1 in rg or g2 in rg): nc += 1
        if REC_MODE == "step" and t_td is None and fsw_ > 1.0 and R_.get("sw_q") is not None: t_td = t; R_["sw_frozen"] = d.qpos[R_["sw_qa"]].copy()   # the foot stays where it landed
        k_c = int(round(t/B.CTRL_DT, 6))
        if k_c != kc: kc = k_c; qt = _rec_targets(S, R_, d, fz > 1.0)
        q = d.qpos[qad]; w = d.qvel[dof]
        cmd = R_["a_hold"] + R_["kp"]*(qt[qad] - q) - R_["kd"]*w
        act += alpha*(cmd - act)
        frac = np.maximum(0.0, 1.0 - np.abs(w)/om_v); hi = np.where(w > 0, tau_v*frac, tau_v); lo = np.where(w < 0, tau_v*frac, tau_v)
        d.ctrl[aid] = np.clip(act, -lo, hi)
        BI.backemf_brake(m, d, S); mujoco.mj_step(m, d)
        travel = max(travel, float(np.linalg.norm(d.xpos[sup_b][:2] - x0))); pel_min = min(pel_min, float(d.xpos[R_["pel"]][2]))
        if s % 5 == 0:
            ov = lim_overrun(S, d.qpos)
            if ov[0] > lim[0]: lim = ov
        if rec_every and s % rec_every == 0: frames.append(d.qpos.copy())
    up = float(d.xmat[R_["bc"].tb][8]); zc = float(d.subtree_com[R_["pel"]][2])
    fell = nonfoot or up < 0.85 or zc < 0.6*R_["z_st"]
    dv = S["ball_dof"]
    if snap is None: snap = (d.qpos.copy(), d.qvel.copy())
    return dict(frames=frames, qpos=snap[0], qvel=snap[1], n_contact=nc, fell=bool(fell), nonfoot=bool(nonfoot), t_fall=t_fall, t_touchdown=t_td, step_xy=(None if R_.get("sw_tgt") is None else (np.asarray(R_["sw_tgt"]) - R_["sole"](d, R_["fsu"])[:2]).tolist()) if REC_MODE == "step" else None, up_end=up, z_com_end=zc, z_com_rel=zc/R_["z_st"],
                travel=travel, pelvis_min=pel_min, lim_over=lim[0], lim_joint=(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(lim[1])) if lim[1] >= 0 else None),
                ball_speed=float(np.linalg.norm(d.qvel[dv:dv + 3])), clear=None)


# ------------------------------------------------------------------ ball flight (ball-only MuJoCo model)
_FM = {}
def goal_frame(S, demand):
    qw, qx, qy, qz = np.asarray(S["q_settled"])[3:7]; yaw0 = float(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))
    g = demand["goal"]; b0 = np.asarray(S["q_settled"])[S["ball_qadr"]:S["ball_qadr"] + 3]; a = yaw0 + float(g["az"])
    n = np.array([np.cos(a), np.sin(a)]); t = np.array([-np.sin(a), np.cos(a)]); c = b0[:2] + float(g["D"])*n
    return c, n, t


def flight_model(S, demand):
    m = S["m"]; gb_ = S["ball_geom"]; bb_ = S["ball_body"]
    key = (id(m), float(m.opt.timestep), json.dumps(demand["goal"], sort_keys=True, default=float), tuple(np.round(m.geom_friction[gb_], 8)), round(float(m.body_mass[bb_]), 8))   # perturbed plants carry their own ball
    if key in _FM: return _FM[key]
    fl = S["floor"]; gb = S["ball_geom"]; bb = S["ball_body"]; c, n, t = goal_frame(S, demand); g = demand["goal"]
    def gp(i): return f'friction="{" ".join(map(str, m.geom_friction[i]))}" solref="{" ".join(map(str, m.geom_solref[i]))}" solimp="{" ".join(map(str, m.geom_solimp[i]))}" condim="{int(m.geom_condim[i])}" priority="{int(m.geom_priority[i])}" solmix="{float(m.geom_solmix[i])}" margin="{float(m.geom_margin[i])}" gap="{float(m.geom_gap[i])}"'
    W, H, pr = float(g["width"]), float(g["height"]), float(g["post_r"])
    pL = c + (W/2 + pr)*t; pR = c - (W/2 + pr)*t
    integ = {0: "Euler", 1: "RK4", 2: "implicit", 3: "implicitfast"}[int(m.opt.integrator)]; cone = {0: "pyramidal", 1: "elliptic"}[int(m.opt.cone)]
    I = m.body_inertia[bb]; r = float(m.geom_size[gb][0])
    xml = f"""<mujoco><option timestep="{m.opt.timestep}" integrator="{integ}" cone="{cone}" iterations="{m.opt.iterations}" impratio="{m.opt.impratio}" density="{AIR['density']}" viscosity="{AIR['viscosity']}" gravity="{' '.join(map(str, m.opt.gravity))}"/>
<worldbody><geom name="floor" type="plane" size="0 0 0.05" {gp(fl)}/>
<body name="ball"><freejoint name="ball_free"/><inertial pos="0 0 0" mass="{float(m.body_mass[bb])}" diaginertia="{I[0]} {I[1]} {I[2]}"/><geom name="ball" type="sphere" size="{r}" mass="0" fluidshape="ellipsoid" {gp(gb)}/></body>
<body name="goal"><geom type="cylinder" fromto="{pL[0]} {pL[1]} 0 {pL[0]} {pL[1]} {H + 2*pr}" size="{pr}" solref="0.002 1"/><geom type="cylinder" fromto="{pR[0]} {pR[1]} 0 {pR[0]} {pR[1]} {H + 2*pr}" size="{pr}" solref="0.002 1"/>
<geom type="cylinder" fromto="{pL[0]} {pL[1]} {H + pr} {pR[0]} {pR[1]} {H + pr}" size="{pr}" solref="0.002 1"/></body></worldbody></mujoco>"""
    fm = mujoco.MjModel.from_xml_string(xml); _FM[key] = (fm, c, n, t); return _FM[key]


def ball_flight(S, demand, qpos, qvel, record_every=10):
    """Continue the ball (only) in MuJoCo from its state at t_rel until it crosses the goal plane, stops, or FLIGHT_T.
    Goal = the ball centre crosses the plane through the posts with |lateral| < W/2 and height < H (posts and bar deflect physically)."""
    fm, c, n, t = flight_model(S, demand); g = demand["goal"]; W, H = float(g["width"]), float(g["height"])
    fd = mujoco.MjData(fm); qa = S["ball_qadr"]; dv = S["ball_dof"]; fd.qpos[:] = qpos[qa:qa + 7]; fd.qvel[:] = qvel[dv:dv + 6]; mujoco.mj_forward(fm, fd)
    def sc(p): return float(np.dot(p[:2] - c, n)), float(np.dot(p[:2] - c, t)), float(p[2])
    s0, u0, z0 = sc(fd.qpos[:3]); path = [fd.qpos[:3].tolist()]; nstep = max(1, int(round(0.005/fm.opt.timestep))); tt = 0.0; crossed = None
    while tt < FLIGHT_T:
        mujoco.mj_step(fm, fd, nstep); tt += nstep*fm.opt.timestep; s1, u1, z1 = sc(fd.qpos[:3])
        if len(path) < 2000 and int(round(tt/fm.opt.timestep)) % (nstep*record_every) == 0: path.append(fd.qpos[:3].tolist())
        if s0 < 0 <= s1:
            f = -s0/max(s1 - s0, 1e-12); crossed = (u0 + f*(u1 - u0), z0 + f*(z1 - z0), tt - (1 - f)*nstep*fm.opt.timestep); break
        s0, u0, z0 = s1, u1, z1
        if np.linalg.norm(fd.qvel[:3]) < 0.02 and fd.qpos[2] < 0.2: break
    path.append(fd.qpos[:3].tolist())
    if crossed is not None:
        u, z, tc = crossed; inside = abs(u) < W/2 and z < H
        margin = min(W/2 - abs(u), H - z); miss = max(0.0, abs(u) - W/2) + max(0.0, z - H)
        return dict(goal=bool(inside), crossed=True, u=u, z=z, t_cross=tc, margin=float(margin), miss=float(miss), stop_s=None, path=path)
    s, u, z = sc(fd.qpos[:3])
    return dict(goal=False, crossed=False, u=u, z=z, t_cross=None, margin=-float(-s), miss=float(-s + max(0.0, abs(u) - W/2)), stop_s=float(s), path=path)


# ------------------------------------------------------------------ prediction input: ball speed needed (no robot, no search)
def need():
    task = GoalKickTask("G1"); out = {}
    for cfg in CONFIGS:
        dem = demand_for(cfg); S = task.plant(dem, core.DT_OPT); qa = S["ball_qadr"]; dv = S["ball_dof"]; rows = []
        for el in (0, 5, 10, 15, 20, 25, 30):
            for v in np.arange(0.5, 30.01, 0.25):
                q = np.array(S["q_settled"]); qv = np.zeros(S["m"].nv); n = flight_model(S, dem)[2]
                a = np.arctan2(n[1], n[0]); qv[dv:dv + 3] = v*np.array([np.cos(R(el))*np.cos(a), np.cos(R(el))*np.sin(a), np.sin(R(el))])
                f = ball_flight(S, dem, q, qv); rows.append(dict(elev=el, v=float(v), goal=f["goal"], z=f["z"], t=f["t_cross"]))
        summ = {}
        for el in (0, 5, 10, 15, 20, 25, 30):
            ok = [r["v"] for r in rows if r["elev"] == el and r["goal"]]
            summ[el] = dict(v_min=min(ok) if ok else None, v_max=max(ok) if ok else None)
        az_tol = float(np.degrees(np.arctan((GOAL["width"]/2)/CONFIGS[cfg]["D"])))
        out[cfg] = dict(by_elev=summ, az_half_window_deg=az_tol); print(cfg, json.dumps(out[cfg]), flush=True)
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(f"{OUT}/speed_needed.json", "w"), indent=1); return out


# ------------------------------------------------------------------ search
RDRAWS = int(os.environ.get("KICK_RDRAWS", "0"))
def search(cfg, seed, iters, pop, out_dir=OUT, warm=None, sigma=0.10, tag=""):
    """CMA-ES. Score = worse-of (1 ms cached plant, 0.5 ms fresh plant) [+ RDRAWS fixed perturbation draws on fresh plants,
    alternating 1 ms / 0.5 ms]. warm = a ledger path whose z is the start point (the start point is always a candidate)."""
    import cma
    T.LAG_TAU = LAG; task = GoalKickTask("G1"); dem = demand_for(cfg); S = task.plant(dem); ref = task.ref(S, dem)
    n = task.n_params(S, dem)
    if warm: z0 = np.asarray(json.load(open(warm))["z"], float)
    else: z0 = np.zeros(n); z0[len(S["types"])*T.NK] = 2*(ref["takeoff_ref"] - T.T_REL_MIN)/(T.HOR - T.T_REL_MIN) - 1
    S2 = task.plant(dem, core.DT_CHECK, fresh=True)
    draws = [(perturbed_plant(task, dem, (core.DT_OPT, core.DT_CHECK)[k % 2], draw_pert(500000 + 1000*seed, k, n), uid=60000 + 10*seed + k), draw_pert(500000 + 1000*seed, k, n)["zn"]) for k in range(RDRAWS)]
    name = f"{cfg}{tag}_s{seed}"; log = open(f"{out_dir}/{name}.log", "a"); t0 = time.time()
    def obj(z):
        recs = [task.run(S, dem, z), task.run(S2, dem, z)] + [task.run(Sk, dem, np.clip(z + zn, -1, 1)) for Sk, zn in draws]
        sc = [task.score(r, dem) for r in recs]
        if all(x < 100.0 for x in sc): return max(sc)
        return 100.0 + sum(x - 100.0 for x in sc if x >= 100.0)
    es = cma.CMAEvolutionStrategy(z0, sigma, {"bounds": [-1, 1], "seed": seed, "verbose": -9, "maxiter": iters, "popsize": pop})
    best = (obj(z0), np.array(z0)); log.write(f"{time.time() - t0:.0f}s start {best[0]:.4f} (warm {warm}, rdraws {RDRAWS}, phys {PHYS})\n"); log.flush()
    while not es.stop():
        X = es.ask(); F = [obj(z) for z in X]; es.tell(X, F); k = int(np.argmin(F))
        if F[k] < best[0]: best = (F[k], np.array(X[k]))
        if es.countiter % 5 == 0: log.write(f"{time.time() - t0:.0f}s it {es.countiter} best {best[0]:.4f}\n"); log.flush()
    z = best[1]; out = TK.measure(task, dem, z); secs = time.time() - t0
    p = task.unpack(S, dem, z); rec = task.rollout(S, dem, p, ctrl_fn=task.controller(S, dem, p), record=True)
    em = EM.analyze(S, rec, dem["effector"]) if rec.get("trace") else None
    led = dict(job="M1_kick_goal", cfg=cfg, seed=seed, robot="G1", phys=PHYS, warm=warm, rdraws=RDRAWS,
               supplied=supplied(dem, iters, pop), prior_steps=dict(windup=ref["windup_steps"], path=ref["path_steps"], prep=ref.get("prep_steps")),
               result=dict(goal_both_dt=bool(out["ok_all"]), score=float(best[0]), meas=[dict(dt=m_["dt"], valid=bool(m_["valid"]), reasons=m_["reasons"],
                           **{k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else v) for k, v in m_["extra"].items() if k.startswith(("goal_", "rel_", "ball_", "slip", "eff_speed", "dmin", "t_rel", "feet_bw", "overspeed", "com_speed", "cont_", "rec_", "lim_"))}) for m_ in out["meas"]],
                           amp=p.get("amp"), tscale=p.get("tscale"), secs=round(secs)),
               emerged=em, flight_path=rec.get("flight_path"), z=np.asarray(z).tolist())
    json.dump(led, open(f"{out_dir}/{name}.json", "w"), indent=1, default=float)
    mm = led["result"]["meas"]; print(f"[{name}] goal(both dt) {out['ok_all']} | " + " | ".join(f"dt {x['dt']}: {x['valid']} {x['reasons']} v {x['ball_speed']} u {x.get('goal_u')} z {x.get('goal_z')} slip {x.get('slip')}" for x in mm) + f" | {secs:.0f} s", flush=True)
    return led


def supplied(dem, iters, pop):
    return dict(body="stock Menagerie G1 (MJX model used by the construct campaigns), unchanged geometry/masses/ranges/speeds",
                ratings="min(MJCF peak, Unitree URDF effort, vendor published joint max 120 N m): knee 120, ankle pitch/roll 35, waist roll/pitch 35, hip roll 88, hip pitch/yaw 88, waist yaw 88, shoulders/elbows 25, wrists 5 N m; applied as the torque envelope of driven joints and the force limit of held servos",
                lag=f"first-order actuator lag {LAG*1000:.0f} ms (throw.LAG_TAU), torque-speed droop and back-EMF brake (construct plant)",
                contact=("robot-floor: Menagerie MJX explicit pairs (foot solref 0.008); ball geom solref 0.002, robot/floor geoms 0.02 -> MIXED ball-floor and ball-robot solref 0.011 (default solmix; near-inelastic, no bounce); posts 0.002; ball condim 6, rolling friction %.4f m (c_rr 0.05), torsional %.3f m; effector collider = 4 cm sphere at the foot site (ImpactTask)" % (ROLL, TORSION)) + ("; PHYS 2: the ball collides with EVERY robot collision geom (group 3: shins, ankle braces, feet, thighs, hips, pelvis, torso, head, arms, hands; contype 2 / conaffinity 0 so no new robot-robot or robot-floor contact)" if PHYS >= 2 else "; PHYS 1: the ball collides ONLY with the kicking-foot body geoms (shin/brace transparent)"),
                ball="0.43 kg, r 0.11 m, hollow-shell inertia, on the floor at (0.30, -0.17) in front of the right foot (kick_ball placement)",
                goal=dict(dem["goal"], az_deg=float(np.degrees(dem["goal"]["az"])), note="D = distance from the ball's start to the goal plane along the bearing az; inner mouth 2.0 x 1.0 m; posts/bar cylinders r 0.06 m outside the mouth"),
                flight=(("robot + ball continued together for %.2f s after t_rel (last driven command held, servos at targets; any robot-ball contact = BALL_RECONTACT), then " % CONT_T) if PHYS >= 2 else "robot removed at t_rel, then ") + "ball-only MuJoCo continuation (requires no robot contact and separating velocity at t_rel), same dt and contact params, air drag via MuJoCo ellipsoid fluid model (rho 1.2, mu 1.8e-5), cap %.0f s" % FLIGHT_T,
                support="left foot (right-foot kick): ImpactTask stance gates + support-foot horizontal travel <= %.2f m" % SLIP_MAX,
                scaffold="kick_ball construction: body-derived wind-up -> release reference for effector direction (elev 15 deg [kick_ball's value], az = goal bearing), tracker omega 40, tscale range 0.9-1.45, 21 driven joints",
                objective=("valid (all gates + goal at both dt) first; among valid, maximise ball speed (power-shot side study)" if dem.get("goal_objective") == "speed" else ("valid first; among valid, maximise min(mouth margin min(W/2-|u|, H-z) [m], (0.02 - slip)/0.02) + 0.01 x ball speed (gate-margin objective)" if PHYS >= 2 else "valid (all gates + goal at both dt) first; among valid, maximise the mouth margin min(W/2-|u|, H-z) + 0.01 x ball speed")),
                search=dict(score="worse-of over (1 ms, 0.5 ms fresh)" + (" + %d fixed perturbation draws on fresh plants (alternating dt)" % RDRAWS if RDRAWS else ""), cma_seed_rule="seeds >= 1"),
                budget=dict(iters=iters, pop=pop, evals=iters*pop))


# ------------------------------------------------------------------ robustness: 4 perturbation draws x 2 dt, fresh plants
PERT = dict(joint_mrad=2.0, ball_xy_m=0.005, ball_mass_rel=0.03, roll_rel=0.20, z_sigma=0.02)
def draw_pert(seed_base, k, n_z):
    rng = np.random.default_rng(seed_base + 17*k + 1)
    return dict(zn=PERT["z_sigma"]*rng.standard_normal(n_z), jn=rng.standard_normal(64), bxy=PERT["ball_xy_m"]*rng.standard_normal(2),
                mr=1 + PERT["ball_mass_rel"]*rng.uniform(-1, 1), rr=1 + PERT["roll_rel"]*rng.uniform(-1, 1))


def perturbed_plant(task, dem, dt, pd, uid, nominal_ref=True):
    """A FRESH plant with one perturbation draw applied (start joints, ball xy/mass/rolling friction); mj_setConst after the mass
    change. The solution's reference is the nominal plant's (as in the search)."""
    S = task.fresh_plant(dem, dt, uid=uid); m = S["m"]
    if nominal_ref:
        nom = task.plant(dem, dt, fresh=(dt != core.DT_OPT)); dr = dem["direction"]
        task._refs[(id(m), tuple(np.round(dr, 4)), dem.get("capacity_mode", "power"))] = task.ref(nom, dem)
    q = np.array(S["q_settled"]); nqr = S["ball_qadr"]; q[7:nqr] += 1e-3*PERT["joint_mrad"]*pd["jn"][:nqr - 7]; q[nqr:nqr + 2] += pd["bxy"]; S["q_settled"] = q
    bb = S["ball_body"]; m.body_mass[bb] *= pd["mr"]; m.body_inertia[bb] *= pd["mr"]; m.geom_friction[S["ball_geom"]][2] *= pd["rr"]
    if PHYS >= 2: d_ = mujoco.MjData(m); mujoco.mj_setConst(m, d_)
    return S


def robust(path, n_draws=None, seed_base=None, suffix="_robust"):
    T.LAG_TAU = LAG; led = json.load(open(path)); cfg = led["cfg"]; task = GoalKickTask("G1"); dem = demand_for(cfg); z = np.asarray(led["z"])
    n_draws = n_draws or N_ROBUST; sb = 1000*int(led["seed"]) if seed_base is None else seed_base
    rows = []
    for k in range(n_draws):
        pd = draw_pert(sb, k, len(z)); zk = np.clip(z + pd["zn"], -1, 1); res = []
        for i, dt in enumerate((core.DT_OPT, core.DT_CHECK)):
            S = perturbed_plant(task, dem, dt, pd, uid=100*k + i + 7*int(led["seed"]) + (0 if seed_base is None else 50000))
            r = task.run(S, dem, zk); ex = r["extra"]
            res.append(dict(dt=dt, valid=bool(r["valid"]), reasons=r["reasons"], ball_speed=round(float(r["value"]), 3), goal_u=ex.get("goal_u"), goal_z=ex.get("goal_z"), slip=round(float(ex["slip"]), 4), cont_contacts=ex.get("cont_contacts"),
                            **{k_: ex.get(k_) for k_ in ("lim_kick_deg", "lim_kick_joint", "lim_rec_deg", "rec_fell", "rec_t_fall", "rec_travel", "rec_t_touchdown") if k_ in ex}))
        rows.append(dict(draw=k, perturbation=dict(ball_dxy=pd["bxy"].tolist(), mass_x=pd["mr"], roll_x=pd["rr"], joint_mrad=PERT["joint_mrad"], z_sigma=PERT["z_sigma"]), both_dt=all(x["valid"] for x in res), runs=res))
        print(f"[{cfg} s{led['seed']} draw {k}] both {rows[-1]['both_dt']} " + " | ".join(f"{x['dt']}: {x['valid']} {x['reasons']} v {x['ball_speed']} u {x['goal_u']}" for x in res), flush=True)
    npass = sum(r["both_dt"] for r in rows)
    out = dict(ledger=os.path.basename(path), cfg=cfg, seed=led["seed"], phys=PHYS, perturbations=PERT, seed_base=sb, draws=rows, n_pass=npass, n=n_draws, robust=bool(npass >= 0.75*n_draws - 1e-9))
    json.dump(out, open(path.replace(".json", f"{suffix}.json"), "w"), indent=1, default=float); return out


def regate(path, out_dir):
    """Round-0 ledger under the repaired physics: nominal on FRESH plants at 1 ms and 0.5 ms + N_ROBUST new draws -> <out_dir>/<name>_regate.json"""
    T.LAG_TAU = LAG; led = json.load(open(path)); task = GoalKickTask("G1"); dem = demand_for(led["cfg"]); z = np.asarray(led["z"]); nom = []
    for i, dt in enumerate((core.DT_OPT, core.DT_CHECK)):
        S = task.fresh_plant(dem, dt, uid=90000 + i); m = S["m"]; nomS = task.plant(dem, dt, fresh=(i > 0))
        task._refs[(id(m), tuple(np.round(dem["direction"], 4)), dem.get("capacity_mode", "power"))] = task.ref(nomS, dem)
        r = task.run(S, dem, z); ex = r["extra"]
        nom.append(dict(dt=dt, valid=bool(r["valid"]), reasons=r["reasons"], ball_speed=round(float(r["value"]), 3), goal_u=ex.get("goal_u"), goal_z=ex.get("goal_z"), slip=round(float(ex["slip"]), 4),
                        cont_contacts=ex.get("cont_contacts"), cont_clear=ex.get("cont_clear"), rel_sep_speed=ex.get("rel_sep_speed")))
    print(led["cfg"], led["seed"], json.dumps(nom, default=float), flush=True)
    import shutil; os.makedirs(out_dir, exist_ok=True); dst = f"{out_dir}/{os.path.basename(path)}"; shutil.copy(path, dst)
    nd = int(os.environ.get("KICK_REGATE_DRAWS", N_ROBUST))
    rb = robust(dst, n_draws=nd, seed_base=7000 + 1000*int(led["seed"]), suffix="_regate_draws") if nd > 0 else dict(n_pass=None, n=0, robust=None)
    out = dict(ledger=os.path.basename(path), cfg=led["cfg"], seed=led["seed"], phys=PHYS, nominal_fresh=nom, nominal_both=all(x["valid"] for x in nom), n_pass=rb["n_pass"], n=rb["n"], robust=rb["robust"])
    json.dump(out, open(dst.replace(".json", "_regate.json"), "w"), indent=1, default=float); return out


def technique(path):
    """Replay a ledger at 1 ms (nominal plant, recorded) and analyse the chain/ownership AT IMPACT (first step the ball moves > 0.5 m/s)
    as well as at t_rel; writes <ledger>_technique.json."""
    T.LAG_TAU = LAG; L = json.load(open(path)); task = GoalKickTask("G1"); dem = demand_for(L["cfg"]); S = task.plant(dem); z = np.asarray(L["z"])
    p = task.unpack(S, dem, z); rec = task.rollout(S, dem, p, ctrl_fn=task.controller(S, dem, p), record=True); tr = rec["trace"]
    qv = np.asarray(tr["qvel"]); dv = S["ball_dof"]; sp = np.linalg.norm(qv[:, dv:dv + 3], axis=1); hit = np.nonzero(sp > 0.5)[0]
    qp, qvf = rec["final_state"]; w = np.asarray(qvf[dv + 3:dv + 6]); vb = np.asarray(qvf[dv:dv + 3]); r_ = BALL["radius"]
    qa = S["ball_qadr"]; Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, np.asarray(qp[qa + 3:qa + 7])); ww = Rm.reshape(3, 3) @ w      # free-joint angular velocity is in the body frame
    vh = vb.copy(); vh[2] = 0.0; e = vh/max(np.linalg.norm(vh), 1e-9); v_contact = vh + np.cross(ww, np.array([0, 0, -r_]))         # floor-contact point velocity (0 = pure rolling)
    spin = dict(ball_v=vb.tolist(), ball_w_world=ww.tolist(), slip_speed_at_floor=float(np.dot(v_contact, e)), roll_ratio=float(-np.dot(np.cross(ww, np.array([0, 0, -r_])), e)/max(np.linalg.norm(vh), 1e-9)))   # 1 = pure rolling, 0 = no spin, < 0 = backspin
    out = dict(ledger=os.path.basename(path), spin_at_release=spin, replay_valid=bool(rec["valid"]), replay_reasons=rec["reasons"], ball_speed=float(rec["value"]), at_release=EM.analyze(S, rec, dem["effector"]))
    if len(hit):
        k = int(hit[0]); trk = {kk: (v[:k] if isinstance(v, list) else v) for kk, v in tr.items()}; out["impact_t"] = k*float(tr["dt"])
        out["at_impact"] = EM.analyze(S, dict(rec, trace=trk), dem["effector"]); out["foot_speed_at_impact"] = out["at_impact"]["speed"]
        out["ball_to_foot_speed"] = float(rec["value"])/max(out["foot_speed_at_impact"], 1e-9)
        dt_ = float(tr["dt"]); bv = qv[:, dv:dv + 3]; acc = np.linalg.norm(np.diff(bv, axis=0), axis=1)/dt_; cont = np.nonzero(acc > 30.0)[0]     # ball pushed by the robot (rolling/air decel << 30 m/s^2)
        m_ = S["m"]; d_ = mujoco.MjData(m_); v6 = np.zeros(6); fs = []
        for i in range(len(qv)):
            d_.qpos[:] = tr["q"][i]; d_.qvel[:] = qv[i]; mujoco.mj_forward(m_, d_); mujoco.mj_objectVelocity(m_, d_, mujoco.mjtObj.mjOBJ_SITE, S["site"], v6, 0); fs.append(float(np.linalg.norm(v6[3:])))
        fs = np.asarray(fs); out["foot_speed_peak"] = float(fs.max()); out["foot_speed_peak_t"] = float(np.argmax(fs)*dt_)
        if len(cont): out["contact_window"] = [float(cont[0]*dt_), float((cont[-1] + 1)*dt_)]; out["contact_s"] = float((cont[-1] + 1 - cont[0])*dt_); out["foot_speed_peak_in_contact"] = float(fs[cont[0]:cont[-1] + 2].max())
    json.dump(out, open(path.replace(".json", "_technique.json"), "w"), indent=1, default=float); return out


def render_goal(path, out, W_=960, H_=540, fps=30):
    """Video of a ledger: the kick (full robot + ball simulation at the plant dt) then the simulated ball flight (robot frozen at its
    t_rel pose: the flight model has no robot). Goal posts/bar drawn as visual geoms. Camera between robot and ball."""
    import subprocess, tempfile, shutil
    from PIL import Image, ImageDraw, ImageFont
    T.LAG_TAU = LAG; L = json.load(open(path)); task = GoalKickTask("G1"); dem = demand_for(L["cfg"]); S = task.plant(dem); z = np.asarray(L["z"])
    p = task.unpack(S, dem, z); rec = task.rollout(S, dem, p, ctrl_fn=task.controller(S, dem, p), record=True); tr = rec["trace"]; dt = float(tr["dt"])
    qpos, qvel = rec["final_state"]; fm, c, n, t = flight_model(S, dem); fd = mujoco.MjData(fm); qa = S["ball_qadr"]; dv = S["ball_dof"]; Qc = []
    if PHYS >= 2:          # the 0.3 s robot+ball continuation (last command held), shown at 1/4 speed like the kick, then the flight from its end state
        ctrl = getattr(task, "_last_ctrl", None); cont = continuation(S, qpos, qvel, ctrl, rec_every=max(1, int(round(1.0/fps/dt/4)))); Qc = cont["frames"]; qpos, qvel = cont["qpos"], cont["qvel"]
    fd.qpos[:] = qpos[qa:qa + 7]; fd.qvel[:] = qvel[dv:dv + 6]; mujoco.mj_forward(fm, fd); fl = ball_flight(S, dem, qpos, qvel)
    t_end = (fl["t_cross"] or 3.0) + 0.6; Qf = []
    for k in range(int(t_end/(1.0/fps))):
        Qf.append(np.concatenate([qpos[:qa], fd.qpos[:7]])); mujoco.mj_step(fm, fd, int(round(1.0/fps/fm.opt.timestep)))
    Qk = np.asarray(tr["q"])[::max(1, int(round(1.0/fps/dt/4)))]          # the kick at 1/4 speed
    spec = mujoco.MjSpec.from_file(task.scene(dem)); g = dem["goal"]; W, H, pr = float(g["width"]), float(g["height"]), float(g["post_r"])
    pL = c + (W/2 + pr)*t; pR = c - (W/2 + pr)*t
    for a_, b_ in ((np.r_[pL, 0], np.r_[pL, H + 2*pr]), (np.r_[pR, 0], np.r_[pR, H + 2*pr]), (np.r_[pL, H + pr], np.r_[pR, H + pr])):
        gg = spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, fromto=list(a_) + list(b_), size=[pr, 0, 0], rgba=[0.95, 0.95, 0.95, 1]); gg.contype = 0; gg.conaffinity = 0
    m = spec.compile(); m.vis.global_.offwidth = W_; m.vis.global_.offheight = H_; rd = mujoco.Renderer(m, H_, W_); d = mujoco.MjData(m)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.azimuth = -60; cam.elevation = -14
    try: fb = ImageFont.truetype("DejaVuSans-Bold.ttf", 22); fs_ = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception: fb = fs_ = ImageFont.load_default()
    tmp = tempfile.mkdtemp(prefix="m1_render_"); frames = []; mm = L["result"]["meas"][0]
    title = f"Stock G1 kicks a 0.43 kg ball into a 2 x 1 m goal at {g['D']:.0f} m, {np.degrees(g['az']):.0f} deg"
    sub = f"ball {mm.get('ball_speed', 0):.1f} m/s | goal at 1 ms and 0.5 ms | 10 ms lag, URDF/vendor ratings | " + ("whole-body ball contact; robot+ball simulated to t_rel+0.3 s, then robot frozen" if PHYS >= 2 else "flight simulated in MuJoCo (robot frozen after t_rel)")
    allQ = [(q_, "kick (1/4 speed)") for q_ in Qk] + [(q_, "kick: 0.3 s continuation, robot + ball (1/4 speed)") for q_ in Qc] + [(q_, "ball flight (real time)") for q_ in Qf]
    Qk = list(Qk) + list(Qc)
    for k, (q_, lab) in enumerate(allQ):
        d.qpos[:len(q_)] = q_; mujoco.mj_forward(m, d); rb = np.asarray(q_[:2]); bl = np.asarray(q_[qa:qa + 2]); cen = 0.5*(rb + c) if lab.startswith("ball") else rb
        cam.lookat[:] = [cen[0], cen[1], 0.6]; cam.distance = 3.2 if lab.startswith("kick") else max(3.2, 0.75*np.linalg.norm(c - rb) + 2.5)
        rd.update_scene(d, camera=cam); im = Image.fromarray(rd.render()); dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, W_, 56], fill=(10, 16, 26)); dr.text((12, 5), title, fill=(240, 244, 250), font=fb); dr.text((12, 33), sub, fill=(170, 186, 205), font=fs_)
        dr.rectangle([0, H_ - 28, W_, H_], fill=(10, 16, 26)); dr.text((12, H_ - 23), lab + ("   GOAL" if lab.startswith("ball") and fl["goal"] and k - len(Qk) >= (fl["t_cross"] or 1e9)*fps else ""), fill=(255, 214, 140), font=fs_)
        im.save(f"{tmp}/f{k:04d}.png"); frames.append(f"{tmp}/f{k:04d}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{tmp}/f%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", out], check=True)
    pick = [frames[int(round(j))] for j in np.linspace(0, len(frames) - 1, 6)]; ims = [Image.open(p_).resize((W_//2, H_//2)) for p_ in pick]
    cs = Image.new("RGB", (3*W_//2, 2*H_//2))
    for j, im in enumerate(ims): cs.paste(im, ((j % 3)*W_//2, (j//3)*H_//2))
    cs.save(os.path.splitext(out)[0] + "_sheet.png"); shutil.rmtree(tmp); return out


def summarize():
    import glob; rows = []
    for f in sorted(glob.glob(f"{OUT}/g*_s*.json")):
        import re
        if not re.search(r"_s\d+\.json$", f): continue
        L = json.load(open(f)); rb = f.replace(".json", "_robust.json"); RB = json.load(open(rb)) if os.path.exists(rb) else None
        mm = L["result"]["meas"]; e = L.get("emerged") or {}
        rows.append(dict(cfg=L["cfg"], seed=L["seed"], goal_both_dt=L["result"]["goal_both_dt"], reasons=[x["reasons"] for x in mm],
                         ball_speed=[x.get("ball_speed") for x in mm], elev_deg=[round(float(np.degrees(x.get("ball_elev", 0))), 1) for x in mm], az_deg=[round(float(np.degrees(x.get("ball_az", 0))), 1) for x in mm],
                         cross_u=[x.get("goal_u") for x in mm], cross_z=[x.get("goal_z") for x in mm], t_cross=[x.get("goal_t_cross") for x in mm], margin=[x.get("goal_margin") for x in mm], slip=[x.get("slip") for x in mm],
                         robust=None if RB is None else f"{RB['n_pass']}/{RB['n']}", robust_ok=None if RB is None else RB["robust"],
                         peaks=e.get("peak_omega"), order=e.get("peak_order"), ownership=e.get("ownership_group"), ownership_base=e.get("ownership_base"), beyond_human=e.get("beyond_human"), W_pos=e.get("W_pos"),
                         amp=L["result"].get("amp"), tscale=L["result"].get("tscale"), secs=L["result"]["secs"], evals=L["supplied"]["budget"]["evals"]))
    json.dump(rows, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
    for r in rows: print(json.dumps(r, default=float))
    return rows


if __name__ == "__main__":
    T.LAG_TAU = LAG; os.makedirs(OUT, exist_ok=True); cmd = sys.argv[1]
    if cmd == "need": need()
    elif cmd == "search": search(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 150, int(sys.argv[5]) if len(sys.argv) > 5 else 16)
    elif cmd == "search_r1":     # search_r1 <cfg> <seed> <iters> <pop> <out_dir> [warm_ledger] [sigma]
        search(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), out_dir=sys.argv[6], warm=(sys.argv[7] if len(sys.argv) > 7 and sys.argv[7] != "-" else None), sigma=float(sys.argv[8]) if len(sys.argv) > 8 else 0.10)
    elif cmd == "robust": robust(sys.argv[2])
    elif cmd == "confirm":       # confirm <ledger>: independent 16-draw set (R2 seed base 910000 + 1000 seed, used by no search)
        L_ = json.load(open(sys.argv[2])); robust(sys.argv[2], seed_base=910000 + 1000*int(L_["seed"]), suffix="_confirm")
    elif cmd == "regate": regate(sys.argv[2], sys.argv[3])
    elif cmd == "summarize": summarize()
    elif cmd == "render": render_goal(sys.argv[2], sys.argv[3])
    elif cmd == "technique":
        for f_ in sys.argv[2:]: o_ = technique(f_); print(f_, o_["replay_valid"], o_.get("impact_t"), o_.get("foot_speed_at_impact"), (o_.get("at_impact") or {}).get("ownership_group"), (o_.get("at_impact") or {}).get("peak_order"), flush=True)
