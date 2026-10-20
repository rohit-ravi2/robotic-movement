"""Phase 14: feedback balance controller for G1 landing (and standing), following this structure:
  (1) keep the trunk line (crown -> navel) vertical, (2) keep the feet parallel to the floor, (3) crouch on landing
  to absorb the shock, plus the dynamic refinement: keep the CAPTURE POINT (COM + COM velocity / omega0) over the
  centre of the feet, not just the COM.

Actuation: every joint is a stiff implicit position servo (the mechanism that holds the robot at rest in settle());
the controller sets servo TARGETS each step. Correction directions come from the robot's own kinematics with both
feet held fixed (constrained Jacobians), computed once at the standing pose, so no sign is hand-tuned.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, branch_i as BI, plant as P

G = 9.80665
# Per-robot joint roles. H1 (Phase 14b) has no ankle-roll joint (lateral capture-point correction unavailable; lateral
# balance is left to the held hip-roll servos + trunk-roll feedback) and no waist pitch.
JN = {"G1": dict(ap=("left_ankle_pitch_joint", "right_ankle_pitch_joint"), ar=("left_ankle_roll_joint", "right_ankle_roll_joint"),
                 kn=("left_knee_joint", "right_knee_joint"), hp=("left_hip_pitch_joint", "right_hip_pitch_joint"),
                 hr=("left_hip_roll_joint", "right_hip_roll_joint"), wp=("waist_pitch_joint",)),
      "H1": dict(ap=("left_ankle", "right_ankle"), ar=(), kn=("left_knee", "right_knee"), hp=("left_hip_pitch", "right_hip_pitch"),
                 hr=("left_hip_roll", "right_hip_roll"), wp=())}
# Landing-crouch direction per joint role (knee-normalised). G1: hand-set in Phase 14 (tuned controller, kept verbatim).
# H1: derived from the analytic bank reference (crouch frame vs sole-flat stand: hip -0.547, knee +1.013, ankle -0.457 rad).
CROUCH_DIR = {"G1": {"kn": 1.0, "hp": -0.75, "ap": -0.45}, "H1": {"kn": 1.0, "hp": -0.54, "ap": -0.45}}
JN["G1H"] = JN["G1"]; CROUCH_DIR["G1H"] = CROUCH_DIR["G1"]
DEFAULT = dict(stiff=250.0, k_cp=1.0, k_cp_d=0.0, k_trunk=1.0, k_trunk_d=0.0, k_foot=1.0,
               crouch=0.6, t_down=0.15, t_hold=0.25, t_up=0.6, pre_bend=0.15)


def _jid(m, n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)


def _act_of(m, jid): return [a for a in range(m.nu) if m.actuator_trnid[a][0] == jid][0]


def foot_points(S):
    """Where the feet are, per foot: (body id, site id or None, local offset). G1 has sole sites (left_foot/right_foot,
    used verbatim); a robot without them (H1) uses the centroid of its foot collision geoms in the foot-body frame."""
    m = S["m"]; out = []
    for fb in BI.ROBOTS[S.get("robot", "G1")]["foot_bodies"]:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, fb)
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_foot" if fb.startswith("left") else "right_foot")
        if sid >= 0: out.append((b, sid, None)); continue
        d = mujoco.MjData(m); d.qpos[:] = S["q0"]; mujoco.mj_forward(m, d)
        gs = [g for g in S["foot_geoms"] if m.geom_bodyid[g] == b]
        cen = np.mean([d.geom_xpos[g] for g in gs], 0); loc = d.xmat[b].reshape(3, 3).T @ (cen - d.xpos[b])
        out.append((b, None, loc))
    return out


def foot_xy(d, fp):
    b, sid, loc = fp
    return d.site_xpos[sid][:2] if sid is not None else (d.xpos[b] + d.xmat[b].reshape(3, 3) @ loc)[:2]


def constrained_maps(m, q, feet):
    """With both feet held still: d(COM_x, COM_y)/d(q_j) and d(trunk pitch, trunk roll)/d(q_j) for every hinge dof."""
    d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_forward(m, d)
    Jf = []
    for (b, sid, loc) in feet:
        a = np.zeros((3, m.nv)); r = np.zeros((3, m.nv))
        if sid is not None: mujoco.mj_jacSite(m, d, a, r, sid)
        else: mujoco.mj_jac(m, d, a, r, d.xpos[b] + d.xmat[b].reshape(3, 3) @ loc, b)
        Jf += [a, r]
    Jf = np.vstack(Jf)
    Jc = np.zeros((3, m.nv)); mujoco.mj_jacSubtreeCom(m, d, Jc, 0)
    tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    Jt = np.zeros((3, m.nv)); Jr = np.zeros((3, m.nv)); mujoco.mj_jacBody(m, d, Jt, Jr, tb)
    base = np.arange(6); out = {}
    for j in range(1, m.njnt):
        dof = m.jnt_dofadr[j]
        vb = -np.linalg.lstsq(Jf[:, base], Jf[:, dof], rcond=None)[0]
        com = Jc[:, base] @ vb + Jc[:, dof]; rot = Jr[:, base] @ vb + Jr[:, dof]
        out[j] = dict(com=com[:2], trunk=np.array([rot[1], rot[0]]))     # pitch about world y, roll about world x
    return out


class BalanceController:
    def __init__(self, S, q_stand, q_land=None, params=None):
        self.S = S; m = S["m"]; self.m = m; self.p = dict(DEFAULT, **(params or {}))
        self.q_stand = np.array(q_stand); self.q_land = np.array(q_land if q_land is not None else q_stand)
        self.robot = S.get("robot", "G1")
        self.j = {k: [_jid(m, n) for n in v] for k, v in JN[self.robot].items()}
        self.qa = {k: [m.jnt_qposadr[j] for j in v] for k, v in self.j.items()}
        self.feet = foot_points(S)
        maps = constrained_maps(m, self.q_stand, self.feet)
        # resolved gains: joint motion per unit of COM / trunk error (both legs move together)
        self.g_ap = 1.0/sum(maps[j]["com"][0] for j in self.j["ap"])          # rad per m of COM x
        self.g_ar = 1.0/sum(maps[j]["com"][1] for j in self.j["ar"]) if self.j["ar"] else None   # rad per m of COM y
        self.g_hp = 1.0/sum(maps[j]["trunk"][0] for j in self.j["hp"])        # rad per rad of trunk pitch
        self.g_hr = 1.0/sum(maps[j]["trunk"][1] for j in self.j["hr"])
        self.fbodies = [fp[0] for fp in self.feet]
        self.tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        # crouch direction: from the stand pose toward the deep-crouch shape (knees flex, hips flex, ankles dorsiflex)
        self.dq_crouch = CROUCH_DIR[self.robot]
        # foot-body tilt at the flat-sole stand: the sole-parallel target is RELATIVE to it on a robot whose foot
        # body is not level at stance (H1: 12 mrad roll). G1 keeps the Phase 14 absolute form (its tuned controller
        # was tuned on it; the stance offset there is 0.6 mrad).
        d0 = mujoco.MjData(m); d0.qpos[:] = self.q_stand; mujoco.mj_forward(m, d0)
        self.foot_tilt0 = [self._tilt(d0, fb) if fp[1] is None else np.zeros(2) for fb, fp in zip(self.fbodies, self.feet)]
        self.z_stand = float(d0.subtree_com[0][2])
        self.t_touch = None if self.p.get("mode", "land") == "land" else -1e9   # "stand": absorb already finished

    def _tilt(self, d, body):
        R = d.xmat[body].reshape(3, 3); z = R[:, 2]
        return np.array([np.arctan2(z[0], z[2]), -np.arctan2(z[1], z[2])])  # pitch (about y), roll (about x)

    def targets(self, t, d, in_contact):
        """Servo targets for every actuator (array indexed by actuator id)."""
        m = self.m; p = self.p; q = self.q_land.copy() if (self.t_touch is None) else self.q_stand.copy()
        if self.t_touch is None and in_contact: self.t_touch = t
        if self.t_touch is None:
            # FLIGHT: landing posture (slight pre-bend) with soles parallel to the floor
            for k, s in self.dq_crouch.items():
                for a in self.qa[k]: q[a] += p["pre_bend"]*s
            for i_f, (fb, a_p) in enumerate(zip(self.fbodies, self.qa["ap"])):
                tilt = self._tilt(d, fb) - self.foot_tilt0[i_f]; q[a_p] -= p["k_foot"]*tilt[0]
                if self.qa["ar"]: q[self.qa["ar"][i_f]] -= p["k_foot"]*tilt[1]
        else:
            # TOUCHDOWN: compliant crouch (down, hold, back up) to absorb the impact
            tau = t - self.t_touch
            if tau < p["t_down"]: w = tau/p["t_down"]
            elif tau < p["t_down"] + p["t_hold"]: w = 1.0
            else: w = max(0.0, 1.0 - (tau - p["t_down"] - p["t_hold"])/p["t_up"])
            w *= p["crouch"]
            for k, s in self.dq_crouch.items():
                for a in self.qa[k]: q[a] += w*s
        # BALANCE (all phases with contact): capture point over the feet centre, trunk line vertical
        com = d.subtree_com[0]; vcom = P.com_velocity(m, d)
        w0 = np.sqrt(G/max(com[2], 0.2)); cp = com[:2] + vcom[:2]/w0
        feet = 0.5*(foot_xy(d, self.feet[0]) + foot_xy(d, self.feet[1]))
        e = cp - feet - getattr(self, "cp_offset", 0.0)      # cp_offset: the stance's own capture point relative to the feet centre (H1 stands certified with its COM 8 cm behind the feet centre; pushing it to the centre fights the stance). Landing keeps 0.
        if in_contact:
            for a in self.qa["ap"]: q[a] -= p["k_cp"]*self.g_ap*e[0]
            if self.g_ar is not None:
                for a in self.qa["ar"]: q[a] -= p["k_cp"]*self.g_ar*e[1]
        tilt = self._tilt(d, self.tb) - getattr(self, "tilt_ref", 0.0); vel = np.zeros(6)   # tilt_ref: a tracked reference posture's torso tilt (stand mode following a reference); 0 = vertical
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.tb, vel, 0)   # world frame: [angular; linear]
        rate = np.array([vel[1], vel[0]])
        for a in self.qa["hp"]: q[a] -= self.g_hp*(p["k_trunk"]*tilt[0] + p["k_trunk_d"]*rate[0])
        for a in self.qa["hr"]: q[a] -= self.g_hr*(p["k_trunk"]*tilt[1] + p["k_trunk_d"]*rate[1])
        tgt = np.zeros(m.nu)
        for a_ in range(m.nu):
            jid = int(m.actuator_trnid[a_][0]); lo, hi = m.jnt_range[jid]
            tgt[a_] = np.clip(q[m.jnt_qposadr[jid]], lo, hi)
        return tgt


def all_servos(S, omega):
    """Make every driven actuator an implicit position servo with PHYSICAL stiffness: natural frequency omega (rad/s),
    kp = omega^2 * I_eff, critically damped, torque limits kept (dt-independent; the earlier kp ~ 1/dt^2 form did not
    transfer between 2 ms and 1 ms). Returns a restore function (driven actuators back to torque motors)."""
    m = S["m"]; J = S["jarr"]
    for i, (aid, dof) in enumerate(zip(J["aid"], J["dof"])):
        ieff = 1.0/float(S["Minv_key"][dof, dof]); kp = omega*omega*ieff
        BI._as_servo(m, int(aid), kp, 2.0*np.sqrt(kp*ieff), float(J["tau"][i]))
    def restore():
        for i, aid in enumerate(J["aid"]): BI._as_motor(m, int(aid), float(J["tau"][i]))
    return restore


CTRL_DT = 0.002   # fixed 500 Hz control rate with zero-order hold (independent of the simulation timestep)


def simulate(S, ctrl, q0, v0=None, T=2.0, push=None, record=False):
    """Run the controller from state (q0, v0). push = (t0, duration, force_xyz on the torso). Returns outcome metrics."""
    m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = q0; d.qvel[:] = 0 if v0 is None else v0; mujoco.mj_forward(m, d)
    restore = all_servos(S, ctrl.p["stiff"]); fl = S["floor"]; foot = S["foot_geoms"]; tb = ctrl.tb
    nonfoot = False; peak = 0.0; tilt_max = 0.0; trace = []; ctrl._kc = None
    try:
        for s in range(int(T/m.opt.timestep)):
            t = s*m.opt.timestep; fz = 0.0
            for ci in range(d.ncon):
                c = d.contact[ci]
                if fl in (c.geom1, c.geom2):
                    fv = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fv); fz += fv[0]
                    o = c.geom2 if c.geom1 == fl else c.geom1
                    if o not in foot: nonfoot = True
            d.xfrc_applied[:] = 0
            if push and push[0] <= t < push[0] + push[1]: d.xfrc_applied[tb, :3] = push[2]
            k_c = int(round(t/CTRL_DT, 6))
            if k_c != getattr(ctrl, "_kc", None):
                ctrl._kc = k_c; d.ctrl[:] = ctrl.targets(k_c*CTRL_DT, d, fz > 1.0)
            BI.backemf_brake(m, d, S)
            mujoco.mj_step(m, d)
            peak = max(peak, fz); tilt_max = max(tilt_max, float(np.arccos(np.clip(d.xmat[tb][8], -1, 1))))
            if record and s % 10 == 0: trace.append(d.qpos.copy())
    finally:
        restore()
    up = float(d.xmat[tb][8]); z = float(d.subtree_com[0][2]); v = float(np.linalg.norm(P.com_velocity(m, d)))
    ok = (not nonfoot) and up > 0.85 and z > 0.6*ctrl.z_stand and v < 0.5     # 0.6 * 0.666 = 0.40 on G1 (Phase 14 rule)
    out = dict(ok=bool(ok), nonfoot=nonfoot, up_end=up, z_end=z, v_end=v, peak_grf_bw=peak/(S["M"]*G), tilt_max=tilt_max,
               t_touch=ctrl.t_touch)
    if record: out["trace"] = np.array(trace)
    return out
