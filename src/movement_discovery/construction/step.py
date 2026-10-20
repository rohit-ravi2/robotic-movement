"""STEP AS AN EFFECT: the body-derived stepping prior for the stepping / base regime (locomotion P01-P08, trip P20, carry P70,
negotiation P81-P84, P87, P88). Representation "servo_knots+step_prior".

A STEP = place the swing foot at a target (x, y, yaw) by a time T with a landing state the balance controller can hold. Everything
below is DERIVED, nothing is authored per program:
  * FOOTSTEPS from the program's goal (GoalSpec in demand["goal"]: com_displacement target_xy / x_min / along_slope, com_velocity,
    heading, path arc, stability with x_min; else demand["path"] / demand["direction"]) and the BODY: the nominal step width is the
    settled stance's foot separation, the per-step forward / lateral / yaw limits are the largest double-support poses the leg IK can
    reach inside 95% of the joint ranges (x 0.85 margin), the travel heading (face the goal or keep the heading) is the one that needs
    fewer steps under those limits; number of steps = what the goal needs at that stride, capped by the program horizon.
  * FOOTHOLDS and CLEARANCE from the world as built (rays in the plant's own scene, robot moved out of the way): a foothold must be
    flat under the sole rectangle (the foot collision geometry), otherwise it is shifted along the walking direction (stairs: off the
    nosing; a barrier: before / beyond it); the swing apex is the highest ground swept by the sole on the way plus one foot height.
  * PREPARATION + TRANSFER from the support contract: the ZMP must sit on the stance sole during single support, so the COM reference
    is the linear inverted pendulum (omega = sqrt(g / z_com) of this body) driven by the footstep ZMP plan (least squares, at rest at
    both ends): slow timing reproduces the certified quasi-static weight shift of throw.prepare_support (COM over the stance foot;
    single-support poses are checked with the same static certificate, throw._cert_ok), fast timing gives the dynamic sway.
  * SWING = a reach-style path for the FOOT as effector: whole-body differential IK per frame (stance sole fixed, swing sole on a
    lift / traverse / lower min-jerk path to the target, COM on the LIPM reference, pelvis upright at the stance pelvis height above the
    ground under the COM), touchdown, double-support transfer to the next stance foot.
EXECUTION (what worked physically for quasi-static manipulation in tasks/reach.py): servo mode. The Phase-14 capture-point balance
controller (balance.BalanceController, stand mode, made yaw-aware through its cp_offset / tilt_ref hooks) owns the legs; every 2 ms the
legs' targets come from an ANCHORED online IK: the reference pelvis, capture point and swing sole are mapped by the actual stance
sole's planar pose (double support: blended by the ZMP), support soles stay at their latched poses, the swing sole goes to its
anchored reference pose (no balance correction on a foot in the air); double support shares the load by an admittance on the leg
lengths. Arms / waist follow the reference (start pose + the momentum-cancelling shoulder swing). The search refines on top: knot
offsets (rad, +-0.15) on the 12 leg joints and the driven trunk / arm joints, a time scale, the capture-point / trunk gain scales and
two quantised plan levels (pace, stride) that select another cached reference.
Walking = the chained sequence of constructed steps: each step is built from the previous step's planned end state and executed in ONE
continuous servo rollout (the handoff between steps is the identity on the same plant and MjData); a chained Plan step or a handed-off
start enters through demand["initial_state"] (engine.core.handoff), and the reference is then built from that state.
Record: format C (trace q / foot_load_bw every 10 physics steps) + base_displacement / phases like ContactSchedule, so the locomotion,
negotiation, balance and payload evaluators read it unchanged. MuJoCo only inside functions."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import contextlib, json
import numpy as np
from .base import Representation

SIDES = ("left", "right")
LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
FPS = 100
G = 9.80665
KNOT_DT = 0.5          # reference seconds per knot
KNOT_AMP = 0.15        # rad
CTRL_DT = 0.002
STRIDE_FRAC = 0.65     # fraction of the kinematic single-support reach used per step (dynamic margin; see Body.limits)
REF_KEYS = ("rise_at_end", "psi_max", "momentum_comp", "momentum_waist", "momentum_lam_v", "momentum_lam_p", "stride_frac", "step_time_scale")      # demand keys that change the reference


def _mj():
    import mujoco
    return mujoco


def _Rz(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _yaw(R): return float(np.arctan2(R[1, 0], R[0, 0]))


def _wrap(a): return float((a + np.pi) % (2*np.pi) - np.pi)


def _rot_err(R, Rt):
    """0.5 sum_i R[:, i] x Rt[:, i] (the small-angle rotation taking R to Rt, world frame) = 0.5 vee(Rt R^T - R Rt^T)."""
    M = Rt @ R.T; return 0.5*np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])


def _mj_s(s): s = np.clip(s, 0.0, 1.0); return s*s*s*(10 - 15*s + 6*s*s)


def _tilt_to(n):
    """Rotation taking world z to the unit normal n (ground plane under a foothold)."""
    n = np.asarray(n, float)/np.linalg.norm(n); z = np.array([0.0, 0.0, 1.0]); v = np.cross(z, n); s = np.linalg.norm(v); c = float(z @ n)
    if s < 1e-9: return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]); return np.eye(3) + K + K @ K*((1 - c)/s**2)


def _compose(a, b):
    """SE(2): pose a (x, y, yaw) then b expressed in a's frame."""
    c, s = np.cos(a[2]), np.sin(a[2]); return np.array([a[0] + c*b[0] - s*b[1], a[1] + s*b[0] + c*b[1], _wrap(a[2] + b[2])])


def _relative(a, b):
    """b expressed in a's frame."""
    c, s = np.cos(a[2]), np.sin(a[2]); dx, dy = b[0] - a[0], b[1] - a[1]; return np.array([c*dx + s*dy, -s*dx + c*dy, _wrap(b[2] - a[2])])


# ------------------------------------------------------------------------------------------------------------------ the body
class Body:
    """What the step prior reads from the robot model, derived once per plant structure: foot sites and sole rectangles, the leg chains,
    the settled stance's step width / pelvis height / COM height, the per-step limits (IK reachability) and the world probe."""
    def __init__(self, S):
        mj = _mj(); m = S["m"]; self.S = S; self.m = m; self.robot = S.get("robot", "G1")
        self.fsite = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, f"{s}_foot") for s in SIDES]
        if min(self.fsite) < 0: raise RuntimeError("step prior: the robot has no left_foot / right_foot sites (H1): unsupported")
        self.fbody = [int(m.site_bodyid[i]) for i in self.fsite]
        self.pel = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "pelvis"); self.torso = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "torso_link")
        fj = [j for j in range(m.njnt) if int(m.jnt_type[j]) == int(mj.mjtJoint.mjJNT_FREE) and int(m.jnt_bodyid[j]) == self.pel][0]
        self.qfree = int(m.jnt_qposadr[fj]); self.vfree = int(m.jnt_dofadr[fj])
        self.leg = {}
        for i, s in enumerate(SIDES):
            L = []
            for k in LEG:
                nm = f"{s}_{k}_joint"; j = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, nm)
                if j < 0: raise RuntimeError(f"step prior: joint {nm} missing")
                L.append((nm, j, int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])))
            self.leg[i] = L
        self.leg_q = np.array([x[2] for i in (0, 1) for x in self.leg[i]]); self.leg_v = np.array([x[3] for i in (0, 1) for x in self.leg[i]])
        jid = [x[1] for i in (0, 1) for x in self.leg[i]]; r = np.array([m.jnt_range[j] for j in jid]); span = r[:, 1] - r[:, 0]
        self.lo, self.hi = r[:, 0] + 0.05*span, r[:, 1] - 0.05*span
        self.ik_v = np.r_[np.arange(self.vfree, self.vfree + 6), self.leg_v]
        self.act_of = {int(m.actuator_trnid[a][0]): a for a in range(m.nu)}
        self.leg_act = {i: [self.act_of[x[1]] for x in self.leg[i]] for i in (0, 1)}
        self.leg_qa = {i: [x[2] for x in self.leg[i]] for i in (0, 1)}
        # collision geoms of each foot (explicit pairs on the MJX G1; contype/conaffinity elsewhere)
        pair = set(np.asarray(m.pair_geom1).tolist()) | set(np.asarray(m.pair_geom2).tolist())
        self.coll = [{g for g in S["feet"][i] if (g in pair or (m.geom_contype[g] | m.geom_conaffinity[g])) and int(m.geom_type[g]) != int(mj.mjtGeom.mjGEOM_MESH)} or set(S["feet"][i]) for i in (0, 1)]
        q0 = np.asarray(S.get("q_stance0", S["q_settled"]), float); d = mj.MjData(m); d.qpos[:] = q0; mj.mj_forward(m, d); self.q_nom = q0.copy()
        # sole rectangle in the site frame (corners of the collision geoms' boxes)
        self.sole = []; self.foot_h = 0.0
        for i in (0, 1):
            Rs = d.site_xmat[self.fsite[i]].reshape(3, 3); ps = d.site_xpos[self.fsite[i]]; pts = []
            for g in self.coll[i]:
                aabb = m.geom_aabb[g]; c, h = aabb[:3], aabb[3:]; Rg = d.geom_xmat[g].reshape(3, 3); pg = d.geom_xpos[g]
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        for sz in (-1, 1): pts.append(Rs.T @ (pg + Rg @ (c + h*np.array([sx, sy, sz])) - ps))
            pts = np.array(pts); self.sole.append((pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max())); self.foot_h = max(self.foot_h, float(pts[:, 2].max() - max(pts[:, 2].min(), -0.01)))
        self.R0 = [d.site_xmat[self.fsite[i]].reshape(3, 3).copy() for i in (0, 1)]
        self.R0 = [_Rz(-_yaw(R)) @ R for R in self.R0]                      # stance sole orientation with its yaw removed
        pf = [d.site_xpos[self.fsite[i]].copy() for i in (0, 1)]; mid = 0.5*(pf[0] + pf[1]); yaw0 = _yaw(d.xmat[self.pel].reshape(3, 3))
        Rt = _Rz(-yaw0)
        self.w = float((Rt @ (pf[0] - pf[1]))[1])                           # step width = settled stance foot separation
        self.site_z = float(np.mean([p[2] for p in pf]))                    # site height above the ground at stance (ground = 0 or start_z handled by rays)
        self.pel_h = float(d.xpos[self.pel][2] - mid[2]); self.pel_off = (Rt @ (d.xpos[self.pel] - mid))[:2]
        self.com_h = float(d.subtree_com[self.pel][2] - mid[2]); self.com_off = (Rt @ (d.subtree_com[self.pel] - mid))[:2]
        self.omega = float(np.sqrt(G/max(self.com_h, 0.2))); self.tau = 1.0/self.omega
        self.Rpel0 = _Rz(-yaw0) @ d.xmat[self.pel].reshape(3, 3)
        self._lim = None; self._probe = None
        # upper-body dofs that can cancel the vertical angular momentum of the swing (waist yaw, shoulder pitches): driven joints only
        drv = set(S["types"]); self.up = []
        for nm in ("waist_yaw_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "torso"):
            j = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, nm)
            if j >= 0 and nm in drv: lo_, hi_ = m.jnt_range[j]; sp_ = hi_ - lo_; self.up.append((nm, int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]), lo_ + 0.05*sp_, hi_ - 0.05*sp_))
    # ---------------------------------------------------------------------------------------- kinematics
    def fk(self, q):
        mj = _mj(); d = mj.MjData(self.m); d.qpos[:] = q; mj.mj_kinematics(self.m, d); mj.mj_comPos(self.m, d); return d
    def feet_poses(self, q):
        """(x, y, yaw, z) of each sole site."""
        d = self.fk(q); return [np.array([*d.site_xpos[s][:2], _yaw(d.site_xmat[s].reshape(3, 3)), d.site_xpos[s][2]]) for s in self.fsite]
    def ik(self, q, feet, com_xy, pel_z, pel_yaw, iters=8, tol=2e-4, d=None):
        """Damped least-squares whole-body IK over the floating base + 12 leg joints: both soles at (pos, R), COM xy, pelvis height and
        an upright pelvis at pel_yaw. Joint limits at 95%. Returns (q, residual norm)."""
        mj = _mj(); m = self.m; d = d if d is not None else mj.MjData(m); q = np.array(q, float); nv = m.nv; iv = self.ik_v
        jp = np.zeros((3, nv)); jr = np.zeros((3, nv)); jc = np.zeros((3, nv)); Rp_t = _Rz(pel_yaw) @ self.Rpel0; res = 1e9
        for it in range(iters):
            d.qpos[:] = q; mj.mj_kinematics(m, d); mj.mj_comPos(m, d); rows = []; errs = []
            for i in (0, 1):
                pos, Rt = feet[i][:2]; wr = feet[i][2] if len(feet[i]) > 2 else 0.5; s = self.fsite[i]; mj.mj_jacSite(m, d, jp, jr, s)
                rows += [jp[:, iv].copy(), wr*jr[:, iv]]; errs += [pos - d.site_xpos[s], wr*_rot_err(d.site_xmat[s].reshape(3, 3), Rt)]
            mj.mj_jacSubtreeCom(m, d, jc, self.pel); rows.append(jc[:2, iv].copy()); errs.append(np.asarray(com_xy) - d.subtree_com[self.pel][:2])
            mj.mj_jacBody(m, d, jp, jr, self.pel); rows += [jp[2:3, iv].copy(), 0.5*jr[:, iv]]; errs += [np.array([pel_z - d.xpos[self.pel][2]]), 0.5*_rot_err(d.xmat[self.pel].reshape(3, 3), Rp_t)]
            A = np.vstack(rows); e = np.concatenate(errs); res = float(np.linalg.norm(e))
            if res < tol: break
            dq = np.linalg.solve(A.T @ A + 1e-5*np.eye(A.shape[1]), A.T @ e); n_ = np.linalg.norm(dq)
            if n_ > 0.2: dq *= 0.2/n_
            v = np.zeros(nv); v[iv] = dq; mj.mj_integratePos(m, q, v, 1.0); q[self.leg_q] = np.clip(q[self.leg_q], self.lo, self.hi)
        else:
            res = self._res(q, feet, com_xy, pel_z, Rp_t, d)
        return q, res
    def _res(self, q, feet, com_xy, pel_z, Rp_t, d):
        mj = _mj(); m = self.m; d.qpos[:] = q; mj.mj_kinematics(m, d); mj.mj_comPos(m, d); e = []
        for i in (0, 1): pos, Rt = feet[i][:2]; wr = feet[i][2] if len(feet[i]) > 2 else 0.5; s = self.fsite[i]; e += [pos - d.site_xpos[s], wr*_rot_err(d.site_xmat[s].reshape(3, 3), Rt)]
        e += [np.asarray(com_xy) - d.subtree_com[self.pel][:2], np.array([pel_z - d.xpos[self.pel][2]]), 0.5*_rot_err(d.xmat[self.pel].reshape(3, 3), Rp_t)]
        return float(np.linalg.norm(np.concatenate(e)))
    def Lz(self, d, v) -> float:
        """Vertical angular momentum of the robot about its COM for joint velocity v at the kinematics already in d."""
        mj = _mj(); d.qvel[:] = v; mj.mj_comVel(self.m, d); mj.mj_subtreeVel(self.m, d); return float(d.subtree_angmom[self.pel][2])
    def foot_target(self, i, x, y, yaw, z, normal=(0.0, 0.0, 1.0)):
        return (np.array([x, y, z], float), _tilt_to(normal) @ _Rz(yaw) @ self.R0[i])
    # ---------------------------------------------------------------------------------------- per-step limits (reachability)
    def limits(self) -> dict:
        """Per-step limits from SINGLE-SUPPORT REACH at touchdown (the binding pose of a step): stance sole fixed, the landing sole flat
        at (dx, +-(w + dw), psi), the COM 85% of the way from the feet midpoint to the stance foot (where the LIPM puts it at touchdown),
        joints inside 95% of their ranges, IK residual < 2 mm. The walking pelvis height = stance height - dh, dh the drop (grid 0..8 cm)
        that maximises min(forward, backward) reach. Each limit x STRIDE_FRAC (0.65: the dynamic margin; at 0.8 x reach the executed
        walk fell at both timesteps, at 0.63 it held). Derived from leg length and joint ranges; cached per robot model."""
        key = (self.robot, self.S.get("mjcf"), np.round(self.q_nom[self.leg_q], 2).tobytes())
        if key in _LIMITS: self._lim = _LIMITS[key]
        if self._lim is not None: return self._lim
        q0 = self.q_nom; fp = self.feet_poses(q0); st = fp[0]; z = float(st[3])
        def ok(dh, dx=0.0, dw=0.0, psi=0.0):
            yaw0 = float(st[2]); R = _Rz(yaw0)[:2, :2]; sw = np.asarray(st[:2]) + R @ np.array([dx, -(self.w + dw)])
            feet = [self.foot_target(0, st[0], st[1], yaw0, z), self.foot_target(1, sw[0], sw[1], yaw0 + psi, z)]
            com = np.asarray(st[:2]) + 0.15*(0.5*(sw - np.asarray(st[:2]))); q = q0.copy()
            for _ in range(10):
                q, r = self.ik(q, feet, com, z + self.pel_h - dh, yaw0 + 0.5*psi, iters=10)
                if r < 2e-3: break
            return r < 2e-3 and bool(np.all(q[self.leg_q] > self.lo + 1e-4) and np.all(q[self.leg_q] < self.hi - 1e-4))
        def bis(f, a, b, n=8):
            if not f(a): return 0.0
            for _ in range(n):
                c = 0.5*(a + b)
                if f(c): a = c
                else: b = c
            return a
        best = None
        for dh in (0.0, 0.02, 0.04, 0.06, 0.08):
            fw = bis(lambda v: ok(dh, dx=v), 0.0, 0.6); bk = bis(lambda v: ok(dh, dx=-v), 0.0, 0.6)
            if best is None or min(fw, bk) > min(best[1], best[2]) + 1e-3: best = (dh, fw, bk)
        dh, fw, bk = best; dw = bis(lambda v: ok(dh, dw=v), 0.0, 0.3); psi = bis(lambda v: ok(dh, psi=v), 0.0, 1.0); psi_in = bis(lambda v: ok(dh, psi=-v), 0.0, 1.0)
        f = STRIDE_FRAC
        self._lim = _LIMITS[key] = dict(s_f=f*fw, s_b=f*bk, dw=f*dw, psi=f*min(psi, psi_in), dh=dh, w=self.w, frac=f, raw=dict(fw=fw, bk=bk, dw=dw, psi_out=psi, psi_in=psi_in))
        return self._lim
    # ---------------------------------------------------------------------------------------- the world probe (rays)
    def ground(self, xy) -> float:
        """Height of the highest static or world surface at xy (the robot moved out of the way)."""
        mj = _mj()
        if self._probe is None:
            d = mj.MjData(self.m); d.qpos[:] = self.q_nom; d.qpos[self.qfree + 2] += 50.0; mj.mj_forward(self.m, d); self._probe = (d, np.zeros(1, np.int32))
        d, gid = self._probe; dist = mj.mj_ray(self.m, d, np.array([xy[0], xy[1], 5.0]), np.array([0.0, 0.0, -1.0]), None, 1, -1, gid)
        return 0.0 if dist < 0 else float(5.0 - dist)
    def sole_points(self, i, x, y, yaw, nx=5, ny=3, grow=0.0):
        x0, x1, y0, y1 = self.sole[i]; R = _Rz(yaw)[:2, :2]
        return [np.array([x, y]) + R @ np.array([a, b]) for a in np.linspace(x0 - grow, x1 + grow, nx) for b in np.linspace(y0 - grow, y1 + grow, ny)]
    def foothold(self, i, x, y, yaw, grow=0.02):
        """(flat, mean height, normal): a plane fitted to the ground under the sole rectangle grown by the swing sweep's 2 cm margin
        (a foot placed 1 cm short of a barrier made its own swing clear the barrier: a 0.27 m lift and a shin strike); flat = residual <= 1 cm."""
        P = self.sole_points(i, x, y, yaw, grow=grow); h = np.array([self.ground(p) for p in P]); X = np.c_[np.array(P), np.ones(len(P))]
        c, *_ = np.linalg.lstsq(X, h, rcond=None); res = float(np.max(np.abs(X @ c - h))); n = np.array([-c[0], -c[1], 1.0]); n /= np.linalg.norm(n)
        return res <= 0.01, float(c[0]*x + c[1]*y + c[2]), n
    def sweep_height(self, i, p0, p1, yaw0, yaw1, n=12) -> float:
        """Highest ground swept by the sole rectangle (grown 2 cm) moving from p0 to p1."""
        hs = []
        for s in np.linspace(0, 1, n):
            xy = p0 + s*(p1 - p0); hs += [self.ground(p) for p in self.sole_points(i, xy[0], xy[1], yaw0 + s*_wrap(yaw1 - yaw0), nx=3, ny=3, grow=0.02)]
        return float(max(hs))


_BODIES: Dict[Tuple, Body] = {}
_LIMITS: Dict[Tuple, dict] = {}


def body_of(S) -> Body:
    key = (id(S["m"]),)
    if key not in _BODIES: _BODIES[key] = Body(S)
    return _BODIES[key]


# ------------------------------------------------------------------------------------------------------------------ goal -> waypoints
def goal_waypoints(demand: dict, start: np.ndarray, H: float) -> dict:
    """The program's goal as base waypoints [(x, y, yaw or None)] in WORLD coordinates (start = pelvis (x, y, yaw)), a desired speed
    (None = no speed demand) and the goal error function's spec. Reads demand["goal"] (GoalSpec) first, then demand["path"] /
    demand["direction"]. Nothing per program: one rule per goal kind."""
    g = demand.get("goal"); kind = getattr(g, "kind", None); gp = dict(getattr(g, "params", {}) or {}); x0, y0, yaw0 = map(float, start)
    def rel(dx, dy): c, s = np.cos(yaw0), np.sin(yaw0); return np.array([x0 + c*dx - s*dy, y0 + s*dx + c*dy])
    wps: List[Tuple[float, float, Optional[float]]] = []; speed = None; spec = dict(kind="none")
    if kind == "com_displacement" and gp.get("target_xy") is not None:
        t = np.array([x0, y0]) + np.asarray(gp["target_xy"], float); wps = [(t[0], t[1], None)]; spec = dict(kind="point", xy=t.tolist())
    elif kind == "com_displacement" and gp.get("along_slope") is not None:
        t = rel(float(gp["along_slope"])*np.cos(float(gp.get("angle", 0.0))), 0.0); wps = [(t[0], t[1], yaw0)]; spec = dict(kind="point", xy=t.tolist())
    elif kind in ("com_displacement", "stability") and gp.get("x_min") is not None:
        t = rel(float(gp["x_min"]) + 0.12, 0.0); wps = [(t[0], t[1], yaw0)]; spec = dict(kind="x_min", x_min=float(x0 + gp["x_min"]))
    elif kind == "com_velocity":
        if gp.get("v_xy") is not None: v = np.asarray(gp["v_xy"], float)
        else: sp_ = float(gp.get("speed", 0.0)); h_ = float(gp.get("heading", 0.0)); v = sp_*np.array([np.cos(h_), np.sin(h_)])
        speed = float(np.linalg.norm(v)); hd = float(np.arctan2(v[1], v[0])); D = speed*H; t = rel(D*np.cos(hd), D*np.sin(hd))
        wps = [(t[0], t[1], _wrap(yaw0 + hd))]; spec = dict(kind="velocity", v=(_Rz(yaw0)[:2, :2] @ v).tolist(), window=gp.get("window"))
    elif kind == "heading":
        speed = float(gp.get("speed", 0.5)); dpsi = float(gp["dpsi"]); D = speed*H; a = rel(0.35*D, 0.0); b = a + 0.65*D*np.array([np.cos(yaw0 + dpsi), np.sin(yaw0 + dpsi)])
        wps = [(a[0], a[1], yaw0), (b[0], b[1], _wrap(yaw0 + dpsi))]; spec = dict(kind="heading", yaw=_wrap(yaw0 + dpsi))
    elif kind == "path" and gp.get("shape") == "arc":
        R, arc, side = float(gp["R"]), float(gp["arc"]), int(gp["side"]); c = rel(0.0, side*R); n = max(2, int(np.ceil(R*arc/0.25)))
        for k in range(1, n + 1):
            a = k*arc/n; p = rel(R*np.sin(a), side*R*(1 - np.cos(a))); wps.append((p[0], p[1], _wrap(yaw0 + side*a)))
        speed = float(gp.get("speed")) if gp.get("speed") else None; spec = dict(kind="arc", centre=c.tolist(), R=R, arc=arc, side=side)
    elif demand.get("path") is not None:
        P = np.asarray(demand["path"], float); P = P - P[0]
        for p in P[1:]: t = np.array([x0, y0]) + p; wps.append((t[0], t[1], None))
        spec = dict(kind="point", xy=wps[-1][:2]) if wps else spec
    elif demand.get("direction") is not None and demand.get("distance"):
        e, a = map(float, demand["direction"]); D = float(demand["distance"]); t = rel(D*np.cos(a), D*np.sin(a)); wps = [(t[0], t[1], None)]; spec = dict(kind="point", xy=t.tolist())
    return dict(waypoints=[tuple(map(lambda v: None if v is None else float(v), w)) for w in wps], speed=speed, spec=spec)


# ------------------------------------------------------------------------------------------------------------------ footstep planner
def plan_footsteps(body: Body, feet0: List[np.ndarray], waypoints, n_max: int, first: Optional[int] = None, L: Optional[dict] = None) -> List[dict]:
    """Alternating footsteps toward the waypoints under the body's per-step limits. Each swing foot goes to its nominal place in the
    current waypoint frame (+-w/2 lateral), clipped in the stance-foot frame to [-s_b, s_f] forward, [w, w + dw] lateral (no crossing),
    +-psi yaw; the target is moved along the walking direction to the nearest flat foothold. Stops when both feet are within 2 cm /
    0.03 rad of the last waypoint's nominal placement, or at n_max."""
    L = L or body.limits(); w = body.w; feet = [np.array(f[:3], float) for f in feet0]; steps = []; wi = 0; wps = list(waypoints)
    if not wps: return steps
    def nominal(i, wp): return _compose(wp, np.array([0.0, (0.5*w if i == 0 else -0.5*w), 0.0]))
    def frame(wp): return np.array([float(wp[0]), float(wp[1]), float(wp[2])])
    # pelvis waypoint -> feet-midpoint frame (the pelvis sits pel_off ahead of the feet midpoint)
    def mid_frame(wp): f = frame(wp); c, s = np.cos(f[2]), np.sin(f[2]); po = body.pel_off; return np.array([f[0] - c*po[0] + s*po[1], f[1] - s*po[0] - c*po[1], f[2]])
    capped = True
    swing = first
    if swing is None:                       # the foot on the goal's side moves first (a lateral goal: lead with that foot), else the right
        g = mid_frame(wps[0]); rel = _relative(0.5*(feet[0] + feet[1]), g); swing = 0 if rel[1] > 0.02 else 1
    for k in range(n_max):
        M = mid_frame(wps[wi]); tgt = nominal(swing, M); st = feet[1 - swing]; r = _relative(st, tgt); sgn = 1.0 if swing == 0 else -1.0
        dx = float(np.clip(r[0], -L["s_b"], L["s_f"])); lat = float(np.clip(sgn*r[1], w, w + L["dw"])); dyaw = float(np.clip(r[2], -L["psi"], L["psi"]))
        new = _compose(st, np.array([dx, sgn*lat, dyaw]))
        flat, z, nrm = body.foothold(swing, new[0], new[1], new[2])
        if not flat:                       # nearest flat foothold along the stance foot's forward axis, inside the stride limits
            best = None
            for dd in sorted(np.arange(-0.30, 0.301, 0.02), key=abs):
                if not (-L["s_b"] <= dx + dd <= L["s_f"]*1.15): continue
                cand = _compose(st, np.array([dx + dd, sgn*lat, dyaw])); f_, z_, n_ = body.foothold(swing, cand[0], cand[1], cand[2])
                if f_: best = (cand, z_, n_); break
            if best is not None: new, z, nrm = best; flat = True
        steps.append(dict(foot=int(swing), pose=new.tolist(), z=float(z), normal=np.asarray(nrm).tolist(), flat=bool(flat)))
        feet[swing] = new
        done = all(np.linalg.norm(feet[i][:2] - nominal(i, M)[:2]) < 0.02 and abs(_wrap(feet[i][2] - M[2])) < 0.03 for i in (0, 1))
        if done or (wi < len(wps) - 1 and np.linalg.norm(0.5*(feet[0][:2] + feet[1][:2]) - M[:2]) < 0.6*L["s_f"]):
            if wi == len(wps) - 1: capped = False; break
            wi += 1
        swing = 1 - swing
    if capped and len(steps) >= 2:
        # the horizon cut the plan short: the LAST step squares the trailing foot up beside the stance foot (nominal width, same yaw), so
        # the final transfer starts from a well-formed stance instead of a staggered, twisted one
        i = steps[-1]["foot"]; st = np.asarray(steps[-2]["pose"], float); sgn = 1.0 if i == 0 else -1.0
        new = _compose(st, np.array([0.0, sgn*w, 0.0])); flat, z, nrm = body.foothold(i, new[0], new[1], new[2])
        if flat: steps[-1] = dict(foot=int(i), pose=new.tolist(), z=float(z), normal=np.asarray(nrm).tolist(), flat=True, closing=True)
    return steps


def choose_heading(body: Body, feet0, start, wps, n_max, L=None):
    """Fill missing waypoint yaws with the option that ends CLOSEST to the goal under the step cap (then: fewer steps): keep the current
    heading, face the travel direction while walking, or turn in place first and then walk."""
    if all(w[2] is not None for w in wps): return [tuple(w) for w in wps], None
    keep, face = [], []; prev = np.asarray(start[:2], float)
    for (x, y, yw) in wps:
        d = np.array([x, y]) - prev; tr = float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 1e-6 else float(start[2])
        keep.append((x, y, float(start[2]) if yw is None else yw)); face.append((x, y, tr if yw is None else yw)); prev = np.array([x, y])
    def score(steps, wps_):
        f = [np.array(x[:3], float) for x in feet0]
        for s_ in steps: f[s_["foot"]] = np.asarray(s_["pose"], float)
        mid = 0.5*(f[0][:2] + f[1][:2]); g = np.asarray(wps_[-1][:2], float) - _Rz(wps_[-1][2])[:2, :2] @ body.pel_off
        return (round(float(np.linalg.norm(mid - g)), 2), len(steps))          # remaining distance first (cm resolution), then the step count
    turn = [(float(start[0]), float(start[1]), face[0][2])] + face                 # turn in place first, then walk facing the travel direction
    cands = [(keep, "keep"), (face, "face"), (turn, "turn_then_walk")]
    sc = [score(plan_footsteps(body, feet0, w_, n_max, L=L), w_) for w_, _ in cands]
    return cands[int(min(range(3), key=lambda i: sc[i]))]


# ------------------------------------------------------------------------------------------------------------------ LIPM COM reference
def lipm_com(p_ref: np.ndarray, dt: float, omega: float, x0: np.ndarray, xN: np.ndarray) -> np.ndarray:
    """COM xy whose LIPM ZMP x - x''/omega^2 follows p_ref in least squares, at rest at x0 at the start and at xN at the end
    (sparse normal equations; the dense solve cost seconds per reference)."""
    import scipy.sparse as sp, scipy.sparse.linalg as spl
    N = len(p_ref); k2 = 1.0/(omega*omega*dt*dt); W = 1e3; rows, cols, vals = [], [], []
    for k in range(1, N - 1):
        rows += [k - 1]*3; cols += [k - 1, k, k + 1]; vals += [-k2, 1 + 2*k2, -k2]
    for j, idx in enumerate((0, 1, N - 1, N - 2)): rows.append(N - 2 + j); cols.append(idx); vals.append(W)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N + 2, N)); AtA = (A.T @ A).tocsc(); out = np.zeros((N, 2))
    for ax in (0, 1):
        b = np.zeros(N + 2); b[:N - 2] = p_ref[1:N - 1, ax]; b[N - 2:] = W*np.array([x0[ax], x0[ax], xN[ax], xN[ax]])
        out[:, ax] = spl.spsolve(AtA, A.T @ b)
    return out


# ------------------------------------------------------------------------------------------------------------------ the reference
def timing(body: Body, n_steps: int, H: float, speed: Optional[float], stride: float, scale: float = 1.0) -> dict:
    """Body-derived phase durations in units of the LIPM time constant tau = 1/omega: single support 2.4 tau, double support 1.6 tau,
    initial shift 3 tau, final transfer 3 tau + 0.6 s rest. NOT compressed to fit a horizon or a demanded speed (compressed steps of
    0.8 s fell at both timesteps): the horizon caps the number of steps instead; the search's tscale (0.7..1.3) may speed it up."""
    tau = body.tau*float(scale); t0 = 3.0*tau; t_end = 3.0*tau + 0.6; ss = 2.4*tau; ds = 1.6*tau; per = ss + ds
    budget = max(H - t0 - t_end, 0.0)
    return dict(t0=t0, ss=ss, ds=ds, t_end=t_end, per=per, n_fit=int(budget//per) if per > 0 else n_steps, speed_demand=speed, speed_nominal=stride/per)


def build_reference(body: Body, q_start: np.ndarray, demand: dict, H: float) -> dict:
    """The stepping reference from the current start state: goal -> waypoints -> footsteps -> timing -> ZMP plan -> LIPM COM ->
    per-frame swing-foot path + whole-body IK at FPS. Pure kinematics (no simulation)."""
    q_start = np.asarray(q_start, float); fp = body.feet_poses(q_start); d0 = body.fk(q_start)
    start = np.array([*d0.xpos[body.pel][:2], _yaw(d0.xmat[body.pel].reshape(3, 3))])
    L = dict(body.limits())
    if demand.get("stride_frac") is not None:                # a searched stride level: the same reach, another fraction of it
        f_ = float(demand["stride_frac"])/L["frac"]; L.update(s_f=L["s_f"]*f_, s_b=L["s_b"]*f_, dw=L["dw"]*f_, psi=L["psi"]*f_, frac=float(demand["stride_frac"]))
    if demand.get("psi_max") is not None: L["psi"] = min(L["psi"], float(demand["psi_max"]))
    if demand.get("footsteps") is not None:          # explicit footsteps (tests / a caller that already has targets): relative to the start feet
        steps = []; feet = [f[:3].copy() for f in fp]
        for s in demand["footsteps"]:
            i = SIDES.index(s["foot"]) if isinstance(s["foot"], str) else int(s["foot"]); pose = _compose(feet[i], np.array([s.get("dx", 0.0), s.get("dy", 0.0), s.get("dyaw", 0.0)]))
            flat, z, n = body.foothold(i, *pose); steps.append(dict(foot=i, pose=pose.tolist(), z=z, normal=n.tolist(), flat=flat)); feet[i] = pose
        gw = dict(waypoints=[], speed=None, spec=dict(kind="footsteps")); mode = "explicit"
    else:
        gw = goal_waypoints(demand, start, H); n_cap = timing(body, 0, H, None, L["s_f"], float(demand.get("step_time_scale", 1.0)))["n_fit"]
        wps, mode = choose_heading(body, fp, start, gw["waypoints"], n_cap, L) if gw["waypoints"] else ([], None)
        steps = plan_footsteps(body, fp, wps, n_cap, L=L) if wps else []
    tm = timing(body, len(steps), H, gw.get("speed"), L["s_f"], float(demand.get("step_time_scale", 1.0)))
    # ---- phase timeline and ZMP plan
    feet = [np.array([*f[:3]]) for f in fp]; fz = [float(f[3]) for f in fp]
    segs = []; t = 0.0; com0 = d0.subtree_com[body.pel][:2].copy()
    def centre(i, pose): return np.asarray(pose[:2], float)
    zmp_from = com0.copy()
    if steps:
        st0 = 1 - steps[0]["foot"]; segs.append(dict(kind="ds", t0=t, t1=t + tm["t0"], p0=zmp_from, p1=centre(st0, feet[st0]))); t += tm["t0"]
        for k, s in enumerate(steps):
            i = s["foot"]; st = 1 - i; segs.append(dict(kind="ss", t0=t, t1=t + tm["ss"], stance=st, swing=i, p=centre(st, feet[st]), k=k)); t += tm["ss"]
            feet[i] = np.asarray(s["pose"], float)
            nxt = (1 - steps[k + 1]["foot"]) if k + 1 < len(steps) else None
            p_to = centre(nxt, feet[nxt]) if nxt is not None else 0.5*(feet[0][:2] + feet[1][:2])
            segs.append(dict(kind="ds", t0=t, t1=t + (tm["ds"] if nxt is not None else tm["t_end"] - 0.6), p0=centre(st, feet[st]), p1=p_to)); t = segs[-1]["t1"]
    mid = 0.5*(feet[0][:2] + feet[1][:2]); yaw_end = float(np.arctan2(np.sin(feet[0][2]) + np.sin(feet[1][2]), np.cos(feet[0][2]) + np.cos(feet[1][2])))
    segs.append(dict(kind="hold", t0=t, t1=t + 0.6 + (0.0 if steps else 0.4), p=mid)); T = segs[-1]["t1"]
    N = int(round(T*FPS)) + 1; ts = np.arange(N)/FPS; zmp = np.zeros((N, 2)); stance = -np.ones(N, int)
    for sg in segs:
        k0, k1 = int(round(sg["t0"]*FPS)), min(N, int(round(sg["t1"]*FPS)) + 1)
        for k in range(k0, k1):
            if sg["kind"] == "ds": a = _mj_s((ts[k] - sg["t0"])/max(sg["t1"] - sg["t0"], 1e-9)); zmp[k] = (1 - a)*sg["p0"] + a*sg["p1"]
            elif sg["kind"] == "ss": zmp[k] = sg["p"]; stance[k] = sg["stance"]
            else: zmp[k] = sg["p"]
    com_end = mid + _Rz(yaw_end)[:2, :2] @ body.com_off
    com = lipm_com(zmp, 1.0/FPS, body.omega, com0, com_end) if N > 4 else np.repeat(com0[None], N, 0)
    comd = np.gradient(com, 1.0/FPS, axis=0)
    # ---- per-frame targets + IK
    feet = [np.array([*f[:3]]) for f in fp]; fzc = list(fz); Q = np.zeros((N, len(q_start))); q = q_start.copy(); res_max = 0.0; cert = []; clear = []
    ground0 = [fz[i] - body.site_z for i in (0, 1)]
    mj = _mj(); dd = mj.MjData(body.m); seg_i = 0; cur = [f.copy() for f in feet]; curz = list(fzc); curn = [np.array([0.0, 0.0, 1.0])]*2; gnd = list(ground0)
    pyaw_prev = float(start[2]); yseg = -1; y_seg0 = pyaw_prev
    sw = None; n_unreached = 0; res_k = np.zeros(N); lam_v = float(demand.get("momentum_lam_v", 1e-3)); lam_p = float(demand.get("momentum_lam_p", 2e-3))
    # walking pelvis height: down by the body-derived dh over the initial shift, back up over the final transfer
    t_dn = segs[0]["t1"] if steps else 0.0; fin = [sg for sg in segs if sg["kind"] == "ds"]; t_up0, t_up1 = (fin[-1]["t0"], fin[-1]["t1"]) if steps else (0.0, 0.0)
    def dh_of(tk):
        if not steps: return 0.0
        up = _mj_s((tk - t_up0)/max(t_up1 - t_up0, 1e-6)) if demand.get("rise_at_end", False) else 0.0      # default: stay at the walking height (rising during the last transfer cost stability)
        return L["dh"]*(_mj_s(tk/max(t_dn, 1e-6)) - up)
    for k in range(N):
        tk = ts[k]
        while seg_i < len(segs) - 1 and tk > segs[seg_i]["t1"] + 1e-9: seg_i += 1
        sg = segs[seg_i]; tgt = [None, None]
        if sg["kind"] == "ss":
            s = steps[sg["k"]]; i = sg["swing"]
            if sw is None or sw["k"] != sg["k"]:
                p0 = cur[i].copy(); p1 = np.asarray(s["pose"], float); z0_ = curz[i]; z1_ = float(s["z"]) + body.site_z
                hs = body.sweep_height(i, p0[:2], p1[:2], p0[2], p1[2]) + body.site_z
                ztop = max(hs, z0_, z1_) + body.foot_h; clear.append(dict(step=sg["k"], apex=float(ztop - body.site_z), swept=float(hs - body.site_z)))
                sw = dict(k=sg["k"], i=i, p0=p0, p1=p1, z0=z0_, z1=z1_, ztop=ztop, n0=curn[i], n1=np.asarray(s["normal"], float))
            u = (tk - sg["t0"])/max(sg["t1"] - sg["t0"], 1e-9)
            hz = _mj_s((u - 0.15)/0.70); lift = _mj_s(u/0.25); low = _mj_s((u - 0.75)/0.25)
            z = sw["z0"] + (sw["ztop"] - sw["z0"])*lift
            if u > 0.75: z = sw["ztop"] + (sw["z1"] - sw["ztop"])*low
            xy = sw["p0"][:2] + hz*(sw["p1"][:2] - sw["p0"][:2]); yw = sw["p0"][2] + hz*_wrap(sw["p1"][2] - sw["p0"][2])
            nrm = sw["n0"] if u < 0.5 else sw["n1"]
            tgt[i] = body.foot_target(i, xy[0], xy[1], yw, z, nrm if u < 0.1 or u > 0.9 else (0, 0, 1))
            tgt[i] = (*tgt[i], 0.5*max(1.0 - _mj_s(u/0.15), _mj_s((u - 0.7)/0.2)) + 0.02)    # the swing sole may pitch in mid-swing (a flat trailing sole runs out of ankle range); flat again for touchdown
            st = 1 - i
            tgt[st] = body.foot_target(st, cur[st][0], cur[st][1], cur[st][2], curz[st], curn[st])
            if u >= 1.0 - 1e-9 or (k + 1 < N and ts[k + 1] > sg["t1"] + 1e-9):
                cur[i] = sw["p1"].copy(); curz[i] = sw["z1"]; curn[i] = sw["n1"]; gnd[i] = float(s["z"])
        else:
            for i in (0, 1): tgt[i] = body.foot_target(i, cur[i][0], cur[i][1], cur[i][2], curz[i], curn[i])
        # pelvis: upright, yaw = mean of the feet yaws, height above the ground under the COM (feet weighted by proximity of the COM)
        # pelvis yaw turns only in DOUBLE support (both soles resist the yaw reaction; turning the whole body over one stance sole
        # slipped it 35 deg): fixed through single support, min-jerk to the mean of the feet yaws over each double support
        fy = [cur[i][2] for i in (0, 1)]; ymean = float(np.arctan2(np.sin(fy[0]) + np.sin(fy[1]), np.cos(fy[0]) + np.cos(fy[1])))
        if seg_i != yseg: yseg = seg_i; y_seg0 = pyaw_prev
        if sg["kind"] == "ss": pyaw = y_seg0
        else: pyaw = float(y_seg0 + _mj_s((tk - sg["t0"])/max(sg["t1"] - sg["t0"], 1e-9))*_wrap(ymean - y_seg0))
        pyaw_prev = pyaw
        wts = [1.0/(np.linalg.norm(com[k] - cur[i][:2]) + 0.03) for i in (0, 1)]
        if sg["kind"] == "ss": wts[sw["i"]] = 0.0
        gz = float(np.dot(wts, gnd)/sum(wts)); pz = gz + body.pel_h - dh_of(tk)
        q, r = body.ik(q, tgt, com[k], pz, pyaw, iters=8, d=dd); res_max = max(res_max, r); n_unreached += int(r > 5e-3); res_k[k] = r
        if k > 0 and body.up and float(demand.get("momentum_comp", 0.5) or 0.0) > 0:
            # ZERO VERTICAL ANGULAR MOMENTUM: the upper body (waist yaw, shoulder pitches) counter-rotates against the swing leg's yaw momentum
            # (a stance sole on three contact capsules slipped 16-27 deg in yaw under the uncompensated swing), sprung back to its start pose
            v = np.zeros(body.m.nv); mj.mj_differentiatePos(body.m, v, 1.0/FPS, Q[k - 1], q); iu = [u_[2] for u_ in body.up]; qu = [u_[1] for u_ in body.up]
            v[iu] = 0.0; dd.qpos[:] = q; mj.mj_kinematics(body.m, dd); mj.mj_comPos(body.m, dd); L0 = float(demand.get("momentum_comp", 0.5))*body.Lz(dd, v); a_ = np.zeros(len(iu))
            for j_, dof_ in enumerate(iu): e_ = np.zeros(body.m.nv); e_[dof_] = 1.0; a_[j_] = body.Lz(dd, e_)*(0.0 if (body.up[j_][0] in ("waist_yaw_joint", "torso") and not demand.get("momentum_waist", False)) else 1.0)
            A_ = np.vstack([a_[None, :], np.sqrt(lam_v)*np.eye(len(iu))]); b_ = np.r_[-L0, -np.sqrt(lam_v)*0.0*np.ones(len(iu))]
            A_ = np.vstack([A_, np.sqrt(lam_p)*np.eye(len(iu))/FPS]); b_ = np.r_[b_, -np.sqrt(lam_p)*(q[qu] - q_start[qu])]
            vu = np.linalg.lstsq(A_, b_, rcond=None)[0]; q[qu] = np.clip(q[qu] + vu/FPS, [u_[3] for u_ in body.up], [u_[4] for u_ in body.up]); dd.qvel[:] = 0
        Q[k] = q; stance[k] = sg["stance"] if sg["kind"] == "ss" else -1
        if sg["kind"] == "ss" and abs((tk - sg["t0"]) - 0.5*(sg["t1"] - sg["t0"])) < 0.5/FPS: cert.append(dict(step=sg["k"], t=float(tk), q=q.copy()))
    nb = min(N, int(round(0.15*FPS)))              # blend from the actual start (a handed-off pose differs from the IK's first frame)
    for k in range(nb): a = _mj_s(k/max(nb - 1, 1)); Q[k] = q_start + a*(Q[k] - q_start)
    meta = [dict(foot=SIDES[s["foot"]], target=[float(v) for v in s["pose"]], z=float(s["z"]), flat=bool(s["flat"])) for s in steps]
    for sg in segs:
        if sg["kind"] == "ss": meta[sg["k"]].update(t_lift=float(sg["t0"]), t_land=float(sg["t1"]))
    return dict(Q=Q, fps=FPS, T=float(T), N=N, stance=stance, com=com, comd=comd, zmp=zmp, steps=meta, n_steps=len(steps), segs=[{k_: (v.tolist() if isinstance(v, np.ndarray) else v) for k_, v in sg.items()} for sg in segs],
                timing=tm, limits=L, goal=gw["spec"], heading_mode=mode, waypoints=(gw["waypoints"]), ik_res_max=float(res_max), ik_unreached=int(n_unreached), ik_res=res_k, cert_q=cert, clearance=clear, start=start.tolist())


def certify(body: Body, ref: dict) -> List[dict]:
    """Static certificate (plant.static_certificate through throw._cert_ok, robot collision geoms) of each step's mid-single-support
    pose: True where the COM reference is quasi-static (slow timing), expectedly False where the timing is dynamic."""
    import plant as P, throw as T
    mj = _mj(); m = body.m; S = body.S
    cg = [g for g in P.PlantContractVerifier(m, mj.MjData(m), P.PlantContract(S["mjcf"], "torque")).collision_geoms() if g in S.get("robot_geoms", set(range(m.ngeom)))]
    out = []
    for c in ref.get("cert_q", []):
        try: r = P.static_certificate(m, mj.MjData(m), c["q"], coll_geoms=cg); out.append(dict(step=c["step"], ok=bool(T._cert_ok(r)), status=r["status"], margin=r.get("support_margin")))
        except Exception as e: out.append(dict(step=c["step"], ok=False, status=f"error {type(e).__name__}"))
    return out


# ------------------------------------------------------------------------------------------------------------------ the representation
class StepPrior(Representation):
    """Servo-mode stepping: the body-derived footstep / LIPM / swing-IK reference tracked by the balance-controlled legs, plus knot
    offsets, a time scale and gain scales (the search)."""
    name = "servo_knots+step_prior"; label = "step as an effect: goal-derived footsteps, LIPM COM, swing-foot IK; balance-controlled legs + knot offsets"
    N_GLOBAL = 5                                   # tscale, k_cp scale, k_trunk scale, pace level, stride level
    PACE = (0.7, 0.85, 1.0, 1.15, 1.3)             # step-time scale levels (z = 0 -> 1.0): the plan's step count follows (more steps fit a horizon)
    STRIDE = (0.5, 0.65, 0.8)                      # fraction of the single-support reach per step (z = 0 -> STRIDE_FRAC)
    @staticmethod
    def _level(zv, levels): return levels[int(np.clip(round((float(zv) + 1.0)/2.0*(len(levels) - 1)), 0, len(levels) - 1))]
    def _dem(self, demand, p=None):
        """The demand the reference is built from: the searched pace / stride levels (None = the prior's)."""
        if p is None or (p.get("pace", 1.0) == 1.0 and p.get("stride", STRIDE_FRAC) == STRIDE_FRAC): return demand
        return dict(demand, step_time_scale=float(p["pace"]) if p.get("pace", 1.0) != 1.0 else demand.get("step_time_scale"), stride_frac=float(p["stride"]) if p.get("stride", STRIDE_FRAC) != STRIDE_FRAC else demand.get("stride_frac"))
    def __init__(self, task, horizon: Optional[float] = None):
        super().__init__(task, horizon); self._steprefs: Dict = {}
    def native(self) -> bool: return False         # the wrapped task supplies the plant only (bind_horizon not used)
    # ---- horizon and reference
    def H(self, demand) -> float:
        if self.horizon is not None: return float(self.horizon)
        h = demand.get("horizon")
        if isinstance(h, (list, tuple)): return float(h[1])
        if h is not None: return float(h)
        if demand.get("phases"): return float(sum(float(ph.get("horizon", 0.0)) for ph in demand["phases"]))     # a program that declares its horizon as phases (P70's carry)
        g = demand.get("goal"); return float(getattr(g, "horizon", 3.0) or 3.0)
    def _dkey(self, demand):
        g = demand.get("goal"); gk = None if g is None else json.dumps(dict(kind=g.kind, params=g.params), sort_keys=True, default=str)
        return (gk, json.dumps(demand.get("footsteps"), default=str), json.dumps(demand.get("path"), default=str), str(demand.get("direction")), str(demand.get("distance")), self.H(demand),
                json.dumps({k: demand.get(k) for k in REF_KEYS}, sort_keys=True, default=str))
    def ref(self, S, demand) -> dict:
        q = np.asarray(S["q_settled"], float); key = (self._dkey(demand), np.round(q[:min(len(q), 36)], 2).tobytes())
        if key not in self._steprefs:
            b = body_of(S); self._steprefs[key] = build_reference(b, q, demand, self.H(demand))
        return self._steprefs[key]
    def joints(self, S) -> List[str]:
        """Offset joints: the 12 leg joints + the task's driven non-leg joints (waist, arms)."""
        b = body_of(S); legs = [x[0] for i in (0, 1) for x in b.leg[i]]
        return legs + [t for t, g in zip(S["types"], S["groups"]) if g != "legs" and t not in legs]
    def nk(self, S, demand) -> int: return max(2, int(round(self.ref(S, demand)["T"]/KNOT_DT)) + 1)
    def n_params(self, S, demand): return len(self.joints(S))*self.nk(S, demand) + self.N_GLOBAL
    def unpack(self, S, demand, z):
        z = np.asarray(z, float); nJ = len(self.joints(S)); nk = self.nk(S, demand); kn = z[:nJ*nk]; g = z[nJ*nk:nJ*nk + self.N_GLOBAL]
        pace = self._level(g[3], self.PACE); stride = self._level(g[4], self.STRIDE); p = dict(pace=pace, stride=stride)
        ref = self.ref(S, self._dem(demand, p)); ts_max = max(1.0, self.H(demand)/max(ref["T"], 1e-6))
        ts = float(np.clip(1.0 + 0.15*g[0], 0.85, max(0.85, min(1.15, ts_max))))
        return dict(x=z.tolist(), knots=kn.tolist(), nk=nk, tscale=ts, k_cp_scale=float(1.0 + 0.5*g[1]), k_trunk_scale=float(1.0 + 0.5*g[2]), pace=pace, stride=stride)
    def controller(self, S, demand, p): return None
    def z0(self, S, demand): return np.zeros(self.n_params(S, demand))
    def sigma0(self, demand, warm): return 0.1
    def floor(self, demand): return None
    def warm(self, S, demand, stored): return np.asarray(stored["x"], float)
    def objective(self, rec, demand): return float(rec["extra"].get("goal_err", 0.0)) + 0.5*float(rec["extra"].get("placement_err_mean", 0.0) or 0.0) + 0.2*float(rec["extra"].get("v_end", 0.0))
    def describe(self, demand=None):
        return dict(super().describe(), params="knot offsets (12 legs + driven trunk/arm joints) x nk (one per 0.5 s of the prior's reference, over normalised reference time) +-0.15 rad; tscale 0.85-1.15; k_cp, k_trunk scales; pace level (step time x 0.7-1.3) and stride level (0.5/0.65/0.8 of reach), each a cached reference", prior="step (goal-derived footsteps, LIPM COM, swing IK)")
    # ---- rollout
    def rollout(self, S, demand, p, ctrl_fn=None, record=False):
        with self.start_ctx(S, demand): return self._rollout(S, demand, p, ctrl_fn, record)
    def _rollout(self, S, demand, p, ctrl_fn=None, record=False):
        import mujoco, branch_i as BI, plant as P, balance as B_
        from engine import core
        from validity import TrajectoryValidityReport
        m = S["m"]; d = mujoco.MjData(m); BI.settle(S); BI._set_capacity_limits(m, S, None); b = body_of(S)
        m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
        d.qpos[:] = S["q_settled"]; d.qvel[:] = S["v_settled"]; mujoco.mj_forward(m, d)
        ref = self.ref(S, self._dem(demand, p)); Q = ref["Q"]; fps = ref["fps"]; ts = float(p.get("tscale", 1.0)); T_ref = ref["T"]
        H_roll = T_ref*ts + 0.2; dt = m.opt.timestep; nst = int(round(H_roll/dt)); J = S["jarr"]; tau_v = J["tau"]
        rd = S.get("robot_dofs"); rj = [i for i in (rd if rd is not None else range(m.nv)) if i >= 6]
        E0 = float(d.energy[0] + d.energy[1]); start_qdot = float(np.max(np.abs(d.qvel[rj])))
        # knot offsets over the reference time
        jn = self.joints(S); nk = int(p["nk"]); kn = np.asarray(p["knots"], float).reshape(len(jn), nk)*KNOT_AMP; jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in jn]
        jact = np.array([b.act_of[j] for j in jid]); kt = np.linspace(0.0, T_ref, nk)
        # balance controller (stand mode) owns the legs; its stand pose follows the reference frame
        ctl = core.landing_controller(S, self.robot)[0]; ctl.p["mode"] = "stand"; ctl.t_touch = -1e9; ctl._kc = None
        ctl.p["k_cp"] = float(demand.get("k_cp", 0.5))*float(p.get("k_cp_scale", 1.0)); ctl.p["k_trunk"] = float(demand.get("k_trunk", 0.75))*float(p.get("k_trunk_scale", 1.0))
        d_r = mujoco.MjData(m); d_r.qpos[:] = Q[0]; mujoco.mj_kinematics(m, d_r); ctl.tilt_ref = ctl._tilt(d_r, ctl.tb)          # the reference's own torso tilt (constant: pelvis upright, waist held)
        ctl.p["k_trunk_d"] = 0.0; ctl.p["k_cp_d"] = 0.0                                   # the rate terms read world-frame rates (not re-expressed below)
        zr_ = _Rz(-_yaw(d_r.xmat[b.pel].reshape(3, 3))) @ d_r.xmat[ctl.tb].reshape(3, 3)[:, 2]; tilt_ref_b = np.array([np.arctan2(zr_[0], zr_[2]), -np.arctan2(zr_[1], zr_[2])])
        omega_s = float(demand.get("omega_servo", 60.0)); omega_legs = float(demand.get("omega_legs", ctl.p["stiff"]))
        ieff = np.array([1.0/float(S["Minv_key"][dof, dof]) for dof in J["dof"]]); is_leg = np.array([g == "legs" for g in S["groups"]])
        for i, aid in enumerate(J["aid"]):
            om = omega_legs if is_leg[i] else omega_s; kp = om*om*ieff[i]; BI._as_servo(m, int(aid), kp, 2.0*np.sqrt(kp*ieff[i]), float(tau_v[i]))
        for (a_, dof, qad, tau, kp, kd) in S["held"]: d.ctrl[a_] = S["q_settled"][qad]
        # the HELD leg joints (hip roll, ankle roll on the G1) get the same dt-INDEPENDENT stiffness as the driven legs (balance.all_servos'
        # lesson: kp ~ I/dt^2 does not transfer between timesteps; the held form made the 0.5 ms double support 4x stiffer laterally)
        leg_dofs = set(b.leg_v.tolist()); held_leg = [h for h in S["held"] if h[1] in leg_dofs]
        if demand.get("held_legs_physical", True):
            for (a_, dof, qad, tau, kp, kd) in held_leg:
                ie = 1.0/float(S["Minv_key"][dof, dof]); kp_ = omega_legs*omega_legs*ie; BI._as_servo(m, int(a_), kp_, 2.0*np.sqrt(kp_*ie), float(tau))
        # held objects (carry): engine.world weld engaged at the start (ContactSchedule convention; LABEL held_start_abstraction)
        wd = None; held = list(demand.get("held") or []); held_note = None
        if held and demand.get("world_objects"):
            from engine import world as EW
            if "md_world" not in S: S["md_world"] = EW.World(m, demand.get("world_objects"), demand.get("world_surfaces") or [], self.robot)
            wd = S["md_world"]
            for nm in held:
                if nm in wd.eq and not wd.held(d, nm):
                    if wd.palm_dist(d, nm) > 0.10:
                        jj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm + "_free"); a_ = m.jnt_qposadr[jj]
                        d.qpos[a_:a_ + 3] = d.site_xpos[wd.palm]; d.qpos[a_ + 3:a_ + 7] = [1, 0, 0, 0]; d.qvel[m.jnt_dofadr[jj]:m.jnt_dofadr[jj] + 6] = 0; mujoco.mj_forward(m, d); held_note = "held_start_abstraction"
                    wd.grasp(d, nm)
        robot_g = S.get("robot_geoms", set()); foot_g = [b.coll[0] | S["feet"][0], b.coll[1] | S["feet"][1]]
        W_act = W_con = W_pas = 0.0; nonfoot = False; nonfoot_what = None; peak_grf = 0.0; util = 0.0; work = np.zeros(len(J["aid"]))
        z0 = float(d.xpos[b.pel][2]); fell = False; t_fell = None; trace = []; loads = []; tg = np.array(d.ctrl); ff = [0.0, 0.0]; fz_prev = 1e3
        land_err = {}; land_rel = {}; steps = ref["steps"]; land_t = [(k, s["t_land"]*ts + 0.15) for k, s in enumerate(steps)]
        flight = 0.0; flight_max = 0.0; tilt_max = 0.0; BW = S["M"]*G
        d_x = mujoco.MjData(m); quat_ = np.zeros(4); q_leg_prev = None; hold_xy = [None, None]; EYE6 = 1e-4*np.eye(6)
        legset = set(b.leg_act[0]) | set(b.leg_act[1]); nonleg_act = np.array([a_ for a_ in range(m.nu) if a_ not in legset], int)
        nonleg_qa = np.array([m.jnt_qposadr[int(m.actuator_trnid[a_][0])] for a_ in nonleg_act], int)
        act_lo = np.array([m.jnt_range[int(m.actuator_trnid[a_][0])][0] for a_ in range(m.nu)]); act_hi = np.array([m.jnt_range[int(m.actuator_trnid[a_][0])][1] for a_ in range(m.nu)])
        is_robot = np.zeros(m.ngeom, bool); is_robot[list(robot_g)] = True; foot_of = -np.ones(m.ngeom, int)
        for i_ in (0, 1): foot_of[list(foot_g[i_])] = i_
        held_bodies = set()
        if wd is not None: held_bodies = {wd.bid.get(nm, -2) for nm in held}
        fv = np.zeros(6); t_blend = float(demand.get("t_touch_blend", 0.15)); st_prev = -1; td = None
        k_adm = float(demand.get("k_load", 0.03)); adm_max = float(demand.get("load_dz_max", 0.04)); delta = 0.0; ff_f = [0.5*S["M"]*G, 0.5*S["M"]*G]
        dbg = demand.get("_dbg"); h_td = float(demand.get("touchdown_height", 0.004)); early = -1; ff_sw_prev = [0.0, 0.0]
        i_seg = np.full(len(Q), -1); i_land = np.full(len(Q), 1e9)
        for k_, s_ in enumerate(steps):
            a0_, a1_ = int(round(s_["t_lift"]*fps)), int(round(s_["t_land"]*fps)); i_seg[a0_:a1_ + 1] = k_; i_land[a0_:a1_ + 1] = s_["t_land"]*fps
        jp_ = np.zeros((3, m.nv)); jr_ = np.zeros((3, m.nv)); sw_dof = {i: [x[3] for x in b.leg[i]] for i in (0, 1)}
        try:
            for s_ in range(nst):
                tt = s_*dt; tr_ = min(tt/ts, T_ref); fi = tr_*fps; i0 = min(int(fi), len(Q) - 1); i1 = min(i0 + 1, len(Q) - 1); a = fi - i0
                k_c = int(round(tt/CTRL_DT, 6))
                if k_c != ctl._kc:
                    qr = (1 - a)*Q[i0] + a*Q[i1]; st = int(ref["stance"][i0])
                    if st >= 0 and early >= 0 and early == i_seg[i0]: st = -1                       # this swing already touched down (contact before the reference's touchdown)
                    if st >= 0 and fi > i_land[i0] - 0.12*fps and ff_sw_prev[1 - st] > 0.08*BW: early = i_seg[i0]; st = -1
                    # ANCHORED ONLINE LEG IK. Per foot, the planar map (yaw + translation) taking the reference sole to the ACTUAL sole; the
                    # desired pelvis, capture point and swing sole are the reference ones mapped by the stance foot's anchor (double support:
                    # blended by the ZMP's position between the feet). Each leg is then solved (6-DOF IK, base at the desired pelvis): a support
                    # sole stays where it is (flat, at the planned ground height), the swing sole goes to its anchored reference pose. Raw
                    # reference joint angles dragged a landed, unloaded foot around once the body had drifted from the plan.
                    d_r.qpos[:] = qr; mujoco.mj_kinematics(m, d_r)
                    fa = [d.site_xpos[b.fsite[i]][:2] for i in (0, 1)]; fr = [d_r.site_xpos[b.fsite[i]][:2] for i in (0, 1)]; anc = []
                    for i in (0, 1):
                        dy_ = _wrap(_yaw(d.site_xmat[b.fsite[i]].reshape(3, 3)) - _yaw(d_r.site_xmat[b.fsite[i]].reshape(3, 3))); Rz_ = _Rz(dy_)
                        anc.append((dy_, Rz_, d.site_xpos[b.fsite[i]] - Rz_ @ d_r.site_xpos[b.fsite[i]]))
                    pz_ = ref["zmp"][i0]; sg_ = fr[1] - fr[0]; alpha = float(np.clip((pz_ - fr[0]) @ sg_/max(float(sg_ @ sg_), 1e-9), 0.0, 1.0))
                    wa = ([1.0, 0.0] if st == 0 else [0.0, 1.0]) if st >= 0 else [1.0 - alpha, alpha]
                    def amap(p_, wa=wa, anc=anc): return wa[0]*(anc[0][1] @ p_ + anc[0][2]) + wa[1]*(anc[1][1] @ p_ + anc[1][2])
                    dyaw_a = wa[0]*anc[0][0] + wa[1]*anc[1][0]; anc_z = wa[0]*anc[0][2][2] + wa[1]*anc[1][2][2]
                    P_des = amap(d_r.xpos[b.pel].copy()); P_des[2] = d_r.xpos[b.pel][2]; R_des = _Rz(dyaw_a) @ d_r.xmat[b.pel].reshape(3, 3)
                    w0 = np.sqrt(G/max(float(d.subtree_com[b.pel][2]), 0.2)); cp_r = ref["com"][i0] + ref["comd"][i0]/w0; cp_des = amap(np.r_[cp_r, 0.0])[:2]
                    # YAW-AWARE BALANCE through the controller's own hooks: balance.BalanceController maps the capture-point error and the trunk
                    # tilt to ankle / hip corrections in WORLD x / y (exact only at zero yaw: a robot turned 50 deg got corrections rotated by 50 deg).
                    # cp_offset / tilt_ref are chosen so that its errors come out expressed in the support frame (stance sole yaw; double support:
                    # the mean sole yaw). balance.py is not edited; its cp and tilt are recomputed here exactly as it computes them.
                    psi_s = (_yaw(d.site_xmat[b.fsite[st]].reshape(3, 3)) if st >= 0 else float(np.arctan2(sum(np.sin(_yaw(d.site_xmat[b.fsite[i]].reshape(3, 3))) for i in (0, 1)), sum(np.cos(_yaw(d.site_xmat[b.fsite[i]].reshape(3, 3))) for i in (0, 1)))))
                    Rs_ = _Rz(-psi_s)[:2, :2]; com_c = d.subtree_com[0]; vcom_c = P.com_velocity(m, d); cp_c = com_c[:2] + vcom_c[:2]/np.sqrt(G/max(com_c[2], 0.2))
                    fm_ = 0.5*(B_.foot_xy(d, ctl.feet[0]) + B_.foot_xy(d, ctl.feet[1]))
                    ctl.cp_offset = cp_c - fm_ - Rs_ @ (cp_c - cp_des); shift = cp_des - cp_r
                    zt_ = d.xmat[ctl.tb].reshape(3, 3)[:, 2]; zb_ = _Rz(-psi_s) @ zt_; tilt_b = np.array([np.arctan2(zb_[0], zb_[2]), -np.arctan2(zb_[1], zb_[2])])
                    ctl.tilt_ref = ctl._tilt(d, ctl.tb) - (tilt_b - tilt_ref_b)
                    d_x.qpos[:] = qr; d_x.qpos[b.qfree:b.qfree + 3] = P_des; mujoco.mju_mat2Quat(quat_, R_des.flatten()); d_x.qpos[b.qfree + 3:b.qfree + 7] = quat_
                    d_x.qpos[b.leg_q] = q_leg_prev if q_leg_prev is not None else qr[b.leg_q]
                    ftg = []
                    for i in (0, 1):
                        if st >= 0 and i == 1 - st:
                            p_ = amap(d_r.site_xpos[b.fsite[i]].copy()); p_[2] = d_r.site_xpos[b.fsite[i]][2] + (h_td if fi > i_land[i0] - 0.25*fps else 0.0)   # touchdown a few mm ABOVE the loaded stance's penetration depth
                            ftg.append((p_, _Rz(dyaw_a) @ d_r.site_xmat[b.fsite[i]].reshape(3, 3)))
                        else:
                            # a support sole is held at its LATCHED pose (start / touchdown), which follows the actual sole only while that sole
                            # carries load (a real slip is accepted; an unloaded sole is not allowed to wander with its own target)
                            if hold_xy[i] is None or (st_prev >= 0 and i == 1 - st_prev and st < 0): hold_xy[i] = np.r_[d.site_xpos[b.fsite[i]][:2], _yaw(d.site_xmat[b.fsite[i]].reshape(3, 3))]
                            elif ff_f[i] > 0.3*BW: hold_xy[i] += 0.05*(np.r_[d.site_xpos[b.fsite[i]][:2], _yaw(d.site_xmat[b.fsite[i]].reshape(3, 3))] - hold_xy[i])
                            p_ = np.r_[hold_xy[i][:2], d_r.site_xpos[b.fsite[i]][2]]; yr_ = _yaw(d_r.site_xmat[b.fsite[i]].reshape(3, 3))
                            ftg.append((p_, _Rz(_wrap(hold_xy[i][2] - yr_)) @ d_r.site_xmat[b.fsite[i]].reshape(3, 3)))
                    for _it in range(3):
                        mujoco.mj_kinematics(m, d_x); mujoco.mj_comPos(m, d_x)
                        for i in (0, 1):
                            s_i = b.fsite[i]; e6 = np.concatenate((ftg[i][0] - d_x.site_xpos[s_i], _rot_err(d_x.site_xmat[s_i].reshape(3, 3), ftg[i][1])))
                            mujoco.mj_jacSite(m, d_x, jp_, jr_, s_i); J6 = np.vstack([jp_[:, sw_dof[i]], jr_[:, sw_dof[i]]])
                            d_x.qpos[b.leg_qa[i]] += np.clip(np.linalg.solve(J6.T @ J6 + EYE6, J6.T @ e6), -0.2, 0.2)
                    d_x.qpos[b.leg_q] = np.clip(d_x.qpos[b.leg_q], b.lo, b.hi); q_leg_prev = d_x.qpos[b.leg_q].copy()
                    q_st = qr.copy(); q_st[b.leg_q] = q_leg_prev; ctl.q_stand = q_st
                    ctl._kc = k_c; tg = ctl.targets(k_c*CTRL_DT, d, fz_prev > 1.0)
                    if st >= 0:                                   # the swing leg: its IK solution (no balance correction on a foot in the air)
                        sw = 1 - st; hold_xy[sw] = None
                        for a_, qa_ in zip(b.leg_act[sw], b.leg_qa[sw]): tg[a_] = q_st[qa_]
                        sw_tg = np.array([tg[a_] for a_ in b.leg_act[sw]])
                    elif st_prev >= 0:                            # touchdown: the landed leg blends from its swing targets into the balance controller's
                        td = (k_c*CTRL_DT, 1 - st_prev, sw_tg.copy())
                    if st < 0 and td is not None and k_c*CTRL_DT - td[0] < t_blend:
                        a_b = _mj_s((k_c*CTRL_DT - td[0])/t_blend)
                        for j_, a_ in enumerate(b.leg_act[td[1]]): tg[a_] = (1 - a_b)*td[2][j_] + a_b*tg[a_]
                    # LOAD DISTRIBUTION in double support (admittance on the leg lengths): the reference ZMP's position between the feet sets each
                    # foot's share of the load; the legs lengthen / shorten differentially until the measured share follows it. Two stiff
                    # position-servoed legs otherwise split the load by millimetre geometry, not by the commanded ZMP (a landing foot took 100%).
                    if st < 0 and k_adm > 0:
                        share = ff_f[1]/max(ff_f[0] + ff_f[1], 1.0) if (ff_f[0] + ff_f[1]) > 0.2*BW else 0.5
                        delta = float(np.clip(delta + k_adm*(alpha - share)*CTRL_DT, -adm_max, adm_max))
                    else: delta *= 0.9
                    if abs(delta) > 1e-5:
                        for i_ in (0, 1):
                            mujoco.mj_jacSite(m, d, jp_, jr_, b.fsite[i_]); J6 = np.vstack([jp_[:, sw_dof[i_]], jr_[:, sw_dof[i_]]])
                            e6 = np.r_[0.0, 0.0, (-delta if i_ == 1 else delta), 0.0, 0.0, 0.0]          # +delta: the right sole goes DOWN relative to the pelvis, the left UP
                            dqi = np.linalg.solve(J6.T @ J6 + EYE6, J6.T @ e6)
                            for j_, a_ in enumerate(b.leg_act[i_]): tg[a_] += dqi[j_]
                    st_prev = st
                    if dbg is not None:
                        dbg.append(dict(t=tt, st=st, fi=fi, delta=delta, ff=list(ff_f), com=d.subtree_com[b.pel][:2].copy(), cp_t=(cp_r + shift).copy(), feet=[d.site_xpos[b.fsite[i]].copy() for i in (0, 1)], feet_r=[d_r.site_xpos[b.fsite[i]].copy() for i in (0, 1)],
                                        fyaw=[_yaw(d.site_xmat[b.fsite[i]].reshape(3, 3)) for i in (0, 1)], q_leg=d.qpos[b.leg_q].copy(), tg_leg=np.array([tg[a_] for i in (0, 1) for a_ in b.leg_act[i]]), qr_leg=qr[b.leg_q].copy(), pel=d.qpos[:7].copy()))
                    tg[nonleg_act] = qr[nonleg_qa]
                    kk = np.interp(tr_, kt, np.arange(nk)); k0_ = min(int(kk), nk - 1); k1_ = min(k0_ + 1, nk - 1); f_ = kk - k0_
                    off = kn[:, k0_]*(1 - f_) + kn[:, k1_]*f_; tg[jact] += off
                    d.ctrl[:] = np.clip(tg, act_lo, act_hi)
                if ctrl_fn is not None: ctrl_fn(tt, d, S)          # hooks (mocap obstacles, wrenches) + observers; servo mode ignores the return
                BI.backemf_brake(m, d, S)
                qv0 = d.qvel.copy(); mujoco.mj_step(m, d); qm = 0.5*(qv0 + d.qvel)
                tq = d.actuator_force[J["aid"]]; util = max(util, float(np.max(np.abs(tq)/tau_v)))
                W_act += float(np.dot(d.actuator_force, qm[S["act_dof"]]))*dt; W_con += float(np.dot(d.qfrc_constraint, qm))*dt; W_pas += float(np.dot(d.qfrc_passive, qm))*dt + float(np.dot(d.qfrc_applied, qm))*dt
                pw = tq*d.qvel[J["dof"]]; work += np.where(pw > 0, pw, 0.0)*dt
                ff = [0.0, 0.0]; fz = 0.0; nc = d.ncon
                if nc:
                    G1 = d.contact.geom1[:nc]; G2 = d.contact.geom2[:nc]; R1 = is_robot[G1]; R2 = is_robot[G2]
                    for ci in np.nonzero(R1 != R2)[0]:            # robot-world contacts only
                        o, w_ = (int(G1[ci]), int(G2[ci])) if R1[ci] else (int(G2[ci]), int(G1[ci])); fo = foot_of[o]
                        if fo >= 0: mujoco.mj_contactForce(m, d, int(ci), fv); ff[fo] += fv[0]; fz += fv[0]
                        elif int(m.geom_bodyid[w_]) not in held_bodies:
                            if not nonfoot: nonfoot_what = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, o), round(tt, 3))
                            nonfoot = True
                peak_grf = max(peak_grf, fz); fz_prev = fz; ff_sw_prev = ff; a_f = min(1.0, dt/0.02); ff_f = [ff_f[0] + a_f*(ff[0] - ff_f[0]), ff_f[1] + a_f*(ff[1] - ff_f[1])]
                if fz < 1.0: flight += dt; flight_max = max(flight_max, flight)
                else: flight = 0.0
                up = float(d.xmat[b.torso][8]); tilt_max = max(tilt_max, float(np.arccos(np.clip(up, -1, 1))))
                if not fell and (float(d.xpos[b.pel][2]) < z0 - 0.35 or up < 0.5): fell = True; t_fell = tt
                for (k, tl) in land_t:
                    if k not in land_err and tt >= tl:
                        sd = SIDES.index(steps[k]["foot"]); tgt = np.asarray(steps[k]["target"][:2]); pa = d.site_xpos[b.fsite[sd]][:2]; land_err[k] = float(np.linalg.norm(pa - tgt))
                        land_rel[k] = float(ff[sd]/BW)                 # landing foot load (BW) at the check instant
                if record and s_ % 10 == 0: trace.append(d.qpos.copy()); loads.append([ff[0]/BW, ff[1]/BW])
                if fell and tt > t_fell + 0.3: break
        finally:
            for i, aid in enumerate(J["aid"]): BI._as_motor(m, int(aid), float(tau_v[i]))
            for (a_, dof, qad, tau, kp, kd) in held_leg: BI._as_servo(m, int(a_), kp, kd, float(tau))
        E1 = float(d.energy[0] + d.energy[1]); resid = (E1 - E0) - W_act; closure = resid - W_con - W_pas
        rep = TrajectoryValidityReport(start_qdot_max=(start_qdot if start_qdot >= 0.5 else min(start_qdot, 0.04)), energy_residual=resid, energy_tol=max(1.0, 0.02*abs(W_act)), actuator_work=W_act,
                                       contact_energy=0.0, ke_takeoff=0.0, nonfoot_contact=nonfoot, took_off=True, peak_grf_bw=peak_grf/BW,
                                       energy_closure=closure, closure_tol=max(5.0, 0.02*(abs(W_act) + abs(W_con)))).evaluate()
        if demand.get("initial_state") is not None: rep.reasons = [r for r in rep.reasons if r != "START_NOT_AT_REST"]       # a handed-off start is mid-motion by construction
        # end state: at rest, both feet loaded, capture point inside the feet's support
        com = d.subtree_com[b.pel].copy(); vcom = P.com_velocity(m, d); w0 = np.sqrt(G/max(com[2], 0.2)); cp = com[:2] + vcom[:2]/w0
        fa = [d.site_xpos[b.fsite[i]][:2] for i in (0, 1)]; seg = fa[1] - fa[0]; u_ = np.clip(float((cp - fa[0]) @ seg)/max(float(seg @ seg), 1e-9), 0, 1); cp_err = float(np.linalg.norm(cp - (fa[0] + u_*seg)))
        v_end = float(np.linalg.norm(vcom[:2])); end_ok = (not fell) and v_end < 0.15 and min(ff) > 0.05*BW and cp_err < 0.08
        reasons = list(rep.reasons) + (["FELL"] if fell else []) + ([] if (end_ok or fell) else ["END_UNBALANCED"])
        viol = 0.0
        if fell: viol += 5.0
        if nonfoot: viol += 5.0
        if "END_UNBALANCED" in reasons: viol += 1.0 + 5.0*max(0.0, v_end - 0.15) + 10.0*max(0.0, cp_err - 0.08)
        if any(z_ in rep.reasons for z_ in ("ENERGY_CLOSURE", "ENERGY_ACCOUNTING_VIOLATION")): viol += 2.0
        if "START_NOT_AT_REST" in reasons: viol += 1.0
        # the goal measure (the program's goal as the representation's value; the PROGRAM's evaluate stays the judge)
        q0 = np.asarray(S["q_settled"]); pel = d.xpos[b.pel].copy(); disp = pel - q0[b.qfree:b.qfree + 3]
        gs = ref["goal"]; ge = 0.0
        if gs["kind"] == "point": ge = float(np.linalg.norm(pel[:2] - np.asarray(gs["xy"])))
        elif gs["kind"] == "x_min": ge = float(max(0.0, gs["x_min"] - pel[0]))
        elif gs["kind"] == "heading": ge = abs(_wrap(_yaw(d.xmat[b.pel].reshape(3, 3)) - gs["yaw"]))
        elif gs["kind"] in ("velocity", "arc"):
            tgt_last = np.asarray(ref["waypoints"][-1][:2]) if ref["waypoints"] else pel[:2]; ge = float(np.linalg.norm(pel[:2] - tgt_last))
        elif gs["kind"] == "footsteps": ge = float(np.mean(list(land_err.values()))) if land_err else 0.0
        gw = {}
        for g_, wv in zip(S["groups"], work): gw[g_] = gw.get(g_, 0.0) + float(wv)
        tot = float(sum(gw.values()))
        pl = [land_err[k] for k in sorted(land_err)]
        extra = dict(goal=gs, goal_err=ge, base_displacement=disp.tolist(), n_steps=ref["n_steps"], steps=[dict(s, placement_err=land_err.get(k), landing_load_bw=land_rel.get(k), landed=(k in land_err)) for k, s in enumerate(steps)],
                     placement_err=pl, placement_err_mean=(float(np.mean(pl)) if pl else None), placement_err_max=(float(np.max(pl)) if pl else None), steps_landed=len(pl),
                     v_end=v_end, cp_err_end=cp_err, feet_load_end_bw=[ff[0]/BW, ff[1]/BW], fell=fell, t_fell=t_fell, nonfoot=nonfoot_what, flight_max=flight_max, tilt_max=tilt_max,
                     peak_util=util, work=gw, W_act=W_act, closure=closure, peak_grf_bw=peak_grf/BW, tscale=ts, t_rel=float(nst*dt), t_total=float(nst*dt), heading_mode=ref["heading_mode"],
                     timing=ref["timing"], limits=ref["limits"], ik_res_max=ref["ik_res_max"], clearance=ref["clearance"], final_state=(d.qpos.copy(), d.qvel.copy()),
                     held_after=([nm for nm in held if wd is not None and wd.held(d, nm)] if held else None), held_note=held_note,
                     phases=[dict(phase=0, support="steps", horizon=float(nst*dt), held_after=([nm for nm in held if wd is not None and wd.held(d, nm)] if held else None))])
        out = dict(value=-ge, valid=(viol <= 0 and not reasons), reasons=reasons, violation=float(viol), cost=tot, final_state=(d.qpos.copy(), d.qvel.copy()), extra=extra)
        if record: out["trace"] = dict(q=np.asarray(trace).tolist(), foot_load_bw=loads, dt=10*dt)
        return out
