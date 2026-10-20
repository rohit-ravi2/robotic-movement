"""Posture-family helpers: INITIAL CONFIGURATIONS (seated, kneeling, prone, supine) as the problem's initial condition, built from
joint angles + base pose and settled under gravity on the scene (plant build check), and the standing test. Body-floor contact on the
G1 exists through the base scene's explicit floor pairs (pelvis, torso, thighs, shins, hands, head); world pieces replicate them, so a
seat or the floor carries the body. Rollouts that flag NONFOOT_CONTACT are not invalid here (declared per program). MuJoCo inside
functions only."""
from __future__ import annotations
import numpy as np
from ..locomotion._local import (BodyProgram, terrain, q_trace, pelvis_kin, upright, base_validity, no_trace, fail, schedule_demand, phase, CS_NAME, quat_to_yaw_tilt, foot_loads)  # noqa: F401

# joint-angle sets of the initial postures (rad; G1 convention: hip_pitch negative = flexion, knee positive = flexion, ankle_pitch
# negative = dorsiflexion). These are INITIAL CONDITIONS of the problem, not targets of the movement.
POSTURES = {
    "seated":   dict(joints={"hip_pitch": -1.50, "knee": 1.50, "ankle_pitch": 0.0, "shoulder_pitch": 0.2}, rpy=(0.0, 0.0, 0.0), z=0.48),   # knee re-solved from the seat height (seated_knee)
    "kneeling": dict(joints={"hip_pitch": -0.10, "knee": 2.00, "ankle_pitch": 0.90, "shoulder_pitch": 0.2}, rpy=(0.0, 0.0, 0.0), z=0.50),   # settles on shins + feet (check 2026-10-12: pelvis 0.48, tilt 11 deg)
    "prone":    dict(joints={"hip_pitch": 0.0, "knee": 0.1, "shoulder_pitch": -2.6, "elbow": 0.3}, rpy=(0.0, 1.45, 0.0), z=0.20),
    "supine":   dict(joints={"hip_pitch": -0.1, "knee": 0.1, "shoulder_pitch": 0.3, "elbow": 0.3}, rpy=(0.0, -1.45, 0.0), z=0.20),
}

def _quat(rpy):
    r, p, y = rpy; cr, sr, cp, sp, cy, sy = np.cos(r/2), np.sin(r/2), np.cos(p/2), np.sin(p/2), np.cos(y/2), np.sin(y/2)
    return np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])

PELVIS_GEOMS = ("pelvis_collision", "left_hip_collision", "right_hip_collision")
SEAT_RANGE = (0.22, 0.32)        # seat tops the G1 can sit on with the thighs horizontal and the feet on the floor (pelvis bottom 0.33 at knee 1.5 -> 0.21 at knee 2.5)

def _placed(S, q):
    import mujoco, plant as P; m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_forward(m, d); q = q.copy(); q[2] -= min(P.geom_low(m, d, g) for g in S["robot_geoms"]); d.qpos[:] = q; mujoco.mj_forward(m, d); return q, d

def seated_knee(S, seat_h, hip=-1.5):
    """Knee angle (ankle = -(knee - 1.5), sole level) at which the pelvis bottom rests on a seat of height seat_h with the feet on the floor."""
    import mujoco, plant as P; m = S["m"]; pel = [g for g in S["robot_geoms"] if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "") in PELVIS_GEOMS]
    def bottom(k):
        q = np.array(S["q0"], float)
        for side in ("left", "right"):
            for jn, v in (("hip_pitch", hip), ("knee", k), ("ankle_pitch", 1.5 - k)):
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{jn}_joint"); lo, hi = m.jnt_range[j]; q[m.jnt_qposadr[j]] = float(np.clip(v, lo + 0.02, hi - 0.02))
        q, d = _placed(S, q); return min(P.geom_low(m, d, g) for g in pel)
    lo, hi = 1.5, 2.6
    for _ in range(25):
        k = 0.5*(lo + hi); lo, hi = (k, hi) if bottom(k) > seat_h + 0.004 else (lo, k)
    return 0.5*(lo + hi)

def posed_qpos(S, posture, seat_h=None):
    """qpos for a named posture on the plant S: bilateral joints set by role, base orientation from rpy; a seated start solves the knee
    from the seat height (pelvis bottom 4 mm above the seat top, feet on the floor)."""
    import mujoco; m = S["m"]; q = np.array(S["q0"], float); P = dict(POSTURES[posture]); P["joints"] = dict(P["joints"])
    if posture == "seated" and seat_h is not None: k = seated_knee(S, seat_h); P["joints"].update(knee=k, ankle_pitch=1.5 - k)
    for jn_role, val in P["joints"].items():
        for side in ("left", "right"):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{jn_role}_joint")
            if j >= 0: lo, hi = m.jnt_range[j]; q[m.jnt_qposadr[j]] = float(np.clip(val, lo + 0.02, hi - 0.02))
    q[3:7] = _quat(P["rpy"]); q[2] = float(P["z"] if seat_h is None else seat_h + 0.12)
    return q

def posed_plant(robot, posture, spec=None, seat_h=None, dt=0.001):
    """Plant settled in the posture on the world (MJCF from world.build). The plant's settle rests the lowest geom on the floor plane
    and holds every joint; the record of the settle (pelvis height, residual speed) is the plant-build check for the posture."""
    import throw as TH, branch_i as BI, mujoco
    from movement_discovery import world as W
    mjcf = W.build(spec, robot) if spec is not None and ((spec.terrain and spec.terrain.get("kind") != "flat") or spec.mechanisms or spec.objects or spec.surfaces) else None
    S0 = TH.setup(robot, ball=False, mjcf=mjcf) if mjcf else TH.setup(robot, ball=False)
    q = posed_qpos(S0, posture, seat_h); S = TH.setup(robot, start=q, ball=False, mjcf=mjcf) if mjcf else TH.setup(robot, start=q, ball=False)
    BI.apply_dt(S, dt); BI.settle(S); d = mujoco.MjData(S["m"]); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(S["m"], d)
    return S, dict(pelvis_z=float(d.qpos[2]), tilt_deg=float(np.degrees(quat_to_yaw_tilt(d.qpos[3:7])[1])), settle_qdot=float(S.get("settle_qdot_max", np.nan)), com_z=float(d.subtree_com[0][2]))

def standing(q, dt, z_stand, hold=0.4, tol_z=0.06, tilt_max=np.radians(15), v_rest=0.15):
    """First time the pelvis is within tol_z of the standing height, tilt < tilt_max and speed < v_rest, held for `hold` s; None if never."""
    xyz, yaw, tilt, v = pelvis_kin(q, dt); sp = np.linalg.norm(v, axis=1); ok = (np.abs(xyz[:, 2] - z_stand) < tol_z) & (tilt < tilt_max) & (sp < v_rest); n = max(1, int(round(hold/dt)))
    for i in range(len(ok) - n + 1):
        if ok[i:i+n].all(): return float(i*dt)
    return None

def posture_validity(rec):
    """Base validity with NONFOOT_CONTACT removed (body contact with the floor or the seat is the task) and FELL ignored (lying starts)."""
    v = base_validity(rec); v.reasons = [r for r in v.reasons if r not in ("NONFOOT_CONTACT", "FELL", "STANCE_LOST")]; v.contact = True; v.support = True; return v


class PostureProgram(BodyProgram):
    Z_STAND = 0.72            # G1 settled standing pelvis height (m); scenarios may override with params["z_stand"]
    def strategy_space(self, scenario): return ["torque_knots", CS_NAME, "servo_knots"]
    def start_posture(self, scenario): return scenario.params.get("start")
    def initial_plant(self, scenario):
        """Plant settled in the scenario's initial posture (the problem's initial condition); None = the canonical stance on the world."""
        robot = scenario.params.get("robot", "G1"); st = self.start_posture(scenario)
        if st is None:
            import throw as TH; from movement_discovery import world as W; w = self.world(scenario); mjcf = W.build(w, robot) if (w.terrain and w.terrain.get("kind") != "flat") or w.mechanisms or w.objects or w.surfaces else None
            return (TH.setup(robot, ball=False, mjcf=mjcf) if mjcf else TH.setup(robot, ball=False)), dict(note="canonical stance")
        return posed_plant(robot, st, self.world(scenario), seat_h=scenario.params.get("seat_h"))
    def demand(self, scenario):
        p = scenario.params; return schedule_demand(scenario, p["horizon"], [phase("any", self.goal(scenario), initial_posture=self.start_posture(scenario))], self.goal(scenario), allowed=("both", "left", "right", "any"), notes="hands and body may bear load; initial posture settled by posture._local.posed_plant")
