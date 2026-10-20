"""Aerial-family helpers: the JumpTask bridge (tasks/jump.py: torque knots + tracked 'ascent' prior + the Phase 14 landing controller) and
readers for its record (format B in locomotion/_local.py). MuJoCo only inside functions."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import ValidityReport
from ..locomotion._local import (BodyProgram, terrain, q_trace, pelvis_kin, upright, base_validity, no_trace, fail, schedule_demand, phase, quat_to_yaw_tilt, CS_NAME)  # noqa: F401
G = 9.80665


def jump_task(scenario, land=True):
    from tasks.jump import JumpTask
    t = JumpTask(scenario.params.get("robot", "G1")); return t, dict(t.demand_default(), land=land, prior=scenario.params.get("prior", "ascent"))

def flight(rec):
    """Height from the takeoff speed (rec['value'], m/s) or from the COM trace; landing summary; the post-takeoff pelvis trace."""
    ex = (rec or {}).get("extra") or {}; v = rec.get("value") if rec else None; tr = rec.get("trace") if rec else None
    h = (float(v)**2/(2*G)) if v is not None else None
    if h is None and isinstance(tr, dict) and tr.get("z") is not None: z = np.asarray(tr["z"], float); h = float(z.max() - z[0])
    L = ex.get("landing") or {}; q, dt = q_trace(rec)
    return dict(height=h, landed=bool(L.get("ok", False)), touchdown_t=L.get("touchdown_t"), up_end=L.get("up_end"), z_end=L.get("z_end"), v_end=L.get("v_end"), rot=ex.get("rot"), q=q, dt=dt, t0=(ex.get("land_trace") or [(0.0, None)])[0][0])

def landing_displacement(f):
    """Pelvis xy at touchdown minus pelvis xy at the first post-takeoff sample, and the yaw change between them."""
    if f["q"] is None or f["touchdown_t"] is None: return None, None
    ts = f["t0"] + np.arange(len(f["q"]))*f["dt"]; i = int(np.searchsorted(ts, f["touchdown_t"])); i = min(max(i, 1), len(f["q"]) - 1)
    yaw0, _ = quat_to_yaw_tilt(f["q"][0, 3:7]); yaw1, _ = quat_to_yaw_tilt(f["q"][i, 3:7])
    return (f["q"][i, :2] - f["q"][0, :2]), float((yaw1 - yaw0 + np.pi) % (2*np.pi) - np.pi)

def aerial_validity(rec, f, need_landing=True):
    val = base_validity(rec); val.reasons = [r for r in val.reasons if r != "NOT_LANDED"]
    if need_landing and not f["landed"]: fail(val, "NOT_LANDED"); val.support = False
    if f["height"] is None or f["height"] <= 0: fail(val, "NO_FLIGHT")
    return val

def airborne(S, q, lift=0.01):
    """Per sample: both foot sites higher than their settled height + lift (JumpTask records carry no foot-force trace)."""
    import mujoco; m = S["m"]; ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in ("left_foot", "right_foot")]; d = mujoco.MjData(m)
    d.qpos[:] = S["q_settled"]; mujoco.mj_kinematics(m, d); z0 = [float(d.site_xpos[i][2]) for i in ids]; out = []
    for r in q: d.qpos[:] = r; mujoco.mj_kinematics(m, d); out.append(all(d.site_xpos[i][2] > z0[k] + lift for k, i in enumerate(ids)))
    return np.asarray(out, bool)

def site_xyz(S, q, site):
    import mujoco; m = S["m"]; d = mujoco.MjData(m); sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site); out = []
    for r in q: d.qpos[:] = r; mujoco.mj_kinematics(m, d); out.append(d.site_xpos[sid].copy())
    return np.asarray(out)


class AerialProgram(BodyProgram):
    observe_site = "right_foot"
    def strategy_space(self, scenario): return ["torque_knots+tracked_prior", "torque_knots"]
    def demand(self, scenario): return jump_task(scenario, land=True)
    def observe(self, rec, scenario):
        f = flight(rec); base = super().observe(rec, scenario) or {}
        return base | dict(height=f["height"], landed=f["landed"], flight_rotation=f["rot"], crouch=((rec or {}).get("extra") or {}).get("crouch"), T_push=((rec or {}).get("extra") or {}).get("T_push"))
