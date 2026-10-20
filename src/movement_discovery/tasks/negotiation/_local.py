"""Negotiation-family helpers: obstacle worlds (world.obstacles.* dicts go in WorldSpec.mechanisms), clearance tests on the pelvis
trace and, where a model is at hand, on body sites. Policy: a non-foot contact is a violation unless the program declares the
contact (hand on a rail: P85). MuJoCo inside functions only."""
from __future__ import annotations
import numpy as np
from ..locomotion._local import (BodyProgram, terrain, obstacle, impulse, q_trace, pelvis_kin, upright, base_validity, no_trace, fail, schedule_demand, phase, CS_NAME, foot_loads)  # noqa: F401


def travel_goal(p, GoalSpec):
    """Reach x >= x_goal at rest, upright (the common 'get past it' goal)."""
    return GoalSpec(kind="com_displacement", params=dict(x_min=p["x_goal"], terminal="rest"), tolerance=dict(v_rest=0.2), horizon=p["horizon"][1])

def past(q, dt, x_goal, t_rest=0.3):
    """(reached, measures): pelvis x beyond x_goal, at rest and upright over the last t_rest s."""
    xyz, yaw, tilt, v = pelvis_kin(q, dt); n = max(1, int(round(t_rest/dt))); v_end = float(np.linalg.norm(v[-n:], axis=1).max()); up, m = upright(q, dt, T=t_rest)
    return bool(xyz[-1, 0] >= x_goal and v_end <= 0.2 and up), dict(x_end=float(xyz[-1, 0]), short=max(0.0, x_goal - float(xyz[-1, 0])), v_end=v_end, **m)

def sites_xyz(S, q, names):
    import mujoco; m = S["m"]; d = mujoco.MjData(m); ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in names]; out = []
    for r in q: d.qpos[:] = r; mujoco.mj_kinematics(m, d); out.append([d.site_xpos[i].copy() for i in ids])
    return np.asarray(out)


class NegotiationProgram(BodyProgram):
    def strategy_space(self, scenario): return [CS_NAME, "servo_knots+step_prior"]
    def demand(self, scenario):
        p = scenario.params; g = self.goal(scenario)
        return schedule_demand(scenario, p["horizon"], [phase("feet_any", g)], g, effect=dict(effector="pelvis", direction=(0.0, 0.0)), notes=self.notes)
    notes = "feet only; any contact sequence"
