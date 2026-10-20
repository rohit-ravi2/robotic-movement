"""Program 07 (locomotion): slope traversal. Travel a commanded distance up or down a uniform incline (world terrain 'slope'); outcome =
along-slope displacement of the base toward the target, ending upright. The terrain is the physics that differs (gravity component along
the support, friction cone use). integration check: 2026-10-12 world builds; demand = contact_schedule+knots over BaseEffectTask (sampled schedule, prior-only = standing phases); not run (one rollout per family representative: P02)."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import BodyProgram, terrain, q_trace, pelvis_kin, upright, base_validity, no_trace, fail, schedule_demand, phase, CS_NAME


@register
class SlopeTraversal(BodyProgram):
    id = "P07_slope_traversal"; family = Family.LOCOMOTION; title = "Travel a set distance up or down a uniform incline"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE, Capability.ANKLE_ROLL]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P01 on the flat; here the support surface is inclined (angle, friction) and P08's support is discontinuous (steps)."
    def world(self, scenario):
        p = scenario.params; return WorldSpec(terrain=terrain("slope", angle=p["angle"], length=max(3.0, p["dist"] + 1.5), x0=0.4), friction=p["friction"], tag=f"slope{int(round(np.degrees(p['angle'])))}")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); ang = float(rng.choice([-1, 1])*np.radians(rng.uniform(5, 20)))
        return self.scenario(seed, dict(angle=ang, dist=float(rng.uniform(0.5, 2.0)), friction=float(rng.choice([0.6, 0.9])), tol=0.15, horizon=[1.5, 6.0]), **ov)
    def goal(self, scenario):
        p = scenario.params; return GoalSpec(kind="com_displacement", params=dict(along_slope=p["dist"], angle=p["angle"], terminal="rest"), tolerance=dict(pos=p["tol"]), horizon=p["horizon"][1])
    def strategy_space(self, scenario): return [CS_NAME, "servo_knots+step_prior"]
    def demand(self, scenario):
        p = scenario.params; return schedule_demand(scenario, p["horizon"], [phase("feet_any", self.goal(scenario))], self.goal(scenario), effect=dict(effector="pelvis", direction=(p["angle"], 0.0)))
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; xyz, yaw, tilt, v = pelvis_kin(q, dt); u = np.array([np.cos(p["angle"]), 0.0, np.sin(p["angle"])]); s = float((xyz[-1] - xyz[0]) @ u)
        err = abs(s - p["dist"]); lateral = float(abs(xyz[-1, 1] - xyz[0, 1])); n = max(1, int(0.3/dt)); v_end = float(np.linalg.norm(v[-n:], axis=1).max())
        up, m = upright(q, dt, T=0.3, tilt_max=np.radians(30) + abs(p["angle"])); val = base_validity(rec)   # the pelvis may lean with the slope
        if not up: fail(val, "NOT_UPRIGHT_AT_END")
        if v_end > 0.2: fail(val, "NOT_AT_REST", v_end - 0.2)
        return Outcome(value=-err, achieved=(err <= p["tol"] and val.ok), measures=dict(along_slope=s, lateral=lateral, v_end=v_end, **m), validity=val)
