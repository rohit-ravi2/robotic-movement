"""Program 01 (locomotion): point-to-point travel. Move the base from rest to a target point in the ground plane and come to rest
there, by any foot-contact sequence (no gait or step pattern is specified). Outcome = terminal position error; validity = upright at
the end, at rest, feet-only support throughout. Representation: contact_schedule with a variable horizon ().
integration check: 2026-10-12 world builds; demand = contact_schedule+knots over BaseEffectTask (sampled schedule, prior-only = standing phases); not run (one rollout per family representative: P02)."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import BodyProgram, terrain, q_trace, pelvis_kin, upright, base_validity, no_trace, fail, schedule_demand, phase, CS_NAME


@register
class PointToPoint(BodyProgram):
    id = "P01_point_to_point"; family = Family.LOCOMOTION; title = "Point-to-point travel of the base over flat ground"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P02 scores a sustained velocity, not arrival; P04 starts moving. This one starts and ends at rest with a position goal."
    def world(self, scenario): return WorldSpec(terrain=terrain("flat"), friction=scenario.params.get("friction"), tag="flat")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); dist = float(rng.uniform(0.5, 3.0)); ang = float(rng.uniform(-np.pi, np.pi))
        return self.scenario(seed, dict(target=[dist*np.cos(ang), dist*np.sin(ang)], tol=0.10, v_rest=0.15, t_rest=0.3, friction=float(rng.choice([0.6, 0.9, 1.2])), horizon=[1.0, 2.0 + 2.5*dist]), **ov)
    def goal(self, scenario):
        p = scenario.params; return GoalSpec(kind="com_displacement", params=dict(target_xy=p["target"], terminal="rest"), tolerance=dict(pos=p["tol"], v_rest=p["v_rest"]), horizon=p["horizon"][1])
    def strategy_space(self, scenario): return [CS_NAME, "servo_knots+step_prior"]
    def demand(self, scenario):
        p = scenario.params; return schedule_demand(scenario, p["horizon"], [phase("feet_any", self.goal(scenario))], self.goal(scenario), notes="feet only; any contact sequence; must end at rest within tol", effect=dict(effector="pelvis", direction=(0.0, float(np.arctan2(p["target"][1], p["target"][0])))))
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; xyz, yaw, tilt, v = pelvis_kin(q, dt); disp = xyz[-1, :2] - xyz[0, :2]; err = float(np.linalg.norm(disp - np.asarray(p["target"])))
        n = max(1, int(round(p["t_rest"]/dt))); v_end = float(np.linalg.norm(v[-n:], axis=1).max()); up, m = upright(q, dt, T=p["t_rest"])
        val = base_validity(rec)
        if not up: fail(val, "NOT_UPRIGHT_AT_END")
        if v_end > p["v_rest"]: fail(val, "NOT_AT_REST", v_end - p["v_rest"])
        return Outcome(value=-err, achieved=(err <= p["tol"] and val.ok), measures=dict(err=err, disp=disp.tolist(), v_end=v_end, duration=len(q)*dt, **m), validity=val)
