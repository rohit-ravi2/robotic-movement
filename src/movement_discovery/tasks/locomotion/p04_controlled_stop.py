"""Program 04 (locomotion): controlled deceleration to a stop. The rollout STARTS MOVING (initial base velocity v0 is part of the
scenario: the construction must start from a handed-off moving state, engine.core.handoff) and must come to rest upright in the
shortest distance. Outcome = -stopping distance. integration check: 2026-10-12 world builds; demand = contact_schedule+knots over BaseEffectTask (sampled schedule, prior-only = standing phases); not run (one rollout per family representative: P02)."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import BodyProgram, terrain, q_trace, pelvis_kin, upright, base_validity, no_trace, fail, schedule_demand, phase, CS_NAME


@register
class ControlledStop(BodyProgram):
    id = "P04_controlled_stop"; family = Family.LOCOMOTION; title = "Deceleration from a moving state to rest in minimum distance"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P01/P03 start at rest; this starts with momentum and the objective is the braking distance, not a position."
    def world(self, scenario): return WorldSpec(terrain=terrain("flat"), friction=scenario.params["friction"], tag="flat")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); sp = float(rng.uniform(0.5, 2.0))
        return self.scenario(seed, dict(v0=[sp, float(rng.uniform(-0.2, 0.2))*sp], friction=float(rng.choice([0.5, 0.9])), v_rest=0.1, t_rest=0.4, horizon=[0.8, 2.5]), **ov)
    def goal(self, scenario):
        p = scenario.params; return GoalSpec(kind="stability", params=dict(initial_state=dict(vxy=p["v0"]), terminal="rest", objective="min_stop_distance"), tolerance=dict(v_rest=p["v_rest"]), horizon=p["horizon"][1])
    def strategy_space(self, scenario): return [CS_NAME]
    def demand(self, scenario):
        p = scenario.params; return schedule_demand(scenario, p["horizon"], [phase("feet_any", self.goal(scenario), initial_state=dict(vxy=p["v0"]))], self.goal(scenario), notes="initial base velocity v0 must be realised by the construction before t=0 (handoff), not teleported", effect=dict(effector="pelvis", direction=(0.0, float(np.arctan2(p["v0"][1], p["v0"][0]) + np.pi))))
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; xyz, yaw, tilt, v = pelvis_kin(q, dt); sp = np.linalg.norm(v, axis=1); n = max(1, int(round(p["t_rest"]/dt)))
        idx = [i for i in range(len(sp) - n + 1) if (sp[i:i+n] <= p["v_rest"]).all()]; i_stop = idx[0] if idx else None
        dist = float(np.linalg.norm(xyz[(i_stop if i_stop is not None else -1), :2] - xyz[0, :2])); up, m = upright(q, dt, T=p["t_rest"]); val = base_validity(rec)
        if abs(float(sp[0]) - np.linalg.norm(p["v0"])) > 0.3: fail(val, "INITIAL_VELOCITY_NOT_REALISED", abs(float(sp[0]) - np.linalg.norm(p["v0"])))
        if not up: fail(val, "NOT_UPRIGHT_AT_END")
        if i_stop is None: fail(val, "NOT_STOPPED", float(sp[-n:].max()))
        return Outcome(value=-dist, achieved=(i_stop is not None and val.ok), measures=dict(stop_distance=dist, t_stop=(None if i_stop is None else i_stop*dt), v_start=float(sp[0]), **m), validity=val)
