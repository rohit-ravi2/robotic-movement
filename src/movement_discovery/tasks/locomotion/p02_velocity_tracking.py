"""Program 02 (locomotion): velocity tracking. Hold a commanded planar base velocity (speed and direction) over a window after a free
transient. Outcome = mean velocity error over the tracking window. No gait is specified: any foot-contact sequence that produces the
velocity is a solution. integration check: 2026-10-12 round 2, runner.check (contact_schedule+knots over BaseEffectTask, 4 sampled phases, prior-only = standing): no violations, velocity error 1.06 m/s (stands still, as a zero-knot prior must); 766 params; 13 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import BodyProgram, terrain, q_trace, pelvis_kin, upright, base_validity, no_trace, fail, schedule_demand, phase, CS_NAME


@register
class VelocityTracking(BodyProgram):
    id = "P02_velocity_tracking"; family = Family.LOCOMOTION; title = "Sustained tracking of a commanded base velocity"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P01 scores where the base ends; this scores the velocity held over a window (steady state), direction included."
    def world(self, scenario): return WorldSpec(terrain=terrain("flat"), tag="flat")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); sp = float(rng.uniform(0.3, 1.5)); ang = float(rng.uniform(-np.pi/2, np.pi/2))
        return self.scenario(seed, dict(v_cmd=[sp*np.cos(ang), sp*np.sin(ang)], t_transient=1.0, t_window=2.0, tol=0.15*sp + 0.05, horizon=[3.0, 4.0]), **ov)
    def goal(self, scenario):
        p = scenario.params; return GoalSpec(kind="com_velocity", params=dict(v_xy=p["v_cmd"], window=[p["t_transient"], p["t_transient"] + p["t_window"]]), tolerance=dict(v=p["tol"]), horizon=p["horizon"][1])
    def strategy_space(self, scenario): return [CS_NAME, "servo_knots+step_prior"]
    def demand(self, scenario):
        p = scenario.params; return schedule_demand(scenario, p["horizon"], [phase("feet_any", self.goal(scenario))], self.goal(scenario), effect=dict(effector="pelvis", direction=(0.0, float(np.arctan2(p["v_cmd"][1], p["v_cmd"][0])))))
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; xyz, yaw, tilt, v = pelvis_kin(q, dt); i0 = int(p["t_transient"]/dt); i1 = min(len(q), int((p["t_transient"] + p["t_window"])/dt))
        if i1 - i0 < 10: return no_trace("TRACE_SHORTER_THAN_WINDOW")
        e = np.linalg.norm(v[i0:i1] - np.asarray(p["v_cmd"]), axis=1); err = float(e.mean()); up, m = upright(q, dt, T=0.5)
        val = base_validity(rec)
        if not up: fail(val, "NOT_UPRIGHT_AT_END")
        return Outcome(value=-err, achieved=(err <= p["tol"] and val.ok), measures=dict(err_mean=err, err_max=float(e.max()), v_mean=v[i0:i1].mean(0).tolist(), **m), validity=val)
