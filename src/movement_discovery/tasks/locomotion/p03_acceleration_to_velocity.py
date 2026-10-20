"""Program 03 (locomotion): acceleration from rest to a terminal speed in minimum time. Outcome = time at which the base speed first
reaches and then holds the terminal speed for t_hold; the transient organisation is free. integration check: not executable until
construction/contact_schedule lands."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import BodyProgram, terrain, q_trace, pelvis_kin, upright, base_validity, no_trace, fail, schedule_demand, phase, CS_NAME


@register
class AccelerationToVelocity(BodyProgram):
    id = "P03_acceleration_to_velocity"; family = Family.LOCOMOTION; title = "Minimum-time acceleration from rest to a terminal speed"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P02 scores the steady state; this scores the transient (time to speed) and tolerates any steady-state error after it."
    def world(self, scenario): return WorldSpec(terrain=terrain("flat"), tag="flat")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); return self.scenario(seed, dict(v_term=float(rng.uniform(0.8, 2.0)), heading=float(rng.uniform(-0.5, 0.5)), t_hold=0.4, horizon=[1.0, 3.0]), **ov)
    def goal(self, scenario):
        p = scenario.params; return GoalSpec(kind="com_velocity", params=dict(speed=p["v_term"], heading=p["heading"], objective="min_time"), tolerance=dict(speed=0.1*p["v_term"], heading=0.3), horizon=p["horizon"][1])
    def strategy_space(self, scenario): return [CS_NAME, "servo_knots+step_prior"]
    def demand(self, scenario):
        p = scenario.params; return schedule_demand(scenario, p["horizon"], [phase("feet_any", self.goal(scenario))], self.goal(scenario), effect=dict(effector="pelvis", direction=(0.0, p["heading"])))
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; xyz, yaw, tilt, v = pelvis_kin(q, dt); u = np.array([np.cos(p["heading"]), np.sin(p["heading"])]); s = v @ u
        n = max(1, int(round(p["t_hold"]/dt))); ok = s >= 0.9*p["v_term"]; t_reach = None
        for i in range(len(s) - n + 1):
            if ok[i:i+n].all(): t_reach = i*dt; break
        up, m = upright(q, dt, T=0.3); val = base_validity(rec)
        if not up: fail(val, "NOT_UPRIGHT_AT_END")
        if t_reach is None: fail(val, "SPEED_NOT_REACHED", max(0.0, p["v_term"] - float(s.max())))
        value = -(t_reach if t_reach is not None else len(q)*dt + (p["v_term"] - float(s.max())))
        return Outcome(value=value, achieved=(t_reach is not None and val.ok), measures=dict(t_reach=t_reach, s_max=float(s.max()), **m), validity=val)
