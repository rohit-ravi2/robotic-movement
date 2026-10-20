"""Program 05 (locomotion): heading change while moving. Keep moving at a commanded speed while the base heading turns by a commanded
angle; outcome = heading error at the end of the window with the speed held. integration check: not executable until
construction/contact_schedule lands."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import BodyProgram, terrain, q_trace, pelvis_kin, upright, base_validity, no_trace, fail, schedule_demand, phase, CS_NAME


@register
class HeadingChange(BodyProgram):
    id = "P05_heading_change"; family = Family.LOCOMOTION; title = "Turn the direction of travel by a commanded angle without stopping"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE, Capability.HIP_ROLL]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P06 constrains the whole path (an arc); this constrains only the initial and final travel directions and the speed floor."
    def world(self, scenario): return WorldSpec(terrain=terrain("flat"), tag="flat")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); return self.scenario(seed, dict(speed=float(rng.uniform(0.4, 1.2)), dpsi=float(rng.choice([-1, 1])*rng.uniform(np.radians(30), np.radians(150))), t_turn=[1.0, 2.5], speed_floor=0.5, tol=np.radians(10), horizon=[2.5, 4.0]), **ov)
    def goal(self, scenario):
        p = scenario.params; return GoalSpec(kind="heading", params=dict(dpsi=p["dpsi"], speed=p["speed"], speed_floor_frac=p["speed_floor"]), tolerance=dict(psi=p["tol"]), horizon=p["horizon"][1])
    def strategy_space(self, scenario): return [CS_NAME, "servo_knots+step_prior"]
    def demand(self, scenario):
        p = scenario.params; return schedule_demand(scenario, p["horizon"], [phase("feet_any", self.goal(scenario))], self.goal(scenario), effect=dict(effector="pelvis", direction=(0.0, p["dpsi"])))
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; xyz, yaw, tilt, v = pelvis_kin(q, dt); sp = np.linalg.norm(v, axis=1); i0 = int(p["t_turn"][0]/dt)
        head = np.arctan2(v[:, 1], v[:, 0]); n = max(1, int(0.3/dt)); h_end = float(np.arctan2(np.mean(np.sin(head[-n:])), np.mean(np.cos(head[-n:]))))
        err = float(abs((h_end - p["dpsi"] + np.pi) % (2*np.pi) - np.pi)); drop = float(max(0.0, p["speed_floor"]*p["speed"] - sp[i0:].min())) if len(sp) > i0 else 1.0
        up, m = upright(q, dt, T=0.3); val = base_validity(rec)
        if not up: fail(val, "NOT_UPRIGHT_AT_END")
        if drop > 0: fail(val, "SPEED_DROPPED", drop)
        return Outcome(value=-err, achieved=(err <= p["tol"] and val.ok), measures=dict(heading_err_deg=float(np.degrees(err)), heading_end_deg=float(np.degrees(h_end)), speed_min=float(sp[i0:].min()) if len(sp) > i0 else None, yaw_change_deg=float(np.degrees(yaw[-1] - yaw[0])), **m), validity=val)
