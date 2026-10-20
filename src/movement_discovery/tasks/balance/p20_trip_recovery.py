"""Program 20 (balance): trip recovery. While travelling forward (initial base velocity, handed-off state) a low bar across the path
catches a foot; the robot must stay upright and continue past the bar. Outcome = -settling time after the bar; validity = upright, base
beyond the bar. Needs a moving initial state and an obstacle that collides with the FEET (foot geoms do collide; the bar is a world
obstacle). Representation: contact_schedule. integration check: 2026-10-12 world builds (barrier); not executable: needs a moving initial state and the base-level effect mode in contact_schedule+knots (AssertionError: pelvis at plant build)."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import BalanceProgram, terrain, obstacle, CS_NAME, q_trace, pelvis_kin, upright, settle_time, base_validity, no_trace, fail, schedule_demand, phase


@register
class TripRecovery(BalanceProgram):
    id = "P20_trip_recovery"; family = Family.BALANCE; title = "Stay upright when a foot is caught by a low bar while moving"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "The disturbance acts on a swing foot (an angular impulse about the stance foot) during travel, not on the torso of a standing robot (P17)."
    def world(self, scenario): p = scenario.params; return WorldSpec(terrain=terrain("flat"), mechanisms=[obstacle("barrier", height=p["h_bar"], width=1.0, x=p["x_bar"], thick=0.03)], tag="trip")   # a low wall across the path (world.obstacles.barrier); foot geoms collide with world pieces
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); return self.scenario(seed, dict(robot="G1", v0=float(rng.uniform(0.5, 1.2)), x_bar=float(rng.uniform(0.4, 0.8)), h_bar=float(rng.uniform(0.03, 0.08)), T=3.0, t_hold=0.5, horizon=[2.0, 3.5]), **ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="stability", params=dict(initial_state=dict(vxy=[p["v0"], 0.0]), obstacle="bar", x_min=p["x_bar"] + 0.3, objective="min_settling_time"), tolerance=dict(tilt=np.radians(30), t_hold=p["t_hold"]), horizon=p["horizon"][1])
    def strategy_space(self, scenario): return [CS_NAME, "servo_knots+step_prior"]
    def demand(self, scenario): p = scenario.params; return schedule_demand(scenario, p["horizon"], [phase("feet_any", self.goal(scenario), initial_state=dict(vxy=[p["v0"], 0.0]))], self.goal(scenario), effect=dict(effector="pelvis", direction=(0.0, 0.0)))
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; xyz, yaw, tilt, v = pelvis_kin(q, dt); up, m = upright(q, dt, T=p["t_hold"]); val = base_validity(rec)
        i_bar = int(np.argmax(xyz[:, 0] > p["x_bar"])) if (xyz[:, 0] > p["x_bar"]).any() else None; ts = settle_time(q, dt, i_bar*dt) if i_bar is not None else None
        if not up: fail(val, "NOT_UPRIGHT_AT_END")
        if i_bar is None or xyz[-1, 0] < p["x_bar"] + 0.3: fail(val, "NOT_PAST_BAR", max(0.0, p["x_bar"] + 0.3 - float(xyz[-1, 0])))
        if ts is None: fail(val, "NOT_SETTLED")
        return Outcome(value=-(ts if ts is not None else p["T"]), achieved=(up and ts is not None and val.ok), measures=dict(t_settle=ts, x_end=float(xyz[-1, 0]), **m), validity=val)
