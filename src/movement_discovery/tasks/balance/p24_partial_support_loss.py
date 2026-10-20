"""Program 24 (balance): recovery after partial support loss. At t0 the pad under ONE foot drops away (world.mechanisms.trapdoor: two
5 cm pads under the feet, the named one is a mocap slab that drops by `drop` at t0); the robot must stay upright on the remaining support or re-establish a
support within the horizon. Outcome = -settling time after the loss. Bridge: StandTask (standing controller; the hook chain drops the pad).
integration check: 2026-10-12 round 2, runner.check (StandTask on WORLD's trapdoor scene, start_z 0.05, pad dropped by mocap at t0): the standing controller fell after the drop (NONFOOT_CONTACT, FELL, NOT_SETTLED); 11 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import BalanceProgram, terrain, CS_NAME, q_trace, upright, settle_time, base_validity, no_trace, fail, schedule_demand, phase


@register
class PartialSupportLoss(BalanceProgram):
    id = "P24_partial_support_loss"; family = Family.BALANCE; title = "Stay upright when the floor drops under one foot"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE, Capability.HIP_ROLL, Capability.ANKLE_ROLL]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "The support polygon collapses to one foot without warning (no torso force, no slip): the disturbance is the removal of a constraint, not an applied load."
    def world(self, scenario):
        from movement_discovery.world import mechanisms as M
        p = scenario.params; return WorldSpec(terrain=terrain("flat"), mechanisms=[M.trapdoor(foot=p["foot"], t0=p["t0"], drop=p["drop"])], tag="trapdoor")   # two 5 cm pads under the feet; the named one drops at t0 (mocap, via physics.hooks)
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); return self.scenario(seed, dict(robot="G1", foot=str(rng.choice(["left", "right"])), t0=1.0, drop=float(rng.uniform(0.05, 0.20)), T=3.5, t_hold=0.5, horizon=[3.0, 3.5]), **ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="stability", params=dict(disturbance="support_loss", foot=p["foot"], drop=p["drop"], t0=p["t0"], objective="min_settling_time"), tolerance=dict(tilt=np.radians(30), t_hold=p["t_hold"]), horizon=p["T"])
    def strategy_space(self, scenario): return ["servo_knots", CS_NAME]
    def schedule(self, scenario):
        """(BaseEffectTask, demand) for contact_schedule+knots; `demand` (inherited) is the StandTask bridge."""
        p = scenario.params; other = "left" if p["foot"] == "right" else "right"
        return schedule_demand(scenario, p["horizon"], [phase("both", self.goal(scenario), until=p["t0"]), phase("feet_any", self.goal(scenario), note=f"the {p['foot']} foot's floor is {p['drop']} m lower after t0; support may be {other} only or both at the new height")], self.goal(scenario))
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; up, m = upright(q, dt, T=p["t_hold"], z_frac=0.6); ts = settle_time(q, dt, p["t0"]); val = base_validity(rec)
        if not up: fail(val, "NOT_UPRIGHT_AT_END")
        if ts is None: fail(val, "NOT_SETTLED")
        return Outcome(value=-(ts if ts is not None else p["T"]), achieved=(up and ts is not None and val.ok), measures=dict(t_settle=ts, drop=p["drop"], **m), validity=val)
