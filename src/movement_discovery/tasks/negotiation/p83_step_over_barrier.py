"""Program 83 (negotiation): step over a barrier. A low wall (world.obstacles.barrier) crosses the path; get past it to x_goal without
touching it (the feet collide with world pieces: a hit shows as a foot stuck or as a fall). Outcome = -shortfall. Whether it is stepped
over, straddled or jumped is not specified. integration check: 2026-10-12 round 2, runner.check (contact_schedule+knots over BaseEffectTask, prior-only): stands, NOT_PAST_OBSTACLE (short 1.19 m), no other violation; 4 phases."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import NegotiationProgram, terrain, obstacle, q_trace, pelvis_kin, past, base_validity, no_trace, fail, travel_goal, CS_NAME


@register
class StepOverBarrier(NegotiationProgram):
    id = "P83_step_over_barrier"; family = Family.NEGOTIATION; title = "Get past a low wall across the path"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE, Capability.HIP_ROLL]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "A single discrete foot-clearance event at a known place (unlike P08's repeated height gain and P84's many small footholds); the barrier is too high to ignore and too low to need a flight."
    def world(self, scenario): p = scenario.params; return WorldSpec(terrain=terrain("flat"), mechanisms=[obstacle("barrier", height=p["height"], width=1.5, x=p["x_obs"], thick=p["thick"])], tag=f"barrier{int(p['height']*100)}")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); return self.scenario(seed, dict(robot="G1", height=float(rng.uniform(0.10, 0.30)), thick=float(rng.uniform(0.03, 0.15)), x_obs=0.55, x_goal=1.2, horizon=[1.5, 6.0]), **ov)
    def goal(self, scenario): return travel_goal(scenario.params, GoalSpec)
    def strategy_space(self, scenario): return [CS_NAME, "torque_knots+tracked_prior", "servo_knots+step_prior"]
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; ok, m = past(q, dt, p["x_goal"]); val = base_validity(rec); xyz = pelvis_kin(q, dt)[0]
        if not ok: fail(val, "NOT_PAST_OBSTACLE", m["short"])
        return Outcome(value=-m["short"], achieved=(ok and val.ok), measures=dict(barrier=[p["height"], p["thick"]], pelvis_z_min=float(xyz[:, 2].min()), **m), validity=val)
