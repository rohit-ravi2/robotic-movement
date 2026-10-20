"""Program 88 (negotiation): moving obstacles. One or two boxes (world.obstacles.moving_obstacle: kinematic, looping waypoint paths across
the corridor of travel) sweep the path; reach x_goal without being touched (a hit is a NONFOOT_CONTACT with a mover or a fall). Outcome =
-shortfall to x_goal. Timing (waiting, passing) is the construction's. integration check: not executable until a representation drives
world.apply_mocap each step and contact_schedule+knots gets a base-level effect mode."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import NegotiationProgram, terrain, obstacle, q_trace, pelvis_kin, past, base_validity, no_trace, fail, travel_goal


@register
class MovingObstacles(NegotiationProgram):
    id = "P88_moving_obstacles"; family = Family.NEGOTIATION; title = "Get past boxes sweeping across the path"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "The obstacle geometry is time-varying: the feasible path exists only in time windows, so the solution's timing (not its shape) is the physics being tested; every other negotiation program is static."
    def world(self, scenario):
        p = scenario.params; movers = [obstacle("moving_obstacle", path=[[x, -p["span"], 0.4], [x, p["span"], 0.4]], period=per, half=(0.15, 0.15, 0.4), name=f"mover{i}") for i, (x, per) in enumerate(zip(p["xs"], p["periods"]))]
        return WorldSpec(terrain=terrain("flat"), mechanisms=movers, tag=f"movers{len(movers)}")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); n = int(rng.integers(1, 3)); xs = [0.8 + 0.9*i for i in range(n)]
        return self.scenario(seed, dict(robot="G1", xs=xs, periods=[float(rng.uniform(2.5, 5.0)) for _ in xs], span=0.8, x_goal=xs[-1] + 0.6, horizon=[2.0, 10.0]), **ov)
    def goal(self, scenario): return travel_goal(scenario.params, GoalSpec)
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; ok, m = past(q, dt, p["x_goal"]); val = base_validity(rec)
        if not ok: fail(val, "NOT_PAST_OBSTACLE", m["short"])
        return Outcome(value=-m["short"], achieved=(ok and val.ok), measures=dict(periods=p["periods"], **m), validity=val)
