"""Program 67 (payload): OVERHEAD lift: raise a load from a table to above head height (grip point z >= `z_over`) and hold it there.
Value = height gained; achieved iff the grip point ends above z_over with the payload held. Hold = G1 weld (experimental
simplification). The arm's reach above the head bounds the feasible load and height; the search decides how the trunk helps.
integration check: 2026-10-12 — prior-only rollout, plant built, value -0.727, achieved False, gates fired [s1:NONFOOT_CONTACT,s1:STANCE_LOST,s1:FELL,s1:NOT_REACHED,s1:END_UNBALANCED,SEQUENCE_NOT_OK]; lift_height=-0.727 f_max=0 (39 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from . import _pay as P


@register
class OverheadLift(P.PayloadProgram, TaskProgram):
    id = "P67_overhead_lift"; family = Family.PAYLOAD; title = "Raise a load above the head and hold it"
    requires = [Capability.PALM_SITE_RIGHT, Capability.WELD_GRASP, Capability.SERVO_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED
    distinct_from = "P66 ends with the load resting on a shelf; here nothing supports the load: the hold above the head must be sustained against gravity with the arm near full extension, and the end state must still be balanced."
    def parts(self, p):
        from engine import world as W
        return [O.payload_with_handle(name="payload", half=[0.08, 0.08, 0.06], mass=p["mass"], pos=[0.36, -0.2, 0.75 + 0.06])], [W.surface("table", pos=(0.45, -0.2, 0.375), size=(0.25, 0.3, 0.375))]
    def world(self, scenario): o, s = self.parts(scenario.params); return WorldSpec(objects=o, surfaces=s)
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(mass=float(rng.uniform(0.5, 4.0)), z_over=float(rng.uniform(1.55, 1.7)), lift=0.0, hold=0.5); p["lift"] = p["z_over"] - (0.75 + 0.06 + 0.09); return P.B.scenario(self, seed, p, ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="sequence", params=dict(subgoals=[dict(event="held", object="payload"), dict(event="grip_above", object="payload", z=p["z_over"], held=True, hold=p["hold"])]), horizon=6.0)
    def strategy_space(self, scenario): return ["servo_knots", "servo_knots+brace_prior"]
    def demand(self, scenario):
        p = scenario.params; t = P.task(p.get("robot", "G1"), self.world(scenario))
        return P.plan(t, [P.reach_grasp(t, "payload", horizon=3.0), P.move_held(t, "payload", [-0.10, 0.05, p["lift"]], hold=p["hold"], horizon=3.5)], events=("held", "overhead_held"))
