"""Program 65 (payload): FLOOR-LEVEL lift. A handled box (world.objects.payload_with_handle) stands on the floor; its handle is at
~0.23 m: the arm alone cannot reach it from the stance, so whatever lowers the body (squat, trunk bend: the reach prior squats when
the target is short, the search refines) is DERIVED from the constraint, never scripted. Sub-goals: handle held, payload lifted by
>= `lift` and still held. Hold = G1 weld at the grip site (experimental simplification).
integration check: 2026-10-12 — prior-only rollout, plant built, value 0, achieved False, gates fired [s0:NOT_REACHED,s0:NOT_GRASPED,s1:NONFOOT_CONTACT,s1:STANCE_LOST,s1:FELL,s1:NOT_REACHED,s1:END_UNBALANCED,s1:HOLD_LOST,SEQUENCE_NOT_OK,DROPPED]; lift_height=0 f_max=0 (46 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from . import _pay as P


@register
class FloorLevelLift(P.PayloadProgram, TaskProgram):
    id = "P65_floor_level_lift"; family = Family.PAYLOAD; title = "Lift a handled load from the floor"
    requires = [Capability.PALM_SITE_RIGHT, Capability.WELD_GRASP, Capability.SERVO_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED
    distinct_from = "P66 starts from a table and ends on a shelf (height transfer, release); here the load starts at the floor, so the body must get its hand to 0.23 m and come back up loaded."
    def obj(self, p): return O.payload_with_handle(name="payload", half=[0.12, 0.10, 0.10], mass=p["mass"], pos=[p["x"], -0.2, 0.10])
    def world(self, scenario): return WorldSpec(objects=[self.obj(scenario.params)])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(mass=float(rng.uniform(2.0, 10.0)), x=float(rng.uniform(0.28, 0.38)), lift=float(rng.uniform(0.3, 0.5)), hold=0.3); return P.B.scenario(self, seed, p, ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="sequence", params=dict(subgoals=[dict(event="held", object="payload"), dict(event="object_displacement", object="payload", axis="z", min=p["lift"], held=True)]), horizon=6.0)
    def strategy_space(self, scenario): return ["servo_knots", "servo_knots+brace_prior"]
    def demand(self, scenario):
        p = scenario.params; t = P.task(p.get("robot", "G1"), self.world(scenario))
        return P.plan(t, [P.reach_grasp(t, "payload", horizon=3.5, squat=0.5), P.move_held(t, "payload", [0.0, 0.0, p["lift"]], hold=p["hold"], horizon=3.0)], events=("held", "lifted"))
