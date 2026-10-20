"""Program 69 (payload): TWO-HANDED lift of a wide load. Capability.TWO_HANDS: the G1 has a left_palm site. WELD SIMPLIFICATION,
labelled: the engine.world weld exists for the RIGHT hand only, so the load hangs on one weld and the second hand's contribution is
judged GEOMETRICALLY: the left palm must stay within `two_hand_tol` of the load's left handle point throughout the lift (gate
TWO_HAND_LOST, checked at the end of each step from the recorded state). The left shoulder joints are in the driven set, so the
search can place that hand; nothing scripts it. Value = lifted height.
integration check: 2026-10-12 — prior-only rollout, plant built, value 0, achieved False, gates fired [s0:NOT_GRASPED,s1:HOLD_LOST,SEQUENCE_NOT_OK,DROPPED,TWO_HAND_LOST]; lift_height=0 f_max=0 left_palm_to_handle=0.348 (25 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from . import _pay as P


@register
class TwoHandedLift(P.PayloadProgram, TaskProgram):
    id = "P69_two_handed_lift"; family = Family.PAYLOAD; title = "Lift a wide load with both hands"
    requires = [Capability.PALM_SITE_RIGHT, Capability.PALM_SITE_LEFT, Capability.TWO_HANDS, Capability.WELD_GRASP, Capability.SERVO_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED
    distinct_from = "Every other lift is one-handed; here the load is wider than one arm can hold level and the second hand's placement is a validity condition that couples both arms and the trunk."
    def parts(self, p):
        from engine import world as W
        return [O.payload_with_handle(name="payload", half=[0.10, p["half_y"], 0.06], mass=p["mass"], pos=[0.36, 0.0, 0.75 + 0.06], handle_len=2*p["half_y"] - 0.02)], [W.surface("table", pos=(0.45, 0.0, 0.375), size=(0.25, 0.45, 0.375))]
    def world(self, scenario): o, s = self.parts(scenario.params); return WorldSpec(objects=o, surfaces=s)
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(mass=float(rng.uniform(4.0, 12.0)), half_y=float(rng.uniform(0.22, 0.3)), lift=0.2, two_hand_tol=0.08, hold=0.4); return P.B.scenario(self, seed, p, ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="sequence", params=dict(subgoals=[dict(event="held", object="payload", hands=2), dict(event="object_displacement", object="payload", axis="z", min=p["lift"], held=True)]), horizon=6.0)
    def strategy_space(self, scenario): return ["servo_knots"]
    def demand(self, scenario):
        p = scenario.params; t = P.task(p.get("robot", "G1"), self.world(scenario)); hy = p["half_y"] - 0.04
        return P.plan(t, [P.reach_grasp(t, "payload", horizon=3.0, offset=[0.0, -hy, 0.0], push=[0.0, -hy, 0.01]), P.move_held(t, "payload", [0.0, 0.0, p["lift"]], hold=p["hold"], horizon=3.0)], events=("held", "lifted_two_hands"))
    def evaluate(self, rec, scenario):
        out = super().evaluate(rec, scenario); fs = P.final_state(rec); dl = None
        if fs is not None:
            import mujoco; t = self.demand(scenario).steps[0].task; m = t.plant(t.demand_default())["m"]; d = mujoco.MjData(m); d.qpos[:] = fs[0]; mujoco.mj_forward(m, d)
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "payload"); hy = scenario.params["half_y"] - 0.04; lp = d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.array([0.0, hy, 0.09])
            dl = float(np.linalg.norm(d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_palm")] - lp))
        out.measures.update(left_palm_to_handle=dl, two_hand_tol=scenario.params["two_hand_tol"], note="weld simplification: right-hand weld carries the load; the left hand is judged geometrically")
        if dl is None or dl > scenario.params["two_hand_tol"]: out.validity.reasons.append("TWO_HAND_LOST"); out.validity.contact = False; out.validity.violation += 1.0 + 5.0*((dl or 1.0) - scenario.params["two_hand_tol"]); out.achieved = False
        return out
