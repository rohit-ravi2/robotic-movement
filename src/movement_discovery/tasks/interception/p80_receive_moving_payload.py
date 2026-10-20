"""Program 80 (interception): RECEIVE a moving payload into a holding region. A box slides along a table toward the robot (qvel0 on a free box, low friction); it must come to rest INSIDE a region in front of the robot (x in [x_lo, x_hi] of the table, |y| band) with its speed below v_rest, stopped by the hand. Value = -(distance of the final box centre from the region centre). Distinct: a sliding (not flying) object with friction, and a POSITION outcome rather than a hold.
integration check: 2026-10-12 — prior-only rollout (results/md_build/checks_interception.json), plant built, value -0.433, achieved False, gates fired [NOT_IN_REGION]; rel_speed_end=3.88 t_contact=0 dmin_obj=0.87 dist_to_region=0.433 captured=False (27 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from movement_discovery.tasks.force_interaction import _base as B
from . import _catch as C


@register
class ReceiveMovingPayload(C.CatchProgram, TaskProgram):
    id = "P80_receive_moving_payload"; family = Family.INTERCEPTION; title = "Stop a sliding box inside a holding region"
    requires = [Capability.PALM_SITE_RIGHT, Capability.EFFECTOR_COLLIDER, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED
    distinct_from = "Every other interception ends with the object at the effector; here the object must end in a REGION on its support with its speed killed, so braking friction and the stopping position are the effect."
    def parts(self, p):
        from engine import world as W
        o = O.box(name="ball", half=[0.08, 0.08, 0.06], mass=p["mass"], pos=[p["dist"], -0.2, 0.75 + 0.06], graspable=False, mu=p["mu"], qvel0=[-p["v"], 0.0, 0.0, 0.0, 0.0, 0.0])
        return [o], [W.surface("table", pos=(0.9, -0.2, 0.375), size=(0.8, 0.3, 0.375))]
    def world(self, scenario): o, s = self.parts(scenario.params); return WorldSpec(objects=o, surfaces=s)
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(dist=float(rng.uniform(1.2, 1.6)), v=float(rng.uniform(1.0, 2.0)), mass=float(rng.uniform(0.5, 3.0)), mu=float(rng.uniform(0.05, 0.15)), region=[0.3, 0.5], y_band=0.15, v_rest=0.2, f_max=300.0); return B.scenario(self, seed, p, ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="object_displacement", params=dict(object="ball", region_x=p["region"], y_band=p["y_band"], v_rest=p["v_rest"]), horizon=0.8)
    def demand(self, scenario):
        p = scenario.params; t = C.make(p.get("robot", "G1"), self.world(scenario)); o = self.parts(p)[0][0]
        return t, C.demand(t, object=dict(kind="box", pos=o["pos"], size=o["size"], mass=p["mass"], tee=False), rel_speed_max=p["v_rest"], f_max=p["f_max"], mass=0.0)
    def evaluate(self, rec, scenario):
        p = scenario.params; e = rec["extra"]; pos = e.get("ball_pos_end") or [9, 9, 9]; cx = 0.5*sum(p["region"]); dist = float(np.hypot(pos[0] - cx, pos[1] + 0.2)); vs = float(e.get("ball_speed_end", 9.0))
        inside = p["region"][0] <= pos[0] <= p["region"][1] and abs(pos[1] + 0.2) <= p["y_band"] and vs <= p["v_rest"]
        reasons = [r for r in rec["reasons"] if r != "NOT_CAPTURED"] + ([] if inside else ["NOT_IN_REGION"])
        return B.outcome(dict(rec, reasons=reasons), value=-dist, measures=self.measures(rec) | dict(final_pos=pos, final_speed=vs, dist_to_region=dist, inside=inside), achieved=inside)
    def strategy_space(self, scenario): return C.CatchProgram.strategy_space(self, scenario) + ["torque_knots+absorb_prior"]     # ABSORB (construction/absorb.py) appended
