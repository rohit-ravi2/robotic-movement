"""Program 66 (payload): lift a load from a table to a SHELF and release it there. Sub-goals: held, lifted clear of the table, resting
on the shelf (released, inside the shelf's extent). Value = height gained by the payload; achieved iff it rests on the shelf.
Hold = G1 weld at the grip site (experimental simplification). integration check: 2026-10-12 (Plan runner, 3 steps) — prior-only sequence, value -0.000, achieved False, gates fired [s1:ENERGY_CLOSURE,s1:NONFOOT_CONTACT,s1:STANCE_LOST,s1:FELL,s1:NOT_REACHED,s1:END_UNBALANCED,s2:START_NOT_AT_REST,s2:NONFOOT_CONTACT,s2:STANCE_LOST,s2:NOT_REACHED,s2:END_UNBALANCED,SEQUENCE_NOT_OK,NOT_PLACED]; s0 held ok, the lift step falls (reproduces the compose-path line) (120 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from . import _pay as P


@register
class LiftToShelfHeight(P.PayloadProgram, TaskProgram):
    id = "P66_lift_to_shelf_height"; family = Family.PAYLOAD; title = "Transfer a load from a table onto a higher shelf"
    requires = [Capability.PALM_SITE_RIGHT, Capability.WELD_GRASP, Capability.SERVO_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED; releases = True
    distinct_from = "P67 holds the load overhead with no support to put it on; P72 places with centimetre precision at table height; here the effect is a HEIGHT transfer ending with the load resting on a shelf above the start."
    def parts(self, p):
        from engine import world as W
        table = W.surface("table", pos=(0.45, -0.2, p["table"]/2), size=(0.25, 0.3, p["table"]/2)); shelf = W.surface("shelf", pos=(0.38, 0.25, p["shelf"] - 0.02), size=(0.2, 0.2, 0.02))
        return [O.payload_with_handle(name="payload", half=[0.08, 0.08, 0.08], mass=p["mass"], pos=[0.38, -0.2, p["table"] + 0.08])], [table, shelf]
    def world(self, scenario): o, s = self.parts(scenario.params); return WorldSpec(objects=o, surfaces=s)
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(mass=float(rng.uniform(1.0, 6.0)), table=0.75, shelf=float(rng.uniform(1.05, 1.3)), lift=0.0, tol=0.06); p["lift"] = p["shelf"] - p["table"]; return P.B.scenario(self, seed, p, ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="sequence", params=dict(subgoals=[dict(event="held", object="payload"), dict(event="object_on_surface", object="payload", surface="shelf", released=True)]), horizon=9.0)
    def strategy_space(self, scenario): return ["servo_knots", "servo_knots+brace_prior"]
    def demand(self, scenario):
        p = scenario.params; t = P.task(p.get("robot", "G1"), self.world(scenario)); tgt = [0.38, 0.25, p["shelf"] + 0.08 + 0.11 + 0.02]   # grip point (handle) 11 cm above the box centre, 2 cm above the shelf
        return P.plan(t, [P.reach_grasp(t, "payload", horizon=3.0), P.move_held(t, "payload", [0.0, 0.0, p["lift"] + 0.06], horizon=3.0), P.place_held(t, "payload", tgt, horizon=3.0)], events=("held", "lifted_clear", "object_on_surface"))
    def evaluate(self, rec, scenario):
        out = super().evaluate(rec, scenario); p = scenario.params; e = P.final_extra(rec); pos = ((e.get("world_after") or {}).get("payload") or {}).get("pos")
        on = bool(pos is not None and abs(pos[0] - 0.38) <= 0.2 and abs(pos[1] - 0.25) <= 0.2 and abs(pos[2] - 0.08 - p["shelf"]) <= p["tol"])
        out.measures.update(on_shelf=on, final_pos=pos)
        if not on: out.validity.reasons.append("NOT_PLACED"); out.validity.object_interaction = False; out.validity.violation += 1.0; out.achieved = False
        return out
