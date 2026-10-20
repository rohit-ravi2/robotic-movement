"""Program 90 (compound): RETRIEVE an object, CARRY it, PLACE it elsewhere. Sub-goals with events: (1) held (weld, G1 experimental
simplification); (2) lifted clear: object bottom >= table top + clearance while held; (3) resting at the target, released (loose tolerance:
this is a transfer, P72 is the precision version). Chained through compose.run_sequence from the real handoff states (engine/demo.run_place
machinery). integration check: 2026-10-12 (Plan runner, 3 steps) — prior-only sequence, value -0.058, achieved True, reasons [] (s0 held, s1 clear, s2 placed; placement_error 0.058; reproduces the compose-path line) (92 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, Plan
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from movement_discovery.tasks.force_interaction import _base as B
from movement_discovery.tasks.payload import _pay as P
from . import _seq as Q


@register
class RetrieveCarryPlace(Q.CompoundProgram, TaskProgram):
    id = "P90_retrieve_carry_place"; family = Family.COMPOUND; title = "Pick up an object, carry it across the body, set it down"
    requires = [Capability.PALM_SITE_RIGHT, Capability.WELD_GRASP, Capability.SERVO_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED
    distinct_from = "P72 scores centimetre precision at the same place; here the effect is a TRANSFER across the workspace (lateral carry of 0.3-0.5 m) with a loose landing, so the carry posture and the handoffs are the physics."
    subgoals = [dict(goal="payload:grasp", event="held", object="box", horizon=3.0), dict(goal="payload:lift", event="object_clear", object="box", clearance=0.08, horizon=2.0), dict(goal="payload:place", event="object_at", object="box", released=True, horizon=3.0)]
    def parts(self, p):
        from engine import world as W
        return [O.box(name="box", half=[0.05, 0.05, 0.05], mass=p["mass"], pos=[0.36, -0.25, 0.75 + 0.05])], [W.surface("table", pos=(0.4, 0.0, 0.375), size=(0.25, 0.5, 0.375))]
    def world(self, scenario): o, s = self.parts(scenario.params); return WorldSpec(objects=o, surfaces=s)
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(mass=float(rng.uniform(0.3, 2.0)), target=[float(rng.uniform(0.3, 0.4)), float(rng.uniform(0.1, 0.3)), 0.75 + 0.05], tol=0.06, clearance=0.08); return B.scenario(self, seed, p, ov)
    def demand(self, scenario):
        p = scenario.params; t = P.task(p.get("robot", "G1"), self.world(scenario)); tgt = p["target"]
        d1 = dict(t.demand_default(), target_object="box", tol=0.05, hold=0.15, grasp_on_reach="box", omega_servo=120.0, trunk_cap=0.05, horizon=3.0, refine=1, aim_z=True)
        d2 = dict(t.demand_default(), target_object="box", offset=[0.0, 0.0, p["clearance"] + 0.04], tol=0.05, hold=0.15, held=["box"], omega_servo=120.0, trunk_cap=0.05, horizon=2.0)
        d3 = dict(t.demand_default(), target_world=[tgt[0], tgt[1], tgt[2] + 0.02], tol=0.05, hold=0.3, held=["box"], release_at_end=True, omega_servo=120.0, trunk_cap=0.05, horizon=3.0)
        return Plan(steps=[Q.step(t, d1, event="held"), Q.step(t, d2, event="object_clear"), Q.step(t, d3, event="object_at")], robot=t.robot, settle_T=0.0, freeze_T=0.4)
    def evaluate(self, rec, scenario):
        p = scenario.params; e = P.final_extra(rec); pos = ((e.get("world_after") or {}).get("box") or {}).get("pos"); err = float(np.linalg.norm(np.asarray(pos[:2]) - np.asarray(p["target"][:2]))) if pos else 1.0
        placed = bool(pos and err <= p["tol"] and abs(pos[2] - p["target"][2]) <= 2*p["tol"] and not P.held(rec, "box"))
        return B.seq_outcome(rec, value=-err, measures=dict(placement_error=err, final_pos=pos, carry=float(np.linalg.norm(np.asarray(p["target"][:2]) - np.array([0.36, -0.25])))), extra_reasons=([] if placed else ["NOT_PLACED"]), achieved=placed)
