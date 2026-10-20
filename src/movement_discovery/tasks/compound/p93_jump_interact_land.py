"""Program 93 (compound): JUMP, INTERACT with an object in the air (touch a suspended target), LAND. Sub-goals with events: (1) airborne
(both feet unloaded >= FLIGHT_CONFIRM); (2) contact: hand touches the suspended target above standing reach (world.objects.suspended_target)
while airborne; (3) landed and settled (the Phase 14 landing controller gate). The jump+landing primitive lives in tasks/jump.py (bare
plant, torque knots); the suspended target lives in a world plant: chaining them needs the plant state mapping (CORE) or a jump task
that accepts a world scene; the aerial family's P16 (airborne contact) is the single-primitive version. Status SCAFFOLDED (no partial
run: no single step is both executable and meaningful here). integration check: 2026-10-12 — not executable (tasks/jump.py (bare plant) and the suspended target (world plant) differ in qpos layout: compose.run_sequence requires one qpos layout acro)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, Plan
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from movement_discovery.tasks.force_interaction import _base as B
from . import _seq as Q


@register
class JumpInteractLand(Q.CompoundProgram, TaskProgram):
    id = "P93_jump_interact_land"; family = Family.COMPOUND; title = "Jump, touch a hanging target in the air, land"
    requires = [Capability.PALM_SITE_RIGHT, Capability.EFFECTOR_COLLIDER, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE, Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT]; status = Status.SCAFFOLDED
    distinct_from = "P16 (aerial) scores the airborne touch alone; here the landing is a required sub-goal with its own gate, so the touch must not cost the landing (angular momentum at contact is the coupling)."
    subgoals = [dict(goal="aerial:jump", event="airborne", horizon=0.8), dict(goal="P16_airborne_contact", event="contact", object="bag", while_airborne=True, horizon=0.3), dict(goal="balance:landing", event="landed_settled", horizon=1.0)]
    def world(self, scenario): p = scenario.params; return WorldSpec(objects=[O.suspended_target(name="bag", anchor=[0.3, -0.2, 2.2], length=2.2 - p["z_target"], radius=0.08, mass=0.5)])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(z_target=float(rng.uniform(1.45, 1.6)), reach_standing=1.3, h_min=0.03); return B.scenario(self, seed, p, ov)
    def demand(self, scenario):
        return Plan(steps=[], robot=scenario.params.get("robot", "G1"), missing=[dict(goal="sequence", why="tasks/jump.py (bare plant) has no scene: the suspended target cannot exist in the jump step (" + Q.NQ_MAPPING + "); asset: a JumpTask that accepts a world scene")])
    def evaluate(self, rec, scenario):
        e = rec["steps"][-1]["extra"] if "steps" in rec else rec.get("extra", {}); touched = bool(e.get("touched")); landed = bool(e.get("landed"))
        return B.outcome(rec, value=float(e.get("h_apex", 0.0)), measures=dict(touched=touched, landed=landed), extra_reasons=([] if touched else ["NOT_HIT"]) + ([] if landed else ["NOT_LANDED"]), achieved=touched and landed)
