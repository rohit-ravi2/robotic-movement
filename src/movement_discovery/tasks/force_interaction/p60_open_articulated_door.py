"""Program 60 (force_interaction): OPEN an articulated door by pushing its panel (no latch). Resistance = hinge damping + dry
friction + optional closer spring; outcome = door angle. Bare push through the effector sphere collider (the MJX G1 body carries
no collision bits: the hand's group-3 geoms and the sphere are enabled, labelled). Bridge: MechanismReachTask (servo construction).
integration check: 2026-10-12 (Plan runner) — prior-only rollout, value 0.153, achieved False, gates fired [NOT_MOVED,CONTACT_TOO_BRIEF]; viol 4.81 (reproduces the earlier line) (60 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec
from movement_discovery.core.registry import register
from movement_discovery.world import mechanisms as MK
from . import _base as B, _mech as M


@register
class OpenArticulatedDoor(M.MechProgram, TaskProgram):
    id = "P60_open_articulated_door"; family = Family.FORCE_INTERACTION; title = "Open a hinged door by pushing the panel"
    requires = [Capability.PALM_SITE_RIGHT, Capability.EFFECTOR_COLLIDER, Capability.SERVO_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED
    unit = "rad"; distinct_from = "P63's lever is a short bar with a hard range on a horizontal axis; the door is a tall heavy panel on a vertical hinge whose moment arm changes with the angle and whose closer resists continuously."
    def mech(self, p): return MK.door(name="door", pos=p["pos"], width=p["width"], hinge_side=p["hinge_side"], resist=p["damping"], frictionloss=p["friction"], spring=p["closer_k"], handle_height=1.0, max_open=1.57)
    def world(self, scenario): return WorldSpec(mechanisms=[self.mech(scenario.params)])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed)
        p = dict(pos=[float(rng.uniform(0.5, 0.65)), float(rng.uniform(0.15, 0.3))], width=0.8, hinge_side=str(rng.choice(["left", "right"])), damping=float(rng.uniform(1.0, 5.0)), friction=float(rng.uniform(0.5, 3.0)), closer_k=float(rng.choice([0.0, 2.0, 5.0])), min_disp=np.radians(30.0), push=0.30, f_max=400.0, t_contact_min=0.3)
        return B.scenario(self, seed, p, ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="mechanism_displacement", params=dict(mechanism="door", joint="door_hinge", min=p["min_disp"]), horizon=2.5)
    def strategy_space(self, scenario): return ["servo_knots", "torque_knots+tracked_prior", "servo_knots+brace_prior", "torque_knots+brace_prior"]
    def demand(self, scenario):
        p = scenario.params; mk = self.mech(p); t = M.make(p.get("robot", "G1"), self.world(scenario))
        return t, M.demand(t, mk, push=[p["push"], 0.0, 0.0], horizon=2.5, min_disp=p["min_disp"], f_max=p["f_max"], t_contact_min=p["t_contact_min"], sign=mk["open_sign"])
