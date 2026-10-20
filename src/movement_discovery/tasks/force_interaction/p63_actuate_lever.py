"""Program 63 (force_interaction): ACTUATE a lever: push the handle end of a hinged bar DOWN to lift a load box at the far end (world.mechanisms.lever: y hinge, hard range, damping). Bare push through the effector sphere collider (labelled). Outcome = lever angle.
integration check: 2026-10-12 — prior-only rollout, plant built, value 4.28e-07, achieved False, gates fired [NOT_MOVED,CONTACT_TOO_BRIEF]; displacement=4.28e-07 f_max=0 t_contact=0 (27 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec
from movement_discovery.core.registry import register
from movement_discovery.world import mechanisms as MK
from . import _base as B, _mech as M


@register
class ActuateLever(M.MechProgram, TaskProgram):
    id = "P63_actuate_lever"; family = Family.FORCE_INTERACTION; title = "Throw a resistive lever through its range"
    requires = [Capability.PALM_SITE_RIGHT, Capability.EFFECTOR_COLLIDER, Capability.SERVO_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED
    unit = "rad"; distinct_from = "P60's door is a vertical-axis panel pushed through its face; the lever is a short horizontal-axis bar pushed at its tip through a hard range, so the contact point travels a sagittal arc and the required force is set by the load's weight times its arm."
    def mech(self, p): return MK.lever(name="lever", pos=p["pos"], height=p["height"], length=p["length"], pivot=0.5, resist=p["resist"], load=p["load"], max_angle=p["range"], bar_mass=1.0)
    def world(self, scenario): return WorldSpec(mechanisms=[self.mech(scenario.params)])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(pos=[float(rng.uniform(0.75, 0.9)), -0.25], height=float(rng.uniform(0.7, 0.9)), length=float(rng.uniform(0.6, 0.9)), resist=float(rng.uniform(0.2, 1.0)), load=float(rng.uniform(1.0, 5.0)), range=0.6, min_disp=np.radians(20.0), push=0.25, f_max=300.0, t_contact_min=0.2)
        return B.scenario(self, seed, p, ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="mechanism_displacement", params=dict(mechanism="lever", joint="lever_hinge", min=p["min_disp"]), horizon=2.5)
    def strategy_space(self, scenario): return ["servo_knots", "torque_knots+tracked_prior", "servo_knots+brace_prior", "torque_knots+brace_prior"]
    def demand(self, scenario):
        p = scenario.params; mk = self.mech(p); t = M.make(p.get("robot", "G1"), self.world(scenario))
        return t, M.demand(t, mk, push=[0.0, 0.0, -p["push"]], horizon=2.5, min_disp=p["min_disp"], f_max=p["f_max"], t_contact_min=p["t_contact_min"])   # WORLD lever: handle end faces the robot, pushing it DOWN lifts the load (positive angle)
