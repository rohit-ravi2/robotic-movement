"""Program 61 (force_interaction): OPERATE a drawer: pull it open (hold required: G1 weld = experimental simplification,
Capability.WELD_GRASP; G1H Dex3 wrap declared, not exercised) or push it closed from an open start (bare push through the
effector collider). Guided slide at counter height (world.mechanisms.drawer), dry friction + damping; outcome = drawer travel in the commanded sense.
integration check: 2026-10-12 — prior-only rollout, plant built, value 0.28, achieved False, gates fired [CONTACT_TOO_BRIEF]; displacement=0.28 f_max=120 t_contact=0.222 (19 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec
from movement_discovery.core.registry import register
from movement_discovery.world import mechanisms as MK
from . import _base as B, _mech as M


@register
class OperateDrawer(M.MechProgram, TaskProgram):
    id = "P61_operate_drawer"; family = Family.FORCE_INTERACTION; title = "Open or close a drawer on a guided slide"
    requires = [Capability.PALM_SITE_RIGHT, Capability.WELD_GRASP, Capability.EFFECTOR_COLLIDER, Capability.SERVO_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED
    unit = "m"; distinct_from = "P59 pulls a free-standing friction load whose resistance scales with mass; the drawer is a kinematically guided slide with a hard travel stop and a fixed friction, operated in either sense."
    def mech(self, p): return MK.drawer(name="drawer", pos=p["pos"], height=p["height"], half=[0.2, 0.18, 0.08], travel=p["travel"], mass=p["mass"], frictionloss=p["friction"], resist=5.0)
    def world(self, scenario): return WorldSpec(mechanisms=[self.mech(scenario.params)])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed)
        p = dict(mode=str(rng.choice(["pull", "push"])), pos=[0.75, -0.22], height=float(rng.uniform(0.65, 0.95)), travel=0.35, mass=float(rng.uniform(2.0, 8.0)), friction=float(rng.uniform(4.0, 15.0)), min_disp=0.12, stroke=0.25, f_max=300.0, t_contact_min=0.25)
        return B.scenario(self, seed, p, ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="mechanism_displacement", params=dict(mechanism="drawer", joint="drawer_slide", min=p["min_disp"], sense=p["mode"]), horizon=5.0)
    def strategy_space(self, scenario): return ["servo_knots", "torque_knots+tracked_prior", "servo_knots+brace_prior", "torque_knots+brace_prior"]
    def demand(self, scenario):
        p = scenario.params; mk = self.mech(p); t = M.make(p.get("robot", "G1"), self.world(scenario))
        if p["mode"] == "push":   # the drawer starts OPEN (q0 = 0.8 travel set on the plant by WorldReachTask.plant from demand["start_open"]); closing = pushing the front face in (+x)
            return t, M.demand(t, mk, push=[p["stroke"], 0.0, 0.0], horizon=2.5, min_disp=p["min_disp"], f_max=p["f_max"], t_contact_min=p["t_contact_min"], sign=-1.0, start_open=0.8*p["travel"])
        d1 = M.demand(t, mk, push=[-0.03, 0.0, 0.0], horizon=2.5, grasp=mk["name"], min_disp=0.0)
        d2 = M.demand(t, mk, push=[-p["stroke"], 0.0, 0.0], horizon=2.5, min_disp=p["min_disp"], f_max=p["f_max"], held=[mk["name"]])
        return M.plan(t, [d1, d2], events=("held", "mechanism_displacement"))
