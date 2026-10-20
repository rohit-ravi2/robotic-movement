"""Program 62 (force_interaction): ROTATE a crank against a load torque. The knob must be carried around the circle: the hold is the
G1 weld to the knob body (experimental simplification, Capability.WELD_GRASP; the G1H Dex3 wrap on the cylindrical knob is the
declared real hold, not exercised). Outcome = turns of the crank joint (rad / 2 pi). The sequence is a list of EXTERNAL sub-goals:
knob held, then the knob at each successive quadrant of its circle (events on the crank angle), never a hand trajectory.
integration check: 2026-10-12 — prior-only rollout, plant built, value 0.0607, achieved False, gates fired [s0:NOT_MOVED,s1:NOT_MOVED,s2:STANCE_LOST,s2:END_UNBALANCED,s2:NOT_MOVED,s3:START_NOT_AT_REST,s3:STANCE_LOST,s3:NOT_MOVED,s4:STANCE_LOST,s4:END_UNBALANCED,s4:NOT_MOVED,SEQUENCE_NOT_OK]; displacement=0.0607 f_max=279 t_contact=1.99 (39 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec
from movement_discovery.core.registry import register
from movement_discovery.world import mechanisms as MK
from . import _base as B, _mech as M


@register
class RotateCrank(M.MechProgram, TaskProgram):
    id = "P62_rotate_crank"; family = Family.FORCE_INTERACTION; title = "Turn a loaded crank through a given number of turns"
    requires = [Capability.PALM_SITE_RIGHT, Capability.WELD_GRASP, Capability.SERVO_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED
    unit = "turns"; min_key = "min_turns"; distinct_from = "P64 holds a torque in a band on a fixed-axis object through a bounded angle; the crank is unbounded continuous rotation under a constant load, where the hand must follow a circle with a changing lever arm."
    def mech(self, p): return MK.crank(name="crank", pos=p["pos"], height=p["height"], radius=p["radius"], axis=[1, 0, 0], resist=0.5, load=p["load_torque"], mass=1.0)
    def world(self, scenario): return WorldSpec(mechanisms=[self.mech(scenario.params)])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(pos=[0.45, -0.25], height=float(rng.uniform(0.75, 0.95)), radius=float(rng.uniform(0.08, 0.15)), load_torque=float(rng.uniform(0.5, 4.0)), min_turns=0.5, quadrants=4, f_max=300.0)
        return B.scenario(self, seed, p, ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="mechanism_displacement", params=dict(mechanism="crank", joint="crank_hinge", min=p["min_turns"]*2*np.pi, unit="rad"), horizon=10.0)
    def strategy_space(self, scenario): return ["servo_knots", "servo_knots+brace_prior"]
    def _value(self, rec):
        v = super()._value(rec) if "steps" not in rec else float(sum(s["extra"].get("mech_disp", 0.0) for s in rec["steps"][1:]))
        return v/(2*np.pi)
    def demand(self, scenario):
        """Hold the knob, then one reach per quadrant: the target is the knob's NEXT quadrant position on its circle (an external event)."""
        p = scenario.params; mk = self.mech(p); t = M.make(p.get("robot", "G1"), self.world(scenario)); r = p["radius"]
        steps = [M.demand(t, mk, push=[-0.03, 0.0, 0.0], horizon=2.5, grasp=mk["name"], min_disp=0.0)]
        for k in range(1, int(p["quadrants"]) + 1):     # WORLD crank: arm along perp = axis x z = -y at rest; a positive turn about +x carries the knob -y -> +z -> +y -> -z
            a = k*np.pi/2; tgt = np.array([0.0, -r*np.cos(a), r*np.sin(a)]) - np.array([0.0, -r, 0.0])      # knob displacement from its start position (an external event on the crank angle)
            steps.append(M.demand(t, mk, push=(np.array([-0.03, 0.0, 0.0]) + tgt).tolist(), horizon=2.0, min_disp=np.pi/2*0.6, f_max=p["f_max"], held=[mk["name"]]))
        return M.plan(t, steps, events=("held",) + ("quadrant_passed",)*(len(steps) - 1), freeze_T=0.3)
