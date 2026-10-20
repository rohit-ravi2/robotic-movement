"""Program 96 (compound): RETRIEVE a tool (a rod on a table) and OPERATE a mechanism WITH it (press a lever handle down with the rod's
tip). Sub-goals with events: (1) rod held (weld, G1 experimental simplification); (2) lever angle >= min_disp with the contact made by
the ROD (object<->mechanism contact: both are world geoms; the hand itself has no collider here); the lever's load is lifted. Chained
through compose.run_sequence; the second step is a WorldReachTask mechanism demand whose palm target is offset by the held rod's
half-length so the rod tip, not the palm, meets the handle. integration check: 2026-10-12 — prior-only rollout, plant built, value 1.75e-07, achieved False, gates fired [s1:STANCE_LOST,s1:NOT_MOVED,SEQUENCE_NOT_OK]; f_max=0 lever_angle=1.75e-07 (44 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, Plan
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O, mechanisms as MK
from movement_discovery.tasks.force_interaction import _base as B, _mech as M
from . import _seq as Q


@register
class RetrieveAndOperateMechanism(Q.CompoundProgram, TaskProgram):
    id = "P96_retrieve_and_operate_mechanism"; family = Family.COMPOUND; title = "Pick up a rod and press a lever with it"
    requires = [Capability.PALM_SITE_RIGHT, Capability.WELD_GRASP, Capability.SERVO_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED
    distinct_from = "P63 presses the lever with the hand; here the contact is made through a held tool whose length extends the reach and whose pose in the hand (set by the pick) decides where the tip lands: the pick and the press are one coupled problem."
    subgoals = [dict(goal="payload:grasp", event="held", object="rod", horizon=3.0), dict(goal="P63_actuate_lever", event="mechanism_displacement", mechanism="lever", min="min_disp", via="rod", horizon=3.0)]
    def parts(self, p):
        from engine import world as W
        rod = O.cylinder(name="rod", radius=0.015, half_h=p["rod_half"], mass=0.3, pos=[0.36, -0.25, 0.75 + 0.015], quat=[0.7071, 0.0, 0.7071, 0.0])   # lying along x on the table
        rod["grip_point"] = [0.0, 0.0, 0.0]
        return [rod], [W.surface("table", pos=(0.4, -0.25, 0.375), size=(0.2, 0.2, 0.375))], [MK.lever(name="lever", pos=[p["x_lever"], 0.1], height=p["height"], length=0.7, pivot=0.5, resist=0.5, load=p["load"], max_angle=0.6)]
    def world(self, scenario): o, s, m = self.parts(scenario.params); return WorldSpec(objects=o, surfaces=s, mechanisms=m)
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(rod_half=float(rng.uniform(0.15, 0.25)), x_lever=float(rng.uniform(0.85, 1.0)), height=float(rng.uniform(0.7, 0.85)), load=float(rng.uniform(1.0, 3.0)), min_disp=np.radians(15.0), press=0.2, f_max=300.0); return B.scenario(self, seed, p, ov)
    def demand(self, scenario):
        p = scenario.params; spec = self.world(scenario); t = M.make(p.get("robot", "G1"), spec, colliders=(), hand_collision=False); lever = spec.mechanisms[0]
        d1 = dict(t.demand_default(), target_object="rod", push=[0.0, 0.0, 0.01], tol=0.05, hold=0.3, grasp_on_reach="rod", grasp_tol=0.05, omega_servo=120.0, trunk_cap=0.05, horizon=3.0, refine=1, aim_z=True)
        d2 = M.demand(t, lever, push=[-p["rod_half"] - 0.03, 0.0, 0.05 - p["press"]], horizon=3.0, min_disp=p["min_disp"], f_max=p["f_max"], held=["rod"])   # palm target = handle - rod half-length (the rod points +x from the hand) - press depth
        return Plan(steps=[Q.step(t, d1, event="held"), Q.step(t, d2, event="mechanism_displacement")], robot=t.robot, settle_T=0.0, freeze_T=0.4)
    def evaluate(self, rec, scenario):
        e = rec["steps"][-1]["extra"] if "steps" in rec else rec["extra"]; v = float(e.get("mech_disp", 0.0)); held = "rod" in (e.get("held_after") or [])
        return (B.seq_outcome if "steps" in rec else B.outcome)(rec, value=v, measures=dict(lever_angle=v, required=scenario.params["min_disp"], rod_held=held, f_max=e.get("f_max_obj")), extra_reasons=([] if held else ["DROPPED"]), achieved=held and v >= scenario.params["min_disp"])
