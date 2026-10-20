"""Program 77 (interception): LATERAL interception. The ball's path passes 0.3-0.6 m to the robot's side at shoulder height; the hand must get out to the line of flight and capture (contact at the end, relative speed below rel_speed_max). The lateral offset is the physics: the reach is off the sagittal plane and the trunk/pelvis must contribute.
integration check: 2026-10-12 — prior-only rollout (results/md_build/checks_interception.json), plant built, value -6.61, achieved False, gates fired [NOT_CAPTURED]; rel_speed_end=5.63 t_contact=0 dmin_obj=0.493 captured=False (18 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from movement_discovery.tasks.force_interaction import _base as B
from . import _catch as C


@register
class LateralInterception(C.CatchProgram, TaskProgram):
    id = "P77_lateral_interception"; family = Family.INTERCEPTION; title = "Intercept a ball passing to the side"
    requires = [Capability.PALM_SITE_RIGHT, Capability.EFFECTOR_COLLIDER, Capability.WAIST_YAW, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED
    distinct_from = "P73's ball arrives at the chest; here it would miss the body, so the capture requires a lateral displacement of the hand and a waist/pelvis contribution rather than a frontal match."
    def obj(self, p): return O.launched_ball(pos=[p["dist"], p["side"]*p["offset"], 1.1], vel=[-p["v"], 0.0, p["vz"]], name="ball", radius=0.05, mass=p["mass"])
    def world(self, scenario): o = self.obj(scenario.params); return WorldSpec(objects=[o], surfaces=[C.pedestal(o)])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); dist = float(rng.uniform(1.2, 1.6)); v = float(rng.uniform(2.5, 3.5)); tf = dist/v
        p = dict(dist=dist, v=v, vz=float(0.5*9.81*tf), side=float(rng.choice([-1.0, 1.0])), offset=float(rng.uniform(0.3, 0.6)), mass=float(rng.uniform(0.1, 0.3)), rel_speed_max=0.5, f_max=150.0); return B.scenario(self, seed, p, ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="capture", params=dict(object="ball", lateral_offset=p["side"]*p["offset"], rel_speed_max=p["rel_speed_max"]), horizon=0.8)
    def demand(self, scenario):
        p = scenario.params; t = C.make(p.get("robot", "G1"), self.world(scenario)); o = self.obj(p)
        return t, C.demand(t, object=dict(pos=o["pos"], radius=0.05, mass=p["mass"], tee=True), rel_speed_max=p["rel_speed_max"], f_max=p["f_max"], direction=(0.0, float(np.sign(p["side"]*p["offset"]))*np.radians(60)), mass=0.0)
    def evaluate(self, rec, scenario):
        out = super().evaluate(rec, scenario); out.measures["lateral_offset"] = scenario.params["side"]*scenario.params["offset"]; return out
    def strategy_space(self, scenario): return C.CatchProgram.strategy_space(self, scenario) + ["torque_knots+absorb_prior"]     # ABSORB (construction/absorb.py) appended
