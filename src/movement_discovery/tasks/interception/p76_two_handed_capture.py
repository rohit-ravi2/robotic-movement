"""Program 76 (interception): TWO-HANDED capture (Capability.TWO_HANDS: left_palm site on the G1). A larger ball (radius 0.10-0.14 m) comes at the chest; captured = BOTH palm colliders in contact at the end with the relative speed below rel_speed_max. Labelled simplification: each palm is a 4 cm sphere; the left hand's joints (left shoulder pitch/roll) are in the driven set, so the search places it; nothing scripts a two-hand posture.
integration check: 2026-10-12 — prior-only rollout (results/md_build/checks_interception.json), plant built, value -7.26, achieved False, gates fired [NOT_CAPTURED]; rel_speed_end=5.84 t_contact=0 dmin_obj=0.707 captured=False (22 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from movement_discovery.tasks.force_interaction import _base as B
from . import _catch as C


@register
class TwoHandedCapture(C.CatchProgram, TaskProgram):
    id = "P76_two_handed_capture"; family = Family.INTERCEPTION; title = "Catch a large ball between both hands"
    requires = [Capability.PALM_SITE_RIGHT, Capability.PALM_SITE_LEFT, Capability.TWO_HANDS, Capability.EFFECTOR_COLLIDER, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED
    distinct_from = "P73/P75 capture with one hand; the ball here is too large for one hand and the capture condition couples both arms: two contacts at the end, which the single-arm wind-up prior cannot supply on its own."
    def obj(self, p): return O.launched_ball(pos=[p["dist"], 0.0, 1.0], vel=[-p["v"], 0.0, p["vz"]], name="ball", radius=p["radius"], mass=p["mass"])
    def world(self, scenario): o = self.obj(scenario.params); return WorldSpec(objects=[o], surfaces=[C.pedestal(o)])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); dist = float(rng.uniform(1.2, 1.6)); v = float(rng.uniform(2.0, 3.2)); tf = dist/v
        p = dict(dist=dist, v=v, vz=float(0.5*9.81*tf + 0.05/tf), radius=float(rng.uniform(0.10, 0.14)), mass=float(rng.uniform(0.4, 1.0)), rel_speed_max=0.5, f_max=200.0); return B.scenario(self, seed, p, ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="capture", params=dict(object="ball", hands=2, rel_speed_max=p["rel_speed_max"]), horizon=0.8)
    def demand(self, scenario):
        p = scenario.params; t = C.make(p.get("robot", "G1"), self.world(scenario), colliders=("right_palm", "left_palm")); o = self.obj(p)
        return t, C.demand(t, object=dict(pos=o["pos"], radius=p["radius"], mass=p["mass"], tee=True), rel_speed_max=p["rel_speed_max"], f_max=p["f_max"], hands=2, mass=0.0)
    def strategy_space(self, scenario): return C.CatchProgram.strategy_space(self, scenario) + ["torque_knots+absorb_prior"]     # ABSORB (construction/absorb.py) appended
