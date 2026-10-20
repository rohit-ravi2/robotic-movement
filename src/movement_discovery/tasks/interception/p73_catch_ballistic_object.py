"""Program 73 (interception): CATCH a ballistic incoming object. A ball launched from a pedestal 1.2-1.8 m away toward the chest (world.objects.launched_ball: qvel0 at t = 0) must be received by the hand: captured = contact at the end of the horizon with the ball's speed relative to the hand below rel_speed_max (G1, effector sphere collider) or >= 2 Dex3 hand links in contact (G1H, fingers close when the ball enters close_dist: an event). Peak-force and contact-duration gates as tasks/impact.py.
integration check: 2026-10-12 — prior-only rollout (results/md_build/checks_interception.json), plant built, value -7.36, achieved False, gates fired [NOT_CAPTURED]; rel_speed_end=6.48 t_contact=0 dmin_obj=0.44 captured=False (19 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from movement_discovery.tasks.force_interaction import _base as B
from . import _catch as C


@register
class CatchBallisticObject(C.CatchProgram, TaskProgram):
    id = "P73_catch_ballistic_object"; family = Family.INTERCEPTION; title = "Catch a ball thrown at the chest"
    requires = [Capability.PALM_SITE_RIGHT, Capability.EFFECTOR_COLLIDER, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED
    distinct_from = "P77 intercepts a ball that would MISS the body laterally; P78 asks for deceleration of a heavy object; here the ball comes straight at the chest and the whole problem is matching the hand to a falling arc and killing the relative speed."
    def obj(self, p): return O.launched_ball(pos=[p["dist"], -0.2, p["z0"]], vel=[-p["v"], 0.0, p["vz"]], name="ball", radius=0.05, mass=p["mass"])
    def world(self, scenario): o = self.obj(scenario.params); return WorldSpec(objects=[o], surfaces=[C.pedestal(o)])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); dist = float(rng.uniform(1.2, 1.8)); v = float(rng.uniform(2.5, 4.0)); tf = dist/v
        p = dict(dist=dist, v=v, z0=1.0, vz=float(0.5*9.81*tf + (1.05 - 1.0)/tf), mass=float(rng.uniform(0.1, 0.4)), rel_speed_max=0.5, f_max=150.0, close_dist=0.12); return B.scenario(self, seed, p, ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="capture", params=dict(object="ball", rel_speed_max=p["rel_speed_max"], arrival_t=p["dist"]/p["v"]), horizon=0.8)
    def demand(self, scenario):
        p = scenario.params; t = C.make(p.get("robot", "G1"), self.world(scenario)); o = self.obj(p)
        return t, C.demand(t, object=dict(pos=o["pos"], radius=0.05, mass=p["mass"], tee=True), rel_speed_max=p["rel_speed_max"], f_max=p["f_max"], close_dist=p["close_dist"], mass=0.0)
    def strategy_space(self, scenario): return C.CatchProgram.strategy_space(self, scenario) + ["torque_knots+absorb_prior"]     # ABSORB (construction/absorb.py) appended
