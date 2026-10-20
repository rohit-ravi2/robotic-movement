"""Program 74 (interception): ARREST a rolling ball with the foot. A football rolls at the robot along the floor (world.objects.rolling_ball: no-slip spin); the right foot (effector sphere tangent to the sole, left-leg support) must stop it: captured = foot contact at the end with the ball's speed relative to the foot below rel_speed_max. Distinct physics: floor rolling (spin + rolling contact), a foot effector on one-leg support.
integration check: 2026-10-12 — prior-only rollout (results/md_build/checks_interception.json), plant built, value -2.97, achieved False, gates fired [NOT_CAPTURED]; rel_speed_end=1.95 t_contact=0 dmin_obj=0.509 captured=False (18 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from movement_discovery.tasks.force_interaction import _base as B
from . import _catch as C


@register
class ArrestRollingObject(C.CatchProgram, TaskProgram):
    id = "P74_arrest_rolling_object"; family = Family.INTERCEPTION; title = "Stop a rolling ball with the foot"
    requires = [Capability.FOOT_SITE_RIGHT, Capability.EFFECTOR_COLLIDER, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED
    distinct_from = "P73 is an airborne catch by the hand on two feet; here the object rolls on the floor with spin and the stopping effector is a foot on one-leg support, so the support contract and the rolling contact are the physics."
    def obj(self, p): return O.rolling_ball(name="ball", radius=0.11, mass=0.43, pos=[p["dist"], -0.17, 0.11], vel=[-p["v"], 0.0, 0.0])
    def world(self, scenario): return WorldSpec(objects=[self.obj(scenario.params)])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(dist=float(rng.uniform(1.0, 1.6)), v=float(rng.uniform(1.5, 3.0)), rel_speed_max=0.4, f_max=300.0); return B.scenario(self, seed, p, ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="capture", params=dict(object="ball", effector="foot_r", rel_speed_max=p["rel_speed_max"]), horizon=0.8)
    def demand(self, scenario):
        from engine import construct as K
        p = scenario.params; t = C.make(p.get("robot", "G1"), self.world(scenario), colliders=("right_foot",)); o = self.obj(p)
        return t, C.demand(t, effector="right_foot", support="left", object=dict(pos=o["pos"], radius=0.11, mass=0.43, tee=False), rel_speed_max=p["rel_speed_max"], f_max=p["f_max"], driven=K.KICK_DRIVEN, omega=40.0, tscale_range=(0.9, 1.45), direction=(np.radians(15), 0.0), mass=0.0)
    def strategy_space(self, scenario): return C.CatchProgram.strategy_space(self, scenario) + ["torque_knots+absorb_prior"]     # ABSORB (construction/absorb.py) appended
