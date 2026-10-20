"""Program 34 (reaching): touch a MOVING target: a ball rolling on a table (velocity set at t = 0; rolling after the slide, ~5/7 of
the launch speed). The reach prior aims at the ball's PREDICTED position at t_meet (supplied geometry, not a technique); the
outcome gate is physical: the palm must come within tol of the ball's surface at some time (NO_CONTACT otherwise). value = -work.
integration check: 2026-10-12 prior-only rollout (table + rolling ball scene, seed 0): ball launched at t=0, robot FELL (reference fragility, see P33), NO_CONTACT fired with the ReachTask gates; 18 s. Re-checked 2026-10-12 after the scene moved to world.build (Plan runner): value -887.1, gates [NONFOOT_CONTACT,STANCE_LOST,FELL,NOT_REACHED,END_UNBALANCED,NO_CONTACT], 48 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, GoalSpec, Scenario
from movement_discovery.core.registry import register
from ._local import ReachProgram
TABLE = dict(name="table", pos=[0.50, -0.30, 0.275], size=[0.30, 0.55, 0.275], rgba=[0.6, 0.4, 0.2, 1], rigid=True)


@register
class MovingTargetContact(ReachProgram):
    id = "P34_moving_target_contact"; family = Family.REACHING; title = "Moving target contact"
    requires = [Capability.PALM_SITE_RIGHT, Capability.SERVO_CONTROL, Capability.FLOATING_BASE]
    distinct_from = "P33's target waits; here the target position is a function of time, so arrival TIME (not only place) decides contact."
    def scene_objects(self, scenario):
        p = scenario.params; return [dict(name="launched_ball", kind="ball", size=[p["radius"]], mass=0.1, pos=list(p["x0"]), graspable=False, rgba=[0.2, 0.8, 0.3, 1], quat=[1, 0, 0, 0], rigid=True, soft=None)]
    def scene_surfaces(self, scenario): return [dict(TABLE)]
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); r = 0.04; sgn = float(rng.choice([-1, 1])); v = float(rng.uniform(0.15, 0.35)); t_meet = float(rng.uniform(0.9, 1.3))
        x0 = [float(rng.uniform(0.35, 0.50)), float(-0.25 - sgn*0.5*0.9*v*t_meet*2), 0.55 + r]
        p = dict(x0=x0, v=[0.0, sgn*v, 0.0], radius=r, t_meet=t_meet, tol=0.03, hold=0.05, horizon=2.0); p.update(ov)
        return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=p)
    def predicted(self, p): return (np.asarray(p["x0"]) + (5.0/7.0)*np.asarray(p["v"])*p["t_meet"] + np.array([0, 0, p["radius"]])).tolist()
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="pose", params=dict(effector="hand_r", target_motion=dict(x0=p["x0"], v=p["v"]), contact="touch"), tolerance=dict(pos=p["tol"]), horizon=p["horizon"])
    def demand(self, scenario):
        p = scenario.params; t = self.task(scenario)
        return t, dict(t.demand_default(), target_world=self.predicted(p), tol=p["tol"], hold=p["hold"], horizon=p["horizon"], prior="ascent", approach="direct", observe=dict(moving_target=dict(name="launched_ball", v=p["v"], radius=p["radius"])))
