"""Program 52 (impact): kick a ball resting on the floor with the right foot on LEFT-foot support (engine.construct 'kick_ball':
KICK_DRIVEN joint set, tracker omega 40, slower reference range; the one-foot preparation comes from the support contract).
value = ball speed in the cone. integration check: 2026-10-12 prior-only rollout (seed 0): one-foot (left support) plant and prepared reference built, NOT_HIT dmin 0.29, foot 0.02 m/s at the prior's release; 9 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, GoalSpec, Scenario
from movement_discovery.core.registry import register
from ._local import ImpactProgram


@register
class FootMediatedImpact(ImpactProgram):
    id = "P52_foot_mediated_impact"; family = Family.IMPACT; title = "Foot-mediated impact (kick of a ball on the floor)"
    requires = [Capability.FOOT_SITE_RIGHT, Capability.HIP_ROLL, Capability.ANKLE_ROLL, Capability.EFFECTOR_COLLIDER, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]
    effector = "right_foot"; support = "left"
    distinct_from = "P49 strikes with the arm on two feet; here the striking limb is a support limb, so the body must first move onto one foot and the ball is at floor level."
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); r = float(rng.uniform(0.08, 0.12))
        p = dict(object=dict(pos=[float(rng.uniform(0.25, 0.40)), float(rng.uniform(-0.22, -0.12)), r], radius=r, mass=float(rng.uniform(0.3, 0.6)), tee=False), direction=(float(np.radians(rng.uniform(5, 30))), float(np.radians(rng.uniform(-20, 20)))), elev_tol=np.radians(20.0), az_tol=np.radians(20.0)); p.update(ov)
        return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=p)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="object_velocity", params=dict(object="ball", effector="foot_r", direction=p["direction"]), tolerance=dict(elev=p["elev_tol"], az=p["az_tol"]))
    def demand_extra(self, p):
        from engine.construct import KICK_DRIVEN; return dict(driven=list(KICK_DRIVEN), omega=40.0, tscale_range=(0.9, 1.45))
