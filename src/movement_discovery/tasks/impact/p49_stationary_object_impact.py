"""Program 49 (impact): strike a STATIONARY ball on a tee with the palm; value = ball speed after impact inside a cone about the
requested direction (ImpactTask: effector sphere collider, graded miss, cone judged on the ball). Scenario varies the ball's
position in the sweep zone, its mass and radius, and the direction. integration check: 2026-10-12 prior-only rollout (seed 0): NOT_HIT, closest approach 0.30 m, effector 3.1 m/s (the prior does not know where the ball is: the search must put the effector through it); 15 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, GoalSpec, Scenario
from movement_discovery.core.registry import register
from ._local import ImpactProgram


@register
class StationaryObjectImpact(ImpactProgram):
    id = "P49_stationary_object_impact"; family = Family.IMPACT; title = "Impact on a stationary object (ball on a tee)"
    requires = [Capability.PALM_SITE_RIGHT, Capability.EFFECTOR_COLLIDER, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]
    distinct_from = "P41/P48 move the effector with a merged mass; here momentum is transferred in a short rigid contact to a free object whose position the prior does not know."
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed)
        p = dict(object=dict(pos=[float(rng.uniform(0.25, 0.40)), float(rng.uniform(-0.35, -0.15)), float(rng.uniform(0.55, 0.95))], radius=float(rng.uniform(0.04, 0.07)), mass=float(rng.choice([0.06, 0.1, 0.25])), tee=True), direction=(float(np.radians(rng.uniform(-10, 20))), float(np.radians(rng.uniform(-15, 15)))), elev_tol=np.radians(20.0), az_tol=np.radians(20.0)); p.update(ov)
        return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=p)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="object_velocity", params=dict(object="ball", direction=p["direction"]), tolerance=dict(elev=p["elev_tol"], az=p["az_tol"]))
