"""Program 54 (impact): strike a COMPLIANT ball (contact time constant 0.03-0.15 s on the object's geom, engine.world's per-geom
compliance via _local object.soft; WORLD name `objects.compliant_target`). The soft contact spreads the impulse over a longer
dwell: value = ball speed in the cone, and the record carries t_contact and f_max for the comparison with P49.
integration check: 2026-10-12 prior-only rollout (seed 0): compliant geom (solref 0.03-0.15 s) compiled, NOT_HIT dmin 0.30; 15 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, GoalSpec, Scenario
from movement_discovery.core.registry import register
from ._local import ImpactProgram


@register
class CompliantTargetImpact(ImpactProgram):
    id = "P54_compliant_target_impact"; family = Family.IMPACT; title = "Impact on a compliant target"
    requires = [Capability.PALM_SITE_RIGHT, Capability.EFFECTOR_COLLIDER, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]
    distinct_from = "P49's contact is rigid (ms); here the target's compliance makes the contact dwell comparable to the swing's time scale, so follow-through, not peak speed, transfers momentum."
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed)
        p = dict(object=dict(pos=[float(rng.uniform(0.25, 0.40)), float(rng.uniform(-0.35, -0.15)), float(rng.uniform(0.55, 0.9))], radius=float(rng.uniform(0.05, 0.09)), mass=float(rng.choice([0.1, 0.25, 0.5])), tee=True, soft=float(rng.uniform(0.03, 0.15))), direction=(float(np.radians(rng.uniform(-10, 20))), 0.0), elev_tol=np.radians(20.0), az_tol=np.radians(20.0)); p.update(ov)
        return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=p)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="object_velocity", params=dict(object="compliant_target", soft=p["object"]["soft"], direction=p["direction"]), tolerance=dict(elev=p["elev_tol"], az=p["az_tol"]))
