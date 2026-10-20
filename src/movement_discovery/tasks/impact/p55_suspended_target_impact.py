"""Program 55 (impact): strike a SUSPENDED target: a hinge pendulum (pivot overhead, bob at the strike height; WORLD name
`objects.suspended_target`, here built by _local.ImpactTaskX.scene). value = the pendulum's peak swing energy (J) over the rollout
(kinetic + potential above rest); no cone (the pendulum has one degree of freedom). The ImpactTask ball is a far-away dummy.
integration check: 2026-10-12 prior-only rollout (seed 0): pendulum hinge body compiled and keyframes padded, swing energy 0 J, NOT_HIT (closest approach to the bob); 15 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, GoalSpec, Scenario, WorldSpec
from movement_discovery.core.registry import register
from ._local import ImpactProgram


@register
class SuspendedTargetImpact(ImpactProgram):
    id = "P55_suspended_target_impact"; family = Family.IMPACT; title = "Impact on a suspended (pendulum) target"
    requires = [Capability.PALM_SITE_RIGHT, Capability.EFFECTOR_COLLIDER, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]
    distinct_from = "P49's free ball leaves with whatever velocity the contact gives it; a pendulum bob is constrained to an arc, so only the impulse component along the arc counts and the outcome is energy, not a velocity vector."
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); L_ = float(rng.uniform(0.8, 1.5)); z_bob = float(rng.uniform(0.6, 0.9)); x = float(rng.uniform(0.28, 0.42)); y = float(rng.uniform(-0.35, -0.15))
        p = dict(pendulum=dict(pivot=[x, y, z_bob + L_], length=L_, mass=float(rng.uniform(0.3, 1.5)), radius=0.06), object=dict(pos=[3.0, 3.0, 0.05], radius=0.05, mass=0.1, tee=False), direction=(0.0, 0.0)); p.update(ov)
        return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=p)
    def world(self, scenario):
        return super().world(scenario)            # the pendulum is built by ImpactTaskX.scene from demand["pendulum"] (world.build has no "pendulum" kind; a WORLD suspended_target here would be a second, unused pendulum)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="object_velocity", params=dict(object="suspended_target", measure="swing_energy", direction=p["direction"]), tolerance={})
    def demand_extra(self, p): return dict(pendulum=dict(p["pendulum"]), ignore_cone=True, elev_tol=np.radians(89.0), az_tol=np.radians(179.0))
