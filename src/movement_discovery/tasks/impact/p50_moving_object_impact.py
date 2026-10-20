"""Program 50 (impact): strike a MOVING ball: the object is launched toward the strike zone (WORLD `objects.launched_ball`; here
object.qvel0 set in the settled state by _local.ImpactTaskX) and arrives at the strike point at t_arrive on a symmetric parabola.
value = ball speed after the hit in the cone; contact is REQUIRED (an untouched moving ball has speed but is NOT_HIT).
integration check: 2026-10-12 prior-only rollout (seed 0): launched ball starts moving at t=0 (START_NOT_AT_REST stripped for the object only), NOT_HIT dmin 0.37, cone term swapped for the graded miss; 17 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, GoalSpec, Scenario
from movement_discovery.core.registry import register
from ._local import ImpactProgram, G


@register
class MovingObjectImpact(ImpactProgram):
    id = "P50_moving_object_impact"; family = Family.IMPACT; title = "Impact on a moving object (launched ball)"
    requires = [Capability.PALM_SITE_RIGHT, Capability.EFFECTOR_COLLIDER, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]
    distinct_from = "P49's ball waits; here the ball is only at the strike point for a few ms, so the swing must be TIMED and the relative velocity, not the effector speed, sets the outgoing speed."
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); strike = np.array([float(rng.uniform(0.28, 0.40)), float(rng.uniform(-0.35, -0.15)), float(rng.uniform(0.6, 0.9))]); ta = float(rng.uniform(0.35, 0.55))
        vh = np.array([-float(rng.uniform(2.0, 4.0)), float(rng.uniform(-0.5, 0.5))]); vz0 = 0.5*G*ta; pos0 = [float(strike[0] - vh[0]*ta), float(strike[1] - vh[1]*ta), float(strike[2])]
        p = dict(object=dict(pos=pos0, radius=0.05, mass=0.1, tee=False, qvel0=[float(vh[0]), float(vh[1]), float(vz0)]), t_arrive=ta, strike_point=strike.tolist(), direction=(float(np.radians(rng.uniform(0, 25))), 0.0), elev_tol=np.radians(25.0), az_tol=np.radians(30.0)); p.update(ov)
        return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=p)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="object_velocity", params=dict(object="launched_ball", direction=p["direction"], arrives_at=p["t_arrive"]), tolerance=dict(elev=p["elev_tol"], az=p["az_tol"]))
    def demand_extra(self, p): return dict(contact_required=True)
