"""Program 53 (impact): REDIRECT an incoming ball: it arrives with velocity v_in (launched, _local qvel0) and must leave inside a
cone far from its incoming direction (a deflection of 60-150 deg in azimuth), with contact required. value = outgoing speed in the
cone. integration check: 2026-10-12 prior-only rollout (seed 0): incoming ball launched, NOT_HIT dmin 0.375; 15 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, GoalSpec, Scenario
from movement_discovery.core.registry import register
from ._local import ImpactProgram, G


@register
class MomentumRedirection(ImpactProgram):
    id = "P53_momentum_redirection"; family = Family.IMPACT; title = "Directional redirection of an incoming object"
    requires = [Capability.PALM_SITE_RIGHT, Capability.EFFECTOR_COLLIDER, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]
    distinct_from = "P50 adds speed to a moving ball along the swing; here the incoming momentum must be TURNED, so the contact normal (effector pose at impact) is the controlled quantity, not the effector speed."
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); strike = np.array([float(rng.uniform(0.25, 0.38)), float(rng.uniform(-0.30, -0.10)), float(rng.uniform(0.6, 0.9))]); ta = float(rng.uniform(0.35, 0.55))
        az_in = float(np.radians(rng.uniform(150, 210))); s_in = float(rng.uniform(1.5, 3.0)); vh = np.array([s_in*np.cos(az_in), s_in*np.sin(az_in)]); vz0 = 0.5*G*ta
        az_out = float(np.radians(rng.choice([-1, 1])*rng.uniform(60, 120))); pos0 = [float(strike[0] - vh[0]*ta), float(strike[1] - vh[1]*ta), float(strike[2])]
        p = dict(object=dict(pos=pos0, radius=0.05, mass=0.1, tee=False, qvel0=[float(vh[0]), float(vh[1]), float(vz0)]), t_arrive=ta, incoming_az=az_in, direction=(0.0, az_out), elev_tol=np.radians(30.0), az_tol=np.radians(30.0), hit_min=0.5); p.update(ov)
        return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=p)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="object_velocity", params=dict(object="launched_ball", incoming_az=p["incoming_az"], direction=p["direction"]), tolerance=dict(elev=p["elev_tol"], az=p["az_tol"]))
    def demand_extra(self, p): return dict(contact_required=True)
