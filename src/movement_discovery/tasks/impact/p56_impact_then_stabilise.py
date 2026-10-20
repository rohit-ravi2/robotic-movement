"""Program 56 (impact): strike a ball on a tee AND end in a RECOVERABLE state (the J22 rule in tasks/throw.py: `recoverable=True`
bounds the COM speed at release, keeps the capture point near the feet centre and both feet loaded; NOT_RECOVERABLE otherwise),
on the two-foot stance rule. value = ball speed in the cone. integration check: 2026-10-12 prior-only rollout (seed 0): recoverable gate active (cp_err 0.028, com 0.08 m/s), NOT_HIT dmin 0.30; 13 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, GoalSpec, Scenario
from movement_discovery.core.registry import register
from ._local import ImpactProgram


@register
class ImpactThenStabilise(ImpactProgram):
    id = "P56_impact_then_stabilise"; family = Family.IMPACT; title = "Impact followed by post-contact stabilisation"
    requires = [Capability.PALM_SITE_RIGHT, Capability.EFFECTOR_COLLIDER, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]
    distinct_from = "P49 may end in any state (a dive after the hit is valid); here the end state must be one a standing controller can recover, which trades speed for a bounded COM state."
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed)
        p = dict(object=dict(pos=[float(rng.uniform(0.25, 0.40)), float(rng.uniform(-0.35, -0.15)), float(rng.uniform(0.55, 0.95))], radius=0.05, mass=float(rng.choice([0.1, 0.25])), tee=True), direction=(float(np.radians(rng.uniform(-10, 20))), 0.0), elev_tol=np.radians(20.0), az_tol=np.radians(20.0), vcom_max=float(rng.uniform(0.4, 0.8)), cp_max=0.10, feet_min_bw=0.10); p.update(ov)
        return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=p)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="sequence", params=dict(object="ball", direction=p["direction"], then="recoverable_state", vcom_max=p["vcom_max"], cp_max=p["cp_max"]), tolerance=dict(elev=p["elev_tol"], az=p["az_tol"]))
    def demand_extra(self, p): return dict(recoverable=True, vcom_max=p["vcom_max"], cp_max=p["cp_max"], feet_min_bw=p["feet_min_bw"])
