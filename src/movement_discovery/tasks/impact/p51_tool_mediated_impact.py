"""Program 51 (impact): strike a ball with a HAND-HELD TOOL (a 0.3-0.5 m rod). On G1 the hold is a WELD: the rod is a capsule fixed
to the hand body from the palm site (EXPERIMENTAL SIMPLIFICATION, declared as Capability.WELD_GRASP; on G1H the Dex3 wrap of
engine/grasps.py would hold a real tool body). The ball sits at the rod's reach, beyond the hand; the hit must be by the tool
(HIT_WITHOUT_TOOL otherwise). value = ball speed in the cone. integration check: 2026-10-12 prior-only rollout (seed 0): tool capsule compiled on the hand body (0.3-0.5 m), NOT_HIT dmin 0.60; 16 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, GoalSpec, Scenario, WorldSpec
from movement_discovery.core.registry import register
from ._local import ImpactProgram


@register
class ToolMediatedImpact(ImpactProgram):
    id = "P51_tool_mediated_impact"; family = Family.IMPACT; title = "Tool-mediated impact (rod welded in the hand)"
    requires = [Capability.PALM_SITE_RIGHT, Capability.WELD_GRASP, Capability.EFFECTOR_COLLIDER, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]
    distinct_from = "P49 hits with the body's own effector; here a rigid lever with its own mass extends the effector and its tip speed, and the contact point is off the hand."
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); L_ = float(rng.uniform(0.3, 0.5))
        p = dict(tool=dict(length=L_, radius=0.015, mass=float(rng.uniform(0.2, 0.5))), object=dict(pos=[float(0.30 + 0.8*L_), float(rng.uniform(-0.35, -0.15)), float(rng.uniform(0.55, 0.9))], radius=0.04, mass=float(rng.choice([0.06, 0.1])), tee=True), direction=(float(np.radians(rng.uniform(-10, 20))), 0.0), elev_tol=np.radians(25.0), az_tol=np.radians(25.0)); p.update(ov)
        return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=p)
    def world(self, scenario):
        w = super().world(scenario); p = scenario.params; w.objects.append(dict(name="tool", kind="capsule", size=[p["tool"]["radius"], 0.5*p["tool"]["length"]], mass=p["tool"]["mass"], pos=[0.06, -0.24, 0.62], graspable=True, held="weld", rgba=[0.3, 0.3, 0.3, 1])); return w
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="object_velocity", params=dict(object="ball", via="tool", direction=p["direction"]), tolerance=dict(elev=p["elev_tol"], az=p["az_tol"]))
    def demand_extra(self, p): return dict(tool=dict(p["tool"]))
