"""Program 37 (reaching): reach a point OVERHEAD (1.45-1.75 m, above the standing palm by 0.8-1.1 m) and hold it: full arm
elevation with trunk extension, the COM shifting forward under the raised arm. value = -work. integration check: 2026-10-12 prior-only rollout (seed 0, overhead target): FELL, NOT_REACHED dmin 0.11; gates fire; 20 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, GoalSpec, Scenario
from movement_discovery.core.registry import register
from ._local import ReachProgram


@register
class HighPositionReach(ReachProgram):
    id = "P37_high_position_reach"; family = Family.REACHING; title = "High-position reaching (overhead)"
    requires = [Capability.PALM_SITE_RIGHT, Capability.SERVO_CONTROL, Capability.FLOATING_BASE]
    distinct_from = "P36 lowers the body; here the target is above the head, so shoulder elevation and trunk extension, not the legs, bound the reach and gravity loads the arm along its length."
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(target_world=[float(rng.uniform(0.10, 0.35)), float(rng.uniform(-0.35, -0.05)), float(rng.uniform(1.45, 1.75))], load=float(rng.choice([0.0, 0.5, 1.0])), tol=0.03, hold=0.30, horizon=2.0); p.update(ov)
        return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=p)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="pose", params=dict(effector="hand_r", target=p["target_world"], load=p["load"]), tolerance=dict(pos=p["tol"], hold=p["hold"]), horizon=p["horizon"])
    def demand(self, scenario):
        p = scenario.params; t = self.task(scenario); return t, dict(t.demand_default(), target_world=list(p["target_world"]), load=p["load"], tol=p["tol"], hold=p["hold"], horizon=p["horizon"], prior="ascent", approach="direct", trunk_cap=0.6)
