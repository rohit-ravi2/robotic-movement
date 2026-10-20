"""Program 36 (reaching): reach a point near the FLOOR (0.08-0.30 m) and hold it: outside the arm's reach from the stance, so the
prior's squat-on-demand (legs enter only when arm+trunk fall short) and the balance controller carry the body down. value = -work.
integration check: 2026-10-12 prior-only rollout (seed 0, target z 0.08-0.30): no fall, NOT_REACHED with dmin 0.51 m (the squat-on-demand prior stops short of floor level); 15 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, GoalSpec, Scenario
from movement_discovery.core.registry import register
from ._local import ReachProgram


@register
class LowPositionReach(ReachProgram):
    id = "P36_low_position_reach"; family = Family.REACHING; title = "Low-position reaching (floor level)"
    requires = [Capability.PALM_SITE_RIGHT, Capability.SERVO_CONTROL, Capability.FLOATING_BASE, Capability.WAIST_PITCH]
    distinct_from = "P33 stays in the arm's workspace; here the target is below the knees so the legs or trunk must lower the shoulder and the COM must stay over the feet while they do."
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(target_world=[float(rng.uniform(0.25, 0.45)), float(rng.uniform(-0.30, -0.05)), float(rng.uniform(0.08, 0.30))], load=float(rng.choice([0.0, 1.0])), tol=0.03, hold=0.30, horizon=2.5, squat=0.35); p.update(ov)
        return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=p)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="pose", params=dict(effector="hand_r", target=p["target_world"], load=p["load"]), tolerance=dict(pos=p["tol"], hold=p["hold"]), horizon=p["horizon"])
    def demand(self, scenario):
        p = scenario.params; t = self.task(scenario); return t, dict(t.demand_default(), target_world=list(p["target_world"]), load=p["load"], tol=p["tol"], hold=p["hold"], horizon=p["horizon"], prior="ascent", squat=p["squat"], omega_legs=150.0)
