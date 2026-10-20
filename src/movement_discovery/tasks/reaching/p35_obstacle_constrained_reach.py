"""Program 35 (reaching): reach a target behind a BARRIER (a vertical slab between the standing palm and the target; WORLD name
`obstacles.barrier`). The palm, wrist and elbow must not enter the slab's inflated box at any time (OBSTACLE_HIT, graded by the
deepest penetration, computed geometrically: the MJX G1 body carries no collision geoms, so the barrier cannot stop the arm
physically; declared, not approximated). value = -work. integration check: 2026-10-12 prior-only rollout (barrier scene, seed 0): OBSTACLE_HIT fired (geometric gate) alongside FELL/NOT_REACHED; 19 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, GoalSpec, Scenario
from movement_discovery.core.registry import register
from ._local import ReachProgram, PALM0


@register
class ObstacleConstrainedReach(ReachProgram):
    id = "P35_obstacle_constrained_reach"; family = Family.REACHING; title = "Obstacle-constrained reaching"
    requires = [Capability.PALM_SITE_RIGHT, Capability.SERVO_CONTROL, Capability.FLOATING_BASE]
    distinct_from = "P33's straight-line prior is admissible; here the direct path is blocked, so the arm must go over or around and the hold happens in a confined space."
    def barrier(self, p):
        c = PALM0 + np.array([p["gap"], 0.0, -0.05]); return dict(name="barrier", pos=c.tolist(), size=[0.02, 0.18, p["h_half"]], rgba=[0.5, 0.5, 0.55, 1], rigid=True)
    def scene_surfaces(self, scenario): return [self.barrier(scenario.params)]
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(target=[float(rng.uniform(0.38, 0.48)), float(rng.uniform(-0.08, 0.10)), float(rng.uniform(-0.10, 0.08))], gap=float(rng.uniform(0.18, 0.26)), h_half=float(rng.uniform(0.10, 0.18)), margin=0.03, tol=0.03, hold=0.20, horizon=2.0); p.update(ov)
        return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=p)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="pose", params=dict(effector="hand_r", target=(PALM0 + np.asarray(p["target"])).tolist(), avoid="barrier"), tolerance=dict(pos=p["tol"], hold=p["hold"], clearance=p["margin"]), horizon=p["horizon"])
    def demand(self, scenario):
        p = scenario.params; t = self.task(scenario); b = self.barrier(p)
        return t, dict(t.demand_default(), target_world=(PALM0 + np.asarray(p["target"])).tolist(), tol=p["tol"], hold=p["hold"], horizon=p["horizon"], prior="ascent", approach="direct", observe=dict(obstacle=dict(pos=b["pos"], size=b["size"], margin=p["margin"])))
