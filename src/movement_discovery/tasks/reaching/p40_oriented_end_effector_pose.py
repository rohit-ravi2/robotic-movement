"""Program 40 (reaching): reach a point AND hold a prescribed palm orientation there (the palm's forward axis pointed down, sideways
or up; ReachTask's wrist orientation loop over `orient` items). The observer measures the orientation error during the hold
(ORIENT_OFF above orient_hold rad). value = -work. integration check: 2026-10-12 prior-only rollout (seed 0): FELL, NOT_REACHED; orientation observer ran (ORIENT_OFF not reached because the hold never started); 11 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, GoalSpec, Scenario
from movement_discovery.core.registry import register
from ._local import ReachProgram, PALM0
DIRS = {"down": [0, 0, -1], "up": [0, 0, 1], "left": [0, 1, 0], "forward": [1, 0, 0]}


@register
class OrientedEndEffectorPose(ReachProgram):
    id = "P40_oriented_end_effector_pose"; family = Family.REACHING; title = "End-effector pose under orientation constraints"
    requires = [Capability.PALM_SITE_RIGHT, Capability.SERVO_CONTROL, Capability.FLOATING_BASE, Capability.WRIST_3DOF]
    distinct_from = "P33 fixes 3 position coordinates; here 5 (position + a pointing axis) are fixed, so the wrist and the elbow/shoulder redundancy are consumed by the orientation and a different body posture results."
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(target=[float(rng.uniform(0.20, 0.38)), float(rng.uniform(-0.15, 0.15)), float(rng.uniform(-0.10, 0.20))], point=str(rng.choice(list(DIRS))), orient_hold=0.20, tol=0.03, hold=0.30, horizon=2.0); p.update(ov)
        return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=p)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="pose", params=dict(effector="hand_r", target=(PALM0 + np.asarray(p["target"])).tolist(), palm_z_axis=DIRS[p["point"]]), tolerance=dict(pos=p["tol"], orient=p["orient_hold"], hold=p["hold"]), horizon=p["horizon"])
    def demand(self, scenario):
        p = scenario.params; t = self.task(scenario); ori = [dict(body="palm", axis=[0, 0, 1], target=DIRS[p["point"]], k=6.0, max=1.2)]
        return t, dict(t.demand_default(), target_world=(PALM0 + np.asarray(p["target"])).tolist(), orient=ori, tol=p["tol"], hold=p["hold"], horizon=p["horizon"], prior="ascent", approach="direct", observe=dict(orient_hold=p["orient_hold"]))
