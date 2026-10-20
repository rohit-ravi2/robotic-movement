"""Program 39 (reaching): place the TIP of a hand-held rod (0.25-0.35 m, WELD hold: ReachTask welds a declared-held object at the
palm at t = 0) on a point in space and hold it. The IK controls the tip (ReachTask ik_point in the palm frame), an orientation item
keeps the rod's axis as at the start (so the palm target = tip target + rod offset), and the observer judges the TIP (TIP_OFF).
value = -work. integration check: 2026-10-12 prior-only rollout (rod welded at the palm, seed 0): TIP_OFF fired with the base gates (FELL, NOT_REACHED); tip point and orientation items resolved; 22 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, GoalSpec, Scenario
from movement_discovery.core.registry import register
from ._local import ReachProgram, PALM0, palm_frame


@register
class ToolTipPositioning(ReachProgram):
    id = "P39_tool_tip_positioning"; family = Family.REACHING; title = "Tool-tip positioning with a held rod"
    requires = [Capability.PALM_SITE_RIGHT, Capability.SERVO_CONTROL, Capability.FLOATING_BASE, Capability.WELD_GRASP]
    distinct_from = "P33 positions the palm itself; here the controlled point is at the end of a lever fixed to the hand, so wrist orientation errors are amplified by the rod length and the rod's mass loads the wrist."
    def scene_objects(self, scenario):
        p = scenario.params; return [dict(name="tool", kind="cylinder", size=[0.012, 0.5*p["rod_len"]], mass=p["rod_mass"], pos=PALM0.tolist(), graspable=True, rgba=[0.3, 0.3, 0.3, 1], quat=[1, 0, 0, 0], rigid=True, soft=None)]
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(rod_len=float(rng.uniform(0.25, 0.35)), rod_mass=float(rng.uniform(0.2, 0.6)), tip_target=[float(rng.uniform(0.30, 0.50)), float(rng.uniform(-0.30, 0.05)), float(rng.uniform(0.15, 0.45))], tol=0.03, hold=0.30, horizon=2.0); p.update(ov)
        return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=p)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="pose", params=dict(point="tool_tip", target=p["tip_target"], rod=p["rod_len"]), tolerance=dict(pos=p["tol"], hold=p["hold"]), horizon=p["horizon"])
    def demand(self, scenario):
        p = scenario.params; t = self.task(scenario); pos, R = palm_frame(scenario.params.get("robot", "G1")); tip_world_off = np.array([0.0, 0.0, -0.5*p["rod_len"]])   # the rod is welded vertical at the palm (identity quat), tip at its lower end
        r_pt = (R.T @ tip_world_off).tolist(); ori = [dict(body="palm", axis=[0, 0, 1], target=(R @ [0, 0, 1]).tolist(), k=4.0, max=0.8), dict(body="palm", axis=[1, 0, 0], target=(R @ [1, 0, 0]).tolist(), k=4.0, max=0.8)]
        return t, dict(t.demand_default(), target_world=(np.asarray(p["tip_target"]) - tip_world_off).tolist(), held=["tool"], ik_point=r_pt, orient=ori, tol=p["tol"], hold=p["hold"], horizon=p["horizon"], prior="ascent", approach="direct", observe=dict(point=r_pt, tip_target=list(p["tip_target"])))
