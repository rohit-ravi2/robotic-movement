"""Program 38 (reaching): trace a CONTINUOUS PATH on a table with a stylus (a small ball welded in the hand, WELD_GRASP) in light
contact: WipeTask's coverage measure (every waypoint passed within tol with the tool on the surface) with a wide force band and
the tool-tilt gate disabled for a point stylus. value = path coverage in [0, 1]. Free-space tracing (no surface) would need a
chained servo reference from CORE. integration check: 2026-10-12 prior-only WipeTask rollout (stylus welded, table scene, seed 0): ran 32 s, coverage 0.0 with PATH_NOT_COVERED and TOOL_OFF_SURFACE; no fall."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, GoalSpec, Scenario
from movement_discovery.core.registry import register
from ._local import ReachProgram
TABLE = dict(name="table", pos=[0.45, -0.25, 0.30], size=[0.25, 0.35, 0.30], rgba=[0.6, 0.4, 0.2, 1], rigid=True)


@register
class ContinuousPathTracing(ReachProgram):
    id = "P38_continuous_path_tracing"; family = Family.REACHING; title = "Continuous path tracing on a surface"
    requires = [Capability.PALM_SITE_RIGHT, Capability.SERVO_CONTROL, Capability.FLOATING_BASE, Capability.WELD_GRASP]
    distinct_from = "P33 is a point goal; here the goal is a CURVE held in contact over seconds, so tracking error accumulates along the path and the stance must hold under a sustained lean."
    def scene_objects(self, scenario): return [dict(name="stylus", kind="ball", size=[0.02], mass=0.05, pos=[0.45, -0.25, 0.62], graspable=True, rgba=[0.2, 0.6, 0.3, 1], quat=[1, 0, 0, 0], rigid=True, soft=None)]
    def scene_surfaces(self, scenario): return [dict(TABLE)]
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); shape = str(rng.choice(["L", "zigzag", "arc"])); c = np.array([0.42, -0.25]); s = float(rng.uniform(0.10, 0.16))
        if shape == "L": pts = [c + [-s, -s], c + [s, -s], c + [s, s]]
        elif shape == "zigzag": pts = [c + [-s, -s], c + [0, s], c + [s, -s], c + [s, s]]
        else: pts = [c + s*np.array([np.cos(a), np.sin(a)]) for a in np.linspace(-2.0, 2.0, 5)]
        p = dict(shape=shape, path=[[float(x), float(y)] for x, y in pts], tol=0.04, force=[0.5, 25.0], duration=float(rng.uniform(2.5, 4.0))); p.update(ov)
        return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=p)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="sequence", params=dict(effector="hand_r", path=p["path"], surface="table", contact="light"), tolerance=dict(path=p["tol"], force=p["force"]), horizon=p["duration"] + 2.0)
    def demand(self, scenario):
        p = scenario.params; t = self.task(scenario, wipe=True)
        return t, dict(t.demand_default(), tool="stylus", surface="table", held=["stylus"], path=list(p["path"]), force=list(p["force"]), duration=p["duration"], tol=p["tol"], press=0.005, tilt_max=180.0, force_control=True, adm_k=0.02, F_ref=3.0)
