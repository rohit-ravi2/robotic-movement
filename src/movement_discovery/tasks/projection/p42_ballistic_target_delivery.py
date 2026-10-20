"""Program 42 (projection): deliver a released mass to a ground target. The construction (EffectTask) produces a release state;
the flight is drag-free ballistics from the release position and height (_local.landing); outcome = -landing error (m), achieved
when inside the target tolerance. The direction cone is wide and only keeps the body-derived prior aimed; the landing judges.
integration check: 2026-10-12 prior-only rollout (seed 0): release 3.0 m/s at h 0.69 m, landing error 3.55 m -> TARGET_MISSED + OUT_OF_CONE; 11 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec, Scenario
from movement_discovery.core.registry import register
from movement_discovery.embodiment import resolve
from . import _local as L


@register
class BallisticTargetDelivery(TaskProgram):
    id = "P42_ballistic_target_delivery"; family = Family.PROJECTION; title = "Ballistic delivery of a released mass to a ground target"
    requires = [Capability.PALM_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P41 maximises speed in a cone; here speed and direction are coupled by a landing point (a slower, higher release can be the better one)."
    def embodiment(self, robot): return resolve(robot, self)
    def world(self, scenario):
        p = scenario.params; return WorldSpec(surfaces=[dict(name="target_mark", pos=[p["target"][0], p["target"][1], 0.005], size=[p["tol"], p["tol"], 0.005], rgba=[0.9, 0.8, 0.1, 1], rigid=True)], tag=None)
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); R_ = float(rng.uniform(2.0, 6.0)); y = float(rng.uniform(-1.0, 1.0))
        p = dict(mass=float(rng.choice([0.145, 0.3])), target=[R_, y, 0.0], tol=0.5, elev_pref=float(np.radians(rng.uniform(25, 40))), support="both"); p.update(ov)
        return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=p)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="object_displacement", params=dict(landing=p["target"], model="drag_free_ballistic"), tolerance=dict(xy=p["tol"]))
    def demand(self, scenario):
        p = scenario.params; az = float(np.arctan2(p["target"][1], p["target"][0]))
        return L.bridge(scenario, mass=p["mass"], support=p["support"], direction=(p["elev_pref"], az), elev_tol=np.radians(25.0), az_tol=np.radians(20.0))
    def _landing(self, rec, scenario):
        t, d = self.demand(scenario); S = t.plant(d); rs = L.release_state(S, rec); lp, tf, vl = L.landing(rs["pos"], rs["v"], 0.0); return rs, lp, tf, vl
    def evaluate(self, rec, scenario):
        p = scenario.params; rs, lp, tf, vl = self._landing(rec, scenario); err = float(np.linalg.norm(lp[:2] - np.asarray(p["target"][:2])))
        miss = err > p["tol"]; return L.outcome(-err, rec, dict(landing=lp.tolist(), err=err, flight_time=tf, release_speed=float(np.linalg.norm(rs["v"])), h_rel=float(rs["pos"][2])), extra_reasons=(["TARGET_MISSED"] if miss else []), extra_violation=(1.0 + 2.0*(err - p["tol"]) if miss else 0.0))
    def validity(self, rec, scenario): return self.evaluate(rec, scenario).validity
    def observe(self, rec, scenario):
        if not rec.get("trace"): return None
        from engine import emergence as EM; t, d = self.demand(scenario); return EM.analyze(t.plant(d), rec, d["effector"])
