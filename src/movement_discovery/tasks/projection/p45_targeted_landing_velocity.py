"""Program 45 (projection): land a released mass on a target WITH a prescribed arrival: descent angle >= theta_min (a drop-in, not a
line drive) and landing speed <= v_max. Flight drag-free from the release state; value = -(landing error + angle shortfall penalty),
achieved when the landing is within tol, the descent steep enough and the speed under the cap. Higher releases are the physics here.
integration check: 2026-10-12 prior-only rollout (seed 0): err 2.06 m, descent 42 deg, 6.3 m/s -> TARGET_MISSED, ARRIVAL_TOO_FLAT, ARRIVAL_TOO_FAST (+STANCE_LOST); 13 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec, Scenario
from movement_discovery.core.registry import register
from movement_discovery.embodiment import resolve
from . import _local as L


@register
class TargetedLandingVelocity(TaskProgram):
    id = "P45_targeted_landing_velocity"; family = Family.PROJECTION; title = "Landing on a target with a prescribed arrival angle and speed cap"
    requires = [Capability.PALM_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P42 accepts any trajectory through the landing point; here the arrival velocity is constrained, which selects the high (lob) solution branch and caps speed."
    def embodiment(self, robot): return resolve(robot, self)
    def world(self, scenario):
        p = scenario.params; return WorldSpec(surfaces=[dict(name="target_mark", pos=[p["target"][0], p["target"][1], 0.005], size=[p["tol"], p["tol"], 0.005], rgba=[0.9, 0.8, 0.1, 1], rigid=True)])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(mass=0.145, target=[float(rng.uniform(1.5, 4.0)), float(rng.uniform(-0.8, 0.8)), 0.0], tol=0.4, theta_min=float(np.radians(rng.uniform(50, 70))), v_max=float(rng.uniform(5.0, 8.0)), support="both"); p.update(ov)
        return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=p)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="object_velocity", params=dict(landing=p["target"], descent_min=p["theta_min"], speed_max=p["v_max"], model="drag_free_ballistic"), tolerance=dict(xy=p["tol"]))
    def demand(self, scenario):
        p = scenario.params; az = float(np.arctan2(p["target"][1], p["target"][0]))
        return L.bridge(scenario, mass=p["mass"], support=p["support"], direction=(np.radians(55.0), az), elev_tol=np.radians(30.0), az_tol=np.radians(20.0))
    def evaluate(self, rec, scenario):
        p = scenario.params; t, d = self.demand(scenario); rs = L.release_state(t.plant(d), rec); lp, tf, vl = L.landing(rs["pos"], rs["v"])
        err = float(np.linalg.norm(lp[:2] - np.asarray(p["target"][:2]))); theta = float(np.arctan2(-vl[2], np.hypot(vl[0], vl[1]))); vland = float(np.linalg.norm(vl))
        short = max(0.0, p["theta_min"] - theta); over = max(0.0, vland - p["v_max"]); reasons = (["TARGET_MISSED"] if err > p["tol"] else []) + (["ARRIVAL_TOO_FLAT"] if short > 0 else []) + (["ARRIVAL_TOO_FAST"] if over > 0 else [])
        return L.outcome(-(err + 2.0*short), rec, dict(err=err, descent_angle=theta, landing_speed=vland, landing=lp.tolist(), h_rel=float(rs["pos"][2])), extra_reasons=reasons, extra_violation=((1.0 + 2.0*max(0.0, err - p["tol"]) + 3.0*short + 0.5*over) if reasons else 0.0))
    def validity(self, rec, scenario): return self.evaluate(rec, scenario).validity
    def observe(self, rec, scenario):
        if not rec.get("trace"): return None
        from engine import emergence as EM; t, d = self.demand(scenario); return EM.analyze(t.plant(d), rec, d["effector"])
