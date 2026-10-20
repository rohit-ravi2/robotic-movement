"""Program 43 (projection): clear an obstacle of height h at range x_obs with a released mass. Flight modelled drag-free from the
release state (_local.height_at_range); value = clearance margin at the obstacle plane (m, higher is better; negative = blocked),
achieved when the margin is >= 0 and the mass lands beyond the obstacle. The wide cone keeps the prior aimed over the obstacle.
integration check: 2026-10-12 prior-only rollout (seed 0): margin -4.3 m -> BLOCKED, LANDS_SHORT; 13 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec, Scenario
from movement_discovery.core.registry import register
from movement_discovery.embodiment import resolve
from . import _local as L


@register
class ProjectileClearance(TaskProgram):
    id = "P43_projectile_clearance"; family = Family.PROJECTION; title = "Projectile clearance over an obstacle at a given range"
    requires = [Capability.PALM_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P42 scores a landing point; here a height constraint at an intermediate range binds the apex (a lob can beat a line drive that lands further)."
    def embodiment(self, robot): return resolve(robot, self)
    def world(self, scenario):
        p = scenario.params; return WorldSpec(surfaces=[dict(name="barrier", pos=[p["x_obs"], 0.0, 0.5*p["h_obs"]], size=[0.05, 1.5, 0.5*p["h_obs"]], rgba=[0.5, 0.5, 0.5, 1], rigid=True)])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(mass=0.145, x_obs=float(rng.uniform(1.5, 3.5)), h_obs=float(rng.uniform(0.8, 2.0)), elev_pref=float(np.radians(rng.uniform(35, 55))), support="both"); p.update(ov)
        return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=p)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="object_displacement", params=dict(clear_plane_x=p["x_obs"], min_height=p["h_obs"], model="drag_free_ballistic"), tolerance=dict(margin=0.0))
    def demand(self, scenario):
        p = scenario.params; return L.bridge(scenario, mass=p["mass"], support=p["support"], direction=(p["elev_pref"], 0.0), elev_tol=np.radians(30.0), az_tol=np.radians(25.0))
    def evaluate(self, rec, scenario):
        p = scenario.params; t, d = self.demand(scenario); rs = L.release_state(t.plant(d), rec); h = L.height_at_range(rs["pos"], rs["v"], p["x_obs"]); lp, tf, _ = L.landing(rs["pos"], rs["v"])
        margin = (h - p["h_obs"]) if h is not None else -(p["h_obs"] + max(0.0, p["x_obs"] - lp[0])); blocked = margin < 0 or h is None
        reasons = (["BLOCKED"] if blocked else []) + (["LANDS_SHORT"] if lp[0] <= p["x_obs"] else [])
        return L.outcome(margin, rec, dict(height_at_obstacle=h, margin=float(margin), landing=lp.tolist(), release_speed=float(np.linalg.norm(rs["v"])), h_rel=float(rs["pos"][2])), extra_reasons=reasons, extra_violation=(1.0 + 2.0*max(0.0, -margin) if reasons else 0.0))
    def validity(self, rec, scenario): return self.evaluate(rec, scenario).validity
    def observe(self, rec, scenario):
        if not rec.get("trace"): return None
        from engine import emergence as EM; t, d = self.demand(scenario); return EM.analyze(t.plant(d), rec, d["effector"])
