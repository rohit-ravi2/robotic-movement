"""Program 44 (projection): a released mass must meet a MOVING target (constant-velocity point target at height z_t, time origin at
the rollout start; WORLD name `objects.launched_ball`, modelled here kinematically). Flight drag-free from the release state; the
miss distance is the minimum over the flight of |projectile(t) - target(t)| after release; value = -miss (m), achieved within tol.
The lead (aiming ahead of the target) is not supplied: the cone centre points at the target's start position with wide tolerances.
integration check: 2026-10-12 prior-only rollout (seed 0): miss 3.4 m at t_meet 1.06 s -> TARGET_MISSED; 12 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec, Scenario
from movement_discovery.core.registry import register
from movement_discovery.embodiment import resolve
from . import _local as L


@register
class MovingTargetInterception(TaskProgram):
    id = "P44_moving_target_interception"; family = Family.PROJECTION; title = "Interception of a moving target by a released mass"
    requires = [Capability.PALM_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P42's target is fixed; here the target moves during the wind-up and the flight, so release TIME and lead matter, not only the release vector."
    def embodiment(self, robot): return resolve(robot, self)
    def world(self, scenario):
        p = scenario.params; return WorldSpec(objects=[dict(name="launched_ball", kind="ball", size=[0.1], mass=0.2, pos=list(p["x0"]), graspable=False, rgba=[0.2, 0.8, 0.3, 1], qvel0=list(p["v_t"]) + [0, 0, 0], kinematic=True)])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); x0 = [float(rng.uniform(3.0, 5.0)), float(rng.uniform(-2.0, 2.0)), float(rng.uniform(0.6, 1.6))]
        v_t = [float(rng.uniform(-0.8, 0.8)), float(rng.choice([-1, 1])*rng.uniform(1.0, 2.5)), 0.0]
        p = dict(mass=0.145, x0=x0, v_t=v_t, tol=0.3, elev_pref=float(np.radians(rng.uniform(10, 30))), support="both"); p.update(ov)
        return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=p)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="object_displacement", params=dict(target_motion=dict(x0=p["x0"], v=p["v_t"]), model="drag_free_ballistic"), tolerance=dict(miss=p["tol"]))
    def demand(self, scenario):
        p = scenario.params; az = float(np.arctan2(p["x0"][1], p["x0"][0]))
        return L.bridge(scenario, mass=p["mass"], support=p["support"], direction=(p["elev_pref"], az), elev_tol=np.radians(25.0), az_tol=np.radians(45.0))
    def evaluate(self, rec, scenario):
        p = scenario.params; t, d = self.demand(scenario); rs = L.release_state(t.plant(d), rec); t_rel = float(rec["extra"]["t_rel"])
        lp, tf, _ = L.landing(rs["pos"], rs["v"]); ts = np.linspace(0.0, max(tf, 1e-3), 400)
        proj = np.array([L.position_at(rs["pos"], rs["v"], s) for s in ts]); tgt = np.asarray(p["x0"]) + np.outer(t_rel + ts, np.asarray(p["v_t"]))
        dist = np.linalg.norm(proj - tgt, axis=1); i = int(np.argmin(dist)); miss = float(dist[i]); missed = miss > p["tol"]
        return L.outcome(-miss, rec, dict(miss=miss, t_meet=float(t_rel + ts[i]), target_at_meet=tgt[i].tolist(), release_speed=float(np.linalg.norm(rs["v"])), t_rel=t_rel), extra_reasons=(["TARGET_MISSED"] if missed else []), extra_violation=(1.0 + 2.0*(miss - p["tol"]) if missed else 0.0))
    def validity(self, rec, scenario): return self.evaluate(rec, scenario).validity
    def observe(self, rec, scenario):
        if not rec.get("trace"): return None
        from engine import emergence as EM; t, d = self.demand(scenario); return EM.analyze(t.plant(d), rec, d["effector"])
