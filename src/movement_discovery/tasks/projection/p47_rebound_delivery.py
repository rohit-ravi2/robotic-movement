"""Program 47 (projection): deliver a released ball to a floor target VIA ONE REBOUND, off a wall at x_wall (mode "wall") or off the
floor before the target (mode "floor"). Modelled: drag-free flight, a bounce with normal restitution e and tangential loss mu
(scenario parameters standing in for the contact solref/friction of a simulated ball), then flight to the floor. value = -landing
error after the rebound; the required rebound kind must occur (NO_REBOUND otherwise).
integration check: 2026-10-12 prior-only rollout (seed 0, floor mode): one floor rebound modelled, landing err 1.03 m -> TARGET_MISSED; 15 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec, Scenario
from movement_discovery.core.registry import register
from movement_discovery.embodiment import resolve
from . import _local as L


@register
class ReboundDelivery(TaskProgram):
    id = "P47_rebound_delivery"; family = Family.PROJECTION; title = "Delivery to a target through one wall or floor rebound"
    requires = [Capability.PALM_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P42 is a single free flight; here an inelastic rebound (restitution e) sits between release and landing, so the aim point is a reflection and speed is lost at the bounce."
    def embodiment(self, robot): return resolve(robot, self)
    def world(self, scenario):
        p = scenario.params; walls = [dict(name="wall", pos=[p["x_wall"], 0.0, 1.0], size=[0.05, 2.0, 1.0], rgba=[0.6, 0.6, 0.7, 1], rigid=True)] if p["mode"] == "wall" else []
        return WorldSpec(surfaces=walls + [dict(name="target_mark", pos=[p["target"][0], p["target"][1], 0.005], size=[p["tol"], p["tol"], 0.005], rgba=[0.9, 0.8, 0.1, 1], rigid=True)])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); mode = str(rng.choice(["wall", "floor"])); x_wall = float(rng.uniform(2.0, 3.5))
        target = [float(rng.uniform(0.6, x_wall - 0.4)), float(rng.uniform(-0.5, 0.5)), 0.0] if mode == "wall" else [float(rng.uniform(3.0, 6.0)), float(rng.uniform(-0.5, 0.5)), 0.0]
        p = dict(mass=0.145, mode=mode, x_wall=x_wall, target=target, tol=0.4, e=float(rng.uniform(0.6, 0.85)), mu=0.1, support="both"); p.update(ov)
        return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=p)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="object_displacement", params=dict(landing=p["target"], rebound=p["mode"], restitution=p["e"], model="drag_free+one_bounce"), tolerance=dict(xy=p["tol"]))
    def demand(self, scenario):
        p = scenario.params; elev = np.radians(10.0) if p["mode"] == "wall" else np.radians(-5.0); az = 0.0 if p["mode"] == "wall" else float(np.arctan2(p["target"][1], p["target"][0]))
        return L.bridge(scenario, mass=p["mass"], support=p["support"], direction=(elev, az), elev_tol=np.radians(30.0), az_tol=np.radians(25.0))
    def evaluate(self, rec, scenario):
        p = scenario.params; t, d = self.demand(scenario); rs = L.release_state(t.plant(d), rec)
        p1, v1, kind = L.bounce(rs["pos"], rs["v"], p["e"], p["mu"], wall_x=(p["x_wall"] if p["mode"] == "wall" else None)); lp, tf, _ = L.landing(p1, v1)
        ok_kind = (kind == p["mode"]) and (p["mode"] == "wall" or p1[0] < p["target"][0]); err = float(np.linalg.norm(lp[:2] - np.asarray(p["target"][:2])))
        reasons = ([] if ok_kind else ["NO_REBOUND"]) + (["TARGET_MISSED"] if err > p["tol"] else [])
        return L.outcome(-err, rec, dict(err=err, rebound=kind, rebound_point=p1.tolist(), landing=lp.tolist(), speed_after=float(np.linalg.norm(v1)), h_rel=float(rs["pos"][2])), extra_reasons=reasons, extra_violation=((1.0 + 2.0*max(0.0, err - p["tol"]) + (2.0 if not ok_kind else 0.0)) if reasons else 0.0))
    def validity(self, rec, scenario): return self.evaluate(rec, scenario).validity
    def observe(self, rec, scenario):
        if not rec.get("trace"): return None
        from engine import emergence as EM; t, d = self.demand(scenario); return EM.analyze(t.plant(d), rec, d["effector"])
