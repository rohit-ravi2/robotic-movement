"""Program 46 (projection): release a ball ONTO THE FLOOR so that it rolls to rest at a target. Modelled: drag-free fall from the
release state to the floor; if the vertical impact speed exceeds v_bounce the ball bounces (NOT_ROLLED: it is a throw, not a roll);
otherwise the horizontal speed carries into a roll decelerated by rolling resistance c_rr*g (_local.rollout_distance). value =
-|stop point - target| (m). The low release and the forward-down direction are what distinguish it physically from P42.
integration check: 2026-10-12 prior-only rollout (seed 0): impact vz -3.5 m/s -> NOT_ROLLED, err 16 m (the throw-like prior is exactly what the gate must exclude); 12 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec, Scenario
from movement_discovery.core.registry import register
from movement_discovery.embodiment import resolve
from . import _local as L


@register
class RollingDelivery(TaskProgram):
    id = "P46_rolling_delivery"; family = Family.PROJECTION; title = "Rolling delivery of a ball released onto the floor"
    requires = [Capability.PALM_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P42 flies to the target; here the ball must reach the floor gently (bounce gate) and the floor's rolling resistance, not gravity, sets the range."
    def embodiment(self, robot): return resolve(robot, self)
    def world(self, scenario):
        p = scenario.params; return WorldSpec(surfaces=[dict(name="target_mark", pos=[p["target"][0], p["target"][1], 0.005], size=[p["tol"], p["tol"], 0.005], rgba=[0.9, 0.8, 0.1, 1], rigid=True)], friction=p["c_rr"])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(mass=0.145, target=[float(rng.uniform(2.0, 6.0)), float(rng.uniform(-0.5, 0.5)), 0.0], tol=0.4, c_rr=float(rng.uniform(0.02, 0.08)), v_bounce=1.0, support="both"); p.update(ov)
        return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=p)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="object_displacement", params=dict(stop_point=p["target"], model="fall+roll", c_rr=p["c_rr"]), tolerance=dict(xy=p["tol"]))
    def demand(self, scenario):
        p = scenario.params; az = float(np.arctan2(p["target"][1], p["target"][0]))
        return L.bridge(scenario, mass=p["mass"], support=p["support"], direction=(np.radians(-15.0), az), elev_tol=np.radians(25.0), az_tol=np.radians(20.0))
    def evaluate(self, rec, scenario):
        p = scenario.params; t, d = self.demand(scenario); rs = L.release_state(t.plant(d), rec); cp, tf, vi = L.landing(rs["pos"], rs["v"]); vh = np.array([vi[0], vi[1]]); sh = float(np.linalg.norm(vh))
        bounced = -vi[2] > p["v_bounce"]; run = L.rollout_distance(sh, p["c_rr"]); stop = cp[:2] + (vh/max(sh, 1e-9))*run; err = float(np.linalg.norm(stop - np.asarray(p["target"][:2])))
        reasons = (["NOT_ROLLED"] if bounced else []) + (["TARGET_MISSED"] if err > p["tol"] else [])
        return L.outcome(-err, rec, dict(err=err, stop=stop.tolist(), floor_contact=cp.tolist(), impact_vz=float(vi[2]), run_out=run, h_rel=float(rs["pos"][2])), extra_reasons=reasons, extra_violation=((1.0 + 2.0*max(0.0, err - p["tol"]) + 2.0*max(0.0, -vi[2] - p["v_bounce"])) if reasons else 0.0))
    def validity(self, rec, scenario): return self.evaluate(rec, scenario).validity
    def observe(self, rec, scenario):
        if not rec.get("trace"): return None
        from engine import emergence as EM; t, d = self.demand(scenario); return EM.analyze(t.plant(d), rec, d["effector"])
