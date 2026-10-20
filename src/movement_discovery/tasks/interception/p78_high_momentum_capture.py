"""Program 78 (interception): HIGH-MOMENTUM capture and deceleration. A heavy ball (2-5 kg at 3-5 m/s: 6-25 N s) must be received and brought to rest relative to the hand; the OUTCOME is the peak contact force (lower is better) subject to the capture condition (contact at the end, relative speed <= rel_speed_max). The deceleration distance the body affords (arm give, trunk, legs) is what the search must discover.
integration check: 2026-10-12 — prior-only rollout (results/md_build/checks_interception.json), plant built, value -0, achieved False, gates fired [NOT_CAPTURED]; rel_speed_end=6.98 t_contact=0 peak_force=0 dmin_obj=0.309 captured=False (25 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from movement_discovery.tasks.force_interaction import _base as B
from . import _catch as C


@register
class HighMomentumCapture(C.CatchProgram, TaskProgram):
    id = "P78_high_momentum_capture"; family = Family.INTERCEPTION; title = "Receive a heavy fast object with the least peak force"
    requires = [Capability.PALM_SITE_RIGHT, Capability.EFFECTOR_COLLIDER, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED
    distinct_from = "P73 scores the capture only; here the momentum is 10-50x larger and the objective is the PEAK FORCE of the capture, so compliance (give) over a distance, not matching, is the physics."
    def obj(self, p): return O.launched_ball(pos=[p["dist"], -0.15, 1.0], vel=[-p["v"], 0.0, p["vz"]], name="ball", radius=0.10, mass=p["mass"])
    def world(self, scenario): o = self.obj(scenario.params); return WorldSpec(objects=[o], surfaces=[C.pedestal(o)])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); dist = float(rng.uniform(1.2, 1.6)); v = float(rng.uniform(3.0, 5.0)); tf = dist/v
        p = dict(dist=dist, v=v, vz=float(0.5*9.81*tf), mass=float(rng.uniform(2.0, 5.0)), rel_speed_max=0.6, f_cap=1500.0); p["momentum"] = p["mass"]*p["v"]; return B.scenario(self, seed, p, ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="contact_force", params=dict(object="ball", minimise="peak_force", momentum=p["momentum"], rel_speed_max=p["rel_speed_max"]), horizon=0.8)
    def demand(self, scenario):
        p = scenario.params; t = C.make(p.get("robot", "G1"), self.world(scenario)); o = self.obj(p)
        return t, C.demand(t, object=dict(pos=o["pos"], radius=0.10, mass=p["mass"], tee=True), rel_speed_max=p["rel_speed_max"], f_max=p["f_cap"], mass=0.0)
    def evaluate(self, rec, scenario):
        e = rec["extra"]; fpk = float(e.get("f_max_obj", 0.0)); out = B.outcome(rec, value=-fpk, measures=self.measures(rec) | dict(peak_force=fpk, momentum=scenario.params["momentum"], rel_speed_end=e.get("rel_speed_end")), achieved=bool(e.get("captured"))); return out
    def strategy_space(self, scenario): return C.CatchProgram.strategy_space(self, scenario) + ["torque_knots+absorb_prior"]     # ABSORB (construction/absorb.py) appended
