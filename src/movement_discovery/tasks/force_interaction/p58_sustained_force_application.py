"""Program 58 (force_interaction): SUSTAINED force application. A box must be moved by a contact force that stays inside a band
for a minimum contact duration (the slap of the velocity construction is forbidden by the reference gates f_max / t_contact_min of
tasks/impact.py). Outcome = displacement with the force window satisfied. Bridge: ImpactTask push_box_sustained demand.
integration check: 2026-10-12 — prior-only rollout, plant built, value 0, achieved False, gates fired [NOT_HIT,CONTACT_TOO_BRIEF,FORCE_TOO_LOW]; displacement=0 t_contact=0 dmin_obj=0.552 (33 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from . import _base as B


@register
class SustainedForceApplication(TaskProgram):
    id = "P58_sustained_force_application"; family = Family.FORCE_INTERACTION; title = "Move a box with a force held in a band over a window"
    requires = [Capability.PALM_SITE_RIGHT, Capability.EFFECTOR_COLLIDER, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED
    distinct_from = "P57 is a threshold problem (nothing moves until mu m g); here the box moves freely and the constraint is on the force-time profile (peak cap, duration floor, force floor)."
    def embodiment(self, robot): return B.embodiment(robot, self.requires)
    def world(self, scenario): p = scenario.params; return WorldSpec(objects=[O.box(name="ball", half=p["size"], mass=p["mass"], pos=p["pos"], graspable=False)])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); f_hi = float(rng.choice([200.0, 300.0, 400.0]))
        p = dict(mass=float(rng.uniform(4.0, 10.0)), size=[0.15, 0.15, 0.30], pos=[0.50, -0.20, 0.30], support="both", f_max=f_hi, f_min=0.15*f_hi, t_contact_min=float(rng.choice([0.2, 0.3, 0.4])), min_disp=0.02)
        return B.scenario(self, seed, p, ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="contact_force", params=dict(object="ball", band=(p["f_min"], p["f_max"]), window=p["t_contact_min"], min_disp=p["min_disp"]), horizon=0.8)
    def strategy_space(self, scenario): return ["torque_knots+tracked_prior", "torque_knots", "servo_knots", "servo_knots+brace_prior", "torque_knots+brace_prior"]
    def demand(self, scenario):
        p = scenario.params
        return B.impact_task(p.get("robot", "G1"), effector="right_palm", support=p["support"], direction=(0.0, 0.0), measure="displacement", hit_min=p["min_disp"], elev_tol=np.radians(89), az_tol=np.radians(30), f_max=p["f_max"], t_contact_min=p["t_contact_min"], object=dict(kind="box", pos=p["pos"], size=p["size"], mass=p["mass"]))
    def evaluate(self, rec, scenario):
        p = scenario.params; e = rec.get("extra", {}); disp = float(e.get("ball_speed", rec.get("value", 0.0))); fmax = float(e.get("f_max_obj", 0.0))
        extra = ["FORCE_TOO_LOW"] if fmax < p["f_min"] else []
        return B.outcome(rec, value=disp, measures=B.impact_measures(rec) | dict(displacement=disp, force_in_band=(p["f_min"] <= fmax <= p["f_max"])), extra_reasons=extra, extra_violation=(1.0 + 0.01*(p["f_min"] - fmax) if extra else 0.0))
    def validity(self, rec, scenario): return self.evaluate(rec, scenario).validity
    def observe(self, rec, scenario): t, d = self.demand(scenario); return B.observe_impact(t, d, rec, "right_palm")
