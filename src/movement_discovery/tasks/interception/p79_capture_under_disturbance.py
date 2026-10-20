"""Program 79 (interception): CAPTURE under a support disturbance. While the ball is in flight a horizontal force (world.disturbances.force: t0-t1, vec, body) shoves the pelvis; the catch must still succeed and the stance hold (STANCE_LOST / FELL gates of the reference rollout). The disturbance is applied by physics.disturbance.apply when CORE provides it, else the local fallback in _catch (xfrc_applied, accounted in the energy closure).
integration check: 2026-10-12 — prior-only rollout (results/md_build/checks_interception.json), plant built, value -6.55, achieved False, gates fired [STANCE_LOST,NOT_CAPTURED]; rel_speed_end=5.18 t_contact=0 dmin_obj=0.683 captured=False W_ext=11.5 (24 s) Re-checked 2026-10-12 through runner.check (world.colliders + physics.disturbance.apply_power): value -6.550, gates [STANCE_LOST,NOT_CAPTURED], viol 42.51 (vs 42.33: runner.evaluate re-judges the energy closure with W_ext), wrench_steps 150, 51 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from movement_discovery.tasks.force_interaction import _base as B
from . import _catch as C


@register
class CaptureUnderDisturbance(C.CatchProgram, TaskProgram):
    id = "P79_capture_under_disturbance"; family = Family.INTERCEPTION; title = "Catch a ball while being shoved"
    requires = [Capability.PALM_SITE_RIGHT, Capability.EFFECTOR_COLLIDER, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED
    distinct_from = "P73 is the same catch on an undisturbed support; here an external wrench on the pelvis during the flight makes balance recovery and the catch one coupled problem."
    def obj(self, p): return O.launched_ball(pos=[p["dist"], -0.2, 1.0], vel=[-p["v"], 0.0, p["vz"]], name="ball", radius=0.05, mass=p["mass"])
    def dist(self, p):
        from movement_discovery.world import disturbances as D; return D.force(p["t0"], p["t0"] + p["dur"], [p["F"]*np.cos(p["dir"]), p["F"]*np.sin(p["dir"]), 0.0], body="pelvis")
    def world(self, scenario): o = self.obj(scenario.params); return WorldSpec(objects=[o], surfaces=[C.pedestal(o)], disturbances=[self.dist(scenario.params)])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); dist = float(rng.uniform(1.2, 1.6)); v = float(rng.uniform(2.5, 3.5)); tf = dist/v
        p = dict(dist=dist, v=v, vz=float(0.5*9.81*tf), mass=float(rng.uniform(0.1, 0.3)), t0=float(rng.uniform(0.05, 0.25)), dur=0.15, F=float(rng.uniform(60.0, 150.0)), dir=float(rng.uniform(-np.pi, np.pi)), rel_speed_max=0.5, f_max=150.0); return B.scenario(self, seed, p, ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="capture", params=dict(object="ball", rel_speed_max=p["rel_speed_max"], disturbance=self.dist(p)), horizon=0.8)
    def demand(self, scenario):
        p = scenario.params; t = C.make(p.get("robot", "G1"), self.world(scenario)); o = self.obj(p)
        return t, C.demand(t, object=dict(pos=o["pos"], radius=0.05, mass=p["mass"], tee=True), rel_speed_max=p["rel_speed_max"], f_max=p["f_max"], disturbances=[self.dist(p)], mass=0.0)
    def evaluate(self, rec, scenario):
        out = super().evaluate(rec, scenario); out.measures["disturbance_Ns"] = scenario.params["F"]*scenario.params["dur"]; out.measures["feet_bw"] = rec["extra"].get("feet_bw"); return out
    def strategy_space(self, scenario): return C.CatchProgram.strategy_space(self, scenario) + ["torque_knots+absorb_prior"]     # ABSORB (construction/absorb.py) appended
