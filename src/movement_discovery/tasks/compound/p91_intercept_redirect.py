"""Program 91 (compound): INTERCEPT an incoming ball, then REDIRECT it (throw it toward a target direction). Sub-goals with events:
(1) captured (P73 CatchTask: contact at the end, relative speed killed); (2) released along a direction cone at speed >= v_min
(P41-type effect on the ball now in the hand). Executable today: the catch step; the redirect needs the caught ball to stay with the
hand across the handoff (a weld-on-capture event) and the bare/world plant state mapping (CORE): status SCAFFOLDED, partial run.
integration check: 2026-10-12 — prior-only rollout, plant built, value -5.13, achieved False, gates fired [s0:NOT_CAPTURED,SEQUENCE_NOT_OK,MISSING:projection:release]; rel_speed_end=3.8 captured=False (17 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, Plan
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from movement_discovery.tasks.force_interaction import _base as B
from movement_discovery.tasks.interception import _catch as C
from . import _seq as Q


@register
class InterceptRedirect(Q.CompoundProgram, TaskProgram):
    id = "P91_intercept_redirect"; family = Family.COMPOUND; title = "Catch a ball and throw it back out"
    requires = [Capability.PALM_SITE_RIGHT, Capability.EFFECTOR_COLLIDER, Capability.WELD_GRASP, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SCAFFOLDED
    distinct_from = "P73 ends at the capture; here the capture state is the START of a projection, so the catch must end in a throwable posture (balanced, ball controlled) and the two effects share one continuous state."
    subgoals = [dict(goal="P73_catch_ballistic_object", event="captured", object="ball", horizon=0.8), dict(goal="projection:release", event="object_velocity", object="ball", cone=("elev", "az"), v_min="v_min", horizon=0.8)]
    def obj(self, p): return O.launched_ball(pos=[p["dist"], -0.2, 1.0], vel=[-p["v"], 0.0, p["vz"]], name="ball", radius=0.05, mass=p["mass"])
    def world(self, scenario): o = self.obj(scenario.params); return WorldSpec(objects=[o], surfaces=[C.pedestal(o)])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); dist = float(rng.uniform(1.2, 1.6)); v = float(rng.uniform(2.5, 3.5)); tf = dist/v
        p = dict(dist=dist, v=v, vz=float(0.5*9.81*tf), mass=float(rng.uniform(0.1, 0.3)), rel_speed_max=0.5, f_max=150.0, elev=float(np.radians(20)), az=float(np.radians(rng.uniform(-30, 30))), v_min=4.0); return B.scenario(self, seed, p, ov)
    def demand(self, scenario):
        p = scenario.params; t = C.make(p.get("robot", "G1"), self.world(scenario)); o = self.obj(p)
        step = Q.step(t, C.demand(t, object=dict(pos=o["pos"], radius=0.05, mass=p["mass"], tee=True), rel_speed_max=p["rel_speed_max"], f_max=p["f_max"], mass=0.0), event="captured", representation=Q.TORQUE)
        return Plan(steps=[step], robot=t.robot, missing=[dict(goal=self.subgoals[1]["goal"], why="weld-on-capture event (hold the caught ball across the handoff) + " + Q.NQ_MAPPING)], settle_T=0.0, freeze_T=0.4)
    def evaluate(self, rec, scenario):
        if "steps" in rec: e = rec["steps"][0]["extra"]; return B.seq_outcome(rec, value=float(rec["steps"][0]["value"]), measures=dict(captured=e.get("captured"), rel_speed_end=e.get("rel_speed_end")), extra_reasons=Q.missing_reasons(rec) or ["MISSING:redirect"], achieved=False)
        return B.outcome(rec, value=rec["value"], measures=dict(captured=rec["extra"].get("captured")), extra_reasons=["MISSING:redirect"], achieved=False)
