"""Program 68 (payload): ASYMMETRIC-load lift. The load's centre of mass is offset from the handle (payload_with_handle com_offset),
so a hold at the handle carries a gravity moment that tends to tip the load and twists the wrist; the lift must keep the load level
(tilt gate) while rising. Value = lifted height; validity adds TILTED (object tilt > tilt_max at the end). Hold = G1 weld
(experimental simplification: the weld transmits the moment rigidly; a real grip would slip, which is the G1H finger follow-up).
integration check: 2026-10-12 — prior-only rollout, plant built, value 0.0996, achieved False, gates fired [s1:NOT_REACHED,SEQUENCE_NOT_OK,TILTED]; lift_height=0.0996 f_max=0 tilt=0.506 (43 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from . import _pay as P


@register
class AsymmetricLoadLift(P.PayloadProgram, TaskProgram):
    id = "P68_asymmetric_load_lift"; family = Family.PAYLOAD; title = "Lift a load whose mass centre is offset from its handle"
    requires = [Capability.PALM_SITE_RIGHT, Capability.WELD_GRASP, Capability.SERVO_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED
    distinct_from = "P65/P67 lift balanced loads; here the gravity moment about the hold (mass x offset) is the physics: the load tips unless the hold supplies a moment, and the level-tilt gate makes that moment the objective's constraint."
    def parts(self, p):
        from engine import world as W
        return [O.payload_with_handle(name="payload", half=[0.16, 0.08, 0.06], mass=p["mass"], pos=[0.36, -0.2, 0.75 + 0.06], com_offset=[p["offset"], 0.0, 0.0])], [W.surface("table", pos=(0.45, -0.2, 0.375), size=(0.25, 0.3, 0.375))]
    def world(self, scenario): o, s = self.parts(scenario.params); return WorldSpec(objects=o, surfaces=s)
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(mass=float(rng.uniform(1.0, 5.0)), offset=float(rng.choice([-1, 1]))*float(rng.uniform(0.06, 0.14)), lift=0.25, tilt_max=np.radians(20.0), hold=0.4); return P.B.scenario(self, seed, p, ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="sequence", params=dict(subgoals=[dict(event="held", object="payload"), dict(event="object_displacement", object="payload", axis="z", min=p["lift"], held=True, tilt_max=p["tilt_max"])]), horizon=6.0)
    def strategy_space(self, scenario): return ["servo_knots", "servo_knots+brace_prior"]
    def demand(self, scenario):
        p = scenario.params; t = P.task(p.get("robot", "G1"), self.world(scenario))
        return P.plan(t, [P.reach_grasp(t, "payload", horizon=3.0), P.move_held(t, "payload", [0.0, 0.0, p["lift"]], hold=p["hold"], horizon=3.0)], events=("held", "lifted_level"))
    def evaluate(self, rec, scenario):
        out = super().evaluate(rec, scenario); fs = P.final_state(rec); tilt = None
        if fs is not None:
            import mujoco; t = self.demand(scenario).steps[0].task; m = t.plant(t.demand_default())["m"]; qa = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "payload_free")])
            w, x, y, z = np.asarray(fs[0])[qa + 3:qa + 7]; zz = 1 - 2*(x*x + y*y); tilt = float(np.arccos(np.clip(zz, -1, 1)))   # angle between the load's z axis and vertical
        out.measures.update(tilt=tilt, tilt_max=scenario.params["tilt_max"], moment_Nm=scenario.params["mass"]*9.81*abs(scenario.params["offset"]))
        if tilt is not None and tilt > scenario.params["tilt_max"]: out.validity.reasons.append("TILTED"); out.validity.object_interaction = False; out.validity.violation += 1.0 + 2.0*(tilt - scenario.params["tilt_max"]); out.achieved = False
        return out
