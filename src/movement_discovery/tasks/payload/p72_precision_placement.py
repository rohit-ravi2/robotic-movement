"""Program 72 (payload): PRECISION placement. Pick a small object from a table and set it down on a marked target with a tight
tolerance, released and resting (engine.world.effect_place: the object settles under free physics after release, the robot frozen).
Value = -placement error (m); achieved iff error <= tol and on the surface. The three external sub-goals (held, lifted clear,
resting at target) follow engine/demo.run_place. Hold = G1 weld (experimental simplification).
integration check: 2026-10-12 — prior-only rollout, plant built, value -0.0497, achieved False, gates fired [NOT_PLACED]; placement_error=0.0497 (34 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from . import _pay as P


@register
class PrecisionPlacement(P.PayloadProgram, TaskProgram):
    id = "P72_precision_placement"; family = Family.PAYLOAD; title = "Place an object on a target within centimetres"
    requires = [Capability.PALM_SITE_RIGHT, Capability.WELD_GRASP, Capability.SERVO_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED; releases = True
    distinct_from = "P66 transfers height with a loose landing on a shelf; here the height is unchanged and the effect is the FINAL RESTING error after release and settling, so the release dynamics and the approach speed are the physics."
    def parts(self, p):
        from engine import world as W
        return [O.box(name="cube", half=[0.03, 0.03, 0.03], mass=p["mass"], pos=[0.36, -0.22, 0.75 + 0.03])], [W.surface("table", pos=(0.45, -0.05, 0.375), size=(0.25, 0.4, 0.375))]
    def world(self, scenario): o, s = self.parts(scenario.params); return WorldSpec(objects=o, surfaces=s)
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(mass=float(rng.uniform(0.1, 1.0)), target=[float(rng.uniform(0.32, 0.42)), float(rng.uniform(0.02, 0.15)), 0.75 + 0.03], tol=float(rng.choice([0.01, 0.02, 0.03])), lift=0.10); return P.B.scenario(self, seed, p, ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="object_displacement", params=dict(object="cube", target=p["target"], released=True, surface="table"), tolerance=dict(xy=p["tol"], z=2*p["tol"]), horizon=9.0)
    def strategy_space(self, scenario): return ["servo_knots"]
    def demand(self, scenario):
        p = scenario.params; t = P.task(p.get("robot", "G1"), self.world(scenario)); tgt = list(p["target"])
        d1 = dict(t.demand_default(), target_object="cube", tol=0.03, hold=0.15, grasp_on_reach="cube", omega_servo=120.0, trunk_cap=0.05, horizon=3.0, refine=1, aim_z=True)
        d2 = dict(t.demand_default(), target_object="cube", offset=[0.0, 0.0, p["lift"]], tol=0.05, hold=0.15, held=["cube"], omega_servo=120.0, trunk_cap=0.05, horizon=2.0)
        d3 = dict(t.demand_default(), target_world=[tgt[0], tgt[1], tgt[2] + 0.01], tol=0.03, hold=0.3, held=["cube"], release_at_end=True, omega_servo=120.0, trunk_cap=0.05, horizon=3.0)
        return P.plan(t, [d1, d2, d3], events=("held", "lifted", "object_at_rest"))
    def evaluate(self, rec, scenario):
        p = scenario.params; e = P.final_extra(rec); fs = P.final_state(rec); eff = None
        if fs is not None:
            import mujoco; from engine import world as W; t = self.demand(scenario).steps[0].task; S = t.plant(t.demand_default()); m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = fs[0]; mujoco.mj_forward(m, d)
            eff = W.effect_place(t.world(S), None, d, "cube", p["target"], tol=p["tol"], surface="table")
        err = float(eff["dxy"]) if eff else 1.0; extra = [] if (eff and eff["ok"]) else ["NOT_PLACED"]
        m_ = dict(placement_error=err, effect=eff, tol=p["tol"])
        if "steps" in rec: return P.B.seq_outcome(rec, value=-err, measures=m_, extra_reasons=extra, achieved=not extra)
        return P.B.outcome(rec, value=-err, measures=m_, extra_reasons=extra, achieved=not extra)
