"""Program 71 (payload): REORIENT a held payload: rotate a box in the hand by a target angle about a world axis (yaw about z or roll
about x) and hold it there. Needs the 3-DOF wrist (G1H) and a hold: the weld (experimental simplification, Capability.WELD_GRASP; the
Dex3 wrap is declared for cylindrical handles only). The object starts IN THE HAND (labelled abstraction of tasks/reach.py: an object
declared held starts at the palm). Value = rotation achieved about the requested axis; error vs target reported.
integration check: 2026-10-12 — prior-only rollout, plant built, value 4.17e-17, achieved False, gates fired [DROPPED,ORIENTATION_MISSED]; rotation=4.17e-17 (30 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from . import _pay as P


@register
class PayloadReorientation(P.PayloadProgram, TaskProgram):
    id = "P71_payload_reorientation"; family = Family.PAYLOAD; title = "Rotate a held payload to a target orientation"
    requires = [Capability.PALM_SITE_RIGHT, Capability.WRIST_3DOF, Capability.WELD_GRASP, Capability.SERVO_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED
    distinct_from = "Lifts move the payload's position; here its position is free and its ORIENTATION is the effect, which only the wrist and shoulder rotations can produce while the hold carries the gravity moment."
    def world(self, scenario): p = scenario.params; return WorldSpec(objects=[O.box(name="payload", half=[0.06, 0.04, 0.03], mass=p["mass"], pos=[0.35, -0.25, 0.9])])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(robot="G1H", mass=float(rng.uniform(0.2, 1.5)), axis=str(rng.choice(["z", "x"])), angle=float(np.radians(rng.uniform(30, 90))), tol=np.radians(10.0), hold=0.5, lift=-1.0); return P.B.scenario(self, seed, p, ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="pose", params=dict(object="payload", axis=p["axis"], angle=p["angle"], held=True), tolerance=dict(angle=p["tol"]), horizon=3.0)
    def strategy_space(self, scenario): return ["servo_knots"]
    def _target(self, p):
        a = p["angle"]
        return ([1, 0, 0], [float(np.cos(a)), float(np.sin(a)), 0.0]) if p["axis"] == "z" else ([0, 1, 0], [0.0, float(np.cos(a)), float(np.sin(a))])
    def demand(self, scenario):
        p = scenario.params; t = P.task(p.get("robot", "G1H"), self.world(scenario)); ax, tgt = self._target(p)
        d = dict(t.demand_default(), target=[0.0, 0.0, 0.0], tol=0.08, hold=p["hold"], held=["payload"], start_in_hand=["payload"], omega_servo=120.0, trunk_cap=0.05, horizon=3.0, orient=[dict(body="payload", axis=ax, target=tgt, k=6.0, max=1.5)])
        return t, d
    def evaluate(self, rec, scenario):
        p = scenario.params; e = P.final_extra(rec); fs = e.get("final_state"); rot = 0.0
        if fs is not None:
            import mujoco; t = self.demand(scenario).steps[0].task; m = t.plant(t.demand_default())["m"]; qa = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "payload_free")])
            q = np.asarray(fs[0])[qa + 3:qa + 7]; R = np.zeros(9); mujoco.mju_quat2Mat(R, q); R = R.reshape(3, 3); ax, tgt = self._target(p); v = R @ np.asarray(ax, float)
            rot = float(np.arctan2(v[1], v[0])) if p["axis"] == "z" else float(np.arctan2(v[2], v[1]))
        err = abs(rot - p["angle"]); extra = [] if P.held(rec, "payload") else ["DROPPED"]
        if err > p["tol"]: extra.append("ORIENTATION_MISSED")
        return P.B.outcome(rec, value=rot, measures=P.B.reach_measures(rec) | dict(rotation=rot, target=p["angle"], error=err, ori_err_max=e.get("ori_err_max")), extra_reasons=extra, extra_violation=(1.0 + 2.0*max(0.0, err - p["tol"])), achieved=err <= p["tol"]) if extra else P.B.outcome(rec, value=rot, measures=P.B.reach_measures(rec) | dict(rotation=rot, target=p["angle"], error=err, ori_err_max=e.get("ori_err_max")), achieved=True)
