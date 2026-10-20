"""Program 48 (projection): maximum release speed in a cone from a CONSTRAINED INITIAL POSTURE: the plant starts from a named
keyframe of the robot model (G1 'knees_bent' crouch; the scenario names it) instead of the standing stance; the body-derived
reference and the search both start there (_local.StartPoseEffectTask). A seated/kneeling start is NOT offered on the MJX G1: its
body geoms carry no collision, so a knee or pelvis cannot rest on anything (would need Capability.BODY_COLLISION).
integration check: 2026-10-12 prior-only rollout from the 'knees_bent' keyframe (seed 0): plant built (start knee 0.67 rad), reference built from the crouch, 2.2 m/s OUT_OF_CONE; 14 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec, Scenario, Outcome
from movement_discovery.core.registry import register
from movement_discovery.embodiment import resolve
from . import _local as L


@register
class ConstrainedStartProjection(TaskProgram):
    id = "P48_constrained_start_projection"; family = Family.PROJECTION; title = "Release speed in a cone from a constrained start posture"
    requires = [Capability.PALM_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P41 starts from the nominal stance; here the initial posture (a crouch) removes the usual leg-drive geometry, so the wind-up must be built from a different start state."
    def embodiment(self, robot): return resolve(robot, self)
    def world(self, scenario): return WorldSpec(tag="bare")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(mass=float(rng.choice([0.145, 0.3])), start_key="knees_bent", elev=float(np.radians(rng.uniform(10, 35))), az=float(np.radians(rng.uniform(-20, 20))), elev_tol=np.radians(8.0), az_tol=np.radians(8.0), support="both"); p.update(ov)
        return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=p)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="effector_velocity", params=dict(effector="hand_r", direction=(p["elev"], p["az"]), mass=p["mass"], start=p["start_key"]), tolerance=dict(elev=p["elev_tol"], az=p["az_tol"]))
    def demand(self, scenario):
        p = scenario.params; return L.bridge(scenario, mass=p["mass"], support=p["support"], direction=(p["elev"], p["az"]), elev_tol=p["elev_tol"], az_tol=p["az_tol"], start_key=p["start_key"])
    def evaluate(self, rec, scenario):
        t, d = self.demand(scenario); S = t.plant(d); i = S["types"].index("right_knee_joint") if "right_knee_joint" in S["types"] else None
        kn = float(np.asarray(S["q_settled"])[S["jarr"]["qad"][i]]) if i is not None else None
        return L.outcome(rec["value"], rec, dict(elev=rec["extra"]["elev"], az=rec["extra"]["az"], t_rel=rec["extra"]["t_rel"], start_knee=kn, h_rel=float(L.release_state(S, rec)["pos"][2])))
    def validity(self, rec, scenario): return self.evaluate(rec, scenario).validity
    def observe(self, rec, scenario):
        if not rec.get("trace"): return None
        from engine import emergence as EM; t, d = self.demand(scenario); return EM.analyze(t.plant(d), rec, d["effector"])
