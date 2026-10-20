"""Program 41 (projection): maximise the release speed of a held point mass along a direction cone. WORKED EXAMPLE of wrapping the
reference construction (tasks/effect.py) as a TaskProgram: the program owns objective, world, scenario sampling, validity and
observability; the representation ("torque_knots+tracked_prior") and the gates come from the reference implementation.
integration check: 2026-10-12 — capacity scenario (bank row recipe=capacity, tau_scale arm x0.5) through runner.check: demand carries tau_scale, EffectTask.caps = 0.5 on the 5 right-arm types (1.0 elsewhere), prior-only rollout value 0.887, achieved False, gates fired [OUT_OF_CONE] (27 s); baseline seed-0 check not run."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, EmbodimentSpec, WorldSpec, GoalSpec, Scenario, Outcome, ValidityReport
from movement_discovery.core.registry import register


@register
class MaxReleaseSpeed(TaskProgram):
    id = "P41_max_release_speed"; family = Family.PROJECTION; title = "Maximum feasible release speed of a held mass"
    requires = [Capability.PALM_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P42 asks for a ballistic landing point (direction+speed coupled); this asks for speed inside a cone only."
    def embodiment(self, robot):
        import throw as T
        return EmbodimentSpec(robot=robot, mjcf=T.CFG[robot].get("mjcf") or "", driven=list(T.CFG[robot]["driven"]), effectors={"hand_r": T.CFG[robot]["hand_site"]}, capabilities=[Capability.PALM_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE])
    def world(self, scenario): return WorldSpec(tag="bare")                      # mass merged at the palm by the plant
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(mass=float(rng.choice([0.145, 0.3, 0.6])), elev=float(np.radians(rng.uniform(5, 40))), az=float(np.radians(rng.uniform(-30, 30))), elev_tol=np.radians(5.0), az_tol=np.radians(5.0), support="both"); p.update(ov)
        return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=p)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="effector_velocity", params=dict(effector="hand_r", direction=(p["elev"], p["az"]), mass=p["mass"]), tolerance=dict(elev=p["elev_tol"], az=p["az_tol"]))
    def demand(self, scenario):
        """Bridge to the reference construction: an EffectTask demand (the representation's native input)."""
        from tasks.effect import EffectTask; p = scenario.params; t = EffectTask(scenario.params.get("robot", "G1"))
        return t, dict(t.demand_default(), mass=p["mass"], support=p["support"], direction=(p["elev"], p["az"]), elev_tol=p["elev_tol"], az_tol=p["az_tol"])
    def evaluate(self, rec, scenario):
        v = ValidityReport(reasons=list(rec["reasons"]), violation=float(rec["violation"]), direction=("OUT_OF_CONE" not in rec["reasons"]), envelope=("OVERSPEED" not in rec["reasons"]), support=("STANCE_LOST" not in rec["reasons"]))
        return Outcome(value=float(rec["value"]), achieved=v.ok, measures={k: rec["extra"].get(k) for k in ("elev", "az", "t_rel", "work")}, validity=v, mechanics=None)
    def validity(self, rec, scenario): return self.evaluate(rec, scenario).validity
    def observe(self, rec, scenario):
        if not rec.get("trace"): return None
        from engine import emergence as EM; t, d = self.demand(scenario); return EM.analyze(t.plant(d), rec, d["effector"])
