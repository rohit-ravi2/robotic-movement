"""Program 14 (aerial): targeted landing. Land upright with the base over a commanded ground point (dx, dy) after one flight; outcome =
-planar landing error. Bridge: JumpTask. integration check: 2026-10-12 round 2, runner.check through delegation (JumpTask): landing err 0.32 m, NOT_LANDED (prior alone); 9 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import AerialProgram, terrain, flight, landing_displacement, aerial_validity, no_trace, fail


@register
class TargetedLanding(AerialProgram):
    id = "P14_targeted_landing"; family = Family.AERIAL; title = "Land on a commanded ground point"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE, Capability.HIP_ROLL]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P11/P12 ask for a minimum or a 1-D displacement; this is a 2-D accuracy problem at a sub-maximal range (both axes penalised, overshoot included)."
    def world(self, scenario): return WorldSpec(terrain=terrain("flat"), tag="flat")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); r = float(rng.uniform(0.10, 0.45)); a = float(rng.uniform(-np.pi/2, np.pi/2))
        return self.scenario(seed, dict(robot="G1", prior="ascent", target=[r*np.cos(a), r*np.sin(a)], tol=0.05), **ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="com_displacement", params=dict(target_xy=p["target"], landing="upright"), tolerance=dict(pos=p["tol"]), horizon=2.6)
    def evaluate(self, rec, scenario):
        if not rec: return no_trace()
        p = scenario.params; f = flight(rec); val = aerial_validity(rec, f); d, _ = landing_displacement(f)
        if d is None: fail(val, "NO_LANDING_TRACE"); return Outcome(value=-float(np.linalg.norm(p["target"])), achieved=False, validity=val)
        err = float(np.linalg.norm(d - np.asarray(p["target"])))
        return Outcome(value=-err, achieved=(err <= p["tol"] and val.ok), measures=dict(landing=d.tolist(), err=err, height=f["height"], landed=f["landed"]), validity=val)
