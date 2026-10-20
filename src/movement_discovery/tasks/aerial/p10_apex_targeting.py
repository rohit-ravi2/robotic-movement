"""Program 10 (aerial): apex-height targeting. Flight whose COM rise matches a commanded height within tolerance, landing upright;
outcome = -|h - h*|. Bridge: JumpTask. integration check: 2026-10-12 one prior-only JumpTask rollout (ascent prior, zero knots, tscale 1; shared by P09-P16): flight 0.151 m (takeoff 1.72 m/s), landing controller did NOT recover (up_end 0.16, NOT_LANDED), 8 s; err 0.009 m to the sampled target, achieved False (NOT_LANDED)."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import AerialProgram, terrain, flight, aerial_validity, no_trace


@register
class ApexTargeting(AerialProgram):
    id = "P10_apex_targeting"; family = Family.AERIAL; title = "Flight to a commanded apex height"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P09 maximises; this penalises both under- and overshoot of a sub-maximal apex (a graded, not a ceiling, problem)."
    def world(self, scenario): return WorldSpec(terrain=terrain("flat"), tag="flat")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); return self.scenario(seed, dict(robot="G1", prior="ascent", h_target=float(rng.uniform(0.04, 0.20)), tol=0.015), **ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="com_displacement", params=dict(axis="z", target=p["h_target"], landing="upright"), tolerance=dict(h=p["tol"]), horizon=2.6)
    def evaluate(self, rec, scenario):
        if not rec: return no_trace()
        p = scenario.params; f = flight(rec); val = aerial_validity(rec, f); h = f["height"] or 0.0; err = abs(h - p["h_target"])
        return Outcome(value=-float(err), achieved=(err <= p["tol"] and val.ok), measures=dict(height=h, err=err, landed=f["landed"], rot=f["rot"]), validity=val)
