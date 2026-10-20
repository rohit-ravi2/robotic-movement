"""Program 12 (aerial): lateral jump. Flight whose touchdown is displaced sideways (body y) by a commanded distance with little forward
drift, landing upright. Outcome = -|dy - d*|. Bridge: JumpTask (the vertical prior; the lateral component must emerge in the knots).
integration check: 2026-10-12 one prior-only JumpTask rollout (ascent prior, zero knots, tscale 1; shared by P09-P16): flight 0.151 m (takeoff 1.72 m/s), landing controller did NOT recover (up_end 0.16, NOT_LANDED), 8 s; dy 0.005 m, err 0.18 m, FORWARD_DRIFT, NOT_LANDED."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import AerialProgram, terrain, flight, landing_displacement, aerial_validity, no_trace, fail


@register
class LateralJump(AerialProgram):
    id = "P12_lateral_jump"; family = Family.AERIAL; title = "Sideways flight to a commanded lateral displacement"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE, Capability.HIP_ROLL, Capability.ANKLE_ROLL]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P11 is sagittal (hip/knee/ankle pitch plane); lateral flight loads the roll joints and lands with a laterally offset capture point."
    def world(self, scenario): return WorldSpec(terrain=terrain("flat"), tag="flat")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); return self.scenario(seed, dict(robot="G1", prior="ascent", dy=float(rng.choice([-1, 1])*rng.uniform(0.10, 0.40)), tol=0.05, dx_max=0.15), **ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="com_displacement", params=dict(axis="y", target=p["dy"], dx_max=p["dx_max"], landing="upright"), tolerance=dict(dy=p["tol"]), horizon=2.6)
    def evaluate(self, rec, scenario):
        if not rec: return no_trace()
        p = scenario.params; f = flight(rec); val = aerial_validity(rec, f); d, _ = landing_displacement(f)
        if d is None: fail(val, "NO_LANDING_TRACE"); return Outcome(value=-abs(p["dy"]), achieved=False, validity=val)
        err = abs(float(d[1]) - p["dy"])
        if abs(float(d[0])) > p["dx_max"]: fail(val, "FORWARD_DRIFT", abs(float(d[0])) - p["dx_max"])
        return Outcome(value=-err, achieved=(err <= p["tol"] and val.ok), measures=dict(dy=float(d[1]), dx=float(d[0]), err=err, height=f["height"], landed=f["landed"]), validity=val)
