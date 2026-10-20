"""Program 09 (aerial): maximum vertical displacement. Leave the ground and land again upright; outcome = COM rise from takeoff (v^2/2g).
Bridge: JumpTask (ascent prior + landing controller). integration check: 2026-10-12 one prior-only JumpTask rollout (ascent prior, zero knots, tscale 1; shared by P09-P16): flight 0.151 m (takeoff 1.72 m/s), landing controller did NOT recover (up_end 0.16, NOT_LANDED), 8 s; value 0.151 m, achieved False (NOT_LANDED)."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import AerialProgram, terrain, flight, aerial_validity, no_trace


@register
class MaxVertical(AerialProgram):
    id = "P09_max_vertical"; family = Family.AERIAL; title = "Maximum vertical flight with a standing landing"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P10 asks for a given apex (accuracy), P11/P12 for horizontal travel; this is the vertical ceiling with a landing."
    def world(self, scenario): return WorldSpec(terrain=terrain("flat"), tag="flat")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); return self.scenario(seed, dict(robot="G1", prior="ascent", h_min=0.02, land_T=0.5), **ov)
    def goal(self, scenario): return GoalSpec(kind="com_displacement", params=dict(axis="z", objective="max", landing="upright"), tolerance=dict(h_min=scenario.params["h_min"]), horizon=2.6)
    def evaluate(self, rec, scenario):
        if not rec: return no_trace()
        f = flight(rec); val = aerial_validity(rec, f); h = f["height"] or 0.0
        return Outcome(value=float(h), achieved=(h >= scenario.params["h_min"] and val.ok), measures=dict(height=h, landed=f["landed"], up_end=f["up_end"], z_end=f["z_end"], v_end=f["v_end"], rot=f["rot"]), validity=val)
