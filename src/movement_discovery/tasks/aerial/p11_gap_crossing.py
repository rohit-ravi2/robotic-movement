"""Program 11 (aerial): horizontal gap crossing. A gap of width w opens in the floor ahead (world terrain 'gap'); the robot must take off
before the near edge and land upright beyond the far edge. Outcome = pelvis horizontal displacement from takeoff to touchdown; achieved
iff the touchdown point clears the far edge. Bridge: JumpTask on the flat floor (the gap is judged ballistically from the pelvis trace until
world/terrain gap lands). integration check: 2026-10-12 round 2, runner.check through delegation (torque_knots+tracked_prior -> JumpTask, on the gap scene, start_z 0.30): flight, NOT_LANDED (prior alone), SHORT_OF_FAR_EDGE; 133 params; 14 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import AerialProgram, terrain, flight, landing_displacement, aerial_validity, no_trace, fail


@register
class GapCrossing(AerialProgram):
    id = "P11_gap_crossing"; family = Family.AERIAL; title = "Cross a floor gap in one flight"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P09/P10 are vertical; here the objective is forward range over a hole, the landing surface is at the same height, and a short flight falls in."
    def world(self, scenario): p = scenario.params; return WorldSpec(terrain=terrain("gap", width=p["width"], depth=0.3, x_edge=p["x_gap"]), tag=f"gap{int(p['width']*100)}")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); return self.scenario(seed, dict(robot="G1", prior="ascent", x_gap=0.35, width=float(rng.uniform(0.15, 0.60)), margin=0.05), **ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="com_displacement", params=dict(axis="x", min=p["x_gap"] + p["width"] + p["margin"], landing="upright"), tolerance=dict(margin=p["margin"]), horizon=2.6)
    def evaluate(self, rec, scenario):
        if not rec: return no_trace()
        p = scenario.params; f = flight(rec); val = aerial_validity(rec, f); d, dyaw = landing_displacement(f)
        dx = float(d[0]) if d is not None else 0.0; need = p["x_gap"] + p["width"] + p["margin"]
        if d is None: fail(val, "NO_LANDING_TRACE")
        elif dx < need: fail(val, "SHORT_OF_FAR_EDGE", need - dx)
        return Outcome(value=dx, achieved=(d is not None and dx >= need and val.ok), measures=dict(dx=dx, dy=(float(d[1]) if d is not None else None), height=f["height"], landed=f["landed"], need=need), validity=val)
