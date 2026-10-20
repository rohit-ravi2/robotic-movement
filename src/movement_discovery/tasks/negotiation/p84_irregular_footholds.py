"""Program 84 (negotiation): irregular footholds. A field of random raised blocks (world terrain 'uneven': n, amp, size, seeded) between
the start and x_goal; travel across and stand at rest. Outcome = -shortfall to x_goal. No foot placement is given; the terrain seed is
part of the scenario. integration check: 2026-10-12 world builds; demand = contact_schedule+knots over BaseEffectTask; not run (family representative: P83)."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import NegotiationProgram, terrain, q_trace, pelvis_kin, past, base_validity, no_trace, fail, travel_goal


@register
class IrregularFootholds(NegotiationProgram):
    id = "P84_irregular_footholds"; family = Family.NEGOTIATION; title = "Cross a field of uneven blocks"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE, Capability.ANKLE_ROLL]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "Support height and foot tilt vary at every contact (ankle roll/pitch compliance), with no single clearance event (P83) and no regular rise (P08)."
    def world(self, scenario): p = scenario.params; return WorldSpec(terrain=terrain("uneven", seed=p["terrain_seed"], n=p["n"], amp=p["amp"], extent=p["extent"], size=p["size"], clear_radius=0.3), tag=f"uneven{p['terrain_seed']}")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); return self.scenario(seed, dict(robot="G1", terrain_seed=int(seed), n=int(rng.integers(25, 60)), amp=float(rng.uniform(0.03, 0.10)), size=float(rng.uniform(0.12, 0.30)), extent=2.0, x_goal=float(rng.uniform(1.0, 1.8)), horizon=[2.0, 8.0]), **ov)
    def goal(self, scenario): return travel_goal(scenario.params, GoalSpec)
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; ok, m = past(q, dt, p["x_goal"]); val = base_validity(rec); xyz = pelvis_kin(q, dt)[0]
        if not ok: fail(val, "NOT_PAST_OBSTACLE", m["short"])
        return Outcome(value=-m["short"], achieved=(ok and val.ok), measures=dict(z_range=[float(xyz[:, 2].min()), float(xyz[:, 2].max())], amp=p["amp"], **m), validity=val)
