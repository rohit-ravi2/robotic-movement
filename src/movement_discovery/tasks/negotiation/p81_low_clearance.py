"""Program 81 (negotiation): low-clearance passage. An overhead slab (world.obstacles.low_clearance: underside at `height`, spanning
`depth` along x) blocks the standing height; get past it to x_goal and stand again. Outcome = -shortfall to x_goal (0 when past), no
contact with the slab (NONFOOT_CONTACT is a violation). Ducking, crawling or anything else is the construction's.
integration check: 2026-10-12 world builds; demand = contact_schedule+knots over BaseEffectTask; not run (family representative: P83)."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import NegotiationProgram, terrain, obstacle, q_trace, pelvis_kin, past, base_validity, no_trace, fail, travel_goal


@register
class LowClearance(NegotiationProgram):
    id = "P81_low_clearance"; family = Family.NEGOTIATION; title = "Pass under an overhead slab lower than standing height"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "The constraint is vertical (head/torso height) over a stretch of travel; P82 constrains lateral width, P83 forces foot clearance."
    def world(self, scenario): p = scenario.params; return WorldSpec(terrain=terrain("flat"), mechanisms=[obstacle("low_clearance", height=p["height"], depth=p["depth"], width=1.5, x=p["x_obs"])], tag=f"lowbar{int(p['height']*100)}")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); depth = float(rng.uniform(0.3, 1.0)); return self.scenario(seed, dict(robot="G1", height=float(rng.uniform(0.75, 1.15)), depth=depth, x_obs=0.6, x_goal=0.6 + depth + 0.5, horizon=[2.0, 8.0]), **ov)
    def goal(self, scenario): return travel_goal(scenario.params, GoalSpec)
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; ok, m = past(q, dt, p["x_goal"]); val = base_validity(rec); xyz = pelvis_kin(q, dt)[0]
        under = (xyz[:, 0] > p["x_obs"]) & (xyz[:, 0] < p["x_obs"] + p["depth"]); z_under = float(xyz[under, 2].max()) if under.any() else None
        if not ok: fail(val, "NOT_PAST_OBSTACLE", m["short"] + (0.0 if m["tilt_end_deg"] < 30 else 1.0))
        return Outcome(value=-m["short"], achieved=(ok and val.ok), measures=dict(pelvis_z_under=z_under, slab_height=p["height"], **m), validity=val)
