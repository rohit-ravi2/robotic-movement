"""Program 87 (negotiation): changing-friction terrain. Travel to x_goal across a floor whose friction alternates between patches
(world.disturbances.slip_patch -> static friction patches; patch layout seeded), ending upright at rest. Outcome = -shortfall to x_goal.
integration check: 2026-10-12 world builds; demand = contact_schedule+knots over BaseEffectTask; not run (family representative: P83)."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import NegotiationProgram, terrain, q_trace, pelvis_kin, past, base_validity, no_trace, fail, travel_goal


def _slip(xy, mu, size):
    try: from movement_discovery.world.disturbances import slip_patch; return slip_patch(xy, mu, size)
    except Exception: return dict(kind="slip_patch", t0=0.0, t1=None, vec=[0.0, 0.0, 0.0], body="floor", xy=[float(xy[0]), float(xy[1])], mu=float(mu), size=float(size))


@register
class ChangingFriction(NegotiationProgram):
    id = "P87_changing_friction"; family = Family.NEGOTIATION; title = "Travel across alternating low- and high-friction patches"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "The support geometry is flat (unlike P84) but the tangential force budget changes from step to step; P19 is a single standing slip, not travel."
    def world(self, scenario):
        p = scenario.params; return WorldSpec(terrain=terrain("flat"), disturbances=[_slip((x, 0.0), mu, p["patch"]) for x, mu in zip(p["xs"], p["mus"])], friction=p["mu_base"], tag=f"patches{len(p['xs'])}")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); n = int(rng.integers(2, 5)); patch = float(rng.uniform(0.3, 0.5)); xs = [0.5 + i*2*patch for i in range(n)]
        return self.scenario(seed, dict(robot="G1", xs=xs, mus=[float(rng.uniform(0.05, 0.25)) for _ in xs], mu_base=0.9, patch=patch, x_goal=xs[-1] + patch + 0.3, horizon=[2.0, 10.0]), **ov)
    def goal(self, scenario): return travel_goal(scenario.params, GoalSpec)
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; ok, m = past(q, dt, p["x_goal"]); val = base_validity(rec)
        if not ok: fail(val, "NOT_PAST_OBSTACLE", m["short"])
        return Outcome(value=-m["short"], achieved=(ok and val.ok), measures=dict(mus=p["mus"], **m), validity=val)
