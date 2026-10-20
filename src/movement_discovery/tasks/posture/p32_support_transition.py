"""Program 32 (posture): transition between support configurations. From standing on both feet, stand on ONE named foot (the other
lifted by at least `clearance`) for t_hold, then return to both feet at rest, within the horizon. Outcome = -total transition time. The
support sequence IS the task statement here (both -> one -> both); how the weight is shifted is not. The contact_schedule+knots
representation takes these phases natively. integration check: 2026-10-12 plant build: canonical stance on the world, 0.0 s; no rollout (contact_schedule+knots needs a base-level effect mode)."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import PostureProgram, terrain, q_trace, pelvis_kin, upright, base_validity, no_trace, fail, schedule_demand, phase, CS_NAME, foot_loads


@register
class SupportTransition(PostureProgram):
    id = "P32_support_transition"; family = Family.POSTURE; title = "Both feet -> one foot (held) -> both feet"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE, Capability.HIP_ROLL, Capability.ANKLE_ROLL]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P21/P24 are reactive (external narrowing/loss of support); here the support change is voluntary and must be reversed, with no external disturbance."
    def world(self, scenario): return WorldSpec(terrain=terrain("flat"), tag="flat")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); return self.scenario(seed, dict(robot="G1", start=None, foot=str(rng.choice(["left", "right"])), t_hold=float(rng.uniform(0.5, 1.5)), clearance=0.04, t_rest=0.3, horizon=[2.0, 4.0]), **ov)
    def start_posture(self, scenario): return None
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="stability", params=dict(sequence=["both", p["foot"], "both"], t_hold=p["t_hold"], clearance=p["clearance"], terminal="rest"), tolerance=dict(tilt=np.radians(30)), horizon=p["horizon"][1])
    def strategy_space(self, scenario): return [CS_NAME, "torque_knots"]
    def demand(self, scenario):
        p = scenario.params; g = self.goal(scenario)
        return schedule_demand(scenario, p["horizon"], [phase("both", g), phase(p["foot"], g, min_duration=p["t_hold"]), phase("both", g)], g, allowed=("both", p["foot"]), notes="phase order is the task statement; horizons free")
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec); L = foot_loads(rec)
        if q is None or L is None: return no_trace("NO_TRACE_OR_FOOT_LOADS")
        p = scenario.params; i_sw = 0 if p["foot"] == "right" else 1; sw = L[:, i_sw] < 1e-3; n = max(1, int(round(p["t_hold"]/dt))); runs = np.diff(np.concatenate([[0], sw.astype(int), [0]]))
        starts, ends = np.where(runs == 1)[0], np.where(runs == -1)[0]; held = [(s, e) for s, e in zip(starts, ends) if e - s >= n and e < len(sw) - int(p["t_rest"]/dt)]
        up, m = upright(q, dt, T=p["t_rest"]); val = base_validity(rec); val.reasons = [r for r in val.reasons if r != "STANCE_LOST"]; val.support = True   # a lifted foot is the task
        both_end = bool((L[-int(p["t_rest"]/dt):] > 0.05).all())
        if not held: fail(val, "ONE_FOOT_PHASE_NOT_HELD", p["t_hold"] - (float((ends - starts).max())*dt if len(starts) else 0.0))
        if not both_end: fail(val, "NOT_BACK_ON_BOTH_FEET")
        if not up: fail(val, "NOT_UPRIGHT_AT_END")
        t_tot = float(held[0][1]*dt) if held else len(q)*dt + 1.0
        return Outcome(value=-t_tot, achieved=(bool(held) and both_end and up and val.ok), measures=dict(one_foot_s=(float((held[0][1] - held[0][0])*dt) if held else 0.0), t_total=t_tot, **m), validity=val)
