"""Program 85 (negotiation): auxiliary support surface use. A waist-high rail (static box, engine.world.surface convention) runs beside
the path while a lateral impulse hits the torso; the hands MAY load the rail (declared body contact: NONFOOT_CONTACT with the rail is not a
violation; with the floor it is). Outcome = -settling time after the impulse, upright at the end. Whether the rail is used is the
construction's. Needs hand-rail contact: EFFECTOR_COLLIDER (the G1 palms have no collision with world pieces other than through the
replicated floor pairs: hand_collision IS in the base scene's floor pairs, so world pieces do carry the hands).
integration check: not executable until a representation applies disturbances (physics.disturbance.wrap) on a world plant."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import NegotiationProgram, terrain, impulse, q_trace, pelvis_kin, upright, base_validity, no_trace, fail, schedule_demand, phase
from ..balance._local import settle_time


@register
class AuxiliarySupport(NegotiationProgram):
    id = "P85_auxiliary_support"; family = Family.NEGOTIATION; title = "Recover from a push with a rail within reach"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.PALM_SITE_RIGHT, Capability.PALM_SITE_LEFT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P17 has only the feet; here a third support (the rail, at hand height, on one side) exists and the push is sized beyond the feet-only capture limit, so using it changes what is recoverable."
    def world(self, scenario):
        p = scenario.params; return WorldSpec(terrain=terrain("flat"), surfaces=[dict(name="rail", pos=[0.0, p["side"]*p["rail_y"], p["rail_h"]], size=[0.8, 0.02, 0.02], rgba=[0.5, 0.5, 0.55, 1], rigid=True), dict(name="rail_post", pos=[0.0, p["side"]*p["rail_y"], p["rail_h"]/2], size=[0.02, 0.02, p["rail_h"]/2], rgba=[0.5, 0.5, 0.55, 1], rigid=True)], disturbances=[impulse(p["t0"], p["Jvec"])], tag="rail")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); J = float(rng.uniform(15.0, 35.0)); side = int(rng.choice([-1, 1]))
        return self.scenario(seed, dict(robot="G1", side=side, rail_y=float(rng.uniform(0.35, 0.55)), rail_h=float(rng.uniform(0.85, 1.05)), J=J, Jvec=[0.0, side*J, 0.0], t0=0.5, T=3.0, t_hold=0.5, horizon=[3.0, 3.0]), **ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="stability", params=dict(disturbance="impulse", J=p["J"], auxiliary_support="rail", objective="min_settling_time"), tolerance=dict(tilt=np.radians(30), t_hold=p["t_hold"]), horizon=p["T"])
    def strategy_space(self, scenario): return ["servo_knots", "torque_knots"]
    def demand(self, scenario):
        p = scenario.params; g = self.goal(scenario); return schedule_demand(scenario, p["horizon"], [phase("any", g, hands_allowed_on="rail")], g, allowed=("both",), notes="hands may load the rail; feet stay the primary support")
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; up, m = upright(q, dt, T=p["t_hold"]); ts = settle_time(q, dt, p["t0"] + 0.05); val = base_validity(rec)
        rail_contact = bool((rec.get("extra") or {}).get("rail_contact", False)); val.reasons = [r for r in val.reasons if not (r == "NONFOOT_CONTACT" and rail_contact)]; val.contact = "NONFOOT_CONTACT" not in val.reasons
        if not up: fail(val, "NOT_UPRIGHT_AT_END")
        if ts is None: fail(val, "NOT_SETTLED")
        return Outcome(value=-(ts if ts is not None else p["T"]), achieved=(up and ts is not None and val.ok), measures=dict(t_settle=ts, rail_used=rail_contact, J=p["J"], **m), validity=val)
