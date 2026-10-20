"""Program 17 (balance): recovery from a sudden impulse. A horizontal impulse (impulse in N s on the torso, random horizontal direction) hits the
standing robot; it must be upright and at rest again. Outcome = -settling time after the impulse. Bridge: stand bridge (capture-point
controller, zero knots). integration check: 2026-10-12 round 2, runner.check (servo_knots -> StandTask local stand loop, wrench via physics.hooks, 1 step of 20.9 N s): fell (tilt 155 deg), NOT_SETTLED; hooks delivered; 10 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import BalanceProgram, terrain, impulse, q_trace, upright, settle_time, base_validity, no_trace, fail


@register
class ImpulseRecovery(BalanceProgram):
    id = "P17_impulse_recovery"; family = Family.BALANCE; title = "Stay upright after a horizontal impulse on the torso"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE, Capability.ANKLE_ROLL]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P18 applies a sustained force (a quasi-static lean, no momentum step); here the whole disturbance is momentum delivered in 0.1 s."
    def world(self, scenario): p = scenario.params; return WorldSpec(terrain=terrain("flat"), disturbances=[impulse(p["t0"], p["Jvec"])], tag="flat")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); J = float(rng.uniform(5.0, 30.0)); a = float(rng.uniform(-np.pi, np.pi))
        return self.scenario(seed, dict(robot="G1", J=J, dir=a, Jvec=[J*np.cos(a), J*np.sin(a), 0.0], t0=0.5, T=3.0, t_hold=0.5), **ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="stability", params=dict(disturbance="impulse", J=p["J"], dir=p["dir"], objective="min_settling_time"), tolerance=dict(tilt=np.radians(30), t_hold=p["t_hold"]), horizon=p["T"])
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; up, m = upright(q, dt, T=p["t_hold"]); ts = settle_time(q, dt, p["t0"] + 0.05); val = base_validity(rec)
        if not up: fail(val, "NOT_UPRIGHT_AT_END")
        if ts is None: fail(val, "NOT_SETTLED")
        return Outcome(value=-(ts if ts is not None else p["T"]), achieved=(up and ts is not None and val.ok), measures=dict(t_settle=ts, J=p["J"], **m), validity=val)
