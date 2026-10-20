"""Program 19 (balance): slip recovery. The floor under the feet is low-friction (world friction override) and a horizontal impulse pushes
the robot; the feet may slide. Outcome = -settling time; validity = upright at the end. The friction cone, not the torque budget, is the
limiting physics. Bridge: stand bridge with the floor friction overridden. integration check: 2026-10-12 one StandTask rollout (mu 0.058, J 5 N s): settled 1.97 s after the push, slide 9 mm, achieved True, 3.2 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import BalanceProgram, terrain, impulse, q_trace, pelvis_kin, upright, settle_time, base_validity, no_trace, fail


@register
class SlipRecovery(BalanceProgram):
    id = "P19_slip_recovery"; family = Family.BALANCE; title = "Recover from a push on a low-friction floor"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE, Capability.ANKLE_ROLL]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "Same disturbance class as P17 but the tangential ground force is capped by mu*N: ankle strategies saturate and the feet move."
    def world(self, scenario): p = scenario.params; return WorldSpec(terrain=terrain("flat"), friction=p["mu"], disturbances=[impulse(p["t0"], p["Jvec"])], tag=f"mu{int(p['mu']*100)}")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); J = float(rng.uniform(3.0, 15.0)); a = float(rng.uniform(-np.pi, np.pi))
        return self.scenario(seed, dict(robot="G1", mu=float(rng.uniform(0.05, 0.25)), J=J, dir=a, Jvec=[J*np.cos(a), J*np.sin(a), 0.0], t0=0.5, T=3.0, t_hold=0.5), **ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="stability", params=dict(disturbance="impulse", J=p["J"], mu=p["mu"], objective="min_settling_time"), tolerance=dict(tilt=np.radians(30), t_hold=p["t_hold"]), horizon=p["T"])
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; xyz = pelvis_kin(q, dt)[0]; up, m = upright(q, dt, T=p["t_hold"]); ts = settle_time(q, dt, p["t0"] + 0.05); val = base_validity(rec)
        if not up: fail(val, "NOT_UPRIGHT_AT_END")
        if ts is None: fail(val, "NOT_SETTLED")
        return Outcome(value=-(ts if ts is not None else p["T"]), achieved=(up and ts is not None and val.ok), measures=dict(t_settle=ts, slide_xy=float(np.linalg.norm(xyz[-1, :2] - xyz[0, :2])), mu=p["mu"], **m), validity=val)
