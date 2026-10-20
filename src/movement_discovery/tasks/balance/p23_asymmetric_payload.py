"""Program 23 (balance): asymmetric payload balance. A payload mass is merged at the right palm (one-sided load that shifts the COM
laterally and forward); the robot must stand upright for T s and absorb a small impulse. Outcome = -settling time; measures include the
lateral drift. Bridge: stand bridge with mass_at=('right_palm', m). integration check: 2026-10-12 one StandTask rollout (1.68 kg at the right palm, J 2-10 N s): settled 0.66 s, drift_y -3 mm, achieved True, 3.9 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import BalanceProgram, terrain, impulse, q_trace, pelvis_kin, upright, settle_time, base_validity, no_trace, fail


@register
class AsymmetricPayload(BalanceProgram):
    id = "P23_asymmetric_payload"; family = Family.BALANCE; title = "Stand and absorb a push while holding a one-sided payload"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.PALM_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE, Capability.HIP_ROLL]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "The equilibrium posture itself is asymmetric (COM offset by the load) before any disturbance; P17's stance is symmetric."
    def world(self, scenario): p = scenario.params; return WorldSpec(terrain=terrain("flat"), disturbances=[impulse(p["t0"], p["Jvec"])], tag="flat")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); J = float(rng.uniform(2.0, 10.0)); a = float(rng.uniform(-np.pi, np.pi))
        return self.scenario(seed, dict(robot="G1", mass=float(rng.uniform(1.5, 6.0)), site="right_palm", J=J, Jvec=[J*np.cos(a), J*np.sin(a), 0.0], t0=1.0, T=3.0, t_hold=0.5), **ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="stability", params=dict(payload=dict(site=p["site"], mass=p["mass"]), disturbance="impulse", J=p["J"], objective="min_settling_time"), tolerance=dict(tilt=np.radians(30), t_hold=p["t_hold"]), horizon=p["T"])
    def bridge_kwargs(self, scenario): return dict(mass_at=(scenario.params["site"], scenario.params["mass"]))
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; xyz = pelvis_kin(q, dt)[0]; up, m = upright(q, dt, T=p["t_hold"]); ts = settle_time(q, dt, p["t0"] + 0.05); val = base_validity(rec)
        if not up: fail(val, "NOT_UPRIGHT_AT_END")
        if ts is None: fail(val, "NOT_SETTLED")
        return Outcome(value=-(ts if ts is not None else p["T"]), achieved=(up and ts is not None and val.ok), measures=dict(t_settle=ts, mass=p["mass"], drift_y=float(xyz[-1, 1] - xyz[0, 1]), **m), validity=val)
