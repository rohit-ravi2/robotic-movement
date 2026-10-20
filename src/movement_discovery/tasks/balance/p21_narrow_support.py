"""Program 21 (balance): narrow-support balancing. Stand on a beam of width w (world terrain 'narrow_beam' along x under the start; the floor beside it is 15 cm lower)
for T s while a lateral impulse arrives; both foot sites must stay inside the strip |y| <= w/2 - foot half-width and the robot upright.
Outcome = -settling time; the stance on the beam is the construction's to find. Bridge: stand bridge on the flat floor (strip judged from
the foot sites; the normal stance fails the strip, which is the point). integration check: 2026-10-12 one StandTask rollout on the FLAT floor (beam terrain under the start needs a terrain-aware settle: refused loudly by StandTask.plant): feet y 0.12 m vs strip half-width 0.03 -> FOOT_OFF_BEAM, as expected for the normal stance; 2.0 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import BalanceProgram, terrain, impulse, q_trace, upright, settle_time, foot_sites_xy, base_validity, no_trace, fail


@register
class NarrowSupport(BalanceProgram):
    id = "P21_narrow_support"; family = Family.BALANCE; title = "Balance on a narrow beam under a lateral impulse"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE, Capability.HIP_ROLL, Capability.ANKLE_ROLL]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P17 has the full support polygon; here the lateral support width is below the foot separation, so lateral capture must come from the hips/arms or a tandem stance."
    def world(self, scenario): p = scenario.params; return WorldSpec(terrain=terrain("narrow_beam", width=p["w"], length=2.0, height=0.15, x0=-1.0), disturbances=[impulse(p["t0"], p["Jvec"])], tag=f"beam{int(p['w']*100)}")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); J = float(rng.uniform(1.0, 6.0)); s = float(rng.choice([-1, 1]))
        return self.scenario(seed, dict(robot="G1", w=float(rng.uniform(0.06, 0.16)), foot_half_width=0.03, J=J, Jvec=[0.0, s*J, 0.0], t0=1.0, T=3.0, t_hold=0.5), **ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="stability", params=dict(support_strip=p["w"], disturbance="impulse", J=p["J"], objective="min_settling_time"), tolerance=dict(tilt=np.radians(30), t_hold=p["t_hold"]), horizon=p["T"])
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; up, m = upright(q, dt, T=p["t_hold"]); ts = settle_time(q, dt, p["t0"] + 0.05); val = base_validity(rec); S = rec.get("plant")
        if S is None: t, d = self.demand(scenario); S = t.plant(d)                                             # the model for foot-site kinematics
        fy = foot_sites_xy(S, q)[:, :, 1]; out = float(np.max(np.abs(fy)) + p["foot_half_width"] - p["w"]/2)
        if out > 0: fail(val, "FOOT_OFF_BEAM", out); val.support = False
        if not up: fail(val, "NOT_UPRIGHT_AT_END")
        if ts is None: fail(val, "NOT_SETTLED")
        return Outcome(value=-(ts if ts is not None else p["T"]), achieved=(up and ts is not None and val.ok), measures=dict(t_settle=ts, foot_y_max=float(np.max(np.abs(fy))), strip_excess=out, **m), validity=val)
