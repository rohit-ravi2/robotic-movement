"""Program 18 (balance): resistance to sustained loading. A constant horizontal force acts on the torso for a window and is then released;
the robot must stay upright throughout and after. Outcome = -peak pelvis tilt during the loading (how little the posture yields).
Bridge: stand bridge. integration check: 2026-10-12 round 2, runner.check (StandTask + hooks: 1500 wrench steps of the 0.5-2.0 s force): fell during the load (tilt 129 deg); 11 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import BalanceProgram, terrain, force, q_trace, pelvis_kin, upright, base_validity, no_trace, fail


@register
class SustainedLoading(BalanceProgram):
    id = "P18_sustained_loading"; family = Family.BALANCE; title = "Hold the stance under a sustained horizontal force"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE, Capability.ANKLE_ROLL]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P17 is an impulse (momentum); here the static equilibrium itself moves (the COM must lean against the load) and the release is a second disturbance."
    def world(self, scenario): p = scenario.params; return WorldSpec(terrain=terrain("flat"), disturbances=[force(p["t0"], p["t1"], p["F"])], tag="flat")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); F = float(rng.uniform(15.0, 70.0)); a = float(rng.uniform(-np.pi, np.pi))
        return self.scenario(seed, dict(robot="G1", Fmag=F, dir=a, F=[F*np.cos(a), F*np.sin(a), 0.0], t0=0.5, t1=2.0, T=3.5, t_hold=0.5), **ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="stability", params=dict(disturbance="force", F=p["Fmag"], dir=p["dir"], window=[p["t0"], p["t1"]], objective="min_tilt"), tolerance=dict(tilt=np.radians(30), t_hold=p["t_hold"]), horizon=p["T"])
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; xyz, yaw, tilt, v = pelvis_kin(q, dt); i0, i1 = int(p["t0"]/dt), min(len(q), int(p["t1"]/dt)); tilt_load = float(np.degrees(tilt[i0:i1].max())) if i1 > i0 else 90.0
        up, m = upright(q, dt, T=p["t_hold"]); up_all = bool((tilt < np.radians(30)).all() and (xyz[:, 2] > 0.7*xyz[0, 2]).all()); val = base_validity(rec)
        if not up: fail(val, "NOT_UPRIGHT_AT_END")
        if not up_all: fail(val, "NOT_UPRIGHT_DURING_LOAD")
        return Outcome(value=-tilt_load, achieved=(up and up_all and val.ok), measures=dict(tilt_during_load_deg=tilt_load, drift_xy=float(np.linalg.norm(xyz[-1, :2] - xyz[0, :2])), **m), validity=val)
