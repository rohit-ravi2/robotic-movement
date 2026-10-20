"""Program 22 (balance): moving-platform stabilisation. The support surface translates with a commanded profile (world terrain 'platform' with
moving=dict(axis, amp, freq); disturbances.platform_motion sets its mocap pose); the robot must stay upright for the whole horizon and keep its base within a
bound of the platform. Outcome = -peak pelvis tilt. Needs the platform pose in the record (trace['platform_xy']); no rollout today calls
world.apply_mocap each step. integration check: 2026-10-12 round 2, runner.check (StandTask on the platform scene, start_z 0.10, mocap applied 4000/4000 steps): stayed upright, tilt max 2.0 deg; the evaluator then lacked the platform pose (NO_PLATFORM_TRACE) -> now recomputed from the WorldSpec (mocap_targets; synthetic test passes), not re-run; 15 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import BalanceProgram, terrain, platform_motion, q_trace, pelvis_kin, upright, base_validity, no_trace, fail


@register
class MovingPlatform(BalanceProgram):
    id = "P22_moving_platform"; family = Family.BALANCE; title = "Stand on a translating platform"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE, Capability.ANKLE_ROLL]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "The disturbance enters through the feet (support acceleration), continuously and predictably, not through the torso (P17/P18)."
    def world(self, scenario):
        p = scenario.params; mv = dict(axis=p["axis"], amp=p["amp"], freq=p["freq"], phase=0.0)
        return WorldSpec(terrain=terrain("platform", size=(0.6, 0.6), height=0.10, pos=(0.0, 0.0), moving=mv), disturbances=[platform_motion("platform", p["axis"], p["amp"], p["freq"], 0.0, p["t0"], None)], tag="platform")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); return self.scenario(seed, dict(robot="G1", axis=[1.0, 0.0, 0.0] if rng.random() < 0.5 else [0.0, 1.0, 0.0], amp=float(rng.uniform(0.03, 0.12)), freq=float(rng.uniform(0.3, 1.2)), t0=0.5, T=4.0, t_hold=0.5, drift_max=0.3), **ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="stability", params=dict(disturbance="platform", axis=p["axis"], amplitude=p["amp"], frequency=p["freq"], objective="min_tilt"), tolerance=dict(tilt=np.radians(30), drift=p["drift_max"]), horizon=p["T"])
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; xyz, yaw, tilt, v = pelvis_kin(q, dt); up, m = upright(q, dt, T=p["t_hold"]); val = base_validity(rec)
        up_all = bool((tilt < np.radians(30)).all() and (xyz[:, 2] > 0.7*xyz[0, 2]).all()); plat = (rec.get("trace") or {}).get("platform_xy")
        if plat is None:                        # the platform pose is kinematic: recompute it from the WorldSpec (world.disturbances.mocap_targets)
            from movement_discovery.world.disturbances import mocap_targets; w = self.world(scenario)
            plat = np.array([mocap_targets(w, i*dt)["platform"][:2] for i in range(len(q))]) - mocap_targets(w, 0.0)["platform"][:2]
        drift = float(np.max(np.linalg.norm((xyz[:, :2] - xyz[0, :2]) - np.asarray(plat, float), axis=1)))
        if not up or not up_all: fail(val, "NOT_UPRIGHT")
        if drift > p["drift_max"]: fail(val, "LEFT_PLATFORM", drift - p["drift_max"])
        return Outcome(value=-float(np.degrees(tilt.max())), achieved=(up and up_all and val.ok), measures=dict(drift_max=drift, **m), validity=val)
