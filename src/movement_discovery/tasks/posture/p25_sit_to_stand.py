"""Program 25 (posture): sit-to-stand. Start seated on a box seat (world terrain 'platform' under the pelvis, thighs horizontal) with the feet on the floor;
end standing on both feet at rest. Outcome = -time to reach standing. No rise strategy (trunk lean, arm use, foot placement) is given.
integration check: 2026-10-12 round 2, runner.check (torque_knots over BaseEffectTask from demand['initial_posture'] = the seated start on the seat scene; prior-only): stays seated, NOT_STANDING, viol 0.29; 171 params; 8 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import PostureProgram, terrain, q_trace, pelvis_kin, standing, posture_validity, no_trace, fail, SEAT_RANGE


@register
class SitToStand(PostureProgram):
    id = "P25_sit_to_stand"; family = Family.POSTURE; title = "Rise from a seat to standing"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P27 starts with the shins on the floor (no seat, knees loaded); P29 starts lying. Here the pelvis is supported at seat height and the feet are already planted."
    def strategy_space(self, scenario): return super().strategy_space(scenario) + ["torque_knots+rise_prior"]      # RISE: certified support-transition prior (appended)
    def world(self, scenario): p = scenario.params; return WorldSpec(terrain=terrain("platform", size=(0.25, 0.3), height=p["seat_h"], pos=(p["seat_x"], 0.0)), tag=f"seat{int(p['seat_h']*100)}")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); return self.scenario(seed, dict(robot="G1", start="seated", seat_h=float(rng.uniform(*SEAT_RANGE)), seat_x=0.0, z_stand=self.Z_STAND, horizon=[1.5, 4.0]), **ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="pose", params=dict(final="standing", z_stand=p["z_stand"], terminal="rest", objective="min_time"), tolerance=dict(z=0.06, tilt=np.radians(15)), horizon=p["horizon"][1])
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; ts = standing(q, dt, p["z_stand"]); val = posture_validity(rec); xyz = pelvis_kin(q, dt)[0]
        if ts is None: fail(val, "NOT_STANDING", max(0.0, p["z_stand"] - float(xyz[:, 2].max())))
        return Outcome(value=-(ts if ts is not None else len(q)*dt + 1.0), achieved=(ts is not None and val.ok), measures=dict(t_stand=ts, z_max=float(xyz[:, 2].max()), z_end=float(xyz[-1, 2])), validity=val)
