"""Program 28 (posture): prone-to-stand. Start lying face down on the floor; end standing on both feet at rest. Outcome = -time to
standing; hands and knees may bear load. integration check: 2026-10-12 round 2, runner.check (torque_knots over BaseEffectTask from the prone initial posture; prior-only): stays prone, NOT_STANDING; 6 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import PostureProgram, terrain, q_trace, pelvis_kin, standing, posture_validity, no_trace, fail


@register
class ProneToStand(PostureProgram):
    id = "P28_prone_to_stand"; family = Family.POSTURE; title = "Rise from lying face down to standing"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.PALM_SITE_RIGHT, Capability.PALM_SITE_LEFT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "Face-down the arms can push directly against the floor and the hips are already flexion-ready; P29 (face-up) must first turn or sit up. P27 starts with the trunk already vertical."
    def strategy_space(self, scenario): return super().strategy_space(scenario) + ["torque_knots+rise_prior"]      # RISE: certified support-transition prior (appended)
    def world(self, scenario): return WorldSpec(terrain=terrain("flat"), tag="flat")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); return self.scenario(seed, dict(robot="G1", start="prone", z_stand=self.Z_STAND, friction=float(rng.choice([0.6, 0.9])), horizon=[2.0, 6.0]), **ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="pose", params=dict(final="standing", z_stand=p["z_stand"], terminal="rest", objective="min_time"), tolerance=dict(z=0.06, tilt=np.radians(15)), horizon=p["horizon"][1])
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; ts = standing(q, dt, p["z_stand"]); val = posture_validity(rec); xyz = pelvis_kin(q, dt)[0]
        if ts is None: fail(val, "NOT_STANDING", max(0.0, p["z_stand"] - float(xyz[:, 2].max())))
        return Outcome(value=-(ts if ts is not None else len(q)*dt + 1.0), achieved=(ts is not None and val.ok), measures=dict(t_stand=ts, z_max=float(xyz[:, 2].max()), z_end=float(xyz[-1, 2])), validity=val)
