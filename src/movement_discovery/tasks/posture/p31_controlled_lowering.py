"""Program 31 (posture): controlled lowering. From standing, bring the pelvis down to a commanded height and hold it at rest, never
exceeding a descent speed bound (no drop) and ending with the feet still the support. Outcome = -|z_end - z*|. Whether this is a squat,
a kneel-less crouch or anything else is the construction's. integration check: 2026-10-12 plant build: canonical stance on the world, 0.0 s; no rollout (contact_schedule+knots needs a base-level effect mode)."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import PostureProgram, terrain, q_trace, pelvis_kin, base_validity, no_trace, fail, upright


@register
class ControlledLowering(PostureProgram):
    id = "P31_controlled_lowering"; family = Family.POSTURE; title = "Lower the body to a commanded height and hold"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P26 ends on a seat (support transfers to the pelvis); here the feet stay the only support and the end state is held by the legs. P09's crouch is a means; here the low posture is the goal."
    def strategy_space(self, scenario): return super().strategy_space(scenario) + ["torque_knots+rise_prior"]      # RISE: certified support-transition prior (appended)
    def world(self, scenario): return WorldSpec(terrain=terrain("flat"), tag="flat")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); return self.scenario(seed, dict(robot="G1", start=None, z_target=float(rng.uniform(0.30, 0.55)), vz_max=0.6, tol=0.03, t_hold=0.5, horizon=[1.0, 3.0]), **ov)
    def start_posture(self, scenario): return None
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="com_displacement", params=dict(axis="z", pelvis_target=p["z_target"], terminal="rest", support="feet"), tolerance=dict(z=p["tol"], vz_max=p["vz_max"]), horizon=p["horizon"][1])
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; xyz, yaw, tilt, v = pelvis_kin(q, dt); vz = np.gradient(xyz[:, 2], dt); n = max(1, int(round(p["t_hold"]/dt))); val = base_validity(rec)
        err = float(abs(xyz[-n:, 2].mean() - p["z_target"])); drop = float(max(0.0, -vz.min() - p["vz_max"])); v_end = float(np.linalg.norm(v[-n:], axis=1).max())
        if drop > 0: fail(val, "DESCENT_TOO_FAST", drop)
        if v_end > 0.15: fail(val, "NOT_AT_REST", v_end - 0.15)
        if tilt[-n:].max() > np.radians(45): fail(val, "NOT_UPRIGHT_AT_END")
        return Outcome(value=-err, achieved=(err <= p["tol"] and val.ok), measures=dict(z_end=float(xyz[-1, 2]), err=err, vz_min=float(vz.min()), v_end=v_end, tilt_end_deg=float(np.degrees(tilt[-1]))), validity=val)
