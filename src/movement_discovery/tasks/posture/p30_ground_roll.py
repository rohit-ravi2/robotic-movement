"""Program 30 (posture): ground roll to orientation. Start lying (supine or prone) and end lying in the commanded orientation (rolled
by +-180 deg about the body's long axis, or a quarter turn onto the side) WITHOUT standing (pelvis stays below z_max). Outcome = -roll
error at the end, at rest. integration check: 2026-10-12 plant build: prone start settled on the world (pelvis z 0.15 m, tilt 87 deg, 0.4 s); no rollout (no representation starts from a non-standing posture yet)."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import PostureProgram, terrain, q_trace, pelvis_kin, posture_validity, no_trace, fail


def _roll_pitch(q4):
    w, x, y, z = q4; roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)); pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1)); return float(roll), float(pitch)


@register
class GroundRoll(PostureProgram):
    id = "P30_ground_roll"; family = Family.POSTURE; title = "Roll on the ground to a commanded lying orientation"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.PALM_SITE_RIGHT, Capability.PALM_SITE_LEFT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE, Capability.HIP_ROLL]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "No rise at all: the whole-body angular momentum about the long axis must be generated against the floor by limb motion while the COM stays low (P28/P29 end standing)."
    def world(self, scenario): return WorldSpec(terrain=terrain("flat"), tag="flat")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); start = str(rng.choice(["supine", "prone"])); return self.scenario(seed, dict(robot="G1", start=start, droll=float(rng.choice([-1, 1])*rng.choice([np.pi/2, np.pi])), tol=np.radians(15), z_max=0.45, t_rest=0.4, horizon=[1.5, 5.0]), **ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="pose", params=dict(final="lying", droll=p["droll"], z_max=p["z_max"], terminal="rest"), tolerance=dict(roll=p["tol"]), horizon=p["horizon"][1])
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; xyz, yaw, tilt, v = pelvis_kin(q, dt); r0, _ = _roll_pitch(q[0, 3:7]); r1, _ = _roll_pitch(q[-1, 3:7]); droll = (r1 - r0 + np.pi) % (2*np.pi) - np.pi
        err = float(abs((droll - p["droll"] + np.pi) % (2*np.pi) - np.pi)); val = posture_validity(rec); n = max(1, int(round(p["t_rest"]/dt)))
        if xyz[:, 2].max() > p["z_max"]: fail(val, "ROSE_TOO_HIGH", float(xyz[:, 2].max()) - p["z_max"])
        if np.linalg.norm(v[-n:], axis=1).max() > 0.15: fail(val, "NOT_AT_REST")
        return Outcome(value=-err, achieved=(err <= p["tol"] and val.ok), measures=dict(droll_deg=float(np.degrees(droll)), err_deg=float(np.degrees(err)), z_max=float(xyz[:, 2].max())), validity=val)
