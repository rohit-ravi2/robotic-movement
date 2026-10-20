"""Program 82 (negotiation): narrow corridor. Two walls (world.obstacles.corridor, inner faces at y = +-width/2) narrower than the
shoulder span; travel through to x_goal without touching them. Outcome = -shortfall to x_goal. integration check: not executable until
construction/contact_schedule+knots gets a base-level effect mode."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import NegotiationProgram, terrain, obstacle, q_trace, pelvis_kin, past, base_validity, no_trace, fail, travel_goal


@register
class NarrowCorridor(NegotiationProgram):
    id = "P82_narrow_corridor"; family = Family.NEGOTIATION; title = "Travel through a corridor narrower than the shoulders"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE, Capability.WAIST_YAW]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "A lateral width constraint along the whole path (arm swing and foot placement width both limited); P81 is a height constraint, P86 a single aperture."
    def world(self, scenario): p = scenario.params; return WorldSpec(terrain=terrain("flat"), mechanisms=[obstacle("corridor", width=p["width"], length=p["length"], x0=p["x0"], wall_h=1.5)], tag=f"corridor{int(p['width']*100)}")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); L = float(rng.uniform(1.0, 2.5)); return self.scenario(seed, dict(robot="G1", width=float(rng.uniform(0.32, 0.50)), length=L, x0=0.5, x_goal=0.5 + L + 0.3, horizon=[2.0, 10.0]), **ov)
    def goal(self, scenario): return travel_goal(scenario.params, GoalSpec)
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; ok, m = past(q, dt, p["x_goal"]); val = base_validity(rec); xyz, yaw, tilt, v = pelvis_kin(q, dt)
        inside = (xyz[:, 0] > p["x0"]) & (xyz[:, 0] < p["x0"] + p["length"]); y_max = float(np.abs(xyz[inside, 1]).max()) if inside.any() else 0.0
        if y_max > p["width"]/2: fail(val, "PELVIS_IN_WALL", y_max - p["width"]/2)
        if not ok: fail(val, "NOT_PAST_OBSTACLE", m["short"])
        return Outcome(value=-m["short"], achieved=(ok and val.ok), measures=dict(pelvis_y_max_inside=y_max, yaw_max_deg=float(np.degrees(np.abs(yaw - yaw[0]).max())), **m), validity=val)
