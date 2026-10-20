"""Program 26 (posture): stand-to-sit. Start standing with a box seat behind the feet; end seated (pelvis over the seat at seat height,
at rest) with a bounded contact speed at seating. Outcome = -pelvis descent speed at seat contact (gentler is better). The descent
strategy is free. integration check: 2026-10-12 plant build: canonical stance on the world, 0.8 s; no rollout (contact_schedule+knots needs a base-level effect mode)."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import PostureProgram, terrain, q_trace, pelvis_kin, posture_validity, no_trace, fail, SEAT_RANGE


@register
class StandToSit(PostureProgram):
    id = "P26_stand_to_sit"; family = Family.POSTURE; title = "Lower onto a seat from standing"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "The reverse of P25 is not its time-reversal: the goal is a soft arrival on a support behind the feet (impact-bounded), and P31 lowers without any seat."
    def strategy_space(self, scenario): return super().strategy_space(scenario) + ["torque_knots+rise_prior"]      # RISE: certified support-transition prior (appended)
    def world(self, scenario): p = scenario.params; return WorldSpec(terrain=terrain("platform", size=(0.25, 0.3), height=p["seat_h"], pos=(p["seat_x"], 0.0)), tag=f"seat{int(p['seat_h']*100)}")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); return self.scenario(seed, dict(robot="G1", start=None, seat_h=float(rng.uniform(*SEAT_RANGE)), seat_x=-0.20, seat_half=0.25, vz_max=0.5, t_rest=0.4, horizon=[1.0, 3.5]), **ov)
    def start_posture(self, scenario): return None            # standing start: the plant's canonical stance
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="pose", params=dict(final="seated", seat_h=p["seat_h"], seat_x=p["seat_x"], terminal="rest", objective="min_contact_speed"), tolerance=dict(vz_max=p["vz_max"], x=p["seat_half"]), horizon=p["horizon"][1])
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; xyz, yaw, tilt, v = pelvis_kin(q, dt); vz = np.gradient(xyz[:, 2], dt); z_seat = p["seat_h"] + 0.15; hit = np.where(xyz[:, 2] <= z_seat + 0.02)[0]; val = posture_validity(rec)
        n = max(1, int(round(p["t_rest"]/dt))); seated_end = bool(abs(xyz[-1, 2] - z_seat) < 0.06 and abs(xyz[-1, 0] - p["seat_x"]) < p["seat_half"] and np.linalg.norm(v[-n:], axis=1).max() < 0.15 and tilt[-1] < np.radians(30))
        vz_hit = float(-vz[hit[0]]) if len(hit) else None
        if not seated_end: fail(val, "NOT_SEATED")
        if vz_hit is not None and vz_hit > p["vz_max"]: fail(val, "HARD_SEATING", vz_hit - p["vz_max"])
        return Outcome(value=-(vz_hit if vz_hit is not None else 5.0), achieved=(seated_end and val.ok), measures=dict(vz_at_seat=vz_hit, z_end=float(xyz[-1, 2]), x_end=float(xyz[-1, 0])), validity=val)
