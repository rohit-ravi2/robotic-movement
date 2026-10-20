"""Program 06 (locomotion): curved path following. Travel along a circular arc (radius R, arc angle) at a commanded speed; outcome =
mean cross-track error of the base over the path. integration check: 2026-10-12 world builds; demand = contact_schedule+knots over BaseEffectTask (sampled schedule, prior-only = standing phases); not run (one rollout per family representative: P02)."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import BodyProgram, terrain, q_trace, pelvis_kin, upright, base_validity, no_trace, fail, schedule_demand, phase, CS_NAME


@register
class CurvedPath(BodyProgram):
    id = "P06_curved_path"; family = Family.LOCOMOTION; title = "Follow a circular arc at speed with bounded cross-track error"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE, Capability.HIP_ROLL]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P05 is free between two headings; here the lateral acceleration v^2/R is sustained along the whole path and scored continuously."
    def world(self, scenario): return WorldSpec(terrain=terrain("flat"), tag="flat")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); R = float(rng.uniform(0.8, 3.0)); arc = float(rng.uniform(np.radians(60), np.radians(180)))
        return self.scenario(seed, dict(R=R, arc=arc, side=int(rng.choice([-1, 1])), speed=float(rng.uniform(0.4, 1.0)), tol=0.15, horizon=[1.5, 1.5 + 1.5*R*arc/0.4]), **ov)
    def goal(self, scenario):
        p = scenario.params; return GoalSpec(kind="path", params=dict(shape="arc", R=p["R"], arc=p["arc"], side=p["side"], speed=p["speed"]), tolerance=dict(cross_track=p["tol"]), horizon=p["horizon"][1])
    def strategy_space(self, scenario): return [CS_NAME, "servo_knots+step_prior"]
    def demand(self, scenario):
        p = scenario.params; return schedule_demand(scenario, p["horizon"], [phase("feet_any", self.goal(scenario))], self.goal(scenario), effect=dict(effector="pelvis", direction=(0.0, 0.0)))
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; xyz, yaw, tilt, v = pelvis_kin(q, dt); c = xyz[0, :2] + np.array([0.0, p["side"]*p["R"]])   # arc centre: R to the side of the start, travel initially along +x
        rel = xyz[:, :2] - c; ct = np.abs(np.linalg.norm(rel, axis=1) - p["R"]); ang0 = np.arctan2(rel[0, 1], rel[0, 0]); ang = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0])) - ang0
        prog = float(p["side"]*ang[-1]); frac = float(np.clip(prog/p["arc"], 0.0, 1.0)); up, m = upright(q, dt, T=0.3); val = base_validity(rec)
        if not up: fail(val, "NOT_UPRIGHT_AT_END")
        if frac < 0.95: fail(val, "ARC_INCOMPLETE", 1.0 - frac)
        err = float(ct.mean())
        return Outcome(value=-(err + (1.0 - frac)), achieved=(err <= p["tol"] and val.ok), measures=dict(cross_track_mean=err, cross_track_max=float(ct.max()), arc_fraction=frac, **m), validity=val)
