"""Program 08 (locomotion): step / stair negotiation. Reach the top landing of a flight of steps (world terrain 'steps': n, rise, run; ascending) and stand there; outcome = height gained toward the landing with forward progress past the last riser. Nothing about foot placement
or sequence is specified. integration check: 2026-10-12 world builds; demand = contact_schedule+knots over BaseEffectTask (sampled schedule, prior-only = standing phases); not run (one rollout per family representative: P02)."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import BodyProgram, terrain, q_trace, pelvis_kin, upright, base_validity, no_trace, fail, schedule_demand, phase, CS_NAME


@register
class StepNegotiation(BodyProgram):
    id = "P08_step_negotiation"; family = Family.LOCOMOTION; title = "Climb a flight of steps to its landing"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P07's support is continuous; here the support height is discontinuous and foot clearance of each riser is forced by the terrain. P83 (one barrier) has no height gain."
    def world(self, scenario):
        p = scenario.params; return WorldSpec(terrain=terrain("steps", n=p["n"], rise=p["rise"], run=p["run"], x0=p["x0"]), tag=f"steps{p['n']}")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); n = int(rng.integers(1, 5)); rise = float(rng.uniform(0.08, 0.18)); run = float(rng.uniform(0.25, 0.35))
        return self.scenario(seed, dict(n=n, rise=rise, run=run, x0=0.5, tol_h=0.03, tol_x=0.15, horizon=[1.5, 2.0 + 2.0*n]), **ov)
    def goal(self, scenario):
        p = scenario.params; x_top = p["x0"] + p["n"]*p["run"]; return GoalSpec(kind="com_displacement", params=dict(dz=p["n"]*p["rise"], x_min=x_top + 0.1, terminal="rest"), tolerance=dict(dz=p["tol_h"], x=p["tol_x"]), horizon=p["horizon"][1])
    def strategy_space(self, scenario): return [CS_NAME, "servo_knots+step_prior"]
    def demand(self, scenario):
        p = scenario.params; return schedule_demand(scenario, p["horizon"], [phase("feet_any", self.goal(scenario))], self.goal(scenario), notes="terrain forces riser clearance; no foot order is given", effect=dict(effector="pelvis", direction=(float(np.arctan2(p["rise"], p["run"])), 0.0)))
    def evaluate(self, rec, scenario):
        q, dt = q_trace(rec)
        if q is None: return no_trace()
        p = scenario.params; g = self.goal(scenario).params; xyz, yaw, tilt, v = pelvis_kin(q, dt); dz = float(xyz[-1, 2] - xyz[0, 2]); dx = float(xyz[-1, 0] - xyz[0, 0])
        h_err = abs(dz - g["dz"]); x_short = max(0.0, g["x_min"] - dx); n = max(1, int(0.3/dt)); v_end = float(np.linalg.norm(v[-n:], axis=1).max())
        up, m = upright(q, dt, T=0.3, z_frac=0.5); val = base_validity(rec)        # z_frac vs the START height: a descent legitimately ends lower
        if not up: fail(val, "NOT_UPRIGHT_AT_END")
        if v_end > 0.2: fail(val, "NOT_AT_REST", v_end - 0.2)
        if x_short > 0: fail(val, "LANDING_NOT_REACHED", x_short)
        return Outcome(value=-(h_err + x_short), achieved=(h_err <= p["tol_h"] and val.ok), measures=dict(dz=dz, dx=dx, v_end=v_end, **m), validity=val)
