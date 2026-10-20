"""Program 86 (negotiation): reach through a constrained opening. A wall with a rectangular hole (world.obstacles.opening) stands ahead;
the right palm must reach a target point BEHIND the wall, passing through the hole, and hold there, feet planted. Outcome = -palm miss
distance at the hold. Bridge: the quasi-static servo construction (tasks/reach.py: ReachTask on the world scene; representation
servo_knots). The hole constraint is judged geometrically on the palm path (the wall pieces carry hand/forearm floor pairs: a collision
also shows as a stalled reach). integration check: 2026-10-12 one prior-only ReachTask rollout on the opening scene (servo construction, zero offsets): the straight-line ascent prior drives the arm into the wall and the robot falls (NONFOOT_CONTACT, STANCE_LOST, FELL; miss 1.71 m), 19.6 s: the constraint bites; the prior alone cannot satisfy it."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import NegotiationProgram, terrain, obstacle, q_trace, base_validity, no_trace, fail, sites_xyz


@register
class ConstrainedOpening(NegotiationProgram):
    id = "P86_constrained_opening"; family = Family.NEGOTIATION; title = "Reach a point behind a wall through a hole"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.PALM_SITE_RIGHT, Capability.SERVO_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "A manipulation-space constraint (the arm's path, not the body's travel): the straight-line reach is blocked, the feasible palm paths pass through a window smaller than the arm's sweep."
    observe_site = "right_palm"
    def world(self, scenario):
        p = scenario.params; return WorldSpec(terrain=terrain("flat"), mechanisms=[obstacle("opening", size=(p["hole_w"], p["hole_h"]), height=p["sill"], x=p["x_wall"], wall_w=2.0, wall_h=1.8, thick=0.04)], tag=f"opening{int(p['hole_w']*100)}")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); sill = float(rng.uniform(0.6, 1.0)); hw = float(rng.uniform(0.18, 0.35)); hh = float(rng.uniform(0.18, 0.35)); xw = 0.38
        return self.scenario(seed, dict(robot="G1", x_wall=xw, hole_w=hw, hole_h=hh, sill=sill, hole_y=float(rng.uniform(-0.25, 0.0)), target=[xw + float(rng.uniform(0.10, 0.22)), float(rng.uniform(-0.25, 0.0)), sill + hh/2 + float(rng.uniform(-0.05, 0.05))], tol=0.03, hold=0.2, horizon=[1.5, 1.5]), **ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="effector_position", params=dict(effector="hand_r", target=p["target"], through=dict(x=p["x_wall"], y=[-p["hole_w"]/2, p["hole_w"]/2], z=[p["sill"], p["sill"] + p["hole_h"]])), tolerance=dict(pos=p["tol"], hold=p["hold"]), horizon=p["horizon"][1])
    def strategy_space(self, scenario): return ["servo_knots"]
    def demand(self, scenario):
        from tasks.reach import ReachTask; from movement_discovery import world as W
        p = scenario.params; t = ReachTask(p.get("robot", "G1"), scene=W.build(self.world(scenario), p.get("robot", "G1")))
        return t, dict(t.demand_default(), target=p["target"], target_frame="world", tol=p["tol"], hold=p["hold"], horizon=p["horizon"][1], prior="ascent")
    def evaluate(self, rec, scenario):
        if not rec: return no_trace()
        p = scenario.params; ex = rec.get("extra") or {}; val = base_validity(rec); val.reasons = [r for r in val.reasons if r not in ("NOT_REACHED", "NOT_HELD")]
        palm_hold = np.asarray(ex.get("palm_hold", ex.get("palm_end", [np.nan]*3)), float); miss = float(np.linalg.norm(palm_hold - np.asarray(p["target"]))) if np.isfinite(palm_hold).all() else 1.0
        q, dt = q_trace(dict(trace=dict(q=ex.get("trace"), dt=10*rec.get("dt", 0.001)))) if ex.get("trace") is not None else (None, None); through = None
        if q is not None:
            t, d = self.demand(scenario); palm = sites_xyz(t.plant(d), q, ["right_palm"])[:, 0]; beyond = palm[:, 0] > p["x_wall"]
            if beyond.any():
                i = int(np.argmax(beyond)); yz = palm[i, 1:]; through = bool(-p["hole_w"]/2 <= yz[0] <= p["hole_w"]/2 and p["sill"] <= yz[1] <= p["sill"] + p["hole_h"])
                if not through: fail(val, "CROSSED_WALL_OUTSIDE_HOLE")
        if miss > p["tol"]: fail(val, "NOT_REACHED", miss - p["tol"])
        if not ex.get("hold_ok", False): fail(val, "NOT_HELD")
        return Outcome(value=-miss, achieved=(miss <= p["tol"] and val.ok), measures=dict(miss=miss, through_hole=through, t_reach=ex.get("t_reach"), work=ex.get("work")), validity=val)
