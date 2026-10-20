"""Program 70 (payload): CARRY a held load along a floor path. Needs LOCOMOTION under load: the representation is the
contact_schedule+knots (per-phase support contract + chained torque-knot segments) with the payload welded to the hand
(G1, experimental simplification). This program builds the PHASE DEMAND (support contract sequence derived from the path length and
a step count, held object, path waypoints, events = COM passing each waypoint with the load still held) and the evaluation; it does
NOT write a gait. Executable since the Plan runner: a one-step Plan on contact_schedule+knots over BaseEffectTask (_pay.carry_task).
integration check: 2026-10-12 (Plan runner) — prior-only rollout of the sampled 7-phase schedule (zero knots, no prior), value 0.105 m carried, achieved False, gates fired [P0:FELL,P1:NONFOOT_CONTACT,P1:STANCE_LOST,P1:FELL,P2:NONFOOT_CONTACT,P2:STANCE_LOST,P3:NONFOOT_CONTACT,P3:STANCE_LOST,P4:NONFOOT_CONTACT,P5:NONFOOT_CONTACT,P5:STANCE_LOST,P6:NONFOOT_CONTACT]; payload held through all phases (held_after ['payload']), path 1.96 m; the zero-knot one-foot phases fall as expected (41 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec, Plan, Step
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from . import _pay as P


@register
class CarryAlongPath(P.PayloadProgram, TaskProgram):
    id = "P70_carry_along_path"; family = Family.PAYLOAD; title = "Carry a held load along a path on foot"
    requires = [Capability.PALM_SITE_RIGHT, Capability.WELD_GRASP, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE, Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT]; status = Status.INTEGRATED
    distinct_from = "P65-P69 lift in place on two planted feet; here the support contract must change (steps) while the load is held, which couples the carry to the locomotion family's phase machinery."
    def world(self, scenario): p = scenario.params; return WorldSpec(objects=[O.payload_with_handle(name="payload", half=[0.08, 0.08, 0.06], mass=p["mass"], pos=[0.35, -0.2, 0.06])])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); L = float(rng.uniform(1.0, 2.5)); p = dict(mass=float(rng.uniform(1.0, 5.0)), path=[[0.0, 0.0], [L/2, 0.0], [L, 0.0]], tol=0.15, steps=int(round(L/0.3)), lift=0.0); return P.B.scenario(self, seed, p, ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="sequence", params=dict(subgoals=[dict(event="held", object="payload")] + [dict(event="com_passes", xy=w, tol=p["tol"], held=True) for w in p["path"][1:]]), horizon=2.0 + 1.5*len(p["path"]))
    def strategy_space(self, scenario): return ["contact_schedule+knots", "servo_knots+step_prior"]
    def demand(self, scenario):
        """One-step Plan on the contact_schedule+knots representation (construction/contact_schedule over BaseEffectTask, P.carry_task):
        alternating single-support phases sampled from the path length, the payload held throughout (engine.world weld; a declared-held
        object starts in the hand: LABEL held_start_abstraction), the effect = base displacement along the path direction. The waypoint
        events are judged by evaluate on the base displacement (checked through runner.check: see the module docstring)."""
        p = scenario.params; n = p["steps"]; phases = [dict(support=("left" if k % 2 == 0 else "right"), horizon=0.6, held=["payload"]) for k in range(n)]
        path = np.asarray(p["path"], float); dxy = path[-1] - path[0]; az = float(np.arctan2(dxy[1], dxy[0]))
        t, d = P.carry_task(p.get("robot", "G1"), phases, held=["payload"], direction=(0.0, az)); d.update(path=p["path"], events=[dict(kind="com_passes", xy=w, tol=p["tol"]) for w in p["path"][1:]])
        return Plan(steps=[Step(task=t, demand=d, held=["payload"], event="com_passes", representation="contact_schedule+knots")], robot=t.robot)
    def evaluate(self, rec, scenario):
        e = P.final_extra(rec); path = np.asarray(scenario.params["path"], float); bd = e.get("base_displacement"); com = e.get("com_xy_end") or e.get("com_xy")
        dist = float(np.linalg.norm(np.asarray(bd)[:2])) if bd is not None else (float(np.linalg.norm(np.asarray(com)[:2] - path[0])) if com is not None else 0.0)
        phases = e.get("phases") or []; held = ("payload" in (phases[-1].get("held_after") or [])) if phases else P.held(rec, "payload")
        return P.B.outcome(rec, value=dist, measures=dict(carry_distance=dist, path_length=float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))), held_after=(phases[-1].get("held_after") if phases else e.get("held_after")), n_phases=len(phases)),
                           extra_reasons=([] if held else ["DROPPED"]) + (["MISSING:" + m["goal"] for m in rec.get("missing", [])]), achieved=held and dist >= float(np.linalg.norm(path[-1] - path[0])) - scenario.params["tol"])
