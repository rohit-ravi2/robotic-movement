"""Payload-family helpers (): WorldReachTask (force_interaction/_mech) steps over WORLD objects/surfaces, returned as
core.contracts.Plan (runner.run_sequence executes them from the real handed-off states). Holds on the G1 = engine.world WELD at the
object's grip site (experimental simplification, Capability.WELD_GRASP); G1H = Dex3 fingers (declared, the wrap recipe of
engine/grasps applies to cylindrical handles only). Value conventions: lifted height (m) of the payload at the end of the last step,
placement error (m, negated), rotation (rad). carry_task: the LOCOMOTION-under-load step (contact_schedule+knots over
construction.base_effect.BaseEffectTask with the payload held). MuJoCo-free at import."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Plan, Step
from movement_discovery.tasks.force_interaction import _base as B, _mech as M


def task(robot, spec): return M.make(robot, spec, colliders=(), hand_collision=False)   # weld abstraction: hand<->payload contact is the weld, not a collider (labelled)


def reach_grasp(t, name, lift_clear=0.01, hold=0.3, horizon=3.0, **kw):
    d = dict(t.demand_default(), target_object=name, push=[0.0, 0.0, lift_clear], tol=0.05, hold=hold, grasp_on_reach=name, grasp_tol=0.05, omega_servo=120.0, trunk_cap=0.05, horizon=horizon, refine=1, aim_z=True); d.update(kw); return d


def move_held(t, name, push, hold=0.3, horizon=3.0, **kw):
    """Carry the held object so its grip point moves by `push` (world) from where it is at the handoff."""
    d = dict(t.demand_default(), target_object=name, push=list(push), tol=0.05, hold=hold, held=[name], omega_servo=120.0, trunk_cap=0.05, horizon=horizon); d.update(kw); return d


def place_held(t, name, target_world, hold=0.3, horizon=3.0, **kw):
    d = dict(t.demand_default(), target_world=list(target_world), tol=0.05, hold=hold, held=[name], release_at_end=True, omega_servo=120.0, trunk_cap=0.05, horizon=horizon); d.update(kw); return d


def plan(t, demands, events=(), settle_T=0.0, freeze_T=0.4, missing=(), **kw) -> Plan:
    """Plan of WorldReachTask steps (one demand each; held objects read from the demands; events name the world condition per step)."""
    ev = list(events) + [None]*(len(demands) - len(events))
    return Plan(steps=[Step(task=t, demand=dict(d), held=list(d.get("held") or []), event=e) for d, e in zip(demands, ev)], robot=t.robot, settle_T=settle_T, freeze_T=freeze_T, missing=list(missing), **kw)


def carry_task(robot, phases, held=("payload",), direction=None, measure="displacement"):
    """(BaseEffectTask, demand) for a carry step: the contact_schedule+knots representation chains the phase plants (support contract
    per phase, the held payload welded/placed in the hand through engine.world: LABEL held_start_abstraction), the effect is the base
    displacement along `direction` (None = horizontal magnitude). The program samples the phases; nothing here is a gait."""
    from movement_discovery.construction.base_effect import BaseEffectTask
    t = BaseEffectTask(robot); d = dict(t.demand_default(), effector="pelvis", measure=measure, direction=direction, support="both", prior=None, phases=[dict(ph) for ph in phases], held=list(held))
    return t, d


def final_extra(rec): return rec["steps"][-1]["extra"] if "steps" in rec else rec["extra"]


def lift_height(rec, name):
    e = final_extra(rec); d = (e.get("obj_disp") or {}).get(name); return float(d[2]) if d is not None else 0.0


def held(rec, name): return name in (final_extra(rec).get("held_after") or [])


def final_state(rec):
    """(qpos, qvel) at the end: run_sequence keeps it at the top level (after the freeze); a single rollout in extra."""
    return rec.get("final_state") if "steps" in rec else rec["extra"].get("final_state")


class PayloadProgram:
    """Mixin: lifted-height evaluate with the hold gate (DROPPED when the payload is not in the hand at the end unless `releases`)."""
    obj_name = "payload"; releases = False; height_key = "lift"
    def embodiment(self, robot):
        from movement_discovery import embodiment as E; return E.resolve(robot, self)
    def evaluate(self, rec, scenario):
        h = lift_height(rec, self.obj_name); req = float(scenario.params.get(self.height_key, 0.0)); hd = held(rec, self.obj_name); extra = [] if (hd or self.releases) else ["DROPPED"]
        m = dict(lift_height=h, required=req, held_after=final_extra(rec).get("held_after"), f_max=final_extra(rec).get("f_max_obj"), obj_disp=final_extra(rec).get("obj_disp"))
        if "steps" in rec: return B.seq_outcome(rec, value=h, measures=m, extra_reasons=extra, achieved=h >= req)
        return B.outcome(rec, value=h, measures=B.reach_measures(rec) | m, extra_reasons=extra, achieved=h >= req)
    def validity(self, rec, scenario): return self.evaluate(rec, scenario).validity
    def observe(self, rec, scenario):
        e = final_extra(rec); return dict(work=e.get("work"), trunk_pitch_max=e.get("trunk_pitch_max"), knee_max=e.get("knee_max"), peak_util=e.get("peak_util"), cp_err_max=e.get("cp_err_max"))
    def run(self, scenario):
        """Thin wrapper: ONE prior-only rollout of the Plan through the runner (raw record)."""
        from movement_discovery import runner
        return runner.rollout(self.id, scenario, robot=scenario.params.get("robot", "G1"))["rec"]
