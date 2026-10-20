"""Shared helpers for the OBJECT families (force_interaction, payload, interception, compound). MuJoCo-free at import: every
reference-engine import happens inside functions. Nothing here prescribes a movement; it wraps records into contracts.
Validity flags come from the one table in physics.validity (classify); sequences run through runner.run_sequence (Plans); the
prior-only rollout is runner.check / runner.rollout (the former prior_rollout / run_steps copies are gone)."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import EmbodimentSpec, Capability, Scenario, Outcome, ValidityReport
from movement_discovery.physics import validity as V
R = np.radians


def embodiment(robot, caps, driven=None, effectors=None, notes=""):
    import throw as T
    cfg = T.CFG[robot]; eff = {"hand_r": cfg["hand_site"], "foot_r": "right_foot", "foot_l": "left_foot"}; eff.update(effectors or {})
    return EmbodimentSpec(robot=robot, mjcf=cfg.get("mjcf") or "", driven=list(driven or cfg["driven"]), effectors=eff, capabilities=list(caps), notes=notes)


def scenario(prog, seed, params, overrides):
    p = dict(params); p.update(overrides); return Scenario(id=f"{prog.id}_s{seed}", program_id=prog.id, seed=seed, params=p)


def impact_task(robot, **effect):
    """(ImpactTask, demand): the reference impact construction (effector collider, bare-robot reference, sustained-force gates)."""
    from tasks.impact import ImpactTask
    t = ImpactTask(robot); return t, dict(t.demand_default(), **effect)


def validity(rec, extra_reasons=()):
    """ValidityReport from a record: physics.validity.classify (the one reason -> flag table; step prefixes s<i>: stripped)."""
    return V.classify(rec, extra_reasons)


def outcome(rec, value, measures=None, extra_reasons=(), extra_violation=0.0, achieved=None):
    v = validity(rec, extra_reasons); v.violation += float(extra_violation)
    return Outcome(value=float(value), achieved=bool(v.ok if achieved is None else (achieved and v.ok)), measures=dict(measures or {}), validity=v, trace_ref=None)


def impact_measures(rec):
    e = rec.get("extra", {}); keys = ("ball_speed", "ball_elev", "ball_az", "eff_speed", "dmin_obj", "f_max_obj", "t_contact", "t_rel", "work", "com_speed")
    return {k: (float(e[k]) if isinstance(e.get(k), (int, float, np.floating)) else e.get(k)) for k in keys if k in e}


def reach_measures(rec):
    e = rec.get("extra", {}); keys = ("dmin", "t_reach", "t_grasp", "hold_ok", "peak_util", "cp_err_max", "trunk_pitch_max", "knee_max", "W_act", "held_after", "end_qdot")
    out = {k: e.get(k) for k in keys if k in e}; out["work"] = e.get("work"); return out


def observe_impact(task, demand, rec, effector):
    """engine.emergence summary of a RECORDED impact/effect rollout (descriptive, never a filter)."""
    if not rec.get("trace"): return None
    from engine import emergence as EM
    return EM.analyze(task.plant(demand), rec, effector)


def box(name, size, mass, pos, rgba=(0.2, 0.4, 0.9, 1), **kw):
    from engine import world as W
    return W.obj(name, kind="box", size=tuple(size), mass=float(mass), pos=list(pos), rgba=rgba, **kw)


def summary(rec):
    return dict(value=round(float(rec.get("value", 0.0)), 4), valid=bool(rec.get("valid")), reasons=list(rec.get("reasons", [])), violation=round(float(rec.get("violation", 0.0)), 3))


def seq_outcome(res, value, measures=None, extra_reasons=(), achieved=None):
    """Outcome of a runner.run_sequence record: validity = every step valid + handoffs ok + settled; reasons prefixed by step index."""
    rs = [f"s{s['i']}:{r}" for s in res["steps"] for r in s["reasons"]] + ([] if res["ok"] else ["SEQUENCE_NOT_OK"]) + list(extra_reasons)
    viol = float(sum(s["violation"] for s in res["steps"]))
    rec = dict(reasons=rs, violation=viol); v = validity(rec); v.support = all(s["handoff"]["ok"] for s in res["steps"])
    return Outcome(value=float(value), achieved=bool(v.ok if achieved is None else (achieved and v.ok)), measures=dict(measures or {}, steps=[dict(task=s["task"], value=s["value"], valid=s["valid"], reasons=s["reasons"]) for s in res["steps"]], t_total=res["t_total"]), validity=v)
