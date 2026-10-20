"""Capacity perturbations of a SCENARIO forwarded into every Plan step's demand (applied by runner.assemble, once, not per program):
    scenario.params["tau_scale"]  = {group or joint name: factor}  -> demand["tau_scale"]   (tasks/effect.py EffectTask.caps: the prior's
                                     capacity weights, the search and the envelope gate see the scaled torque; inherited by ImpactTask,
                                     BaseEffectTask, the families' EffectTask subclasses)
    scenario.params["jnt_ranges"] = {joint: (lo, hi)} in any robot's naming -> demand["jnt_ranges"] re-keyed onto this robot's joints
                                     (embodiment.joint_ranges; unknown joints dropped) -> EffectTask.plant / BaseEffectTask.plant apply
                                     them to the model before the settle. tasks/impact.py ImpactTask.plant (frozen) ignores the key:
                                     ensure_ranges() applies it on the built plant (after the settle; LABEL jnt_ranges_applied=runner).
The generators emit these for the projection and impact families (scenarios/generators.CAPACITY_FAMILIES); consumed_by() reports, per
step, whether the step's task honours each key, so an inert perturbation is visible in the check line instead of silent."""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np

KEYS = ("tau_scale", "jnt_ranges")


def apply(demand: dict, params: dict, robot: str) -> dict:
    """demand with the scenario's capacity keys copied in (the program's own values win when it set them)."""
    out = dict(demand); ts = params.get("tau_scale"); jr = params.get("jnt_ranges")
    if ts and out.get("tau_scale") is None: out["tau_scale"] = {str(k): float(v) for k, v in dict(ts).items()}
    if jr and out.get("jnt_ranges") is None:
        from .adapter import joint_ranges
        rk = joint_ranges(robot, {str(k): (float(v[0]), float(v[1])) for k, v in dict(jr).items()})
        out["jnt_ranges"] = {k: [float(lo), float(hi)] for k, (lo, hi) in rk.items()}
        if not rk: out["jnt_ranges_dropped"] = sorted(jr)
    return out


def consumed_by(task, demand: dict) -> dict:
    """Which capacity keys this task honours: tau_scale needs EffectTask.caps (any EffectTask-family task); jnt_ranges needs a plant that
    reads the key (EffectTask / BaseEffectTask own it; ImpactTask's frozen plant does not: the runner applies it, see ensure_ranges)."""
    req = [k for k in KEYS if demand.get(k)]
    if not req: return dict(requested=[])
    tau_ok = hasattr(task, "caps"); jr_native = type(task).plant.__qualname__.split(".")[0] in ("EffectTask", "BaseEffectTask") if hasattr(task, "plant") else False
    return dict(requested=req, tau_scale=("task" if tau_ok else "NOT consumed") if "tau_scale" in req else None, jnt_ranges=("task" if jr_native else ("runner" if hasattr(task, "plant") else "NOT consumed")) if "jnt_ranges" in req else None)


def ensure_ranges(S, demand: dict) -> int:
    """Apply demand["jnt_ranges"] to the plant's model when the task's plant did not (ImpactTask and its subclasses: their cache key
    ignores the ranges). Returns the number of joints changed; S["jnt_ranges_applied"] records who applied them."""
    jr = demand.get("jnt_ranges")
    if not jr: return 0
    import mujoco
    m = S["m"]; n = 0
    for jn, (lo, hi) in jr.items():
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if j < 0: continue
        if abs(float(m.jnt_range[j][0]) - float(lo)) > 1e-9 or abs(float(m.jnt_range[j][1]) - float(hi)) > 1e-9: m.jnt_range[j] = [float(lo), float(hi)]; m.jnt_limited[j] = 1; n += 1
    S["jnt_ranges_applied"] = S.get("jnt_ranges_applied") or ("runner" if n else "task"); return n


def caps_of(task, S, demand: dict) -> Optional[np.ndarray]:
    """The per-type torque factors the rollout will use (EffectTask.caps), for a check line; None when the task has no caps."""
    return task.caps(S, demand) if hasattr(task, "caps") else None
