"""Rollout hooks for world plants, as ONE ctrl_fn to chain in front of the task's controller (the frozen rollouts call ctrl_fn once
per step, after mj_forward and before mj_step, starting at t = 0):
    world.apply_initial_velocities(m, d, spec)  at the first call of each rollout (objects' qvel0; the rollouts reset from the settle)
    world.apply_mocap(m, d, spec, t)            every step (moving platforms / obstacles)
    physics.disturbance.apply(dists, m, d, t)   every step (impulse / force / torque wrenches; slip_patch and platform_motion are not wrenches)
A new rollout is detected by t falling below the previous call's t. Tasks whose rollout ignores ctrl_fn (balance.simulate behind
StandTask) do not receive these hooks: the runner reports `hooks_applied` so that is visible."""
from __future__ import annotations
from typing import Callable, Optional
import numpy as np
from . import disturbance as D

WRENCH_KINDS = ("impulse", "force", "torque")


def world_hooks(spec, n_out: int, inner: Optional[Callable] = None) -> Callable:
    from movement_discovery import world as W
    kin = bool(W.kinematic_pieces(spec)) if spec is not None else False
    dists = [d for d in (spec.disturbances if spec is not None else []) if d.get("kind") in WRENCH_KINDS]
    if spec is not None and any(o.get("kinematic") for o in spec.objects):   # kinematic=True objects are modelled analytically by the program (never bodies in the plant): no qvel0 to apply
        import dataclasses
        spec = dataclasses.replace(spec, objects=[o for o in spec.objects if not o.get("kinematic")])
    v0 = any(o.get("qvel0") for o in (spec.objects if spec is not None else []))
    if v0:                                        # a 3-vector qvel0 (linear only) is padded to the free joint's 6 dofs
        import dataclasses
        spec = dataclasses.replace(spec, objects=[(dict(o, qvel0=list(o["qvel0"]) + [0.0]*(6 - len(o["qvel0"]))) if (o.get("qvel0") and o["kind"] != "suspended" and len(o["qvel0"]) < 6) else o) for o in spec.objects])
    st = dict(last_t=None, calls=0, mocap=0, wrench_steps=0, v0_sets=0)
    def fn(t, d, S):
        m = S["m"]; st["calls"] += 1
        if st["last_t"] is None or t < st["last_t"]:
            if v0: st["v0_sets"] += W.apply_initial_velocities(m, d, spec)
        st["last_t"] = t
        if kin: st["mocap"] += W.apply_mocap(m, d, spec, t)
        if dists:
            D.apply(dists, m, d, t)
            if np.any(d.xfrc_applied): st["wrench_steps"] += 1
        return inner(t, d, S) if inner is not None else np.zeros(len(S["jarr"]["aid"]) if "jarr" in S else n_out)   # per-ACTUATOR zeros: bilateral plants (jump: 6 types, 11 actuators) differ from len(S["types"])
    fn.stats = st; fn.active = bool(kin or dists or v0); return fn


def chain(outer: Callable, inner: Optional[Callable], n_out: int) -> Callable:
    """outer(t, d, S) runs first (hooks), inner's command is returned (None -> zeros)."""
    def fn(t, d, S):
        outer(t, d, S); return inner(t, d, S) if inner is not None else np.zeros(len(S["jarr"]["aid"]) if "jarr" in S else n_out)
    return fn
