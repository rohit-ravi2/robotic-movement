"""Disturbance dict factories (MuJoCo-free) + the mocap driver for moving supports/obstacles.
SCHEMA (agreed with physics.disturbance.apply(disturbances, m, d, t)): every dict has
    kind: "impulse" | "force" | "platform_motion" | "slip_patch"
    t0:   start time (s);  t1: end time (s) or None (impulse: a single step; platform_motion: forever)
    vec:  3-vector in the WORLD frame (impulse: N s, applied as F = vec/dt on the step containing t0; force: N, applied for t0<=t<t1)
    body: robot body name the wrench acts on (default "torso_link"; H1 also has "torso_link")
`platform_motion` and `moving_obstacle` motions are KINEMATIC: apply_mocap(m, d, spec, t) sets d.mocap_pos from the world dicts
(the pieces' `motion` field); physics.disturbance.apply may delegate to it. `slip_patch` is NOT a wrench: build() turns it into a
friction patch on the floor (terrain.friction_patches); apply() must ignore kind "slip_patch".
compile check: 2026-10-12 n/a (dict factories); apply_mocap exercised on platform(moving) + moving_obstacle scene ok."""
from __future__ import annotations
import math
import numpy as np

KINDS = ("impulse", "force", "platform_motion", "slip_patch")


def impulse(t: float, vec, body: str = "torso_link") -> dict:
    return dict(kind="impulse", t0=float(t), t1=None, vec=[float(v) for v in vec], body=body)


def force(t0: float, t1: float, vec, body: str = "torso_link") -> dict:
    return dict(kind="force", t0=float(t0), t1=float(t1), vec=[float(v) for v in vec], body=body)


def platform_motion(body: str = "platform", axis=(0, 0, 1), amp: float = 0.05, freq: float = 1.0, phase: float = 0.0, t0: float = 0.0, t1=None) -> dict:
    """Overrides/declares the sinusoidal motion of a mocap support named `body` (also settable in terrain.platform(moving=...))."""
    return dict(kind="platform_motion", t0=float(t0), t1=(None if t1 is None else float(t1)), vec=[float(v)*float(amp) for v in axis], body=body, freq=float(freq), phase=float(phase))


def slip_patch(xy, mu: float, size: float = 0.3) -> dict:
    """A low-friction floor patch (terrain, static): build() adds it; physics.disturbance ignores it."""
    return dict(kind="slip_patch", t0=0.0, t1=None, vec=[0.0, 0.0, 0.0], body="floor", xy=[float(xy[0]), float(xy[1])], mu=float(mu), size=float(size))


def _motion_pos(pos0, motion, t):
    p = np.asarray(pos0, float)
    if "drop" in motion: return p - np.array([0.0, 0.0, motion["drop"]]) if t >= motion["t0"] else p   # trapdoor: step drop at t0
    if "path" in motion:
        path = np.asarray(motion["path"], float); n = len(path)
        if n == 1: return path[0]
        u = (t % motion["period"])/motion["period"]*n; i = int(u) % n; f = u - int(u); return (1 - f)*path[i] + f*path[(i + 1) % n]
    return p + np.asarray(motion["axis"], float)*motion["amp"]*math.sin(2*math.pi*motion["freq"]*t + motion.get("phase", 0.0))


def mocap_targets(spec, t: float) -> dict:
    """body name -> world position at time t for every kinematic piece in the WorldSpec (platform_motion disturbances override)."""
    from .build import kinematic_pieces
    out = {p["name"]: _motion_pos(p["pos"], p["motion"], t) for p in kinematic_pieces(spec)}
    for dd in spec.disturbances:
        if dd["kind"] == "platform_motion" and dd["body"] in out and t >= dd["t0"] and (dd["t1"] is None or t < dd["t1"]):
            base = next(p["pos"] for p in kinematic_pieces(spec) if p["name"] == dd["body"])
            out[dd["body"]] = np.asarray(base, float) + np.asarray(dd["vec"], float)*math.sin(2*math.pi*dd["freq"]*(t - dd["t0"]) + dd["phase"])
    return out


def apply_mocap(m, d, spec, t: float) -> int:
    """Set d.mocap_pos for every kinematic piece at time t (call before each mj_step). Returns the number of bodies moved."""
    import mujoco
    n = 0
    for name, pos in mocap_targets(spec, t).items():
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name); mid = int(m.body_mocapid[bid])
        if mid < 0: raise KeyError(f"{name} is not a mocap body in this model")
        d.mocap_pos[mid] = pos; n += 1
    return n
