"""Object factories (MuJoCo-free at import). Dicts follow engine.world.obj() conventions (name, kind, size, mass, pos, quat, graspable,
rgba, soft, rigid, geoms) plus: `mu` (geom friction, priority 1 so it beats the floor's), `qvel0` (6-vector initial free-joint
velocity applied by `apply_initial_velocities(m, d, spec)` after reset; NOT applied by the keyframe), `grip_point`/`grip_site`
(body-frame grasp point; build adds a site `<name>_grip`), `com_offset` (payload: box geom shifted from the body origin).
kind "suspended": a pendulum (two hinges x/y at a fixed anchor + massless rod + bob), joints `<name>_hx`, `<name>_hy`.
Robot contact: the G1 body has NO collision bits; objects touch the robot only through the effector collider (tasks/impact.py) or a
grasp (weld `grasp_<name>` inactive equality, or Dex3 fingers on the G1H). Objects DO collide with terrain/obstacles/mechanisms.
compile check: 2026-10-12 ball ok; box ok; cylinder ok; payload_with_handle ok; tool ok; compliant_target ok; suspended_target ok;
rolling_ball ok (qvel0 applied, omega = z x v / r); launched_ball ok; heavy_block ok."""
from __future__ import annotations
import math
import numpy as np


def _quat_z_to(v):
    """Quaternion (w,x,y,z) rotating +z onto unit vector v (numpy replacement for mju_quatZ2Vec)."""
    v = np.asarray(v, float); v = v/np.linalg.norm(v); z = np.array([0.0, 0.0, 1.0]); c = float(np.clip(z @ v, -1, 1))
    if c > 1 - 1e-9: return [1.0, 0.0, 0.0, 0.0]
    if c < -1 + 1e-9: return [0.0, 1.0, 0.0, 0.0]
    ax = np.cross(z, v); ax /= np.linalg.norm(ax); a = math.acos(c); return [math.cos(a/2)] + (math.sin(a/2)*ax).tolist()


def _base(name, kind, size, mass, pos, graspable, rgba, soft=None, quat=(1, 0, 0, 0), rigid=True, mu=None, qvel0=None, **extra):
    d = dict(name=name, kind=kind, size=[float(s) for s in size], mass=float(mass), pos=[float(p) for p in pos], graspable=bool(graspable), rgba=list(rgba), soft=(None if soft is None else float(soft)), quat=[float(q) for q in quat], rigid=bool(rigid), mu=(None if mu is None else float(mu)), qvel0=(None if qvel0 is None else [float(v) for v in qvel0]))
    d.update(extra); return d


def ball(name="ball", radius=0.036, mass=0.145, pos=(0.5, -0.2, 0.46), graspable=True, rgba=(0.9, 0.2, 0.2, 1), **kw) -> dict:
    return _base(name, "ball", (radius,), mass, pos, graspable, rgba, **kw)


def box(name="box", half=(0.05, 0.05, 0.05), mass=0.5, pos=(0.5, -0.2, 0.46), graspable=True, rgba=(0.2, 0.4, 0.9, 1), **kw) -> dict:
    return _base(name, "box", half, mass, pos, graspable, rgba, **kw)


def cylinder(name="cyl", radius=0.03, half_h=0.08, mass=0.3, pos=(0.5, -0.2, 0.46), graspable=True, rgba=(0.3, 0.7, 0.4, 1), **kw) -> dict:
    return _base(name, "cylinder", (radius, half_h), mass, pos, graspable, rgba, **kw)


def payload_with_handle(name="payload", half=(0.12, 0.08, 0.08), mass=3.0, pos=(0.5, 0.0, 0.08), com_offset=(0.0, 0.0, 0.0), handle_r=0.015, handle_len=0.12, handle_mass=0.1, rgba=(0.6, 0.35, 0.15, 1), **kw) -> dict:
    """Box load + a horizontal (y-axis) handle cylinder centred above the box top. com_offset shifts the box geom (hence the COM)
    relative to the body origin / handle for asymmetric loads. pos is the body origin = box centre before the offset."""
    o = _base(name, "compound", half, mass + handle_mass, pos, True, rgba, **kw); co = [float(v) for v in com_offset]
    ht = half[2] + 0.03   # handle 3 cm above the box top so fingers fit under it
    o["geoms"] = [dict(type="box", size=list(half), pos=co, mass=float(mass)), dict(type="cylinder", size=[handle_r, handle_len/2], pos=[0, 0, ht], quat=[math.sqrt(0.5), math.sqrt(0.5), 0, 0], mass=float(handle_mass))]
    o["com_offset"] = co; o["grip_point"] = [0.0, 0.0, ht]; o["handle_axis"] = [0, 1, 0]; o["half_z"] = float(half[2]); return o


def tool(name="tool", head=(0.04, 0.025, 0.015), handle_r=0.015, handle_len=0.16, mass=0.15, pos=(0.3, -0.25, 0.655), lean=None, rgba=(0.2, 0.6, 0.3, 1), **kw) -> dict:
    """Handle (cylinder) + head (box), one free body with origin at the head centre; lean = handle axis in the body frame (None = +z)."""
    o = _base(name, "compound", head, mass, pos, True, rgba, **kw); ax = np.array([0, 0, 1.0]) if lean is None else np.asarray(lean, float)/np.linalg.norm(lean)
    mid = np.array([0, 0, head[2]]) + ax*(handle_len/2)
    o["geoms"] = [dict(type="box", size=list(head), pos=[0, 0, 0], mass=0.8*mass), dict(type="cylinder", size=[handle_r, handle_len/2], pos=mid.tolist(), quat=_quat_z_to(ax), mass=0.2*mass)]
    o["half_z"] = float(head[2]); o["handle_axis"] = ax.tolist(); o["grip_point"] = mid.tolist(); o["grip_tip"] = (np.array([0, 0, head[2]]) + ax*handle_len).tolist(); return o


def compliant_target(name="pad", half=(0.15, 0.15, 0.04), mass=1.0, pos=(0.5, 0.0, 1.0), soft=0.05, graspable=False, rgba=(0.9, 0.7, 0.2, 1), **kw) -> dict:
    """Soft box (contact time constant `soft` s via the engine `soft` convention: solref [soft, 1], priority 1)."""
    return _base(name, "box", half, mass, pos, graspable, rgba, soft=soft, rigid=False, **kw)


def suspended_target(name="bag", anchor=(0.6, 0.0, 1.8), length=0.8, radius=0.1, mass=2.0, damping=0.05, rgba=(0.8, 0.3, 0.3, 1), qvel0=None) -> dict:
    """Bob of `mass` hung from a fixed anchor by a massless rod on two hinges (x then y). Joints `<name>_hx`, `<name>_hy` (padded with
    0 in keyframes). qvel0 = (wx, wy) initial hinge rates. Not a free body; graspable=False."""
    return dict(name=name, kind="suspended", anchor=[float(v) for v in anchor], length=float(length), size=[float(radius)], mass=float(mass), damping=float(damping), rgba=list(rgba), graspable=False, pos=[anchor[0], anchor[1], anchor[2] - length], joints=[f"{name}_hx", f"{name}_hy"], qvel0=(None if qvel0 is None else [float(v) for v in qvel0]))


def rolling_ball(name="ball", radius=0.11, mass=0.43, pos=(2.0, 0.0, 0.11), vel=(-2.0, 0.0, 0.0), graspable=False, rgba=(0.9, 0.9, 0.9, 1), **kw) -> dict:
    """Ball on the floor with an initial linear velocity and the matching no-slip spin (omega = z x v / r)."""
    v = np.asarray(vel, float); w = np.cross([0, 0, 1.0], v)/float(radius)
    return ball(name, radius, mass, pos, graspable, rgba, qvel0=list(v) + list(w), **kw)


def launched_ball(pos, vel, name="ball", radius=0.05, mass=0.1, graspable=False, rgba=(0.9, 0.5, 0.1, 1), **kw) -> dict:
    """Airborne ball with initial velocity `vel` (m/s) at `pos`; no spin. For interception tasks."""
    return ball(name, radius, mass, pos, graspable, rgba, qvel0=list(vel) + [0.0, 0.0, 0.0], **kw)


def heavy_block(mass=20.0, mu=0.6, name="block", half=(0.2, 0.2, 0.2), pos=(0.7, 0.0, 0.2), graspable=False, rgba=(0.35, 0.35, 0.4, 1), **kw) -> dict:
    """Massive box with its own floor friction (geom priority 1 so `mu` overrides the floor's)."""
    return box(name, half, mass, pos, graspable, rgba, mu=mu, **kw)


def apply_initial_velocities(m, d, spec) -> int:
    """Set d.qvel for every object carrying qvel0 (free bodies: 6 dofs at joint `<name>_free`; suspended: the two hinges). Call after
    mj_resetDataKeyframe / the plant's settle, before the first mj_step. Returns the number of objects set."""
    import mujoco
    n = 0
    for o in spec.objects:
        if not o.get("qvel0"): continue
        if o["kind"] == "suspended":
            for jn, w in zip(o["joints"], o["qvel0"]): d.qvel[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]] = w
        else:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, o["name"] + "_free")
            if jid < 0: raise KeyError(f"object {o['name']} has no free joint in this model")
            a = int(m.jnt_dofadr[jid]); d.qvel[a:a + 6] = np.asarray(o["qvel0"], float)
        n += 1
    if n: mujoco.mj_forward(m, d)
    return n
