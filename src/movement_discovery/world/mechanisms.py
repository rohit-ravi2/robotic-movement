"""Articulated mechanisms: factories (MuJoCo-free) + `add(ctx, mech)` (MuJoCo inside) used by build.py.
Every factory dict exposes: kind, name, pos, params, and the names tasks measure: `joint` (hinge/slide, qpos padded with 0 in
keyframes = closed/rest), `body` (the moving link), `handle_geom`, `handle_site` (site at the handle centre, on the moving link),
`open_sign` (door: multiply the joint angle by this to get the opening angle >= 0), `range`.
Grasp: an INACTIVE weld `grasp_<name>` between the robot hand body and the moving link (engine.world convention: World.grasp
activates it) unless graspable=False; Dex3 fingers (G1H) close on the handle geom directly. The G1 hand/shin/torso capsules touch
mechanism geoms through replicated floor pairs (pushing a door with the body works; grazing a 1.5 cm handle needs the grasp).
Simplifications: door handle = fixed lever cylinder (no latch dof); drawer = box sliding inside a solid cabinet (contacts between
them excluded); crank = arm + axial handle on a post; lever = bar on a stand, `load` kg box at the far end.
compile check: 2026-10-12 door(left/right) ok; drawer ok; crank(x and z axis) ok; lever ok; trapdoor(left/right) ok (G1, G1H, H1; mocap drop verified)."""
from __future__ import annotations
import math


def door(width=0.8, height=2.0, hinge_side="left", handle_height=1.0, resist=2.0, pos=(1.0, 0.0), max_open=1.57, thick=0.04, frictionloss=0.0, spring=0.0, name="door", graspable=True) -> dict:
    """Panel on a z hinge at the `hinge_side` edge (left = +y, facing +x), swinging towards +x (away from the robot). resist = joint
    damping (N m s/rad); frictionloss = dry friction; spring = closer stiffness (N m/rad)."""
    return dict(kind="door", name=name, pos=[float(pos[0]), float(pos[1])], width=float(width), height=float(height), hinge_side=hinge_side, handle_height=float(handle_height), resist=float(resist), max_open=float(max_open), thick=float(thick), frictionloss=float(frictionloss), spring=float(spring), graspable=bool(graspable),
                joint=f"{name}_hinge", body=name, handle_geom=f"{name}_handle", handle_site=f"{name}_handle_site", open_sign=(1.0 if hinge_side == "left" else -1.0), range=[0.0, float(max_open)])


def drawer(travel=0.3, resist=5.0, pos=(0.9, 0.0), height=0.7, half=(0.2, 0.2, 0.08), mass=2.0, frictionloss=0.0, name="drawer", graspable=True) -> dict:
    """Box drawer (half-extents) sliding out towards -x (the robot) from a solid cabinet at `pos`; the drawer top sits at `height`."""
    return dict(kind="drawer", name=name, pos=[float(pos[0]), float(pos[1])], travel=float(travel), resist=float(resist), height=float(height), half=[float(v) for v in half], mass=float(mass), frictionloss=float(frictionloss), graspable=bool(graspable),
                joint=f"{name}_slide", body=name, handle_geom=f"{name}_handle", handle_site=f"{name}_handle_site", open_sign=1.0, range=[0.0, float(travel)])


def crank(radius=0.15, axis=(1, 0, 0), resist=1.0, pos=(0.7, 0.0), height=1.0, load=0.0, handle_len=0.1, mass=0.5, name="crank", graspable=True) -> dict:
    """Arm of `radius` turning about `axis` (world frame) at the top of a post of `height`; handle parallel to the axis at the arm tip.
    resist = damping; load = dry friction torque (N m) to overcome. Unlimited rotation (angle accumulates)."""
    return dict(kind="crank", name=name, pos=[float(pos[0]), float(pos[1])], radius=float(radius), axis=[float(v) for v in axis], resist=float(resist), height=float(height), load=float(load), handle_len=float(handle_len), mass=float(mass), graspable=bool(graspable),
                joint=f"{name}_hinge", body=name, handle_geom=f"{name}_handle", handle_site=f"{name}_handle_site", open_sign=1.0, range=None)


def lever(length=0.8, pivot=0.5, resist=0.5, load=2.0, pos=(0.9, 0.0), height=0.9, max_angle=0.6, bar_mass=1.0, name="lever", graspable=True) -> dict:
    """Bar of `length` on a y hinge at fraction `pivot` from the handle end (handle end faces the robot, -x); `load` kg box at the far end.
    Pushing the handle DOWN lifts the load (positive joint angle = handle down)."""
    return dict(kind="lever", name=name, pos=[float(pos[0]), float(pos[1])], length=float(length), pivot=float(pivot), resist=float(resist), load=float(load), height=float(height), max_angle=float(max_angle), bar_mass=float(bar_mass), graspable=bool(graspable),
                joint=f"{name}_hinge", body=name, handle_geom=f"{name}_handle", handle_site=f"{name}_handle_site", open_sign=1.0, range=[-float(max_angle), float(max_angle)])


def trapdoor(foot="left", t0=0.5, drop=0.1, size=(0.16, 0.09), height=0.05, xy=None, name="trapdoor") -> dict:
    """Partial support loss: two 5 cm pads under the feet (robot stands on them: start_z_offset = height); the pad under `foot`
    is a MOCAP slab that drops by `drop` m at t0 (apply_mocap). xy = pad centre (default: the G1/G1H foot site at the keyframes,
    (0.03, +-0.119)). Keys: kind, body (the dropping pad's mocap body), fixed_body, t0, drop, foot. No joints, no grasp."""
    s = 1.0 if foot == "left" else -1.0; xy = [0.03, s*0.119] if xy is None else [float(xy[0]), float(xy[1])]
    return dict(kind="trapdoor", name=name, body=name, fixed_body=f"{name}_fixed", foot=foot, t0=float(t0), drop=float(drop), size=[float(size[0]), float(size[1])], height=float(height), xy=xy, graspable=False)


def trapdoor_pieces(mech: dict) -> list:
    from .terrain import _box
    h, (sx, sy), (x, y) = mech["height"], mech["size"], mech["xy"]
    mv = _box(mech["body"], (sx, sy, h/2), (x, y, h/2), rgba=(0.8, 0.3, 0.2, 1)); mv["motion"] = dict(drop=mech["drop"], t0=mech["t0"])
    return [mv, _box(mech["fixed_body"], (sx, sy, h/2), (x, -y, h/2), rgba=(0.3, 0.5, 0.7, 1))]


KINDS = ("door", "drawer", "crank", "lever")
STATIC_KINDS = ("trapdoor",)
_QY90 = [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]   # cylinder axis z -> x
_QX90 = [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]   # cylinder axis z -> y


def add(ctx, mech: dict) -> None:
    """Add one mechanism to ctx.spec (see build._Ctx). Raises NotImplementedError for unknown kinds."""
    import mujoco
    k = mech["kind"]; n = mech["name"]; x, y = mech["pos"]; B = mujoco.mjtGeom.mjGEOM_BOX; C = mujoco.mjtGeom.mjGEOM_CYLINDER; H = mujoco.mjtJoint.mjJNT_HINGE
    if k == "door":
        w, h, th, s = mech["width"], mech["height"], mech["thick"], (1.0 if mech["hinge_side"] == "left" else -1.0)
        yh = y + s*w/2                                                   # hinge edge; the panel extends towards -s
        for i, yp in enumerate((yh + s*0.06, yh - s*(w + 0.06))): ctx.world_geom(ctx.spec.worldbody, f"{n}_post{i}", B, [0.05, 0.05, h/2 + 0.06], [x, yp, h/2 + 0.06], rgba=(0.5, 0.4, 0.3, 1))   # 1 cm clearance to the panel
        ctx.world_geom(ctx.spec.worldbody, f"{n}_lintel", B, [0.05, w/2 + 0.11, 0.05], [x, y, h + 0.06], rgba=(0.5, 0.4, 0.3, 1))
        b = ctx.spec.worldbody.add_body(name=n, pos=[x, yh, 0.01])   # 1 cm floor clearance
        rng = [0.0, mech["max_open"]] if s > 0 else [-mech["max_open"], 0.0]
        ctx.joint(b, mech["joint"], H, [0, 0, 1], rng, mech["resist"], mech["frictionloss"], stiffness=mech["spring"])
        ctx.world_geom(b, f"{n}_panel", B, [th/2, w/2, h/2], [0.0, -s*w/2, h/2], rgba=(0.7, 0.55, 0.35, 1), mass=15.0*w*h/1.6)
        hp = [-th/2 - 0.05, -s*(w - 0.08), mech["handle_height"]]   # lever handle 5 cm proud of the robot-side face, 8 cm from the free edge
        ctx.world_geom(b, mech["handle_geom"], C, [0.012, 0.05, 0], hp, quat=_QY90, rgba=(0.8, 0.8, 0.2, 1), mass=0.2)
        b.add_site(name=mech["handle_site"], pos=hp, size=[0.01, 0, 0])
    elif k == "drawer":
        hx, hy, hz = mech["half"]; z = mech["height"] - hz; cab = [hx + 0.03, hy + 0.03, hz + 0.03]
        ctx.world_geom(ctx.spec.worldbody, f"{n}_cabinet", B, cab, [x + 0.03, y, z], rgba=(0.5, 0.4, 0.3, 1))
        ctx.world_geom(ctx.spec.worldbody, f"{n}_cabinet_base", B, [cab[0], cab[1], (z - cab[2])/2], [x + 0.03, y, (z - cab[2])/2], rgba=(0.5, 0.4, 0.3, 1))
        b = ctx.spec.worldbody.add_body(name=n, pos=[x, y, z]); ctx.joint(b, mech["joint"], mujoco.mjtJoint.mjJNT_SLIDE, [-1, 0, 0], [0.0, mech["travel"]], mech["resist"], mech["frictionloss"])
        ctx.world_geom(b, f"{n}_box", B, [hx, hy, hz], [0, 0, 0], rgba=(0.7, 0.55, 0.35, 1), mass=mech["mass"])
        hp = [-hx - 0.04, 0.0, 0.0]; ctx.world_geom(b, mech["handle_geom"], C, [0.012, 0.06, 0], hp, quat=_QX90, rgba=(0.8, 0.8, 0.2, 1), mass=0.1)
        b.add_site(name=mech["handle_site"], pos=hp, size=[0.01, 0, 0]); ctx.exclude_world(n)   # the drawer sits inside the solid cabinet: no drawer-vs-static contacts (robot pairs still apply)
    elif k == "crank":
        r, hgt, ax = mech["radius"], mech["height"], mech["axis"]
        ctx.world_geom(ctx.spec.worldbody, f"{n}_post", C, [0.03, hgt/2, 0], [x, y, hgt/2], rgba=(0.4, 0.4, 0.4, 1))
        b = ctx.spec.worldbody.add_body(name=n, pos=[x, y, hgt + 0.05]); ctx.joint(b, mech["joint"], H, ax, None, mech["resist"], mech["load"])   # hub 5 cm above the post top
        from .objects import _quat_z_to
        import numpy as np
        a = np.asarray(ax, float)/np.linalg.norm(ax); perp = np.cross(a, [0, 0, 1.0]); perp = np.cross(a, [0, 1.0, 0]) if np.linalg.norm(perp) < 1e-6 else perp; perp /= np.linalg.norm(perp)
        ctx.world_geom(b, f"{n}_arm", C, [0.012, r/2, 0], (perp*r/2).tolist(), quat=_quat_z_to(perp), rgba=(0.4, 0.4, 0.4, 1), mass=mech["mass"])
        hp = (perp*r - a*(0.03 + mech["handle_len"]/2)).tolist()        # handle on the robot-facing side of the arm (axis points away from the robot by default: -x side)
        ctx.world_geom(b, mech["handle_geom"], C, [0.012, mech["handle_len"]/2, 0], hp, quat=_quat_z_to(a), rgba=(0.8, 0.8, 0.2, 1), mass=0.1)
        b.add_site(name=mech["handle_site"], pos=hp, size=[0.01, 0, 0])
    elif k == "lever":
        L, p, hgt = mech["length"], mech["pivot"], mech["height"]
        ctx.world_geom(ctx.spec.worldbody, f"{n}_stand", B, [0.03, 0.03, hgt/2], [x, y, hgt/2], rgba=(0.4, 0.4, 0.4, 1))
        b = ctx.spec.worldbody.add_body(name=n, pos=[x, y, hgt + 0.03]); ctx.joint(b, mech["joint"], H, [0, -1, 0], [-mech["max_angle"], mech["max_angle"]], mech["resist"], 0.0)
        ctx.world_geom(b, f"{n}_bar", B, [L/2, 0.02, 0.015], [(0.5 - p)*L, 0, 0], rgba=(0.6, 0.6, 0.65, 1), mass=mech["bar_mass"])
        lh = 0.5*(mech["load"]/800.0)**(1/3) if mech["load"] > 0 else 0.03
        if mech["load"] > 0: ctx.world_geom(b, f"{n}_load", B, [lh, lh, lh], [(1 - p)*L - lh, 0, lh + 0.015], rgba=(0.35, 0.35, 0.4, 1), mass=mech["load"])
        hp = [-p*L + 0.05, 0.0, 0.0]; ctx.world_geom(b, mech["handle_geom"], C, [0.012, 0.05, 0], hp, quat=_QX90, rgba=(0.8, 0.8, 0.2, 1), mass=0.05)
        b.add_site(name=mech["handle_site"], pos=hp, size=[0.01, 0, 0])
    else:
        raise NotImplementedError(f"mechanism kind {k!r} is not implemented (known: {KINDS})")
    if mech.get("graspable", True): ctx.grasp_weld(mech["body"])
