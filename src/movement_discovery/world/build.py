"""build(spec, robot) -> MJCF path: composes terrain, objects, mechanisms, obstacles and static slip patches onto the robot's base
scene with MjSpec, pads every keyframe for the added joints, and caches the file next to the robot assets as
`scene_md_<robot>_<spec.hash()>_<base6>v<VERSION>.xml` (base6 = hash of the base scene file, VERSION = this builder's: neither a
changed base nor a changed builder reuses a stale scene).
Routing: spec.terrain (one terrain dict, use terrain.compose for several), spec.objects (objects.* dicts or engine.world.obj),
spec.surfaces (engine.world.surface), spec.mechanisms (mechanisms.* AND obstacles.* dicts, routed by kind), spec.disturbances
(only kind "slip_patch" is built; the rest are runtime wrenches/motions), spec.friction (floor + floor pairs), spec.gravity.
Robot contact on the G1 (every robot geom contype 0): each world geom gets explicit pairs replicated from the base scene's
floor pairs (same robot geoms, solref, condim; friction = the piece's mu or the floor's). G1H/H1 collide by contype.
ROBOT START: the plant's settle puts the lowest foot on the floor plane; keyframes here are raised by start_z_offset(spec) but a
scenario that settles must re-add that offset (see terrain.py).
compile check: 2026-10-12 all 36 factory scenes (results/md_build/world_check.py) compile + mj_forward on G1, G1H and H1; 0 failures."""
from __future__ import annotations
import hashlib, json, os
import numpy as np
from movement_discovery.core.contracts import WorldSpec, Capability
from . import terrain as T, objects as O, mechanisms as M, obstacles as OB
from .disturbances import KINDS as DKINDS

VERSION = 3          # bump when geometry/contact logic changes: it is part of the cache file name
RIGID = dict(solref=[0.002, 1.0], solimp=[0.99, 0.999, 0.001, 0.5, 2.0])


def _pieces(spec: WorldSpec) -> list:
    out = list(T.pieces(spec.terrain))
    for mech in spec.mechanisms:
        if mech["kind"] in OB.KINDS: out += OB.pieces(mech)
        elif mech["kind"] == "trapdoor": out += M.trapdoor_pieces(mech)
        elif mech["kind"] not in M.KINDS: raise NotImplementedError(f"mechanism/obstacle kind {mech['kind']!r} (known: {M.KINDS + M.STATIC_KINDS + OB.KINDS})")
    patches = [dict(xy=dd["xy"], size=dd["size"], mu=dd["mu"]) for dd in spec.disturbances if dd["kind"] == "slip_patch"]
    if patches: out += T.pieces(T.friction_patches(patches), "slip")
    for dd in spec.disturbances:
        if dd["kind"] not in DKINDS: raise NotImplementedError(f"disturbance kind {dd['kind']!r} (known: {DKINDS})")
    return out


def kinematic_pieces(spec: WorldSpec) -> list:
    return [p for p in _pieces(spec) if p.get("motion")]


FOOTPRINT = ((0.0, 0.0), (0.03, 0.119), (0.03, -0.119))     # pelvis + G1 foot sites at the keyframes


def start_z_offset(spec: WorldSpec, xy=None) -> float:
    """Height of the terrain/obstacle top under the robot's start footprint (max over FOOTPRINT, or the given xy; 0 on bare floor):
    add to pelvis z after the plant's settle."""
    if xy is None: return max(start_z_offset(spec, q) for q in FOOTPRINT)
    z = 0.0
    for p in _pieces(spec):
        if p.get("slope") is not None:
            x0, a, L = p["slope"]; u = (xy[0] - x0)/np.cos(a)
            if 0 <= u <= L and abs(xy[1]) <= p["size"][1]: z = max(z, u*np.sin(a))
        elif abs(xy[0] - p["pos"][0]) <= p["size"][0] and abs(xy[1] - p["pos"][1]) <= p["size"][1]: z = max(z, p["pos"][2] + p["size"][2])
    return float(z)


class _Ctx:
    def __init__(self, spec, robot, hand):
        import mujoco
        self.mj = mujoco; self.spec = spec; self.robot = robot; self.hand = hand; self.pad_q = []; self.pad_v = 0; self.ngeom = 0
        self.floor_pairs = [p for p in spec.pairs if "floor" in (p.geomname1, p.geomname2)]
    def world_geom(self, parent, name, gtype, size, pos, quat=(1, 0, 0, 0), mu=None, rgba=(0.5, 0.5, 0.5, 1), mass=0.0, priority=0, robot_contact=True, soft=None):
        g = parent.add_geom(name=name, type=gtype, size=list(size) + [0]*(3 - len(size)), pos=list(pos), quat=list(quat), mass=float(mass), rgba=list(rgba)); g.contype = 1; g.conaffinity = 1; self.ngeom += 1
        if soft: g.solref = [float(soft), 1.0]; g.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]; g.priority = max(priority, 1)
        else: g.solref = RIGID["solref"]; g.solimp = RIGID["solimp"]; g.priority = priority
        if mu is not None: g.friction = [float(mu), 0.005, 0.0001]; g.priority = max(int(g.priority), 1)
        if robot_contact: self.robot_pairs(name, mu)
        return g
    def robot_pairs(self, gname, mu=None):
        for p in self.floor_pairs:
            rg = p.geomname1 if p.geomname2 == "floor" else p.geomname2; q = self.spec.add_pair(geomname1=rg, geomname2=gname); q.solref = list(p.solref); q.condim = int(p.condim); fr = list(p.friction)
            if mu is not None: fr[0] = fr[1] = float(mu)
            q.friction = fr
    def joint(self, body, name, jtype, axis, rng, damping, frictionloss=0.0, q0=0.0, stiffness=0.0):
        j = body.add_joint(name=name, type=jtype, axis=list(axis), damping=float(damping), frictionloss=float(frictionloss)); j.stiffness = float(stiffness)
        if rng is not None: j.range = list(rng); j.limited = self.mj.mjtLimited.mjLIMITED_TRUE
        self.pad_q.append(float(q0)); self.pad_v += 1; return j
    def free(self, body, name, pos, quat): body.add_freejoint(name=name); self.pad_q += list(pos) + list(quat); self.pad_v += 6
    def grasp_weld(self, bodyname):
        e = self.spec.add_equality(name=f"grasp_{bodyname}", type=self.mj.mjtEq.mjEQ_WELD, objtype=self.mj.mjtObj.mjOBJ_BODY, name1=self.hand, name2=bodyname); e.active = False
    def exclude_world(self, bodyname): self.spec.add_exclude(name=f"x_{bodyname}", bodyname1="world", bodyname2=bodyname)


def _add_piece(ctx, p):
    mj = ctx.mj; kinds = dict(box=mj.mjtGeom.mjGEOM_BOX, cylinder=mj.mjtGeom.mjGEOM_CYLINDER, sphere=mj.mjtGeom.mjGEOM_SPHERE)
    if p.get("motion"):
        b = ctx.spec.worldbody.add_body(name=p["name"], mocap=True, pos=list(p["pos"]), quat=list(p["quat"]))
        ctx.world_geom(b, p["name"] + "_geom", kinds[p["type"]], p["size"], [0, 0, 0], mu=p.get("mu"), rgba=p["rgba"], priority=p.get("priority", 0))
    else: ctx.world_geom(ctx.spec.worldbody, p["name"], kinds[p["type"]], p["size"], p["pos"], quat=p["quat"], mu=p.get("mu"), rgba=p["rgba"], priority=p.get("priority", 0))


def _add_object(ctx, o):
    mj = ctx.mj; kinds = dict(ball=mj.mjtGeom.mjGEOM_SPHERE, box=mj.mjtGeom.mjGEOM_BOX, cylinder=mj.mjtGeom.mjGEOM_CYLINDER, capsule=mj.mjtGeom.mjGEOM_CAPSULE)
    if o["kind"] == "suspended":
        a = ctx.spec.worldbody.add_body(name=o["name"] + "_anchor", pos=list(o["anchor"])); a.add_geom(name=o["name"] + "_anchor_geom", type=mj.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0], mass=0.0, contype=0, conaffinity=0, rgba=[0.3, 0.3, 0.3, 1])
        b = a.add_body(name=o["name"], pos=[0, 0, 0]); ctx.joint(b, o["joints"][0], mj.mjtJoint.mjJNT_HINGE, [1, 0, 0], None, o["damping"]); ctx.joint(b, o["joints"][1], mj.mjtJoint.mjJNT_HINGE, [0, 1, 0], None, o["damping"])
        L, r = o["length"], o["size"][0]
        b.add_geom(name=o["name"] + "_rod", type=mj.mjtGeom.mjGEOM_CYLINDER, size=[0.005, (L - r)/2, 0], pos=[0, 0, -(L - r)/2], mass=0.0, contype=0, conaffinity=0, rgba=[0.3, 0.3, 0.3, 1])
        g = b.add_geom(name=o["name"] + "_geom", type=mj.mjtGeom.mjGEOM_SPHERE, size=[r, 0, 0], pos=[0, 0, -L], mass=float(o["mass"]), rgba=list(o["rgba"])); g.solref = RIGID["solref"]; g.solimp = RIGID["solimp"]
        b.add_site(name=o["name"] + "_grip", pos=[0, 0, -L], size=[0.01, 0, 0]); return
    b = ctx.spec.worldbody.add_body(name=o["name"], pos=list(o["pos"]), quat=list(o.get("quat", [1, 0, 0, 0]))); ctx.free(b, o["name"] + "_free", o["pos"], o.get("quat", [1, 0, 0, 0]))
    glist = o.get("geoms") or [dict(type=o["kind"], size=o["size"], pos=[0, 0, 0])]; mass_each = o["mass"]/len(glist)
    for gi, gd in enumerate(glist):
        g = b.add_geom(name=o["name"] + ("_geom" if gi == 0 else f"_geom{gi}"), type=kinds[gd["type"]], size=list(gd["size"]) + [0]*(3 - len(gd["size"])), pos=gd.get("pos", [0, 0, 0]), quat=gd.get("quat", [1, 0, 0, 0]), mass=float(gd.get("mass", mass_each)), rgba=list(o["rgba"]))
        if gd.get("soft"): g.solref = [float(gd["soft"]), 1.0]; g.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]; g.priority = 1
        elif o.get("soft"): g.solref = [float(o["soft"]), 1.0]; g.priority = 1
        elif o.get("rigid", True): g.solref = RIGID["solref"]; g.solimp = RIGID["solimp"]
        if o.get("mu") is not None: g.friction = [float(o["mu"]), 0.005, 0.0001]; g.priority = max(int(g.priority), 1)
    if o.get("grip_point"): b.add_site(name=o["name"] + "_grip", pos=list(o["grip_point"]), size=[0.01, 0, 0])
    if o.get("graspable"): ctx.grasp_weld(o["name"])


def build(spec: WorldSpec, robot: str = "G1") -> str:
    """Compile the WorldSpec onto `robot`'s base scene; returns the cached MJCF path ("" for an empty spec = the bare base scene)."""
    import mujoco
    from engine import world as W
    if not (spec.objects or spec.surfaces or spec.mechanisms or spec.terrain or spec.disturbances or spec.friction or spec.gravity): return ""
    base_h = hashlib.sha256(open(W.BASE[robot], "rb").read()).hexdigest()[:6]
    path = f"{W.SCENE_DIRS[robot]}/scene_md_{robot}_{spec.hash()}_{base_h}v{VERSION}.xml"
    if os.path.exists(path): return path
    pieces = _pieces(spec)                                            # validates kinds before touching MuJoCo
    ms = mujoco.MjSpec.from_file(W.BASE[robot]); ctx = _Ctx(ms, robot, W.HAND[robot])
    if spec.gravity: ms.option.gravity = list(spec.gravity)
    if spec.friction is not None:
        for p in ctx.floor_pairs: fr = list(p.friction); fr[0] = fr[1] = float(spec.friction); p.friction = fr
        for g in ms.worldbody.geoms:
            if g.name == "floor": g.friction = [float(spec.friction), 0.005, 0.0001]; g.priority = 2
    for s in spec.surfaces:
        b = ms.worldbody.add_body(name=s["name"], pos=list(s["pos"])); ctx.world_geom(b, s["name"] + "_top", mujoco.mjtGeom.mjGEOM_BOX, s["size"], [0, 0, 0], rgba=s["rgba"], robot_contact=False)   # engine.world convention: massless slab, no robot pairs
    for p in pieces: _add_piece(ctx, p)
    for o in spec.objects: _add_object(ctx, o)
    for mech in spec.mechanisms:
        if mech["kind"] in M.KINDS: M.add(ctx, mech)
    dz = start_z_offset(spec)
    for k in ms.keys:
        q = list(k.qpos); q[2] += dz; k.qpos = q + ctx.pad_q
        if len(k.qvel): k.qvel = list(k.qvel) + [0.0]*ctx.pad_v
    ms.compile(); open(path, "w").write(ms.to_xml()); return path


def describe(spec: WorldSpec) -> str:
    """One-line ledger description."""
    parts = []
    if spec.terrain: parts.append("terrain=" + (",".join(p["kind"] for p in spec.terrain["parts"]) if spec.terrain["kind"] == "composite" else spec.terrain["kind"]))
    if spec.objects: parts.append("objects=" + ",".join(f"{o['name']}:{o['kind']}" + ("(v0)" if o.get("qvel0") else "") for o in spec.objects))
    if spec.surfaces: parts.append("surfaces=" + ",".join(s["name"] for s in spec.surfaces))
    if spec.mechanisms: parts.append("mechanisms=" + ",".join(f"{m['name']}:{m['kind']}" for m in spec.mechanisms))
    if spec.disturbances: parts.append("disturbances=" + ",".join(f"{d['kind']}@{d['t0']}" for d in spec.disturbances))
    if spec.friction is not None: parts.append(f"mu={spec.friction}")
    if spec.gravity: parts.append(f"g={list(spec.gravity)}")
    return f"world[{spec.hash()}] " + (" ".join(parts) or "bare") + (f" start_dz={start_z_offset(spec):.3f}" if start_z_offset(spec) else "")


def required_capabilities(spec: WorldSpec) -> list:
    """List of alternatives: each entry is a tuple of Capabilities of which the embodiment must offer at least ONE.
    graspable object/mechanism handle -> (FINGER_GRASP, WELD_GRASP); non-graspable free object -> (EFFECTOR_COLLIDER, BODY_COLLISION)
    (the G1 body cannot touch it otherwise); terrain, obstacles, moving platforms, disturbances -> nothing (feet pairs are built in)."""
    req = []
    if any(o.get("graspable") for o in spec.objects) or any(m.get("graspable", True) for m in spec.mechanisms if m["kind"] in M.KINDS): req.append((Capability.FINGER_GRASP, Capability.WELD_GRASP))
    if any(not o.get("graspable") and o["kind"] != "suspended" for o in spec.objects): req.append((Capability.EFFECTOR_COLLIDER, Capability.BODY_COLLISION))
    if any(o["kind"] == "suspended" for o in spec.objects): req.append((Capability.EFFECTOR_COLLIDER, Capability.BODY_COLLISION))
    return req
