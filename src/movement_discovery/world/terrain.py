"""Terrain factories (MuJoCo-free). Every factory returns a plain dict `dict(kind=..., **params)`; `pieces(t)` expands it into
static (or mocap) geom descriptors that build.py adds on the base scene's floor plane "floor" (z = 0).
Piece schema: dict(name, type="box"|"cylinder", size=half-extents, pos, quat=(w,x,y,z), mu, rgba, priority, motion=None|dict,
slope=None|(x0, angle, length)).  motion = dict(axis, amp, freq, phase) marks a MOCAP body (kinematic support) whose pose
`disturbances.apply_mocap` sets each step; `path`/`period` marks a looping waypoint path.
ROBOT START HEIGHT: the plant's settle (branch_i.settle: `qpos[2] -= lowest_z()`) puts the lowest foot geom at z = 0, i.e. on the
floor PLANE, not on the terrain. build() also raises the keyframes by `start_z_offset(spec)` (terrain top under the robot's
start xy), but a scenario that settles must re-add that offset to pelvis z after settle (or start the robot where the terrain is
flat, e.g. steps with x0 > 0.3). Robot-terrain contact on the G1 (contype 0 everywhere) is by explicit pairs replicated from the
base scene's floor pairs (feet, shins, hands, torso...); G1H/H1 collide by contype.
compile check: 2026-10-12 flat ok; slope ok; steps ok; stairs ok; uneven ok; gap ok; platform(static+moving) ok; narrow_beam ok;
friction_patches ok; composite ok (G1 scene_mjx base)."""
from __future__ import annotations
import math
import numpy as np

RGBA = (0.45, 0.42, 0.38, 1.0)


def flat() -> dict: return dict(kind="flat")


def slope(angle: float, length: float = 3.0, width: float = 2.0, x0: float = 0.4, thick: float = 0.05) -> dict:
    """Incline rising along +x from (x0, z=0) at `angle` rad (negative = decline; the ramp still starts at z = 0 at x0)."""
    return dict(kind="slope", angle=float(angle), length=float(length), width=float(width), x0=float(x0), thick=float(thick))


def steps(n: int, rise: float, run: float, width: float = 2.0, x0: float = 0.4) -> dict:
    """n ascending steps starting at x0; step i spans x in [x0+i*run, x0+(i+1)*run] with top at (i+1)*rise."""
    return dict(kind="steps", n=int(n), rise=float(rise), run=float(run), width=float(width), x0=float(x0))


def stairs(n: int, rise: float, run: float, width: float = 2.0, x0: float = 0.4, landing: float = 0.8, descend: bool = True) -> dict:
    """A full flight: n steps up, a landing, and (descend) n steps down."""
    return dict(kind="stairs", n=int(n), rise=float(rise), run=float(run), width=float(width), x0=float(x0), landing=float(landing), descend=bool(descend))


def uneven(seed: int, n: int = 40, amp: float = 0.06, extent: float = 2.0, size: float = 0.25, clear_radius: float = 0.3) -> dict:
    """Field of n random boxes (half-side `size`, heights U(amp/4, amp)) in [-extent, extent]^2; none within clear_radius of the start."""
    return dict(kind="uneven", seed=int(seed), n=int(n), amp=float(amp), extent=float(extent), size=float(size), clear_radius=float(clear_radius))


def gap(width: float, depth: float = 0.3, x_edge: float = 0.6, slab: float = 2.5, slab_width: float = 2.0) -> dict:
    """Two raised slabs of height `depth` separated by a gap of `width` whose near edge is at x_edge; the robot starts on slab A."""
    return dict(kind="gap", width=float(width), depth=float(depth), x_edge=float(x_edge), slab=float(slab), slab_width=float(slab_width))


def platform(size=(0.5, 0.5), height: float = 0.1, pos=(0.0, 0.0), moving: dict | None = None, name: str = "platform") -> dict:
    """Box support (half-extents size, top at `height`). moving=dict(axis=(x,y,z), amp, freq, phase=0) -> mocap-driven kinematic support."""
    return dict(kind="platform", size=[float(size[0]), float(size[1])], height=float(height), pos=[float(pos[0]), float(pos[1])], moving=(None if moving is None else dict(moving)), name=name)


def narrow_beam(width: float = 0.1, length: float = 2.0, height: float = 0.15, x0: float = 0.4) -> dict:
    return dict(kind="narrow_beam", width=float(width), length=float(length), height=float(height), x0=float(x0))


def friction_patches(patches) -> dict:
    """patches: list of (xy, size, mu) or dicts; each a 5 mm slab (top at z=0.005) with its own friction (priority 2: wins over feet)."""
    out = []
    for p in patches:
        if isinstance(p, dict): out.append(dict(xy=[float(p["xy"][0]), float(p["xy"][1])], size=float(p.get("size", 0.3)), mu=float(p["mu"])))
        else: xy, s, mu = p; out.append(dict(xy=[float(xy[0]), float(xy[1])], size=float(s), mu=float(mu)))
    return dict(kind="friction_patches", patches=out)


def compose(*parts: dict) -> dict:
    """Several terrain features in one spec (e.g. compose(slope(...), friction_patches(...)))."""
    return dict(kind="composite", parts=[dict(p) for p in parts])


KINDS = ("flat", "slope", "steps", "stairs", "uneven", "gap", "platform", "narrow_beam", "friction_patches", "composite")


def _quat_y(theta):
    return [math.cos(theta/2), 0.0, math.sin(theta/2), 0.0]


def _box(name, half, pos, mu=None, quat=(1, 0, 0, 0), rgba=RGBA, slope=None):
    return dict(name=name, type="box", size=[float(v) for v in half], pos=[float(v) for v in pos], quat=[float(v) for v in quat], mu=mu, rgba=list(rgba), priority=0, motion=None, slope=slope)


def pieces(t: dict | None, prefix: str = "ter") -> list:
    """Expand a terrain dict into geom descriptors. Raises NotImplementedError for unknown kinds."""
    if t is None: return []
    k = t["kind"]
    if k == "flat": return []
    if k == "composite": return [p for i, part in enumerate(t["parts"]) for p in pieces(part, f"{prefix}{i}")]
    if k == "slope":
        a, L, w, x0, th = t["angle"], t["length"], t["width"], t["x0"], t["thick"]
        mid = np.array([x0 + 0.5*L*math.cos(a), 0.0, 0.5*L*math.sin(a)]); nrm = np.array([-math.sin(a), 0.0, math.cos(a)])
        return [_box(f"{prefix}_slope", (L/2, w/2, th), mid - th*nrm, quat=_quat_y(-a), slope=[x0, a, L])]
    if k == "steps":
        return [_box(f"{prefix}_step{i}", (t["run"]/2, t["width"]/2, (i + 1)*t["rise"]/2), (t["x0"] + (i + 0.5)*t["run"], 0.0, (i + 1)*t["rise"]/2)) for i in range(t["n"])]
    if k == "stairs":
        n, r, run, w, x0 = t["n"], t["rise"], t["run"], t["width"], t["x0"]; out = pieces(dict(kind="steps", n=n, rise=r, run=run, width=w, x0=x0), prefix)
        xl = x0 + n*run; out.append(_box(f"{prefix}_landing", (t["landing"]/2, w/2, n*r/2), (xl + t["landing"]/2, 0.0, n*r/2)))
        if t["descend"]:
            xd = xl + t["landing"]
            out += [_box(f"{prefix}_down{i}", (run/2, w/2, (n - 1 - i)*r/2), (xd + (i + 0.5)*run, 0.0, (n - 1 - i)*r/2)) for i in range(n - 1)]
        return out
    if k == "uneven":
        rng = np.random.default_rng(t["seed"]); out = []; tries = 0
        while len(out) < t["n"] and tries < 50*t["n"]:
            tries += 1; xy = rng.uniform(-t["extent"], t["extent"], 2); h = rng.uniform(t["amp"]/4, t["amp"])
            if np.linalg.norm(xy) < t["clear_radius"] + t["size"]: continue
            out.append(_box(f"{prefix}_u{len(out)}", (t["size"], t["size"], h/2), (xy[0], xy[1], h/2)))
        return out
    if k == "gap":
        d, xe, L, w = t["depth"], t["x_edge"], t["slab"], t["slab_width"]
        return [_box(f"{prefix}_slabA", (L/2, w/2, d/2), (xe - L/2, 0.0, d/2)), _box(f"{prefix}_slabB", (L/2, w/2, d/2), (xe + t["width"] + L/2, 0.0, d/2))]
    if k == "platform":
        p = _box(t.get("name", "platform"), (t["size"][0], t["size"][1], t["height"]/2), (t["pos"][0], t["pos"][1], t["height"]/2), rgba=(0.3, 0.5, 0.7, 1))
        if t.get("moving"): mv = t["moving"]; p["motion"] = dict(axis=[float(v) for v in mv.get("axis", (0, 0, 1))], amp=float(mv["amp"]), freq=float(mv["freq"]), phase=float(mv.get("phase", 0.0)))
        return [p]
    if k == "narrow_beam":
        return [_box(f"{prefix}_beam", (t["length"]/2, t["width"]/2, t["height"]/2), (t["x0"] + t["length"]/2, 0.0, t["height"]/2))]
    if k == "friction_patches":
        out = []
        for i, p in enumerate(t["patches"]):
            b = _box(f"{prefix}_patch{i}", (p["size"], p["size"], 0.0025), (p["xy"][0], p["xy"][1], 0.0025), mu=p["mu"], rgba=(0.2, 0.2, 0.8, 1)); b["priority"] = 2; out.append(b)
        return out
    raise NotImplementedError(f"terrain kind {k!r} is not implemented (known: {KINDS})")


def height_at(t: dict | None, xy=(0.0, 0.0)) -> float:
    """Terrain top under xy (0 on bare floor). Moving platforms are evaluated at their initial pose."""
    z = 0.0
    for p in pieces(t):
        if p["slope"] is not None:
            x0, a, L = p["slope"]; u = (xy[0] - x0)/math.cos(a)
            if 0 <= u <= L and abs(xy[1]) <= p["size"][1]: z = max(z, u*math.sin(a))
            continue
        if abs(xy[0] - p["pos"][0]) <= p["size"][0] and abs(xy[1] - p["pos"][1]) <= p["size"][1]: z = max(z, p["pos"][2] + p["size"][2])
    return float(z)
