"""Obstacle factories (MuJoCo-free): static geoms the robot must negotiate, expanded by `pieces(o)` into the terrain piece schema
(see terrain.py) so build.py treats them exactly like terrain (contype 1, rigid, robot floor-pairs replicated on the G1).
`moving_obstacle` is a MOCAP body following a looping waypoint path (piece["motion"] = dict(path, period); pose set by
disturbances.apply_mocap). WorldSpec has no obstacle field: put obstacle dicts in WorldSpec.mechanisms (build.py routes by kind).
Robot start xy = (0, 0), facing +x.
compile check: 2026-10-12 barrier ok; low_clearance ok; corridor ok; opening ok; moving_obstacle ok."""
from __future__ import annotations
from .terrain import _box

KINDS = ("barrier", "low_clearance", "corridor", "opening", "moving_obstacle")


def barrier(height=0.3, width=1.5, x=0.8, thick=0.05, name="barrier") -> dict:
    """Wall across the path (normal +x) from the floor to `height`."""
    return dict(kind="barrier", name=name, height=float(height), width=float(width), x=float(x), thick=float(thick))


def low_clearance(height=1.0, depth=0.6, width=1.5, x=0.8, slab=0.1, name="lowbar") -> dict:
    """Overhead slab whose underside is at `height`, spanning `depth` along x (crawl/duck under)."""
    return dict(kind="low_clearance", name=name, height=float(height), depth=float(depth), width=float(width), x=float(x), slab=float(slab))


def corridor(width=0.6, length=3.0, x0=0.3, wall_h=1.5, thick=0.05, name="corridor") -> dict:
    """Two parallel walls (inner faces at y = +-width/2) from x0 to x0+length."""
    return dict(kind="corridor", name=name, width=float(width), length=float(length), x0=float(x0), wall_h=float(wall_h), thick=float(thick))


def opening(size=(0.6, 1.2), height=0.3, x=0.8, wall_w=2.0, wall_h=2.2, thick=0.05, name="opening") -> dict:
    """Wall with a rectangular hole: size = (width, height) of the hole, `height` = sill top (0 = floor-level doorway)."""
    return dict(kind="opening", name=name, size=[float(size[0]), float(size[1])], height=float(height), x=float(x), wall_w=float(wall_w), wall_h=float(wall_h), thick=float(thick))


def moving_obstacle(path, period=4.0, half=(0.15, 0.15, 0.4), name="mover") -> dict:
    """Box (half-extents, origin at its centre) that loops through `path` waypoints (xyz) with linear interpolation over `period` s."""
    return dict(kind="moving_obstacle", name=name, path=[[float(v) for v in p] for p in path], period=float(period), half=[float(v) for v in half])


def pieces(o: dict) -> list:
    k, n = o["kind"], o["name"]; W = (0.6, 0.3, 0.3, 1)
    if k == "barrier": return [_box(n, (o["thick"]/2, o["width"]/2, o["height"]/2), (o["x"], 0.0, o["height"]/2), rgba=W)]
    if k == "low_clearance":
        h, s, w = o["height"], o["slab"], o["width"]
        return [_box(f"{n}_slab", (o["depth"]/2, w/2, s/2), (o["x"] + o["depth"]/2, 0.0, h + s/2), rgba=W)] + [_box(f"{n}_post{i}", (o["depth"]/2, 0.05, (h + s)/2), (o["x"] + o["depth"]/2, sy*(w/2 + 0.05), (h + s)/2), rgba=W) for i, sy in enumerate((1, -1))]
    if k == "corridor":
        return [_box(f"{n}_wall{i}", (o["length"]/2, o["thick"]/2, o["wall_h"]/2), (o["x0"] + o["length"]/2, sy*(o["width"]/2 + o["thick"]/2), o["wall_h"]/2), rgba=W) for i, sy in enumerate((1, -1))]
    if k == "opening":
        (hw, hh), sill, x, ww, wh, t = o["size"], o["height"], o["x"], o["wall_w"], o["wall_h"], o["thick"]; top = sill + hh; side = (ww - hw)/2
        out = [_box(f"{n}_side{i}", (t/2, side/2, wh/2), (x, sy*(hw/2 + side/2), wh/2), rgba=W) for i, sy in enumerate((1, -1))]
        out.append(_box(f"{n}_lintel", (t/2, hw/2, (wh - top)/2), (x, 0.0, top + (wh - top)/2), rgba=W))
        if sill > 0: out.append(_box(f"{n}_sill", (t/2, hw/2, sill/2), (x, 0.0, sill/2), rgba=W))
        return out
    if k == "moving_obstacle":
        p = _box(n, o["half"], o["path"][0], rgba=(0.8, 0.3, 0.1, 1)); p["motion"] = dict(path=o["path"], period=o["period"]); return [p]
    raise NotImplementedError(f"obstacle kind {k!r} is not implemented (known: {KINDS})")
