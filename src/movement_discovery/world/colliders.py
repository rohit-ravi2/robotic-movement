"""EFFECTOR SPHERE COLLIDERS: the one helper that gives a compiled scene a 4 cm, massless, contype-1 sphere at effector sites (palm /
foot), exactly as tasks/impact.py (reference, frozen) does for its own scenes, because the MJX G1 body carries NO collision bits: a
hand cannot touch a door, a box or a ball otherwise. Foot spheres sit tangent to the sole (site + r on z). Used by the OBJECT
families (force_interaction/_mech, interception/_catch); impact/_local keeps the reference's own collider. Labelled simplification:
the palm is a 4 cm sphere. MuJoCo only inside functions."""
from __future__ import annotations
import os
from typing import Sequence
import numpy as np
from movement_discovery.core.contracts import WorldSpec

EFF_R = 0.04


def effector_spheres(base: str, sites: Sequence[str], r: float = EFF_R) -> str:
    """MJCF path = `base` + spheres at `sites` (cached as <base>_eff_<sites>_r<mm>.xml next to it)."""
    if not sites: return base
    import mujoco
    tag = "_".join(s.replace("_", "") for s in sites) + f"_r{int(round(r*1000))}"
    out = (base[:-4] if base.endswith(".xml") else base) + f"_eff_{tag}.xml"
    if os.path.exists(out): return out
    sp = mujoco.MjSpec.from_file(base); mb = mujoco.MjModel.from_xml_path(base)
    for sname in sites:
        sid = mujoco.mj_name2id(mb, mujoco.mjtObj.mjOBJ_SITE, sname); assert sid >= 0, sname
        bname = mujoco.mj_id2name(mb, mujoco.mjtObj.mjOBJ_BODY, mb.site_bodyid[sid]); body = sp.body(bname) if hasattr(sp, "body") else sp.find_body(bname)
        pos = np.array(mb.site_pos[sid], float)
        if "foot" in sname: pos[2] += r                                      # foot site at the sole: sphere tangent to it (tasks/impact.py)
        g = body.add_geom(name=f"eff_{sname}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[r, 0, 0], pos=pos.tolist(), mass=0.0, group=3, rgba=[1, 0.6, 0.1, 0.6]); g.contype = 1; g.conaffinity = 1
    sp.compile(); open(out, "w").write(sp.to_xml()); return out


def scene(spec: WorldSpec, robot: str, sites: Sequence[str] = ("right_palm",), r: float = EFF_R) -> str:
    """world.build(spec, robot) (the bare base scene for an empty spec) + effector spheres at `sites`."""
    from .build import build
    base = build(spec, robot)
    if not base:
        from engine import world as EW; base = EW.BASE[robot]
    return effector_spheres(base, sites, r)


def enable_effector_geoms(S, site_name=None) -> int:
    """Turn on the collision bits of the effector body's group-3 geoms (hand capsule / foot spheres) on a built plant, as
    tasks/impact.py does; returns the number of geoms enabled."""
    import mujoco
    m = S["m"]; sid = S["site"] if site_name is None else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
    hb = int(m.site_bodyid[sid]); n = 0
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) == hb and int(m.geom_group[g]) == 3: m.geom_contype[g] = 1; m.geom_conaffinity[g] = 1; n += 1
    return n
