"""Terrain-aware settle. branch_i.settle places the robot by dropping its lowest collision geom to z = 0 (the floor plane) before
the 4 s servo settle; on raised terrain (platform, beam, step top) that drops the robot INTO the piece. settle_on_terrain(S, dz)
offsets the plant's RobotBody.lowest_z() by -dz on that plant instance (so every later settle, including branch_i.apply_dt's
re-settle, places the lowest geom at z = dz) and re-settles. dz comes from world.start_z_offset(spec) and the runner passes it as
demand["start_z"]. No edit to branch_i."""
from __future__ import annotations


def settle_on_terrain(S: dict, dz: float, resettle: bool = True) -> dict:
    import branch_i as BI
    rb = S["rb"]; dz = float(dz)
    if getattr(rb, "_md_dz", None) == dz and S.get("start_z") == dz: return S
    orig = getattr(rb, "_md_lowest_orig", None) or rb.lowest_z
    rb._md_lowest_orig = orig; rb.lowest_z = (lambda o=orig, z=dz: o() - z); rb._md_dz = dz; S["start_z"] = dz
    if resettle:
        for k in ("q_settled", "v_settled", "a_hold", "ctrl_settled"): S.pop(k, None)
        BI.settle(S)
    return S


def standing_height(S) -> dict:
    """Pelvis z, lowest robot geom z and max |qdot| of the settled state (the plant-build check on terrain)."""
    import mujoco, plant as P, numpy as np
    m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d)
    return dict(pelvis_z=float(d.qpos[2]), lowest_z=float(min(P.geom_low(m, d, g) for g in S["robot_geoms"])), settle_qdot=float(S.get("settle_qdot_max", np.nan)), start_z=float(S.get("start_z", 0.0)))
