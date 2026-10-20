"""Spec-only lift capacity of stock G1H and H1 (2026-10-17): heaviest load (split across both hands, or on the head) for which the
static, contact-consistent joint torques stay within the rated torques (leverage.solve, LP, feet on the floor, mu 0.9), per posture.
Upper bound: no dynamics, no grip limit (the load is attached at the palms), no balance margin beyond the support polygon."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
import throw as T
from engine import leverage as L

def handles(robot, m):
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_palm"); lp = np.array(m.site_pos[sid]); rb = int(m.site_bodyid[sid])
    lb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, m.body(rb).name.replace("right", "left"))
    arms = [m.joint(j).name for j in range(m.njnt) if any(k in m.joint(j).name for k in ("shoulder", "elbow", "wrist"))]
    waist = [m.joint(j).name for j in range(m.njnt) if any(k in m.joint(j).name for k in ("waist", "torso"))]
    tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    fingers = [a for a in range(m.nu) if "hand" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a][0])) or "")]
    return (rb, lp), (lb, lp*np.array([1, -1, 1])), arms, waist, tb, fingers

def top_point(m, d, body):
    best = None
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != body: continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            a = m.mesh_vertadr[m.geom_dataid[g]]; V = m.mesh_vert[a:a + m.mesh_vertnum[m.geom_dataid[g]]] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        else: V = d.geom_xpos[g][None] + np.array([[0, 0, m.geom_rbound[g]]])
        v = V[int(np.argmax(V[:, 2]))]
        if best is None or v[2] > best[2]: best = v
    return best.copy()

def capacity(robot):
    S = T.setup(robot, ball=False); m = S["m"]; q0 = S["q0"].copy(); d = mujoco.MjData(m)
    R, Lh, arms, waist, tb, fing = handles(robot, m); fg = sorted(S["foot_geoms"])
    d.qpos[:] = q0; mujoco.mj_forward(m, d); pal = lambda h: d.xpos[h[0]] + d.xmat[h[0]].reshape(3, 3) @ h[1]
    p_r0, p_l0 = pal(R), pal(Lh); hip_z = p_r0[2]
    poses = {"suitcase (arms hanging at the sides)": (q0, None),
             "hug at chest": ("ik", (0.25, p_r0[2] + 0.40)),
             "held low in front (bent at the waist)": ("ikw", (0.30, 0.30)),
             "on the head (hands steadying)": (q0, "head")}
    out = {}
    for name, (q, how) in poses.items():
        err = 0.0
        if isinstance(q, str):
            x, z = how; y = abs(p_r0[1]) * 0.8
            q, err = L.site_ik(m, q0, [(R[0], R[1], (x, -y, z)), (Lh[0], Lh[1], (x, y, z))], arms + (waist if q == "ikw" else []))
        d.qpos[:] = q; mujoco.mj_forward(m, d)
        def util(M):
            if how == "head": loads = [L.point_load(m, d, tb, top_point(m, d, tb), M)]
            else: loads = [L.point_load(m, d, R[0], pal(R), M/2), L.point_load(m, d, Lh[0], pal(Lh), M/2)]
            r = L.solve(m, q, loads, fg, robot_bodies=S["robot_bodies"], exclude_act=fing)
            return (r["max_util"] if r["status"] == "OK" else 9.0), r
        lo, hi = 0.0, 300.0
        for _ in range(16):
            mid = 0.5*(lo + hi); u, _ = util(mid)
            lo, hi = (mid, hi) if u <= 1.0 else (lo, mid)
        u, r = util(lo + 1e-6)
        out[name] = dict(max_kg=round(lo, 1), worst_joint=r.get("worst_joint"), ik_err=round(float(err), 4))
        print(f"{robot:4s} {name:40s} max {lo:6.1f} kg  binding {r.get('worst_joint')}  ik_err {err:.3f}", flush=True)
    return dict(mass=float(m.body_mass.sum()), poses=out)

if __name__ == "__main__":
    res = {r: capacity(r) for r in ("G1H", "H1")}
    os.makedirs("../results/lift", exist_ok=True); json.dump(res, open("../results/lift/capacity.json", "w"), indent=1)
