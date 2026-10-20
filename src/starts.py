"""Generic certified-start construction for any robot in branch_i.ROBOTS:
lean the whole body about the ankles (pelvis pitch + compensating ankle pitch keeps the soles
flat), choose the lean that maximises the static-equilibrium support margin, certify."""
import numpy as np, mujoco
import plant as P


def foot_pitch(m, d, body):
    R = d.xmat[body].reshape(3, 3); return float(np.arctan2(-R[2, 0], R[2, 2]))


def certified_lean_start(S, ankle_joints, foot_bodies, q_base=None, grid=np.arange(-0.3, 0.301, 0.01)):
    m = S["m"]; d = mujoco.MjData(m); rb = S["rb"]
    q_base = np.array(S["q0"] if q_base is None else q_base)
    cg = P.PlantContractVerifier(m, d, P.PlantContract(S["mjcf"], "torque")).collision_geoms()
    fbs = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in foot_bodies]
    jas = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ankle_joints]
    best = None
    for p in grid:
        q = q_base.copy(); q[3:7] = [np.cos(p/2), 0, np.sin(p/2), 0]
        for fb, ja in zip(fbs, jas):
            for _ in range(20):
                d.qpos[:] = q; mujoco.mj_forward(m, d); fp = foot_pitch(m, d, fb); q[ja] -= fp
                if abs(fp) < 1e-7: break
        rb.d.qpos[:] = q; mujoco.mj_forward(m, rb.d); rb.d.qpos[2] -= rb.lowest_z(); mujoco.mj_forward(m, rb.d)
        q = rb.d.qpos.copy()
        c = P.static_certificate(m, d, q, coll_geoms=cg)
        if c["status"] == "STATIC_FEASIBLE" and (best is None or c["support_margin"] > best[1]["support_margin"]):
            best = (q, c, float(p))
    return best


def sole_flat_start(S, ankle_joints, foot_bodies, q_base=None, grid=np.arange(-0.3, 0.301, 0.01), tol=0.0005):
    """Phase 9 (H1 diagnosis): 'feet flat' must mean the SOLE (collision geometry) is flat, not the
    ankle link frame. H1's heel capsule sits 1.7 mm above its toe capsule when the link is level;
    the robot then starts on its toe bar, rocks through the gap and topples over the heel.
    For each lean, pitch each ankle until the lowest points of the foot's collision geoms are
    coplanar with the ground (max gap <= tol), then certify at the same tight tolerance and keep
    the lean with the largest support margin."""
    m = S["m"]; d = mujoco.MjData(m); rb = S["rb"]
    q_base = np.array(S["q0"] if q_base is None else q_base)
    cg = P.PlantContractVerifier(m, d, P.PlantContract(S["mjcf"], "torque")).collision_geoms()
    fbs = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in foot_bodies]
    jas = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ankle_joints]
    def sole_pitch(q, fb):
        d.qpos[:] = q; mujoco.mj_forward(m, d)
        pts = []
        for g in cg:
            if m.geom_bodyid[g] != fb: continue
            c = d.geom_xpos[g]; R = d.geom_xmat[g].reshape(3, 3); r = m.geom_size[g][0]
            if m.geom_type[g] in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
                ax = R[:, 2]*m.geom_size[g][1]; pts += [(c-ax)[[0, 2]] - [0, r], (c+ax)[[0, 2]] - [0, r]]
        pts = np.array(pts); i, j = np.argmin(pts[:, 0]), np.argmax(pts[:, 0])
        return float(np.arctan2(pts[j, 1]-pts[i, 1], pts[j, 0]-pts[i, 0]))   # >0: toe higher than heel
    best = None
    for p in grid:
        q = q_base.copy(); q[3:7] = [np.cos(p/2), 0, np.sin(p/2), 0]
        for fb, ja in zip(fbs, jas):
            for _ in range(30):
                a = sole_pitch(q, fb)
                if abs(a) < 1e-6: break
                q[ja] += a                      # rotate the foot to level the sole
        rb.d.qpos[:] = q; mujoco.mj_forward(m, rb.d); rb.d.qpos[2] -= rb.lowest_z(); mujoco.mj_forward(m, rb.d)
        q = rb.d.qpos.copy()
        c = P.static_certificate(m, d, q, coll_geoms=cg, tol=tol)
        if c["status"] == "STATIC_FEASIBLE" and (best is None or c["support_margin"] > best[1]["support_margin"]):
            best = (q, c, float(p))
    return best
