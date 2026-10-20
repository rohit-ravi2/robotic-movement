"""GRASP LAB (J27): fixed-base G1H, right palm posed by IK (fingers horizontal, palm z vertical), a brush teleported into the open
hand with its handle across the palm (axis = palm z), fingers close, then the arm is SHAKEN; reports whether the wrap survives.
usage: grasp_lab.py [gx list] [gy list] [zsign]   e.g. grasp_lab.py 0.02,0.03,0.04 0.03,0.04,0.05 1"""
import sys, time; sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), 'src')); import numpy as np, mujoco
from engine import world as W
import branch_i as BI
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint", "right_wrist_yaw_joint"]
HEAD = (0.04, 0.025, 0.015); HR = 0.015; HH = 0.16; H_GRIP = HEAD[2] + HH/2    # grasp point: mid-handle above the head centre

def build():
    spec = mujoco.MjSpec.from_file(W.BASE["G1H"]); nq0 = sum(len(k.qpos) for k in spec.keys[:1])
    e = spec.add_equality(name="fix_base", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY, name1="world", name2="pelvis"); e.active = True
    b = spec.worldbody.add_body(name="brush", pos=[0.5, -0.5, 1.0]); b.add_freejoint()
    g = b.add_geom(name="brush_head", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(HEAD), pos=[0, 0, 0], mass=0.12, rgba=[0.2, 0.6, 0.3, 1]); g.solref = [0.002, 1.0]; g.solimp = [0.99, 0.999, 0.001, 0.5, 2.0]
    g = b.add_geom(name="brush_handle", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[HR, HH/2, 0], pos=[0, 0, HEAD[2] + HH/2], mass=0.03, rgba=[0.2, 0.6, 0.3, 1]); g.solref = [0.002, 1.0]; g.solimp = [0.99, 0.999, 0.001, 0.5, 2.0]
    for k in spec.keys: k.qpos = list(k.qpos) + [0.5, -0.5, 1.0, 1, 0, 0, 0]; k.qvel = (list(k.qvel) + [0.0]*6) if len(k.qvel) else k.qvel
    m = spec.compile(); return m, nq0

def servos(m, d, omega_arm=120.0, omega_other=368.0):
    mujoco.mj_forward(m, d); M = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, M, d.qM); Minv = np.linalg.inv(M)
    arm_j = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM}
    for a in range(m.nu):
        j = int(m.actuator_trnid[a][0]); n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        if "hand" in n: continue                                     # finger position servos as in the model (kp 500, +-1.4 N m)
        dof = int(m.jnt_dofadr[j]); ieff = 1.0/max(float(Minv[dof, dof]), 1e-9); om = omega_arm if j in arm_j else omega_other; kp = om*om*ieff
        tau = float(m.actuator_forcerange[a][1]) if m.actuator_forcelimited[a] else 1e3
        BI._as_servo(m, a, kp, 2.0*np.sqrt(kp*ieff), tau); d.ctrl[a] = d.qpos[m.jnt_qposadr[j]]

def ik(m, d, sid, p_star, zsign=1, iters=300, u=None, w=None, c=None):
    """Right-arm damped LS: palm axis u (default palm z) -> world w (default zsign*z), palm x -> world +x (soft); the point c (palm frame, default 0) lands on p_star."""
    u = np.array([0, 0, 1.0]) if u is None else np.asarray(u, float)/np.linalg.norm(u); w = np.array([0, 0, float(zsign)]) if w is None else np.asarray(w, float)/np.linalg.norm(w); c = np.zeros(3) if c is None else np.asarray(c, float)
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM]; dofs = [int(m.jnt_dofadr[j]) for j in jids]; qads = [int(m.jnt_qposadr[j]) for j in jids]
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv)); lo = m.jnt_range[jids, 0] + 0.02; hi = m.jnt_range[jids, 1] - 0.02
    for _ in range(iters):
        mujoco.mj_forward(m, d); R = d.site_xmat[sid].reshape(3, 3); p = d.site_xpos[sid] + R @ c
        e_p = p_star - p; e_r = 3.0*np.cross(R @ u, w) + 0.3*np.cross(R[:, 0], [1, 0, 0])
        mujoco.mj_jacSite(m, d, jacp, jacr, sid); Jp = jacp[:, dofs] - np.cross(np.eye(3)[None, :, :]*0 + (R @ c)[None, :], jacr[:, dofs].T)[:, :].T if False else jacp[:, dofs] + np.cross(jacr[:, dofs].T, R @ c).T
        Jm = np.vstack([Jp, jacr[:, dofs]]); e = np.concatenate([e_p, e_r])
        dq = np.linalg.solve(Jm.T @ Jm + 1e-3*np.eye(len(dofs)), Jm.T @ e)*0.5
        q = np.clip(d.qpos[qads] + dq, lo, hi); d.qpos[qads] = q
        if np.linalg.norm(e_p) < 1e-3 and np.linalg.norm(e_r) < 2e-2: break
    mujoco.mj_forward(m, d); R = d.site_xmat[sid].reshape(3, 3); return float(np.linalg.norm(p_star - d.site_xpos[sid] - R @ c)), float(np.degrees(np.arccos(np.clip(np.dot(R @ u, w), -1, 1))))

def run(gx, gy, zsign=1, p_star=(0.30, -0.30, 0.85), shake=True, verbose=False, T=5.0, snaps=None, amp=1.0, head_mass=None, frame=None):
    """frame=(u_eq, c_eq): grasp FRAME mode - the brush stands vertical (axis world z) with its mid-handle at p_star; the palm is posed so that palm axis u_eq is vertical and palm point c_eq is on the handle (gx, gy ignored)."""
    m, nq0 = build(); d = mujoco.MjData(m); k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "ready"); d.qpos[:] = m.key_qpos[k]
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_palm"); bb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "brush")
    fj = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in W.FINGER_JOINTS}; fa = {n: [a for a in range(m.nu) if int(m.actuator_trnid[a][0]) == j][0] for n, j in fj.items()}
    for n, j in fj.items(): d.qpos[m.jnt_qposadr[j]] = W.FINGER_OPEN[n]
    if head_mass is not None:
        gh = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "brush_head"); m.body_mass[bb] += head_mass - 0.12; m.body_inertia[bb] *= (m.body_mass[bb]/(m.body_mass[bb] - head_mass + 0.12))
    if frame is None:
        e_p, e_z = ik(m, d, sid, np.asarray(p_star), zsign)
        R = d.site_xmat[sid].reshape(3, 3); p = d.site_xpos[sid]
        # brush: handle axis = palm z (zsign): body z up in world when zsign=+1 (head below the hand); the grasp point sits at palm (gx, gy, 0)
        gp = p + R @ np.array([gx, gy, 0.0]); zb = zsign*R[:, 2]
    else:
        u_eq, c_eq = frame; e_p, e_z = ik(m, d, sid, np.asarray(p_star), 1, u=u_eq, w=[0, 0, 1], c=c_eq)
        R = d.site_xmat[sid].reshape(3, 3); gp = np.asarray(p_star, float); zb = np.array([0, 0, 1.0])
    jb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "brush"); a0 = m.jnt_qposadr[jb]
    # brush frame: z along the handle; choose x = R x
    xb = R[:, 0] - np.dot(R[:, 0], zb)*zb; xb /= np.linalg.norm(xb); yb = np.cross(zb, xb); Rb = np.stack([xb, yb, zb], 1)
    qb = np.zeros(4); mujoco.mju_mat2Quat(qb, Rb.ravel()); d.qpos[a0:a0+3] = gp - Rb @ np.array([0, 0, H_GRIP]); d.qpos[a0+3:a0+7] = qb; d.qvel[:] = 0
    servos(m, d); hand_b = {b for b in range(m.nbody) if "right_hand" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")} | {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")}
    mujoco.mj_forward(m, d); pen0 = min([c.dist for c in d.contact[:d.ncon]] + [0.0])
    def contacts():
        links = set(); F = 0.0
        for ci in range(d.ncon):
            c = d.contact[ci]; b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if bb in (b1, b2):
                o = b2 if b1 == bb else b1
                if o in hand_b: links.add(o); f = np.zeros(6); mujoco.mj_contactForce(m, d, ci, f); F += abs(float(f[0]))
        return links, F
    qb_hold = d.qpos[a0:a0+7].copy(); q0 = d.ctrl.copy(); sp = [a for a in range(m.nu) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) == "right_shoulder_pitch_joint"][0]; wp = [a for a in range(m.nu) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) == "right_wrist_pitch_joint"][0]
    dt = m.opt.timestep; n = int(T/dt); log = []; rel0 = None; held_frac = 0; n_meas = 0; slip = 0.0; rot = 0.0; Fmin = 1e9; dropped = None
    for s in range(n):
        t = s*dt; w = float(np.clip((t - 0.2)/0.4, 0, 1))
        for nme, a in fa.items(): d.ctrl[a] = (1 - w)*W.FINGER_OPEN[nme] + w*W.FINGER_CLOSE[nme]
        if shake and 1.5 <= t < 4.5:
            d.ctrl[sp] = q0[sp] + amp*0.25*np.sin(2*np.pi*1.5*(t - 1.5)); d.ctrl[wp] = q0[wp] + amp*0.30*np.sin(2*np.pi*2.5*(t - 1.5))
        else: d.ctrl[sp] = q0[sp]; d.ctrl[wp] = q0[wp]
        if t < 0.8: d.qpos[a0:a0+7] = qb_hold; d.qvel[m.jnt_dofadr[jb]:m.jnt_dofadr[jb]+6] = 0.0     # kinematic support until the fingers have closed (the counter's role in the pick)
        mujoco.mj_step(m, d)
        if snaps and any(abs(t - ts) < 0.5*dt for ts in snaps["t"]):
            for az in snaps.get("az", (180,)): snapshot(m, d, sid, f"{snaps['dir']}/lab_{gx:.3f}_{gy:.3f}_t{t:.1f}_az{az}.png", az=az)
        if s % 10 == 0 and t >= 1.4:
            links, F = contacts(); R = d.site_xmat[sid].reshape(3, 3); rel = R.T @ (d.xpos[bb] - d.site_xpos[sid]); axz = float(np.degrees(np.arccos(np.clip(abs(np.dot(d.xmat[bb].reshape(3, 3)[:, 2], R[:, 2])), -1, 1))))
            if rel0 is None: rel0 = rel.copy(); cmap = [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, l)) for l in links]; F0 = F
            held = (len(links) >= 2 and F >= 1.0); held_frac += held; n_meas += 1; slip = max(slip, float(np.linalg.norm(rel - rel0))); rot = max(rot, axz); Fmin = min(Fmin, F)
            if not held and dropped is None: dropped = t
    links, F = contacts(); R = d.site_xmat[sid].reshape(3, 3); rel_end = R.T @ (d.xpos[bb] - d.site_xpos[sid]); ax_end = R.T @ d.xmat[bb].reshape(3, 3)[:, 2]
    grip_end = rel_end + ax_end*H_GRIP      # the mid-handle grasp point in the palm frame after settling
    res = dict(gx=gx, gy=gy, zsign=zsign, palm_R=np.round(d.site_xmat[sid].reshape(3, 3), 2).tolist() if frame is not None else None, axis_palm=np.round(ax_end, 3).tolist(), grip_palm=np.round(grip_end, 3).tolist(), centre_palm=np.round(rel_end, 3).tolist(), ik_pos=round(e_p, 4), ik_zdeg=round(e_z, 1), pen0=round(pen0, 4), links0=sorted(l.replace("right_hand_", "") for l in cmap), F0=round(F0, 1), held_frac=round(held_frac/max(n_meas, 1), 2), slip=round(slip, 4), rot_deg=round(rot, 1), Fmin=round(Fmin, 1), F_end=round(F, 1), dropped=dropped)
    print(res, flush=True); return res

if __name__ == "__main__":
    gxs = [float(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0.02,0.03,0.04,0.05").split(",")]; gys = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0.03,0.04,0.05,0.06").split(",")]; zs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    for gx in gxs:
        for gy in gys: run(gx, gy, zs)


def snapshot(m, d, sid, path, az=180, el=-25, dist=0.35):
    import os; os.environ.setdefault("MUJOCO_GL", "egl")
    r = mujoco.Renderer(m, 480, 640); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = d.site_xpos[sid]; cam.distance = dist; cam.azimuth = az; cam.elevation = el
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[3] = 1; opt.geomgroup[0] = 1     # collision geometry (what the physics sees)
    r.update_scene(d, cam, opt); img = r.render(); r.close()
    from PIL import Image; Image.fromarray(img).save(path)


# ---------------------------------------------------------------- pick rehearsal with an ANGLED-handle brush on a slab
U_EQ = np.array([0.69, 0.32, 0.65]); U_EQ /= np.linalg.norm(U_EQ)      # settled handle axis in the palm frame (standard pose: palm x -> world x, palm z -> world z)
C_EQ = np.array([0.054, 0.056, 0.0])                                    # mid-handle grasp point in the palm frame

def build_pick(lean_dir, slab_top=0.64, hh=HH):
    """G1H fixed base + slab + brush whose handle leans along the unit vector lean_dir (world frame at rest = body frame)."""
    spec = mujoco.MjSpec.from_file(W.BASE["G1H"]); nq0 = sum(len(k.qpos) for k in spec.keys[:1])
    e = spec.add_equality(name="fix_base", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY, name1="world", name2="pelvis"); e.active = True
    g = spec.worldbody.add_geom(name="slab", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.35, 0.4, 0.02], pos=[0.55, 0.0, slab_top - 0.02], rgba=[0.6, 0.4, 0.2, 1]); g.solref = [0.002, 1.0]; g.solimp = [0.99, 0.999, 0.001, 0.5, 2.0]
    b = spec.worldbody.add_body(name="brush", pos=[0.5, -0.5, 1.0]); b.add_freejoint()
    g = b.add_geom(name="brush_head", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(HEAD), pos=[0, 0, 0], mass=0.12, rgba=[0.2, 0.6, 0.3, 1]); g.solref = [0.002, 1.0]; g.solimp = [0.99, 0.999, 0.001, 0.5, 2.0]
    ax = np.asarray(lean_dir, float)/np.linalg.norm(lean_dir); mid = np.array([0, 0, HEAD[2]]) + ax*(hh/2)
    zq = np.zeros(4); mujoco.mju_quatZ2Vec(zq, ax)       # quaternion rotating body z onto the handle axis
    g = b.add_geom(name="brush_handle", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[HR, hh/2, 0], pos=mid.tolist(), quat=zq.tolist(), mass=0.03, rgba=[0.2, 0.6, 0.3, 1]); g.solref = [0.002, 1.0]; g.solimp = [0.99, 0.999, 0.001, 0.5, 2.0]
    for k in spec.keys: k.qpos = list(k.qpos) + [0.5, -0.5, 1.0, 1, 0, 0, 0]; k.qvel = (list(k.qvel) + [0.0]*6) if len(k.qvel) else k.qvel
    m = spec.compile(); return m, nq0, ax, mid

def run_pick(xy=(0.30, -0.30), slab_top=0.64, approach=0.10, lift=0.05, amp=1.0, snaps=None, T=7.0, u_eq=U_EQ, c_eq=C_EQ, close_T=0.4):
    """Hand in the standard pose slides forward (world +x) over the open fingers onto the leaning handle, closes, lifts, shakes."""
    m, nq0, ax, mid = build_pick(u_eq, slab_top); d = mujoco.MjData(m); k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "ready"); d.qpos[:] = m.key_qpos[k]
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_palm"); bb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "brush"); jb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "brush"); a0 = m.jnt_qposadr[jb]
    fj = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in W.FINGER_JOINTS}; fa = {n: [a for a in range(m.nu) if int(m.actuator_trnid[a][0]) == j][0] for n, j in fj.items()}
    for n, j in fj.items(): d.qpos[m.jnt_qposadr[j]] = W.FINGER_OPEN[n]
    # brush at rest: head bottom on the slab, identity orientation (the handle leans along u_eq in WORLD = palm-frame vector under the standard pose)
    head_c = np.array([xy[0], xy[1], slab_top + HEAD[2]]); grip = head_c + mid; d.qpos[a0:a0+3] = head_c; d.qpos[a0+3:a0+7] = [1, 0, 0, 0]
    Rstd = np.eye(3); p_grasp = grip - Rstd @ c_eq; p_via = p_grasp - np.array([approach, 0, 0]); p_lift = p_grasp + np.array([0, 0, lift])
    arm_q = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in ARM]
    e1 = ik(m, d, sid, p_via, 1); q_via = d.qpos[arm_q].copy(); e2 = ik(m, d, sid, p_grasp, 1); q_grasp = d.qpos[arm_q].copy(); e3 = ik(m, d, sid, p_lift, 1); q_lift = d.qpos[arm_q].copy()
    d.qpos[arm_q] = q_via; d.qvel[:] = 0; servos(m, d); mujoco.mj_forward(m, d)
    hand_b = {b for b in range(m.nbody) if "right_hand" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")} | {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")}
    arm_a = [[a for a in range(m.nu) if int(m.actuator_trnid[a][0]) == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)][0] for n in ARM]
    def contacts():
        links = set(); F = 0.0
        for ci in range(d.ncon):
            c = d.contact[ci]; b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if bb in (b1, b2):
                o = b2 if b1 == bb else b1
                if o in hand_b: links.add(o); f = np.zeros(6); mujoco.mj_contactForce(m, d, ci, f); F += abs(float(f[0]))
        return links, F
    # schedule: 0-0.5 settle | 0.5-1.5 slide in | 1.5-1.9 close | 1.9-2.4 hold | 2.4-2.9 lift | 2.9-5.9 shake | 5.9-T hold
    sp = arm_a[0]; wp = arm_a[4]; dt = m.opt.timestep; n = int(T/dt); stats = dict(held=0, nmeas=0, slip=0.0, rot=0.0, Fmin=1e9, dropped=None, brush_moved_before_close=0.0); rel0 = None; ax0 = None; b_pos0 = d.qpos[a0:a0+3].copy()
    for s in range(n):
        t = s*dt
        if t < 0.5: qa_t = q_via
        elif t < 1.5: w = (t - 0.5)/1.0; qa_t = (1 - w)*q_via + w*q_grasp
        elif t < 2.4: qa_t = q_grasp
        elif t < 2.9: w = (t - 2.4)/0.5; qa_t = (1 - w)*q_grasp + w*q_lift
        else: qa_t = q_lift.copy()
        if 2.9 <= t < 5.9: qa_t = qa_t.copy(); qa_t[0] += amp*0.25*np.sin(2*np.pi*1.5*(t - 2.9)); qa_t[4] += amp*0.30*np.sin(2*np.pi*2.5*(t - 2.9))
        d.ctrl[arm_a] = qa_t
        wc = float(np.clip((t - 1.5)/close_T, 0, 1))
        for nme, a in fa.items(): d.ctrl[a] = (1 - wc)*W.FINGER_OPEN[nme] + wc*W.FINGER_CLOSE[nme]
        mujoco.mj_step(m, d)
        if snaps and any(abs(t - ts) < 0.5*dt for ts in snaps["t"]):
            for az in snaps.get("az", (180,)): snapshot(m, d, sid, f"{snaps['dir']}/pick_t{t:.1f}_az{az}.png", az=az)
        if t < 1.5 and s % 10 == 0: stats["brush_moved_before_close"] = max(stats["brush_moved_before_close"], float(np.linalg.norm(d.qpos[a0:a0+3] - b_pos0)))
        if s % 10 == 0 and t >= 2.3:
            links, F = contacts(); R = d.site_xmat[sid].reshape(3, 3); rel = R.T @ (d.xpos[bb] - d.site_xpos[sid]); axp = R.T @ (d.xmat[bb].reshape(3, 3) @ ax)
            if rel0 is None: rel0 = rel.copy(); ax0 = axp.copy(); stats["links_at_close"] = sorted(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, l).replace("right_hand_", "") for l in links); stats["F_at_close"] = round(F, 1); stats["axis_palm_at_close"] = np.round(axp, 3).tolist(); stats["grip_palm_at_close"] = np.round(rel + axp*0 + R.T @ (d.xmat[bb].reshape(3, 3) @ mid), 3).tolist()
            held = (len(links) >= 2 and F >= 1.0); stats["held"] += held; stats["nmeas"] += 1; stats["slip"] = max(stats["slip"], float(np.linalg.norm(rel - rel0))); stats["rot"] = max(stats["rot"], float(np.degrees(np.arccos(np.clip(np.dot(axp, ax0), -1, 1))))); stats["Fmin"] = min(stats["Fmin"], F)
            if not held and stats["dropped"] is None: stats["dropped"] = round(t, 2)
    links, F = contacts(); z_b = float(d.xpos[bb][2])
    res = dict(ik_err=(round(e1[0], 4), round(e2[0], 4), round(e3[0], 4)), moved_before_close=round(stats["brush_moved_before_close"], 4), links_at_close=stats.get("links_at_close"), F_at_close=stats.get("F_at_close"), axis_palm_at_close=stats.get("axis_palm_at_close"), grip_palm_at_close=stats.get("grip_palm_at_close"),
               held_frac=round(stats["held"]/max(stats["nmeas"], 1), 2), slip=round(stats["slip"], 4), rot_deg=round(stats["rot"], 1), Fmin=round(stats["Fmin"], 1), F_end=round(F, 1), dropped=stats["dropped"], brush_z_end=round(z_b, 3), lifted=z_b > slab_top + HEAD[2] + 0.5*lift)
    print(res, flush=True); return res
