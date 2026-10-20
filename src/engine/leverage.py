"""LEVERAGE / LOAD-PATH PREDICTOR (L0, 2026-10-16).

For a body pose q, the floor contacts of the support feet, optional extra unilateral contacts and a set of external point loads,
solve the static contact-consistent joint torques

    S tau + Jc^T f = g(q) - sum_k J_k^T F_k          (qvel = 0; F_k = the load's force on the robot at its point)

with |tau_j| <= s tau_max_j, f_n >= 0 and an inner-pyramid friction cone (the static certificate's LP, plant.static_certificate),
minimising s (= max utilisation) and then, at that s, the L1 utilisation (a well-defined torque split). The LP is solved in
SCALED variables u = tau/tau_max and f/W (W = body weight): the unscaled LP mixes 1e-3 and 1e3 coefficients, which is what gave
HiGHS its status-4 artefact (INTEGRATION_NOTES). Returns per-joint utilisation |tau_j|/tau_max_j, its max and RMS, the worst
joint, and per-load moment arms (per joint: |((p - x_j) x F) . a_j| / |F|; whole body: horizontal distance of the load line from
the support-polygon centroid). This is the PRINCIPLE: it is computed from the robot spec and a pose only, before any search.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
from scipy.optimize import linprog
import plant as P

G = 9.80665


def tau_max(m):
    """Per-actuator torque limit (the certificate's rule: forcerange, else joint actfrcrange, else motor ctrlrange)."""
    kinds = P.actuator_kinds(m); out = np.zeros(m.nu)
    for a in range(m.nu):
        jid = int(m.actuator_trnid[a][0])
        if m.actuator_forcelimited[a]: t = float(m.actuator_forcerange[a][1])
        elif m.jnt_actfrclimited[jid]: t = float(m.jnt_actfrcrange[jid][1])
        else: t = float(m.actuator_ctrlrange[a][1]) if kinds[a] == "torque_motor" else np.inf
        out[a] = t
    return out


def point_load(m, d, body, point, mass=None, force=None):
    """A load on `body` at world `point`: a mass (weight -mg z) or an explicit world force."""
    F = np.array([0, 0, -float(mass)*G]) if force is None else np.asarray(force, float)
    return dict(body=int(body), point=np.asarray(point, float), force=F)


def solve(m, q, loads=(), floor_geoms=None, extra_contacts=(), mu=0.9, tol=0.003, robot_bodies=None, exclude_act=(), held=None):
    """extra_contacts: [dict(body, point, normal)] unilateral frictional contacts (e.g. a hand on a table).
    exclude_act: actuator ids ignored in the max/RMS (e.g. finger servos, which no load reaches).
    held: dict(mass, com (world), grips=[(body, world point)]): an object held by WELDS at the grips. Each grip carries a free 6-D
    wrench (force + moment) on the robot; the object's own equilibrium (sum of grip forces = its weight, moments about its COM = 0)
    closes the system, so the LP chooses how the weight and the moment split between the hands (statically indeterminate)."""
    d = mujoco.MjData(m); d.qpos[:] = q; d.qvel[:] = 0; mujoco.mj_forward(m, d)
    nv, nu = m.nv, m.nu
    rd = np.arange(nv) if robot_bodies is None else np.array([i for i in range(nv) if int(m.dof_bodyid[i]) in robot_bodies])
    if floor_geoms is None:
        floor_geoms = [g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g] != 0]
    cp = P.potential_contacts(m, d, sorted(set(floor_geoms)), tol, with_geom=True)
    cons = [dict(body=int(m.geom_bodyid[int(c[3])]), point=np.array(c[:3], float), normal=np.array([0, 0, 1.0])) for c in cp] + list(extra_contacts)
    nc = len(cons)
    if nc == 0: return dict(status="NO_CONTACT")
    W = float(sum(m.body_mass[b] for b in (robot_bodies or range(1, m.nbody))))*G
    tm = tau_max(m); fin = np.isfinite(tm); tm_s = np.where(fin, tm, 1e3)
    b = np.array(d.qfrc_bias, float)
    jp = np.zeros((3, nv)); jr = np.zeros((3, nv))
    for L in loads:
        mujoco.mj_jac(m, d, jp, jr, L["point"], L["body"]); b -= jp.T @ L["force"]
    Sel = np.zeros((nv, nu))
    for a in range(nu): Sel[int(m.jnt_dofadr[int(m.actuator_trnid[a][0])]), a] = float(m.actuator_gear[a][0])*tm_s[a]
    Jc = np.zeros((nv, 3*nc))
    for i, c in enumerate(cons):
        n = c["normal"]/np.linalg.norm(c["normal"]); t1 = np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0]); t1 /= np.linalg.norm(t1); t2 = np.cross(n, t1)
        mujoco.mj_jac(m, d, jp, jr, c["point"], c["body"]); Jc[:, 3*i:3*i+3] = jp.T @ np.stack([n, t1, t2], 1)*W
    # variables x = [u (nu), f/W (3nc), s, t (nu), held wrenches/W (6 per grip)]
    ng = len(held["grips"]) if held else 0
    n = 2*nu + 3*nc + 1 + 6*ng; iS = nu + 3*nc; it = iS + 1; iw = it + nu
    Aeq = np.zeros((nv, n)); Aeq[:, :nu] = Sel; Aeq[:, nu:iS] = Jc
    if ng:
        for h, (hb, hp) in enumerate(held["grips"]):
            mujoco.mj_jac(m, d, jp, jr, np.asarray(hp, float), int(hb)); Aeq[:, iw+6*h:iw+6*h+3] = jp.T*W; Aeq[:, iw+6*h+3:iw+6*h+6] = jr.T*W
    Aeq, beq = Aeq[rd]/W, b[rd]/W
    if ng:   # object equilibrium (scaled by W): sum F_h = -m g z ; sum (p_h - c) x F_h + M_h = 0
        Ao = np.zeros((6, n)); bo = np.zeros(6); bo[2] = -float(held["mass"])*G/W; c0 = np.asarray(held["com"], float)
        for h, (hb, hp) in enumerate(held["grips"]):
            r_ = np.asarray(hp, float) - c0; Rx = np.array([[0, -r_[2], r_[1]], [r_[2], 0, -r_[0]], [-r_[1], r_[0], 0]])
            Ao[:3, iw+6*h:iw+6*h+3] = np.eye(3); Ao[3:, iw+6*h:iw+6*h+3] = Rx; Ao[3:, iw+6*h+3:iw+6*h+6] = np.eye(3)
        Aeq = np.vstack([Aeq, Ao]); beq = np.concatenate([beq, bo])
    rows = []
    for i in range(nu):
        for sg in (1, -1):
            r = np.zeros(n); r[i] = sg; r[iS] = -1.0; rows.append(r)          # |u_i| <= s
            r = np.zeros(n); r[i] = sg; r[it+i] = -1.0; rows.append(r)        # |u_i| <= t_i
    k = mu/np.sqrt(2.0)
    for i in range(nc):
        o = nu + 3*i
        for tt in (1, 2):
            for sg in (1, -1):
                r = np.zeros(n); r[o+tt] = sg; r[o] = -k; rows.append(r)
    A = np.array(rows); bub = np.zeros(len(rows))
    bnds = [(None, None)]*nu + [(0, None) if j % 3 == 0 else (None, None) for j in range(3*nc)] + [(0, None)] + [(0, None)]*nu + [(None, None)]*(6*ng)
    c1 = np.zeros(n); c1[iS] = 1.0
    r1 = linprog(c1, A_ub=A, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bnds, method="highs")
    if not r1.success: return dict(status="INFEASIBLE", detail=r1.message, ncon=nc)
    s1 = float(r1.x[iS]); bnds2 = list(bnds); bnds2[iS] = (0, s1*1.001 + 1e-6)
    c2 = np.zeros(n); c2[it:] = 1.0
    r2 = linprog(c2, A_ub=A, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bnds2, method="highs")
    x = r2.x if r2.success else r1.x
    u = np.abs(x[:nu]); tau = x[:nu]*tm_s; f = x[nu:iS].reshape(nc, 3)*W
    keep = np.array([a not in set(exclude_act) for a in range(nu)])
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a][0])) for a in range(nu)]
    wj = int(np.argmax(np.where(keep, u, -1)))
    # moment arms
    pts = np.array([c["point"][:2] for c in cons]); cen = pts.mean(0); arms = []
    for L in loads:
        F = L["force"]; Fn = np.linalg.norm(F) + 1e-12; per = {}; anc = set(); x_ = L["body"]
        while x_ > 0: anc.add(x_); x_ = int(m.body_parentid[x_])
        for a in range(nu):
            if not keep[a] or int(m.jnt_bodyid[int(m.actuator_trnid[a][0])]) not in anc: continue   # joints on the load's chain to the root
            jid = int(m.actuator_trnid[a][0]); xa = d.xanchor[jid]; ax = d.xaxis[jid]
            per[names[a]] = float(abs(np.cross(L["point"] - xa, F) @ ax)/Fn)
        top = dict(sorted(per.items(), key=lambda kv: -kv[1])[:5])
        arms.append(dict(body=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, L["body"]), point=L["point"].round(3).tolist(),
                         arm_to_support_centre=float(np.linalg.norm(L["point"][:2] - cen)), top_joint_arms=top,
                         sum_joint_arms=float(sum(per.values()))))
    ug = u[keep]
    grip_w = [dict(force=(x[iw+6*h:iw+6*h+3]*W).tolist(), moment=(x[iw+6*h+3:iw+6*h+6]*W).tolist()) for h in range(ng)]
    return dict(status="OK", grip_wrenches=grip_w, max_util=float(ug.max()), rms_util=float(np.sqrt(np.mean(ug**2))), mean_util=float(ug.mean()),
                worst_joint=names[wj], util={names[a]: float(u[a]) for a in range(nu) if keep[a]}, tau=tau.tolist(),
                normal_force=float(f[:, 0].sum()), weight=W, ncon=nc, com_xy=np.array(d.subtree_com[1][:2]).tolist(),
                support_margin=float(P._support_margin(pts, np.array(d.subtree_com[1][:2]))) if nc >= 3 else None, load_paths=arms)


def site_ik(m, q, targets, joints, iters=200, step=0.5, damp=1e-3, tol=1e-3):
    """Damped least-squares IK: move only `joints` (names) so each (body, local point) in targets reaches its world point.
    targets: [(body_id, local_point, world_target)]. Joint limits are clipped. Base and other joints stay fixed."""
    d = mujoco.MjData(m); q = np.array(q, float)
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints]
    dofs = [int(m.jnt_dofadr[j]) for j in jids]; qa = [int(m.jnt_qposadr[j]) for j in jids]
    jp = np.zeros((3, m.nv))
    for _ in range(iters):
        d.qpos[:] = q; mujoco.mj_forward(m, d); E = []; Js = []
        for bid, lp, tgt in targets:
            p = d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ np.asarray(lp); E.append(np.asarray(tgt) - p)
            mujoco.mj_jac(m, d, jp, None, p, bid); Js.append(jp[:, dofs].copy())
        e = np.concatenate(E); J = np.vstack(Js)
        if np.linalg.norm(e) < tol: break
        dq = J.T @ np.linalg.solve(J @ J.T + damp*np.eye(len(e)), e)*step
        for i, j in enumerate(jids):
            q[qa[i]] = np.clip(q[qa[i]] + dq[i], *m.jnt_range[j]) if m.jnt_limited[j] else q[qa[i]] + dq[i]
    return q, float(np.linalg.norm(e))


def g1h_parts(m):
    """Named G1H handles: palms (right site + its mirror on the left wrist), head top point (torso_link head mesh), act ids of fingers."""
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_palm"); lp = np.array(m.site_pos[sid])
    rb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"); lb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    head = max((g for g in range(m.ngeom) if m.geom_bodyid[g] == tb and m.geom_contype[g]), key=lambda g: m.geom_pos[g][2])
    fingers = [a for a in range(m.nu) if "hand" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a][0])) or "")]
    return dict(rpalm=(rb, lp), lpalm=(lb, lp*np.array([1, -1, 1])), torso=tb, head_geom=head, fingers=fingers)


def head_top(m, d, head_geom):
    """World point at the top of the head collision mesh (its highest vertex at the current pose)."""
    mid = m.geom_dataid[head_geom]; a = m.mesh_vertadr[mid]; V = m.mesh_vert[a:a + m.mesh_vertnum[mid]] @ d.geom_xmat[head_geom].reshape(3, 3).T + d.geom_xpos[head_geom]
    return V[int(np.argmax(V[:, 2]))].copy()


ARMS = [f"{s}_{j}" for s in ("left", "right") for j in ("shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint", "elbow_joint",
                                                         "wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint")]


def acceptance(mass=10.0, chest=(0.30, 0.12, 1.00)):
    """L0 acceptance: G1H standing, a mass (a) welded to both palms at chest height vs (b) resting on top of the head."""
    import throw as T
    S = T.setup("G1H", ball=False); m = S["m"]; q0 = S["q0"].copy(); H = g1h_parts(m); d = mujoco.MjData(m)
    fg = sorted(S["foot_geoms"]); ex = H["fingers"]; out = {}
    out["unloaded"] = solve(m, q0, (), fg, robot_bodies=S["robot_bodies"], exclude_act=ex)
    x, y, z = chest
    qa, err = site_ik(m, q0, [(H["rpalm"][0], H["rpalm"][1], (x, -y, z)), (H["lpalm"][0], H["lpalm"][1], (x, y, z))], ARMS)
    d.qpos[:] = qa; mujoco.mj_forward(m, d)
    P_ = [d.xpos[b] + d.xmat[b].reshape(3, 3) @ lp for b, lp in (H["rpalm"], H["lpalm"])]
    loads = [point_load(m, d, H["rpalm"][0], P_[0], mass/2), point_load(m, d, H["lpalm"][0], P_[1], mass/2)]
    out["arms_unloaded"] = solve(m, qa, (), fg, robot_bodies=S["robot_bodies"], exclude_act=ex)
    out["hands_chest"] = solve(m, qa, loads, fg, robot_bodies=S["robot_bodies"], exclude_act=ex); out["hands_chest"]["ik_err"] = err
    d.qpos[:] = q0; mujoco.mj_forward(m, d); ht = head_top(m, d, H["head_geom"])
    out["head"] = solve(m, q0, [point_load(m, d, H["torso"], ht, mass)], fg, robot_bodies=S["robot_bodies"], exclude_act=ex)
    return out


if __name__ == "__main__":
    import json
    o = acceptance(float(sys.argv[1]) if len(sys.argv) > 1 else 10.0)
    for k, v in o.items():
        if v["status"] != "OK": print(k, v); continue
        print(f"{k:14s} max_util={v['max_util']:.3f} rms={v['rms_util']:.3f} worst={v['worst_joint']} N={v['normal_force']:.0f}/{v['weight']:.0f} margin={v['support_margin']}",
              *[(lp['body'], round(lp['arm_to_support_centre'], 3), list(lp['top_joint_arms'].items())[:2]) for lp in v["load_paths"]])
    os.makedirs("../results/overnight/leverage", exist_ok=True)
    json.dump(o, open("../results/overnight/leverage/L0_acceptance.json", "w"), indent=1, default=float)
