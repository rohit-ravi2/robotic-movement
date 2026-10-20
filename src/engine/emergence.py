"""EMERGENCE instrument (construct, 2026-10-11): what a recorded effect rollout did, read the way a human analyst reads the g1t_3
animation. Input: the plant S and a throw.rollout(record=True) record. Output: segment peak speeds and times, release-speed
ownership by joint and group (v = J qdot along the effect direction), joint excursions against the model's limits and against
a human range-of-motion table, positive work by group, the arm/leg lag power signature. All quantities from the simulator state."""
import numpy as np, mujoco
# human range of motion (deg) for the joints the G1 can exceed; (lo, hi) in the robot's joint convention where known
HUMAN = {"right_shoulder_pitch_joint": (-180, 60), "left_shoulder_pitch_joint": (-180, 60), "right_shoulder_yaw_joint": (-90, 90), "left_shoulder_yaw_joint": (-90, 90),
         "right_elbow_joint": (-10, 145), "left_elbow_joint": (-10, 145), "waist_yaw_joint": (-45, 45), "waist_pitch_joint": (-30, 30), "waist_roll_joint": (-30, 30),
         "right_hip_yaw_joint": (-45, 45), "left_hip_yaw_joint": (-45, 45), "right_hip_pitch_joint": (-120, 30), "left_hip_pitch_joint": (-120, 30), "right_knee_joint": (0, 140), "left_knee_joint": (0, 140)}
CHAINS = {"palm": ["pelvis", "torso_link", "{s}_shoulder_yaw_link", "{s}_elbow_link", "{s}_wrist_yaw_link"], "foot": ["pelvis", "{s}_hip_yaw_link", "{s}_knee_link", "{s}_ankle_roll_link"]}

def chain_bodies(m, site_name):
    s = "left" if site_name.startswith("left") else "right"; kind = "foot" if "foot" in site_name else "palm"
    names = [n.format(s=s) for n in CHAINS[kind]]; ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
    return [(n, b) for n, b in zip(names, ids) if b >= 0]

def analyze(S, rec, site_name):
    m = S["m"]; J = S["jarr"]; tr = rec["trace"]; types = S["types"]; groups = S["groups"]; nT = len(types)
    q = np.asarray(tr["q"]); qd = np.asarray(tr["qvel"]); tq = np.asarray(tr["torque"]); P = np.asarray(tr["power"]); dt = float(tr["dt"]); t = np.arange(len(q))*dt
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name); seg = chain_bodies(m, site_name)
    d = mujoco.MjData(m); vel6 = np.zeros(6); W = np.zeros((len(q), len(seg))); vsite = np.zeros((len(q), 3))
    for i in range(len(q)):
        d.qpos[:] = q[i]; d.qvel[:] = qd[i]; mujoco.mj_forward(m, d)
        for k, (_, b) in enumerate(seg): mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel6, 0); W[i, k] = np.linalg.norm(vel6[:3])
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, sid, vel6, 0); vsite[i] = vel6[3:]
    peaks = {n: dict(omega=float(W[:, k].max()), t=float(t[np.argmax(W[:, k])])) for k, (n, _) in enumerate(seg)}
    order = [n for n, _ in sorted(peaks.items(), key=lambda kv: kv[1]["t"])]
    # ownership at release
    d.qpos[:] = q[-1]; d.qvel[:] = qd[-1]; mujoco.mj_forward(m, d); jacp = np.zeros((3, m.nv)); mujoco.mj_jacSite(m, d, jacp, None, sid)
    v = vsite[-1]; u = v/max(np.linalg.norm(v), 1e-9); c = (jacp.T @ u)*d.qvel
    own = {types[i].replace("_joint", ""): float(c[int(J["dof"][i])]) for i in range(nT)}; own_g = {}
    for i in range(nT): own_g[groups[i]] = own_g.get(groups[i], 0.0) + float(c[int(J["dof"][i])])
    # excursions vs limits and human ROM
    qa = J["qad"]; exc = {}
    for i, tn in enumerate(types):
        j = int(m.dof_jntid[int(J["dof"][i])]); lo, hi = np.degrees(m.jnt_range[j]); mn, mx = np.degrees(q[:, int(qa[i])].min()), np.degrees(q[:, int(qa[i])].max())
        h = HUMAN.get(tn); exc[tn.replace("_joint", "")] = dict(min=round(float(mn), 1), max=round(float(mx), 1), limit=[round(float(lo), 1), round(float(hi), 1)], beyond_human=(bool(mn < h[0] - 1 or mx > h[1] + 1) if h else None), limit_use=round(float(max((mx - 0)/(hi + 1e-9) if hi > 0 else 0, (mn - 0)/(lo - 1e-9) if lo < 0 else 0)), 2))
    Wpos = {}; 
    for i in range(nT): Wpos[groups[i]] = Wpos.get(groups[i], 0.0) + float(np.sum(np.clip(P[:, i], 0, None))*dt)
    # lag signature: power of the distal groups while the proximal segment (index 1 of the chain: torso or thigh) accelerates
    k_prox = min(1, len(seg) - 1); i_pk = int(np.argmax(W[:, k_prox])); rise = slice(max(0, i_pk - 60), max(1, i_pk))
    dist = [i for i in range(nT) if groups[i] in ("shoulder", "elbow", "wrist")] if "palm" in site_name else [i for i in range(nT) if groups[i] == "legs" and not S.get("support_mask", np.zeros(nT, bool))[i]]
    lag = dict(prox_peak_t=float(t[i_pk]), distal_power_during_prox_rise_W=float(P[rise][:, dist].sum(1).mean()) if dist else None, distal_power_after_W=float(P[i_pk:][:, dist].sum(1).mean()) if dist else None)
    speed = float(np.linalg.norm(v)); amp = [round(float(W[:, k].max()), 1) for k in range(len(seg))]
    return dict(effector=site_name, speed=speed, release_t=float(t[-1]), chain=[n for n, _ in seg], peak_omega=amp, peak_t=[peaks[n]["t"] for n, _ in seg], peak_order=order,
                strict_proximal_to_distal=bool(order == [n for n, _ in seg]), ownership_joint={k: round(v_, 2) for k, v_ in sorted(own.items(), key=lambda kv: -abs(kv[1]))[:8]}, ownership_group={k: round(v_, 2) for k, v_ in own_g.items()},
                ownership_base=float(c[:6].sum()), excursions=exc, beyond_human=[k for k, e in exc.items() if e["beyond_human"]], W_pos=Wpos, lag=lag, W_total=float(sum(Wpos.values())))
