"""Phase 10 B4 (second simulator), MuJoCo side.
(1) export_mjcf(): write a Drake-parsable MJCF from the COMPILED G1 MjModel: explicit inertials,
    hinge joints (axis, range, armature, damping), the floor, and ONLY the foot collision
    primitives (the MJX model's floor contact set). Visual meshes are dropped (Drake's MJCF parser
    accepts .obj only and they carry no mass or contact in scene_mjx).
(2) controller_spec(): everything the rollout's control law needs, simulator-agnostic: knot
    table, knot_dt, lag alpha, torque-speed envelope, initial lag state, tracker reference/gains,
    held-joint PD targets/gains/limits, settled start state; plus the MuJoCo 1 ms trace (COM vz,
    floor normal force, driven-joint q and applied torque) for comparison."""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, branch_i as BI, mocap_baseline as MB

OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase10/xsim")

def export_mjcf(S, path):
    m = S["m"]
    fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    # the MJX model's floor contact set = geoms in an explicit <pair> with the floor
    foot = set(int(m.pair_geom2[i] if m.pair_geom1[i] == fl else m.pair_geom1[i]) for i in range(m.npair)
               if fl in (m.pair_geom1[i], m.pair_geom2[i]))
    def nm(o, i): return mujoco.mj_id2name(m, o, i)
    def f(a): return " ".join(f"{float(x):.9g}" for x in a)
    kids = {b: [] for b in range(m.nbody)}
    for b in range(1, m.nbody): kids[int(m.body_parentid[b])].append(b)
    lines = ['<mujoco model="g1_drake_export">', '  <compiler angle="radian"/>', '  <worldbody>',
             '    <geom name="floor" type="plane" size="20 20 0.1" friction="1 0.005 0.0001"/>']
    def emit(b, ind):
        s = " "*ind
        lines.append(f'{s}<body name="{nm(mujoco.mjtObj.mjOBJ_BODY, b)}" pos="{f(m.body_pos[b])}" quat="{f(m.body_quat[b])}">')
        lines.append(f'{s}  <inertial pos="{f(m.body_ipos[b])}" quat="{f(m.body_iquat[b])}" mass="{m.body_mass[b]:.9g}" diaginertia="{f(m.body_inertia[b])}"/>')
        for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
            jn = nm(mujoco.mjtObj.mjOBJ_JOINT, j); t = m.jnt_type[j]
            if t == mujoco.mjtJoint.mjJNT_FREE:
                lines.append(f'{s}  <freejoint name="{jn}"/>')
            elif t == mujoco.mjtJoint.mjJNT_HINGE:
                dof = m.jnt_dofadr[j]
                lim = f' limited="true" range="{f(m.jnt_range[j])}"' if m.jnt_limited[j] else ' limited="false"'
                lines.append(f'{s}  <joint name="{jn}" type="hinge" pos="{f(m.jnt_pos[j])}" axis="{f(m.jnt_axis[j])}"{lim} '
                             f'armature="{m.dof_armature[dof]:.9g}" damping="{m.dof_damping[dof]:.9g}"/>')
            else:
                raise ValueError(f"unsupported joint type {t} at {jn}")
        for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
            if g not in foot: continue
            t = m.geom_type[g]; sz = m.geom_size[g]
            gt = {mujoco.mjtGeom.mjGEOM_SPHERE: ("sphere", sz[:1]), mujoco.mjtGeom.mjGEOM_CAPSULE: ("capsule", sz[:2]),
                  mujoco.mjtGeom.mjGEOM_BOX: ("box", sz[:3]), mujoco.mjtGeom.mjGEOM_CYLINDER: ("cylinder", sz[:2])}[t]
            lines.append(f'{s}  <geom name="{nm(mujoco.mjtObj.mjOBJ_GEOM, g)}" type="{gt[0]}" size="{f(gt[1])}" pos="{f(m.geom_pos[g])}" '
                         f'quat="{f(m.geom_quat[g])}" mass="0" friction="1 0.005 0.0001"/>')
        for c in kids[b]: emit(c, ind + 2)
        lines.append(f'{s}</body>')
    for c in kids[0]: emit(c, 4)
    lines += ['  </worldbody>', '</mujoco>']
    open(path, "w").write("\n".join(lines))
    return path

def controller_spec(S, ref, x, om, ts, horizon):
    """Mirror of branch_i.rollout's control law + mocap_baseline.Tracker, as plain arrays."""
    m = S["m"]; J = S["jarr"]; ti = J["ti"]; NT = len(S["types"])
    kn = np.asarray(x).reshape(NT, -1); NK = kn.shape[1]
    tr = MB.Tracker(S, ref, omega=om, tscale=ts)
    dt = m.opt.timestep
    jname = lambda dof: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.dof_jntid[dof]))
    act = np.array([S["a_hold"][t] for t in S["types"]])
    held = []
    for (a, dof, qad, tau, kp, kv) in S["held"]:
        held.append(dict(joint=jname(dof), target=float(S["q_settled"][qad]), kp=float(m.actuator_gainprm[a][0]),
                         kd=float(-m.actuator_biasprm[a][2]), limit=float(abs(m.actuator_forcerange[a][1]))))
    return dict(dt=dt, horizon=horizon, knot_dt=horizon/NK, kn=kn, ti=ti, sgn=J["sgn"], tau=J["tau"], om=J["om"],
                driven=[jname(int(dof)) for dof in J["dof"]], alpha=1.0 - np.exp(-dt/0.040),
                act0=act[ti]*J["sgn"], QR=tr.QR, QDR=tr.QDR, tr_kp=tr.kp, tr_kd=tr.kd, held=held,
                q_settled=S["q_settled"], v_settled=S["v_settled"],
                qpos_joint_names=[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)],
                jnt_qposadr=np.array(m.jnt_qposadr), jnt_dofadr=np.array(m.jnt_dofadr), M=S["M"])

def mujoco_trace(S, ref, x, om, ts):
    """1 ms MuJoCo run of the same solution, recording what the comparison needs each step."""
    tr = MB.Tracker(S, ref, omega=om, tscale=ts); J = S["jarr"]; log = dict(t=[], q=[], qd=[], tau=[])
    def fn(t, d, S_):
        log["t"].append(t); log["q"].append(d.qpos[J["qad"]].copy()); log["qd"].append(d.qvel[J["dof"]].copy())
        log["tau"].append(d.ctrl[J["aid"]].copy())
        return tr(t, d, S_)
    r = BI.rollout(np.asarray(x), set(S["types"]), None, record=True, S=S, ctrl_fn=fn, horizon=ref["horizon"], score="best")
    return r, {k: np.array(v) for k, v in log.items()}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    d = json.load(open("../results/phase9/baseline9.json"))
    picks = []
    for meth in ("APRIOR", "OURS", "B1"):
        rows = [r for r in d if r["method"] == meth and r["valid"] and r["task"] == "T2"]
        picks += rows[::2] if meth == "APRIOR" else rows[::8]
    exported = False
    for r in picks:
        c = MB.load_clip(r["clip"], "retargeted_g1_all.pkl"); ref = MB.analytic_clip(c) if r["method"] == "APRIOR" else c
        S = BI.setup(c["start"]); BI.apply_dt(S, 0.001)
        if not exported:
            export_mjcf(S, f"{OUT}/g1_drake.xml"); exported = True
        spec = controller_spec(S, ref, r["x"], r["omega"], r["tscale"], ref["horizon"])
        o, lg = mujoco_trace(S, ref, r["x"], r["omega"], r["tscale"])
        BI.apply_dt(S, 0.004)
        key = f'{r["method"]}_{r["clip"].split("@")[0]}_{r["seed"]}'
        pickle.dump(dict(meta=dict(method=r["method"], clip=r["clip"], seed=r["seed"], v_stored=r["v"], v_mujoco=o["v"],
                                   takeoff_t=o["takeoff_t"], grf_bw=o["grf_bw"], reasons=o["validity"]["reasons"]),
                         spec=spec, mj=dict(trace=o["trace"], **lg)), open(f"{OUT}/{key}.pkl", "wb"))
        print(key, "mujoco v", round(o["v"], 3), "stored", round(r["v"], 3), flush=True)
