"""The DECLARED plant for every G1 dynamic experiment, and its verification.

Every dynamic campaign calls verify_or_die(S, caps_scale) before spending compute. Results
carry the contract hash; a plant change changes the hash, so old and new outputs can never be
silently compared as the same experiment."""
import sys, os, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, plant as P, branch_i as BI


def contract(S):
    m = S["m"]
    jn = lambda a: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a][0]))
    torque = [jn(a) for a in S["driven"]]
    held = [jn(a) for (a, *_r) in S["held"]]
    c = P.PlantContract(S.get("mjcf", BI.MJCF), {"torque": torque, "implicit_position_hold": held},
                        controlled=torque, locked=held, penetration_tol=0.002, mu=float(m.geom_friction[S["floor"]][0]),
                        timestep=float(m.opt.timestep))
    return c


def plant_hash(S):
    m = S["m"]
    blob = json.dumps(dict(contract=contract(S).hash(), dt=float(m.opt.timestep),
                           integ=int(m.opt.integrator), it=int(m.opt.iterations),
                           mass=float(sum(m.body_mass)), q0=np.round(S["q_settled"], 6).tolist(),
                           hold_gain="1.0*Ieff/dt^2", settle=4.0), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def verify(S, caps_scale=None):
    m = S["m"]; BI.settle(S); d = mujoco.MjData(m)
    c = contract(S); v = P.PlantContractVerifier(m, d, c)
    BI._set_capacity_limits(m, S, caps_scale)
    v.pctv1_actuator_semantics()
    v.pctv2_control_coverage()
    v.pctv3_ground_placement(S["q_settled"])
    # PCTV-4: zero torque on study joints, held joints still held -> must not rise, no motor force
    d.qpos[:] = S["q_settled"]; d.qvel[:] = 0; mujoco.mj_forward(m, d)
    z0 = float(d.subtree_com[0][2])
    for (a, *_r) in S["held"]: d.ctrl[a] = S["q_settled"][_r[1]]
    for _ in range(int(0.3/m.opt.timestep)):
        for aid in S["driven"]: d.ctrl[aid] = 0.0
        mujoco.mj_step(m, d)
    fm = max(abs(float(d.actuator_force[a])) for a in S["driven"])
    v._rec("PCTV-4", (float(d.subtree_com[0][2]) <= z0 + 1e-3) and fm < 1e-9,
           f"zero study torque: COM {z0:.3f}->{float(d.subtree_com[0][2]):.3f}, max motor force {fm:.1e}")
    # PCTV-5: static hold at the settled start under the recorded holding torques
    x = BI.hold_params(S, caps_scale)
    r = BI.rollout(x, set(S["types"]), caps_scale, record=True, S=S)
    dz = abs(r["trace"]["z"][-1]-r["trace"]["z"][0]); vzm = float(np.max(np.abs(r["trace"]["vz"])))
    v._rec("PCTV-5", dz < 0.005 and vzm < 0.02 and S["settle_qdot_max"] < 0.05,
           f"0.5 s hold: COM drift {dz*1000:.2f} mm, max|vz| {vzm*1000:.1f} mm/s, "
           f"start qdot {S['settle_qdot_max']:.4f} rad/s")
    # PCTV-10: the settled state IS the declared start (a robot lying on the floor is "stable"
    # too; PCTV-5 alone passed exactly that case for H1)
    d0 = mujoco.MjData(m); d0.qpos[:] = S["q0"]; mujoco.mj_forward(m, d0)
    rb = S["rb"]; rb.d.qpos[:] = S["q0"]; mujoco.mj_forward(m, rb.d); rb.d.qpos[2] -= rb.lowest_z(); mujoco.mj_forward(m, rb.d)
    z_decl = float(rb.d.subtree_com[0][2]); up_decl = float(rb.d.xmat[1][8])
    d1 = mujoco.MjData(m); d1.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d1)
    z_set = float(d1.subtree_com[0][2]); up_set = float(d1.xmat[1][8])
    v._rec("PCTV-10", abs(z_set - z_decl) < 0.01 and abs(up_set - up_decl) < 0.05,
           f"settled COM z {z_set:.3f} vs declared {z_decl:.3f}; upright {up_set:.3f} vs {up_decl:.3f}")
    v.pctv6_known_torque(S["driven"][0])
    decl = {}
    for t in S["types"]:
        st = caps_scale[t][0] if (caps_scale and t in caps_scale) else 1.0
        for (jid, dof, qad, tau, om, sgn, aid) in S["tmap"][t]: decl[aid] = tau*st
    v.pctv9_capacity_realised(decl)
    return v, c


def verify_or_die(S, caps_scale=None):
    v, c = verify(S, caps_scale)
    if not v.summary():
        raise RuntimeError("PLANT CONTRACT FAILED: " + json.dumps(v.results, default=str))
    return dict(contract_hash=c.hash(), plant_hash=plant_hash(S), checks=v.results)


if __name__ == "__main__":
    S = BI.setup()
    out = {}
    for tag, sc in (("baseline", None),
                    ("proximal_strong", {t: (s, s) for t, s in zip(
                        ["ankle_pitch_joint", "knee_joint", "hip_pitch_joint", "waist_pitch_joint",
                         "shoulder_pitch_joint", "elbow_joint"], [2.2, 1.7, 1.3, 0.9, 0.6, 0.45])})):
        v, c = verify(S, sc)
        out[tag] = dict(passed=v.summary(), contract_hash=c.hash(), plant_hash=plant_hash(S),
                        checks=v.results)
        print(tag, "PASS" if v.summary() else "FAIL")
        for k, r in v.results.items(): print(f"  {k}: {'PASS' if r['passed'] else 'FAIL'}  {r['detail']}")
    json.dump(out, open(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/plant_contract_g1.json"), "w"),
              indent=1, default=str)
