"""Held-out onboarding acceptance (2026-10-16): T.setup(robot) builds; after the plant's settle the robot STANDS for 2 s at dt 1 ms and
0.5 ms under the plant's hold (every actuator an implicit position servo at the settled pose, kp = I_eff/dt^2 [capped per ROBOTS],
force-limited at the joint's rating: branch_i._as_servo, the hold used by the settle); body_spec prints with rating sources.
Writes results/heldout/accept_<robot>.json.  usage: python heldout/accept_stand.py Apollo"""
import os, sys, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
import throw as T, branch_i as BI
from engine import sprint as SP, sprint_ceiling as SC
robot = sys.argv[1]; out = {}
for dt in (0.001, 0.0005):
    S = SP.plant(robot, dt); m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; d.qvel[:] = S["v_settled"]
    Minv = S["Minv_key"]; J = S["jarr"]; cfg = BI.ROBOTS[robot]
    saved = (m.actuator_gainprm.copy(), m.actuator_biasprm.copy(), m.actuator_gaintype.copy(), m.actuator_biastype.copy(), m.actuator_ctrlrange.copy(), m.actuator_forcerange.copy())
    for i, aid in enumerate(J["aid"]):
        dof = J["dof"][i]; ieff = 1.0/float(Minv[dof, dof]); kp = ieff/dt**2
        if cfg.get("hold_sat_rad"): kp = min(kp, J["tau"][i]/cfg["hold_sat_rad"])
        BI._as_servo(m, aid, kp, 2*np.sqrt(kp*ieff), J["tau"][i])
    for a in range(m.nu): d.ctrl[a] = S["q_settled"][m.jnt_qposadr[int(m.actuator_trnid[a][0])]]
    mujoco.mj_forward(m, d); c0 = d.subtree_com[1].copy(); z0 = float(d.qpos[2]); vmax = 0.0; tmax = 0.0; nonfoot = False
    for k in range(int(2.0/dt)):
        mujoco.mj_step(m, d); vmax = max(vmax, float(np.abs(d.qvel).max())) if k > int(0.2/dt) else vmax
        tmax = max(tmax, float(np.max(np.abs(d.actuator_force[J["aid"]])/J["tau"])))
        for ci in range(d.ncon):
            c = d.contact[ci]; o = c.geom2 if c.geom1 == S["floor"] else c.geom1 if c.geom2 == S["floor"] else None
            if o is not None and o not in S["foot_geoms"] and o in S["robot_geoms"]: nonfoot = True
    up = float(d.xmat[1].reshape(3, 3)[2, 2])
    out[str(dt)] = dict(settle_qdot_max=S["settle_qdot_max"], settle_certificate=S.get("settle_certificate"), com_drift=float(np.linalg.norm(d.subtree_com[1] - c0)), z_drift=float(d.qpos[2] - z0),
                        qvel_max_after_0p2s=vmax, upright_cos=up, max_torque_util=tmax, nonfoot_contact=nonfoot, z=float(d.qpos[2]),
                        ok=bool(np.linalg.norm(d.subtree_com[1] - c0) < 0.02 and up > 0.98 and not nonfoot and vmax < 0.1))
    m.actuator_gainprm[:], m.actuator_biasprm[:], m.actuator_gaintype[:], m.actuator_biastype[:], m.actuator_ctrlrange[:], m.actuator_forcerange[:] = saved
    print(robot, dt, out[str(dt)], flush=True)
B = SC.body_spec(robot); sp = yaml.safe_load(open(BI.ROBOTS[robot]["spec"]))
src = {j: dict(tau=sp["joints"][j]["tau_max_peak"]["value"], tau_src=sp["joints"][j]["tau_max_peak"]["src"], omega=sp["joints"][j]["omega_max"]["value"],
               omega_src=sp["joints"][j]["omega_max"]["src"], omega_tag=sp["joints"][j]["omega_max"]["tag"]) for j in SP.driven(robot)}
print("body_spec:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in B.items()}))
for j, r in src.items(): print(f"  {j:28s} tau {r['tau']:7.1f} ({r['tau_src']})  omega {r['omega']:6.2f} ({r['omega_src']}, {r['omega_tag']})")
json.dump(dict(robot=robot, stand=out, body_spec=B, ratings=src, accepted=all(v["ok"] for v in out.values())), open(f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/../results/heldout/accept_{robot}.json", "w"), indent=1, default=float)
