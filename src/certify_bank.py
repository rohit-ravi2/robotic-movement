"""Gate 1 — static-equilibrium certification of every pose in the analytic pose bank."""
import sys, os, json, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, plant as P, robot_bound as R, branch_i as BI

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MEN = os.environ.get("MENAGERIE_DIR", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "models", "mujoco_menagerie"))
CFG = {"G1": (f"{MEN}/unitree_g1/scene_mjx.xml", f"{ROOT}/spec/g1.yaml", True),
       "H1": (f"{MEN}/unitree_h1/scene.xml", f"{ROOT}/spec/h1.yaml", False)}

def certify_robot(lab):
    mjcf, spec, convert = CFG[lab]
    rb = R.RobotBody(mjcf, spec, capacity="peak")
    if convert: BI.to_torque_motors(rb)
    m = rb.m; d = mujoco.MjData(m)
    v = P.PlantContractVerifier(m, d, P.PlantContract(mjcf, "torque", [], []))
    cg = v.collision_geoms()
    D = pickle.load(open(f"{ROOT}/results/pose_bank.pkl", "rb"))[lab]
    out = []
    for rec in D["bank"]:
        for tag in ("t0", "tT"):
            rb.set_pose(rec[tag]); rb.drop_to_ground()
            q = np.array(rb.d.qpos)
            try:
                c = P.static_certificate(m, d, q, mu=0.9, coll_geoms=cg)
            except Exception as e:
                c = dict(status="NUMERICALLY_UNRESOLVED", detail=str(e)[:80])
            d.qpos[:] = q; d.qvel[:] = 0; mujoco.mj_forward(m, d)
            M = float(sum(m.body_mass[1:]))
            com = (m.body_mass[1:, None]*d.xipos[1:]).sum(0)/M
            pts = P.potential_contacts(m, d, cg, 0.003)
            xs = [p[0] for p in pts] or [0.0]
            out.append(dict(robot=lab, subset=list(rec["groups"]), pose=tag,
                            status=c["status"], s=c.get("s"),
                            torque_margin=c.get("torque_margin"),
                            ncon=c.get("ncon", 0), com_z=float(com[2]),
                            com_x=float(com[0]),
                            support_x=[float(min(xs)), float(max(xs))],
                            n_pts=len(pts)))
    return out

if __name__ == "__main__":
    res = []
    for lab in ("G1", "H1"):
        t0 = time.time(); r = certify_robot(lab); res += r
        print(f"{lab}: {len(r)} poses certified in {time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(f"{ROOT}/results/static_certificates.json", "w"), indent=1, default=float)
    import collections
    for lab in ("G1", "H1"):
        rs = [x for x in res if x["robot"] == lab]
        for tag, nm in (("t0", "crouch"), ("tT", "takeoff")):
            g = [x for x in rs if x["pose"] == tag]
            c = collections.Counter(x["status"] for x in g)
            feas = [x for x in g if x["status"] == "STATIC_FEASIBLE"]
            tm = np.array([x["torque_margin"] for x in feas]) if feas else np.array([0.0])
            print(f"  {lab} {nm:<8} {len(feas)}/{len(g)} STATIC_FEASIBLE "
                  f"({100*len(feas)/len(g):.0f}%)  torque margin median {np.median(tm):.3f} "
                  f"min {tm.min():.3f} | {dict(c)}", flush=True)
