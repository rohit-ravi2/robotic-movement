"""Static-holdability audit of the pose bank.

The pose optimizer constrained the crouch to be ground-clear and to have the centre of mass
inside the foot support polygon. It never checked whether the pose can be HELD: whether the
gravitational joint torques at rest are within actuator limits.

A crouch the robot cannot hold at rest is not a state it can start a jump from, so a bound
computed from it is not a bound on what the robot can do starting from rest.

Utilisation per joint = |gravity-compensating torque at rest| / tau_max. A pose is holdable
when every joint's utilisation is below 1.
"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, robot_bound as R

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MEN = os.environ.get("MENAGERIE_DIR", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "models", "mujoco_menagerie"))

def audit(label, mjcf, spec, bank_key):
    rb = R.RobotBody(mjcf, spec, capacity="peak")
    D = pickle.load(open(f"{ROOT}/results/pose_bank.pkl", "rb"))[bank_key]
    m, d = rb.m, rb.d
    out = []
    for rec in D["bank"]:
        for tag in ("t0", "tT"):
            # DYNAMIC COLLAPSE TEST. Inverse dynamics was unusable: mj_inverse returned
            # torques clipped to the actuator force range, giving degenerate utilisations of
            # exactly 1.00 (G1) and 0.00 (H1). Instead, hold the pose with a strong PD whose
            # torques are clipped to the real limits and see whether it collapses.
            rb.set_pose(rec[tag]); rb.drop_to_ground()
            q = np.array(rb.d.qpos)
            dd = mujoco.MjData(m); dd.qpos[:] = q; dd.qvel[:] = 0
            mujoco.mj_forward(m, dd)
            com0 = float((m.body_mass[1:, None]*dd.xipos[1:]).sum(0)[2]/rb.M)
            sat = {n: 0 for g, ns in rb.groups.items() for n in ns}
            nst = int(1.0/m.opt.timestep)
            for _ in range(nst):
                for g, names in rb.groups.items():
                    for n in names:
                        r = rb.J[n]; dof = m.jnt_dofadr[r["id"]]; qad = m.jnt_qposadr[r["id"]]
                        aid = [a for a in range(m.nu) if m.actuator_trnid[a][0] == r["id"]][0]
                        raw = 800.0*(q[qad]-dd.qpos[qad]) - 50.0*dd.qvel[dof]
                        cl = float(np.clip(raw, -r["tau"], r["tau"]))
                        if abs(raw) > r["tau"]: sat[n] += 1
                        dd.ctrl[aid] = cl
                mujoco.mj_step(m, dd)
            mujoco.mj_forward(m, dd)
            com1 = float((m.body_mass[1:, None]*dd.xipos[1:]).sum(0)[2]/rb.M)
            drop = com0-com1; qv = float(np.abs(dd.qvel).max())
            satfrac = {n: v/nst for n, v in sat.items()}
            worst = max(satfrac, key=satfrac.get) if satfrac else None
            out.append(dict(robot=label, subset=list(rec["groups"]), pose=tag,
                            com_drop=drop, final_qvel=qv,
                            holdable=bool(drop < 0.02 and qv < 0.5),
                            sat_frac=float(satfrac[worst]) if worst else 0.0,
                            worst=worst, com_z=com0))
    return out

if __name__ == "__main__":
    res = []
    res += audit("G1", f"{MEN}/unitree_g1/scene_mjx.xml", f"{ROOT}/spec/g1.yaml", "G1")
    res += audit("H1", f"{MEN}/unitree_h1/scene.xml", f"{ROOT}/spec/h1.yaml", "H1")
    json.dump(res, open(f"{ROOT}/results/holdability.json", "w"), indent=1, default=float)
    for lab in ("G1", "H1"):
        rs = [r for r in res if r["robot"] == lab]
        cr = [r for r in rs if r["pose"] == "t0"]; to = [r for r in rs if r["pose"] == "tT"]
        print(f"\n{lab}: {len(cr)} crouch poses, {len(to)} takeoff poses")
        for nm, group in (("crouch", cr), ("takeoff", to)):
            nh = [r for r in group if not r["holdable"]]
            dr = np.array([r["com_drop"] for r in group])
            print(f"  {nm:<8} NOT holdable: {len(nh)}/{len(group)} ({100*len(nh)/len(group):.0f}%)"
                  f"   COM drop median {np.median(dr)*100:.1f} cm max {dr.max()*100:.1f} cm")
        worst = sorted(cr, key=lambda r: -r["com_drop"])[:4]
        print(f"  worst crouches (largest COM collapse in 1 s):")
        for w in worst:
            print(f"    drop {w['com_drop']*100:5.1f} cm  qvel {w['final_qvel']:5.2f}  "
                  f"saturated {w['sat_frac']*100:3.0f}% of the time at {str(w['worst']):<24} "
                  f"COMz {w['com_z']:.3f}  {'+'.join(w['subset'])}")
        from collections import Counter
        c = Counter(r["worst"] for r in cr if not r["holdable"])
        print(f"  binding joint among non-holdable crouches: {dict(c)}")
