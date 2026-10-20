"""Rebuild the pose bank with (a) the corrected support polygon and (b) static-equilibrium
feasibility of the START (crouch) pose as a HARD constraint inside the pose optimizer.

Gate 1 found the top-scoring poses of the old bank were systematically infeasible (G1 top-5
1/5, H1 0/5 holdable), 100% centre-of-mass-outside-true-support.

Only the crouch must be holdable: the jump starts from rest there. The takeoff pose is passed
through dynamically, so its static certificate is RECORDED, not required.
Output: results/pose_bank_certified.pkl. The old results/pose_bank.pkl is kept as history."""
import sys, os, json, pickle, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, robot_bound as R, bank as BK, plant as P, branch_i as BI
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MEN = os.environ.get("MENAGERIE_DIR", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "models", "mujoco_menagerie"))
CFG = {"G1": (f"{MEN}/unitree_g1/scene_mjx.xml", f"{ROOT}/spec/g1.yaml", True),
       "H1": (f"{MEN}/unitree_h1/scene.xml", f"{ROOT}/spec/h1.yaml", False)}
T = 0.25


def make_body(lab):
    mjcf, spec, conv = CFG[lab]
    rb = R.RobotBody(mjcf, spec, capacity="peak")
    if conv: BI.to_torque_motors(rb)
    m = rb.m; dd = mujoco.MjData(m)
    cg = P.PlantContractVerifier(m, dd, P.PlantContract(mjcf, "torque", [], [])).collision_geoms()
    def cert(q):
        return P.static_certificate(m, dd, q, mu=0.9, coll_geoms=cg)
    return rb, cert


_B = {}
def solve_sub(args):
    lab, sub, inh = args
    if lab not in _B:
        _B[lab] = make_body(lab)
    rb, cert = _B[lab]
    R.CROUCH_CERTIFIER = lambda q: cert(q)["status"] == "STATIC_FEASIBLE"
    res = R.optimize_subset(rb, sub, T, n_scan=1500, n_restart=6,
                            seed=abs(hash(sub)) % 9999, extra_seeds=inh)
    t0 = res.get("t_crouch", {}); tT = res.get("t_takeoff", {})
    rec = BK.capture(rb, t0, tT, sub)
    rb.set_pose(t0); rb.drop_to_ground(); c0 = cert(np.array(rb.d.qpos))
    rb.set_pose(tT); rb.drop_to_ground(); cT = cert(np.array(rb.d.qpos))
    rec["cert_crouch"] = {k: v for k, v in c0.items() if not isinstance(v, np.ndarray)}
    rec["cert_takeoff"] = {k: v for k, v in cT.items() if not isinstance(v, np.ndarray)}
    rec["analytic_v"] = float(res["v_takeoff"]); rec["n_seeds"] = int(res["n_seeds"])
    return sub, rec, (t0, tT)


def build(lab, pool):
    rb, _ = make_body(lab)
    gs = list(rb.groups); bank = []; solved = {}
    for r_ in range(1, len(gs)+1):               # level by level so subsets seed supersets
        jobs = []
        for sub in itertools.combinations(gs, r_):
            inh = [solved[k] for k in (tuple(x for x in sub if x != dr) for dr in sub) if k in solved]
            jobs.append((lab, sub, inh))
        for sub, rec, tt in pool.map(solve_sub, jobs):
            solved[sub] = tt; bank.append(rec)
        print(f"  {lab} level {r_}: {len(jobs)} subsets", flush=True)
    return rb, bank


if __name__ == "__main__":
    out = {}
    with Pool(int(os.environ.get("NPROC", "14"))) as pool:
        for lab in ("G1", "H1"):
            t0 = time.time(); rb, bank = build(lab, pool)
            out[lab] = dict(bank=bank, groups=list(rb.groups),
                            caps={n: [rb.J[n]["tau"], rb.J[n]["omega"]]
                                  for g in rb.groups for n in rb.groups[g]},
                            mass=np.array(rb.m.body_mass[1:]).tolist(),
                            group_index={g: list(v) for g, v in rb.groups.items()})
            nf = sum(r["cert_crouch"]["status"] == "STATIC_FEASIBLE" for r in bank)
            print(f"{lab}: {len(bank)} poses, crouch certified {nf}/{len(bank)}, "
                  f"{time.time()-t0:.0f}s", flush=True)
    pickle.dump(out, open(f"{ROOT}/results/pose_bank_certified_v2.pkl", "wb"))
    print("saved pose_bank_certified_v2.pkl", flush=True)
