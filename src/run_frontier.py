"""Task 2b: monotone frontier + required_depth(D) for G1 and H1, both capacity sets."""
import sys, os, json, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, robot_bound as R
from multiprocessing import Pool

MEN = os.environ.get("MENAGERIE_DIR", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "models", "mujoco_menagerie"))
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CFG = {"G1": (f"{MEN}/unitree_g1/scene_mjx.xml", f"{ROOT}/spec/g1.yaml"),
       "H1": (f"{MEN}/unitree_h1/scene.xml",     f"{ROOT}/spec/h1.yaml")}
T_PUSH = 0.25

def run(a):
    robot, cap = a
    mj, sp = CFG[robot]
    rb = R.RobotBody(mj, sp, capacity=cap)
    t0 = time.time()
    rows = R.frontier(rb, T_PUSH, n_scan=900, n_restart=6)
    viol = R.check_monotone(rows)
    out = []
    for r in rows:
        out.append(dict(robot=robot, capacity=cap, T_push=T_PUSH, subset=list(r["subset"]),
                        k=len(r["subset"]), v_takeoff=r["v_takeoff"], height=r["height"],
                        W_used=r["W_used"], PE=r["PE"], dz=r["dz"],
                        n_power_bound=r["n_power_bound"], n_seeds=r["n_seeds"],
                        spread=r["spread"], converged=bool(r["converged"]),
                        t_crouch={k: float(v) for k, v in (r.get("t_crouch") or {}).items()},
                        t_takeoff={k: float(v) for k, v in (r.get("t_takeoff") or {}).items()}))
    return dict(robot=robot, capacity=cap, rows=out, violations=viol, secs=time.time()-t0)

if __name__ == "__main__":
    jobs = [(r, c) for r in ("G1", "H1") for c in ("peak", "continuous")]
    with Pool(4) as p: res = p.map(run, jobs)
    json.dump(res, open(f"{ROOT}/results/frontier_monotone.json", "w"), indent=1)
    for r in res:
        print(f"{r['robot']}/{r['capacity']}: {len(r['rows'])} subsets, {r['secs']:.0f}s, "
              f"monotonicity violations: {len(r['violations'])}", flush=True)
        for v in r["violations"][:5]:
            print(f"    {v[0]} ({v[1]:.3f}) > superset {v[2]} ({v[3]:.3f}) by {v[4]:.3f}", flush=True)
