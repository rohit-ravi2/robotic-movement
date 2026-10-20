"""Adversarial verifier for Q3_G1_continuous (2026-10-17). Replays two headline z on FRESH plants (new process per job, env set before
import, plant built from the Menagerie G1 MJX model at the job's dt), exact extrapolated stride, at 1 ms and 0.5 ms, plus small
perturbations of SIMULATION state/choices only (no hardware change): base vx +0.05 m/s at start, joint-angle noise 2e-3 rad (seed 1),
matched actuator lag +25%. Also diffs the built plant against stock scene_mjx.xml. Output results/overnight/VERIFY_Q3_adv.json."""
import os, sys, json, subprocess
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); MEN = os.environ.get("MENAGERIE_DIR", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "models", "mujoco_menagerie"))
CHILD = r'''
import os, sys, json; sys.path.insert(0, "src")
import numpy as np, mujoco
from engine import sprint as SP
pth, ix, dt, pert = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]; R = json.load(open(pth))["rungs"][ix]
st = None
if R.get("stride_extrapolated"):
    st = dict(SP.select_stride(SP.body("G1"), float(R["stride"]["extrapolated_from"])), extrapolated_from=R["stride"]["extrapolated_from"])
_is = SP.initial_state
def initial_state(S, Rr, p):
    q0, v0, ph = _is(S, Rr, p); q0 = np.array(q0, float); v0 = np.array(v0, float)
    if pert == "vx": v0[0] += 0.05
    if pert == "qnoise": q0[7:] += 2e-3*np.random.default_rng(1).standard_normal(len(q0) - 7)
    return q0, v0, ph
SP.initial_state = initial_state
P = SP.Sprint("G1", R["v"], stride=st); S = P.S[dt]; m = S["m"]
stock = mujoco.MjModel.from_xml_path("%s/unitree_g1/scene_mjx.xml")
def bid(mm, n): return mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, n)
dev = []
for b in range(stock.nbody):
    n = mujoco.mj_id2name(stock, mujoco.mjtObj.mjOBJ_BODY, b); j = bid(m, n)
    if j < 0: dev.append(("missing_body", n)); continue
    if abs(stock.body_mass[b] - m.body_mass[j]) > 1e-9: dev.append(("mass", n, float(stock.body_mass[b]), float(m.body_mass[j])))
for jn in range(stock.njnt):
    n = mujoco.mj_id2name(stock, mujoco.mjtObj.mjOBJ_JOINT, jn); j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    if j < 0: dev.append(("missing_joint", n)); continue
    if np.abs(stock.jnt_range[jn] - m.jnt_range[j]).max() > 1e-9: dev.append(("range", n))
    if abs(stock.dof_armature[stock.jnt_dofadr[jn]] - m.dof_armature[m.jnt_dofadr[j]]) > 1e-9: dev.append(("armature", n))
    if abs(stock.dof_damping[stock.jnt_dofadr[jn]] - m.dof_damping[m.jnt_dofadr[j]]) > 1e-9: dev.append(("damping", n, float(stock.dof_damping[stock.jnt_dofadr[jn]]), float(m.dof_damping[m.jnt_dofadr[j]])))
    if abs(stock.dof_frictionloss[stock.jnt_dofadr[jn]] - m.dof_frictionloss[m.jnt_dofadr[j]]) > 1e-9: dev.append(("frictionloss", n))
tau = {t: float(S["jarr"]["tau"][i]) for i, t in enumerate(S["types"])}
stk = {mujoco.mj_id2name(stock, mujoco.mjtObj.mjOBJ_JOINT, stock.actuator_trnid[a, 0]): float(stock.jnt_actfrcrange[stock.actuator_trnid[a, 0], 1]) for a in range(stock.nu)}
r = P.run(np.asarray(R["z"]), dt); s = SP.summarize(r)
print("JSON" + json.dumps(dict(s, ratings=SP.RATINGS, act_tau=SP.ACT_TAU, timestep=float(m.opt.timestep), plant_dev=dev[:20], n_dev=len(dev),
      tau_used=tau, tau_stock=stk, geom_friction_floor=[float(x) for x in m.geom_friction[0]], cone=int(m.opt.cone)), default=float))
''' % MEN
def job(a):
    pth, ix, lag, rt, dt, pert = a
    lag_ = lag*1.25 if pert == "lag125" else lag
    env = dict(os.environ, SPRINT_LAG=str(lag_), SPRINT_RATINGS=rt, OMP_NUM_THREADS="1")
    o = subprocess.run(["nice", "-n", "10", sys.executable, "-c", CHILD, pth, str(ix), str(dt), pert], cwd=ROOT, env=env, capture_output=True, text=True)
    L = [l for l in o.stdout.splitlines() if l.startswith("JSON")]
    r = {**(json.loads(L[0][4:]) if L else dict(error=o.stderr[-800:])), **dict(src=pth, rung=ix, lag=lag_, ratings_job=rt, dt=dt, pert=pert)}
    print(os.path.basename(pth), ix, lag_, rt, dt, pert, r.get("value"), r.get("valid"), r.get("reasons"), r.get("periodic"), r.get("slip_max"), r.get("flight_mean"), r.get("n_dev"), r.get("error", "")[-300:], flush=True)
    return r
if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    S = "results/sprint/"
    specs = [(S + "ladder_G1_cont_lag2ms_c2.json", 2, 0.002, "continuous"), (S + "ladder_G1_cont_lag10ms.json", 0, 0.010, "continuous")]
    jobs = [(p, i, l, rt, dt, pe) for p, i, l, rt in specs for pe in ("none", "vx", "qnoise", "lag125") for dt in (0.001, 0.0005)]
    with ThreadPoolExecutor(2) as ex: res = list(ex.map(job, jobs))
    json.dump(res, open(f"{ROOT}/results/overnight/VERIFY_Q3_adv.json", "w"), indent=1)
