"""J3: atlas builder daemon. Sweeps demand grids for each task with the generic robust optimizer, appending faithful records.
Runs on spare cores; restartable (skips (demand, seed) already present). python engine/build_atlas.py [jobs.json]"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from multiprocessing import Pool
from engine import core, task as TK, atlas as AT

def default_grid():
    jobs = []
    # jump: landed, prior vs native, 4 seeds each (G1)
    for prior in (True, False):
        for s in range(1, 5): jobs.append(dict(robot="G1", task="jump", demand=dict(land=True, prior=prior), seed=s, iters=100, warm="robust4" if prior else None))
    # throw: max speed x range-of-motion scale (the form-factor axis), prior, 3 seeds
    for rom in (1.0, 0.8, 0.6, 0.45):
        for s in range(1, 4): jobs.append(dict(robot="G1", task="throw", demand=dict(target_speed=None, rom_scale=rom, prior=True, omega=20.0), seed=s, iters=150, warm=None))
    # throw: demand sweep (target speed) at full ROM, prior, 2 seeds
    for tg in (6.0, 10.0, 14.0, 18.0, 22.0):
        for s in range(1, 3): jobs.append(dict(robot="G1", task="throw", demand=dict(target_speed=tg, rom_scale=1.0, prior=True, omega=20.0), seed=s, iters=150, warm=None))
    return jobs

def make_task(name, robot):
    if name == "jump": from tasks.jump import JumpTask; return JumpTask(robot)
    if name == "reach": from tasks.reach import ReachTask; return ReachTask(robot)
    from tasks.throw import ThrowTask; return ThrowTask(robot)

def warm_record(job):
    w = job.get("warm")
    if w == "robust4":
        R = {r["seed"]: r for r in json.load(open(f"{core.RES}/phase14/jump14_robust4.json"))}; return R.get(job["seed"]) or R[1]
    if isinstance(w, str) and w.startswith("file:"):
        R = json.load(open(w[5:])); R = {r.get("seed", i + 1): r for i, r in enumerate(R)}; return R.get(job["seed"]) or next(iter(R.values()))
    return None

def run_job(job):
    t = make_task(job["task"], job["robot"]); t0 = time.time()
    out = TK.optimize(t, job["demand"], job["seed"], iters=job["iters"], warm=warm_record(job), robust=True)
    out.update(seed=job["seed"]); AT.append(job["robot"], job["task"], out, builder="build_atlas v0", warm=job.get("warm"))
    return job["task"], job["demand"], job["seed"], round(out["value"], 3), out["ok_all"], round(time.time() - t0)

if __name__ == "__main__":
    jobs = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else default_grid()
    done = set()
    for j in jobs:
        for r in AT.load(j["robot"], j["task"]):
            if AT.key(r["demand"]) == AT.key(j["demand"]) and r.get("seed") == j["seed"]: done.add((j["task"], AT.key(j["demand"]), j["seed"]))
    todo = [j for j in jobs if (j["task"], AT.key(j["demand"]), j["seed"]) not in done]
    print(f"atlas: {len(jobs)} jobs, {len(todo)} to run", flush=True)
    with Pool(int(os.environ.get("NPROC", "6"))) as p:
        for res in p.imap_unordered(run_job, todo): print("done", res, flush=True)
