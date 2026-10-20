"""Q4: re-run the stored best candidates' fresh-process validity replays (q4_sled_run.fresh_eval, unchanged protocol: plant built at each
dt, seeds 0-3) and record t_held, which the run JSON did not store (needed for the pre-registered H_T check: failures at t_held <= 1 s).
usage: python q4_sled_theld.py push MU payload1,payload2 -> results/overnight/Q4/sled_<dir>_mu<MU>_theld.json"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiprocessing as mp
import q4_sled_run as Q
if __name__ == "__main__":
    nm, mu = sys.argv[1], float(sys.argv[2]); sgn = 1 if nm == "push" else -1; out = {}
    for mass in [float(x) for x in sys.argv[3].split(",")]:
        bz = json.load(open(f"{Q.OUT}/sled_{nm}_mu{mu:g}_{mass:g}kg.json"))["best_z"]
        jobs = [(mu, mass, sgn, bz, 4.0, dt, s) for s in (0, 1, 2, 3) for dt in (0.001, 0.0005)]
        with mp.get_context("spawn").Pool(1, maxtasksperchild=1) as p: C = p.map(Q.fresh_eval, jobs, chunksize=1)
        out[f"{mass:g}"] = [dict(seed=j[6], dt=j[5], **c) for j, c in zip(jobs, C)]
        print(mass, [(r["seed"], r["dt"], r["reason"], round(r["t_held"], 2), round(r["disp"], 3)) for r in out[f"{mass:g}"]], flush=True)
    json.dump(out, open(f"{Q.OUT}/sled_{nm}_mu{mu:g}_theld.json", "w"), indent=1, default=float)
