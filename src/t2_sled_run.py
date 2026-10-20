"""T2 held-out sled runs (stock G1, right hand WELDED to the sled post: the track B simulator abstraction; sled floor friction mu set
per run: an ENVIRONMENT parameter, the robot model is unchanged). Pre-registered in results/overnight/T2/PREDICTIONS.md before running.
Search = sled_dyn (track B) unchanged: 19-D representation, score = worse of 1 ms and 0.5 ms, FEET_SLID > 3 cm and falls fail.
Validity (stricter than track B): the best candidate re-run in FRESH processes (a plant built directly at each dt, no re-timed plant
history), start seeds 0..3 (seed > 0 perturbs the start by 2e-5 m): valid = seed 0 moves >= 0.2 m at both dt with no fall and feet slid
<= 3 cm; robust = count of seeds 0..3 that do.
usage: python t2_sled_run.py push|pull MU payload1,payload2,... [iters] [pop] [workers]"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, multiprocessing as mp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "overnight", "T2")
B_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "overnight", "B")
MU = None


def _init(mu):
    import leverage_probes as LP
    LP.SLED_MU = mu; LP._L3.clear()


def fresh_eval(args):
    """(mu, mass, sgn, z, T, dt, seed) in a fresh process (maxtasksperchild=1): plant built directly at dt."""
    mu, mass, sgn, z, T, dt, seed = args
    _init(mu); import sled_dyn as SD
    r = SD.evaluate((mass, sgn, np.asarray(z), T, dt, False, seed))
    return {k: r.get(k) for k in ("disp", "reason", "t_held", "com_lean_mean", "util_rms", "weld_err_max", "weld_ang_max")}


if __name__ == "__main__":
    nm, mu = sys.argv[1], float(sys.argv[2]); payloads = [float(x) for x in sys.argv[3].split(",")]
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 40; pop = int(sys.argv[5]) if len(sys.argv) > 5 else 12; W = int(sys.argv[6]) if len(sys.argv) > 6 else 6
    sgn = 1 if nm == "push" else -1; _init(mu)
    import leverage_probes as LP, sled_dyn as SD
    x0 = None
    bj = f"{B_DIR}/{nm}_20kg.json"
    if os.path.exists(bj): x0 = np.asarray(json.load(open(bj))["search"]["best_z"])       # warm start: track B's 20 kg strategy (mu 0.4)
    for mass in payloads:
        tag = f"sled_{nm}_mu{mu:g}_{mass:g}kg"; jf = f"{OUT}/{tag}.json"
        if os.path.exists(jf): x0 = np.asarray(json.load(open(jf))["best_z"]); continue
        t0 = time.time(); log = open(f"{OUT}/{tag}.log", "w")
        LP.l3_plant(mass, 0.001)      # build the scene file ONCE in the parent: forked/spawned workers otherwise race on writing it (ParseXML: empty file)
        res = LP.cma_search(SD.evaluate, lambda x: (mass, sgn, x, 4.0, "both", False, 0), SD.DIM, iters, pop, W, sigma=0.4, x0=x0, log=log)
        bz = np.asarray(res["best_z"])
        jobs = [(mu, mass, sgn, bz.tolist(), 4.0, dt, s) for s in (0, 1, 2, 3) for dt in (0.001, 0.0005)]
        with mp.get_context("spawn").Pool(W, maxtasksperchild=1) as p: C = p.map(fresh_eval, jobs, chunksize=1)
        per = []
        for i, s in enumerate((0, 1, 2, 3)):
            a, b = C[2*i], C[2*i + 1]; ok = all(c["reason"] is None and c["disp"] >= 0.2 for c in (a, b))
            per.append(dict(seed=s, ok=ok, disp=[a["disp"], b["disp"]], reason=[a["reason"], b["reason"]]))
        out = dict(track="T2", robot="G1 (stock menagerie model, vendor ratings)", direction=nm, sled_mu=mu, payload=mass, sled_total=mass + 1.0, T=4.0,
                   label="right hand WELDED to the sled post (simulator abstraction, weld time constant 0.005 s as track B); sled floor mu is the environment; simulator only",
                   search=dict(best_z=bz.tolist(), best=res.get("best") and {k: res["best"].get(k) for k in ("disp", "disp_min", "reason", "score", "ok_both")}, evals=res.get("evals")),
                   fresh=per, valid=per[0]["ok"], robust=int(sum(p_["ok"] for p_ in per)), disp_min_seed0=min(per[0]["disp"]), best_z=bz.tolist(), wall_s=time.time() - t0, warm_start=x0 is not None)
        json.dump(out, open(jf, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
        print(f"== {tag}: valid={out['valid']} robust={out['robust']}/4 disp seed0 {per[0]['disp']} {per[0]['reason']} wall {out['wall_s']/60:.1f} min", flush=True)
        x0 = bz
