"""Q1: Gate A budget ladder. Squat jump (dip <= 1 cm, now actually enforced, P13), planar human,
held certified crouch, 1 ms. 16 seeds per level; levels span 1.2k -> 192k rollouts (160x)."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cma, gate_a2 as GA
from multiprocessing import Pool
OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase9")
CROUCH0 = list(GA.CROUCH)
LEVELS = [(75, 16), (300, 16), (1200, 16), (1500, 32), (1500, 64), (3000, 64)]

def solve(args):
    seed, iters, pop, squat, extra = args
    GA.CROUCH[:] = extra.get("crouch", CROUCH0)      # reset every job (workers are reused)
    h = GA.make_h(extra.get("scale"))
    if "act_tau" in extra: h.act_tau = extra["act_tau"]
    if "anderson_scale" in extra:           # Phase 10 labelled attribution sweep
        h.rescale_anderson(**extra["anderson_scale"]); h._hold, h._cert = GA.hold_torque(h, h.place(GA.CROUCH))
    if "ta_width" in extra: h.ta = [(p, w*extra["ta_width"], fl) for p, w, fl in h.ta]
    nk = extra.get("nk", GA.NK); hor = extra.get("horizon", GA.HOR)
    def obj(x):
        r = GA.rollout(h, x, horizon=hor, nk=nk); pen = 0.0
        if squat and r["dip"] > 0.01: pen += (r["dip"]-0.01)**2*200.0
        if not r["valid"]: pen += 0.1
        return -r["height"] + pen
    x0 = np.repeat(np.clip(h._hold/h.tau, -1, 1)[:, None], nk, 1).ravel() if h._hold is not None else np.zeros(5*nk)
    es = cma.CMAEvolutionStrategy(x0, 0.35, {"bounds": [-1, 1], "seed": seed, "verbose": -9, "maxiter": iters, "popsize": pop})
    t0 = time.time(); trace = []
    while not es.stop():
        X = es.ask(); f = [obj(x) for x in X]; es.tell(X, f); trace.append(float(-min(f)))
    r = GA.rollout(h, es.result.xbest, horizon=hor, nk=nk)
    ok = r["valid"] and (not squat or r["dip"] <= 0.0105)
    return dict(seed=seed, iters=iters, pop=pop, squat=squat, extra={k: v for k, v in extra.items()}, budget=iters*pop,
                height=r["height"], v=r["v"], dip=r["dip"], valid=bool(ok), T=r["T"], grf=r["peak_grf_bw"],
                mom=r["momentum_error"], secs=time.time()-t0, stop=str(es.stop()), best_trace=trace[::max(1, len(trace)//50)],
                x=np.asarray(es.result.xbest).tolist())

def run(name, jobs, workers=14):
    jobs = sorted(jobs, key=lambda j: -j[1]*j[2])
    with Pool(workers) as p: res = p.map(solve, jobs, chunksize=1)
    json.dump(res, open(f"{OUT}/{name}.json", "w"), default=float)
    return res

if __name__ == "__main__":
    what = sys.argv[1]
    if what == "decomp":
        it, pop = int(sys.argv[2]), int(sys.argv[3]); seeds = range(1, 9)
        C = {"base": {}, "crouch_mid": {"crouch": [-1.04, 1.69, -0.66, -0.64, 0.45]},
             "crouch_shallow": {"crouch": [-0.9, 1.51, -0.62, -0.75, 0.58]},
             "lag30": {"act_tau": 0.030}, "lag50": {"act_tau": 0.050},
             "ta_narrow": {"ta_width": 0.7}, "ta_wide": {"ta_width": 1.4},
             "nk7": {"nk": 7}, "nk28": {"nk": 28}, "hor035": {"horizon": 0.35}, "hor060": {"horizon": 0.60}}
        jobs = [(s, it, pop, True, dict(v, tag=k)) for k, v in C.items() for s in seeds]
        jobs += [(s, it, pop, False, dict(tag="cmj")) for s in seeds]       # countermovement allowed (P-001 evidence)
        run("gate_a_decomp", jobs)
    if what == "ladder":
        run("gate_a_ladder", [(s, it, pop, True, {}) for it, pop in LEVELS for s in range(1, 17)])


def bobbert(it=1200, pop=16, seeds=range(1, 9), scale=1.5):
    """Direction-only control at a converged budget, under the P13-corrected squat constraint:
    strengthen every actuator without re-optimising and check whether height falls."""
    import gate_a2 as GA
    base = run("gate_a_bobbert_base", [(s, it, pop, True, {}) for s in seeds])
    out = []
    for r in base:
        hs = GA.make_h([scale]*5); rr = GA.rollout(hs, np.array(r["x"]))
        out.append(dict(seed=r["seed"], base=r["height"], strong_same_control=rr["height"],
                        base_valid=r["valid"], strong_valid=bool(rr["valid"] and rr["dip"] <= 0.0105), dip=rr["dip"]))
        print(f"  seed {r['seed']}: {r['height']*100:.1f} -> {rr['height']*100:.1f} cm (strong, same control)", flush=True)
    json.dump(out, open(f"{OUT}/gate_a_bobbert.json", "w"), default=float)
    n = sum(o["strong_same_control"] < o["base"] for o in out)
    print(f"Bobbert direction: height DECREASED in {n}/{len(out)} seeds", flush=True)


if __name__ == "__main__" and sys.argv[1:2] == ["bobbert"]:
    bobbert()
