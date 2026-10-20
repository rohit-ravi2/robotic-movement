"""T2 held-out head-carry runs (stock G1H, free object on the head, palms on its sides, no weld; simulator only).
Pre-registered in results/overnight/T2/PREDICTIONS.md BEFORE this runs. Per mass: CMA-ES on the track A 14-D refinement vector
(engine.headcarry.decode; score = engine.headcarry.evaluate at 1 ms), and every generation the 3 best 1 ms passers are re-run with the
SLIP-GATED validity (t2_gate.rollout_gated) at 1 ms AND 0.5 ms; the first candidate valid at both is the result. Then the law-off
ablation of that candidate (1 ms). Ladder continuation: each mass starts from the previous mass's best z.
Valid = held 10 s, no OFF_PATCH / DROPPED / FLOOR / FELL, tilt <= 45 deg, head carries >= 50% of the weight, feet slid <= 3 cm and
system COM drift <= 10 cm, at both dt.
usage: python t2_headcarry_run.py OBJ m1,m2,... [budget] [workers] [x0 spec: 'tallcyl' -> z6 = -1 (dz = 0, the reachable grip)]"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, multiprocessing as mp
from engine import headcarry as HC, dynpred_headcarry as D
import t2_gate as G

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "overnight", "T2")


def J(o):
    return json.loads(json.dumps(o, default=lambda x: x.tolist() if hasattr(x, "tolist") else (float(x) if isinstance(x, (np.floating, np.integer)) else str(x))))


def ev(args):
    D.register_objects(); return HC.evaluate(args)


def gated(args):
    D.register_objects(); return G.eval_job(args)


def run_mass(pool, obj, mass, budget, x0, pop=12, T=10.0, min_gens=3):
    import cma
    tag = f"head_{obj}_{mass:g}kg"; path = f"{OUT}/{tag}.json"
    if os.path.exists(path): return json.load(open(path))
    t0 = time.time(); dim = HC.DIM["head"]
    es = cma.CMAEvolutionStrategy(np.asarray(x0), 0.3, dict(popsize=pop, seed=11, bounds=[-1, 1], verbose=-9))
    nev = 0; best = (np.inf, list(x0), None); validated = None; hist = []; gens = 0
    while nev + pop <= budget:
        X = es.ask(); R = pool.map(ev, [(obj, mass, "head", np.asarray(x), T, 0.001, None) for x in X], chunksize=1); nev += len(X); gens += 1
        f = [r["score"] for r in R]; es.tell(X, f); i = int(np.argmin(f))
        if f[i] < best[0]: best = (f[i], np.asarray(X[i]).tolist(), R[i])
        passers = sorted([(f[k], k) for k in range(len(R)) if HC.gate(R[k], T, "head", mass)])[:3]
        h = dict(gen=gens, evals=nev, best=float(best[0]), n_pass_1ms=len(passers), best_held=best[2]["t_held"], best_reason=best[2]["reason"], wall=time.time() - t0)
        if passers and validated is None:
            C = pool.map(gated, [(obj, mass, "head", np.asarray(X[k]), T, dt, None, True) for _, k in passers for dt in (0.001, 0.0005)], chunksize=1); nev += len(C)
            h["checked"] = []
            for j, (fk, k) in enumerate(passers):
                c1, c2 = C[2*j], C[2*j + 1]; ok = bool(c1.get("valid_slip") and c2.get("valid_slip"))
                h["checked"].append(dict(ok=ok, slide=[c1.get("foot_slide_max"), c2.get("foot_slide_max")], reason=[c1.get("reason"), c2.get("reason")]))
                if ok and validated is None: validated = dict(z=np.asarray(X[k]).tolist(), r1=c1, r05=c2, gen=gens)
        hist.append(h); print(tag, h, flush=True)
        if validated is not None and gens >= min_gens: break
    if validated is not None:
        bz = validated["z"]; abl = G.rollout_gated(obj, mass, "head", np.asarray(bz), T, 0.001, law=False)
    else:
        bz = best[1]; abl = None
        C = pool.map(gated, [(obj, mass, "head", np.asarray(bz), T, dt, None, True) for dt in (0.001, 0.0005)])
        validated_fail = dict(r1=C[0], r05=C[1])
    res = dict(track="T2", robot="G1H (stock menagerie model, vendor ratings)", obj=obj, obj_spec=HC.OBJ[obj], mass=mass, T=T, budget=budget, evals=nev, gens=gens,
               wall_s=time.time() - t0, valid=validated is not None, best_z=bz, params=HC.decode(np.asarray(bz), "head"),
               check=(dict(dt1ms=validated["r1"], dt05ms=validated["r05"]) if validated else dict(dt1ms=validated_fail["r1"], dt05ms=validated_fail["r05"])),
               law_off_1ms=abl, best_1ms_score=best[2], hist=hist,
               label="free object resting on the head; palms on its sides; no weld; object placed at rest at t=0 by a 'helper'; simulator only",
               choices="arm servo compliance and law gains as track A (construction); stand controller; 2 ms control; MuJoCo 3.2.7 default contact; no model parameter changed")
    for k in ("trace",):
        for v in (res["check"]["dt1ms"], res["check"]["dt05ms"]): v.pop(k, None) if isinstance(v, dict) else None
    res = J(res); json.dump(res, open(path, "w"), indent=1)
    c = res["check"]
    print(f"== {tag}: valid={res['valid']} 1ms {c['dt1ms'].get('t_held')} {c['dt1ms'].get('reason')} slide {c['dt1ms'].get('foot_slide_max')} | 0.5ms {c['dt05ms'].get('t_held')} "
          f"{c['dt05ms'].get('reason')} slide {c['dt05ms'].get('foot_slide_max')} | law-off {abl and abl['t_held']} {abl and abl['reason']} | evals {nev} {res['wall_s']/60:.1f} min", flush=True)
    return res


if __name__ == "__main__":
    obj = sys.argv[1]; masses = [float(x) for x in sys.argv[2].split(",")]; budget = int(sys.argv[3]) if len(sys.argv) > 3 else 180
    W = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    D.register_objects(); x0 = np.zeros(HC.DIM["head"])
    if obj == "tallcyl": x0[6] = -1.0          # dz = 0: palms at the COM height (dz = 0.5 is out of the arms' reach, IK residual 0.27 m)
    with mp.get_context("fork").Pool(W) as pool:
        for mass in masses:
            r = run_mass(pool, obj, mass, budget, x0)
            if r["valid"]: x0 = np.asarray(r["best_z"])
