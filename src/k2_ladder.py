"""K2 head-carry ladders with the LAG-AWARE stand (k2_lag.COMP = "pred"), stock G1H, free load, arm-balance law, T1 support gate.
For each (object, lag): masses 8, 16, 24, ... kg UPWARD, warm-started (rung 0 from the Q1 10 ms lag maximum's z, then each rung from
the last valid z). Per rung: the warm-start z is first checked at both dt (FRESH plants); if not valid, CMA-ES (seed 1, pop 6,
bounds [-1, 1], sigma 0.2) on the 14-D track-A vector with lag + support gate in the score (engine.headcarry.evaluate at 1 ms);
each generation's best two 1 ms gate passers are checked at both dt. The ladder stops at the FIRST rung with no valid z within
the per-rung budget (budget-limited: a failure is 'not found within B evals', not an impossibility).
At each ladder maximum: 3 perturbations of z (sigma 0.03, rng 20261018) at both dt, and the law-off ablation (same z) at both dt.
Output results/overnight/K2/ladder_<obj>_lag<ms>ms.json. usage: python k2_ladder.py --lag 0.02 --objs box,pot [--workers 3] [--budget 36] [--wall_h 1.4]"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, multiprocessing as mp
import k2_lag as K

STEP = 8.0
WARM = {"box": "head_box_lag10ms", "pot": "head_pot_lag10ms"}


def gated(a):
    obj, mass, z, dt, lag, law = a
    return K.job(("g", "head", obj, mass, z, dt, lag, K.COMP, True, law, 0.0))[1]


def both(pool, obj, mass, z, lag, law=True):
    r1, r05 = pool.map(gated, [(obj, mass, list(z), dt, lag, law) for dt in (0.001, 0.0005)], chunksize=1)
    return bool(r1.get("valid_slip") and r05.get("valid_slip")), r1, r05


def brief(r): return [r.get("t_held"), r.get("reason"), r.get("foot_travel_max"), r.get("pelvis_drift"), r.get("valid_slip")]


def rung(pool, obj, mass, lag, x0, budget, t_end, pop=6):
    import cma
    ok, r1, r05 = both(pool, obj, mass, x0, lag); rec = dict(mass=mass, warm_ok=ok, warm=[brief(r1), brief(r05)], evals=0, hist=[])
    if ok: rec.update(valid=True, z=list(x0), r1=r1, r05=r05); return rec
    es = cma.CMAEvolutionStrategy(np.asarray(x0), 0.2, dict(popsize=pop, seed=1, bounds=[-1, 1], verbose=-9)); found = None
    while rec["evals"] + pop <= budget and time.time() < t_end and found is None:
        X = [np.asarray(x) for x in es.ask()]
        R = pool.map(K.score, [("head", obj, mass, list(x), lag, K.COMP) for x in X], chunksize=1); rec["evals"] += len(X)
        f = [r["score"] for r in R]; es.tell(X, f)
        pas = sorted([(f[k], k) for k in range(len(R)) if R[k].get("gate1")])[:2]; h = dict(evals=rec["evals"], fbest=float(min(f)), n_pass_1ms=len([r for r in R if r.get("gate1")]), checked=[])
        for _, k in pas:
            ok, c1, c2 = both(pool, obj, mass, X[k], lag); h["checked"].append([ok, brief(c1), brief(c2)])
            if ok: found = (X[k].tolist(), c1, c2); break
        rec["hist"].append(h); K.log(f"L {obj} {mass:g}kg lag{lag*1000:g} gen {len(rec['hist'])}: {json.dumps(K.J(h))[:300]}", f"ladder_lag{int(lag*1000)}.status")
    rec["valid"] = found is not None
    if found: rec.update(z=found[0], r1=found[1], r05=found[2])
    rec["stopped"] = None if found else ("wall" if time.time() >= t_end else f"budget {budget} evals")
    return rec


def ladder(pool, obj, lag, budget, t_end, n_pert=3):
    P = f"{K.OUT}/ladder_{obj}_lag{int(round(lag*1000))}ms.json"
    L = json.load(open(P)) if os.path.exists(P) else dict(obj=obj, lag=lag, comp=K.COMP, step=STEP, rungs=[])
    Q1 = json.load(open(f"{K.ROOT}/results/overnight/Q1/lag_maxima.json"))[WARM[obj]]
    x = list(np.asarray(Q1["lag_max_z"], float)[:14]); L["warm_src"] = f"Q1 {WARM[obj]} lag_max {Q1['lag_max']} kg z"
    done = [r for r in L["rungs"] if r["valid"]]
    if done: x = done[-1]["z"]
    mass = (L["rungs"][-1]["mass"] + STEP) if L["rungs"] and L["rungs"][-1]["valid"] else (L["rungs"][-1]["mass"] if L["rungs"] else 8.0)
    while not (L["rungs"] and not L["rungs"][-1]["valid"]) and time.time() < t_end:
        r = rung(pool, obj, mass, lag, x, budget, t_end); L["rungs"].append(r); json.dump(K.J(L), open(P, "w"), indent=1)
        K.log(f"L {obj} lag {lag*1000:g} ms {mass:g} kg: valid {r['valid']} warm_ok {r['warm_ok']} evals {r['evals']}", f"ladder_lag{int(lag*1000)}.status")
        if not r["valid"]: break
        x = r["z"]; mass += STEP
    val = [r for r in L["rungs"] if r["valid"]]
    if not val: L["max"] = None; json.dump(K.J(L), open(P, "w"), indent=1); return L
    best = val[-1]; L["max"] = best["mass"]; L["max_z"] = best["z"]; L["first_fail"] = (L["rungs"][-1]["mass"] if not L["rungs"][-1]["valid"] else None)
    if "robust" not in L:
        rng = np.random.default_rng(20261018); z0 = np.asarray(best["z"])
        zs = [np.clip(z0 + 0.03*rng.standard_normal(z0.size), -1, 1).tolist() for _ in range(n_pert)]
        jobs = [(obj, best["mass"], zp, dt, lag, True) for zp in zs for dt in (0.001, 0.0005)] + [(obj, best["mass"], best["z"], dt, lag, False) for dt in (0.001, 0.0005)]
        R = pool.map(gated, jobs, chunksize=1)
        pert = [dict(z=zs[i], r1=brief(R[2*i]), r05=brief(R[2*i + 1]), both=bool(R[2*i].get("valid_slip") and R[2*i + 1].get("valid_slip"))) for i in range(n_pert)]
        L["robust"] = dict(sigma=0.03, rng=20261018, draws=pert, n_both=sum(p["both"] for p in pert), n_rollouts_valid=sum(bool(r.get("valid_slip")) for r in R[:2*n_pert]))
        L["lawoff"] = dict(r1=R[-2], r05=R[-1])
        json.dump(K.J(L), open(P, "w"), indent=1)
    K.log(f"L {obj} lag {lag*1000:g} ms MAX {L['max']} kg (first fail {L['first_fail']}); perturb {L['robust']['n_both']}/{n_pert} both-dt; law-off held {L['lawoff']['r1'].get('t_held')}/{L['lawoff']['r05'].get('t_held')} s", f"ladder_lag{int(lag*1000)}.status")
    return L


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--lag", type=float, required=True); ap.add_argument("--objs", default="box,pot")
    ap.add_argument("--workers", type=int, default=3); ap.add_argument("--budget", type=int, default=36); ap.add_argument("--wall_h", type=float, default=1.4); a = ap.parse_args()
    os.makedirs(K.OUT, exist_ok=True); t_end = time.time() + 3600*a.wall_h
    with mp.get_context("fork").Pool(a.workers, maxtasksperchild=6) as pool:
        for obj in a.objs.split(","):
            L = ladder(pool, obj, a.lag, a.budget, t_end)
            os.system(f"~/bin/notify 'K2 {obj} lag {a.lag*1000:g} ms: max {L.get('max')} kg' 2>/dev/null")
    K.log(f"ladder lag {a.lag} done", f"ladder_lag{int(a.lag*1000)}.status")
