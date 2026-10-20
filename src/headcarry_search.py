"""Track A driver (2026-10-17): head-carry with arm balance vs hands-only, G1H, free objects (no weld). Simulator only.
Phases: P predictor (spec only, before search) -> S standing 10 s ladder per (strategy, object, mass) -> X extension of the head
ladder past 16 kg while it holds -> W sway variant (lateral weight shift 0.04 m at 0.5 Hz, 10 s) for the cases that held ->
R render. Every case: prior (z=0: the derived law with rule gains), CMA search on 14 refinement parameters, the best re-run at
0.5 ms, and the LAW-OFF ablation (same posture and arm impedance, no feedback). Gates: holds T at 1 ms AND 0.5 ms; head strategy:
the head carries >= 50% of the weight, the COM never leaves the 7 cm head patch, the object never touches the floor.
Resumable: a case whose JSON exists is skipped. usage: python headcarry_search.py [--workers 6] [--budget 240] [--phase PSXWR]"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from engine import headcarry as HC

OUT = HC.RES
MASSES = [2.0, 5.0, 8.0, 12.0, 16.0]
EXT = [20.0, 24.0, 32.0, 48.0, 64.0]
HANDS = [2.0, 3.0, 4.0, 5.0, 8.0]
DISTURB = [("sway", 0.2, 0.5), ("push", 0.2)]   # lateral capture-point sway 0.2 m at 0.5 Hz (~2 cm COM sway); torso pushes 0.2 x robot weight, 0.2 s


def J(o):
    return json.loads(json.dumps(o, default=lambda x: x.tolist() if hasattr(x, "tolist") else (float(x) if isinstance(x, (np.floating, np.integer)) else str(x))))


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)
    with open(f"{OUT}/status.txt", "a") as f: f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def run_case(pool, strategy, obj, mass, budget, pop=12, T=10.0, sway=None, tag=None, x0=None, min_gens=4):
    import cma
    tag = tag or f"{strategy}_{obj}_{mass:g}kg" + (f"_{sway[0]}" if sway else "")
    path = f"{OUT}/{tag}.json"
    if os.path.exists(path): return json.load(open(path))
    t0 = time.time(); dim = HC.DIM[strategy]
    if sway and x0 is None and os.path.exists(f"{OUT}/{strategy}_{obj}_{mass:g}kg.json"): x0 = json.load(open(f"{OUT}/{strategy}_{obj}_{mass:g}kg.json"))["best_z"]   # warm start: the standing best
    z0 = np.zeros(dim) if x0 is None else np.asarray(x0); prior = HC.evaluate((obj, mass, strategy, z0, T, 0.001, sway))
    es = cma.CMAEvolutionStrategy(np.zeros(dim) if x0 is None else np.asarray(x0), 0.3, dict(popsize=pop, seed=7, bounds=[-1, 1], verbose=-9))
    best = (prior["score"], z0.tolist(), prior); hist = []; nev = 1; solved = None; validated = None
    if HC.gate(prior, T, strategy, mass):
        c0 = HC.evaluate((obj, mass, strategy, z0, T, 0.0005, sway)); nev += 1
        if HC.gate(c0, T, strategy, mass): solved = "prior"; validated = (prior["score"], z0.tolist(), prior, c0)
    gens = 0
    while nev + pop <= budget:
        X = es.ask(); R = pool.map(HC.evaluate, [(obj, mass, strategy, np.asarray(x), T, 0.001, sway) for x in X], chunksize=1); nev += len(X); gens += 1
        f = [r["score"] for r in R]; es.tell(X, f); i = int(np.argmin(f))
        if f[i] < best[0]: best = (f[i], np.asarray(X[i]).tolist(), R[i])
        passers = sorted([(f[k], k) for k in range(len(R)) if HC.gate(R[k], T, strategy, mass)])[:3]
        hist.append(dict(gen=gens, evals=nev, best=float(best[0]), gen_best=float(f[i]), gen_med=float(np.median(f)), n_pass=len(passers), best_held=best[2]["t_held"], wall=time.time() - t0))
        if passers and validated is None:
            # a 1 ms pass counts only once it also passes at 0.5 ms (checked here for the 3 best passers of the generation)
            C = pool.map(HC.evaluate, [(obj, mass, strategy, np.asarray(X[k]), T, 0.0005, sway) for _, k in passers], chunksize=1); nev += len(C)
            for (fk, k), c in zip(passers, C):
                if HC.gate(c, T, strategy, mass): validated = (fk, np.asarray(X[k]).tolist(), R[k], c); solved = f"gen {gens}"; break
            hist[-1]["dt05_checked"] = [bool(HC.gate(c, T, strategy, mass)) for c in C]
        if solved and gens >= min_gens: break
    if validated is not None:   # the reported best is the first candidate that passed BOTH dt (not the lowest 1 ms score)
        bz = np.asarray(validated[1]); best = validated[:3]; chk = validated[3]
    else:
        bz = np.asarray(best[1]); chk = HC.evaluate((obj, mass, strategy, bz, T, 0.0005, sway))
    sw_, pu_ = HC.disturbance(sway); abl = HC.rollout(obj, mass, strategy, bz, T, 0.001, False, sw_, law=False, push=pu_); abl.pop("trace", None)
    ok1 = HC.gate(best[2], T, strategy, mass); ok05 = HC.gate(chk, T, strategy, mass)
    res = dict(track="A", strategy=strategy, obj=obj, mass=mass, T=T, sway=sway, budget=budget, evals=nev, gens=gens, wall_s=time.time() - t0,
               s_per_eval_wall=(time.time() - t0)/max(1, nev + 2), label=("free object resting on the head; palms on its sides, no weld" if strategy == "head" else "free object squeezed between the palms, no weld"),
               prior=prior, prior_gate=HC.gate(prior, T, strategy, mass), best_z=bz.tolist(), best_params=HC.decode(bz, strategy), best=best[2], check_dt0005=chk,
               law_off_ablation=abl, gate_1ms=ok1, gate_05ms=ok05, valid=bool(ok1 and ok05), solved=solved, hist=hist)
    res = J(res); json.dump(res, open(path, "w"), indent=1)
    log(f"{tag}: valid={res['valid']} prior_gate={res['prior_gate']} best held {best[2]['t_held']:.2f} ({best[2]['reason']}) 0.5ms {chk['t_held']:.2f} ({chk['reason']}) "
        f"law-off {abl['t_held']:.2f} ({abl['reason']}) head_frac {best[2].get('head_frac', 0):.2f} evals {nev} {res['wall_s']/60:.1f} min")
    return res


def phase_P():
    path = f"{OUT}/predictor.json"
    if os.path.exists(path): return json.load(open(path))
    import throw as TH
    S = TH.setup("G1H", ball=False); out = {}
    for st in ("head", "hands"):
        for ob in ("box", "pot"):
            t0 = time.time(); out[f"{st}_{ob}"] = HC.predict_max_load(st, ob, S=S); out[f"{st}_{ob}"]["wall_s"] = time.time() - t0
            log(f"predictor {st} {ob}: m_max {out[f'{st}_{ob}']['m_max']} kg (binding {out[f'{st}_{ob}']['binding']})")
    out = J(dict(note="computed from the robot spec and the IK grip posture only, BEFORE any search; theta = 10 deg object tilt margin, sway requirement 0.5 x the unloaded stance's capacity", **out))
    json.dump(out, open(path, "w"), indent=1); return out


def ladder(pool, strategy, obj, masses, budget, sway=None, stop_after=2, small_budget=120, min_gens=4):
    res = []; fails = 0
    for i, mass in enumerate(masses):
        r = run_case(pool, strategy, obj, mass, budget if fails == 0 else small_budget, sway=sway, min_gens=min_gens); res.append(r)
        fails = 0 if r["valid"] else fails + 1
        if fails >= stop_after:
            log(f"{strategy} {obj}{' sway' if sway else ''}: {fails} consecutive failure(s) ending at {mass:g} kg; heavier masses not run: {masses[i+1:]}"); break
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--budget", type=int, default=240)
    ap.add_argument("--phase", default="PSXWR"); ap.add_argument("--hands_budget", type=int, default=720); ap.add_argument("--ext_budget", type=int, default=480); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    import multiprocessing as mp
    if a.smoke:
        OUT = HC.RES = f"{HC.RES}/smoke"; os.makedirs(OUT, exist_ok=True)
        with mp.get_context("fork").Pool(a.workers) as pool:
            r = run_case(pool, "head", "box", 5.0, 2*12 + 1, T=3.0, min_gens=2); print(json.dumps({k: r[k] for k in ("valid", "evals", "wall_s", "s_per_eval_wall")}))
        sys.exit()
    if "P" in a.phase: phase_P()
    with mp.get_context("fork").Pool(a.workers) as pool:
        if "S" in a.phase:
            for st, ob in (("head", "box"), ("head", "pot"), ("hands", "box"), ("hands", "pot")): ladder(pool, st, ob, MASSES, a.budget)
        if "H" in a.phase:   # hands-only, finer ladder and deeper budget (pass 2)
            for ob in ("box", "pot"): ladder(pool, "hands", ob, HANDS, a.hands_budget, small_budget=a.hands_budget)
            os.system("~/bin/notify 'Track A: hands-only ladder done' 2>/dev/null")
        if "X" in a.phase:
            for ob in ("box", "pot"):
                top = f"{OUT}/head_{ob}_16kg.json"
                if os.path.exists(top) and json.load(open(top))["valid"]: ladder(pool, "head", ob, EXT, a.ext_budget, stop_after=1)
            os.system("~/bin/notify 'Track A: head-carry extension past 16 kg done' 2>/dev/null")
        if "W" in a.phase:
            for dist in DISTURB:
                for st, ob in (("head", "box"), ("head", "pot"), ("hands", "box"), ("hands", "pot")):
                    ms = [m for m in sorted(set(MASSES + EXT + HANDS)) if os.path.exists(f"{OUT}/{st}_{ob}_{m:g}kg.json") and json.load(open(f"{OUT}/{st}_{ob}_{m:g}kg.json"))["valid"]]
                    if ms: ladder(pool, st, ob, ms, 120, sway=dist, stop_after=1, min_gens=1)
        if "B" in a.phase:   # refinements: hands box 3 kg warm-started from the 4 kg solution (the ladder was non-monotone);
            # head max bracketed by one bisection step between the last valid and the first failed mass
            hb4 = f"{OUT}/hands_box_4kg.json"
            if os.path.exists(hb4): run_case(pool, "hands", "box", 3.0, a.hands_budget, x0=json.load(open(hb4))["best_z"], tag="hands_box_3kg_warm4")
            for ob, lo, hi in (("box", 24.0, 32.0), ("pot", 20.0, 24.0)):
                mid = 0.5*(lo + hi); f_lo = f"{OUT}/head_{ob}_{lo:g}kg.json"
                run_case(pool, "head", ob, mid, a.ext_budget, x0=(json.load(open(f_lo))["best_z"] if os.path.exists(f_lo) else None))
        if "C" in a.phase:   # CONTINUATION: warm-started head ladder upward (each mass starts from the last valid mass's best z)
            for ob, start in (("box", 28.0), ("pot", 22.0)):
                prev = f"{OUT}/head_{ob}_{start:g}kg.json"; mass = start
                while True:
                    mass = mass + 4.0
                    tag = f"head_{ob}_{mass:g}kg" + ("_warm" if os.path.exists(f"{OUT}/head_{ob}_{mass:g}kg.json") else "")
                    r = run_case(pool, "head", ob, mass, a.ext_budget, x0=json.load(open(prev))["best_z"], tag=tag)
                    if not r["valid"] or mass >= 64: break
                    prev = f"{OUT}/{tag}.json"
        if "M" in a.phase:   # disturbances at the continuation maxima (warm-started from the standing best)
            for st, ob, mass, src in (("head", "box", 32.0, "head_box_32kg_warm"), ("head", "pot", 34.0, "head_pot_34kg")):
                x0 = json.load(open(f"{OUT}/{src}.json"))["best_z"]
                for dist in DISTURB: run_case(pool, st, ob, mass, 120, sway=dist, x0=x0, min_gens=1)
    log("driver finished phases " + a.phase)
