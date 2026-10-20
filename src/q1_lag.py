"""TRACK Q1 (2026-10-17): HEAD-CARRY UNDER ACTUATOR LAG. Stock G1H (menagerie model, vendor ratings), simulator only.
Opt-in lag mode of engine.headcarry (HC_LAG / rollout(lag=, wcap=)): every body actuator's servo output passes a first-order lag
(branch_i exact discretisation) and the torque-speed envelope with droop; the Dex3 finger servos stay ideal. The as-built servo
gains (stand 368 rad/s, 'held' roll joints kp = I/dt^2 = 1000 rad/s at 1 ms) cannot be evaluated explicitly behind a lag (they
fall before the load is placed), so lagged servos are capped at natural frequency c/tau (critically damped) - a controller-gain
choice, recorded; c = 2 fixed from the 40 kg pot scan at 10 ms before any other object was run (the only c in {1.5,2,3,4} that held).
Phases:
  A  replay every stored zero-lag maximum (pot 40, box 32, crate 63, tall cylinder 24 head; hands box 6, pot 8) under 10 ms lag,
     c = 2, FRESH plants at 1 ms and 0.5 ms, gate = t2_gate.valid_slip (engine.headcarry.gate incl. the T1 support gate: loaded
     foot travel <= 0.02 m, pelvis drift <= 0.05 m; plus slide <= 3 cm, COM drift <= 10 cm). Raw-gain (no cap) replays at 1 ms too.
  B  where a maximum fails: CMA-ES (seed 1, pop 8) on the 14-D track A vector + 1 dim for the cap c = 2 * 2^(0.5 z14), lag inside
     the score (engine.headcarry.evaluate, support gate in the score), warm-started from the zero-lag solution; each generation the
     best 1 ms gate passers are checked at both dt; no valid within the per-mass budget -> step the mass down.
  C  the same for the pot at 20 ms (bracket).
  Law-off ablation (same z, same lag) at each lag-robust maximum, both dt.
usage: python q1_lag.py [--phase ABC] [--workers 4] [--budget 48] [--wall_h 2.5]"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, multiprocessing as mp

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = f"{ROOT}/results/overnight/Q1"; T1 = f"{ROOT}/results/overnight/T1"; T2 = f"{ROOT}/results/overnight/T2"
C0 = 2.0
CASES = [  # tag, strategy, obj, stored max kg, source, mass ladder downwards
    ("head_pot", "head", "pot", 40.0, f"{T1}/head_pot_40kg_t1.json", [36.0, 32.0, 28.0, 24.0]),
    ("head_box", "head", "box", 32.0, f"{T1}/head_box_32kg_t1.json", [28.0, 24.0, 20.0, 16.0]),
    ("head_crate", "head", "crate", 63.0, f"{T2}/head_crate_63kg.json", [56.0, 49.0, 42.0, 35.0]),
    ("head_tallcyl", "head", "tallcyl", 24.0, f"{T2}/head_tallcyl_24kg.json", [21.0, 18.0, 15.0, 12.0]),
    ("hands_box", "hands", "box", 6.0, f"{T1}/hands_box_6kg_t1.json", [5.0, 4.0, 3.0]),
    ("hands_pot", "hands", "pot", 8.0, f"{T1}/hands_pot_8kg_t1.json", [7.0, 6.0, 5.0, 4.0]),
]
KEEP = ("t_held", "reason", "head_frac", "tilt_max_deg", "foot_travel_max", "pelvis_drift", "foot_slide_max", "com_drift_max", "gate_hc", "slip_ok", "valid_slip", "e_max", "util_p95", "act_lag", "joint_sat", "error")


def J(o): return json.loads(json.dumps(o, default=lambda x: x.tolist() if hasattr(x, "tolist") else (float(x) if isinstance(x, (np.floating, np.integer)) else str(x))))
def log(s):
    line = time.strftime("%H:%M:%S ") + s; print(line, flush=True); open(f"{OUT}/status.txt", "a").write(line + "\n")
def cap_of(z15): return C0*2**(0.5*float(np.clip(z15[14], -1, 1))) if len(z15) > 14 else C0


def _setlag(lag, c):
    from engine import headcarry as HC, dynpred_headcarry as D
    D.register_objects(); HC.ACT_LAG = float(lag); HC.LAG_WCAP = float(c); return HC


def gated(a):
    """(strategy, obj, mass, z15, dt, lag, law) -> slip+support gated result (fresh plant)."""
    st, obj, mass, z, dt, lag, law = a; HC = _setlag(lag, cap_of(z)); import t2_gate as G
    r = G.eval_job((obj, mass, st, np.asarray(z[:14]), 10.0, dt, None, law)); r = {k: r.get(k) for k in KEEP}; r["wcap"] = cap_of(z); return r


def score(a):
    st, obj, mass, z, lag = a; HC = _setlag(lag, cap_of(z))
    r = HC.evaluate((obj, mass, st, np.asarray(z[:14]), 10.0, 0.001, None)); r["gate1"] = HC.gate(r, 10.0, st, mass) if "foot_travel_max" in r else False
    return {k: r.get(k) for k in ("score", "t_held", "reason", "foot_travel_max", "pelvis_drift", "gate1", "head_frac")}


def z15(z, c=C0): return list(np.asarray(z, float)[:14]) + [float(np.clip(2*np.log2(c/C0), -1, 1))]


def phase_A(pool, lag=0.010):
    path = f"{OUT}/replay_lag{int(lag*1000)}ms.json"; res = json.load(open(path)) if os.path.exists(path) else {}
    jobs = []
    for tag, st, obj, mmax, src, _ in CASES:
        if tag in res: continue
        z = z15(json.load(open(src))["best_z"])
        jobs += [(tag, f"c2_{dt}", (st, obj, mmax, z, dt, lag, True)) for dt in (0.001, 0.0005)]
        jobs += [(tag, "raw_0.001", (st, obj, mmax, z[:14] + [0.0], 0.001, lag, True))]
    out = pool.map(_tagged, jobs, chunksize=1)
    for tag, k, r in out: res.setdefault(tag, {})[k] = r
    for tag, v in res.items():
        if "raw_0.001" in v and v["raw_0.001"].get("wcap") == C0: pass
        v["valid_c2"] = bool(v.get("c2_0.001", {}).get("valid_slip") and v.get("c2_0.0005", {}).get("valid_slip"))
    json.dump(J(res), open(path, "w"), indent=1)
    for tag, v in res.items():
        a, b = v.get("c2_0.001", {}), v.get("c2_0.0005", {})
        log(f"A lag {lag*1000:g} ms {tag}: valid {v['valid_c2']} | 1ms {a.get('t_held')} {a.get('reason')} ft {a.get('foot_travel_max')} pd {a.get('pelvis_drift')} | 0.5ms {b.get('t_held')} {b.get('reason')} ft {b.get('foot_travel_max')} pd {b.get('pelvis_drift')} | raw gains {v.get('raw_0.001', {}).get('t_held')} {v.get('raw_0.001', {}).get('reason')}")
    return res


def _tagged(a):
    tag, k, args = a
    if k.startswith("raw"):
        st, obj, mass, z, dt, lag, law = args; HC = _setlag(lag, 0.0); import t2_gate as G
        r = G.eval_job((obj, mass, st, np.asarray(z[:14]), 10.0, dt, None, law)); r = {x: r.get(x) for x in KEEP}; r["wcap"] = 0.0; return tag, k, r
    return tag, k, gated(args)


def search_mass(pool, tag, st, obj, mass, lag, x0, budget, t_end, pop=8):
    import cma
    name = f"{tag}_{mass:g}kg_lag{int(round(lag*1000))}ms"; path = f"{OUT}/{name}.json"
    if os.path.exists(path):
        old = json.load(open(path))
        if old["valid"] or old["evals"] + pop > budget: return old     # a short earlier search is re-run at the larger budget
    t0 = time.time(); es = cma.CMAEvolutionStrategy(np.asarray(x0), 0.2, dict(popsize=pop, seed=1, bounds=[-1, 1], verbose=-9))
    nev = 0; best = (np.inf, list(x0), None); found = None; hist = []
    while nev + pop <= budget and time.time() < t_end and found is None:
        X = [np.asarray(x) for x in es.ask()]
        if nev == 0: X[0] = np.asarray(x0)      # the warm start itself is always evaluated
        R = pool.map(score, [(st, obj, mass, list(x), lag) for x in X], chunksize=1); nev += len(X)
        f = [r["score"] for r in R]; es.tell(X, f); i = int(np.argmin(f))
        if f[i] < best[0]: best = (f[i], X[i].tolist(), R[i])
        pas = sorted([(f[k], k) for k in range(len(R)) if R[k].get("gate1")])[:2]
        h = dict(evals=nev, best=float(best[0]), n_pass_1ms=sum(bool(r.get("gate1")) for r in R), best_held=best[2]["t_held"], best_reason=best[2]["reason"], checked=[])
        if pas:
            C = pool.map(gated, [(st, obj, mass, X[k].tolist(), dt, lag, True) for _, k in pas for dt in (0.001, 0.0005)], chunksize=1)
            for j, (_, k) in enumerate(pas):
                c1, c2 = C[2*j], C[2*j + 1]; ok = bool(c1.get("valid_slip") and c2.get("valid_slip"))
                h["checked"].append(dict(ok=ok, r1=[c1.get("t_held"), c1.get("reason"), c1.get("foot_travel_max"), c1.get("pelvis_drift")], r05=[c2.get("t_held"), c2.get("reason"), c2.get("foot_travel_max"), c2.get("pelvis_drift")]))
                if ok and found is None: found = dict(z=X[k].tolist(), r1=c1, r05=c2)
        hist.append(h); log(f"B {name} gen {len(hist)}: {json.dumps(J(h))[:400]}")
    res = dict(track="Q1", tag=tag, strategy=st, obj=obj, mass=mass, lag=lag, valid=found is not None, evals=nev, wall_s=time.time() - t0,
               z=(found or {}).get("z", best[1]), wcap=cap_of((found or {}).get("z", best[1])), check=found, best_1ms=best[2], hist=hist, x0=list(x0))
    json.dump(J(res), open(path, "w"), indent=1); return res


def ladder(pool, tag, st, obj, mmax, src, masses, lag, budget, t_end, replay):
    z0 = z15(json.load(open(src))["best_z"]); out = dict(tag=tag, lag=lag, stored_max=mmax, replay_valid=replay, steps=[])
    if replay: out.update(lag_max=mmax, lag_max_z=z0, lag_max_src="stored zero-lag z, c = 2"); return out
    x = z0
    for mass in [mmax] + masses:
        if time.time() > t_end: out["stopped"] = "wall budget"; break
        r = search_mass(pool, tag, st, obj, mass, lag, x, budget, t_end); out["steps"].append(dict(mass=mass, valid=r["valid"], evals=r["evals"], wall_s=r["wall_s"]))
        log(f"B {tag} {mass:g} kg lag {lag*1000:g} ms: valid {r['valid']} evals {r['evals']} {r['wall_s']/60:.1f} min")
        if r["valid"]: out.update(lag_max=mass, lag_max_z=r["z"], lag_max_src=f"Q1 search {tag}_{mass:g}kg"); break
        x = r["z"]
    return out


def ablate(pool, L, lag):
    jobs = []
    for tag, v in L.items():
        if v.get("lag_max") is None: continue
        st, obj = [(c[1], c[2]) for c in CASES if c[0] == tag][0]
        jobs += [(f"{tag}|{dt}|{law}", "x", (st, obj, v["lag_max"], v["lag_max_z"], dt, lag, law)) for dt in (0.001, 0.0005) for law in (True, False)]
    out = {}
    for k, _, r in pool.map(_tagged, jobs, chunksize=1):
        tag, dt, law = k.split("|"); out.setdefault(tag, {})[f"{'law' if law == 'True' else 'lawoff'}_{dt}"] = r
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default="ABC"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget", type=int, default=48); ap.add_argument("--wall_h", type=float, default=2.5); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); t_end = time.time() + 3600*a.wall_h
    with mp.get_context("fork").Pool(a.workers, maxtasksperchild=8) as pool:
        if "A" in a.phase:
            A = phase_A(pool, 0.010); nv = sum(v["valid_c2"] for v in A.values()); os.system(f"~/bin/notify 'Q1 lag replay done: valid at 10ms {nv}/{len(A)}' 2>/dev/null")
        if "B" in a.phase or "C" in a.phase:
            A = json.load(open(f"{OUT}/replay_lag10ms.json")); Lp = f"{OUT}/lag_maxima.json"; L = json.load(open(Lp)) if os.path.exists(Lp) else {}
            plan = []
            if "B" in a.phase: plan += [(c, 0.010) for c in CASES]
            if "C" in a.phase: plan.insert(1, (CASES[0], 0.020))     # the pot at 20 ms right after the 10 ms pot
            for c, lag in plan:
                key = f"{c[0]}_lag{int(lag*1000)}ms"
                if key in L and L[key].get("lag_max") is not None: continue
                rep = A[c[0]]["valid_c2"] if lag == 0.010 else False
                L[key] = ladder(pool, *c, lag, a.budget, t_end, rep); json.dump(J(L), open(Lp, "w"), indent=1)
                lm = L[key].get("lag_max"); os.system(f"~/bin/notify 'Q1 {key}: lag max {lm} kg (stored {c[3]:g})' 2>/dev/null")
            for lag in (0.010, 0.020):
                sub = {k.rsplit("_lag", 1)[0]: v for k, v in L.items() if k.endswith(f"_lag{int(lag*1000)}ms")}
                ab = ablate(pool, sub, lag)
                for tag, v in ab.items(): L[f"{tag}_lag{int(lag*1000)}ms"]["ablation"] = v
                json.dump(J(L), open(Lp, "w"), indent=1)
    log("Q1 driver finished " + a.phase)
