"""Asymmetry: mechanical ceiling vs optimizer accessibility."""
import json, pickle, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bank as BK
R = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results")
T = ["ankle_pitch_joint", "knee_joint", "hip_pitch_joint", "waist_pitch_joint", "shoulder_pitch_joint", "elbow_joint"]
t2g = dict(zip(T, ["ankles", "knees", "hips", "waist", "shoulders", "elbows"]))
out = {}

def analytic(sc, bankfile):
    D = pickle.load(open(f"{R}/{bankfile}", "rb"))["G1"]
    caps = BK.caps_from_scaled(D, {t2g[t]: v for t, v in (sc or {}).items()})
    rows = BK.frontier_from_bank(D["bank"], D["groups"], caps, np.array(D["mass"]), 0.25)
    return max(r["v_takeoff"] for r in rows)

import phase8_campaign as PC
if os.path.exists(f"{R}/phase8/asym_budget.json"):
    d = json.load(open(f"{R}/phase8/asym_budget.json"))["runs"]
    conds = ["PROXIMAL_STRONG", "DISTAL_STRONG", "MIDDLE_STRONG", "UNIFORM"]
    print("ANALYTIC additive bound (old uncertified / certified bank):")
    ana = {}
    for c in conds:
        ana[c] = (analytic(PC.morph(c), "pose_bank.pkl"), analytic(PC.morph(c), "pose_bank_certified.pkl"))
        print(f"   {c:<16} {ana[c][0]:.3f} / {ana[c][1]:.3f} m/s")
    print("\nACCESSIBILITY CURVES: takeoff v (valid runs only) by budget (iters x 20 rollouts)")
    curves = {}
    for c in conds:
        curves[c] = {}
        for it in sorted({r["iters"] for r in d}):
            v = np.array([r["v"] for r in d if r["morph"] == c and r["iters"] == it and r["valid"]])
            nv = sum(1 for r in d if r["morph"] == c and r["iters"] == it)
            curves[c][it] = dict(best=float(v.max()) if len(v) else None, median=float(np.median(v)) if len(v) else None,
                                 sd=float(v.std()) if len(v) else None, q25=float(np.quantile(v, .25)) if len(v) else None,
                                 q75=float(np.quantile(v, .75)) if len(v) else None, n_valid=int(len(v)), n=nv)
    its = sorted({r["iters"] for r in d})
    print(f"{'iters':>6} " + " ".join(f"{c[:12]:>26}" for c in conds))
    for it in its:
        print(f"{it:>6} " + " ".join(f"   best {curves[c][it]['best']:.2f} med {curves[c][it]['median']:.2f} sd {curves[c][it]['sd']:.2f}" if curves[c][it]['best'] else f"{'none valid':>26}" for c in conds))
    # paired by seed at each budget: prox - dist
    print("\nPROX - DIST, paired by seed (same seed = same CMA random stream):")
    gaps = {}
    for it in its:
        g = []
        for s in range(1, 9):
            a = [r for r in d if r["morph"] == "PROXIMAL_STRONG" and r["iters"] == it and r["seed"] == s and r["valid"]]
            b = [r for r in d if r["morph"] == "DISTAL_STRONG" and r["iters"] == it and r["seed"] == s and r["valid"]]
            if a and b: g.append(a[0]["v"] - b[0]["v"])
        g = np.array(g); bs = [np.mean(np.random.default_rng(i).choice(g, len(g))) for i in range(2000)]
        gaps[it] = dict(mean=float(g.mean()), ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                        best_gap=curves["PROXIMAL_STRONG"][it]["best"] - curves["DISTAL_STRONG"][it]["best"], n=int(len(g)))
        print(f"  iters {it:>5}: mean gap {g.mean():+.3f} m/s CI [{gaps[it]['ci'][0]:+.3f},{gaps[it]['ci'][1]:+.3f}]  best-vs-best {gaps[it]['best_gap']:+.3f}  n={len(g)}")
    # failure diagnostics for poor seeds at the top budget
    top = its[-1]
    print(f"\nPOOR-SEED DIAGNOSTICS at {top} iters (below condition median - 1 SD):")
    for c in conds:
        cv = curves[c][top]
        for r in d:
            if r["morph"] == c and r["iters"] == top and (not r["valid"] or r["v"] < cv["median"] - cv["sd"]):
                mc = r["mech"]
                print(f"   {c[:10]} seed {r['seed']} v {r['v']:.2f} valid {r['valid']} reasons {r['validity']['reasons']} T {mc['T_push']:.3f} sat {r['validity']['saturation_frac']:.2f} unload {mc['unload_time']:.3f} PR {r['participation']:.2f}")
    out.update(analytic=ana, curves=curves, gaps=gaps)
    # mechanism: shares, timing at top budget
    print("\nMECHANISM at top budget: work shares and peak-velocity times (ms) [ankle knee hip waist shoulder elbow]")
    mech = {}
    for c in conds:
        rr = [r for r in d if r["morph"] == c and r["iters"] == top and r["valid"]]
        sh = np.array([[r["work"][t]/max(r["total_work"], 1e-9) for t in T] for r in rr])
        tp = []
        for r in rr:
            tr = r["trace"]; to = r["takeoff_t"]; tt = np.array(tr["t"]); k = tt <= to
            tp.append([tt[k][np.argmax(np.array(tr["qvel"][t])[k]*np.sign(np.mean(np.array(tr["qvel"][t])[k]) or 1))]*1000 for t in T])
        tp = np.array(tp)
        mech[c] = dict(shares=sh.mean(0).tolist(), shares_sd=sh.std(0).tolist(), tpeak_ms=np.median(tp, 0).tolist(),
                       T_push=float(np.mean([r["mech"]["T_push"] for r in rr])), PR=float(np.mean([r["participation"] for r in rr])),
                       peak_util=float(np.mean([r["mech"]["peak_util"] for r in rr])), grf_bw=float(np.mean([r["grf_bw"] for r in rr])))
        print(f"   {c:<16} shares {np.round(sh.mean(0),2)} tpeak {np.round(np.median(tp,0)).astype(int)} T {mech[c]['T_push']:.3f} PR {mech[c]['PR']:.2f} util {mech[c]['peak_util']:.2f}")
    out["mechanism"] = mech
for f in ("asym_cem", "ablations", "spec_map"):
    if os.path.exists(f"{R}/phase8/{f}.json"):
        d = json.load(open(f"{R}/phase8/{f}.json"))["runs"]
        print(f"\n== {f}")
        for mname in sorted({r["morph"] for r in d}):
            v = np.array([r["v"] for r in d if r["morph"] == mname and r["valid"]]); n = sum(r["morph"] == mname for r in d)
            if len(v): print(f"   {mname:<34} best {v.max():.3f} median {np.median(v):.3f} sd {v.std():.3f}  valid {len(v)}/{n}")
            out.setdefault(f, {})[mname] = dict(best=float(v.max()) if len(v) else None, median=float(np.median(v)) if len(v) else None,
                                                  sd=float(v.std()) if len(v) else None, n_valid=int(len(v)), n=n)
json.dump(out, open(f"{R}/phase8/asym_analysis.json", "w"), indent=1, default=float)
