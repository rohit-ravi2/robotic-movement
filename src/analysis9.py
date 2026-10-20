"""All Phase 8 headline numbers recomputed from the Phase 9 FAITHFUL replays (A1/A2/A4 fixed)."""
import json, numpy as np, os
R = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase9")
L = lambda n: json.load(open(f"{R}/{n}_faithful.json"))
rng = np.random.default_rng(0)
def boot(x, f=np.mean, n=4000):
    x = np.asarray(x); return [float(np.quantile([f(x[rng.integers(0, len(x), len(x))]) for _ in range(n)], q)) for q in (.025, .975)]
out = {}
# repro
r = L("repro")
for t in ("T1", "T2"):
    v = np.array([o["v"] for o in r if o["task"] == t and o["valid"]])
    out[f"repro_{t}"] = dict(n=len(v), median=float(np.median(v)), best=float(v.max()), sd=float(v.std()))
    print(f"REPRO {t}: n {len(v)} median {np.median(v):.3f} best {v.max():.3f} sd {v.std():.3f}; flight height median {np.median(v)**2/19.6133*100:.1f} cm")
# PR slope
p = L("pr_demand")
for task in ("T1", "reserve"):
    g = [o for o in p if o["task"] == task and o["valid"]]; x = np.array([o["v"] for o in g]); y = np.array([o["PR"] for o in g])
    b = np.polyfit(x, y, 1)[0]; bs = [np.polyfit(x[i], y[i], 1)[0] for i in [rng.integers(0, len(x), len(x)) for _ in range(3000)]]
    by = [round(float(np.mean([o["PR"] for o in g if o["target"] == t])), 2) for t in sorted({o["target"] for o in g})]
    out[f"PR_{task}"] = dict(slope=float(b), ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], by_target=by)
    print(f"PR {task}: slope {b:.2f} CI [{np.quantile(bs,.025):.2f},{np.quantile(bs,.975):.2f}] by target {by}")
d = L("pr_duration"); g = [o for o in d if o["valid"]]
X = np.c_[np.ones(len(g)), [o["v"] for o in g], [o["T_push"] for o in g]]; y = np.array([o["PR"] for o in g])
b = np.linalg.lstsq(X, y, rcond=None)[0]; bs = np.array([np.linalg.lstsq(X[i], y[i], rcond=None)[0] for i in [rng.integers(0, len(g), len(g)) for _ in range(3000)]])
out["PR_duration"] = dict(coef=b.tolist(), ci_v=np.quantile(bs[:, 1], [.025, .975]).tolist(), ci_T=np.quantile(bs[:, 2], [.025, .975]).tolist())
print(f"PR = {b[0]:.2f} + {b[1]:.2f} v [{np.quantile(bs[:,1],.025):.2f},{np.quantile(bs[:,1],.975):.2f}] + {b[2]:.1f} T [{np.quantile(bs[:,2],.025):.1f},{np.quantile(bs[:,2],.975):.1f}]")
# asymmetry
for name in ("asym_budget_v2", "asym_budget"):
    a = L(name); print(f"ASYM {name}:")
    for it in sorted({o["iters"] for o in a}):
        row = []
        for c in ("PROXIMAL_STRONG", "DISTAL_STRONG", "MIDDLE_STRONG", "UNIFORM"):
            v = [o["v"] for o in a if o["morph"] == c and o["iters"] == it and o["valid"]]
            row.append(f"{c[:5]} {np.median(v):.2f}/{max(v):.2f}")
        gaps = []
        for s in range(1, 9):
            pp = [o["v"] for o in a if o["morph"] == "PROXIMAL_STRONG" and o["iters"] == it and o["seed"] == s and o["valid"]]
            dd = [o["v"] for o in a if o["morph"] == "DISTAL_STRONG" and o["iters"] == it and o["seed"] == s and o["valid"]]
            if pp and dd: gaps.append(pp[0] - dd[0])
        ci = boot(gaps) if len(gaps) > 1 else [np.nan, np.nan]
        out.setdefault(name, {})[it] = dict(row=row, gap=float(np.mean(gaps)), ci=ci)
        print(f"   iters {it:>5}: " + " | ".join(row) + f" | PROX-DIST {np.mean(gaps):+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]")
c = L("asym_cem"); print("CEM medians:", {m: round(float(np.median([o['v'] for o in c if o['morph'] == m and o['valid']])), 2) for m in sorted({o['morph'] for o in c})})
sm = L("spec_map"); print("SPEC medians:", {m: round(float(np.median([o['v'] for o in sm if o['morph'] == m and o['valid']])), 2) for m in sorted({o['morph'] for o in sm}, key=lambda z: float(z[2:]))})
T = ["ankle_pitch_joint", "knee_joint", "hip_pitch_joint", "waist_pitch_joint", "shoulder_pitch_joint", "elbow_joint"]
for m_ in sorted({o['morph'] for o in sm}, key=lambda z: float(z[2:])):
    g = [o for o in sm if o["morph"] == m_ and o["valid"]]
    sh = np.mean([np.array([o["work"][t] for t in T])/sum(o["work"][t] for t in T) for o in g], 0)
    print(f"   {m_}: PR {np.mean([o['PR'] for o in g]):.2f} shares {np.round(sh,2)}")
ab = L("ablations"); fac = {m: float(np.median([o['v'] for o in ab if o['morph'] == m and o['valid']])) for m in sorted({o['morph'] for o in ab}) if m.startswith("FAC")}
print("FACTORIAL:", {k: round(v, 2) for k, v in fac.items()})
SS, SW, WS, WW = fac["FAC:knee=S,hip=S"], fac["FAC:knee=S,hip=W"], fac["FAC:knee=W,hip=S"], fac["FAC:knee=W,hip=W"]
print(f"   knee main {(SS+SW-WS-WW)/2:+.2f} hip main {(SS+WS-SW-WW)/2:+.2f} interaction {(SS-SW)-(WS-WW):+.2f}")
# benchmark
for name in ("baseline", "baseline_budget400"):
    bm = L(name); print(f"BENCHMARK {name} (faithful 1 ms; per clip OURS minus B1, median-of-seeds):")
    clips = sorted({o["clip"] for o in bm if o.get("set", "primary") == "primary"})
    for task in ("T2", "T1"):
        for lam in (10.0, 1.0, 0.1, 0.0):
            gs = []
            for cl in clips:
                a = [o["v"] for o in bm if o["clip"] == cl and o["method"] == "OURS" and o["task"] == task and o["valid"]]
                b_ = [o["v"] for o in bm if o["clip"] == cl and o["method"] == "B1" and o["lam"] == lam and o["task"] == task and o["valid"]]
                if a and b_: gs.append(np.median(a) - np.median(b_))
            if gs:
                print(f"   {task} lambda {lam:>4}: gaps {np.round(gs,2)} mean {np.mean(gs):+.3f}")
                out.setdefault(name, {})[f"{task}_{lam}"] = dict(gaps=list(map(float, gs)), mean=float(np.mean(gs)))
    b0 = [o for o in bm if o["method"] == "B0"]
    if b0: print(f"   B0 valid {sum(o['valid'] for o in b0)}/{len(b0)}; v of valid: {[round(o['v'],2) for o in b0 if o['valid']]}")
json.dump(out, open(f"{R}/analysis9.json", "w"), indent=1, default=float)
