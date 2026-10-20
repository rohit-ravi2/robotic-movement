"""Mechanism campaigns, measured by 1 ms replay (physics gates; CONTACT_ENERGY recorded only)."""
import json, numpy as np, os
R = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase8")
def load(name):
    p = f"{R}/{name}_replay_dt0.001.json"
    return json.load(open(p)) if os.path.exists(p) else []
def ok(o): return not [x for x in o["reasons"] if x != "CONTACT_ENERGY"]
out = {}
a = load("asym_budget_v2")
print("ASYMMETRY v2 (optimised at 2 ms, measured at 1 ms): median / best / n_valid")
conds = ["PROXIMAL_STRONG", "DISTAL_STRONG", "MIDDLE_STRONG", "UNIFORM"]
for it in sorted({o["iters"] for o in a}):
    row = []
    for c in conds:
        v = [o["v_new"] for o in a if o["morph"] == c and o["iters"] == it and ok(o)]
        vo = [o["v_old"] for o in a if o["morph"] == c and o["iters"] == it]
        row.append(f"{c[:6]} {np.median(v):.2f}/{max(v):.2f}/{len(v)} (2ms {np.median(vo):.2f})" if v else f"{c[:6]} none")
        out.setdefault("asym", {}).setdefault(c, {})[it] = dict(median=float(np.median(v)) if v else None, best=float(max(v)) if v else None, n=len(v), median_2ms=float(np.median(vo)))
    print(f"  iters {it:>5}: " + " | ".join(row))
print("  paired PROX-DIST (same seed) at 1 ms:")
for it in sorted({o["iters"] for o in a}):
    g = []
    for s in range(1, 7):
        p = [o for o in a if o["morph"] == "PROXIMAL_STRONG" and o["iters"] == it and o["seed"] == s and ok(o)]
        q = [o for o in a if o["morph"] == "DISTAL_STRONG" and o["iters"] == it and o["seed"] == s and ok(o)]
        if p and q: g.append(p[0]["v_new"] - q[0]["v_new"])
    if g:
        g = np.array(g); bs = [np.mean(np.random.default_rng(i).choice(g, len(g))) for i in range(2000)]
        print(f"     iters {it:>5}: mean {g.mean():+.3f} CI [{np.quantile(bs,.025):+.3f},{np.quantile(bs,.975):+.3f}] n={len(g)}")
        out.setdefault("asym_gap", {})[it] = dict(mean=float(g.mean()), ci=[float(np.quantile(bs,.025)), float(np.quantile(bs,.975))], n=len(g))
c = load("asym_cem")
print("\nCEM (independent optimizer), 1 ms: median / best / n")
for m in sorted({o["morph"] for o in c}):
    v = [o["v_new"] for o in c if o["morph"] == m and ok(o)]
    print(f"  {m:<16} {np.median(v):.2f} / {max(v):.2f} / {len(v)}"); out.setdefault("cem", {})[m] = dict(median=float(np.median(v)), best=float(max(v)), n=len(v))
ab = load("ablations")
print("\nABLATIONS / FACTORIAL, 1 ms: median / best / n")
for m in sorted({o["morph"] for o in ab}):
    v = [o["v_new"] for o in ab if o["morph"] == m and ok(o)]
    if v: print(f"  {m:<40} {np.median(v):.2f} / {max(v):.2f} / {len(v)}"); out.setdefault("abl", {})[m] = dict(median=float(np.median(v)), best=float(max(v)), n=len(v))
sm = load("spec_map")
print("\nSPEC MAP (r>0 proximal-strong), 1 ms: median / best / n ; PR ; shares [ank kne hip wai sho elb]")
T = ["ankle_pitch_joint", "knee_joint", "hip_pitch_joint", "waist_pitch_joint", "shoulder_pitch_joint", "elbow_joint"]
for m in sorted({o["morph"] for o in sm}, key=lambda x: float(x[2:])):
    g = [o for o in sm if o["morph"] == m and ok(o)]; v = [o["v_new"] for o in g]
    if not v: continue
    sh = np.mean([[o["work_new"][t] for t in T]/np.sum([o["work_new"][t] for t in T]) for o in g], 0)
    print(f"  {m:<8} {np.median(v):.2f} / {max(v):.2f} / {len(v)}  PR {np.mean([o['PR_new'] for o in g]):.2f}  shares {np.round(sh,2)}")
    out.setdefault("spec", {})[m] = dict(median=float(np.median(v)), best=float(max(v)), n=len(v), PR=float(np.mean([o['PR_new'] for o in g])), shares=sh.tolist())
d = load("pr_duration")
print("\nPR: DEMAND x DURATION control, 1 ms: mean PR (n) [target rows x Tc cols 0.06/0.10/0.14]")
for t in (0.5, 1.1, 1.7):
    row = []
    for Tc in (0.06, 0.1, 0.14):
        g = [o for o in d if o["target"] == t and abs(float(o["task"].strip("[]").split(",")[1]) - Tc) < 1e-9 and ok(o)]
        row.append(f"PR {np.mean([o['PR_new'] for o in g]):.2f} T {np.mean([o['T_push'] for o in g]):.3f} v {np.mean([o['v_new'] for o in g]):.2f} (n{len(g)})" if g else "none")
        out.setdefault("dur", {})[f"{t}/{Tc}"] = dict(PR=float(np.mean([o['PR_new'] for o in g])) if g else None, T=float(np.mean([o['T_push'] for o in g])) if g else None, n=len(g))
    print(f"  target {t}: " + " | ".join(row))
# regression PR ~ v + T across all valid duration runs
g = [o for o in d if ok(o)]
X = np.c_[np.ones(len(g)), [o["v_new"] for o in g], [o["T_push"] for o in g]]; y = np.array([o["PR_new"] for o in g])
b = np.linalg.lstsq(X, y, rcond=None)[0]
bs = []
for i in range(2000):
    k = np.random.default_rng(i).integers(0, len(g), len(g)); bs.append(np.linalg.lstsq(X[k], y[k], rcond=None)[0])
bs = np.array(bs)
print(f"  PR = {b[0]:.2f} + {b[1]:.2f}*v [{np.quantile(bs[:,1],.025):.2f},{np.quantile(bs[:,1],.975):.2f}] + {b[2]:.1f}*T_push [{np.quantile(bs[:,2],.025):.1f},{np.quantile(bs[:,2],.975):.1f}]  (n={len(g)})")
out["dur_regression"] = dict(coef=b.tolist(), ci_v=np.quantile(bs[:,1],[.025,.975]).tolist(), ci_T=np.quantile(bs[:,2],[.025,.975]).tolist(), n=len(g))
json.dump(out, open(f"{R}/mechanism_analysis.json", "w"), indent=1)
