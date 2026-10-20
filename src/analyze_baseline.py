"""North-star benchmark analysis. Primary numbers = 1 ms replay."""
import json, numpy as np, os
R = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase8")
d = json.load(open(f"{R}/baseline.json"))["runs"]
rp_path = f"{R}/baseline_replay_dt0.001.json"
rp = json.load(open(rp_path)) if os.path.exists(rp_path) else None
T = ["ankle_pitch_joint", "knee_joint", "hip_pitch_joint", "waist_pitch_joint", "shoulder_pitch_joint", "elbow_joint"]
def key(r): return (r["clip"], r["method"], r.get("lam"), r.get("seed"), r.get("task"), r.get("tscale") if r["method"] == "B0" else None)
rpk = {key(r): r for r in rp} if rp else {}
rows = []
for r in d:
    k = key(r); q = rpk.get(k)
    rows.append(dict(clip=r["clip"], set=r.get("set"), method=r["method"], lam=r.get("lam"), seed=r.get("seed"), task=r.get("task"),
                     tscale=r.get("tscale"), v4=r["v"], valid4=r["valid"], v1=q["v"] if q else None,
                     # 1 ms replay gate: every physics-consistency check; CONTACT_ENERGY (a coarse-dt
                     # spring bound) is recorded, not gating, since the 1 ms replay IS that test
                     valid1=(q is not None and not [x for x in q["reasons"] if x != "CONTACT_ENERGY"]),
                     dev=r.get("deviation"), rot=r["mech"]["flight_rotation"], T=r["mech"]["T_push"],
                     crouch=r["mech"]["crouch_depth"], PR=r["participation"],
                     shares=[r["work"][t]/max(r["total_work"], 1e-9) for t in T], secs=r.get("secs")))
use = "v1" if rp else "v4"; ok = "valid1" if rp else "valid4"
out = {}
print(f"measurement: {'1 ms replay' if rp else '4 ms (replay pending)'}\n")
for task in ("T2", "T1"):
    print(f"=== task {task}")
    for cl in sorted({r["clip"] for r in rows}):
        g = [r for r in rows if r["clip"] == cl and (r["task"] == task or r["method"] == "B0")]
        line = {}
        b0 = [r for r in g if r["method"] == "B0"]
        for r in b0: line[f"B0 ts{r['tscale']}"] = (r[use] if r[ok] else None, r[use] if r[ok] else None, r["dev"])
        for lam in (10.0, 1.0, 0.1, 0.0):
            gg = [r for r in g if r["method"] == "B1" and r["lam"] == lam and r["task"] == task]
            v = [r[use] for r in gg if r[ok]]
            line[f"B1 l={lam}"] = (max(v) if v else None, np.median(v) if v else None, np.median([r["dev"] for r in gg]) if gg else None, len(v), len(gg))
        gg = [r for r in g if r["method"] == "OURS" and r["task"] == task]
        v = [r[use] for r in gg if r[ok]]
        line["OURS"] = (max(v) if v else None, np.median(v) if v else None, np.median([r["dev"] for r in gg if r["dev"] is not None]) if gg else None, len(v), len(gg))
        out.setdefault(task, {})[cl] = line
        s = "  ".join(f"{k}: {('%.2f/%.2f' % (x[0], x[1])) if x[0] is not None else ' -- '}" + (f"(dev {x[2]:.3f})" if x[2] is not None else "") for k, x in line.items())
        print(f"  {cl[:28]:<28} [{g[0]['set']}] {s}")
    # paired OURS vs best human-anchored B1 per clip (best over seeds; equal budget)
    print(f"  --- OURS minus B1 at each lambda (primary clips; best-of-seeds | median-of-seeds)")
    for lam in (10.0, 1.0, 0.1, 0.0):
        gaps = []
        for cl in out[task]:
            if [r for r in rows if r["clip"] == cl][0]["set"] != "primary": continue
            a = out[task][cl]["OURS"]; b = out[task][cl][f"B1 l={lam}"]
            if a[0] is not None and b[0] is not None: gaps.append((a[0]-b[0], a[1]-b[1]))
        if gaps:
            g = np.array(gaps)
            print(f"     lambda {lam:>5}: best-gaps {np.round(g[:,0],2)} mean {g[:,0].mean():+.3f} | median-gaps {np.round(g[:,1],2)} mean {g[:,1].mean():+.3f}  ({int(sum(g[:,1] > 0))}/{len(g)} clips OURS median higher)")
json.dump(dict(summary=out, rows=rows), open(f"{R}/baseline_analysis.json", "w"), indent=1, default=float)
