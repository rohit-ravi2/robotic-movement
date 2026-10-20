"""BENCHMARK_V2 analysis: lambda=0 primary, paired bootstrap CIs, analytic-prior arm, deviation anatomy."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mocap_baseline as MB, branch_i as BI
from scipy.interpolate import CubicSpline
R = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase9")
d = json.load(open(f"{R}/baseline9.json")); rng = np.random.default_rng(0)
T = ["ankle_pitch_joint", "knee_joint", "hip_pitch_joint", "waist_pitch_joint", "shoulder_pitch_joint", "elbow_joint"]
def boot_mean(x, n=5000):
    x = np.asarray(x); b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]; return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
out = {"clips": {}}
clips = sorted({r["clip"] for r in d})
print("per clip, task T2 (primary) and T1: median v [n valid]")
for task in ("T2", "T1"):
    allgaps = {"B1": [], "APRIOR": []}
    for cl in clips:
        v = {m: {r["seed"]: r["v"] for r in d if r["clip"] == cl and r["method"] == m and r["task"] == task and r["valid"]} for m in ("OURS", "B1", "APRIOR")}
        line = {m: (float(np.median(list(v[m].values()))) if v[m] else None, len(v[m])) for m in v}
        pair = {}
        for other in ("B1", "APRIOR"):
            seeds = sorted(set(v["OURS"]) & set(v[other]))
            if len(seeds) >= 3:
                g = [v["OURS"][s] - v[other][s] for s in seeds]; allgaps[other] += g
                pair[other] = dict(mean=float(np.mean(g)), ci=boot_mean(g), n=len(seeds))
        b0 = [r["v"] for r in d if r["clip"] == cl and r["method"] == "B0" and r["valid"]]
        out["clips"].setdefault(cl, {})[task] = dict(median=line, paired=pair, B0=b0)
        ps = "  ".join(f"OURS-{k} {p['mean']:+.3f} [{p['ci'][0]:+.3f},{p['ci'][1]:+.3f}] n{p['n']}" for k, p in pair.items())
        print(f"  {task} {cl[:30]:<30} OURS {line['OURS'][0]:.2f}[{line['OURS'][1]}]  B1(l=0) {line['B1'][0] if line['B1'][0] else float('nan'):.2f}[{line['B1'][1]}]  APRIOR {line['APRIOR'][0] if line['APRIOR'][0] else float('nan'):.2f}[{line['APRIOR'][1]}]  B0 {np.round(b0,2).tolist()}  | {ps}")
    for other, g in allgaps.items():
        if g: out[f"pooled_{task}_OURS-{other}"] = dict(mean=float(np.mean(g)), ci=boot_mean(g), n=len(g)); print(f"  POOLED {task} OURS-{other}: {np.mean(g):+.3f} {boot_mean(g)} (n={len(g)} seed-pairs; clips are not independent of each other's seeds)")
# deviation anatomy: OURS and B1 vs the mocap reference, per joint type and push-off phase
print("\nDEVIATION ANATOMY (T2): mean squared deviation from mocap reference (rad^2) by joint, and by push-off third")
anat = {}
for cl in clips:
    c = MB.load_clip(cl, "retargeted_g1_all.pkl"); S = BI.setup(c["start"]); J = S["jarr"]
    tq = np.arange(len(c["Q"]))/c["fps"]
    ref = {t: CubicSpline(tq, c["Q"][:, S["tmap"][t][0][2]]*S["tmap"][t][0][5]) for t in T}
    for m in ("OURS", "B1"):
        rows = [r for r in d if r["clip"] == cl and r["method"] == m and r["task"] == "T2" and r["valid"] and r.get("trace")]
        per_j, per_ph, desc = [], [], []
        for r in rows:
            tr = r["trace"]; t = np.array(tr["t"]); to = r["T_push"] or t[-1]; k = t <= to
            dj = {tt: np.array(tr["q"][tt])[k] - ref[tt](np.minimum(t[k], tq[-1])) for tt in T}
            per_j.append([float(np.mean(dj[tt]**2)) for tt in T])
            thirds = np.array_split(np.arange(k.sum()), 3)
            per_ph.append([float(np.mean([np.mean(dj[tt][idx]**2) for tt in T])) for idx in thirds])
            desc.append([r["crouch"], r["T_push"], r["PR"]] + [r["work"][tt]/max(sum(r["work"].values()), 1e-9) for tt in T])
        anat.setdefault(cl, {})[m] = dict(per_joint=np.mean(per_j, 0).tolist() if per_j else None, per_phase=np.mean(per_ph, 0).tolist() if per_ph else None,
                                          descriptors=np.mean(desc, 0).tolist() if desc else None, n=len(rows))
        if per_j: print(f"  {cl[:22]:<22} {m:<4} joints {np.round(np.mean(per_j,0),3).tolist()} phases {np.round(np.mean(per_ph,0),3).tolist()} | crouch {np.mean(desc,0)[0]*100:.1f} cm T {np.mean(desc,0)[1]*1000:.0f} ms PR {np.mean(desc,0)[2]:.2f} shares {np.round(np.mean(desc,0)[3:],2).tolist()}")
out["anatomy"] = anat
json.dump(out, open(f"{R}/bench9_analysis.json", "w"), indent=1, default=float)
