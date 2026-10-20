"""Q4: does analytic required_depth discriminate inside the dynamic operating range?"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, bank as BK
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results")
D = pickle.load(open(f"{R}/pose_bank_certified_v2.pkl", "rb"))["G1"]
caps = {n: tuple(v) for n, v in D["caps"].items()}; mass = np.array(D["mass"])
def frontier(T):
    rows = BK.frontier_from_bank(D["bank"], D["groups"], caps, mass, T)
    return {tuple(r["subset"]): r["v_takeoff"] for r in rows}
def req_depth(F, dem):
    out = []
    for dd in dem:
        ks = [len(s) for s, v in F.items() if v >= dd]
        out.append(min(ks) if ks else np.nan)
    return np.array(out)
dem = np.linspace(0.3, 7.0, 135)
res = {}
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for T in (0.25, 0.15, 0.12):
    F = frontier(T); rd = req_depth(F, dem)
    first = dem[np.argmax(rd > 1)] if np.any(rd > 1) else np.nan
    res[T] = dict(vmax=max(F.values()), first_transition=float(first), depth_at={f"{d:.1f}": float(req_depth(F, [d])[0]) for d in (0.5, 1.0, 1.5, 2.0, 2.4)})
    ax[0].step(dem, rd, where="post", label=f"analytic, T_push = {T} s")
    print(f"T={T}: analytic vmax {max(F.values()):.2f}, required_depth leaves 1 at D = {first:.2f} m/s; depth at 0.5..2.4: {list(res[T]['depth_at'].values())}")
ax[0].axvspan(0.5, 2.42, color="0.85", label="dynamic operating range (0.5-2.42 m/s)")
ax[0].set_xlabel("demanded takeoff velocity D (m/s)"); ax[0].set_ylabel("analytic required_depth (groups)"); ax[0].legend(fontsize=7)
pr = json.load(open(f"{R}/phase9/pr_demand_faithful.json"))
g = [o for o in pr if o["task"] == "T1" and o["valid"]]
ax[1].scatter([o["v"] for o in g], [o["PR"] for o in g], s=10, label="dynamic participation ratio (faithful 1 ms)")
F = frontier(0.25); ax[1].step(dem, req_depth(F, dem), where="post", color="k", label="analytic required_depth (T=0.25)")
ax[1].set_xlim(0, 3); ax[1].set_xlabel("takeoff velocity (m/s)"); ax[1].set_ylabel("groups / effective groups"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{R}/phase9/figs/analytic_vs_dynamic_overlay.png", dpi=130)
json.dump(res, open(f"{R}/phase9/analytic_verdict.json", "w"), indent=1)
