"""PR central test, formal closure."""
import json, numpy as np, sys
import os; R = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase8")
d = json.load(open(f"{R}/pr_demand.json"))["runs"]
rep = json.load(open(f"{R}/repro.json"))["runs"]
G = 9.80665; M = 33.34
TYPES = ["ankle_pitch_joint", "knee_joint", "hip_pitch_joint", "waist_pitch_joint", "shoulder_pitch_joint", "elbow_joint"]

def dims(r):
    """Alternative dimensionality metrics from the push-off trace."""
    tr = r.get("trace"); to = r["takeoff_t"]
    if not tr or to is None: return {}
    t = np.array(tr["t"]); k = t <= to
    V = np.array([tr["qvel"][x] for x in TYPES])[:, k]            # types x time
    Pw = np.array([np.array(tr["tq"][x])[k]*np.array(tr["qvel"][x])[k] for x in TYPES])
    out = {}
    for name, X in (("vel", V), ("power", Pw)):
        C = np.cov(X); ev = np.clip(np.linalg.eigvalsh(C), 0, None); ev = ev[::-1]
        out[f"{name}_eig_PR"] = float(ev.sum()**2/np.sum(ev**2)) if ev.sum() > 0 else 0.0
        cs = np.cumsum(ev)/ev.sum(); out[f"{name}_n90"] = int(np.searchsorted(cs, 0.9) + 1)
        s = np.linalg.svd(X, compute_uv=False); p = s/s.sum()
        out[f"{name}_erank"] = float(np.exp(-np.sum(p*np.log(p + 1e-300))))
    wpos = np.array([max(r["work"][x], 0) for x in TYPES]); wtot = wpos.sum()
    out["PR_leftright_units"] = None
    return out

rows = []
for r in d:
    if not r["valid"]: continue
    mc = r["mech"]
    row = dict(task=r["task"], target=r["target"], seed=r["seed"], v=r["v"], PR=r["participation"],
               PR_full=r["participation_full"], E_req=0.5*M*r["v"]**2 + M*G*(mc["z_takeoff"]-mc["z_start"]),
               impulse_net=mc["impulse_net"], peak_util=mc["peak_util"], mean_util=mc["mean_util"],
               peak_power=mc["peak_power"], T=mc["T_push"], crouch=mc["crouch_depth"],
               exc_sum=sum(mc["excursion"].values()), W=r["total_work"],
               shares={x: r["work"][x]/max(r["total_work"], 1e-9) for x in TYPES}, **dims(r))
    rows.append(row)
for r in rep:
    if r["valid"] and r["task"] == "T1":
        mc = r["mech"]
        rows.append(dict(task="max", target=None, seed=r["seed"], v=r["v"], PR=r["participation"], PR_full=r["participation_full"],
                         E_req=0.5*M*r["v"]**2 + M*G*(mc["z_takeoff"]-mc["z_start"]), impulse_net=mc["impulse_net"],
                         peak_util=mc["peak_util"], mean_util=mc["mean_util"], peak_power=mc["peak_power"], T=mc["T_push"],
                         crouch=mc["crouch_depth"], exc_sum=sum(mc["excursion"].values()), W=r["total_work"],
                         shares={x: r["work"][x]/max(r["total_work"], 1e-9) for x in TYPES}, **dims(r)))
rng = np.random.default_rng(0)
def slope_ci(x, y, n=4000):
    x = np.asarray(x, float); y = np.asarray(y, float); b = np.polyfit(x, y, 1)[0]; bs = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x)); 
        if np.ptp(x[i]) > 0: bs.append(np.polyfit(x[i], y[i], 1)[0])
    return b, np.quantile(bs, [0.025, 0.975])
out = {}
for task in ("T1", "reserve"):
    R_ = [r for r in rows if r["task"] == task]
    print(f"\n=== demand formulation: {task}  (n valid = {len(R_)})")
    tg = sorted({r["target"] for r in R_})
    print(f"{'target':>6} {'v_ach':>6} {'E_req J':>8} {'imp_net':>7} {'pkutil':>6} {'mutil':>6} {'Ppk W':>6} {'W J':>6} {'PR':>10} {'PRfull':>6} {'velPR':>6} {'powPR':>6} {'erank':>6}")
    for t in tg:
        g = [r for r in R_ if r["target"] == t]; f = lambda k: np.array([r[k] for r in g if r.get(k) is not None], float)
        print(f"{t:6.2f} {f('v').mean():6.2f} {f('E_req').mean():8.1f} {f('impulse_net').mean():7.1f} {f('peak_util').mean():6.2f} {f('mean_util').mean():6.2f} {f('peak_power').mean():6.0f} {f('W').mean():6.1f} {f('PR').mean():5.2f}±{f('PR').std():.2f} {f('PR_full').mean():6.2f} {f('vel_eig_PR').mean():6.2f} {f('power_eig_PR').mean():6.2f} {f('vel_erank').mean():6.2f}")
    res = {}
    for k in ("PR", "PR_full", "vel_eig_PR", "power_eig_PR", "vel_erank", "power_erank", "vel_n90"):
        x = [r["v"] for r in R_ if r.get(k) is not None]; y = [r[k] for r in R_ if r.get(k) is not None]
        b, ci = slope_ci(x, y); sd = np.mean([np.std([r[k] for r in R_ if r["target"] == t]) for t in tg])
        span = (max(x)-min(x))*b
        res[k] = dict(slope_per_mps=b, ci95=ci.tolist(), change_over_range=span, mean_within_target_sd=sd)
        print(f"  {k:<14} slope {b:+.3f}/(m/s) CI [{ci[0]:+.3f},{ci[1]:+.3f}]  change over tested range {span:+.2f}  within-target SD {sd:.2f}")
    res["demand_range_v"] = [min(r["v"] for r in R_), max(r["v"] for r in R_)]
    res["E_ratio"] = max(r["E_req"] for r in R_)/min(r["E_req"] for r in R_)
    print(f"  achieved v range {res['demand_range_v'][0]:.2f}-{res['demand_range_v'][1]:.2f} m/s; required-energy ratio {res['E_ratio']:.1f}x")
    out[task] = res
json.dump(dict(summary=out, rows=[{k: v for k, v in r.items()} for r in rows]), open(f"{R}/pr_closure.json", "w"), indent=1, default=float)
