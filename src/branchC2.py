"""Branch C, wide-range test. The G1->H1 interpolation may be too narrow a family for a
normalizer to demonstrate anything. This draws per-group capacities log-uniformly over a
wide range and repeats the collapse test, plus Branch F.4 (does any structure survive
unrealistic specs?)."""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, bank as BK

G = 9.80665; T = 0.25
D = pickle.load(open(os.environ.get("MLAW_BANK", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/pose_bank.pkl")), "rb"))
g1 = D["G1"]; OUT = os.environ.get("MLAW_OUT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results"))
gi = g1["group_index"]; GROUPS = list(gi)
rng = np.random.default_rng(11)

N = 80
bodies = []
for i in range(N):
    caps = {}
    for g in GROUPS:
        base = tuple(g1["caps"][gi[g][0]])
        ft = float(np.exp(rng.uniform(np.log(0.2), np.log(5.0))))
        fo = float(np.exp(rng.uniform(np.log(0.2), np.log(5.0))))
        for n in gi[g]: caps[n] = (base[0]*ft, base[1]*fo)
    ms = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
    mass = np.array(g1["mass"])*ms
    rows = BK.frontier_from_bank(g1["bank"], GROUPS, caps, mass, T)
    bodies.append(dict(caps=caps, mass=mass, vmax=max(r["v_takeoff"] for r in rows),
                       rows=[(len(r["subset"]), r["v_takeoff"]) for r in rows]))

def depth_of(rows, Dm):
    ok = [k for k, v in rows if v >= Dm]
    return min(ok) if ok else None

def stats(b):
    caps = b["caps"]
    tot_tau = sum(c[0] for c in caps.values())
    tot_pow = sum(c[0]*c[1]/4.0 for c in caps.values())
    dist = gi["ankles"]+gi["knees"]
    d_pow = sum(caps[n][0]*caps[n][1]/4.0 for n in dist)
    d_tau = sum(caps[n][0] for n in dist)
    leg = gi["ankles"]+gi["knees"]+gi["hips"]
    l_pow = sum(caps[n][0]*caps[n][1]/4.0 for n in leg)
    M = float(np.sum(b["mass"]))
    return dict(none=1.0, total_power=tot_pow, total_torque=tot_tau, distal_power=d_pow,
                distal_torque=d_tau, leg_power=l_pow, mass=M,
                v_from_total_power=np.sqrt(2*tot_pow*T/M),
                v_from_leg_power=np.sqrt(2*l_pow*T/M),
                v_from_distal_power=np.sqrt(2*d_pow*T/M),
                v_from_torque=np.sqrt(2*tot_tau*2.0/M),
                power_per_mass=tot_pow/M)

DEM = np.linspace(0.5, np.quantile([b["vmax"] for b in bodies], 0.95), 45)
KEYS = list(stats(bodies[0]))

def collapse(key, nbin=14):
    curves = []
    for b in bodies:
        c = stats(b)[key]
        curves.append([(Dm/c, depth_of(b["rows"], Dm)) for Dm in DEM])
        curves[-1] = [(x, d) for x, d in curves[-1] if d is not None]
    allx = np.array([x for cu in curves for x, _ in cu])
    edges = np.quantile(allx, np.linspace(0, 1, nbin+1))
    sds = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        vals = [d for cu in curves for x, d in cu if lo <= x <= hi]
        if len(vals) >= 15: sds.append(np.std(vals))
    return float(np.mean(sds)) if sds else float("nan"), len(sds)

res = {k: collapse(k) for k in KEYS}
vm = [b["vmax"] for b in bodies]
print(f"BRANCH C wide-range: {N} random bodies, per-group capacity 0.2x-5x, mass 0.5x-2x")
print(f"  vmax range {min(vm):.2f} - {max(vm):.2f} m/s (ratio {max(vm)/min(vm):.1f}x)")
dv = [depth_of(b['rows'], 3.0) for b in bodies]
dv = [x for x in dv if x is not None]
print(f"  depth at raw D=3.0 m/s across bodies: min {min(dv)} max {max(dv)} SD {np.std(dv):.3f} (n={len(dv)})")
print(f"\n  {'normalizer':<24}{'collapse SD':>13}{'bins':>7}")
for k, (m, nb) in sorted(res.items(), key=lambda x: (np.isnan(x[1][0]), x[1][0])):
    print(f"  {k:<24}{m:13.4f}{nb:7d}")
good = sorted([(k, v[0]) for k, v in res.items() if not np.isnan(v[0])], key=lambda x: x[1])
print(f"\n  best      : {good[0][0]} at {good[0][1]:.4f}")
print(f"  unnorm    : {res['none'][0]:.4f}")
print(f"  reduction : {(res['none'][0]-good[0][1])/res['none'][0]*100:.1f}%")
json.dump({k: list(v) for k, v in res.items()}, open(f"{OUT}/branchC_wide.json", "w"), indent=1)
