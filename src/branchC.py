"""Branch C, corrected: collapse test against ABSOLUTE demand (m/s), not demand expressed
as a fraction of each body's own maximum. The earlier version pre-normalized by vmax, which
collapses the curves by construction and makes every further normalizer look harmful."""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, bank as BK

G = 9.80665; T = 0.25
D = pickle.load(open(os.environ.get("MLAW_BANK", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/pose_bank.pkl")), "rb"))
g1, h1 = D["G1"], D["H1"]
OUT = os.environ.get("MLAW_OUT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results"))

def h1cap(g):
    gi = h1["group_index"]; return tuple(h1["caps"][gi[g][0]]) if g in gi else None

def body_at(alpha):
    caps = {}
    for g, names in g1["group_index"].items():
        src = tuple(g1["caps"][names[0]]); dst = h1cap(g) or src
        tau = max(src[0] + alpha*(dst[0]-src[0]), 1e-6)
        om = max(src[1] + alpha*(dst[1]-src[1]), 1e-6)
        for n in names: caps[n] = (tau, om)
    m = np.array(g1["mass"])
    sc = 1.0 + alpha*(float(np.sum(h1["mass"]))/float(np.sum(m)) - 1.0)
    return caps, m*sc

ALPHAS = list(np.round(np.linspace(-0.5, 1.5, 21), 3))
bodies = []
for a in ALPHAS:
    caps, mass = body_at(a)
    rows = BK.frontier_from_bank(g1["bank"], g1["groups"], caps, mass, T)
    bodies.append(dict(alpha=float(a), caps=caps, mass=mass,
                       vmax=max(r["v_takeoff"] for r in rows),
                       rows=[(len(r["subset"]), r["v_takeoff"]) for r in rows]))

def depth_of(rows, Dm):
    ok = [k for k, v in rows if v >= Dm]
    return min(ok) if ok else None

def stats(b):
    caps = b["caps"]; gi = g1["group_index"]
    tot_tau = sum(c[0] for c in caps.values())
    tot_pow = sum(c[0]*c[1]/4.0 for c in caps.values())
    dist = gi["ankles"]+gi["knees"]
    d_pow = sum(caps[n][0]*caps[n][1]/4.0 for n in dist)
    d_tau = sum(caps[n][0] for n in dist)
    M = float(np.sum(b["mass"])); L = 0.8
    return dict(none=1.0, total_power=tot_pow, total_torque=tot_tau,
                distal_power=d_pow, distal_torque=d_tau, mass=M, mgh=M*G*L,
                froude=M*np.sqrt(G*L),
                v_from_power=np.sqrt(2*tot_pow*T/M),          # energy-consistent velocity scale
                v_from_distal_power=np.sqrt(2*d_pow*T/M),
                v_from_torque=np.sqrt(2*tot_tau*2.0/M),       # tau * 2 rad sweep
                sqrt_gL=np.sqrt(G*L))

# absolute demand grid spanning all bodies
DEMANDS = np.linspace(0.5, max(b["vmax"] for b in bodies)*0.98, 40)
KEYS = list(stats(bodies[0]))

def collapse(key, nbin=12):
    curves = []
    for b in bodies:
        c = stats(b)[key]
        pts = [(Dm/c, depth_of(b["rows"], Dm)) for Dm in DEMANDS]
        curves.append([(x, d) for x, d in pts if d is not None])
    allx = np.array([x for cu in curves for x, _ in cu])
    if len(allx) < 20 or allx.max() <= allx.min(): return float("nan"), 0
    edges = np.quantile(allx, np.linspace(0, 1, nbin+1))
    sds, ns = [], 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        vals = [d for cu in curves for x, d in cu if lo <= x <= hi]
        if len(vals) >= 6: sds.append(np.std(vals)); ns += 1
    return (float(np.mean(sds)) if sds else float("nan")), ns

res = {}
for k in KEYS:
    m, nb = collapse(k); res[k] = dict(collapse=m, bins=nb)
json.dump(dict(demands=DEMANDS.tolist(), alphas=ALPHAS,
               vmax=[b["vmax"] for b in bodies], collapse=res),
          open(f"{OUT}/branchC.json", "w"), indent=1, default=float)

print("BRANCH C (corrected) — collapse of required_depth across 21 synthetic bodies")
print("  demand is ABSOLUTE (m/s); lower metric = better collapse\n")
print(f"  {'normalizer':<24}{'collapse SD':>13}{'bins':>7}")
for k, v in sorted(res.items(), key=lambda x: (np.isnan(x[1]['collapse']), x[1]['collapse'])):
    print(f"  {k:<24}{v['collapse']:13.4f}{v['bins']:7d}")
good = [(k, v['collapse']) for k, v in res.items() if not np.isnan(v['collapse'])]
good.sort(key=lambda x: x[1]); bestk, bestv = good[0]
nonev = res['none']['collapse']
print(f"\n  best normalizer : {bestk} at {bestv:.4f}")
print(f"  unnormalized    : {nonev:.4f}")
print(f"  reduction in spread vs unnormalized: {(nonev-bestv)/nonev*100:.1f}%")
