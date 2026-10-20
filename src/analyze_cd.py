"""Branch C (normalization discovery), Branch D (morphology continuum), F.1 (triviality)."""
import sys, os, json, pickle, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, bank as BK, robot_bound as R

G = 9.80665; T = 0.25
D = pickle.load(open(os.environ.get("MLAW_BANK", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/pose_bank.pkl")), "rb"))
OUT = os.environ.get("MLAW_OUT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results"))
g1, h1 = D["G1"], D["H1"]

# ---- synthetic morphology continuum on G1 geometry: interpolate capacities and masses
#      toward H1's per-group values. These are SYNTHETIC bodies with real endpoints.
def h1_group_cap(g):
    gi = h1["group_index"]
    if g not in gi: return None
    n = gi[g][0]; return tuple(h1["caps"][n])

def body_at(alpha, extrapolate=False):
    """alpha=0 -> G1 capacities and mass; alpha=1 -> H1 per-group capacities and total mass."""
    caps = {}
    for g, names in g1["group_index"].items():
        src = tuple(g1["caps"][names[0]])
        dst = h1_group_cap(g) or src          # G1's waist has no H1 counterpart: hold fixed
        tau = src[0] + alpha*(dst[0]-src[0])
        om = src[1] + alpha*(dst[1]-src[1])
        for n in names: caps[n] = (max(tau, 1e-6), max(om, 1e-6))
    m = np.array(g1["mass"])
    scale = 1.0 + alpha*(float(np.sum(h1["mass"]))/float(np.sum(m)) - 1.0)
    return caps, m*scale

ALPHAS = list(np.round(np.linspace(-0.5, 1.5, 21), 3))
bodies = []
for a in ALPHAS:
    caps, mass = body_at(a)
    rows = BK.frontier_from_bank(g1["bank"], g1["groups"], caps, mass, T)
    vmax = max(r["v_takeoff"] for r in rows)
    bodies.append(dict(alpha=float(a), extrapolated=bool(a < 0 or a > 1), vmax=vmax,
                       caps={k: list(v) for k, v in caps.items()}, mass_total=float(np.sum(mass)),
                       rows=[dict(subset=list(r["subset"]), v=r["v_takeoff"]) for r in rows]))

def depth_of(rows, Dm):
    ok = [r for r in rows if r["v"] >= Dm]
    return min((len(r["subset"]) for r in ok), default=None)

def minsub(rows, Dm):
    ok = [r for r in rows if r["v"] >= Dm]
    if not ok: return None
    k = min(len(r["subset"]) for r in ok)
    return tuple(sorted(max([r for r in ok if len(r["subset"]) == k], key=lambda r: r["v"])["subset"]))

# ---------------- BRANCH C : which normalization collapses the curves? ----------------
def caps_stats(b):
    caps = b["caps"]; gi = g1["group_index"]
    tot_tau = sum(caps[n][0] for n in caps)
    tot_pow = sum(caps[n][0]*caps[n][1]/4.0 for n in caps)
    dist = gi["ankles"] + gi["knees"]          # "distal" leg groups for a jump
    d_tau = sum(caps[n][0] for n in dist)
    d_pow = sum(caps[n][0]*caps[n][1]/4.0 for n in dist)
    d_work = sum(caps[n][0]*2.0 for n in dist)     # tau * nominal 2 rad sweep
    M = b["mass_total"]; L = 0.8
    return dict(total_power=tot_pow, total_torque=tot_tau, distal_power=d_pow,
                distal_torque=d_tau, distal_work=d_work, mass=M, mgh=M*G*L,
                froude=M*np.sqrt(G*L), power_per_mass=tot_pow/M,
                distal_power_per_mass=d_pow/M, sqrt_pow_over_m=np.sqrt(tot_pow/M))

CANDS = ["total_power", "total_torque", "distal_power", "distal_torque", "distal_work",
         "mass", "mgh", "froude", "power_per_mass", "distal_power_per_mass",
         "sqrt_pow_over_m", "none"]

def collapse_metric(key, use=None):
    """Spread of required_depth across bodies at matched normalized demand."""
    use = use or bodies
    curves = []
    for b in use:
        st = caps_stats(b); c = 1.0 if key == "none" else st[key]
        pts = []
        for frac in np.linspace(0.1, 0.95, 18):
            Dm = frac*b["vmax"]
            d_ = depth_of(b["rows"], Dm)
            if d_ is not None: pts.append((Dm/c, d_))
        curves.append(pts)
    # residual spread: bin by normalized demand, take SD of depth within bin
    allx = np.array([x for c in curves for x, _ in c])
    if allx.max() <= allx.min(): return float("nan")
    bins = np.quantile(allx, np.linspace(0, 1, 13))
    sds = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        vals = [d for c in curves for x, d in c if lo <= x <= hi]
        if len(vals) >= 4: sds.append(np.std(vals))
    return float(np.mean(sds)) if sds else float("nan")

C = {k: collapse_metric(k) for k in CANDS}
C_interp = {k: collapse_metric(k, [b for b in bodies if not b["extrapolated"]]) for k in CANDS}

# ---------------- BRANCH D : depth map, bifurcations, sensitivity ----------------
fracs = np.linspace(0.1, 0.95, 18)
depth_map = []; ident_map = []
for b in bodies:
    depth_map.append([depth_of(b["rows"], f*b["vmax"]) for f in fracs])
    ident_map.append([minsub(b["rows"], f*b["vmax"]) for f in fracs])
bif = []
for j in range(len(fracs)):
    for i in range(1, len(bodies)):
        a, bb = ident_map[i-1][j], ident_map[i][j]
        da, db = depth_map[i-1][j], depth_map[i][j]
        if a and bb and a != bb and da == db:
            bif.append(dict(alpha_from=bodies[i-1]["alpha"], alpha_to=bodies[i]["alpha"],
                            frac=float(fracs[j]), depth=da,
                            subset_from=list(a), subset_to=list(bb)))
# D.4 sensitivity field: d(vmax)/d(log tau_g) and d(log omega_g)
sens = {}
for g in g1["group_index"]:
    row = {}
    for mode in ("tau", "omega"):
        vs = []
        for s in (0.8, 1.25):
            sc = {g: (s, 1.0) if mode == "tau" else (1.0, s)}
            caps = BK.caps_from_scaled(g1, sc)
            rr = BK.frontier_from_bank(g1["bank"], g1["groups"], caps, np.array(g1["mass"]), T)
            vs.append(max(x["v_takeoff"] for x in rr))
        row[mode] = float((vs[1]-vs[0])/(np.log(1.25)-np.log(0.8)))
    sens[g] = row

# ---------------- F.1 : does capacity-normalized beat naive predictors? ----------------
rowsF = []
for key in ["distal_power", "total_power", "power_per_mass", "mass", "total_torque", "none"]:
    rowsF.append(dict(normalizer=key, collapse_all=C[key], collapse_interp=C_interp[key]))

json.dump(dict(bodies=[{k: v for k, v in b.items() if k != "rows"} for b in bodies],
               collapse_all=C, collapse_interp=C_interp,
               fracs=fracs.tolist(), depth_map=depth_map,
               ident_map=[[list(x) if x else None for x in row] for row in ident_map],
               bifurcations=bif, sensitivity=sens, F1=rowsF),
          open(f"{OUT}/analysis_CD.json", "w"), indent=1, default=float)
print("wrote analysis_CD.json")
