"""Branches B, C, D, E, F. All run off the pose bank, so all are seconds-scale."""
import sys, os, json, pickle, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, bank as BK, robot_bound as R

G = 9.80665; T = 0.25
D = pickle.load(open(os.environ.get("MLAW_BANK", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/pose_bank.pkl")), "rb"))
OUT = os.environ.get("MLAW_OUT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results"))

def front(lab, caps=None, mass=None):
    d = D[lab]
    return BK.frontier_from_bank(d["bank"], d["groups"],
                                 caps or {k: tuple(v) for k, v in d["caps"].items()},
                                 np.array(mass if mass is not None else d["mass"]), T)

def rdepth(rows, ds):
    return R.required_depth(rows, ds)

def depth_at(rows, Dm):
    ok = [r for r in rows if r["v_takeoff"] >= Dm]
    return min((len(r["subset"]) for r in ok), default=None)

res = {}

# ---------------- BRANCH B ----------------
B = {}
for lab in ("G1", "H1"):
    rows = front(lab); vmax = max(r["v_takeoff"] for r in rows)
    # B.1 bisect every transition to 0.01 m/s
    coarse = np.arange(0.25, vmax, 0.25)
    trans = []
    prev = depth_at(rows, coarse[0])
    for a, b in zip(coarse[:-1], coarse[1:]):
        da, db = depth_at(rows, a), depth_at(rows, b)
        if da != db and db is not None:
            lo, hi = a, b
            for _ in range(40):
                mid = 0.5*(lo+hi)
                if depth_at(rows, mid) == da: lo = mid
                else: hi = mid
            trans.append(dict(from_depth=da, to_depth=db, at=0.5*(lo+hi), width=hi-lo))
    # B.2 shape
    plateau = trans[0]["at"] if trans else vmax
    esc = (trans[-1]["at"] - trans[0]["at"]) if len(trans) > 1 else 0.0
    wall = vmax - (trans[-1]["at"] if trans else 0.0)
    best, marg = R.best_by_k(rows)
    B[lab] = dict(vmax=vmax, transitions=trans, plateau_width=plateau,
                  escalation_span=esc, wall_gap=wall,
                  per_k={k: dict(subset=list(best[k]["subset"]), v=best[k]["v_takeoff"],
                                 dvdk=marg[k]) for k in sorted(best)})
res["B"] = B

# ---------------- BRANCH E : capacity ablations ----------------
E = {}
SC = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
for lab in ("G1", "H1"):
    d = D[lab]; base = front(lab)
    vmax0 = max(r["v_takeoff"] for r in base)
    probe = [0.4*vmax0, 0.6*vmax0, 0.8*vmax0]
    per_group = {}
    for g in d["groups"]:
        curve = []
        for s in SC:
            for mode, sc in (("both", (s, s)), ("torque", (s, 1.0)), ("speed", (1.0, s))):
                caps = BK.caps_from_scaled(d, {g: sc})
                rows = front(lab, caps=caps)
                curve.append(dict(scale=s, mode=mode,
                                  vmax=max(r["v_takeoff"] for r in rows),
                                  depth=[depth_at(rows, p) for p in probe],
                                  in_min_subset=[
                                      (g in (R.required_depth(rows, [p])[0]["subset"] or []))
                                      for p in probe]))
        per_group[g] = curve
    # elasticity d(depth)/d(log capacity) at the middle probe, both-scaling
    el = {}
    for g, c in per_group.items():
        pts = [(np.log(x["scale"]), x["depth"][1]) for x in c if x["mode"] == "both"
               and x["depth"][1] is not None]
        if len(pts) >= 3:
            xs, ys = zip(*pts); el[g] = float(np.polyfit(xs, ys, 1)[0])
        else: el[g] = None
    E[lab] = dict(probe_demands=probe, per_group=per_group, elasticity=el)
res["E"] = E

# ---------------- BRANCH F : controls ----------------
F = {}
rng = np.random.default_rng(7)
for lab in ("G1", "H1"):
    d = D[lab]; groups = d["groups"]
    base_caps = {k: tuple(v) for k, v in d["caps"].items()}
    base_mass = np.array(d["mass"])
    rows0 = front(lab); vmax0 = max(r["v_takeoff"] for r in rows0)
    ds = np.linspace(0.15*vmax0, 0.95*vmax0, 12)
    truth = [depth_at(rows0, x) for x in ds]
    # F.2 uniform capacity, same total torque and mean speed
    tot_tau = sum(base_caps[n][0] for n in base_caps)
    mean_om = float(np.mean([base_caps[n][1] for n in base_caps]))
    uni = {n: (tot_tau/len(base_caps), mean_om) for n in base_caps}
    rows_u = front(lab, caps=uni)
    # F.3 scrambled: permute (tau,omega) across GROUPS, multiset fixed
    gi = d["group_index"]
    scram = []
    for trial in range(12):
        perm = list(groups); rng.shuffle(perm)
        cmap = {}
        for gsrc, gdst in zip(groups, perm):
            src_cap = base_caps[gi[gsrc][0]]
            for n in gi[gdst]: cmap[n] = src_cap
        for n in base_caps: cmap.setdefault(n, base_caps[n])
        r_ = front(lab, caps=cmap)
        scram.append([depth_at(r_, x) for x in ds])
    F[lab] = dict(demands=ds.tolist(), truth=truth,
                  uniform=dict(vmax=max(r["v_takeoff"] for r in rows_u),
                               depth=[depth_at(rows_u, x) for x in ds]),
                  scrambled=scram)
res["F"] = F
json.dump(res, open(f"{OUT}/analysis_BEF.json", "w"), indent=1, default=float)
print("wrote analysis_BEF.json")
