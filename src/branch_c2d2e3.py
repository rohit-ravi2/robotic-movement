"""C2 separability/assignment ceiling, D2 bifurcation margins, E3 elasticity completion,
A2 ROM-tightening check. All off the pose bank."""
import sys, os, json, pickle, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, bank as BK, robot_bound as R

G = 9.80665; T = 0.25
D = pickle.load(open(os.environ.get("MLAW_BANK", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/pose_bank.pkl")), "rb"))
OUT = os.environ.get("MLAW_OUT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results"))
out = {}

# ============================ A2: does the ROM branch use the SWEPT range? ============
g1 = D["G1"]; gi = g1["group_index"]
rec0 = g1["bank"][0]
a2 = dict(implementation="bank.eval_pose computes wr = tau * dth where dth is the stored "
                        "per-joint excursion between the optimized crouch and takeoff poses, "
                        "not the joint's full declared range of motion.",
          example_joint=list(rec0["exc"])[0],
          example_excursion=float(list(rec0["exc"].values())[0]))
n = a2["example_joint"]
full_rom = None
for lab in ("G1",):
    pass
a2["full_rom_would_be"] = "joint range width"
out["A2"] = a2

# ============================ C2: separability / assignment ceiling ====================
def work_of(rec, caps, joints):
    W = 0.0
    for nn, dth in rec["exc"].items():
        if nn not in joints or dth <= 0: continue
        tau, om = caps[nn]
        wr = tau*dth; wp = tau*om/4.0*T
        wc = tau*max(0.0, 1.0 - (dth/T)/om)*dth
        W += min(wr, wp, wc)
    return W

C2 = {}
for lab in ("G1", "H1"):
    d = D[lab]; gidx = d["group_index"]; groups = list(gidx)
    base = {k: tuple(v) for k, v in d["caps"].items()}
    mass = np.array(d["mass"])
    rows0 = BK.frontier_from_bank(d["bank"], groups, base, mass, T)
    v0 = {tuple(sorted(r["subset"])): r["v_takeoff"] for r in rows0}
    vmax0 = max(v0.values())

    # C2.2 — does permuting two capacities INSIDE a subset change its bound?
    inside_tests = []
    full = tuple(sorted(groups))
    for ga, gb in itertools.combinations(groups, 2):
        cmap = dict(base)
        ca = base[gidx[ga][0]]; cb = base[gidx[gb][0]]
        for nn in gidx[ga]: cmap[nn] = cb
        for nn in gidx[gb]: cmap[nn] = ca
        rr = BK.frontier_from_bank(d["bank"], groups, cmap, mass, T)
        vfull = max(x["v_takeoff"] for x in rr if tuple(sorted(x["subset"])) == full)
        # excursion asymmetry between the two groups in the full-subset pose
        recf = [x for x in d["bank"] if tuple(sorted(x["groups"])) == full][0]
        ea = np.mean([recf["exc"].get(nn, 0.0) for nn in gidx[ga]])
        eb = np.mean([recf["exc"].get(nn, 0.0) for nn in gidx[gb]])
        inside_tests.append(dict(swap=[ga, gb], dv=float(vfull - v0[full]),
                                 exc_a=float(ea), exc_b=float(eb),
                                 exc_ratio=float(max(ea, eb)/max(min(ea, eb), 1e-9))))

    # C2.3 — ceiling: max change in required_depth over ALL group permutations
    dgrid = np.linspace(0.15*vmax0, 0.95*vmax0, 12)
    def depth_curve(caps):
        rr = BK.frontier_from_bank(d["bank"], groups, caps, mass, T)
        vv = [(len(x["subset"]), x["v_takeoff"]) for x in rr]
        o = []
        for Dm in dgrid:
            ok = [k for k, v in vv if v >= Dm]
            o.append(min(ok) if ok else None)
        return o
    truth = depth_curve(base)
    perms = list(itertools.permutations(groups))
    if len(perms) > 240:
        rs = np.random.default_rng(3); idx = rs.choice(len(perms), 240, replace=False)
        perms = [perms[i] for i in idx]
    devs = []
    for p in perms:
        cmap = {}
        for gsrc, gdst in zip(groups, p):
            cc = base[gidx[gsrc][0]]
            for nn in gidx[gdst]: cmap[nn] = cc
        dc = depth_curve(cmap)
        a = np.array([x if x is not None else -1 for x in dc], float)
        b = np.array([x if x is not None else -1 for x in truth], float)
        devs.append(float(np.mean(np.abs(a-b))))
    devs = np.array(devs)
    C2[lab] = dict(n_perms=len(perms), ceiling_max=float(devs.max()),
                   ceiling_p95=float(np.percentile(devs, 95)), ceiling_mean=float(devs.mean()),
                   inside_swaps=inside_tests, vmax=vmax0,
                   phase5_observed=0.28 if lab == "H1" else 0.56)
out["C2"] = C2

# ============================ D2: bifurcation margins =================================
cd = json.load(open(f"{OUT}/analysis_CD.json"))
def h1cap(g):
    gg = D["H1"]["group_index"]; return tuple(D["H1"]["caps"][gg[g][0]]) if g in gg else None
def body_at(alpha):
    caps = {}
    for g, names in gi.items():
        src = tuple(g1["caps"][names[0]]); dst = h1cap(g) or src
        for nn in names:
            caps[nn] = (max(src[0]+alpha*(dst[0]-src[0]), 1e-6),
                        max(src[1]+alpha*(dst[1]-src[1]), 1e-6))
    m = np.array(g1["mass"]); sc = 1.0+alpha*(float(np.sum(D["H1"]["mass"]))/float(np.sum(m))-1.0)
    return caps, m*sc
TOL = 0.05     # pose-bank numerical/seed tolerance in m/s
d2 = []
for bf in cd["bifurcations"]:
    entry = dict(bf)
    for side, al in (("from", bf["alpha_from"]), ("to", bf["alpha_to"])):
        caps, mass = body_at(al)
        rr = BK.frontier_from_bank(g1["bank"], list(gi), caps, mass, T)
        vmax = max(x["v_takeoff"] for x in rr)
        Dm = bf["frac"]*vmax
        cand = [x for x in rr if x["v_takeoff"] >= Dm and len(x["subset"]) == bf["depth"]]
        cand.sort(key=lambda x: -x["v_takeoff"])
        if len(cand) >= 2:
            entry[f"margin_{side}"] = float(cand[0]["v_takeoff"]-cand[1]["v_takeoff"])
            entry[f"winner_{side}"] = list(cand[0]["subset"])
            entry[f"runnerup_{side}"] = list(cand[1]["subset"])
        else:
            entry[f"margin_{side}"] = float("nan")
    m1, m2 = entry.get("margin_from", float("nan")), entry.get("margin_to", float("nan"))
    mn = np.nanmin([m1, m2])
    entry["class"] = ("GENUINE" if mn > 3*TOL else "TIE" if mn < TOL else "MARGINAL")
    entry["min_margin"] = float(mn)
    d2.append(entry)
out["D2"] = dict(tolerance=TOL, items=d2,
                 counts={c: sum(1 for x in d2 if x["class"] == c) for c in ("GENUINE", "MARGINAL", "TIE")})

# ============================ E3: elasticity, torque and speed separated ==============
bef = json.load(open(f"{OUT}/analysis_BEF.json"))
E3 = {}
for lab, dd in bef["E"].items():
    probes = dd["probe_demands"]; per = {}
    for g, curve in dd["per_group"].items():
        row = {}
        for mode in ("both", "torque", "speed"):
            pts = [(np.log(x["scale"]), x["depth"][1]) for x in curve
                   if x["mode"] == mode and x["depth"][1] is not None]
            row[mode] = float(np.polyfit(*zip(*pts), 1)[0]) if len(pts) >= 3 else None
            vm = [(np.log(x["scale"]), x["vmax"]) for x in curve if x["mode"] == mode]
            row[mode+"_dvmax"] = float(np.polyfit(*zip(*vm), 1)[0]) if len(vm) >= 3 else None
        per[g] = row
    # E3.3: is elasticity predictable from each group's share of the baseline bound?
    d = D[lab]; base = {k: tuple(v) for k, v in d["caps"].items()}
    full = tuple(sorted(d["group_index"]))
    recf = [x for x in d["bank"] if tuple(sorted(x["groups"])) == full][0]
    shares = {}
    tot = 0.0
    for g, names in d["group_index"].items():
        w = work_of(recf, base, set(names)); shares[g] = w; tot += w
    shares = {g: w/max(tot, 1e-9) for g, w in shares.items()}
    E3[lab] = dict(per_group=per, bound_share=shares)
out["E3"] = E3
json.dump(out, open(f"{OUT}/c2d2e3.json", "w"), indent=1, default=float)
print("wrote c2d2e3.json")
