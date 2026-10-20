"""Physics-lens verifier, repair round 2 of M6_walking_headcarry (0.75 m/s; 20 kg rIens, 15 kg rFrob, 25 kg rIens).
FRESH plants uid 6 (runs) and uid 7 (trace), never used before (builder uid 1 P / uid 2 V; verifiers uid 3, 4, 5), spawned processes,
1 ms and 0.5 ms. Four NEW perturbations X1-X4 fixed here and committed before any run (seeds/values/directions distinct from P1-P4,
V1-V4, W1-W4, DRAWS d_*/e_*, SEARCH_DRAW).
  python verify_m6_physics_r2.py runs [workers]   -> verify_physics/r2_runs.json
  python verify_m6_physics_r2.py trace            -> verify_physics/r2_trace_pot20_rIens.json
"""
import os, sys, json, time
import numpy as np, mujoco
from engine import core, walk_headcarry as WH

OUT = f"{WH.RES}/verify_physics"
UID = 6
XPERT = {   # verifier round 2, fixed before the runs
    "X1": dict(noise="start", seed=8801, sd=0.025, vx=-0.025),                    # joint noise sd 0.025, slow start
    "X2": dict(mass_noise=(8802, 0.05), load_scale=1.05),                          # robot mass sd 5 %; true load 5 % HEAVIER than the model
    "X3": dict(lag_plant=1.25*WH.LAG, friction=0.9),                               # 12.5 ms plant lag, mu 0.9
    "X4": dict(shove=(0.06, 0.08, 8.1, 0.2), n_cyc=WH.N_TRANS + WH.N_PER + 8),    # 0.1 (M+m) g forward-left, cycle 8.1
}
SPLIT = {"L105": dict(load_scale=1.05), "L110": dict(load_scale=1.10), "M8802": dict(mass_noise=(8802, 0.05))}
C20, C15, C25 = "pot_20kg_v0.75_s2_rIens", "pot_15kg_v0.75_s1_rFrob", "pot_25kg_v0.75_s2_rIens"
SDRAWS = ["search_draw"] + list(WH.DRAWS)


def kw_for(name, nT):
    if name == "nominal": return {}
    if name in XPERT:
        kw = dict(XPERT[name])
        if kw.get("noise") == "start":
            rng = np.random.default_rng(kw.pop("seed")); sd = kw.pop("sd"); kw["noise"] = dict(q=rng.normal(0, sd, nT), vx=kw.pop("vx"))
        return kw
    if name in SPLIT: return dict(SPLIT[name])
    return WH.pert_kw(name, nT)   # P1-P4, DRAWS, search_draw exactly as the builder defines them


def job(a):
    case, name, dt, law = a
    g = json.load(open(f"{WH.RES}/search/{case}.json"))
    C = WH.Carry(g["obj"], g["mass"], g["v"], dt_list=(dt,), uid=UID, law_base=g["law_base"])
    t0 = time.time(); r = C.run(g["x"], dt, law=law, **kw_for(name, len(C.S[dt]["types"]))); o = WH.summ(r); o["secs"] = round(time.time() - t0, 1)
    o["slip_log"] = r.get("slip_log")
    return case, name, dt, law, o


def runs(workers=3):
    import multiprocessing as mp
    os.makedirs(OUT, exist_ok=True)
    B = (core.DT_OPT, core.DT_CHECK)
    jobs = [(C20, nm, dt, True) for nm in ["nominal"] + list(XPERT) + ["P1", "P2", "P3", "P4"] + list(SPLIT) for dt in B]
    jobs += [(C20, "nominal", dt, False) for dt in B]
    jobs += [(C15, nm, dt, True) for nm in ["nominal"] + list(XPERT) for dt in B]
    jobs += [(C20, nm, core.DT_OPT, True) for nm in SDRAWS]
    jobs += [(C25, nm, core.DT_OPT, True) for nm in ["nominal"] + SDRAWS]
    out = {}
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        for case, nm, dt, law, o in pool.imap_unordered(job, jobs):
            out.setdefault(case, {}).setdefault("law" if law else "law_off", {}).setdefault(nm, {})[str(dt)] = o
            print(case, nm, dt, law, o["valid"], o["reasons"], o.get("value"), o.get("dist"), o.get("slip_max"), o["load"], o["secs"], flush=True)
            json.dump(dict(xpert={k: str(v) for k, v in XPERT.items()}, uid=UID, out=out), open(f"{OUT}/r2_runs.json", "w"), indent=1, default=float)
    summ = {}
    for c, sets in ((C20, {"X": list(XPERT), "P": ["P1", "P2", "P3", "P4"]}), (C15, {"X": list(XPERT)})):
        L = out[c]["law"]; ok = {nm: len(L[nm]) == 2 and all(L[nm][k]["valid"] for k in L[nm]) for nm in L if len(L[nm]) == 2}
        summ[c] = dict(nominal_both_dt=ok["nominal"], ok=ok, **{f"{s}_pass": sum(ok[n] for n in ns) for s, ns in sets.items()})
    for c in (C20, C25):
        L = out[c]["law"]; summ[c + "_draws_1ms"] = {n: L[n][str(core.DT_OPT)]["valid"] for n in SDRAWS}
    json.dump(dict(xpert={k: str(v) for k, v in XPERT.items()}, split={k: str(v) for k, v in SPLIT.items()}, uid=UID, summary=summ, out=out),
              open(f"{OUT}/r2_runs.json", "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1))


def gate_trace(C, S, g, r):
    """Gates recomputed from recorded qpos via mj_forward (as round 1)."""
    m = S["m"]; tr = r["trace"]; Lp = C.law_setup(S, WH.law_params(C.split(g["x"])[1], C.law_base)[0])
    d = mujoco.MjData(m); bid = Lp["bid"]; tb = Lp["tb"]; fl = S["floor"]; feetg = set(S["foot_geoms"]); rob = set(S["robot_bodies"])
    e = []; tilt = []; z_rel = []; nonfoot = 0; load_floor = 0; load_contacts = set()
    for q in tr["qpos"]:
        d.qpos[:] = q; mujoco.mj_forward(m, d)
        Rt = d.xmat[tb].reshape(3, 3); ht = d.xpos[tb] + Rt @ Lp["h_loc"]; cl = d.xipos[bid]
        e.append(float(np.linalg.norm(cl[:2] - ht[:2]))); z_rel.append(float(cl[2] - ht[2])); tilt.append(float(np.degrees(np.arccos(np.clip(d.xmat[bid][8], -1, 1)))))
        for i in range(d.ncon):
            c = d.contact[i]; b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if fl in (c.geom1, c.geom2):
                oth = c.geom2 if c.geom1 == fl else c.geom1
                if m.geom_bodyid[oth] == bid: load_floor += 1
                elif oth not in feetg and m.geom_bodyid[oth] in rob and c.dist < 0: nonfoot += 1
            if bid in (b1, b2): load_contacts.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b2 if b1 == bid else b1)))
    tq = np.abs(np.asarray(tr["tq"])); tau = np.asarray(S["jarr"]["tau"]); util = tq/tau; types = S["types"]
    ld = np.asarray(tr["load"]); t = np.asarray(tr["t"]); com = np.asarray(tr["com"]); feet = np.asarray(tr["feet"])
    wp = types.index("waist_pitch_joint")
    travel = []
    for li in (0, 1):
        on = (ld[:, li] > 0.30) & (t > 1.0); i = 0
        while i < len(on):
            if on[i]:
                j = i
                while j + 1 < len(on) and on[j + 1]: j += 1
                P = feet[i:j + 1, 3*li:3*li + 2]; travel.append(float(np.linalg.norm(P.max(0) - P.min(0)))); i = j + 1
            else: i += 1
    both_off = (ld[:, 0] < 0.02) & (ld[:, 1] < 0.02) & (t > 0.5)
    i0 = int(np.searchsorted(t, 2.0)) if t[-1] > 2.5 else 0
    sat = lambda lo, hi: float(np.mean(util[(t >= lo) & (t < hi), wp] > 0.99)) if ((t >= lo) & (t < hi)).any() else None
    return dict(valid=r["valid"], reasons=r["reasons"], v_net=r["value"], dist=r["dist"], t_run=r["t_run"], slip_max=r["slip_max"], slip_log=r.get("slip_log"),
                flight_max=r["flight_max"], ds_min=r["ds_min"], n_eq_constraints=int(m.neq), e_max_trace=max(e), z_rel_min=min(z_rel), z_rel_max=max(z_rel),
                tilt_max_trace=max(tilt), nonfoot_floor_contacts=nonfoot, load_floor_contacts=load_floor, load_touches=sorted(x for x in load_contacts if x),
                util_max=float(util.max()), util_max_joint=types[int(np.argmax(util.max(0)))], waist_pitch_p95=float(np.percentile(util[:, wp], 95)),
                util_p95_by_joint={types[i]: round(float(np.percentile(util[:, i], 95)), 3) for i in range(len(types))},
                frac_over_rating=float(np.mean(util > 1.0 + 1e-6)), ratings={types[i]: float(tau[i]) for i in range(len(types))}, held=WH.ratings_table(S),
                min_support_sum_after_0p5s=float(ld[t > 0.5].sum(1).min()), n_samples_no_support_after_0p5s=int(both_off.sum()),
                stance_origin_travel_max=max(travel) if travel else None, n_stances=len(travel),
                com_dx_2s_to_end=float(com[-1, 0] - com[i0, 0]), v_mean_2s_to_end=float((com[-1, 0] - com[i0, 0])/max(t[-1] - t[i0], 1e-9)),
                com_y_drift=float(com[-1, 1] - com[i0, 1]), load_mass=float(m.body_mass[bid]), load_info=r["load"], lag=WH.LAG,
                wp_sat_0_0p9=sat(0, 0.9), wp_sat_0_1p2=sat(0, 1.2), wp_sat_steady_2p5_end=sat(2.5, 1e9), e_max_first_1p2s=max(np.asarray(e)[t < 1.2]))


def trace(case=C20):
    res = {}
    g = json.load(open(f"{WH.RES}/search/{case}.json"))
    for name, dt in (("nominal", core.DT_OPT), ("nominal", core.DT_CHECK), ("P2", core.DT_OPT)):
        C = WH.Carry(g["obj"], g["mass"], g["v"], dt_list=(dt,), uid=UID + 1, law_base=g["law_base"]); S = C.S[dt]
        r = C.run(g["x"], dt, record=True, **kw_for(name, len(S["types"])))
        res[f"{name}_{dt}"] = o = gate_trace(C, S, g, r)
        print(name, dt, json.dumps({k: v for k, v in o.items() if k not in ("util_p95_by_joint", "ratings", "held", "slip_log")}, default=float), flush=True)
    os.makedirs(OUT, exist_ok=True); json.dump(res, open(f"{OUT}/r2_trace_pot20_rIens.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    if sys.argv[1] == "runs": runs(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif sys.argv[1] == "trace": trace()
