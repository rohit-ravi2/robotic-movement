"""Physics-lens verifier, repair round 1 of M6_walking_headcarry (0.75 m/s rungs).
FRESH plants uid 4 (never used by builder: uid 1 = P validation, uid 2 = V set, uid 3 = round-0 trace), spawned processes, 1 ms and
0.5 ms, with 4 NEW perturbations W1-W4 written before any run (distinct seeds/values/directions from P1-P4, V1-V4, DRAWS, SEARCH_DRAW),
the law-off ablation, and an independent trace check of the headline 15 kg run (gates recomputed from recorded qpos via mj_forward).
  python verify_m6_physics_r1.py runs [workers]   -> verify_physics/r1_runs.json
  python verify_m6_physics_r1.py trace            -> verify_physics/r1_trace_pot15_v075.json
"""
import os, sys, json, time
import numpy as np, mujoco
from engine import core, walk_headcarry as WH

OUT = f"{WH.RES}/verify_physics"
UID = 4
WPERT = {   # verifier round 1, fixed before the runs
    "W1": dict(noise="start", seed=7701, sd=0.03, vx=0.06),                      # larger joint noise, faster start than P1 (+0.04) / V1 (-0.04)
    "W2": dict(mass_noise=(7702, 0.07), load_scale=0.90),                        # robot mass sd 7 %; true load 10 % LIGHTER than the controller assumes
    "W3": dict(lag_plant=1.30*WH.LAG, friction=0.75),                            # 13 ms plant lag (controller assumes 10 ms), mu 0.75
    "W4": dict(shove=(-0.08, 0.06, 6.4, 0.2), n_cyc=WH.N_TRANS + WH.N_PER + 8),  # 0.1 (M+m) g backwards-left diagonal, cycle 6.4
}
CASES = ["pot_15kg_v0.75_s1_rFrob", "pot_20kg_v0.75_s1_rFrob"]


def wkw(name, nT):
    if name == "nominal": return {}
    kw = dict(WPERT[name])
    if kw.get("noise") == "start":
        rng = np.random.default_rng(kw.pop("seed")); sd = kw.pop("sd"); kw["noise"] = dict(q=rng.normal(0, sd, nT), vx=kw.pop("vx"))
    return kw


def job(a):
    case, name, dt, law = a
    g = json.load(open(f"{WH.RES}/search/{case}.json"))
    C = WH.Carry(g["obj"], g["mass"], g["v"], dt_list=(dt,), uid=UID, law_base=g["law_base"])
    t0 = time.time(); r = C.run(g["x"], dt, law=law, **wkw(name, len(C.S[dt]["types"]))); o = WH.summ(r); o["secs"] = round(time.time() - t0, 1)
    o["slip_log"] = r.get("slip_log")
    return case, name, dt, law, o


def runs(workers=3):
    import multiprocessing as mp
    os.makedirs(OUT, exist_ok=True)
    jobs = [(c, nm, dt, True) for c in CASES for nm in ["nominal"] + list(WPERT) for dt in (core.DT_OPT, core.DT_CHECK)]
    jobs += [(CASES[0], "nominal", dt, False) for dt in (core.DT_OPT, core.DT_CHECK)]
    out = {}
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        for case, nm, dt, law, o in pool.imap_unordered(job, jobs):
            out.setdefault(case, {}).setdefault("law" if law else "law_off", {}).setdefault(nm, {})[str(dt)] = o
            print(case, nm, dt, law, o["valid"], o["reasons"], o.get("value"), o.get("dist"), o.get("slip_max"), o["load"], o["secs"], flush=True)
            json.dump(dict(wpert={k: str(v) for k, v in WPERT.items()}, uid=UID, out=out), open(f"{OUT}/r1_runs.json", "w"), indent=1, default=float)
    summ = {}
    for c in CASES:
        L = out[c]["law"]; ok = {nm: all(L[nm][k]["valid"] for k in L[nm]) and len(L[nm]) == 2 for nm in L}
        n = sum(ok[nm] for nm in WPERT); summ[c] = dict(nominal_both_dt=ok["nominal"], w_pass=n, robust=bool(ok["nominal"] and n >= 3), ok=ok)
    json.dump(dict(wpert={k: str(v) for k, v in WPERT.items()}, uid=UID, summary=summ, out=out), open(f"{OUT}/r1_runs.json", "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1))


def trace(case="pot_15kg_v0.75_s1_rFrob"):
    res = {}
    g = json.load(open(f"{WH.RES}/search/{case}.json"))
    for dt in (core.DT_OPT, core.DT_CHECK):
        C = WH.Carry(g["obj"], g["mass"], g["v"], dt_list=(dt,), uid=UID + 1, law_base=g["law_base"]); S = C.S[dt]; m = S["m"]
        r = C.run(g["x"], dt, record=True); tr = r["trace"]; Lp = C.law_setup(S, WH.law_params(C.split(g["x"])[1], C.law_base)[0])
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
        # coarse independent slip bound: foot-body origin xy travel over each loaded (> 30 % body weight) stance after 1 s
        # (includes heel-toe rolling of the origin, so it is an UPPER bound on sole sliding)
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
        i0 = int(np.searchsorted(t, 2.0))
        res[str(dt)] = dict(valid=r["valid"], reasons=r["reasons"], v_net=r["value"], dist=r["dist"], t_run=r["t_run"], slip_max=r["slip_max"], slip_log=r.get("slip_log"),
                            flight_max=r["flight_max"], ds_min=r["ds_min"], n_eq_constraints=int(m.neq), e_max_trace=max(e), z_rel_min=min(z_rel), z_rel_max=max(z_rel),
                            tilt_max_trace=max(tilt), nonfoot_floor_contacts=nonfoot, load_floor_contacts=load_floor, load_touches=sorted(x for x in load_contacts if x),
                            util_max=float(util.max()), util_max_joint=types[int(np.argmax(util.max(0)))], waist_pitch_p95=float(np.percentile(util[:, wp], 95)),
                            util_p95_by_joint={types[i]: round(float(np.percentile(util[:, i], 95)), 3) for i in range(len(types))},
                            frac_over_rating=float(np.mean(util > 1.0 + 1e-6)), ratings={types[i]: float(tau[i]) for i in range(len(types))}, held=WH.ratings_table(S),
                            min_support_sum=float(ld.sum(1).min()), n_samples_no_support_after_0p5s=int(both_off.sum()),
                            stance_origin_travel_max=max(travel), stance_origin_travel_p95=float(np.percentile(travel, 95)), n_stances=len(travel),
                            com_dx_2s_to_end=float(com[-1, 0] - com[i0, 0]), v_mean_2s_to_end=float((com[-1, 0] - com[i0, 0])/(t[-1] - t[i0])),
                            com_y_drift=float(com[-1, 1] - com[i0, 1]), load_mass=float(m.body_mass[bid]), load_info=r["load"], lag=WH.LAG)
        print(json.dumps({k: v for k, v in res[str(dt)].items() if k not in ("util_p95_by_joint", "ratings", "held")}, default=float), flush=True)
    os.makedirs(OUT, exist_ok=True); json.dump(res, open(f"{OUT}/r1_trace_pot15_v075.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    if sys.argv[1] == "runs": runs(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif sys.argv[1] == "trace": trace()
