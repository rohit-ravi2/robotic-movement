"""Physics-lens verifier for M6_walking_headcarry: re-runs the claimed vectors on FRESH plants (uid 2, spawned processes, never
used by the builder) at 1 ms and 0.5 ms with 4 NEW perturbations V1-V4 (distinct seeds/values from P1-P4, DRAWS and SEARCH_DRAW),
the law-off ablation, and an independent trace check (load vs head patch, tilt, non-foot / load-floor contacts, torques vs rating,
support) for the headline pot 20 kg run.
  python verify_m6_physics.py runs [workers]     -> verify_physics/runs.json
  python verify_m6_physics.py trace              -> verify_physics/trace_pot20.json
"""
import os, sys, json, time
import numpy as np, mujoco
from engine import core, walk_headcarry as WH

OUT = f"{WH.RES}/verify_physics"
NC = WH.N_TRANS + WH.N_PER + 2
VPERT = {   # verifier-chosen, written before the runs; never seen by any search
    "V1": dict(noise="start", seed=5501, sd=0.02, vx=-0.04),
    "V2": dict(mass_noise=(5502, 0.05), load_scale=1.08),
    "V3": dict(lag_plant=1.20*WH.LAG, friction=0.8),
    "V4": dict(shove=(0.08, -0.06, 5.6, 0.2), n_cyc=WH.N_TRANS + WH.N_PER + 8),
}
CASES = ["pot_20kg_v0.6_s1", "pot_15kg_v0.6_s1", "pot_10kg_v0.6_s1_rob", "pot_5kg_v0.6_s1_rob", "box_20kg_v0.6_s1", "box_15kg_v0.6_s1", "box_5kg_v0.6_s1"]


def vkw(name, nT):
    if name == "nominal": return {}
    kw = dict(VPERT[name])
    if kw.get("noise") == "start":
        rng = np.random.default_rng(kw.pop("seed")); sd = kw.pop("sd"); kw["noise"] = dict(q=rng.normal(0, sd, nT), vx=kw.pop("vx"))
    return kw


def job(a):
    case, name, dt, law = a
    g = json.load(open(f"{WH.RES}/search/{case}.json"))
    C = WH.Carry(g["obj"], g["mass"], g["v"], dt_list=(dt,), uid=2, law_base=g["law_base"])
    t0 = time.time(); r = C.run(g["x"], dt, law=law, **vkw(name, len(C.S[dt]["types"]))); o = WH.summ(r); o["secs"] = round(time.time() - t0, 1)
    return case, name, dt, law, o


def runs(workers=3):
    import multiprocessing as mp
    os.makedirs(OUT, exist_ok=True)
    for c in CASES:
        g = json.load(open(f"{WH.RES}/search/{c}.json")); WH.Carry(g["obj"], g["mass"], g["v"], law_base=g["law_base"])   # scaffold cache
    jobs = [(c, nm, dt, True) for c in CASES for nm in ["nominal"] + list(VPERT) for dt in (core.DT_OPT, core.DT_CHECK)]
    jobs += [(c, "nominal", dt, False) for c in CASES[:1] + CASES[4:5] for dt in (core.DT_OPT, core.DT_CHECK)]
    out = {}
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        for case, nm, dt, law, o in pool.imap_unordered(job, jobs):
            out.setdefault(case, {}).setdefault("law" if law else "law_off", {}).setdefault(nm, {})[str(dt)] = o
            print(case, nm, dt, law, o["valid"], o["reasons"], o["value"], o["load"]["t_held"], o["load"]["e_max"], flush=True)
            json.dump(dict(vpert={k: str(v) for k, v in VPERT.items()}, out=out), open(f"{OUT}/runs.json", "w"), indent=1, default=float)
    summ = {}
    for c in CASES:
        L = out[c]["law"]; ok = {nm: all(L[nm][k]["valid"] for k in L[nm]) and len(L[nm]) == 2 for nm in L}
        n = sum(ok[nm] for nm in VPERT); summ[c] = dict(nominal_both_dt=ok["nominal"], v_pass=n, robust=bool(ok["nominal"] and n >= 3), ok=ok)
    json.dump(dict(vpert={k: str(v) for k, v in VPERT.items()}, summary=summ, out=out), open(f"{OUT}/runs.json", "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1))


def trace():
    """Independent gate recomputation from the recorded qpos (every 4 physics steps) of the headline run at both dt."""
    res = {}
    g = json.load(open(f"{WH.RES}/search/pot_20kg_v0.6_s1.json"))
    for dt in (core.DT_OPT, core.DT_CHECK):
        C = WH.Carry(g["obj"], g["mass"], g["v"], dt_list=(dt,), uid=3, law_base=g["law_base"]); S = C.S[dt]; m = S["m"]
        r = C.run(g["x"], dt, record=True); tr = r["trace"]; Lp = C.law_setup(S, WH.law_params(C.split(g["x"])[1], C.law_base)[0])
        d = mujoco.MjData(m); bid = Lp["bid"]; tb = Lp["tb"]; fl = S["floor"]; feetg = set(S["foot_geoms"]); rob = set(S["robot_bodies"])
        e = []; tilt = []; z_rel = []; nonfoot = 0; load_floor = 0; load_contacts = set(); eq = int(m.neq); weld_any = eq
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
        ld = np.asarray(tr["load"]); t = np.asarray(tr["t"]); com = np.asarray(tr["com"])
        wp = [i for i, x in enumerate(types) if x == "waist_pitch_joint"][0]
        i0 = int(np.searchsorted(t, 2.0))
        res[str(dt)] = dict(valid=r["valid"], reasons=r["reasons"], v_net=r["value"], dist=r["dist"], t_run=r["t_run"], slip_max=r["slip_max"], flight_max=r["flight_max"], ds_min=r["ds_min"],
                            n_eq_constraints=eq, e_max_trace=max(e), z_rel_min=min(z_rel), tilt_max_trace=max(tilt), nonfoot_floor_contacts=nonfoot, load_floor_contacts=load_floor,
                            load_touches=sorted(x for x in load_contacts if x),
                            util_max=float(util.max()), util_max_joint=types[int(np.argmax(util.max(0)))], waist_pitch_p95=float(np.percentile(util[:, wp], 95)),
                            util_p95_by_joint={types[i]: round(float(np.percentile(util[:, i], 95)), 3) for i in range(len(types))},
                            ratings={types[i]: float(tau[i]) for i in range(len(types))}, held=WH.ratings_table(S),
                            min_support_sum=float(ld.sum(1).min()), frac_samples_no_support=float(np.mean(ld.sum(1) < 0.02)),
                            com_dx_2s_to_end=float(com[-1, 0] - com[i0, 0]), v_mean_2s_to_end=float((com[-1, 0] - com[i0, 0])/(t[-1] - t[i0])),
                            load_mass=float(m.body_mass[bid]), head_frac=r["load"]["head_frac"])
        print(json.dumps({k: v for k, v in res[str(dt)].items() if k not in ("util_p95_by_joint", "ratings", "held")}, default=float), flush=True)
    os.makedirs(OUT, exist_ok=True); json.dump(res, open(f"{OUT}/trace_pot20.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    if sys.argv[1] == "runs": runs(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif sys.argv[1] == "trace": trace()
