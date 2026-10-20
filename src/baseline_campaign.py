"""North-star benchmark: robot-native optimisation vs retargeted human mocap, same plant.
Stage 1 (select): retarget quality check (sec. 71) + certified start projection (sec. 60).
Stage 2 (run): B0 (tracking, gains by fidelity), B1(lambda) refinement Pareto sweep, OURS
native from the SAME certified start and horizon, simple phased heuristic. Tasks T1 and T2."""
import sys, os, json, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, branch_i as BI, mocap_baseline as MB, contract_g1 as CG, lineage as LN
from multiprocessing import Pool

OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase8")
PKL = "retargeted_g1_all.pkl"
LAMS = [10.0, 1.0, 0.1, 0.0]
SEEDS = [1, 2, 3]; ITERS, POP = 100, 16     # identical for OURS and every B1; sized to measured cost (~0.2 s/rollout under shared load)


def quality(key):
    """Numerical retarget sanity (sec. 71) on the raw retargeted clip."""
    D = pickle.load(open(f"{MB.BASE}/{PKL}", "rb"))[key]
    Q = np.array(D["qpos"]); a = D["takeoff_frame"] - D["frame0"]
    S0 = BI.setup(); m = S0["m"]; d = mujoco.MjData(m)
    kn = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_knee_joint")] for s in ("left", "right")]
    seg = Q[a-MB.PRE:a+1]
    knee_range = float(np.max([np.ptp(seg[:, k]) for k in kn]))
    lim = 0
    for j in range(1, m.njnt):
        lo, hi = m.jnt_range[j]; v = Q[:, m.jnt_qposadr[j]]
        lim += int(((v < lo-1e-3) | (v > hi+1e-3)).sum())
    fb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link") for s in ("left", "right")]
    fx = []
    for i in (a-MB.PRE, a-1):
        d.qpos[:] = Q[i]; mujoco.mj_forward(m, d); fx.append([d.xpos[b][:2].copy() for b in fb])
    slip = float(max(np.linalg.norm(fx[1][k]-fx[0][k]) for k in range(2)))
    reasons = []
    if knee_range < 0.5: reasons.append(f"no knee countermovement (knee range {knee_range:.2f} rad)")
    if lim: reasons.append(f"{lim} joint-limit violations")
    if slip > 0.10: reasons.append(f"stance foot slip {slip:.2f} m before takeoff")
    return dict(knee_range=knee_range, limit_violations=lim, stance_slip=slip, reasons=reasons)


def select():
    D = pickle.load(open(f"{MB.BASE}/{PKL}", "rb"))
    rows = {}
    for key in D:
        q = quality(key)
        if q["reasons"]:
            rows[key] = dict(status="EXCLUDED_QUALITY", **q); continue
        c = MB.load_clip(key, PKL)
        cert = c["start_certificate"]
        if cert["status"] != "STATIC_FEASIBLE":
            rows[key] = dict(status="EXCLUDED_START_NOT_CERTIFIABLE", cert=cert["status"], **q); continue
        S = BI.setup(c["start"]); v, _ = CG.verify(S)
        if not v.summary():
            rows[key] = dict(status="EXCLUDED_START_NOT_HOLDABLE", checks={k: r["detail"] for k, r in v.results.items() if not r["passed"]}, **q); continue
        rows[key] = dict(status="SELECTED", lean_change=cert["lean_change"], torque_margin=cert["torque_margin"],
                         support_margin=cert["support_margin"], horizon=c["horizon"], nk=c["nk"],
                         ankle_offsets=c["ankle_offsets"], **q)
    json.dump(rows, open(f"{OUT}/baseline_selection.json", "w"), indent=1, default=float)
    for k, v in rows.items():
        print(f"  {k:<34} {v['status']:<32} " + (f"lean {v['lean_change']:+.2f} knee {v['knee_range']:.2f}" if v["status"] == "SELECTED" else str(v.get("reasons") or v.get("cert"))))
    return [k for k, v in rows.items() if v["status"] == "SELECTED"]


def job(a):
    c = MB.load_clip(a["clip"], PKL); S = BI.setup(c["start"]); BI.settle(S)
    t0 = time.time()
    if a["method"] == "B0":
        om, table = MB.tune_gains(S, c, a["tscale"])
        r = MB.run(S, c, np.zeros(len(S["types"])*c["nk"]), om, a["tscale"], record=True)
        r.update(omega=om, gain_table=table, tscale=a["tscale"])
    elif a["method"] == "B1":
        om, _ = MB.tune_gains(S, c, 1.0)
        r = MB.solve_b1(S, c, a["lam"], a["seed"], om, ITERS, POP, task=a["task"]); r["omega"] = om
    elif a["method"] == "OURS":
        r = MB.solve_b1(S, c, 0.0, a["seed"], 0.0, ITERS, POP, task=a["task"], native=True)
    r.pop("trace", None) if not a.get("keep_trace") else None
    return LN.stamp(dict(a, secs=time.time()-t0, settle_qdot=S["settle_qdot_max"], **r), plant_hash=CG.plant_hash(S))


if __name__ == "__main__":
    clips = select()
    # SENSITIVITY clip: passes certification and settle but drifts 10.6 mm (> 5 mm) in the 0.5 s
    # open-loop hold (0.28 mm at 0.1 s). Run and reported SEPARATELY; the PCTV-5 threshold was
    # NOT changed after seeing it fail.
    SENS = ["fightAndSports1_subject4@4684"]
    print("selected:", clips, "sensitivity:", SENS, flush=True)
    jobs = []
    for cl in clips + SENS:
        for ts in (1.0, 0.87): jobs.append(dict(clip=cl, method="B0", tscale=ts, task="T1", keep_trace=True))
        for task in ("T2", "T1"):
            for s in SEEDS:
                jobs.append(dict(clip=cl, method="OURS", seed=s, task=task, keep_trace=True))
                for lam in LAMS: jobs.append(dict(clip=cl, method="B1", lam=lam, seed=s, task=task, keep_trace=(lam in (0.0, 1.0))))
    for cl in clips:                                     # plant contract at every start pose
        c = MB.load_clip(cl, PKL); S = BI.setup(c["start"]); CG.verify_or_die(S)
    for j in jobs: j["set"] = "sensitivity" if j["clip"] in SENS else "primary"
    LN.manifest(f"{OUT}/baseline.manifest.json", question="Robot-native optimisation vs strongest credible retargeted-mocap baseline (GMR/LAFAN1) on G1, same plant, same start, same horizon, same optimizer budget",
                clips=clips, lambdas=LAMS, seeds=SEEDS, budget=[ITERS, POP],
                acceptance="valid TrajectoryValidityReport; compare OURS vs best B1 at matched budget; lambda=0 separates template vs optimizer")
    t0 = time.time()
    with Pool(14) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(dict(clips=clips, runs=res), open(f"{OUT}/baseline.json", "w"), default=float)
    print(f"baseline: {len(res)} runs in {time.time()-t0:.0f}s", flush=True)


def replay(dt=0.001, name="baseline"):
    """dt-converged MEASUREMENT of every stored benchmark control (P10): replay at 1 ms from the
    same certified start, same tracker (for B0/B1) or none (OURS)."""
    d = json.load(open(f"{OUT}/{name}.json"))
    def one(r):
        c = MB.load_clip(r["clip"], PKL); S = BI.setup(c["start"])
        if abs(S["m"].opt.timestep - dt) > 1e-12:
            S["m"].opt.timestep = dt; S["m"].opt.iterations = 50
            for k in ("q_settled", "v_settled", "a_hold"): S.pop(k, None)
        BI.settle(S)
        x = np.array(r["x"]) if "x" in r else np.zeros(len(S["types"])*c["nk"])
        om = r.get("omega", 0.0) if r["method"] != "OURS" else 0.0
        o = MB.run(S, c, x, om, r.get("tscale", 1.0))
        return dict(clip=r["clip"], method=r["method"], lam=r.get("lam"), seed=r.get("seed"), task=r.get("task"),
                    set=r.get("set"), v_4ms=r["v"], v=o["v"], valid=o["valid"], reasons=o["validity"]["reasons"],
                    deviation=o["deviation"], flight_rotation=o["mech"]["flight_rotation"])
    out = [one(r) for r in d["runs"]]
    json.dump(out, open(f"{OUT}/{name}_replay_dt{dt}.json", "w"), indent=1, default=float)
    return out


def budget_run(iters=400, pop=16, seeds=(1, 2, 3)):
    """Benchmark budget sensitivity: OURS vs mocap-seeded refinement (lambda=0) at 4x budget, T2."""
    global ITERS, POP
    ITERS, POP = iters, pop
    clips = [k for k, v in json.load(open(f"{OUT}/baseline_selection.json")).items() if v["status"] == "SELECTED"]
    jobs = []
    for cl in clips:
        for s in seeds:
            jobs.append(dict(clip=cl, method="OURS", seed=s, task="T2", keep_trace=False, set="primary", budget=iters))
            jobs.append(dict(clip=cl, method="B1", lam=0.0, seed=s, task="T2", keep_trace=False, set="primary", budget=iters))
    with Pool(14) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(dict(clips=clips, runs=res), open(f"{OUT}/baseline_budget{iters}.json", "w"), default=float)
