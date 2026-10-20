"""Phase 13: is the analytic-prior (deep-crouch) G1 jump near-optimal and deployable?
A ladder (4x, 16x budget); B hardware stop margin; C margin + optimised flight/landing controller (APRIOR vs
NATIVE, optimised at 2 ms, valid at 1 AND 0.5 ms); D squat (held deep crouch) vs countermovement (from standing)."""
import sys, os, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cma, branch_i as BI, mocap_baseline as MB, aprior_standard as AS, contract_g1 as CG
from multiprocessing import Pool
OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase13")
MARGIN = 0.10; H_LAND = 2.6; KNOT_DT = 0.05

def crouch_start():
    """The analytic bank's full-set crouch as a HELD start (squat jump), clipped inside the joint ranges."""
    c = AS.clip0(); ref = MB.analytic_clip(c)
    i = int(np.argmin(np.linalg.norm(ref["Q"] - ref["Q"][int(0.55*ref["fps"])], axis=1)))
    D = pickle.load(open(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/pose_bank_certified_v2.pkl"), "rb"))["G1"]
    rec = max(D["bank"], key=lambda r: len(r["groups"])); S0 = BI.setup(); rb = S0["rb"]; q = np.array(c["start"])
    for t, v in rec["t0"].items():
        for n in rb.groups_of_type(t):
            r_ = rb.J[n]; sg = -1.0 if (rb.mirror[t] and r_["side_geom"] == "right") else 1.0
            q[r_["qadr"]] = float(np.clip(sg*v, r_["lo"] + 0.02, r_["hi"] - 0.02))
    return q

def landing_from(z, S):
    J = S["jarr"]; nT = len(S["types"]); delta = 0.6*np.asarray(z[:nT]); om = 10.0*6.0**((z[nT] + 1)/2); zeta = 0.5 + 0.75*(z[nT+1] + 1)
    q = np.array(S["q_settled"])[J["qad"]] + J["sgn"]*delta[J["ti"]]
    ieff = np.array([1.0/S["Minv_key"][dof, dof] for dof in J["dof"]])
    return dict(q=q, kp=om*om*ieff, kd=2*zeta*om*ieff)

def run(S, ref, x, om, ts, margin=None, landing=None):
    fn = MB.Tracker(S, ref, omega=om, tscale=ts) if om > 0 else None
    H = H_LAND if landing else ref["horizon"]
    return BI.rollout(x, set(S["types"]), None, S=S, ctrl_fn=fn, horizon=H, score="best", knot_dt=KNOT_DT, stop_margin=margin, landing=landing)

def score(o):
    reasons = [z for z in o["validity"]["reasons"] if z != "CONTACT_ENERGY"]; viol = 5.0*len(reasons) + (5.0 if o.get("bound_violation") else 0.0)
    if "margin_viol" in o and o["margin_viol"] > 0: viol += 1.0 + 20.0*o["margin_viol"]
    if "slip_viol" in o and o["slip_viol"] > 0: viol += 1.0 + 20.0*o["slip_viol"] + 0.01*o.get("slip_steps", 0)
    L = o.get("landing")
    if L is not None and not L["ok"]:
        viol += 1.0 + 5.0*(L["touchdown_t"] is None) + 5.0*L["nonfoot_after"] + 5.0*max(0.0, 0.85 - L.get("up_end", 0)) + 5.0*max(0.0, 0.40 - L.get("z_end", 0)) + 2.0*max(0.0, L.get("v_end", 9) - 0.5)
    return (100.0 + viol - 0.01*o["v"]) if viol > 0 else -o["v"]

def solve(S, ref, method, seed, iters, margin=None, land=False, pop=16):
    nT = len(S["types"]); nk = ref["nk"]; n = nT*nk + (1 if method == "APRIOR" else 0) + (nT + 2 if land else 0)
    om = MB.tune_gains(S, ref, 1.0)[0] if method == "APRIOR" else 0.0
    def unpack(z):
        x = z[:nT*nk]; k = nT*nk; ts = 1.0
        if method == "APRIOR": ts = 0.87 + 0.3*z[k]; k += 1
        return x, ts, (landing_from(z[k:], S) if land else None)
    def obj(z):
        x, ts, L = unpack(z); return score(run(S, ref, x, om, ts, margin, L))
    es = cma.CMAEvolutionStrategy(np.zeros(n), 0.35, {"bounds": [-1, 1], "seed": seed, "verbose": -9, "maxiter": iters, "popsize": pop})
    while not es.stop():
        X = es.ask(); es.tell(X, [obj(z) for z in X])
    zb = np.asarray(es.result.xbest); x, ts, L = unpack(zb)
    return zb, x, om, ts, L, unpack

def measure(S, ref, x, om, ts, margin, z, unpack, land, dts):
    out = []; dt0 = S["m"].opt.timestep
    for dt in dts:
        BI.apply_dt(S, dt); L = unpack(z)[2] if land else None    # landing gains rebuilt after re-settle
        o = run(S, ref, x, om, ts, margin, L); out.append(o)
    BI.apply_dt(S, dt0)
    ok = [score(o) < 0 for o in out]
    o = out[0]
    return dict(v=o["v"], ok_all=all(ok), ok_each=ok, v_each=[x_["v"] for x_ in out], reasons=o["validity"]["reasons"],
                crouch=o["mech"]["crouch_depth"], T=o["mech"]["T_push"], rot=o["mech"]["flight_rotation"],
                margin_viol=o.get("margin_viol"), landing=o.get("landing"), lim_exc=o["validity"]["joint_limit_excursion"])

def job(a):
    t0 = time.time()
    if a["start"] == "crouch":
        q = crouch_start(); c = dict(key="held_crouch", start=q, Q=np.repeat(q[None], 40, 0), fps=30, takeoff_ref=0.8, horizon=1.1, nk=22)
    else:
        c = AS.clip0()
    S = BI.setup(c["start"]); BI.apply_dt(S, a["dt_opt"]); ref = MB.analytic_clip(c) if a["method"] == "APRIOR" else c
    if a["kind"] == "ladder_t1":       # identical objective to the Phase 9 NS-04 standard-start run
        om, _ = MB.tune_gains(S, ref, 1.0) if a["method"] == "APRIOR" else (0.0, None)
        r = MB.solve_b1(S, ref, 0.0, a["seed"], om, a["iters"], 16, task="T1", native=(a["method"] != "APRIOR"))
        x = np.array(r["x"]); ts = r["tscale"]
        BI.apply_dt(S, 0.001); o = MB.run(S, ref, x, om, ts) if om else BI.rollout(x, set(S["types"]), None, S=S, horizon=ref["horizon"], score="best")
        BI.apply_dt(S, a["dt_opt"])
        ok = not [z for z in o["validity"]["reasons"] if z != "CONTACT_ENERGY"] and not o.get("bound_violation")
        m = dict(v=o["v"], ok_all=bool(ok), omega=om, tscale=ts, reasons=o["validity"]["reasons"], crouch=o["mech"]["crouch_depth"], T=o["mech"]["T_push"],
                 rot=o["mech"]["flight_rotation"], lim_exc=o["validity"]["joint_limit_excursion"])
    else:
        land = a["kind"] == "margin_land"; margin = MARGIN if a["kind"] in ("margin", "margin_land") else None
        z, x, om, ts, L, unpack = solve(S, ref, a["method"], a["seed"], a["iters"], margin, land)
        m = measure(S, ref, x, om, ts, margin, z, unpack, land, (0.001, 0.0005) if land else (0.001,))
        m["z"] = z.tolist()
    return dict(a, **m, x=np.asarray(x).tolist(), secs=time.time() - t0)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "check":
        q = crouch_start(); S = BI.setup(q); BI.apply_dt(S, 0.004)
        print("held crouch settle qdot", round(S["settle_qdot_max"], 4)); CG.verify_or_die(S); print("held crouch passes plant contract")
        for a in (dict(kind="margin_land", method="APRIOR", start="stand", seed=1, iters=2, dt_opt=0.002),
                  dict(kind="margin", method="APRIOR", start="stand", seed=1, iters=2, dt_opt=0.004),
                  dict(kind="ladder_t1", method="APRIOR", start="crouch", seed=1, iters=2, dt_opt=0.004)):
            r = job(a); print(a["kind"], a["start"], "secs", round(r["secs"]), "v", round(r["v"], 2), "ok", r["ok_all"], r.get("landing"), r.get("margin_viol"))
        sys.exit()
    jobs = []
    for it in (400, 1600):
        jobs += [dict(kind="ladder_t1", method="APRIOR", start="stand", seed=s, iters=it, dt_opt=0.004) for s in range(1, 7)]
    jobs += [dict(kind="margin", method="APRIOR", start="stand", seed=s, iters=400, dt_opt=0.004) for s in range(1, 7)]
    jobs += [dict(kind="margin_land", method=m, start="stand", seed=s, iters=400, dt_opt=0.002) for m in ("APRIOR", "NATIVE") for s in range(1, 7)]
    jobs += [dict(kind="ladder_t1", method="APRIOR", start=st, seed=s, iters=400, dt_opt=0.004) for st in ("crouch",) for s in range(1, 7)]
    jobs.sort(key=lambda j: -j["iters"]*(2 if j["dt_opt"] == 0.002 else 1)*(2.4 if j["kind"] == "margin_land" else 1))
    with Pool(int(os.environ.get("NPROC", "14"))) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(f"{OUT}/jump13.json", "w"), default=float); print("done", len(res), flush=True)
