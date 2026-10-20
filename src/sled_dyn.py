"""TRACK B (2026-10-17): sled push / pull on G1 -- dynamic leverage predictor (engine.leverage_dyn) vs a larger search.

Labels: the right hand is WELDED to the sled post (weld attachment, simulator abstraction); the sled is a free box sliding at mu 0.4;
simulator only. The predictor runs from the robot spec BEFORE any search and its numbers are stored first.

Weld fix (general, every mass and direction, not tuned per config): the overnight L3 weld used solref 0.002 s (= 2 dt at 1 ms, MuJoCo's
stiffness floor) with solimp 0.99-0.999. Closing the robot-floor-sled loop with that weld is chaotic: with z = 0 (stand prior, no
drive) a 2e-5 m perturbation of the start decides whether the feet slide > 3 cm (2/4 starts at 1 ms and 3/4 at 0.5 ms slid; weld
force spikes of 3-20 kN, both feet airborne). The stand controller holds without the weld (0/8 slid). With the weld time constant
0.005 s (solimp 0.9/0.95) 0/8 slid. So the overnight defect was the weld stiffness, not the controller; the 0.5 ms vs 1 ms difference
was sensitivity, not a dt effect (fresh 0.5 ms plant held, the same plant re-timed from 1 ms slid: the start differed by 1.7e-5 m).

Search representation (19-D, the same for every config): stand-controller gain scales (2); phase A static offsets on waist pitch,
right shoulder pitch, right elbow, right wrist pitch and symmetric hip/knee/ankle pitch (7) ramped in over 1.5 s; a LEAN SET-POINT
for the controller's capture point (COM ahead (+) / behind (-) the feet centre, +-0.25 m) for phase A; phase B (the drive): deltas
on the same 7 offsets and on the lean set-point, switched at t_B in [1, 3] s and ramped over 0.75 s (8 + 1). Nothing in it says
push or pull or which way to lean: the sign and size are found.
Score (min) = -min(disp, 0.3)/0.3 + 0.2 util_rms + 10 (fell / feet slid > 3 cm), the WORSE of 1 ms and 0.5 ms.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco
import leverage_probes as LP

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "overnight", "B"))
WELD_TC = 0.005
POST = ["waist_pitch_joint", "right_shoulder_pitch_joint", "right_elbow_joint", "right_wrist_pitch_joint"]
LEGS = ("hip_pitch_joint", "knee_joint", "ankle_pitch_joint")
DIM = 19
LEAN_SCALE = 0.25


def decode(z):
    z = np.clip(np.asarray(z, float), -1, 1)
    return dict(gains=z[:2], A=z[2:9], leanA=LEAN_SCALE*z[9], dB=z[10:17], dleanB=LEAN_SCALE*z[17], tB=2.0 + z[18])


def evaluate(args):
    """args = (mass, sgn, z, T, dt, record, perturb_seed). dt = 'both' -> worse of 1 and 0.5 ms."""
    mass, sgn, z, T, dt, record, pseed = args
    if dt == "both":
        r1 = evaluate((mass, sgn, z, T, 0.001, record, pseed))
        # a candidate that already fails at 1 ms scores >= 10 whatever 0.5 ms does: the 0.5 ms run is skipped (it cannot change the ranking
        # between passing and failing candidates; among failures the 1 ms score is used)
        r2 = evaluate((mass, sgn, z, T, 0.0005, False, pseed)) if r1["reason"] is None else dict(r1, reason="SKIPPED_1MS_FAILED")
        r1["dt0005"] = {k: r2[k] for k in ("score", "t_held", "reason", "disp", "com_lean_mean", "weld_err_max", "weld_ang_max")}; r1["score"] = max(r1["score"], r2["score"])
        r1["disp_min"] = min(r1["disp"], r2["disp"]); r1["ok_both"] = (r1["reason"] is None and r2["reason"] is None); return r1
    P = decode(z); S = LP.l3_plant(mass, dt); m = S["m"]; Wd = S["world"]; bid = Wd.bid["sled"]
    q_start = np.array(S["q_settled0"])
    if pseed: q_start[:3] += np.random.default_rng(pseed).normal(0, 2e-5, 3)
    x0 = float(q_start[m.jnt_qposadr[m.body_jntadr[bid]]])
    act = {j: LP.actuator_of(m, j) for j in POST}; legs = [(LP.actuator_of(m, f"{s}_{j}"), k) for k, j in enumerate(LEGS) for s in ("left", "right")]
    scl = np.array([LP.SCALE.get(j, 0.5) for j in POST] + [0.3]*3)
    offA = scl*P["A"]; offB = scl*P["dB"]
    fb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in ("left_foot", "right_foot")]; foot0 = {}; com = []; werr = [0.0, 0.0]
    ctl_ref = {}
    def wB(t): return float(np.clip((t - P["tB"])/0.75, 0.0, 1.0))
    def extra(t, d, w):
        # posture (the stand_rollout posture list is empty here; all offsets are applied in this hook)
        b = wB(t); off = w*offA + b*offB
        for i, j in enumerate(POST):
            if act[j] is not None: d.ctrl[act[j]] += off[i]
        for a, k in legs: d.ctrl[a] += off[4 + k]
    def init(d):
        Wd.grasp(d, "sled"); e = Wd.eq["sled"]; m.eq_solref[e] = [WELD_TC, 1.0]; m.eq_solimp[e] = [0.9, 0.95, 0.001, 0.5, 2.0]
        foot0.update({i: d.site_xpos[i][:2].copy() for i in fb})
    def cb(t, d):
        fd = max(np.linalg.norm(d.site_xpos[i][:2] - foot0[i]) for i in fb)
        if fd > 0.03: return "FEET_SLID"
        if int(round(t/dt)) % int(round(0.02/dt)) == 0:
            # weld compliance: the hand origin's drift in the SLED frame (m) and the relative rotation (rad) since the weld closed
            hb = Wd.hand; Rs = d.xmat[bid].reshape(3, 3); r_ = Rs.T @ (d.xpos[hb] - d.xpos[bid]); Rr = Rs.T @ d.xmat[hb].reshape(3, 3)
            if "r0" not in foot0: foot0["r0"] = r_.copy(); foot0["R0"] = Rr.copy()
            ang = float(np.arccos(np.clip((np.trace(foot0["R0"].T @ Rr) - 1)/2, -1, 1)))
            werr[0] = max(werr[0], float(np.linalg.norm(r_ - foot0["r0"]))); werr[1] = max(werr[1], ang)
            com.append([t, float(d.subtree_com[1][0] - 0.5*(d.site_xpos[fb[0]][0] + d.site_xpos[fb[1]][0])), float(sgn*(d.xpos[bid][0] - x0)), float(d.subtree_com[1][2])])
    def lean_target(t, w):
        return w*P["leanA"] + wB(t)*P["dleanB"]
    r = LP.stand_rollout(S, np.concatenate([P["gains"], []]), T, cb, record=record, posture=[], robot="G1", q_start=q_start if pseed else None,
                         init_cb=init, extra_targets=extra, ramp=1.5, robot_com=True, lean_cb=lean_target)
    C = np.array(com) if com else np.zeros((1, 4)); disp = float(C[-1, 2]); late = C[C[:, 0] >= 0.5*T] if len(C) > 1 else C
    r.update(disp=disp, com_lean_mean=float(late[:, 1].mean()) if len(late) else None, com_lean_minmax=[float(C[:, 1].min()), float(C[:, 1].max())],
             weld_err_max=werr[0], weld_ang_max=werr[1], scene=S["scene"], com_trace=C[::5].tolist() if record else None)
    r["score"] = -min(max(disp, -0.3), 0.3)/0.3 + 0.2*float(r["util_rms"]) + (10.0 if r["reason"] else 0.0); return r


def predictor(out):
    from engine import leverage_dyn as LD
    S = LP.l3_plant(20.0, 0.001); res = {}
    for sgn, nm in ((1, "push"), (-1, "pull")):
        for itf in ("weld", "point"):
            t = time.time(); r = LD.predict(S, sgn, interface=itf); r["time_s"] = time.time() - t; res[f"{nm}_{itf}"] = r
            print(nm, itf, r.get("status"), "m_max_total", r.get("m_max_total"), "lunge bound", r.get("m_max_total_lunge_bound"), "lean", r.get("lean_pred_mean"), flush=True)
    # the overnight static brace number, for the record
    res["static_brace_overnight"] = dict(f_max_N=42.14, limiting="tipping (unloaded-stable pose)", m_total_equiv=42.14/(0.4*9.80665))
    json.dump(res, open(out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)); return res


def run(a):
    os.makedirs(OUT, exist_ok=True)
    pf = f"{OUT}/predictor.json"
    if not os.path.exists(pf): predictor(pf)
    masses = [float(x) for x in a.masses.split(",")]; dirs = [(1, "push"), (-1, "pull")] if a.dirs == "both" else [((1, "push") if a.dirs == "push" else (-1, "pull"))]
    for sgn, nm in dirs:
        x0 = None
        for mass in masses:
            tag = f"{nm}_{mass:g}kg"; jf = f"{OUT}/{tag}.json"
            if os.path.exists(jf) and not a.force:
                J = json.load(open(jf)); x0 = np.asarray(J["search"]["best_z"]); continue
            try:
                log = open(f"{OUT}/{tag}.log", "w"); t0 = time.time()
                prior = evaluate((mass, sgn, np.zeros(DIM), a.T, "both", False, 0))
                res = LP.cma_search(evaluate, lambda x: (mass, sgn, x, a.T, "both", False, 0), DIM, a.iters, a.pop, a.workers, sigma=0.4, x0=x0, log=log)
                bz = np.asarray(res["best_z"])
                rob = LP._pool_map(evaluate, [(mass, sgn, bz, a.T, "both", False, s) for s in (1, 2, 3, 4)], a.workers)
                J = dict(track="B", robot="G1", mass=mass, sled_total_mass=mass + 1.0, direction=nm, mu=0.4, T=a.T, weld_tc=WELD_TC,
                         label="right hand WELDED to the sled post (weld attachment, simulator only); free sled box sliding at mu 0.4; lean set-point and two-phase posture drive searched",
                         prior={k: v for k, v in prior.items() if k != "trace"}, search=res, warm_start=x0 is not None,
                         robust=[dict(seed=s, score=r_["score"], disp_min=r_["disp_min"], ok_both=r_["ok_both"], reason=r_["reason"], reason05=r_["dt0005"]["reason"]) for s, r_ in zip((1, 2, 3, 4), rob)],
                         decoded={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in decode(bz).items()}, wall_s=time.time() - t0)
                b = res["best"]; J["moved_0p2_both"] = bool(b.get("ok_both") and b.get("disp_min", -1) >= 0.2)
                J["robust_moved_0p2"] = int(sum(1 for r_ in J["robust"] if r_["ok_both"] and r_["disp_min"] >= 0.2))
                json.dump(J, open(jf, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
                print(tag, "best", round(res["best_score"], 3), "disp_min", round(b.get("disp_min", 0), 3), "ok_both", b.get("ok_both"), "lean", b.get("com_lean_mean"),
                      "robust", J["robust_moved_0p2"], "/4", "s/eval", round(res["s_per_eval_cpu"], 2), "wall", round(time.time() - t0), flush=True)
                if J["moved_0p2_both"] or b.get("ok_both"): x0 = bz
            except Exception:
                import traceback; traceback.print_exc(); print("CONFIG FAILED; continuing", flush=True)


def render_best(nm, mass, T=4.0):
    from engine import render_any as RA
    J = json.load(open(f"{OUT}/{nm}_{mass:g}kg.json")); sgn = 1 if nm == "push" else -1; bz = np.asarray(J["search"]["best_z"])
    # the search workers' plant history (built at 1 ms, re-timed to 0.5 ms, back to 1 ms) reproduces the stored numbers exactly; a plant
    # built fresh at 1 ms settles 1e-5 m differently and, the welded stand being sensitive near the slip limit, gives a different rollout
    LP.l3_plant(mass, 0.001); LP.l3_plant(mass, 0.0005)
    r = evaluate((mass, sgn, bz, T, 0.001, True, 0)); np.save(f"{OUT}/{nm}_{mass:g}kg_trace.npy", r["trace"])
    RA.render(r["scene"], r["trace"], r["trace_dt"], f"{OUT}/{nm}_{mass:g}kg.mp4", title=f"G1 {dict(push='pushes', pull='pulls')[nm]} a {mass:g} kg sled (mu 0.4, hand welded)",
              sub=f"moved {r['disp']:.2f} m at 1 ms, lean {r['com_lean_mean']:.2f} m", azimuth=90)
    return r


def mechanism(nm, mass, T=4.0, dt=0.001, prefix=""):
    """Re-run a stored best candidate and log the MECHANISM the predictor claims: the sled's floor normal (-> the hand's vertical force
    on the sled, F_z_up = m_s g - N), the sled floor friction, foot friction use (|f_t| / mu f_n per foot sum), the CoP x relative to the
    sole, the most-saturated actuators, and the robot lean. Sampled every 20 ms over the moving phase (sled speed > 1 cm/s)."""
    J = json.load(open(f"{OUT}/{prefix}{nm}_{mass:g}kg.json")); sgn = 1 if nm == "push" else -1; bz = np.asarray(J["search"]["best_z"])
    from engine import leverage as LV
    S = LP.l3_plant(mass, dt); m = S["m"]; Wd = S["world"]; bid = Wd.bid["sled"]; fl = S["floor"]; fg = set(S["foot_geoms"]); sg = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
    tm = LV.tau_max(m); names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a_) for a_ in range(m.nu)]; rows = []; ms = float(m.body_subtreemass[bid])
    orig = LP.stand_rollout
    def wrap(*a_, **k_):
        cb = a_[3]
        def cb2(t, d):
            if int(round(t/dt)) % int(round(0.02/dt)) == 0 and t > 0.3:
                N = 0.0; Fs = 0.0; fn = 0.0; ft = 0.0; cop = 0.0; mu_f = 1.0
                for ci in range(d.ncon):
                    c = d.contact[ci]; fv = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fv)
                    if fl not in (c.geom1, c.geom2): continue
                    o = c.geom2 if c.geom1 == fl else c.geom1
                    if o in sg: N += fv[0]; Fs += np.hypot(fv[1], fv[2])
                    elif o in fg: fn += fv[0]; ft += np.hypot(fv[1], fv[2]); cop += fv[0]*c.pos[0]; mu_f = float(m.geom_friction[o][0])
                u = np.abs(d.actuator_force)/tm; top = np.argsort(-u)[:3]
                rows.append(dict(t=round(t, 2), v_sled=float(d.qvel[m.jnt_dofadr[m.body_jntadr[bid]]]), Fz_up=float(ms*9.80665 - N), sled_fric=float(Fs), foot_fric_use=float(ft/max(mu_f*fn, 1e-9)),
                                 cop_minus_feet=float(cop/max(fn, 1e-9) - 0.5*sum(d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n_)][0] for n_ in ("left_foot", "right_foot"))),
                                 top_util=[(names[i], round(float(u[i]), 2)) for i in top]))
            return cb(t, d)
        a_ = list(a_); a_[3] = cb2; return orig(*a_, **k_)
    LP.stand_rollout = wrap
    try: r = evaluate((mass, sgn, bz, T, dt, False, 0))
    finally: LP.stand_rollout = orig
    mv = [r_ for r_ in rows if abs(r_["v_sled"]) > 0.01] or rows
    sat = {}
    for r_ in mv:
        for n_, u_ in r_["top_util"]:
            if u_ >= 0.98: sat[n_] = sat.get(n_, 0) + 1
    return dict(direction=nm, mass=mass, dt=dt, disp=r["disp"], reason=r["reason"], n_moving_samples=len(mv), Fz_up_mean=float(np.mean([r_["Fz_up"] for r_ in mv])),
                sled_fric_mean=float(np.mean([r_["sled_fric"] for r_ in mv])), foot_fric_use_mean=float(np.mean([r_["foot_fric_use"] for r_ in mv])), foot_fric_use_max=float(np.max([r_["foot_fric_use"] for r_ in rows])),
                cop_minus_feet_mean=float(np.mean([r_["cop_minus_feet"] for r_ in mv])), saturated_actuator_counts=dict(sorted(sat.items(), key=lambda kv: -kv[1])), samples=rows[::5])


def seed_from_predictor(nm):
    """Map the predictor's spec-only strategy (weld interface) into the search vector: phase A = the predicted start pose (waist, right arm
    joints from the predictor's IK, lean set-point = predicted start lean), phase B = the predicted stroke-end pose and lean. Legs = the
    predicted symmetric crouch; gains start at 0 (the predictor's lean is a rigid ankle lean, which the controller's lean set-point realises). Values clip at +-1."""
    from engine import leverage_dyn as LD
    P = json.load(open(f"{OUT}/predictor.json"))[f"{nm}_weld"]; sgn = 1 if nm == "push" else -1; S = LP.l3_plant(20.0, 0.001); m = S["m"]
    q0 = np.array(S["q_settled0"]); poses = P["at_max"]["pose"]; svals = [p_["s"] for p_ in P["per_s_best"]]
    def joints(sd, prm):
        fam = LD.family(S, sd, sgn, [prm[0]], [prm[1]], [prm[2]]); q = fam[0][1]
        return np.array([q[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] - q0[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] for j in POST])
    oA = joints(svals[0], poses[0]); oB = joints(svals[-1], poses[-1]); scl = np.array([LP.SCALE.get(j, 0.5) for j in POST])
    z = np.zeros(DIM); z[2:6] = oA/scl; z[9] = P["lean_pred_start"]/LEAN_SCALE; z[10:14] = (oB - oA)/scl; z[17] = (P["lean_pred_end"] - P["lean_pred_start"])/LEAN_SCALE; z[18] = -0.5
    cA, cB = poses[0][2], poses[-1][2]; z[6:9] = np.array([-cA, 2*cA, -cA])/0.3; z[14:17] = np.array([-(cB - cA), 2*(cB - cA), -(cB - cA)])/0.3   # the predicted crouch
    return np.clip(z, -1, 1), dict(oA=oA.tolist(), oB=oB.tolist(), poses=[poses[0], poses[-1]])


def run_seeded(a):
    """Same objective and gates, x0 = the predictor's strategy (scaffold-seeded) instead of the previous mass's best / zeros."""
    for nm, mass in ((n_, float(m_)) for n_, m_ in (x.split(":") for x in a.seeded.split(","))):
        sgn = 1 if nm == "push" else -1; tag = f"seeded_{nm}_{mass:g}kg"; log = open(f"{OUT}/{tag}.log", "w"); t0 = time.time()
        z0, info = seed_from_predictor(nm); seed_eval = evaluate((mass, sgn, z0, a.T, "both", False, 0))
        res = LP.cma_search(evaluate, lambda x: (mass, sgn, x, a.T, "both", False, 0), DIM, a.iters, a.pop, a.workers, sigma=0.25, x0=z0, log=log)
        bz = np.asarray(res["best_z"]); rob = LP._pool_map(evaluate, [(mass, sgn, bz, a.T, "both", False, s_) for s_ in (1, 2, 3, 4)], a.workers); b = res["best"]
        J = dict(track="B", kind="predictor-seeded search", mass=mass, sled_total_mass=mass + 1.0, direction=nm, z0=z0.tolist(), seed_info=info,
                 seed_eval={k: v for k, v in seed_eval.items() if k not in ("trace", "com_trace")}, search=res, decoded={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in decode(bz).items()},
                 robust=[dict(seed=s_, disp_min=r_["disp_min"], ok_both=r_["ok_both"], reason=r_["reason"], reason05=r_["dt0005"]["reason"]) for s_, r_ in zip((1, 2, 3, 4), rob)],
                 moved_0p2_both=bool(b.get("ok_both") and b.get("disp_min", -1) >= 0.2), wall_s=time.time() - t0)
        J["robust_moved_0p2"] = int(sum(1 for r_ in J["robust"] if r_["ok_both"] and r_["disp_min"] >= 0.2))
        json.dump(J, open(f"{OUT}/{tag}.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
        print(tag, "seed disp_min", round(seed_eval["disp_min"], 3), seed_eval["reason"], seed_eval["dt0005"]["reason"], "| best disp_min", round(b.get("disp_min", 0), 3), "ok_both", b.get("ok_both"),
              "lean", b.get("com_lean_mean"), "robust", J["robust_moved_0p2"], "/4", "wall", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["predict", "search", "render", "time", "seeded"]); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--iters", type=int, default=40); ap.add_argument("--pop", type=int, default=16); ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--masses", default="20,30,40,60,80"); ap.add_argument("--dirs", default="both"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--mass", type=float, default=20.0); ap.add_argument("--name", default="push"); ap.add_argument("--seeded", default="push:30,pull:80")
    a = ap.parse_args()
    if a.cmd == "predict": os.makedirs(OUT, exist_ok=True); predictor(f"{OUT}/predictor.json")
    elif a.cmd == "search": run(a)
    elif a.cmd == "render": render_best(a.name, a.mass, a.T)
    elif a.cmd == "seeded": run_seeded(a)
    elif a.cmd == "time":
        t = time.time(); R = LP._pool_map(evaluate, [(a.mass, 1 if a.name == "push" else -1, np.random.default_rng(i).uniform(-0.5, 0.5, DIM), a.T, "both", False, 0) for i in range(a.workers*2)], a.workers)
        print("s/eval (cpu, both dt)", (time.time() - t)/len(R)*a.workers, [(round(r["disp_min"], 3), r["reason"], r["dt0005"]["reason"]) for r in R])
