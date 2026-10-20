"""LEVERAGE probes (2026-10-16): L1 head load, L2 sheet lift, L3 sled. Each probe = a physical objective only (hold time, effort,
lift clearance); the strategy (posture, grip, orientation, lean) is searched, never scripted. The predictor (engine.leverage, L0)
ranks candidate strategies from the spec before search; the JSON records predicted vs found.

Representation (all probes): the Phase 14 capture-point STAND controller (stand mode) is the prior; the search vector z holds its two
gain scales (0.5..1.5x) plus static POSTURE OFFSETS (SCALE rad per unit) on trunk/arm joint targets, ramped in over 1 s, plus the
probe's own strategy parameters. CLI: python leverage_probes.py <L1|L2|L3> [--workers N] [--iters K] [--pop P] [--smoke] [--tag t]
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "overnight"))
POSTURE = ["waist_pitch_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint",
           "left_elbow_joint", "right_elbow_joint"]
# offset (rad) per unit of z: shoulder pitch reaches overhead (-3 rad from hanging) so the hands CAN reach a load on the head
SCALE = {"waist_pitch_joint": 0.3, "left_shoulder_pitch_joint": 3.0, "right_shoulder_pitch_joint": 3.0, "left_shoulder_roll_joint": 1.2, "right_shoulder_roll_joint": 1.2,
         "left_elbow_joint": 1.5, "right_elbow_joint": 1.5, "left_wrist_pitch_joint": 1.0, "right_wrist_pitch_joint": 1.0, "left_shoulder_yaw_joint": 1.0, "right_shoulder_yaw_joint": 1.0}


def actuator_of(m, jn):
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
    return [a for a in range(m.nu) if int(m.actuator_trnid[a][0]) == j][0] if j >= 0 else None


def stand_rollout(S, z, T, step_cb=None, record=False, ramp=1.0, posture=POSTURE, robot="G1H", q_start=None, extra_targets=None, init_cb=None, robot_com=False, lean_cb=None):
    """robot_com=True: the controller's capture point uses the ROBOT's COM (it reads the world subtree COM, which includes free objects:
    right for a carried load, wrong for a sled resting on the floor), through its cp_offset hook.
    Stand controller + posture offsets. step_cb(t, d) -> None | str (a stop reason) is called every step before mj_step (it may set
    xfrc_applied). Returns time held, utilisation statistics (|actuator force|/limit over non-finger actuators) and a qpos trace."""
    import balance as B, branch_i as BI; from engine import core, leverage as LV
    m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = S["q_settled"] if q_start is None else q_start; d.qvel[:] = S["v_settled"] if q_start is None else 0; mujoco.mj_forward(m, d)
    ctl, par = core.landing_controller(S, "G1H" if robot == "G1H" else robot); ctl.p["mode"] = "stand"; ctl.t_touch = -1e9
    ctl.p["k_cp"] *= 1.0 + 0.5*float(np.clip(z[0], -1, 1)); ctl.p["k_trunk"] *= 1.0 + 0.5*float(np.clip(z[1], -1, 1)); ctl._kc = None
    if init_cb is not None: init_cb(d)
    aids = [actuator_of(m, j) for j in posture]; off = np.array([SCALE.get(j, 0.5) for j in posture])*np.asarray(z[2:2+len(posture)], float)
    restore = B.all_servos(S, ctl.p["stiff"]); tm = LV.tau_max(m)
    keep = np.array([np.isfinite(tm[a]) and "hand" not in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a][0])) or "") for a in range(m.nu)])
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis"); cp0 = np.asarray(getattr(ctl, "cp_offset", 0.0), float)
    tb = ctl.tb; dt = m.opt.timestep; n = int(T/dt); trace = []; u2 = 0.0; umax = []; reason = None; t = 0.0
    try:
        for (a_, dof, qad, tau, kp, kd) in S["held"]: d.ctrl[a_] = S["q_settled"][qad]
        for s_ in range(n):
            t = s_*dt; k_c = int(round(t/B.CTRL_DT, 6))
            if k_c != ctl._kc:
                if robot_com:
                    jc = np.zeros((3, m.nv)); mujoco.mj_jacSubtreeCom(m, d, jc, 0); vw = jc @ d.qvel; mujoco.mj_jacSubtreeCom(m, d, jc, pel); vr = jc @ d.qvel
                    w0 = np.sqrt(9.80665/max(float(d.subtree_com[0][2]), 0.2)); ctl.cp_offset = cp0 + (d.subtree_com[0][:2] - d.subtree_com[pel][:2]) + (vw[:2] - vr[:2])/w0
                    if lean_cb is not None: ctl.cp_offset = ctl.cp_offset + np.array([float(lean_cb(t, min(1.0, t/ramp))), 0.0])   # searched lean set-point (COM ahead + / behind - the feet centre)
                ctl._kc = k_c; d.ctrl[:] = ctl.targets(k_c*B.CTRL_DT, d, True); w = min(1.0, t/ramp)
                for a, o in zip(aids, off):
                    if a is not None: d.ctrl[a] += w*o
                if extra_targets: extra_targets(t, d, w)
            d.xfrc_applied[:] = 0
            if step_cb is not None:
                reason = step_cb(t, d)
                if reason: break
            BI.backemf_brake(m, d, S); mujoco.mj_step(m, d)
            if not np.isfinite(d.qacc).all(): reason = "DIVERGED"; break
            if d.xmat[tb][8] < np.cos(np.radians(35)): reason = "FELL"; break
            if s_ % 10 == 0 and t >= 0.2:   # statistics skip the 0.2 s servo switch-on transient (saturates the legs for ~0.1 s at every pose)
                u = np.abs(d.actuator_force[keep])/tm[keep]
                if os.environ.get("LEV_DEBUG") and s_ % 500 == 0: print(t, [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, int(np.flatnonzero(keep)[i])), round(float(u[i]), 2)) for i in np.argsort(-u)[:3]])
                u2 += float(np.mean(u**2)); umax.append(float(u.max()))
                if record: trace.append(d.qpos.copy())
    finally: restore()
    k = max(1, len(umax))
    return dict(t_held=float(t if reason else T), reason=reason, util_rms=float(np.sqrt(u2/k)), util_p95=float(np.percentile(umax, 95)) if umax else None,
                trace=(np.asarray(trace) if record else None), trace_dt=10*dt)


# ------------------------------------------------------------------------------------------------ L1 head load
OBJ = {"box": dict(kind="box", size=(0.15, 0.15, 0.08)), "pot": dict(kind="cylinder", size=(0.13, 0.09))}


def head_top_settled():
    import throw as TH; from engine import leverage as LV
    S0 = TH.setup("G1H", ball=False); m = S0["m"]; d = mujoco.MjData(m); d.qpos[:] = S0["q_settled"]; mujoco.mj_forward(m, d)
    return LV.head_top(m, d, LV.g1h_parts(m)["head_geom"]), S0


def l1_plant(obj, mass, arm="head", dt=0.001):
    """arm='head': a free object resting on the head (no weld; hand-object contact allowed). arm='hands': the same mass merged at
    BOTH palms (mass/2 each; weld-equivalent, labelled), no object."""
    import throw as TH, branch_i as BI; from engine import world as W
    ht, S0 = head_top_settled()
    if arm == "hands":
        S = TH.setup("G1H", start=np.array(S0["q0"]) + 1e-9, ball=False); m = S["m"]
        from engine import leverage as LV; H = LV.g1h_parts(m)
        for b, lp in (H["rpalm"], H["lpalm"]):
            mb = float(m.body_mass[b]); m.body_ipos[b] = (mb*m.body_ipos[b] + mass/2*lp)/(mb + mass/2); m.body_mass[b] = mb + mass/2
        mujoco.mj_setConst(m, mujoco.MjData(m)); S.pop("q_settled", None); BI.apply_dt(S, dt); return S, None
    o = OBJ[obj]; hz = o["size"][2] if o["kind"] == "box" else o["size"][1]
    ob = W.obj("load", kind=o["kind"], size=o["size"], mass=mass, pos=(float(ht[0]), float(ht[1]), float(ht[2] + hz + 0.004)), graspable=False, rgba=(0.75, 0.45, 0.2, 1))
    path = W.build_scene("G1H", objects=[ob]); S = TH.setup("G1H", mjcf=path, ball=False); BI.apply_dt(S, dt)
    # the object is PLACED at t=0 on the settled robot's head (it slides off a statue's 5-deg-tilted dome during the 4 s settle)
    if True:   # re-placed every call (apply_dt re-settles and resets q_settled)
        from engine import leverage as LV
        m = S["m"]; d = mujoco.MjData(m); q = np.array(S["q_settled"]); d.qpos[:] = q; mujoco.mj_forward(m, d)
        top = LV.head_top(m, d, LV.g1h_parts(m)["head_geom"]); bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "load"); ja = m.jnt_qposadr[m.body_jntadr[bid]]
        q[ja:ja+3] = top + [0, 0, hz + 0.002]; q[ja+3:ja+7] = [1, 0, 0, 0]; v = np.array(S["v_settled"]); v[m.jnt_dofadr[m.body_jntadr[bid]]:][:6] = 0
        S["q_settled"] = q; S["v_settled"] = v; S["_placed"] = True
    return S, path


def l1_eval(args):
    """args = (obj, mass, arm, z, T, dt, record) -> score dict. Objective (minimise) = -(t_held/T) + 0.5 util_rms."""
    obj, mass, arm, z, T, dt, record = args
    S, path = l1_plant(obj, mass, arm, dt); m = S["m"]
    if arm == "head":
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "load"); z0 = float(np.asarray(S["q_settled"])[m.jnt_qposadr[m.body_jntadr[bid]] + 2])
        tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        def cb(t, d):
            if d.xpos[bid][2] < z0 - 0.10 or np.linalg.norm(d.xpos[bid][:2] - d.xpos[tb][:2]) > 0.25: return "DROPPED"
    else: cb = None
    r = stand_rollout(S, z, T, cb, record=record)
    r["score"] = -(r["t_held"]/T) + 0.5*r["util_rms"]; r["scene"] = path; r["z0_settled"] = (float(z0) if arm == "head" else None)
    return r


# ------------------------------------------------------------------------------------------------ search driver
def _pool_map(fn, jobs, workers):
    if workers <= 1: return [fn(j) for j in jobs]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(workers) as p: return p.map(fn, jobs, chunksize=1)


def cma_search(fn, mkjob, dim, iters, pop, workers, seed=0, sigma=0.3, x0=None, log=None):
    import cma
    es = cma.CMAEvolutionStrategy(np.zeros(dim) if x0 is None else x0, sigma, dict(popsize=pop, seed=seed + 1, bounds=[-1, 1], verbose=-9))
    best = (np.inf, None, None); hist = []; t0 = time.time(); nev = 0
    for it in range(iters):
        X = es.ask(); R = _pool_map(fn, [mkjob(np.asarray(x)) for x in X], workers); nev += len(X)
        f = [r["score"] for r in R]; es.tell(X, f); i = int(np.argmin(f))
        if f[i] < best[0]: best = (f[i], np.asarray(X[i]).tolist(), {k: v for k, v in R[i].items() if k != "trace"})
        hist.append(dict(it=it, best=float(best[0]), gen_best=float(f[i]), gen_med=float(np.median(f)), s_per_eval=(time.time() - t0)/nev*max(1, workers)))
        if log: print(json.dumps(hist[-1]), file=log, flush=True)
    return dict(best_score=best[0], best_z=best[1], best=best[2], hist=hist, evals=nev, wall_s=time.time() - t0, s_per_eval_cpu=(time.time() - t0)/nev*max(1, workers))


def predict_l1(mass, chest=(0.30, 0.12, 1.00)):
    """L0 prediction for L1: head vs both hands at chest height, max utilisation each (the predictor's ranking)."""
    from engine import leverage as LV
    import throw as TH
    S = TH.setup("G1H", ball=False); m = S["m"]; H = LV.g1h_parts(m); d = mujoco.MjData(m); q = np.array(S["q_settled"]); fg = sorted(S["foot_geoms"])
    d.qpos[:] = q; mujoco.mj_forward(m, d); ht = LV.head_top(m, d, H["head_geom"])
    head = LV.solve(m, q, [LV.point_load(m, d, H["torso"], ht, mass)], fg, robot_bodies=S["robot_bodies"], exclude_act=H["fingers"])
    qa, err = LV.site_ik(m, q, [(H["rpalm"][0], H["rpalm"][1], (chest[0], -chest[1], chest[2])), (H["lpalm"][0], H["lpalm"][1], (chest[0], chest[1], chest[2]))], LV.ARMS)
    d.qpos[:] = qa; mujoco.mj_forward(m, d); P_ = [d.xpos[b] + d.xmat[b].reshape(3, 3) @ lp for b, lp in (H["rpalm"], H["lpalm"])]
    hands = LV.solve(m, qa, [LV.point_load(m, d, H["rpalm"][0], P_[0], mass/2), LV.point_load(m, d, H["lpalm"][0], P_[1], mass/2)], fg, robot_bodies=S["robot_bodies"], exclude_act=H["fingers"])
    d.qpos[:] = q; mujoco.mj_forward(m, d); P_ = [d.xpos[b] + d.xmat[b].reshape(3, 3) @ lp for b, lp in (H["rpalm"], H["lpalm"])]
    sides = LV.solve(m, q, [LV.point_load(m, d, H["rpalm"][0], P_[0], mass/2), LV.point_load(m, d, H["lpalm"][0], P_[1], mass/2)], fg, robot_bodies=S["robot_bodies"], exclude_act=H["fingers"])
    sm = lambda r: dict(status=r["status"], max_util=r.get("max_util"), rms_util=r.get("rms_util"), worst=r.get("worst_joint"))   # INFEASIBLE = no static solution (tips / friction)
    rank = sorted([(k, (r.get("max_util") if r["status"] == "OK" else float("inf"))) for k, r in (("head", head), ("hands_chest", hands), ("hands_sides", sides))], key=lambda kv: kv[1])
    return dict(hands_sides=sm(sides), ranking=rank, head=sm(head), hands_chest=sm(hands))


def run_l1(a):
    out_dir = f"{RES}/leverage_L1"; os.makedirs(out_dir, exist_ok=True)
    masses = [5.0] if a.smoke else [2.0, 5.0, 8.0]; objs = ["box"] if a.smoke else ["box", "pot"]
    T = 2.0 if a.smoke else 10.0; dim = 2 + len(POSTURE)
    for mass in masses:
        pred = predict_l1(mass)
        for arm, obj in [("head", o) for o in objs] + [("hands", "weld")]:
            try:
                tag = f"{arm}_{obj}_{mass:g}kg{a.tag}"; log = open(f"{out_dir}/{tag}.log", "w")
                prior = l1_eval((obj, mass, arm, np.zeros(dim), T, 0.001, False))
                res = cma_search(l1_eval, lambda x: (obj, mass, arm, x, T, 0.001, False), dim, a.iters, a.pop, a.workers, log=log)
                chk = l1_eval((obj, mass, arm, np.asarray(res["best_z"]), T, 0.0005, False))
                rec = l1_eval((obj, mass, arm, np.asarray(res["best_z"]), T, 0.001, True))
                J = dict(probe="L1_head_load", robot="G1H", arm=arm, obj=obj, mass=mass, T=T, label=("free object on the head, no weld" if arm == "head" else "mass merged at both palms (weld-equivalent)"),
                         predictor=pred, prior={k: v for k, v in prior.items() if k != "trace"}, search={k: v for k, v in res.items()}, check_dt0005={k: v for k, v in chk.items() if k != "trace"},
                         posture_joints=POSTURE)
                np.save(f"{out_dir}/{tag}_trace.npy", rec["trace"]); json.dump(J, open(f"{out_dir}/{tag}.json", "w"), indent=1, default=float)
                if not a.smoke or a.render:
                    from engine import render_any as RA
                    RA.render(rec["scene"] or "G1H", rec["trace"], rec["trace_dt"], f"{out_dir}/{tag}.mp4", title=f"G1H carries {mass:g} kg: {arm} ({obj})",
                              sub=f"held {rec['t_held']:.1f}/{T:g} s, util rms {rec['util_rms']:.2f}", hands_mass=(mass if arm == "hands" else 0))
                print(tag, "prior", round(prior["t_held"], 2), prior["reason"], "best", round(res["best_score"], 3), res["best"]["t_held"], res["best"]["reason"], "s/eval", round(res["s_per_eval_cpu"], 2), flush=True)
            except Exception:
                import traceback; traceback.print_exc(); print("CONFIG FAILED; continuing", flush=True)


# ------------------------------------------------------------------------------------------------ L2 sheet lift
SHEETS = {"plywood": dict(size=(1.2, 0.6, 0.009), mass=20.0, rgba=(0.82, 0.68, 0.45, 1)), "drywall": dict(size=(1.2, 0.6, 0.00625), mass=25.0, rgba=(0.92, 0.92, 0.9, 1))}


def sheet_scene(kind, mass=None):
    """G1H + a free sheet + two INACTIVE welds (left/right wrist_yaw_link -> sheet); activated at the searched grip at t=0."""
    from engine import world as W
    sh = SHEETS[kind]; mass = float(mass or sh["mass"]); path = f"{W.SCENE_DIRS['G1H']}/scene_sheet_{kind}_{mass:g}.xml"
    if os.path.exists(path): return path
    spec = mujoco.MjSpec.from_file(W.BASE["G1H"]); b = spec.worldbody.add_body(name="sheet", pos=[1.5, 0, 0.5]); b.add_freejoint(name="sheet_free")
    g = b.add_geom(name="sheet_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(sh["size"]), mass=mass, rgba=list(sh["rgba"])); g.solref = [0.002, 1.0]; g.solimp = [0.99, 0.999, 0.001, 0.5, 2.0]
    for side in ("left", "right"):
        e = spec.add_equality(name=f"grip_{side}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY, name1=f"{side}_wrist_yaw_link", name2="sheet"); e.active = False
    for bd in spec.bodies:      # the welded hands do not collide with the sheet (the weld IS the grip contact)
        if bd.name and ("wrist_yaw_link" in bd.name or "_hand_" in bd.name): spec.add_exclude(bodyname1=bd.name, bodyname2="sheet")
    for k in spec.keys:
        k.qpos = list(k.qpos) + [1.5, 0, 0.5, 1, 0, 0, 0]
        if len(k.qvel): k.qvel = list(k.qvel) + [0.0]*6
    spec.compile(); tmp = f"{path}.{os.getpid()}.tmp"; open(tmp, "w").write(spec.to_xml()); os.replace(tmp, path); return path


def perim(u, a, b):
    """u in [0,1) -> point on the rectangle [-a,a]x[-b,b] perimeter (counter-clockwise from (-a,-b))."""
    L = 4*a + 4*b; s = (u % 1.0)*L
    for (p0, dv, ln) in (((-a, -b), (1, 0), 2*a), ((a, -b), (0, 1), 2*b), ((a, b), (-1, 0), 2*a), ((-a, b), (0, -1), 2*b)):
        if s <= ln: return np.array([p0[0] + dv[0]*s, p0[1] + dv[1]*s, 0.0])
        s -= ln
    return np.array([-a, -b, 0.0])


def _rot(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])


def l2_decode(x, kind):
    """8 strategy numbers in [-1,1] -> grips (sheet frame), sheet rotation, sheet position (grip midpoint at the hand target)."""
    a, b, _ = SHEETS[kind]["size"]; x = np.clip(np.asarray(x, float), -1, 1)
    uA = (x[0] + 1)/2; uB = uA + np.sign(x[1] or 1)*(0.1 + 0.8*abs(x[1]))/(4*a + 4*b)     # second grip: 0.1-0.9 m of arc from the first (reachable spans)
    gA, gB = perim(uA, a, b), perim(uB, a, b)
    R = _rot(x[2]*np.pi/2, x[3]*np.pi/2, x[4]*np.pi/2); tgt = np.array([0.30 + 0.15*x[5], 0.25*x[7], 0.85 + 0.25*x[6]])
    p = tgt - R @ ((gA + gB)/2); wA, wB = R @ gA + p, R @ gB + p
    if wA[1] < wB[1]: gA, gB, wA, wB = gB, gA, wB, wA       # the grip further left goes to the left hand
    return dict(R=R, p=p, grip_local=[gA, gB], grip_world=[wA, wB], u=[float(uA % 1), float(uB % 1)])


def l2_descriptors(D, kind):
    a, b, _ = SHEETS[kind]["size"]; R, p = D["R"], D["p"]; mid = (D["grip_world"][0] + D["grip_world"][1])/2
    n = R[:, 2]; return dict(grip_u=D["u"], grip_sep=float(np.linalg.norm(D["grip_world"][0] - D["grip_world"][1])), sheet_tilt_from_horizontal_deg=float(np.degrees(np.arccos(abs(n[2])))),
                             lever_com_from_grips_h=float(np.linalg.norm((p - mid)[:2])), com_below_grips=float(mid[2] - p[2]), com_xy=p[:2].tolist(),
                             grip_local=[g.round(3).tolist() for g in D["grip_local"]])


def l2_setup(S, x, kind, mass):
    """IK + feasibility + L0 for one strategy. Returns (q_start, D, info) with info['infeasible'] = penalty or None."""
    from engine import leverage as LV
    m = S["m"]; H = LV.g1h_parts(m); D = l2_decode(x, kind); q = np.array(S["q_settled0"])
    q, err = LV.site_ik(m, q, [(H["lpalm"][0], H["lpalm"][1], D["grip_world"][0]), (H["rpalm"][0], H["rpalm"][1], D["grip_world"][1])], LV.ARMS + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"], iters=300)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sheet"); ja = m.jnt_qposadr[m.body_jntadr[bid]]
    qu = np.zeros(4); mujoco.mju_mat2Quat(qu, D["R"].flatten()); q[ja:ja+3] = D["p"]; q[ja+3:ja+7] = qu
    info = dict(ik_err=err, desc=l2_descriptors(D, kind), infeasible=None)
    a, b, c = SHEETS[kind]["size"]; corners = np.array([D["R"] @ [sx*a, sy*b, sz*c] + D["p"] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    if err > 0.03: info["infeasible"] = 20.0 + err; return q, D, info
    if corners[:, 2].min() < 0.03: info["infeasible"] = 15.0 + max(0.0, 0.03 - corners[:, 2].min()); return q, D, info
    d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_forward(m, d); pen = 0.0
    for i in range(d.ncon):
        c_ = d.contact[i]
        if bid in (int(m.geom_bodyid[c_.geom1]), int(m.geom_bodyid[c_.geom2])): pen = max(pen, -float(c_.dist))
    if pen > 0.01: info["infeasible"] = 12.0 + pen; return q, D, info
    r = LV.solve(m, q, (), sorted(S["foot_geoms"]), robot_bodies=S["robot_bodies"], exclude_act=H["fingers"],
                 held=dict(mass=mass, com=D["p"], grips=[(H["lpalm"][0], D["grip_world"][0]), (H["rpalm"][0], D["grip_world"][1])]))
    info["l0"] = dict(max_util=r.get("max_util"), rms_util=r.get("rms_util"), worst=r.get("worst_joint"), status=r["status"]); return q, D, info


_L2 = {}
def l2_plant(kind, mass, dt):
    import throw as TH, branch_i as BI
    key = (kind, mass)
    if key in _L2:
        S = _L2[key]
        if S["m"].opt.timestep != dt: BI.apply_dt(S, dt); S["q_settled0"] = np.array(S["q_settled"])
        return S
    S = TH.setup("G1H", mjcf=sheet_scene(kind, mass), ball=False); BI.apply_dt(S, dt); S["q_settled0"] = np.array(S["q_settled"]); _L2[key] = S; return S


def l2_eval(args):
    """args = (kind, mass, z[2 gains + 7 posture + 8 strategy], T, dt, record). Score (min) = 10 (1 - t_held/T) + util_p95 + 0.5 util_rms;
    infeasible strategies score 12-20+ (penetration / sheet below 3 cm / unreachable grip)."""
    kind, mass, z, T, dt, record = args; z = np.asarray(z, float); S = l2_plant(kind, mass, dt); m = S["m"]
    q, D, info = l2_setup(S, z[9:17], kind, mass)
    if info["infeasible"] is not None: return dict(score=float(info["infeasible"]), t_held=0.0, reason="INFEASIBLE_START", util_p95=None, util_rms=None, **info)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sheet"); gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "sheet_geom"); a, b, c = SHEETS[kind]["size"]
    eqs = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"grip_{s}") for s in ("left", "right")]; hands = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wrist_yaw_link") for s in ("left", "right")]
    def init(d):
        for e, hb in zip(eqs, hands):
            R1 = d.xmat[hb].reshape(3, 3); m.eq_data[e, :3] = 0; m.eq_data[e, 3:6] = R1.T @ (d.xpos[bid] - d.xpos[hb])
            q1 = d.xquat[hb]; qr = np.zeros(4); mujoco.mju_mulQuat(qr, np.array([q1[0], -q1[1], -q1[2], -q1[3]]), d.xquat[bid]); m.eq_data[e, 6:10] = qr; m.eq_data[e, 10] = 1.0
            m.eq_solref[e] = [0.002, 1.0]; m.eq_solimp[e] = [0.99, 0.999, 0.001, 0.5, 2.0]; d.eq_active[e] = 1
    W_ = mass*9.80665; loc = np.array([[sx*a, sy*b, sz*c] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    def cb(t, d):
        d.xfrc_applied[bid, 2] = (1.0 - min(1.0, t/1.0))*W_          # the initial support is withdrawn over 1 s (lift-off)
        if (loc @ d.xmat[bid].reshape(3, 3).T + d.xpos[bid])[:, 2].min() < 0.01: return "SHEET_GROUNDED"
    S["q_settled"] = q; v0 = S["v_settled"]; S["v_settled"] = np.zeros(m.nv)
    try: r = stand_rollout(S, z, T, cb, record=record, init_cb=init)
    finally: S["q_settled"] = S["q_settled0"]; S["v_settled"] = v0
    r.update(info); r["score"] = 10*(1 - r["t_held"]/T) + float(r["util_p95"] if r["util_p95"] is not None else 1.0) + 0.5*float(r["util_rms"]); r["scene"] = sheet_scene(kind, mass)
    return r


def predict_l2(kind, mass, n=400, seed=0, workers=1):
    """L0 over random strategies (posture = settled + IK): returns the feasible candidates ranked by predicted max utilisation."""
    sheet_scene(kind, mass); rng = np.random.default_rng(seed); X = rng.uniform(-1, 1, (n, 8)); out = _pool_map(_pred_one, [(kind, mass, x) for x in X], workers)
    ok = sorted([o for o in out if o.get("l0") and o["l0"]["status"] == "OK"], key=lambda o: o["l0"]["max_util"])
    return dict(n=n, n_feasible=len(ok), top=ok[:10], feasible_x=[o["x"] for o in ok], util_quantiles=np.percentile([o["l0"]["max_util"] for o in ok], [0, 10, 50, 90]).tolist() if ok else None)


def _pred_one(a):
    kind, mass, x = a; S = l2_plant(kind, mass, 0.001); q, D, info = l2_setup(S, x, kind, mass); info["x"] = x.tolist(); return info


def run_l2(a):
    out_dir = f"{RES}/leverage_L2"; os.makedirs(out_dir, exist_ok=True)
    kinds = ["plywood"] if a.smoke else ["plywood", "drywall"]; T = 2.0 if a.smoke else 6.0
    for kind in kinds:
        try:
            mass = float(a.mass or SHEETS[kind]["mass"]); tag = f"{kind}_{mass:g}kg{a.tag}"; log = open(f"{out_dir}/{tag}.log", "w")
            pred = predict_l2(kind, mass, n=64 if a.smoke else 2000, workers=a.workers)
            x0 = np.zeros(17); x0[9:] = pred["top"][0]["x"] if pred["top"] else 0
            prior = l2_eval((kind, mass, x0, T, 0.001, False))            # the PREDICTOR's best grip, prior posture
            # the search starts from a RANDOM feasible strategy (most random grips are unreachable or penetrate), drawn from the feasible
            # set EXCLUDING the predictor's top decile, so the search is not seeded with the predicted answer
            feas = pred.get("feasible_x") or []; rs = np.random.default_rng(7); xs = np.zeros(17)
            if len(feas) > 1: pool = feas[max(1, len(feas)//10):]; xs[9:] = pool[int(rs.integers(len(pool)))]
            res = cma_search(l2_eval, lambda x: (kind, mass, x, T, 0.001, False), 17, a.iters, a.pop, a.workers, log=log, sigma=0.3, x0=xs); res["x0_strategy"] = xs[9:].tolist()
            bz = np.asarray(res["best_z"]); chk = l2_eval((kind, mass, bz, T, 0.0005, False)); rec = l2_eval((kind, mass, bz, T, 0.001, True))
            J = dict(probe="L2_sheet_lift", robot="G1H", kind=kind, mass=mass, T=T, label="weld attachment at two searched grip points; the choice of grip, sheet orientation and posture is searched; initial support withdrawn over 1 s (table/wall not modelled)",
                     predictor=pred, predicted_best_rollout={k: v for k, v in prior.items() if k != "trace"}, search=res, found=dict(desc=rec.get("desc"), l0=rec.get("l0")), check_dt0005={k: v for k, v in chk.items() if k != "trace"})
            json.dump(J, open(f"{out_dir}/{tag}.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
            if rec.get("trace") is not None and len(rec["trace"]):
                np.save(f"{out_dir}/{tag}_trace.npy", rec["trace"])
                if not a.smoke or a.render:
                    from engine import render_any as RA
                    RA.render(rec["scene"], rec["trace"], rec["trace_dt"], f"{out_dir}/{tag}.mp4", title=f"G1H lifts a {mass:g} kg {kind} sheet (grip searched)", sub=f"held {rec['t_held']:.1f}/{T:g} s, util p95 {rec['util_p95']}", distance=5.5, elevation=-25)
            print(tag, "pred_feasible", pred["n_feasible"], "pred_best", pred["top"][0]["l0"]["max_util"] if pred["top"] else None, "pred_rollout", prior["t_held"], prior["reason"],
                  "best", round(res["best_score"], 3), res["best"].get("t_held"), res["best"].get("reason"), "s/eval", round(res["s_per_eval_cpu"], 2), flush=True)
        except Exception:
            import traceback; traceback.print_exc(); print("CONFIG FAILED; continuing", flush=True)


# ------------------------------------------------------------------------------------------------ L3 sled (G1, brace prior)
L3_POSTURE = ["waist_pitch_joint", "right_shoulder_pitch_joint", "right_elbow_joint"]
L3_LEGS = ("hip_pitch_joint", "knee_joint", "ankle_pitch_joint")
SLED_MU = 0.4
_L3 = {}


def l3_plant(mass, dt):
    """G1 + a sled (box on the floor, sliding mu 0.4 by geom priority) with a post whose top is at the settled right palm; the hand is
    WELDED to the post at t=0 (labelled: weld attachment, single right hand)."""
    import throw as TH, branch_i as BI; from engine import world as W
    key = mass
    if key in _L3:
        S = _L3[key]
        if S["m"].opt.timestep != dt: BI.apply_dt(S, dt); S["q_settled0"] = np.array(S["q_settled"])     # setup() caches ONE plant per scene: re-apply dt
        return S
    S0 = TH.setup("G1", ball=False); m0 = S0["m"]; d0 = mujoco.MjData(m0); d0.qpos[:] = S0["q_settled"]; mujoco.mj_forward(m0, d0); palm = d0.site_xpos[S0["site"]].copy()
    hb = 0.12; ph = (palm[2] - hb)/2       # box clear of the feet (x >= palm + 0.2); post + drawbar are NON-colliding (mass only) so nothing touches the robot
    o = W.obj("sled", kind="compound", size=(0.3, 0.15, hb), mass=mass + 1.0, pos=(float(palm[0] + 0.5), float(palm[1]), hb), rgba=(0.5, 0.35, 0.2, 1))
    o["geoms"] = [dict(type="box", size=[0.3, 0.15, hb], pos=[0, 0, 0], mass=float(mass)), dict(type="cylinder", size=[0.02, float(ph)], pos=[-0.48, 0, float(ph)], mass=0.8),
                  dict(type="box", size=[0.09, 0.02, 0.02], pos=[-0.39, 0, 0], mass=0.2)]
    path = W.build_scene("G1", objects=[o]); S = TH.setup("G1", mjcf=path, ball=False); BI.apply_dt(S, dt); m = S["m"]
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if nm == "sled_geom": m.geom_priority[g] = 2; m.geom_friction[g, 0] = SLED_MU
        elif nm.startswith("sled_geom"): m.geom_contype[g] = 0; m.geom_conaffinity[g] = 0
    S["world"] = W.World(m, [o], robot="G1"); S["scene"] = path; S["q_settled0"] = np.array(S["q_settled"]); _L3[key] = S; return S


def l3_eval(args):
    """args = (mass, direction(+1 push / -1 pull), z[2 gains + 3 arm/trunk + 3 symmetric leg offsets], T, dt, record).
    Score (min) = -min(disp, 0.3)/0.3 + 0.2 util_rms + 10 (fell / feet slid)."""
    mass, sgn, z, T, dt, record = args
    if dt == "both":    # the score is the WORSE of 1 ms and 0.5 ms (the stand prior drifts its feet at 0.5 ms with the sled welded)
        r1 = l3_eval((mass, sgn, z, T, 0.001, record)); r2 = l3_eval((mass, sgn, z, T, 0.0005, False))
        r1["dt0005"] = {k: r2[k] for k in ("score", "t_held", "reason", "disp", "com_lean_mean")}; r1["score"] = max(r1["score"], r2["score"]); return r1
    z = np.asarray(z, float); S = l3_plant(mass, dt); m = S["m"]; Wd = S["world"]
    bid = Wd.bid["sled"]; x0 = float(np.asarray(S["q_settled"])[m.jnt_qposadr[m.body_jntadr[bid]]])
    legs = [(actuator_of(m, f"{s}_{j}"), k) for k, j in enumerate(L3_LEGS) for s in ("left", "right")]; loff = 0.3*np.clip(z[5:8], -1, 1)
    fb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in ("left_foot", "right_foot")]; foot0 = {}; com = []
    def extra(t, d, w):
        for a, k in legs: d.ctrl[a] += w*loff[k]
    def init(d):
        if not os.environ.get("LEV_NOWELD"): Wd.grasp(d, "sled")
        foot0.update({i: d.site_xpos[i][:2].copy() for i in fb})
    def cb(t, d):
        fd = max(np.linalg.norm(d.site_xpos[i][:2] - foot0[i]) for i in fb)
        if os.environ.get("LEV_DEBUG") and int(t*1000) % 100 == 0: print("t", round(t, 2), "foot", round(fd, 3), "sled", round(float(d.xpos[bid][0] - x0), 3), "pelvis", np.round(d.qpos[:3], 3))
        if fd > 0.03: return "FEET_SLID"
        if int(t*1000) % 20 == 0: com.append([t, float(d.subtree_com[1][0] - 0.5*(d.site_xpos[fb[0]][0] + d.site_xpos[fb[1]][0])), float(sgn*(d.xpos[bid][0] - x0))])
    zz = np.concatenate([z[:2], np.clip(z[2:5], -1, 1)*sgn**0])
    r = stand_rollout(S, zz, T, cb, record=record, posture=L3_POSTURE, robot="G1", init_cb=init, extra_targets=extra, ramp=1.5, robot_com=True)
    C = np.array(com) if com else np.zeros((1, 3)); disp = float(C[-1, 2]); late = C[C[:, 0] >= 0.5*T] if len(C) > 1 else C
    r.update(disp=disp, com_lean_mean=float(late[:, 1].mean()) if len(late) else None, scene=S["scene"])
    r["score"] = -min(max(disp, -0.3), 0.3)/0.3 + 0.2*float(r["util_rms"]) + (10.0 if r["reason"] else 0.0); return r


def lean_pose(m, q0, phi):
    """The settled pose rigidly leaned by phi (rad, + = top towards +x) about the ankle-pitch axis midpoint, ankle pitch joints
    counter-rotated so the soles stay flat on the floor."""
    d = mujoco.MjData(m); d.qpos[:] = q0; mujoco.mj_forward(m, d); q = np.array(q0)
    js = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_ankle_pitch_joint") for s in ("left", "right")]; A = 0.5*(d.xanchor[js[0]] + d.xanchor[js[1]])
    Ry = np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]]); q[:3] = A + Ry @ (q0[:3] - A)
    qr = np.zeros(4); mujoco.mju_mat2Quat(qr, Ry.flatten()); qn = np.zeros(4); mujoco.mju_mulQuat(qn, qr, np.asarray(q0[3:7])); q[3:7] = qn
    for j in js: q[m.jnt_qposadr[j]] -= phi
    return q


def predict_l3(mass, sgn, phis=np.radians(np.arange(-35, 35.1, 2.5))):
    """(1) The brace prior (max-force posture, direction push/pull) and its capacity; it is constrained to UNLOADED-stable poses
    (support margin >= 2 cm without the load), so it cannot lean into the load. (2) L0 along a rigid ankle-lean family under the
    required sliding force mu M g at the palm: the lean (COM x minus feet centre) that minimises max utilisation is the prediction."""
    import throw as TH; from engine import leverage as LV; import importlib; BR = importlib.import_module("movement_discovery.construction.priors.brace")
    S = TH.setup("G1", ball=False); m = S["m"]; d = mujoco.MjData(m); q0 = np.array(S["q_settled"]); d.qpos[:] = q0; mujoco.mj_forward(m, d)
    bp = BR.brace_prior(S, d.site_xpos[S["site"]].copy(), [sgn, 0, 0], iters=40)
    F = np.array([-sgn*SLED_MU*mass*9.80665, 0, 0]); fb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in ("left_foot", "right_foot")]; rows = []
    for ph in phis:
        q = lean_pose(m, q0, float(ph)); d.qpos[:] = q; mujoco.mj_forward(m, d)
        r = LV.solve(m, q, [LV.point_load(m, d, S["hand_body"], d.site_xpos[S["site"]].copy(), force=F)], sorted(S["foot_geoms"]), robot_bodies=S["robot_bodies"])
        lean = float(d.subtree_com[1][0] - 0.5*(d.site_xpos[fb[0]][0] + d.site_xpos[fb[1]][0]))
        rows.append(dict(phi_deg=float(np.degrees(ph)), lean=lean, max_util=r.get("max_util"), status=r["status"], worst=r.get("worst_joint"), ncon=r.get("ncon")))
    ok = [r for r in rows if r["status"] == "OK"]; best = min(ok, key=lambda r: r["max_util"]) if ok else None
    return dict(F_required=float(abs(F[0])), brace=dict(f_max=bp["f_max"], limiting=bp["limiting"], caps=bp["caps"], lean=None), family=rows, predicted=best)


def run_l3(a):
    out_dir = f"{RES}/leverage_L3"; os.makedirs(out_dir, exist_ok=True)
    masses = [float(a.mass or 40.0)] if a.smoke else [20.0, 40.0, 80.0]; T = 2.0 if a.smoke else 4.0
    for mass in masses:
        for sgn, nm in ((1, "push"), (-1, "pull")) if not a.smoke else ((1, "push"),):
            try:
                tag = f"{nm}_{mass:g}kg{a.tag}"; log = open(f"{out_dir}/{tag}.log", "w"); pred = predict_l3(mass, sgn)
                prior = l3_eval((mass, sgn, np.zeros(8), T, "both", False))
                res = cma_search(l3_eval, lambda x: (mass, sgn, x, T, "both", False), 8, a.iters, a.pop, a.workers, log=log)
                bz = np.asarray(res["best_z"]); chk = l3_eval((mass, sgn, bz, T, 0.0005, False)); rec = l3_eval((mass, sgn, bz, T, 0.001, True))
                J = dict(probe="L3_sled", robot="G1", mass=mass, direction=nm, mu=SLED_MU, T=T, label="right hand welded to the sled post (weld attachment); lean and arm drive searched",
                         predictor=pred, prior={k: v for k, v in prior.items() if k != "trace"}, search=res, found=dict(com_lean_mean=rec["com_lean_mean"], disp=rec["disp"]), check_dt0005={k: v for k, v in chk.items() if k != "trace"})
                json.dump(J, open(f"{out_dir}/{tag}.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
                if rec.get("trace") is not None and len(rec["trace"]):
                    np.save(f"{out_dir}/{tag}_trace.npy", rec["trace"])
                    if not a.smoke or a.render:
                        from engine import render_any as RA
                        RA.render(rec["scene"], rec["trace"], rec["trace_dt"], f"{out_dir}/{tag}.mp4", title=f"G1 {nm}es a {mass:g} kg sled (mu {SLED_MU})", sub=f"moved {rec['disp']:.2f} m, lean {rec['com_lean_mean']}", azimuth=90)
                pl = pred["predicted"]; print(tag, "F_req", round(pred["F_required"]), "pred_lean", pl and round(pl["lean"], 3), "pred_util", pl and round(pl["max_util"], 2), "prior disp", round(prior["disp"], 3), prior["reason"],
                                              "best", round(res["best_score"], 3), "disp", round(res["best"]["disp"], 3), "lean", res["best"]["com_lean_mean"], res["best"]["reason"], "s/eval", round(res["s_per_eval_cpu"], 2), flush=True)
            except Exception:
                import traceback; traceback.print_exc(); print("CONFIG FAILED; continuing", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("probe"); ap.add_argument("--workers", type=int, default=1); ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--pop", type=int, default=16); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--render", action="store_true"); ap.add_argument("--tag", default="")
    ap.add_argument("--mass", type=float, default=None)
    a = ap.parse_args()
    {"L1": run_l1, "L2": run_l2, "L3": run_l3}[a.probe](a)
