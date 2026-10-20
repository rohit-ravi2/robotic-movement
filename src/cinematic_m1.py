"""Cinematic renders of M1_kick_goal R2 results (2026-10-19), studio kit engine/cinematic.py.
capture: replay a stored ledger on FRESH plants (uids 97000/97001, the physics verifier's) at 1 ms (recorded: kick trace, every step of
         the 1 s recovery window, the ball-only flight at 5 ms) and 0.5 ms; check ball speed, goal crossing u, validity, reasons and
         the kick-phase joint-stop overrun against the stored verifier rows (verify_phys/w*.jsonl) and report the ledger's own meas.
render:  kinematic replay of the cache (refused unless the check matched) -> results/demo/cinematic/M1_kick_goal/<tag>.mp4 (+ stills).
The gate environment (KICK_PHYS/REC/FALLGATE/GUARD) is set per item BEFORE importing engine.kick_goal (it reads them at import),
so each capture runs in its own process.
usage (from src/): python cinematic_m1.py capture <tag> | render <tag>"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M1 = f"{ROOT}/results/movements/M1_kick_goal"; CACHE = f"{ROOT}/results/demo/cinematic/cache"; OUT = f"{ROOT}/results/demo/cinematic/M1_kick_goal"
ITEMS = {
    "m1_g5_0_s1": dict(ledger=f"{M1}/r2/searchA/g5_0_s1.json", env=dict(KICK_PHYS="3", KICK_REC="step", KICK_FALLGATE="0", KICK_GUARD="0"), kind="best",
                       robust="robust: 16/16 independent draws (seed base 911000) + 4/4 verifier draws (970000+), both dt"),
    "m1_r1final_g5_0_s1_FAIL": dict(ledger=f"{M1}/r1/final/g5_0_s1_s1.json", env=dict(KICK_PHYS="3", KICK_REC="step", KICK_FALLGATE="1", KICK_GUARD="0"), kind="fail",
                                    robust="R1 finals under R2 gates: 0/12 valid; 24/24 runs JOINT_LIMIT and fall"),
}
TAG = sys.argv[2] if len(sys.argv) > 2 else None
if TAG: os.environ.update(ITEMS[TAG]["env"])
import numpy as np, mujoco


def stored(item):
    for fn in sorted(os.listdir(f"{M1}/verify_phys")):
        if not fn.endswith(".jsonl"): continue
        for line in open(f"{M1}/verify_phys/{fn}"):
            r = json.loads(line)
            if os.path.normpath(r["ledger"]).endswith(os.path.relpath(item["ledger"], M1)) and all(str(r["env"][k]) == str(v) for k, v in
                    dict(phys=item["env"]["KICK_PHYS"], rec=item["env"]["KICK_REC"], fallgate=item["env"]["KICK_FALLGATE"], guard=item["env"]["KICK_GUARD"]).items()):
                return [dict(dt=x["dt"], valid=x["valid"], reasons=x["reasons"], v=x["v"], u=x["u"], lim_kick_deg=x["lim_kick_deg"], rec_fell=x["rec_fell"]) for x in r["nominal"]], f"verify_phys/{fn}"
    raise KeyError(item["ledger"])


def capture(tag):
    from engine import kick_goal as KG, core
    import throw as T
    it = ITEMS[tag]; T.LAG_TAU = KG.LAG; task = KG.GoalKickTask("G1"); L = json.load(open(it["ledger"])); dem = KG.demand_for(L["cfg"]); z = np.asarray(L["z"])
    st_rows, st_src = stored(it); rec_frames = {}
    orig = KG.recovery
    def rec_wrap(S, qpos, qvel, ctrl, **kw):
        out = orig(S, qpos, qvel, ctrl, rec_every=1); rec_frames["F"] = np.asarray(out["frames"]); return out
    KG.recovery = rec_wrap
    rows = []; keep = None
    for i, dt in enumerate((core.DT_OPT, core.DT_CHECK)):
        S = task.fresh_plant(dem, dt, uid=97000 + i); nomS = task.plant(dem, dt, fresh=(i > 0))
        task._refs[(id(S["m"]), tuple(np.round(dem["direction"], 4)), dem.get("capacity_mode", "power"))] = task.ref(nomS, dem)
        r = task.run(S, dem, z, record=(i == 0)); ex = r["extra"]
        rows.append(dict(dt=dt, valid=bool(r["valid"]), reasons=r["reasons"], v=round(float(r["value"]), 3), u=ex.get("goal_u"), z=ex.get("goal_z"), lim_kick_deg=ex.get("lim_kick_deg"),
                         lim_kick_joint=ex.get("lim_kick_joint"), lim_rec_deg=ex.get("lim_rec_deg"), lim_rec_joint=ex.get("lim_rec_joint"), rec_fell=ex.get("rec_fell"),
                         rec_t_fall=ex.get("rec_t_fall"), rec_t_touchdown=ex.get("rec_t_touchdown"), rec_travel=ex.get("rec_travel"), slip=ex.get("slip"), t_cross=ex.get("goal_t_cross")))
        if i == 0: keep = (S, r, rec_frames["F"])
    KG.recovery = orig
    ok = all(a["valid"] == b["valid"] and sorted(a["reasons"]) == sorted(b["reasons"]) and abs(a["v"] - b["v"]) < 2e-3 and abs(a["u"] - b["u"]) < 2e-3
             and abs(a["lim_kick_deg"] - b["lim_kick_deg"]) < 1e-3 and bool(a["rec_fell"]) == bool(b["rec_fell"]) for a, b in zip(rows, st_rows))
    meas = [dict(dt=m_["dt"], valid=m_["valid"], v=round(m_["ball_speed"], 3), u=round(m_["goal_u"], 3)) for m_ in L["result"]["meas"]]
    chk = dict(stored_src=st_src, stored=st_rows, replay=[{k: r_[k] for k in ("dt", "valid", "reasons", "v", "u", "lim_kick_deg", "rec_fell")} for r_ in rows], match=bool(ok),
               ledger_meas=meas, ledger_meas_note="ledger meas was taken on the search's own plant instances; speed differs by plant instance (R1/R2 dt-spread note)")
    S, r, F = keep; m = S["m"]; tr = r["trace"]; dt = float(tr["dt"]); Q = np.asarray(tr["q"]); t_rel = len(Q)*dt
    # per-step joint-stop overrun over the kick and the recovery window (the gate samples the kick every 2nd step, the window every 5th)
    qa, lo, hi, js = KG._lim_setup(S); names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in js]
    def ovr(A): ov = np.maximum(lo[None] - A[:, qa], A[:, qa] - hi[None]); return np.degrees(np.maximum(ov.max(1), 0)), ov.argmax(1)
    okk, okj = ovr(Q); orr, orj = ovr(F)
    # ball-only flight from the t_rel + 0.3 s snapshot, every 5 ms (the gate's own flight, finer record)
    n_cont = int(round(KG.CONT_T/dt)); snapF = F[n_cont - 1] if len(F) >= n_cont else F[-1]
    # the recovery returns the snapshot state; rerun flight from it via a recorded recovery qvel is not kept per frame: re-run recovery once for the snapshot
    cont = orig(S, *r["final_state"], getattr(task, "_last_ctrl", None)); fl = KG.ball_flight(S, dem, cont["qpos"], cont["qvel"], record_every=1)
    flight_check = dict(u_gate=rows[0]["u"], u_fine=fl["u"], z_fine=fl["z"], t_cross=fl["t_cross"], match=bool(abs(fl["u"] - rows[0]["u"]) < 1e-6))
    P = np.asarray(fl["path"]); TP = t_rel + KG.CONT_T + 0.005*np.arange(len(P)); TP[-1] = t_rel + KG.CONT_T + (fl["t_cross"] or 0.005*(len(P) - 1))
    c, n, t = KG.goal_frame(S, dem); mujoco.mj_saveModel(m, f"{CACHE}/{tag}.mjb", None)
    np.savez(f"{CACHE}/{tag}.npz", Q=Q, F=F, P=P, TP=TP, okk=okk, okj=okj, orr=orr, orj=orj)
    sw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "eff_right_foot") if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "eff_right_foot") >= 0 else int(S["site"])
    info = dict(src=os.path.relpath(it["ledger"], ROOT), tag=tag, cfg=L["cfg"], seed=L["seed"], env=it["env"], kind=it["kind"], robust=it["robust"], check=chk, flight_check=flight_check, rows=rows,
                dt=dt, t_rel=t_rel, n_cont=n_cont, rec_frames=len(F), ball_qadr=int(S["ball_qadr"]), ball_geom=int(S["ball_geom"]), goal=dict(c=c.tolist(), n=n.tolist(), t=t.tolist(), **{k: float(v) for k, v in dem["goal"].items()}),
                lim_names=names, kick_overrun_perstep_max=float(okk.max()), kick_overrun_joint=names[int(okj[int(okk.argmax())])], rec_overrun_perstep_max=float(orr.max()) if len(orr) else 0.0,
                rec_overrun_joint=names[int(orj[int(orr.argmax())])] if len(orr) else None, lag_ms=1000*KG.LAG)
    json.dump(info, open(f"{CACHE}/{tag}.json", "w"), indent=1, default=float); print(tag, json.dumps(chk, default=float), flight_check, flush=True); return info


def lerp(a, b, w): return np.asarray(a, float)*(1 - w) + np.asarray(b, float)*w
def ease(x): x = float(np.clip(x, 0, 1)); return x*x*(3 - 2*x)
def cam_lerp(a, b, w): return (lerp(a[0], b[0], w), a[1] + (b[1] - a[1])*w, a[2] + (b[2] - a[2])*w, a[3] + (b[3] - a[3])*w)


def rot_z_to(v):
    v = np.asarray(v, float); v /= np.linalg.norm(v); a = np.cross([0, 0, 1.0], v); s = np.linalg.norm(a)
    if s < 1e-9: return np.eye(3)
    a /= s; c = v[2]; K_ = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]]); return np.eye(3) + s*K_ + (1 - c)*K_ @ K_


def render(tag):
    from engine import cinematic as K
    info = json.load(open(f"{CACHE}/{tag}.json")); assert info["check"]["match"] and info["flight_check"]["match"], f"{tag}: replay does not reproduce the stored value"
    zz = np.load(f"{CACHE}/{tag}.npz"); Q, F, P, TP = zz["Q"], zz["F"], zz["P"], zz["TP"]; okk, okj, orr, orj = zz["okk"], zz["okj"], zz["orr"], zz["orj"]
    st = K.Studio(f"{CACHE}/{tag}.mjb", fog=(30.0, 110.0)); m = st.m; qa = info["ball_qadr"]; m.geom_rgba[info["ball_geom"]] = [0.97, 0.97, 0.97, 1.0]
    dt, t_rel = info["dt"], info["t_rel"]; nQ, nF = len(Q), len(F); T_rob_end = t_rel + nF*dt; t_fl0 = float(TP[0]); t_cross = float(TP[-1])
    g = info["goal"]; c, n, tt_ = np.array(g["c"]), np.array(g["n"]), np.array(g["t"]); Wg, Hg, pr = g["width"], g["height"], g["post_r"]
    fail = info["kind"] == "fail"; r1, r2 = info["rows"]; names = info["lim_names"]
    yaw = float(np.degrees(np.arctan2(n[1], n[0]))); x0 = np.array(Q[0, :2])
    bv = np.linalg.norm(np.diff(Q[:, qa:qa + 3], axis=0), axis=1)/dt; t_imp = float(np.argmax(bv > 0.5)*dt)          # first ball motion > 0.5 m/s
    def robot(t):
        if t <= t_rel: return Q[int(np.clip(round(t/dt) - 1, 0, nQ - 1))]
        return F[int(np.clip(round((t - t_rel)/dt) - 1, 0, nF - 1))]
    def ball(t):
        if t <= t_rel: return Q[int(np.clip(round(t/dt) - 1, 0, nQ - 1)), qa:qa + 3]
        if t < t_fl0: return F[int(np.clip(round((t - t_rel)/dt) - 1, 0, nF - 1)), qa:qa + 3]
        return np.array([np.interp(min(t, t_cross), TP, P[:, k]) for k in range(3)])
    def overrun(t):
        if t <= t_rel: k = int(np.clip(round(t/dt) - 1, 0, nQ - 1)); return float(okk[k]), names[int(okj[k])], "kick"
        k = int(np.clip(round((t - t_rel)/dt) - 1, 0, nF - 1)); return float(orr[k]), names[int(orj[k])], "recovery"
    pL = c + (Wg/2 + pr)*tt_; pR = c - (Wg/2 + pr)*tt_; post_h = Hg + 2*pr; Rbar = rot_z_to(np.r_[pR - pL, 0])
    def goal_geoms():
        w = [0.97, 0.97, 0.97, 1.0]
        G = [("cyl", [pL[0], pL[1], post_h/2], [pr, post_h/2, 0], w), ("cyl", [pR[0], pR[1], post_h/2], [pr, post_h/2, 0], w),
             ("cyl", [c[0], c[1], Hg + pr], [pr, Wg/2 + pr, 0], w, Rbar),
             ("box", [c[0], c[1], 0.002], [0.04, Wg/2 + pr, 0.002], [0.97, 0.97, 0.97, 0.9], rot_z_to([0, 0, 1]) @ np.array([[n[0], -n[1], 0], [n[1], n[0], 0], [0, 0, 1]]))]   # goal line
        return G
    hist = []; t_peak = [None]
    def geoms(t, b):
        G = goal_geoms(); Pp = np.array(hist[-18:])
        for i, p in enumerate(Pp[:-1]): a = (i + 1)/len(Pp); G.append(("sphere", p, [0.06*(0.3 + 0.7*a)]*3, [1.0, 0.62, 0.22, 0.55*a**1.5]))
        if t > t_rel and len(hist) > 18: G += [("sphere", [p[0], p[1], 0.004], [0.02, 0.02, 0.004], [1.0, 0.55, 0.2, 0.35]) for p in hist[:-18:4]]   # ground track so far
        return G
    who = "Unitree G1 (stock Menagerie MJX)  ·  kick a 0.43 kg football into a 2 x 1 m goal"
    where = f"{g['D']:.0f} m " + ("straight" if abs(g["az"]) < 0.01 else f"{np.degrees(g['az']):.0f} deg")
    if fail:
        value = f"FAILS R2 gates: {r1['lim_kick_deg']:.1f} deg past the {r1['lim_kick_joint'].replace('_joint', '').replace('_', ' ')} stop; falls"
        valid = f"scores at 1 ms and 0.5 ms (u {r1['u']:+.2f} / {r2['u']:+.2f} m) but INVALID at both: {', '.join(r1['reasons'])} ({r1['lim_kick_deg']:.2f} / {r2['lim_kick_deg']:.2f} deg)"
        ttl = (f"FAILURE CASE  ·  R1 final {where} under the R2 gates", f"replayed from {info['src'].replace('results/movements/M1_kick_goal/', 'M1_kick_goal/')} (fresh plant; capture-step recovery)")
        cav = [info["robust"], "actuator lag 10 ms", "conservative ratings: knee 120, ankle/waist roll-pitch 35, hip roll 88 N·m",
               "LABELLED attachment: 4 cm effector sphere at the kicking-foot site", "ball flight after t_rel+0.3 s: ball-only MuJoCo model with air drag",
               "robot shown until its first non-foot floor contact, then held", "stock contact; mid-phase off so the ball meets every body geom"]
    else:
        value = f"GOAL  ·  ball {r1['v']:.2f} m/s, crosses {r1['u']:+.2f} m from centre, rolling"
        valid = f"valid at 1 ms and 0.5 ms  ({r1['v']:.2f} / {r2['v']:.2f} m/s; kick-phase stop overrun {r1['lim_kick_deg']:.2f} / {r2['lim_kick_deg']:.2f} deg, gate 1 deg)"
        ttl = (f"KICK INTO A GOAL  ·  {where}  ·  the one robust config/seed (1/12)", f"replayed from {info['src']} on a fresh plant (verifier uid)")
        cav = [info["robust"], "NOT a standing kick: the robot FALLS after release (fall reported, not gated)",
               f"stop gate covers the kick phase only: {info['rec_overrun_perstep_max']:.2f} deg ({info['rec_overrun_joint'].replace('_joint', '').replace('_', ' ')}) in the recovery window",
               "actuator lag 10 ms", "conservative ratings: knee 120, ankle/waist roll-pitch 35, hip roll 88 N·m", "LABELLED attachment: 4 cm effector sphere at the kicking-foot site",
               "ball flight after t_rel+0.3 s: ball-only MuJoCo model with air drag; ends at the goal plane", "robot shown until its first non-foot floor contact, then held", "stock contact; mid-phase off"]
    def frame(t, fac, cam, extra_note=None):
        b = ball(t); hist.append(np.array(b, float)); q = robot(min(t, T_rob_end)).copy(); q[qa:qa + 3] = b; q[qa + 3:qa + 7] = [1, 0, 0, 0]; st.set(q)
        im = K.finish(st.frame(*cam, geoms=geoms(t, b), key_target=[x0[0] + 0.3*n[0], x0[1] + 0.3*n[1], 0.6]))
        im = K.title(im, *ttl); im = K.lower_third(im, who, value, valid, cav, scale=0.78); im = K.badge(im, fac, t)
        ov, jn, ph = overrun(min(t, T_rob_end - 1e-9)); col = K.WARN if ov > 1.0 else ((255, 214, 140) if ov > 0 else (150, 220, 160))
        im = K.note(im, f"deepest joint-stop overrun now: {ov:4.2f} deg  {jn.replace('_joint', '').replace('_', ' ')}  ({ph}; gate 1 deg" + (")" if ph == "kick" or fail else ", not gated here)"), (40, 92), color=col, size=16)
        y = 130
        if t >= t_imp and t < t_rel: im = K.note(im, f"ball struck at t = {t_imp:.3f} s (first ball speed > 0.5 m/s)", (40, y), color=K.ACC, size=16); y += 38
        if t >= t_rel: im = K.note(im, f"kick phase ends (t_rel = {t_rel:.3f} s)  ·  ball {r1['v']:.2f} m/s; recovery law takes over", (40, y), color=K.ACC, size=16); y += 38
        if t >= T_rob_end and nF < int(round(1.0/dt)): im = K.note(im, f"robot down at t = {T_rob_end:.2f} s (non-foot floor contact): simulation of the robot stops, pose held", (40, y), color=K.WARN, size=16); y += 38
        if t >= t_cross - 1e-9: im = K.note(im, f"GOAL: crosses the line at u = {r1['u']:+.2f} m, z = {r1['z']:.2f} m (flight model stops at the goal plane)", (40, y), color=(150, 220, 160), size=16); y += 38
        if extra_note: im = K.note(im, extra_note, (40, y), color=(140, 200, 255), size=16)
        return im
    W = K.Writer(f"{OUT}/{tag}.mp4"); stills = {}
    ck = ([x0[0] + 0.35*n[0], x0[1] + 0.35*n[1], 0.35], 3.0, yaw + 75.0, -10.0)                       # kick: side view from the kicking-leg side
    behind = ([0.55*(x0[0] + c[0]), 0.55*(x0[1] + c[1]), -0.35], 0.75*g["D"] + 2.2, yaw + 30.0, -13.0)    # ball run: behind-right, robot + goal in frame
    W.add(frame(0.0, 1.0, cam_lerp(behind, ck, 0.0)), hold=20)
    for k in range(30): W.add(frame(0.0, 1.0, cam_lerp(behind, ck, ease(k/29))))
    W.add(frame(0.0, 1.0, ck), hold=10)
    tl = K.timeline(t_cross, windows=[(0.0, t_imp - 0.06, 0.25), (t_imp - 0.06, t_imp + 0.08, 0.08), (t_imp + 0.08, t_cross, 0.4)], base=0.4)
    for t, fac in tl:
        w = ease((t - (t_imp + 0.1))/0.7); cam = cam_lerp(ck, behind, w); im = frame(t, fac, cam); W.add(im)
        if "impact" not in stills and t >= t_imp + 0.02: stills["impact"] = im
        if "overrun" not in stills and t >= (int(np.argmax(okk)) + 1)*dt: stills["overrun"] = im
    im = frame(t_cross, 1.0, behind); W.add(im, hold=60); stills["goal"] = im
    # second pass: the robot after release (the fall is part of the result)
    hist.clear(); crob = ([x0[0] + 0.3*n[0], x0[1] + 0.3*n[1], 0.05], 3.6, yaw + 120.0, -14.0)
    for k in range(25): W.add(frame(t_cross, 1.0, cam_lerp(behind, crob, ease(k/24)), "replay: the robot after release"))
    for t, fac in K.timeline(min(t_rel + 1.0, T_rob_end + 0.3), windows=[], base=0.5, t_start=t_rel - 0.05):
        im = frame(t, fac, crob, "replay: the robot after release (capture-step recovery law)"); W.add(im)
    stills["after"] = im; W.add(im, hold=45)
    out, dur = W.close()
    for k, v in stills.items(): v.convert("RGB").save(f"{OUT}/{tag}_{k}.jpg", quality=88)
    print(out, f"{dur:.1f} s", flush=True); return out, dur


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
    {"capture": capture, "render": render}[sys.argv[1]](sys.argv[2])
