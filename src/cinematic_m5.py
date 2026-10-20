"""Cinematic renders of M5_walking repair round 2 (2026-10-19), studio kit engine/cinematic.py.
Rendered (all verified by the physics lens, VERIFY_physics.md, commit 0e3c421):
  m5_G1_075_r2b          G1 0.75 m/s _r2b nominal walk: robust (F1-F4 4/4; 11/12 with N1-N8; verifier N9-N12 3/4)
  m5_H1_100_r2b          H1 1.0 m/s _r2b nominal walk: robust (F1-F4 4/4; 11/12; N9-N12 4/4)
  m5_G1_075_r2b_F4_shove G1 _r2b under the fresh held-out F4 (+y lateral shove 0.1 BW, 0.2 s, cycle 4.90): recovers
  m5_G1_075_r2b_N12_FAIL G1 _r2b under the verifier's N12 (-y lateral shove 0.1 BW, 0.2 s, cycle 4.06): FALLS (failure case)
capture: one FRESH process per (clip, dt), through the physics verifier's own entry point (verify_physics_r2.job, same perturbation
         code, same scratch scaffold cache); W.Walk.run is wrapped only to keep the recorded trace. The replayed summary + independent
         trace checks are compared field by field with the verifier's stored row (verify_runs_r2.json).
render:  kinematic replay of the 1 ms record (refused unless both dt matched) -> results/demo/cinematic/M5_walking/<tag>.mp4 (+ stills).
usage (from src/): python cinematic_m5.py capture <tag> <1|2> | render <tag>"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M5 = f"{ROOT}/results/movements/M5_walking"; VR = f"{M5}/verify/r2"; CACHE = f"{ROOT}/results/demo/cinematic/cache"; OUT = f"{ROOT}/results/demo/cinematic/M5_walking"
ITEMS = {"m5_G1_075_r2b": ("G1_0.75_r2b", "nominal"), "m5_H1_100_r2b": ("H1_1.0_r2b", "nominal"),
         "m5_G1_075_r2b_F4_shove": ("G1_0.75_r2b", "F4"), "m5_G1_075_r2b_N12_FAIL": ("G1_0.75_r2b", "N12")}
import numpy as np, mujoco


def _cmp(a, b, path=""):
    """Field-by-field comparison (numbers to 1e-4, which is the stored rounding); returns a list of mismatches."""
    bad = []
    if isinstance(b, dict):
        for k in b:
            if k in ("secs",): continue
            if k not in a: bad.append(f"{path}{k} missing"); continue
            bad += _cmp(a[k], b[k], f"{path}{k}.")
    elif isinstance(b, (list, tuple)) and not isinstance(b, str):
        if len(a) != len(b): return [f"{path} len {len(a)} != {len(b)}"]
        for i, (x, y) in enumerate(zip(a, b)): bad += _cmp(x, y, f"{path}{i}.")
    elif isinstance(b, bool) or b is None or isinstance(b, str):
        if a != b: bad.append(f"{path} {a} != {b}")
    elif isinstance(b, (int, float)):
        if not isinstance(a, (int, float)) or abs(float(a) - float(b)) > 1.01e-4: bad.append(f"{path} {a} != {b}")
    return bad


def capture(tag, dtk):
    sys.argv = [sys.argv[0]]; sys.path.insert(0, VR)
    import verify_physics_r2 as V
    from engine import walk as W
    REC = {}; orig = W.Walk.run
    def run(self, z, dt=None, **kw):
        r = orig(self, z, dt, **kw); REC.update(P=self, dt=dt, r=r, z=z); return r
    W.Walk.run = run
    gname, pert = ITEMS[tag]; dtk = int(dtk)
    _, _, _, out = V.job((gname, pert, dtk))
    stored = json.load(open(f"{VR}/verify_runs_r2.json"))[f"{gname}|{pert}|{dtk}"]
    jo = json.loads(json.dumps(out, default=lambda v: v.tolist() if hasattr(v, "tolist") else float(v)))
    bad = _cmp(jo, stored)
    P, dt, r = REC["P"], REC["dt"], REC["r"]; S = P.S[dt]; m = S["m"]; p = P.unpack(np.asarray(REC["z"], float))
    import branch_i as BI
    ankb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in BI.ROBOTS[S["robot"]]["foot_bodies"]]
    shove = dict(N12=(0.0, -0.1, 4.06, 0.2), F4=(0.0, 0.1, 4.90, 0.2), N11=(0.1, 0.0, 5.6, 0.2)).get(pert)
    cyc = float(P.R["Tc"]*p["tscale"])
    tr = r["trace"]; base = f"{CACHE}/{tag}_dt{dtk}"
    np.savez(base + ".npz", T=np.asarray(tr["t"]), Q=np.asarray(tr["qpos"]), COM=np.asarray(tr["com"]), LOAD=np.asarray(tr["load"]))
    mujoco.mj_saveModel(m, base + ".mjb", None)
    bnd = r.get("bounds", [])
    info = dict(tag=tag, gait=gname, pert=pert, dt=float(m.opt.timestep), match=not bad, mismatches=bad, replay=jo, stored=stored,
                stored_src="results/movements/M5_walking/verify/r2/verify_runs_r2.json", robot=S["robot"], v_cmd=jo["v_cmd"],
                cycle_s=cyc, shove=None if shove is None else dict(f=shove[:2], cycle=shove[2], dur=shove[3], t0=shove[2]*cyc),
                fb=[ankb[1], ankb[0]], hf=float(P.R["sole"]["hf"]), td_all=r.get("td_all", []),
                t_win=[bnd[W.N_TRANS]["t"], bnd[-1]["t"]] if len(bnd) > W.N_TRANS + 1 else None, lag=W.LAG,
                mass=float(jo["indep"]["robot_mass"]))
    json.dump(info, open(base + ".json", "w"), indent=1, default=lambda v: v.tolist() if hasattr(v, "tolist") else float(v))
    print(tag, dtk, "MATCH" if not bad else f"MISMATCH {bad[:6]}", jo["value"], jo["valid"], jo["reasons"], flush=True)


def ease(x): x = float(np.clip(x, 0, 1)); return x*x*(3 - 2*x)


VER = json.load(open(f"{M5}/G1/search_v0.75_r2b_valR2.json")), json.load(open(f"{M5}/H1/search_v1.0_r2b_valR2.json"))
RAT = {"G1": "ratings (Unitree published): hip 88, knee 120, ankle 35, waist yaw 88 / roll, pitch 35, arm 25 N·m",
       "H1": "ratings = MJCF limits (Unitree): hip 200, knee 300, ankle 40, torso 200, shoulder 40, elbow 18 N·m"}
SOLV = {"G1": "solver Newton 50 it (MJCF: 5)", "H1": "solver Newton 50 it (MJCF: 100)"}


def render(tag):
    from engine import cinematic as K
    I1, I2 = (json.load(open(f"{CACHE}/{tag}_dt{k}.json")) for k in (1, 2))
    assert I1["match"] and I2["match"], f"{tag}: replay does not reproduce the stored verifier row: {I1['mismatches'][:4]} {I2['mismatches'][:4]}"
    zz = np.load(f"{CACHE}/{tag}_dt1.npz"); T, Q, COM, LOAD = zz["T"], zz["Q"], zz["COM"], zz["LOAD"]
    st = K.Studio(f"{CACHE}/{tag}_dt1.mjb", fog=(14.0, 60.0)); m = st.m
    robot = I1["robot"]; r1, r2 = I1["replay"], I2["replay"]; pert = I1["pert"]; cyc = I1["cycle_s"]; t_end = float(T[-1])
    V = VER[0] if robot == "G1" else VER[1]
    zc = 0.62 if robot == "G1" else 0.95; dist = 3.3 if robot == "G1" else 4.4
    who = f"Unitree {robot} (stock Menagerie model)"
    # ---------- labels
    common = ["10 ms first-order actuator lag (plant)", RAT[robot], "stock MJCF contact: solref 0.02, mu 1, condim 3", SOLV[robot],
              "construction: spec -> LIPM scaffold -> CMA search (stride amp, time scale, lean, 13 controller gains; seed >= 1); no choreography"]
    gates = "slip <= 2 cm/stance, double support >= 20 ms every step, no flight, 10 periodic strides, speed +-5 %"
    if robot == "G1":
        rob = [f"ROBUST: fresh held-out F1-F4 {V['fresh_pass']}/4 at both dt (needs 3/4); 11/12 with N1-N8", "verifier N9-N12: 3/4 (falls on the N12 -y shove)",
               f"lateral shove recovery depends on stride phase (sweep {V['lat_sweep_pass']}/{V['lat_sweep_n']})", "thin margin: N3 double support 21.5 ms vs the 20 ms gate",
               "spec ceiling 1.00 m/s over-predicts the fastest robust walk 1.39x"]
    else:
        rob = [f"ROBUST: fresh held-out F1-F4 {V['fresh_pass']}/4 at both dt (needs 3/4); 11/12 with N1-N8 (N2 slips 3.9 cm)", "verifier N9-N12: 4/4",
               "right ankle at 100 % of its 40 N·m rating on every run", "thin margin: N8 slip 1.96-1.98 cm vs the 2 cm gate",
               f"lateral shove sweep {V['lat_sweep_pass']}/{V['lat_sweep_n']}", "spec ceiling 1.60 m/s over-predicts 1.62x"]
    task = f"{who}  ·  walk straight, commanded {I1['v_cmd']:.2f} m/s (gait _r2b, repair round 2)"
    if pert == "nominal":
        value = f"{r1['value']:.3f} m/s net  ·  {r1['cadence_spm']:.1f} steps/min  ·  {r1['J_per_m']:.1f} J/m (CoT {r1['cot']:.2f})"
        valid = f"valid at 1 ms and 0.5 ms  (net {r1['value']:.3f} / {r2['value']:.3f} m/s;  {gates})"
        cav = rob + common
        ttl = (f"{robot} WALK  ·  {r1['value']:.3f} m/s, robust", f"replayed on fresh plants through the physics verifier (verify_physics_r2); shown: the 1 ms run")
        windows = [(6*cyc, 7*cyc, 0.25)]; t_stop = t_end
    else:
        sh = I1["shove"]; ts0 = sh["t0"]; sgn = "+y" if sh["f"][1] > 0 else "-y"
        if pert == "F4":
            value = f"RECOVERS  ·  {sgn} shove 0.1 BW for 0.2 s at cycle {sh['cycle']:.2f}  ·  net {r1['value']:.3f} m/s"
            valid = f"valid at 1 ms and 0.5 ms under the shove (net {r1['value']:.3f} / {r2['value']:.3f} m/s; energy-closure gate waived: external work)"
            ttl = ("LATERAL SHOVE  ·  fresh held-out perturbation F4", "registered in PREDICTION_R2.md before any repair-2 run; run once; re-run here through the verifier's code")
            cav = ["F4 is one of 4/4 fresh draws passed", "the same gait FALLS on a -y shove at cycle 4.06 (N12, next clip)"] + rob[2:3] + common
            windows = [(ts0 - 0.3, ts0 + 1.5, 0.25), (ts0 + 3.5, t_end, 2.0)]; t_stop = t_end
        else:
            value = f"FALLS  ·  {sgn} shove 0.1 BW for 0.2 s at cycle {sh['cycle']:.2f}  ·  fall detected t = {r1['t_run']:.2f} s"
            valid = f"INVALID at 1 ms and 0.5 ms ({', '.join(r1['reasons'])} / {', '.join(r2['reasons'])}; falls at {r1['t_run']:.2f} / {r2['t_run']:.2f} s)"
            ttl = ("FAILURE CASE  ·  G1 _r2b under the verifier's N12 lateral shove", "N12 phase and sign drawn once from rng(1312); the gait still counts robust: 3/4 of N9-N12, 4/4 of F1-F4")
            cav = ["lateral push recovery depends on when in the stride the shove lands", f"phase sweep: {V['lat_sweep_pass']}/{V['lat_sweep_n']} pass (N8, -y at 3.75, also falls)",
                   "rollout stops at the fall detector; last pose held"] + common
            windows = [(ts0 - 0.3, t_end, 0.25)]; t_stop = t_end
    # ---------- per-frame helpers
    def at(A, t): return K.track_q(A, T, t)
    fb, hf = I1["fb"], I1["hf"]; dk = mujoco.MjData(m)
    prints = []   # footprints at each loaded-onset (visual only): (t, leg, xy, yaw)
    on_prev = [False, False]
    for i, t in enumerate(T):
        for li in (0, 1):
            on = LOAD[i, li] > 0.05
            if on and not on_prev[li]:
                dk.qpos[:] = Q[i]; mujoco.mj_kinematics(m, dk); R_ = dk.xmat[fb[li]].reshape(3, 3); c = dk.xpos[fb[li]] - hf*R_[:, 2]
                prints.append((float(t), li, c[:2].copy(), float(np.arctan2(R_[1, 0], R_[0, 0]))))
            on_prev[li] = on
    LEGC = {0: [0.30, 0.62, 1.00, 0.85], 1: [1.00, 0.70, 0.25, 0.85]}
    sh = I1["shove"]
    def geoms(t):
        g = []
        for (tp, li, xy, yaw) in prints:
            if tp > t: break
            a = 0.85*max(0.25, 1 - (t - tp)/8.0); c, s_ = np.cos(yaw), np.sin(yaw); Rm = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
            g.append(("box", [xy[0], xy[1], 0.0015], [0.10, 0.04, 0.0015], LEGC[li][:3] + [a], Rm))
        cm = at(COM, t); g.append(("sphere", [cm[0], cm[1], 0.003], [0.025, 0.025, 0.003], [0.3, 0.75, 1.0, 0.8], None))
        return g
    def arrow(scn, t):
        if sh is None or not (sh["t0"] <= t <= sh["t0"] + sh["dur"]): return
        cm = at(COM, t); d_ = np.array([sh["f"][0], sh["f"][1], 0.0]); d_ /= np.linalg.norm(d_)
        p1 = np.array([cm[0], cm[1], cm[2]]) - 0.12*d_; p0 = p1 - 0.6*d_
        if scn.ngeom >= scn.maxgeom: return
        gg = scn.geoms[scn.ngeom]; mujoco.mjv_initGeom(gg, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array([1.0, 0.25, 0.2, 1.0], np.float32))
        mujoco.mjv_connector(gg, mujoco.mjtGeom.mjGEOM_ARROW, 0.035, p0, p1); scn.ngeom += 1
    def cam(t):
        c = at(COM, min(t, t_end)); az = 120.0 - 25*ease((t - 1.0)/4.0)
        if sh is not None: az = az + (178 - az)*ease((t - sh["t0"] + 1.0)/0.8)*(1 - ease((t - sh["t0"] - 2.5)/1.0))
        return [c[0], c[1]*0.6, zc], dist, az, -10.0
    def frame(t, fac):
        tt = min(t, t_end); st.set(at(Q, tt)); la, d0, az, el = cam(t)
        # render with footprints + arrow (Studio.frame handles the rig; the arrow is added through a geom hook)
        gl = [(k, p_, s_, rg, mt) for (k, p_, s_, rg, mt) in geoms(tt)]
        img = st.frame(la, d0, az, el, geoms=[(k, p_, s_, rg, mt) for (k, p_, s_, rg, mt) in gl], key_target=[la[0], la[1], 0.8])
        im = K.finish(img); im = K.title(im, *ttl); im = K.lower_third(im, task, value, valid, cav, scale=0.72); im = K.badge(im, fac, t)
        ld = at(LOAD, tt); y = 92; Lr, Ll = float(ld[0]), float(ld[1]); ds = Lr > 0.02 and Ll > 0.02; air = Lr*I1["mass"]*9.81 < 1 and Ll*I1["mass"]*9.81 < 1
        lab = "DOUBLE SUPPORT" if ds else ("NO SUPPORT" if air else "single support")
        im = K.note(im, f"foot load (x body weight)   R {Lr:4.2f}   L {Ll:4.2f}   {lab}      footprints: blue right, orange left", (40, y), color=(150, 220, 160) if ds else ((255, 120, 100) if air else (190, 198, 210)), size=16); y += 38
        if I1["t_win"] and tt >= I1["t_win"][0] and r1["valid"]:
            im = K.note(im, f"measured window from t = {I1['t_win'][0]:.2f} s (after 2 transient cycles): net {r1['value']:.3f} m/s", (40, y), color=(140, 200, 255), size=16); y += 38
        if sh is not None and t >= sh["t0"]:
            im = K.note(im, f"shove {'+y' if sh['f'][1] > 0 else '-y'} 0.1 BW ({0.1*I1['mass']*9.81:.0f} N) for 0.2 s at t = {sh['t0']:.2f} s (cycle {sh['cycle']:.2f})",
                        (40, y), color=K.WARN, size=16); y += 38
        if pert == "N12" and t >= t_end - 0.02:
            im = K.note(im, f"fall detected t = {t_end:.2f} s: rollout stops, pose held", (40, y), color=K.WARN, size=16)
        return im
    # arrow: wrap add of user geoms by patching K.add_geom-driven list is not enough (arrow needs a connector) -> hook Renderer.update_scene
    upd = st.r.update_scene; cur = {"t": 0.0}
    def upd2(*a, **k):
        upd(*a, **k); arrow(st.r.scene, cur["t"])
    st.r.update_scene = upd2
    Wr = K.Writer(f"{OUT}/{tag}.mp4"); stills = {}
    cur["t"] = 0.0; Wr.add(frame(0.0, 1.0), hold=30)
    for t, fac in K.timeline(t_stop, windows=windows, base=1.0):
        cur["t"] = min(t, t_end); im = frame(t, fac); Wr.add(im)
        if pert == "nominal":
            if "ds" not in stills and t >= 6*cyc and Lr_ds(LOAD, T, t): stills["ds"] = im
            if "walk" not in stills and t >= 3.0: stills["walk"] = im
        else:
            if "shove" not in stills and t >= sh["t0"] + 0.1: stills["shove"] = im
            if "after" not in stills and t >= sh["t0"] + (0.8 if pert == "F4" else min(1.2, t_end - sh["t0"] - 0.05)): stills["after"] = im
    stills["end"] = im; Wr.add(im, hold=90)
    out, dur = Wr.close()
    for k, v in stills.items(): v.convert("RGB").save(f"{OUT}/{tag}_{k}.jpg", quality=88)
    print(out, f"{dur:.1f} s", flush=True)


def Lr_ds(LOAD, T, t):
    from engine import cinematic as K
    ld = K.track_q(LOAD, T, t); return ld[0] > 0.02 and ld[1] > 0.02


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
    if sys.argv[1] == "capture": capture(sys.argv[2], sys.argv[3])
    else: render(sys.argv[2])
