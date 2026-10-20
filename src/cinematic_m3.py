"""Cinematic renders of M3_box_jump R2 results (2026-10-19), studio kit engine/cinematic.py.
Nothing in M3 R2 is robust; what the physics verifier (verify_physics_r2) confirmed, and what is rendered here:
  m3_land_vx030      flat-floor landing probe, 0.20 m drop at vx 0.30 m/s: the top of the unbroken range held at both dt (nominal);
                     NOT robust (2/4 new perturbations).
  m3_land_vx050_HOP  the same probe at 0.50 m/s: passes the hold gates nominally but is a HOP (~0.1 s with zero floor force), the
                     R2 headline the verifier refuted as characterised; 0/4 perturbations. Labelled as such.
  m3_v10_D025_s7_FAIL the v10 box search (h 0.20, D 0.25, seed 7): invalid at both dt, 0/4; labelled FAILURE CASE.
capture: replay on FRESH plants at 1 ms and 0.5 ms through the verifier's own entry points (verify_r2.probe / verify_r2 env + BJ.Ctx),
         recording qpos and floor/box normal force every physics step; check the replayed values against the verifier's stored rows.
render:  kinematic replay of the 1 ms record (refused unless the check matched) -> results/demo/cinematic/M3_box_jump/<tag>.mp4 (+ stills).
usage (from src/): python cinematic_m3.py capture <tag> | render <tag>"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M3 = f"{ROOT}/results/movements/M3_box_jump"; VP = f"{M3}/verify_physics_r2"; CACHE = f"{ROOT}/results/demo/cinematic/cache"; OUT = f"{ROOT}/results/demo/cinematic/M3_box_jump"
ITEMS = {"m3_land_vx030": dict(kind="land", vx=0.30, pert="pert_030"), "m3_land_vx050_HOP": dict(kind="hop", vx=0.50, pert="pert_050"),
         "m3_v10_D025_s7_FAIL": dict(kind="fail", rung="G1_D025_v10s7")}
import numpy as np, mujoco

REC = dict(on=False, rows=[], m=None)
_orig_step = mujoco.mj_step
def _step(m, d):
    _orig_step(m, d)
    if not REC["on"]: return
    if REC["m"] is not m: REC["m"] = m; REC["ids"] = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor"), mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box"))
    fl, bx = REC["ids"]; ff = fb = 0.0; fv = np.zeros(6)
    for ci in range(d.ncon):
        c = d.contact[ci]
        if fl in (c.geom1, c.geom2): mujoco.mj_contactForce(m, d, ci, fv); ff += fv[0]
        elif bx in (c.geom1, c.geom2): mujoco.mj_contactForce(m, d, ci, fv); fb += fv[0]
    REC["rows"].append((float(d.time), ff, fb, d.qpos.copy(), d.subtree_com[0].copy()))
mujoco.mj_step = _step


def last_segment(rows):
    t = np.array([r[0] for r in rows]); br = np.where(np.diff(t) < 0)[0]; return rows[br[-1] + 1:] if len(br) else rows


def stored_land(vx, dt):
    for r in json.load(open(f"{VP}/grid_{'1ms' if dt == 0.001 else '05ms'}.json"))["result"]:
        if abs(r["vx"] - vx) < 1e-9: return r
    raise KeyError(vx)


def capture(tag):
    sys.argv = [sys.argv[0], "none"]; sys.path.insert(0, VP)
    import verify_r2 as V
    it = ITEMS[tag]; BJ = V.BJ; keep = None; rows = []
    if it["kind"] in ("land", "hop"):
        for dt in (0.001, 0.0005):
            REC["rows"] = []; REC["on"] = True; r = V.probe(it["vx"], dt); REC["on"] = False
            seg = last_segment(REC["rows"]); st = stored_land(it["vx"], dt)
            ff = np.array([s[1] for s in seg]); tt = np.array([s[0] for s in seg]); td = float(tt[np.argmax(ff >= 1.0)])
            zero = (ff < 1.0) & (tt > td); run = best = 0
            for zz in zero: run = run + 1 if zz else 0; best = max(best, run)
            rows.append(dict(dt=dt, ok=r["ok"], fell=r["fell"], foot_travel=r["foot_travel"], pelvis_drift=r["pelvis_drift"], hops=r["flight_phases_after_touch"], slide=r["slide"],
                             step=r["step"], touchdown_t=td, longest_zero_floor_force_s=best*dt,
                             stored=dict(ok=st["ok"], fell=st["fell"], foot_travel=st["foot_travel"], pelvis_drift=st["pelvis_drift"], hops=st["flight_phases_after_touch"], slide=st["slide"])))
            if dt == 0.001: keep = (seg, REC["m"])
        match = all(a["ok"] == a["stored"]["ok"] and a["fell"] == a["stored"]["fell"] and a["hops"] == a["stored"]["hops"] and
                    all(abs(a[k] - a["stored"][k]) < 1e-6 for k in ("foot_travel", "pelvis_drift", "slide")) for a in rows)
        fc = [x for x in json.load(open(f"{VP}/flightcheck.json")) if abs(x["vx"] - it["vx"]) < 1e-9]
        fmatch = all(abs(a["longest_zero_floor_force_s"] - b["longest_zero_floor_force_s"]) < 1e-9 for a, b in zip(rows, fc)) if fc else None
        P = json.load(open(f"{VP}/{it['pert']}.json"))["result"]; labs = sorted({p["label"] for p in P})
        pert = {l: [bool(p["ok"]) for p in P if p["label"] == l] for l in labs}; npass = sum(all(v) for v in pert.values())
        info = dict(tag=tag, kind=it["kind"], vx=it["vx"], rows=rows, match=bool(match), flight_match=fmatch, flight_stored=fc, pert=pert, pert_pass=npass, pert_n=len(labs),
                    stored_src=[f"verify_physics_r2/grid_1ms.json", "verify_physics_r2/grid_05ms.json", "verify_physics_r2/flightcheck.json", f"verify_physics_r2/{it['pert']}.json"],
                    settle_t=BJ.SETTLE_T, hold_t=BJ.HOLD_T, travel_max=BJ.FOOT_TRAVEL_MAX, drift_max=BJ.PELVIS_DRIFT_MAX)
    else:
        rj = json.load(open(f"{M3}/{it['rung']}/rung_h0.20.json")); ref = BJ.ref_for(rj); z = np.asarray(rj["z"], float)
        c = BJ.Ctx(rj["robot"], rj["h"], rj["D"], om=rj["om"], ref=ref); stv = [x for x in json.load(open(f"{VP}/rung_{it['rung']}.json"))["result"]["runs"] if x["label"] == "nominal"]
        for dt, sv in zip((0.001, 0.0005), stv):
            assert sv["dt"] == dt
            REC["rows"] = []; REC["on"] = True; o = c.run(z, dt); REC["on"] = False; seg = last_segment(REC["rows"]); s = o.get("step") or {}
            rows.append(dict(dt=dt, valid=bool(o["valid"]), failed=o["failed"], vx=o["takeoff"]["vx"], heel_margin=o["heel_margin"], x_touch=o["x_touch"], cp_trig=s.get("cp_trig"),
                             steps=[x.get("dx") for x in s.get("steps", [])], phase=s.get("phase"), takeoff_t=o.get("takeoff_t"), touch_t=o.get("touch_t"), t_end=o.get("t_end"),
                             stored=dict(valid=sv["valid"], failed=sv["failed"], vx=sv["takeoff"]["vx"], heel_margin=sv["heel_margin"], x_touch=sv["x_touch"], cp_trig=(sv["step"] or {}).get("cp_trig"))))
            if dt == 0.001: keep = (seg, REC["m"])
        match = all(a["valid"] == a["stored"]["valid"] and sorted(a["failed"]) == sorted(a["stored"]["failed"]) and
                    all(abs(a[k] - a["stored"][k]) < 1e-6 for k in ("vx", "heel_margin", "x_touch", "cp_trig")) for a in rows)
        Rr = json.load(open(f"{VP}/rung_{it['rung']}.json"))["result"]["runs"]; labs = [l for l in dict.fromkeys(x["label"] for x in Rr) if l != "nominal"]
        pert = {l: [bool(x["valid"]) for x in Rr if x["label"] == l] for l in labs}
        info = dict(tag=tag, kind="fail", rung=it["rung"], h=rj["h"], D=rj["D"], seed=rj["seed"], secs=rj["secs"], iters=rj["iters"], best_f=rj["best_f"], rows=rows, match=bool(match),
                    flight_match=True, pert=pert, pert_pass=sum(all(v) for v in pert.values()), pert_n=len(labs), edge_x=c.S[0.001]["edge_x"],
                    stored_src=[f"verify_physics_r2/rung_{it['rung']}.json"])
    seg, m = keep; mujoco.mj_saveModel(m, f"{CACHE}/{tag}.mjb", None)
    np.savez(f"{CACHE}/{tag}.npz", T=np.array([s[0] for s in seg]), FF=np.array([s[1] for s in seg]), FB=np.array([s[2] for s in seg]),
             Q=np.array([s[3] for s in seg]), COM=np.array([s[4] for s in seg]))
    json.dump(info, open(f"{CACHE}/{tag}.json", "w"), indent=1, default=lambda v: v.tolist() if hasattr(v, "tolist") else (bool(v) if isinstance(v, np.bool_) else float(v)))
    print(tag, "match", info["match"], "flight_match", info["flight_match"], [{k: v for k, v in r.items() if k not in ("step",)} for r in rows], flush=True)


def lerp(a, b, w): return np.asarray(a, float)*(1 - w) + np.asarray(b, float)*w
def ease(x): x = float(np.clip(x, 0, 1)); return x*x*(3 - 2*x)


RATINGS = "ratings (Unitree published): knee 120, ankle pitch/roll 35, waist roll/pitch 35, hip 88 N·m"
def render(tag):
    from engine import cinematic as K
    info = json.load(open(f"{CACHE}/{tag}.json")); assert info["match"] and info["flight_match"] is not False, f"{tag}: replay does not reproduce the stored value"
    zz = np.load(f"{CACHE}/{tag}.npz"); T, FF, FB, Q, COM = zz["T"], zz["FF"], zz["FB"], zz["Q"], zz["COM"]
    st = K.Studio(f"{CACHE}/{tag}.mjb", fog=(12.0, 50.0)); m = st.m
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box")
    if bid >= 0: m.geom_rgba[bid] = [0.80, 0.52, 0.26, 1.0]
    r1, r2 = info["rows"]; kind = info["kind"]; t_end = float(T[-1])
    def q_at(t): return K.track_q(Q, T, t)
    def com_at(t): return K.track_q(COM, T, t)
    def ff_at(t): return float(K.track_q(FF, T, t)), float(K.track_q(FB, T, t))
    who = "Unitree G1 (stock Menagerie model)"
    if kind in ("land", "hop"):
        td = r1["touchdown_t"]; w0 = td + info["settle_t"]; vx = info["vx"]; zf = [r["longest_zero_floor_force_s"] for r in (r1, r2)]
        task = f"{who}  ·  land a 0.20 m drop moving forward at {vx:.2f} m/s, hold 2 s (flat-floor probe)"
        lab = f"robust: {info['pert_pass']}/{info['pert_n']} new verifier perturbations pass at both dt (needs 3/4)  ->  NOT robust"
        base = ["landing servos UNLAGGED (10 ms lag not tested)", "mechanism probe: released in flight posture at vx; no push-off",
                "R2 landing gains: generic CMA tune over vx 0/0.25/0.5/0.75/1.0 (supplied)", RATINGS,
                "contact: foot pairs solref 0.008, mu 1; non-foot 0.02", "the plant's 0.20 m box sits 3 m ahead, never reached"]
        if kind == "land":
            value = f"HELD (nominal)  ·  foot travel {r1['foot_travel']*1000:.1f} mm, pelvis drift {r1['pelvis_drift']*100:.1f} cm"
            valid = (f"valid at 1 ms and 0.5 ms  (travel {r1['foot_travel']*1000:.1f} / {r2['foot_travel']*1000:.1f} mm <= 20;  drift {r1['pelvis_drift']*100:.1f} / {r2['pelvis_drift']*100:.1f} cm <= 5)"
                     f"  ·  top of the unbroken range 0-0.30 m/s")
            ttl = ("FORWARD LANDING  ·  0.30 m/s: top of the range held at both dt", "replayed on a fresh plant (verifier grid); flat floor (the plant's box, 3 m ahead, is never reached)")
            cav = [lab, f"0.5 ms run has a {zf[1]*1000:.0f} ms zero-floor-force interval (1 ms run: {zf[0]*1000:.0f} ms); shown: 1 ms", "no step taken (none needed)"] + base
        else:
            value = f"HOP, NOT A CAPTURE STEP  ·  {zf[0]*1000:.0f} ms with zero floor force; passes the hold gates"
            valid = (f"passes the gates at 1 ms and 0.5 ms (travel {r1['foot_travel']*1000:.1f} / {r2['foot_travel']*1000:.1f} mm, drift {r1['pelvis_drift']*100:.1f} / {r2['pelvis_drift']*100:.1f} cm)"
                     f"; gates open at touchdown + {info['settle_t']:.1f} s, after the hop")
            ttl = ("REFUTED HEADLINE  ·  the 0.50 m/s 'capture step' is a hop", "physics verifier R2: flight phase 99-103 ms; 0.50 is a CMA tuning-grid point (0.475, 0.525 fail at 0.5 ms)")
            cav = [lab, f"zero floor force {zf[0]*1000:.0f} ms (1 ms) / {zf[1]*1000:.0f} ms (0.5 ms); COM rises ~3 cm", "no gate sees the flight phase"] + base
        events = [(td, f"touchdown t = {td:.3f} s", K.ACC), (w0, f"hold window opens (touchdown + {info['settle_t']:.1f} s): gates measured from here", (150, 220, 160))]
        cam0 = ([0.05, 0.0, 0.55], 2.9, 100.0, -8.0); cam_far = 3.4
        windows = [(td - 0.05, min(td + 0.45, t_end), 0.2)]; t_stop = t_end; base_f = 1.0
        def cam(t):
            c = com_at(t); return ([c[0], c[1], 0.52], cam0[1] + (cam_far - cam0[1])*ease((t - td - 0.5)/1.5), 100.0 - 20*ease((t - td - 0.5)/1.5), -8.0)
    else:
        to, tt, tend = r1["takeoff_t"], r1["touch_t"], r1["t_end"]
        task = f"{who}  ·  jump onto a {info['h']:.2f} m box, near edge {info['D']:.2f} m ahead, land and hold 2 s"
        value = f"NOT HELD  ·  takeoff vx {r1['vx']:.2f} m/s, capture point {r1['cp_trig']:.2f} m past the toe; falls"
        valid = (f"INVALID at 1 ms and 0.5 ms (fresh plants)  ·  0.5 ms: vx {r2['vx']:.2f}, cp {r2['cp_trig']:.2f} m, one {r2['steps'][0]:.2f} m step then falls" if r2["steps"] else
                 f"INVALID at 1 ms and 0.5 ms (fresh plants)")
        ttl = ("FAILURE CASE  ·  no box held by G1 or H1 after repair round 2", f"v10 search {info['rung']} (CMA seed {info['seed']}, {info['secs']:.0f} s), best f {info['best_f']:.0f}; replayed from rung_h0.20.json")
        cav = [f"robust: {info['pert_pass']}/{info['pert_n']} new verifier perturbations (S1-S4); all 4 v10 chains 0/4",
               f"failed gates (1 ms): {', '.join(r1['failed'])}", "the landing holds <= 0.30 m/s unbroken (nominal); the search takes off at 0.93-1.91",
               "push-off: 40 ms first-order lag + torque-speed envelope", "landing servos UNLAGGED", RATINGS,
               "scene addition: rigid box with the floor's contact pairs (foot pairs solref 0.008, mu 1)", "H1 not re-run (no ballistic solution at 2.20 m/s)",
               "robot shown until the rollout's fall stop, then held"]
        events = [(to, f"takeoff t = {to:.3f} s  ·  vx {r1['vx']:.2f} m/s (landing holds <= 0.30 nominal)", K.ACC),
                  (tt, f"touches the box t = {tt:.3f} s  ·  heel {r1['heel_margin']:.2f} m past the edge; capture point {r1['cp_trig']:.2f} m past the toe", K.WARN),
                  (tt + 0.01, "capture step triggered at once; the robot falls during the swing", K.WARN)]
        ex = info["edge_x"]; cam0 = ([ex - 0.2, 0.0, 0.5], 3.4, 95.0, -10.0)
        windows = [(to - 0.08, tend, 0.2)]; t_stop = tend; base_f = 0.5
        def cam(t):
            c = com_at(min(t, tend)); return ([0.5*(c[0] + ex + 0.2), c[1]*0.5, 0.5], 3.4, 95.0 - 10*ease((t - tt)/0.3), -10.0)
    def frame(t, fac, cm, extra=None):
        st.set(q_at(min(t, t_end))); c = com_at(min(t, t_end))
        im = K.finish(st.frame(*cm, geoms=[("sphere", [c[0], c[1], 0.003], [0.025, 0.025, 0.003], [0.3, 0.75, 1.0, 0.8])], key_target=[c[0], c[1], 0.6]))
        im = K.title(im, *ttl); im = K.lower_third(im, task, value, valid, cav, scale=0.74); im = K.badge(im, fac, t)
        ff, fb = ff_at(min(t, t_end)); y = 92
        air = ff + fb < 1.0
        im = K.note(im, f"floor force {ff:6.0f} N" + (f"   box force {fb:6.0f} N" if kind == "fail" else "") + ("   AIRBORNE (no support force)" if air else ""), (40, y),
                    color=(255, 120, 100) if air else (150, 220, 160), size=16); y += 38
        for te, txt, col in events:
            if t >= te: im = K.note(im, txt, (40, y), color=col, size=16); y += 38
        if t >= t_end - 1e-9 and kind == "fail": im = K.note(im, f"rollout stops at t = {t_end:.2f} s (fall detected); pose held", (40, y), color=K.WARN, size=16)
        if extra: im = K.note(im, extra, (40, y), color=(140, 200, 255), size=16)
        return im
    W = K.Writer(f"{OUT}/{tag}.mp4"); stills = {}
    W.add(frame(0.0, 1.0, cam(0.0)), hold=30)
    for t, fac in K.timeline(t_stop, windows=windows, base=base_f):
        im = frame(t, fac, cam(t)); W.add(im)
        if kind != "fail":
            if "touchdown" not in stills and t >= r1["touchdown_t"] + 0.03: stills["touchdown"] = im
            if kind == "hop" and "airborne" not in stills and t > r1["touchdown_t"] + 0.12 and ff_at(t)[0] < 1.0: stills["airborne"] = im
        else:
            if "takeoff" not in stills and t >= r1["takeoff_t"] + 0.08: stills["takeoff"] = im
            if "touch" not in stills and t >= r1["touch_t"] + 0.02: stills["touch"] = im
    stills["end"] = im; W.add(im, hold=90)
    out, dur = W.close()
    for k, v in stills.items(): v.convert("RGB").save(f"{OUT}/{tag}_{k}.jpg", quality=88)
    print(out, f"{dur:.1f} s", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
    {"capture": capture, "render": render}[sys.argv[1]](sys.argv[2])
