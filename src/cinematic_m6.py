"""Cinematic renders of M6_walking_headcarry repair round 2 (2026-10-19), studio kit engine/cinematic.py.
Rendered (verified by the physics lens round 2, VERIFY_physics.md, commit 50f0524):
  m6_pot15_v075_rFrob          stock G1H, 15 kg pot on the head at 0.75 m/s: heaviest ROBUST load (P 4/4, V 4/4, W 4/4, X 4/4)
  m6_pot20_v075_rIens          the same task with 20 kg: heaviest VALID load, NOT robust (P 2/4; V 3/4, W 3/4, X 4/4)
  m6_pot20_v075_rIens_P2_FAIL  20 kg under the registered P2 (load x1.10 vs the controller's model + robot mass noise): load drops at 0.94 s
capture: one FRESH process per (clip, dt) on the physics verifier's own plant (verify_m6_physics_r2: uid 6, kw_for, same search vector),
         with record=True. The replayed summary is compared field by field (1e-4) with the verifier's stored row (verify_physics/r2_runs.json).
         The failure clip also captures a CONTINUATION run (M6_LOADSTOP=0, a diagnostic flag of the builder: the rollout is not stopped when
         the load gate fails); its trace must equal the gated run's trace up to the drop.
render:  kinematic replay of the 1 ms record (refused unless both dt matched) -> results/demo/cinematic/M6_walking_headcarry/<tag>.mp4 (+ stills).
usage (from src/): python cinematic_m6.py capture <tag> <1|2|c> | render <tag>"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M6 = f"{ROOT}/results/movements/M6_walking_headcarry"; CACHE = f"{ROOT}/results/demo/cinematic/cache"; OUT = f"{ROOT}/results/demo/cinematic/M6_walking_headcarry"
ITEMS = {"m6_pot15_v075_rFrob": ("pot_15kg_v0.75_s1_rFrob", "nominal"), "m6_pot20_v075_rIens": ("pot_20kg_v0.75_s2_rIens", "nominal"),
         "m6_pot20_v075_rIens_P2_FAIL": ("pot_20kg_v0.75_s2_rIens", "P2")}
import numpy as np, mujoco
from cinematic_m5 import _cmp, ease


def capture(tag, which):
    import verify_m6_physics_r2 as V
    from engine import core, walk_headcarry as WH
    case, pert = ITEMS[tag]; dt = {"1": core.DT_OPT, "2": core.DT_CHECK, "c": core.DT_OPT}[which]
    g = json.load(open(f"{M6}/search/{case}.json"))
    C = WH.Carry(g["obj"], g["mass"], g["v"], dt_list=(dt,), uid=V.UID, law_base=g["law_base"]); S = C.S[dt]; m = S["m"]
    r = C.run(g["x"], dt, law=True, record=True, **V.kw_for(pert, len(S["types"])))
    jo = json.loads(json.dumps(WH.summ(r), default=float))
    stored = json.load(open(f"{M6}/verify_physics/r2_runs.json"))["out"][case]["law"][pert][str(dt)]
    stored = {k: v for k, v in stored.items() if k not in ("secs", "slip_log")}
    bad = _cmp(jo, stored) if which != "c" else []
    Lp = C.law_setup(S, WH.law_params(C.split(g["x"])[1], C.law_base)[0])
    types = list(S["types"]); tr = r["trace"]; base = f"{CACHE}/{tag}_dt{which}"
    np.savez(base + ".npz", T=np.asarray(tr["t"]), Q=np.asarray(tr["qpos"]), COM=np.asarray(tr["com"]), LOAD=np.asarray(tr["load"]), TQ=np.asarray(tr["tq"]))
    mujoco.mj_saveModel(m, base + ".mjb", None)
    bnd = r.get("bounds", [])
    from engine import walk as W
    fb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("right_ankle_roll_link", "left_ankle_roll_link")]
    info = dict(tag=tag, case=case, pert=pert, dt=float(m.opt.timestep), which=which, match=not bad, mismatches=bad, replay=jo, stored=stored,
                stored_src="results/movements/M6_walking_headcarry/verify_physics/r2_runs.json", obj=g["obj"], mass=float(g["mass"]), v_cmd=float(g["v"]),
                bid=int(Lp["bid"]), tb=int(Lp["tb"]), h_loc=np.asarray(Lp["h_loc"]).tolist(), hh=float(Lp["hh"]), types=types,
                tau=np.asarray(S["jarr"]["tau"]).tolist(), wp=types.index("waist_pitch_joint"), wr=types.index("waist_roll_joint"), fb=fb,
                t_win=[bnd[W.N_TRANS]["t"], bnd[-1]["t"]] if len(bnd) > W.N_TRANS + 1 else None, lag=WH.LAG, ratings=WH.ratings_table(S),
                solver_iter=int(m.opt.iterations), neq=int(m.neq), robot_mass=float(sum(m.body_mass[b] for b in S["robot_bodies"])),
                loadstop=WH.LOAD_STOP)
    json.dump(info, open(base + ".json", "w"), indent=1, default=lambda v: v.tolist() if hasattr(v, "tolist") else float(v))
    print(tag, which, "MATCH" if not bad else f"MISMATCH {bad[:6]}", jo["value"], jo["valid"], jo["reasons"], jo["load"]["t_held"], flush=True)


RAT = "ratings (Unitree published): knee 120, hips + waist yaw 88, ankles + waist roll/pitch 35, arms 25 N·m; wrist servos 5"


def render(tag):
    from engine import cinematic as K
    I1, I2 = (json.load(open(f"{CACHE}/{tag}_dt{k}.json")) for k in (1, 2))
    assert I1["match"] and I2["match"], f"{tag}: replay does not reproduce the stored verifier row: {I1['mismatches'][:4]} {I2['mismatches'][:4]}"
    fail = I1["pert"] != "nominal"
    zz = np.load(f"{CACHE}/{tag}_dt1.npz"); T, Q, COM, LOAD, TQ = zz["T"], zz["Q"], zz["COM"], zz["LOAD"], zz["TQ"]
    t_gate = float(T[-1])
    if fail:   # continuation (diagnostic): must equal the gated run up to the drop
        Ic = I1c = json.load(open(f"{CACHE}/{tag}_dtc.json")); zc = np.load(f"{CACHE}/{tag}_dtc.npz"); n = len(T)
        dev = float(np.abs(zc["Q"][:n] - Q).max()); assert dev < 1e-9, f"continuation departs from the gated run before the drop ({dev})"
        I1["cont_dev"] = dev; T, Q, COM, LOAD, TQ = zc["T"], zc["Q"], zc["COM"], zc["LOAD"], zc["TQ"]
    Ic = json.load(open(f"{CACHE}/{tag}_dtc.json")) if fail else None
    st = K.Studio(f"{CACHE}/{tag}_dt1.mjb", fog=(14.0, 60.0)); m = st.m
    r1, r2 = I1["replay"], I2["replay"]; L1, L2 = r1["load"], r2["load"]; mass = I1["mass"]
    t_end = float(T[-1]) if not fail else min(float(T[-1]), t_gate + 2.2)
    bid, tb, hloc = I1["bid"], I1["tb"], np.asarray(I1["h_loc"]); tau = np.asarray(I1["tau"]); wp, wr = I1["wp"], I1["wr"]
    who = "Unitree G1 + Dex3 hands (G1H, stock Menagerie model)"
    task = f"{who}  ·  walk {I1['v_cmd']:.2f} m/s with a free {mass:g} kg pot on the head"
    common = ["10 ms first-order lag on driven joints; wrist + shoulder-yaw held servos lag-free", RAT, "stock MJCF contact (mu 1, condim 3)",
              f"solver Newton {I1['solver_iter']} it (MJCF: 100)", f"no welds (neq {I1['neq']}): the pot is free, steadied by both hands under the arm-balance law",
              "pot helper-placed on the head at t = 0; flying start mid single support", "controller knows the nominal load mass and COM",
              "construction: M5 gait + head-carry law -> CMA search (stride, gains, 6 law params; seed >= 1); no choreography"]
    gates = "gates: load within 7 cm of head top, tilt < 45 deg, slip <= 2 cm, no flight, speed +-5 %"
    if tag == "m6_pot15_v075_rFrob":
        value = f"{r1['value']:.3f} m/s net over {r1['dist']:.2f} m  ·  pot held {L1['t_held']:.1f} s  ·  e_max {100*L1['e_max']:.1f} cm  ·  tilt {L1['tilt_max']:.1f} deg"
        rob = ["ROBUST: P1-P4 4/4, V1-V4 4/4, W1-W4 4/4, verifier X1-X4 4/4 (16/16, both dt)", "heaviest ROBUST load at 0.75 m/s",
               "waist pitch at its 35 N·m rating at p95 (zero margin)", "spec predicted 10.9 kg: found/predicted 1.38", "law off: pot lost at 0.63 s"]
        ttl = ("HEAD CARRY  ·  15 kg pot at 0.75 m/s, robust", "replayed on fresh plants (uid 6) through the physics verifier's code; shown: the 1 ms run")
        windows = [(0.0, 1.3, 0.25), (4.0, 9.0, 2.0)]
    elif tag == "m6_pot20_v075_rIens":
        value = f"{r1['value']:.3f} m/s net over {r1['dist']:.2f} m  ·  pot held {L1['t_held']:.1f} s  ·  e_max {100*L1['e_max']:.1f} cm  ·  tilt {L1['tilt_max']:.1f} deg"
        rob = ["NOT ROBUST: registered P1-P4 2/4 (needs 3/4)", "unscored V1-V4 3/4, W1-W4 3/4; verifier X1-X4 4/4", "every failure: pot dropped at 0.86-1.23 s (gait start)",
               "heaviest VALID load at 0.75 m/s (found/predicted 1.84)", "waist pitch + roll at the 35 N·m rating in the start window", "law off: pot lost at 0.56 s"]
        ttl = ("HEAD CARRY  ·  20 kg pot at 0.75 m/s, valid, not robust", "replayed on fresh plants (uid 6) through the physics verifier's code; shown: the 1 ms run")
        windows = [(0.0, 1.3, 0.25), (4.0, 9.0, 2.0)]
    else:
        value = f"POT DROPPED at {L1['t_lost']:.2f} s  (load x1.10 vs model + mass noise)"
        rob = ["registered P2: true load 10 % heavier than the controller's model + robot mass noise sd 5 %", "one of the two P1-P4 failures that make 20 kg NOT robust (P 2/4)", "split: load x1.10 alone drops it; mass noise alone and load x1.05 pass",
               "waist pitch at its 35 N·m rating 79 % of t < 0.94 s (nominal: 33 % of t < 1.2 s)", "after the gate: continuation run (load-stop off), diagnostic only, not judged"]
        ttl = ("FAILURE CASE  ·  20 kg pot under perturbation P2", "P1-P4 registered in PREDICTION.md before any search; re-run here through the verifier's code (uid 6)")
        windows = [(0.0, t_gate + 0.4, 0.25), (t_gate + 0.4, t_end, 0.5)]
    if fail:
        valid = f"INVALID at 1 ms and 0.5 ms: {', '.join(r1['reasons'])} at {L1['t_lost']:.3f} / {L2['t_lost']:.3f} s (nominal of the same vector is valid at both dt)"
    else:
        valid = f"valid at 1 ms and 0.5 ms  (net {r1['value']:.3f} / {r2['value']:.3f} m/s;  {gates})"
    cav = rob + common
    def at(A, t): return K.track_q(A, T, t)
    dk = mujoco.MjData(m); fbs = I1["fb"]; prints = []; on_prev = [False, False]
    for i, t in enumerate(T):
        for li in (0, 1):
            on = LOAD[i, li] > 0.05
            if on and not on_prev[li]:
                dk.qpos[:] = Q[i]; mujoco.mj_kinematics(m, dk); R_ = dk.xmat[fbs[li]].reshape(3, 3); c = dk.xpos[fbs[li]]
                prints.append((float(t), li, c[:2].copy(), float(np.arctan2(R_[1, 0], R_[0, 0]))))
            on_prev[li] = on
    PC = []   # pot centre path for the trail
    for q in Q: dk.qpos[:] = q; mujoco.mj_kinematics(m, dk); PC.append(dk.xipos[bid].copy())
    PC = np.asarray(PC); LEGC = {0: [0.30, 0.62, 1.00], 1: [1.00, 0.70, 0.25]}
    def geoms(t):
        g = []
        for (tp, li, xy, yaw) in prints:
            if tp > t: break
            a = 0.85*max(0.25, 1 - (t - tp)/8.0); c, s_ = np.cos(yaw), np.sin(yaw); Rm = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
            g.append(("box", [xy[0], xy[1], 0.0015], [0.10, 0.04, 0.0015], LEGC[li] + [a], Rm))
        i1 = int(np.searchsorted(T, t)); i0 = int(np.searchsorted(T, t - 1.5))
        for j in range(i0, min(i1, len(PC)), 6):
            a = 0.7*(1 - (t - T[j])/1.5); g.append(("sphere", PC[j], [0.008]*3, [1.0, 0.85, 0.3, max(a, 0.05)], None))
        return g
    def cam(t):
        c = at(COM, min(t, t_end)); az = 150.0 - 25*ease((t - 1.5)/4.0)
        if fail: az = 150.0 - 15*ease((t - 0.5)/1.5)
        if fail:
            k = ease((t - t_gate)/0.8); return [c[0], c[1]*0.6, 0.92 - 0.55*k], 2.8 + 0.9*k, az, -16.0 - 10*k
        return [c[0], c[1]*0.6, 0.92], 3.0, az, -16.0
    def bar(im, y, lab, u, sat_col=K.WARN):
        from PIL import ImageDraw
        d = ImageDraw.Draw(im); x0, w = 40, 260; d.rounded_rectangle([x0, y, x0 + 480, y + 26], radius=6, fill=(12, 16, 24, 200))
        d.text((x0 + 10, y + 5), lab, font=K.font(14, True), fill=K.SUB); bx = x0 + 205
        d.rectangle([bx, y + 7, bx + w, y + 19], outline=(120, 130, 140)); col = sat_col if u > 0.99 else (140, 200, 255)
        d.rectangle([bx, y + 7, bx + int(w*min(u, 1.0)), y + 19], fill=col)
        return im
    def frame(t, fac):
        tt = min(t, t_end); st.set(at(Q, tt)); la, d0, az, el = cam(t)
        img = st.frame(la, d0, az, el, geoms=geoms(tt), key_target=[la[0], la[1], 0.9])
        im = K.finish(img); im = K.title(im, *ttl); im = K.lower_third(im, task, value, valid, cav, scale=0.70); im = K.badge(im, fac, t)
        dd = st.d; Rt = dd.xmat[tb].reshape(3, 3); ht = dd.xpos[tb] + Rt @ hloc; cl = dd.xipos[bid]; e = float(np.linalg.norm(cl[:2] - ht[:2]))
        tilt = float(np.degrees(np.arccos(np.clip(dd.xmat[bid][8], -1, 1)))); held = e <= 0.07 and cl[2] > ht[2]
        y = 92
        if held: col, txt = (150, 220, 160), f"pot offset from head-top centre {100*e:4.1f} cm (gate 7.0)   tilt {tilt:4.1f} deg"
        else: col, txt = K.WARN, f"POT OFF THE HEAD   offset {100*min(e, 9.99):4.1f} cm   pot height {cl[2]:.2f} m"
        im = K.note(im, txt, (40, y), color=col, size=16); y += 38
        u = np.abs(at(TQ, tt))/tau
        im = bar(im, y, f"waist pitch  {u[wp]*tau[wp]:4.1f}/{tau[wp]:.0f} N·m", u[wp]); y += 30
        im = bar(im, y, f"waist roll   {u[wr]*tau[wr]:4.1f}/{tau[wr]:.0f} N·m", u[wr]); y += 34
        if fail and t >= t_gate:
            im = K.note(im, f"gate failed at t = {t_gate:.2f} s ({L1['reason']}); shown after it: continuation run, not judged (robot falls, t = {Ic['replay']['t_run']:.2f} s)", (40, y), color=K.WARN, size=16)
        elif not fail and I1["t_win"] and tt >= I1["t_win"][0]:
            im = K.note(im, f"measured window from t = {I1['t_win'][0]:.2f} s (after 2 transient cycles): net {r1['value']:.3f} m/s", (40, y), color=(140, 200, 255), size=16)
        return im
    Wr = K.Writer(f"{OUT}/{tag}.mp4"); stills = {}
    Wr.add(frame(0.0, 1.0), hold=30); im = None
    for t, fac in K.timeline(t_end, windows=windows, base=1.0):
        im = frame(t, fac); Wr.add(im)
        if not fail:
            if "start" not in stills and t >= 0.8: stills["start"] = im
            if "walk" not in stills and t >= 6.0: stills["walk"] = im
        else:
            if "start" not in stills and t >= 0.6: stills["start"] = im
            if "drop" not in stills and t >= t_gate: stills["drop"] = im
    stills["end"] = im; Wr.add(im, hold=90)
    out, dur = Wr.close()
    for k, v in stills.items(): v.convert("RGB").save(f"{OUT}/{tag}_{k}.jpg", quality=88)
    print(out, f"{dur:.1f} s", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
    if sys.argv[1] == "capture": capture(sys.argv[2], sys.argv[3])
    else: render(sys.argv[2])
