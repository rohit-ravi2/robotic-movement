"""Cinematic renders of existing results (2026-10-18). Inputs are the replay caches written (and value-checked against the stored
records) by cinematic_capture_{construct,sprint,headcarry}.py; a cache whose check did not match is refused here.
usage: MUJOCO_GL=glfw python cinematic_render.py throw|kick|race|q3|capacity|headcarry   ->  results/demo/cinematic/*.mp4"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco
from PIL import Image, ImageDraw
from engine import cinematic as K

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CACHE, OUT = f"{ROOT}/results/demo/cinematic/cache", f"{ROOT}/results/demo/cinematic"
R_BALL = 0.037      # visual radius of the 0.145 kg baseball (the simulated ball is a point mass merged into the hand)


def load(tag):
    info = json.load(open(f"{CACHE}/{tag}.json")); assert info["check"]["match"], f"{tag}: replay does not reproduce the stored value: {info['check']}"
    d = np.load(f"{CACHE}/{tag}.npz"); return info, d["Q"], d["t"]


def lerp(a, b, w): return np.asarray(a, float)*(1 - w) + np.asarray(b, float)*w
def ease(x): x = float(np.clip(x, 0, 1)); return x*x*(3 - 2*x)


def trail(P, n_keep, rgba, r0, r1=0.25):
    """Fading trail geoms from a list of past positions (newest last)."""
    P = P[-n_keep:]; k = len(P); out = []
    for i, p in enumerate(P[:-1]):
        a = (i + 1)/k; out.append(("sphere", p, [r0*(r1 + (1 - r1)*a)]*3, [rgba[0], rgba[1], rgba[2], 0.65*a**1.5]))
    return out


def vfmt(v): return f"{v:.2f}"


# ---------------------------------------------------------------- 1. THROW with the ball's flight
def throw():
    info, Q, T = load("throw_s0"); st = K.Studio(f"{CACHE}/throw_s0.mjb", fog=(60.0, 170.0)); m = st.m
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_palm"); t_rel = info["release_time"]; T = T + (t_rel - T[-1])   # the last record is the release state
    tf, P = K.ballistic(info["site_pos"], info["site_vel"], R_BALL); t_land = t_rel + tf[-1]; land = P[-1]; rng = float(np.linalg.norm(land[:2] - np.asarray(info["site_pos"][:2])))
    s1, s2 = info["check"]["replay"]; apex = float(P[:, 2].max())
    tl = K.timeline(t_land, windows=[(0.0, t_rel - 0.10, 0.4), (t_rel - 0.10, t_rel + 0.04, 0.1)], base=1.0)
    W = K.Writer(f"{OUT}/throw_G1_ballflight.mp4"); hist = []
    lab = dict(robot_task="Unitree G1 (stock MJX model)  ·  throw, full-budget construction (seed 0)", value=f"{vfmt(s1)} m/s release speed",
               validity=f"valid at 1 ms and 0.5 ms  ({s1:.3f} / {s2:.3f} m/s)",
               caveats=["actuator lag 40 ms", "ball = 0.145 kg point mass merged into the hand; release speed = palm speed (no grip limit)",
                        "ball flight computed from the release state (drag-free)", f"peak joint speed 1.10x rated (gate tolerance 1.1x)",
                        "robot held at its release pose after release (the rollout ends at release)"])
    x0 = Q[0, :2]
    def render(t, fac, cam, after_land=0.0):
        if t < t_rel: st.set(K.track_q(Q, T, t)); b = st.d.site_xpos[site].copy(); hist.append(b)
        else:
            st.set(Q[-1]); i = min(int((t - t_rel)/(tf[1] - tf[0])), len(P) - 1); b = P[i]; hist.append(b)
        g = [("sphere", b, [R_BALL]*3, K.BALL_RGBA)] + trail(np.array(hist), 14, K.BALL_RGBA, 1.6*R_BALL)
        if t >= t_rel: g += [("sphere", p, [0.06]*3, [1, 0.55, 0.15, 0.35]) for p in P[:i:12]]                 # the arc so far
        if t >= t_land - 1e-6: g += [("cyl", land*[1, 1, 0] + [0, 0, 0.002], [0.9, 0.002, 0], [1, 0.45, 0.1, 0.6])]
        img = st.frame(*cam, geoms=g, key_target=[Q[-1, 0], Q[-1, 1], 0.6]); im = K.finish(img)
        im = K.title(im, "THROW  ·  body-derived wind-up -> release construction", "replayed from results/construct_full/throw_s0.json")
        im = K.lower_third(im, lab["robot_task"], lab["value"], lab["validity"], lab["caveats"]); im = K.badge(im, fac, t)
        if t >= t_rel: im = K.note(im, f"release {s1:.2f} m/s at {np.degrees(info['extra']['elev']):.0f} deg  ·  apex {apex:.1f} m" + (f"  ·  lands {rng:.1f} m away (drag-free)" if t >= t_land - 1e-6 else ""), (40, 96), color=K.ACC)
        return im
    cam0 = ([x0[0], x0[1], 0.35], 3.8, 125.0, -7.0)
    W.add(render(0.0, 1.0, cam0), hold=30)                                                        # 1 s still before the motion
    for t, fac in tl:
        if t < t_rel: a = t/t_rel; cam = ([x0[0] + 0.1*a, x0[1], 0.35], 3.8 - 0.4*a, 125.0 - 25*a, -7.0)
        else:
            w = ease((t - t_rel)/0.6); i = min(int((t - t_rel)/(tf[1] - tf[0])), len(P) - 1); b = P[i]
            cam = (lerp([Q[-1, 0], Q[-1, 1], 0.35], [b[0] - 1.0, b[1], max(b[2] - 0.5, 0.35)], w), 3.4 + 6.5*w, 100 - 10*w, -7 + 2*w)
        W.add(render(t, fac, cam))
    b = P[-1]; c_end = ([b[0] - 1.0, b[1], max(b[2] - 0.5, 0.35)], 9.9, 90.0, -5.0); mid = 0.5*(np.asarray(info["site_pos"]) + land); wide = ([mid[0], mid[1], apex*0.30], 0.62*rng + 8, 90.0, -7.0)
    for k in range(45): w = ease(k/44); W.add(render(t_land, 1.0, (lerp(c_end[0], wide[0], w), c_end[1] + (wide[1] - c_end[1])*w, 90.0, -6 - 1*w)))
    W.add(render(t_land, 1.0, wide), hold=75)
    return W.close()


# ---------------------------------------------------------------- 2. FOOTBALL KICK with the simulated ball
def kick():
    info, Q, T = load("kick_ball_s1"); st = K.Studio(f"{CACHE}/kick_ball_s1.mjb", fog=(14.0, 60.0)); m = st.m
    ja = info["ball_qadr"]; r = float(m.geom_size[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")][0])
    bp = Q[:, ja:ja+3]; moved = np.where(np.linalg.norm(bp - bp[0], axis=1) > 2e-3)[0]; t_imp = float(T[moved[0]]) if len(moved) else float(T[-1])
    t_end = float(T[-1]); tf, P = K.ballistic(info["ball_pos"], info["ball_vel"], r); t_land = t_end + tf[-1]; s1, s2 = info["check"]["replay"]
    tl = K.timeline(t_land, windows=[(0.0, t_imp - 0.10, 0.35), (t_imp - 0.10, t_imp + 0.05, 0.1)], base=0.5)
    W = K.Writer(f"{OUT}/kick_G1_ball.mp4"); hist = []
    caveats = ["actuator lag 40 ms", "simulated ball 0.43 kg, r 0.11 m", "4 cm massless contact sphere added at the foot site",
               f"after the rollout ends at {t_end:.2f} s: drag-free ballistic extension (no bounce or roll)", "robot held at its last pose after the rollout ends"]
    def render(t, fac, cam):
        st.set(K.track_q(Q, T, min(t, t_end)))
        if t > t_end: i = min(int((t - t_end)/(tf[1] - tf[0])), len(P) - 1); st.d.qpos[ja:ja+3] = P[i]; st.set(st.d.qpos.copy())
        b = st.d.qpos[ja:ja+3].copy(); hist.append(b)
        g = trail(np.array(hist), 12, [0.95, 0.95, 0.95, 1], r*0.8) if t > t_imp else []
        if t >= t_land - 1e-6: g += [("cyl", P[-1]*[1, 1, 0] + [0, 0, 0.002], [0.5, 0.002, 0], [1, 0.45, 0.1, 0.6])]
        im = K.finish(st.frame(*cam, geoms=g, key_target=[Q[0, 0] + 0.5, Q[0, 1], 0.4]))
        im = K.title(im, "FOOTBALL KICK  ·  impact construction with a simulated ball", "replayed from results/construct/kick_ball_s1.json (2,400-eval tier)")
        im = K.lower_third(im, "Unitree G1 (stock MJX model)  ·  right-foot kick, left-leg support", f"{vfmt(s1)} m/s ball speed after impact",
                           f"valid at 1 ms and 0.5 ms  ({s1:.3f} / {s2:.3f} m/s)", caveats)
        im = K.badge(im, fac, t)
        if t > t_end: im = K.note(im, "ballistic extension (drag-free)" + (f"  ·  lands {np.linalg.norm(P[-1][:2] - bp[0][:2]):.1f} m from the spot" if t >= t_land - 1e-6 else ""), (40, 96), color=K.ACC)
        return im
    x0 = Q[0, :2]; cam0 = ([x0[0] + 0.3, x0[1], 0.25], 3.2, 130.0, -8.0)
    W.add(render(0.0, 1.0, cam0), hold=30)
    for t, fac in tl:
        b = Q[-1, ja:ja+3] if t > t_end else K.track_q(Q, T, t)[ja:ja+3]; w = ease((t - t_imp)/0.3) if t > t_imp else 0.0
        W.add(render(t, fac, (lerp([x0[0] + 0.3, x0[1], 0.25], [0.5*(x0[0] + b[0]) + 0.3, 0.5*(x0[1] + b[1]), 0.1], w), 3.2 + 3.0*w, 130.0 - 35*w, -8.0)))
    W.add(render(t_land, 1.0, ([0.5*(x0[0] + P[-1][0]) + 0.3, 0.5*(x0[1] + P[-1][1]), 0.1], 6.2, 95.0, -8.0)), hold=60)
    return W.close()


# ---------------------------------------------------------------- 3. SPRINT panels
SPR = {
    "G1": dict(tag="sprint_G1", name="Unitree G1  ·  sprint (stock body)", dist=3.0, src="results/sprint/ladder_G1_lag2ms.json rung 2",
               cav=["actuator lag 2 ms", "MJCF peak ratings (ankle and waist 50 N·m)", "gate: 1.1x overspeed tolerance"]),
    "H1": dict(tag="sprint_H1", name="Unitree H1  ·  sprint (stock body)", dist=3.8, src="results/sprint/ladder_H1_lag2ms_extrap.json rung 2",
               cav=["actuator lag 2 ms", "contact solref 0.004 (at the stock 0.02 no gait)", "stride extrapolated past the scaffold ceiling", "point solution"]),
    "Apollo": dict(tag="sprint_Apollo", name="Apptronik Apollo  ·  sprint (NON-STOCK plant)", dist=3.8, src="results/sprint/ladder_Apollo_heldout_lag2ms.json rung 1",
                   cav=["actuator lag 2 ms", "what-if plant: joint damping 0, sole mu 1, condim 3, omega assumed", "not quotable as Apollo capability"]),
}


def sprint_panel(key, w, h):
    c = SPR[key]; info, Q, T = load(c["tag"]); st = K.Studio(f"{CACHE}/{c['tag']}.mjb", w=w, h=h, fog=(10.0, 40.0)); return dict(c, info=info, Q=Q, T=T, st=st, h0=float(Q[0, 2]))


def sprint_frame(p, t, scale, az=90.0, el=-6.0):
    Q, T, st = p["Q"], p["T"], p["st"]; ended = t > T[-1] + 1e-9; st.set(K.track_q(Q, T, t)); x, y = st.d.qpos[0], st.d.qpos[1]
    im = K.finish(st.frame([x, y, 0.42*p["h0"]], p["dist"], az, el, key_target=[x, y, 0.5*p["h0"]]))
    s1, s2 = p["info"]["check"]["replay"]
    im = K.lower_third(im, p["name"], f"{min(s1, s2):.2f} m/s", f"valid at 1 ms and 0.5 ms ({s1:.3f} / {s2:.3f}; lower shown)", p["cav"], x=int(16*scale/0.55), scale=scale, width=int(im.size[0] - 2*16*scale/0.55))
    d = ImageDraw.Draw(im); f = K.font(int(15*scale/0.55), True); d.text((im.size[0] - int(130*scale/0.55), int(14*scale/0.55)), f"x = {x:5.1f} m", font=f, fill=K.INK)
    if ended: im = K.note(im, f"rollout ends at {T[-1]:.2f} s", (int(16*scale/0.55), int(14*scale/0.55)), size=int(14*scale/0.55))
    return im


def t1_card(w, h, scale):
    im = Image.new("RGBA", (w, h), (24, 28, 36, 255)); d = ImageDraw.Draw(im); s = scale/0.55
    d.text((int(30*s), int(40*s)), "Booster T1  ·  sprint (stock body)", font=K.font(int(22*s), True), fill=K.INK)
    d.text((int(30*s), int(80*s)), "no valid gait", font=K.font(int(34*s), True), fill=K.WARN)
    for i, l in enumerate(["The construction finds no valid sprint on the stock T1", "(2 ms actuator lag, both timesteps).",
                           "Dynamic stride predictor: 0 (no feasible stride) - it agrees.", "What-if body tau and omega x1.5: 1.64 m/s (not stock).",
                           "Source: results/heldout, STOCK_RESULTS_2026-10-17.md"]):
        d.text((int(30*s), int((135 + 26*i)*s)), l, font=K.font(int(15*s)), fill=K.SUB)
    return im


def race():
    w, h = 640, 360; sc = 0.68; P = [sprint_panel(k, w, h) for k in ("G1", "H1", "Apollo")]; card = t1_card(w, h, sc)
    t_end = max(float(p["T"][-1]) for p in P) + 0.3; W = K.Writer(f"{OUT}/race_four_robots.mp4"); SLOW = 0.5
    for k, t in enumerate(np.arange(0.0, t_end, SLOW/K.FPS)):
        canvas = Image.new("RGBA", (K.W, K.H)); [canvas.paste(sprint_frame(p, t, sc), xy) for p, xy in zip(P, [(0, 0), (w, 0), (0, h)])]; canvas.paste(card, (w, h))
        d = ImageDraw.Draw(canvas); d.line([(w, 0), (w, K.H)], fill=(10, 12, 16), width=3); d.line([(0, h), (K.W, h)], fill=(10, 12, 16), width=3)
        canvas = K.badge(canvas, SLOW, t, x=K.W//2 - 110, y=K.H//2 - 26, scale=0.85)
        W.add(canvas, hold=(30 if k == 0 else 1))
    W.add(canvas, hold=45); return W.close()


def q3():
    info, Q, T = load("sprint_G1_q3"); st = K.Studio(f"{CACHE}/sprint_G1_q3.mjb", fog=(10.0, 40.0)); s1, s2 = info["check"]["replay"]; h0 = float(Q[0, 2]); SLOW = 0.5
    W = K.Writer(f"{OUT}/sprint_G1_q3_urdf_effort.mp4")
    cav = ["actuator lag 2 ms (compensated), droop and back-EMF brake", "ratings: Unitree URDF effort limits (ankle, waist roll/pitch 35 N·m)",
           "knee 139 N·m, saturated: above Unitree's published 120 N·m (EDU) / 90 N·m (base)", "near-grounded: mean flight 16-24 ms",
           "stride extrapolated past the scaffold ceiling", "gate: 1.1x overspeed tolerance"]
    for k, t in enumerate(np.arange(0.0, float(T[-1]) + 0.3, SLOW/K.FPS)):
        st.set(K.track_q(Q, T, t)); x, y = st.d.qpos[0], st.d.qpos[1]; az = 90 + 25*np.cos(np.pi*min(t/float(T[-1]), 1))
        im = K.finish(st.frame([x, y, 0.40*h0], 3.4, az, -6.0, key_target=[x, y, 0.5*h0]))
        im = K.title(im, "SPRINT  ·  G1 at the Unitree URDF effort limits (track Q3)", "replayed from results/sprint/ladder_G1_cont_lag2ms_c2.json rung 2 (exact extrapolated stride)")
        im = K.lower_third(im, "Unitree G1 (stock MJX body)  ·  sprint", f"{min(s1, s2):.2f} m/s", f"valid at 1 ms and 0.5 ms  ({s1:.3f} / {s2:.3f} m/s; lower shown)", cav)
        im = K.badge(im, SLOW, t, extra=f"x = {x:4.1f} m"); W.add(im, hold=(30 if k == 0 else 1))
    W.add(im, hold=30); return W.close()


# ---------------------------------------------------------------- 4. CAPACITY split-screen
CAP = [("throw_s0", "normal body", None), ("throw_weakarm_s0", "WHAT-IF BODY: arm torque x0.5", "shoulder, elbow, wrist"),
       ("throw_weaklegs_s0", "WHAT-IF BODY: legs + trunk torque x0.5", "legs and trunk")]
GROUPS = ["legs", "trunk", "shoulder", "elbow", "wrist"]


def capacity():
    ws = [427, 426, 427]; panels = []
    for (tag, name, note), w in zip(CAP, ws):
        info, Q, T = load(tag); sp = info["src"]; led = json.load(open(sp if os.path.isabs(sp) else os.path.join(os.path.dirname(os.path.abspath(__file__)), sp)))
        st = K.Studio(f"{CACHE}/{tag}.mjb", w=w, h=K.H, fog=(12.0, 50.0)); site = mujoco.mj_name2id(st.m, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
        T = T + (info["release_time"] - T[-1]); tf, P = K.ballistic(info["site_pos"], info["site_vel"], R_BALL)
        panels.append(dict(tag=tag, name=name, note=note, info=info, Q=Q, T=T, st=st, site=site, tf=tf, P=P, w=w, work=led["result"]["extra"]["work"], hist=[]))
    pre = max(p["info"]["release_time"] for p in panels); post = 0.45
    tl = K.timeline(post, windows=[(-pre, -0.10, 0.3), (-0.10, 0.04, 0.1)], base=0.3, t_start=-pre)
    W = K.Writer(f"{OUT}/capacity_throw_whatif_bodies.mp4")
    def frame(tau, fac):
        canvas = Image.new("RGBA", (K.W, K.H)); x = 0
        for p in panels:
            st, Q, T, tr = p["st"], p["Q"], p["T"], p["info"]["release_time"]; t = tr + tau
            if t < tr: st.set(K.track_q(Q, T, max(t, 0.0))); b = st.d.site_xpos[p["site"]].copy()
            else: st.set(Q[-1]); b = p["P"][min(int((t - tr)/(p["tf"][1] - p["tf"][0])), len(p["P"]) - 1)]
            p["hist"].append(b); g = [("sphere", b, [R_BALL]*3, K.BALL_RGBA)] + trail(np.array(p["hist"]), 10, K.BALL_RGBA, R_BALL)
            im = K.finish(st.frame([Q[0, 0] + 0.1, Q[0, 1], 0.05], 4.3, 115.0, -4.0, geoms=g, key_target=[Q[0, 0], Q[0, 1], 0.6])); d = ImageDraw.Draw(im)
            s1, s2 = p["info"]["check"]["replay"]
            d.rounded_rectangle([10, 10, p["w"] - 10, 64], radius=8, fill=(12, 16, 24, 200)); d.text((22, 16), p["name"], font=K.font(17, True), fill=K.WARN if p["note"] else K.INK)
            d.text((22, 40), "G1, throw construction, seed 0" + ("" if not p["note"] else f"  ·  {p['note']}"), font=K.font(12), fill=K.SUB)
            y0 = K.H - 250; d.rounded_rectangle([10, y0, p["w"] - 10, K.H - 12], radius=10, fill=(12, 16, 24, 205))
            d.text((24, y0 + 10), f"{s1:.2f} m/s release speed", font=K.font(26, True), fill=K.ACC)
            d.text((24, y0 + 44), f"valid at 1 ms and 0.5 ms ({s1:.2f} / {s2:.2f})", font=K.font(13), fill=K.SUB)
            d.text((24, y0 + 68), "positive-work share by joint group", font=K.font(13, True), fill=K.INK)
            tot = sum(max(p["work"][g_], 0) for g_ in GROUPS)
            for i, g_ in enumerate(GROUPS):
                v = max(p["work"][g_], 0)/tot; yy = y0 + 92 + 24*i; d.text((24, yy), g_, font=K.font(13), fill=K.SUB)
                d.rectangle([100, yy + 3, 100 + int((p["w"] - 190)*v), yy + 16], fill=(120, 190, 255) if g_ in ("legs", "trunk") else K.ACC); d.text((106 + int((p["w"] - 190)*v), yy), f"{100*v:.0f}%", font=K.font(12), fill=K.INK)
            canvas.paste(im, (x, 0)); x += p["w"]
        d = ImageDraw.Draw(canvas); [d.line([(xx, 0), (xx, K.H)], fill=(10, 12, 16), width=3) for xx in (ws[0], ws[0] + ws[1])]
        canvas = K.badge(canvas, fac, tau, x=K.W//2 - 120, y=80, extra="(0 = release)", scale=0.85)
        canvas = K.note(canvas, "what-if bodies are counterfactuals, not robot capability  ·  actuator lag 40 ms  ·  ball = 0.145 kg point mass in the hand", (K.W//2 - 420, 400), color=(235, 235, 240), size=14)
        return canvas
    c = frame(-pre, 1.0); W.add(c, hold=30)
    for tau, fac in tl: c = frame(tau, fac); W.add(c)
    W.add(c, hold=60); return W.close()


# ---------------------------------------------------------------- 5. HEAD-CARRY ablation
HC = {"t1pot": dict(out="headcarry_T1_pot40_law_on_off.mp4", head="HEAD CARRY  ·  40 kg pot, arm-balance law on vs off (track T1)",
                    src="replayed from results/overnight/T1/head_pot_40kg_t1.json (+ its law-off ablation)",
                    cav=["ZERO actuator lag (superseded by track Q1)", "this z fails a 10 ms lag replay (0/6 zero-lag maxima survive)", "load placed at rest at t = 0 by a helper",
                         "rigid load contact (solref 0.002)", "servos as torque motors at stock limits, back-EMF brake"]),
      "q1box": dict(out="headcarry_Q1_box32_lag10ms_law_on_off.mp4", head="HEAD CARRY  ·  32 kg box under a 10 ms actuator lag, law on vs off (track Q1)",
                    src="replayed from results/overnight/Q1/lag_maxima.json head_box_lag10ms (+ its law-off ablation)",
                    cav=["actuator lag 10 ms with torque-speed droop", "servo-gain cap (controller choice)", "load placed at rest at t = 0 by a helper",
                         "rigid load contact (solref 0.002)"])}


def headcarry(case):
    c = HC[case]; pn = []
    for law in ("on", "off"):
        info, Q, T = load(f"hc_{case}_{law}"); pn.append(dict(info=info, Q=Q, T=T, st=K.Studio(f"{CACHE}/hc_{case}_{law}.mjb", w=640, h=K.H, fog=(10.0, 40.0)), law=law))
    on = pn[0]["info"]; tl = K.timeline(10.0, windows=[(0.0, 1.3, 0.25)], base=2.0); W = K.Writer(f"{OUT}/{c['out']}")
    def frame(t, fac):
        canvas = Image.new("RGBA", (K.W, K.H))
        for i, p in enumerate(pn):
            Q, T, st = p["Q"], p["T"], p["st"]; st.set(K.track_q(Q, T, t)); im = K.finish(st.frame([Q[0, 0], Q[0, 1], 1.0], 2.9, 150.0, -8.0, key_target=[Q[0, 0], Q[0, 1], 1.0]))
            ch = p["info"]["check"]["replay"]; d = ImageDraw.Draw(im)
            lt = "LAW ON" if p["law"] == "on" else "LAW OFF (ablation: same z, same plant)"; d.rounded_rectangle([14, 100, 42 + d.textlength(lt, font=K.font(18, True)), 140], radius=8, fill=(12, 16, 24, 200))
            d.text((28, 108), lt, font=K.font(18, True), fill=(140, 230, 160) if p["law"] == "on" else K.WARN)
            if p["law"] == "on": val, vl = f"{on['mass']:g} kg {on['obj']} held {ch[0][0]:.0f} s", f"valid at 1 ms and 0.5 ms (pelvis drift {1000*ch[0][2]:.1f} / {1000*ch[1][2]:.1f} mm)"
            else: val, vl = f"drops at {ch[0][0]:.2f} s", f"same at 1 ms and 0.5 ms ({ch[0][0]:.2f} / {ch[1][0]:.2f} s, load off the head patch)"
            im = K.lower_third(im, f"Unitree G1H (stock, 34.4 kg)  ·  {on['mass']:g} kg {on['obj']} on the head", val, vl, c["cav"] if p["law"] == "on" else [], x=16, scale=0.8, width=608)
            if t > T[-1] + 1e-9: im = K.note(im, f"load left the head patch at {T[-1]:.2f} s  ·  rollout ends", (16, 150), size=16)
            canvas.paste(im, (640*i, 0))
        ImageDraw.Draw(canvas).line([(640, 0), (640, K.H)], fill=(10, 12, 16), width=3)
        canvas = K.title(canvas, c["head"], c["src"], x=16, y=16); canvas = K.badge(canvas, fac, t)
        return canvas
    cv = frame(0.0, 1.0); W.add(cv, hold=30)
    for t, fac in tl: cv = frame(t, fac); W.add(cv)
    W.add(cv, hold=45); return W.close()


if __name__ == "__main__":
    job = sys.argv[1]
    if job == "headcarry": res = [headcarry("t1pot"), headcarry("q1box")]
    else: res = [globals()[job]()]
    for out, dur in res: print(f"wrote {out} ({dur:.1f} s)", flush=True)
