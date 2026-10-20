"""Cinematic renders of M2_throw_target R2 results (2026-10-19), studio kit engine/cinematic.py.
capture: replay a stored ledger on FRESH plants at 1 ms (recorded) and 0.5 ms (optionally on a perturbation draw), check both landing errors
against the stored record (ledger meas for nominal, the physics verifier's rows for a draw), save the qpos trace + ball flight + model.
render:  kinematic replay of the cache (refused unless the check matched) -> results/demo/cinematic/M2_throw_target/<tag>.mp4 (+ a still).
usage: python cinematic_m2.py capture <tag> | render <tag> | all"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco
from engine import throw_target as TT, cinematic as K

ROOT = TT.ROOT; CACHE = f"{ROOT}/results/demo/cinematic/cache"; OUT = f"{ROOT}/results/demo/cinematic/M2_throw_target"
R2 = f"{TT.OUT}/r2"; VER = f"{TT.OUT}/verify_physics_r2"
ITEMS = {
    "m2_d10_straight_s26": dict(ledger=f"{R2}/d10_straight_soft_error_s26.json", pert=None, kind="hit",
                                robust="robust: 15/16 draws (101-116) at both dt; pooled 21/24 incl. verifier draws (2/4 on 81-84)"),
    "m2_d10_straight_s26_draw84": dict(ledger=f"{R2}/d10_straight_soft_error_s26.json", pert=84, kind="fail", ver=f"{VER}/d10.json", nominal="m2_d10_straight_s26",
                                       robust="same throw: 15/16 on draws 101-116, pooled 21/24; this draw is one of the 3 misses"),
    "m2_d15_straight_s50": dict(ledger=f"{R2}/d15_straight_soft_error_s50.json", pert=None, kind="nominal_only",
                                robust="NOT robust: 7/16 draws (101-116); pooled 9/20; worst-draw miss 1.4 m"),
}


def stored(item):
    L = json.load(open(item["ledger"]))
    if item["pert"] is None: return [m_["err"] for m_ in L["meas"]], [m_["valid"] for m_ in L["meas"]]
    for e in json.load(open(item["ver"])):
        if os.path.basename(e["path"]) == os.path.basename(item["ledger"]):
            rows = [r for r in e["rows"] if r.get("pert") == item["pert"]]; rows.sort(key=lambda r: -r["dt"])
            return [r["err"] for r in rows], [r["valid"] for r in rows]
    raise KeyError(item)


def capture(tag):
    it = ITEMS[tag]; L, C = TT.load_ledger(it["ledger"]); z = np.array(L["z"]); pert = None if it["pert"] is None else dict(TT.PERTS[0], seed=it["pert"])
    st_err, st_valid = stored(it); rs = []
    for dt in TT.DTS:
        S = TT.plant(dt, pert=pert); r = C.run(S, z, record=(dt == TT.DTS[0]))
        if pert: r = TT._v2(r, S)
        rs.append((S, r))
    rep = [round(float(r["err"]), 4) for _, r in rs]; val = [bool(r["valid"]) for _, r in rs]
    chk = dict(stored=[round(float(e), 4) for e in st_err], replay=rep, stored_valid=st_valid, valid=val,
               match=bool(all(abs(a - b) < 2e-3 for a, b in zip(rep, st_err)) and val == list(st_valid)))
    S, r = rs[0]; tr = r["trace"]; dt = float(tr["dt"]); Q = np.asarray(tr["q"]); n = len(Q); T = (np.arange(n) + 1)*dt
    bt = tr["ball_traj"]; t_end = n*dt; Tb = np.array([t_end] + [t_end + dt*(1 + 10*k) for k in range(len(bt) - 1)])
    qa = S["ball_qadr"]; Pb = np.vstack([Q[:, qa:qa + 3], bt[1:], [r["land"]]]); Tb = np.concatenate([T, Tb[1:], [Tb[-1] + 10*dt]])
    m = S["m"]; mujoco.mj_saveModel(m, f"{CACHE}/{tag}.mjb", None); np.savez(f"{CACHE}/{tag}.npz", Q=Q, t=T, Pb=Pb, Tb=Tb)
    d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d)
    rows = [dict(dt=s_["m"].opt.timestep, valid=r_["valid"], reasons=r_["reasons"], err=round(r_["err"], 4), speed=round(r_["speed"], 3), elev=round(r_["elev"], 2), az=round(r_["az"], 2),
                 slip=round(r_["slip"], 4), land=r_["land"]) for s_, r_ in rs]
    info = dict(src=os.path.relpath(it["ledger"], ROOT), tag=tag, pert=pert, pert_draw=S.get("pert_draw"), check=chk, rows=rows, target_xy=list(map(float, C.xy)), hit_r=TT.HIT_R,
                dist=L["spec"]["dist"], az_deg=float(np.degrees(L["spec"]["az"])), t_rel=float(r["t_rel"]), t_end=t_end, release=r["release"], land=r["land"], speed=float(r["speed"]),
                elev=float(r["elev"]), az=float(r["az"]), lag_ms=1000*float(S["lag"]), ball_geom=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom"), ball_qadr=qa, kind=it["kind"],
                robust=it["robust"], stance=float(np.mean([d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)][0] for b in ("left_ankle_roll_link", "right_ankle_roll_link")])))
    json.dump(info, open(f"{CACHE}/{tag}.json", "w"), indent=1, default=float); print(tag, chk, flush=True); return info


def lerp(a, b, w): return np.asarray(a, float)*(1 - w) + np.asarray(b, float)*w
def ease(x): x = float(np.clip(x, 0, 1)); return x*x*(3 - 2*x)


def render(tag):
    info = json.load(open(f"{CACHE}/{tag}.json")); assert info["check"]["match"], f"{tag}: replay does not reproduce the stored value: {info['check']}"
    z = np.load(f"{CACHE}/{tag}.npz"); Q, T, Pb, Tb = z["Q"], z["t"], z["Pb"], z["Tb"]
    st = K.Studio(f"{CACHE}/{tag}.mjb", fog=(45.0, 140.0)); m = st.m; qa = info["ball_qadr"]; bg = info["ball_geom"]; m.geom_rgba[bg] = K.BALL_RGBA
    txy = np.array(info["target_xy"]); land = np.array(info["land"]); t_rel, t_end = info["t_rel"], info["t_end"]; t_land = float(Tb[-2])
    e1, e2 = info["check"]["replay"]; kind = info["kind"]; hit = all(r["valid"] for r in info["rows"]) and max(e1, e2) <= info["hit_r"]
    nom = nom_err = None
    if kind == "fail": ni = json.load(open(f"{CACHE}/{ITEMS[tag]['nominal']}.json")); nom = np.array(ni["land"]); nom_err = ni["check"]["replay"][0]
    x0 = np.array(Q[0, :2]); rng = float(np.linalg.norm(txy - x0)); fwd = (txy - x0)/rng
    def ball(t): return np.array([np.interp(t, Tb, Pb[:, k]) for k in range(3)])
    def robot(t): return K.track_q(Q, T, min(t, T[-1]))
    hist = []
    def scene_geoms(t, b):
        g = [("cyl", [txy[0], txy[1], 0.003], [info["hit_r"], 0.003, 0], [0.86, 0.16, 0.16, 0.42]),        # hit circle r 0.5 m
             ("cyl", [txy[0], txy[1], 0.005], [0.30, 0.003, 0], [0.97, 0.95, 0.92, 0.55]), ("cyl", [txy[0], txy[1], 0.007], [0.10, 0.003, 0], [0.86, 0.16, 0.16, 0.85])]
        if t > t_rel: g += [("sphere", b, [0.075]*3, [1.0, 0.55, 0.15, 0.22])]                                   # soft halo (visual only)
        P = np.array(hist[-16:])
        for i, p in enumerate(P[:-1]): a = (i + 1)/len(P); g.append(("sphere", p, [0.035*(0.3 + 0.7*a)]*3, [1.0, 0.5, 0.12, 0.6*a**1.5]))
        if len(hist) > 2 and t > t_rel: g += [("sphere", p, [0.022]*3, [1.0, 0.6, 0.2, 0.30]) for p in hist[:-16:3]]   # the arc so far
        if t >= t_land - 1e-9:
            g += [("cyl", [land[0], land[1], 0.009], [0.07, 0.002, 0], [0.1, 0.1, 0.12, 0.9])]
        if nom is not None and t >= t_land - 1e-9: g += [("cyl", [nom[0], nom[1], 0.009], [0.07, 0.002, 0], [0.25, 0.75, 1.0, 0.85])]
        return g
    who = "Unitree G1 (stock Menagerie MJX)  ·  throw a free 0.145 kg ball at a floor target"
    where = f"{info['dist']:.0f} m " + ("straight ahead" if abs(info["az_deg"]) < 1 else f"{info['az_deg']:.0f} deg left")
    if kind == "fail":
        value = f"MISS: lands {e1:.2f} m from centre (hit radius 0.5 m)"; valid = f"valid throw at 1 ms and 0.5 ms ({e1:.3f} / {e2:.3f} m), misses at both"
        ttl = ("FAILURE CASE  ·  " + where + ", perturbation draw 84 (verifier)", f"same command as the hit; lag {info['lag_ms']:.1f} ms (+{info['lag_ms']/10 - 1:.0%}), friction x{info['pert_draw']['friction_scale']:.2f}, mass/start noise")
    else:
        value = f"lands {e1:.2f} m from centre  ·  {info['speed']:.2f} m/s at {info['elev']:.0f} deg"; valid = f"valid at 1 ms and 0.5 ms  (error {e1:.3f} / {e2:.3f} m; hit radius 0.5 m)"
        ttl = ("THROW AT A TARGET  ·  " + where, "replayed from " + info["src"] + (" (nominal plant)" if kind != "fail" else ""))
    cav = ["actuator lag 10 ms" if info["pert"] is None else f"actuator lag {info['lag_ms']:.1f} ms (draw 84)", "rated torques: knee 120, ankle/waist roll-pitch 35, wrist 5 N·m",
           "LABELLED: ball held by a weld (no grip-force limit), released by switching it off", info["robust"],
           "flight after the 0.3 s follow-through: ball+floor twin, vacuum (no drag)", "robot held at its last simulated pose after the follow-through", "stock contact; solver iterations 50"]
    if kind == "nominal_only": cav.insert(0, "nominal hit only")
    def frame(t, fac, cam, flight_note=True):
        b = ball(t) if t > 0 else Pb[0]; hist.append(b.copy()); q = robot(t).copy(); q[qa:qa + 3] = b; q[qa + 3:qa + 7] = [1, 0, 0, 0]; st.set(q)
        im = K.finish(st.frame(*cam, geoms=scene_geoms(t, b), key_target=[x0[0] + 0.3, x0[1], 0.6]))
        im = K.title(im, *ttl); im = K.lower_third(im, who, value, valid, cav, scale=0.85); im = K.badge(im, fac, t)
        if flight_note and t >= t_rel:
            s = f"release {info['speed']:.2f} m/s, elev {info['elev']:.1f} deg, {info['release']['ball_p'][2]:.2f} m high"
            if t >= t_land - 1e-9: s += f"  ·  landed {e1:.2f} m from centre" + ("  (HIT)" if hit else "  (MISS)")
            im = K.note(im, s, (40, 100), color=(K.ACC if hit else K.WARN))
        if nom is not None and t >= t_land - 1e-9: im = K.note(im, f"blue mark: nominal-plant landing of the same throw ({nom_err:.2f} m from centre)", (40, 146), color=(140, 200, 255))
        return im
    W = K.Writer(f"{OUT}/{tag}.mp4")
    c0 = ([x0[0] + 0.25, x0[1], 0.62], 3.6, 135.0, -8.0)
    W.add(frame(0.0, 1.0, c0), hold=30)
    tl = K.timeline(t_land, windows=[(0.0, t_rel - 0.06, 0.2), (t_rel - 0.06, t_rel + 0.05, 0.1), (t_rel + 0.05, t_end, 0.25)], base=0.5)
    mid_w = ([0.5*(x0[0] + txy[0]), 0.5*(x0[1] + txy[1]), 1.6], 0.55*rng + 6.5, 100.0, -9.0)
    stills = {}
    for t, fac in tl:
        if t < t_end: a = ease(t/t_end); cam = (lerp(c0[0], [x0[0] + 0.4, x0[1], 0.75], a), 3.6 + 0.6*a, 135.0 - 30*a, -8.0)
        else:
            w = ease((t - t_end)/(t_land - t_end)); b = ball(t); trk = ([b[0] - 1.5*fwd[0], b[1] - 1.5*fwd[1], max(0.6, 0.6*b[2])], 5.0 + 3.0*w, 105.0 - 5*w, -6.0)
            k = ease((t - t_end)/0.25); cam = (lerp([x0[0] + 0.4, x0[1], 0.75], trk[0], k), 4.2 + (trk[1] - 4.2)*k, 105.0 - 5*w*k, -8 + 2*k)
        im = frame(t, fac, cam); W.add(im)
        if abs(t - t_rel) < 0.006 and "release" not in stills: stills["release"] = im
    last = cam
    for k in range(45): w = ease(k/44); W.add(frame(t_land, 1.0, (lerp(last[0], mid_w[0], w), last[1] + (mid_w[1] - last[1])*w, last[2] + (mid_w[2] - last[2])*w, last[3] + (mid_w[3] - last[3])*w)))
    im = frame(t_land, 1.0, mid_w); W.add(im, hold=45); stills["wide"] = im
    close = ([txy[0] - 0.2*fwd[0], txy[1] - 0.2*fwd[1], 0.0], 3.4, 120.0, -42.0)
    for k in range(40): w = ease(k/39); W.add(frame(t_land, 1.0, (lerp(mid_w[0], close[0], w), mid_w[1] + (close[1] - mid_w[1])*w, mid_w[2] + (close[2] - mid_w[2])*w, mid_w[3] + (close[3] - mid_w[3])*w)))
    im = frame(t_land, 1.0, close); W.add(im, hold=75); stills["landing"] = im
    out, dur = W.close()
    for k, v in stills.items(): v.convert("RGB").save(f"{OUT}/{tag}_{k}.jpg", quality=88)
    print(out, f"{dur:.1f} s", flush=True); return out, dur


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); cmd = sys.argv[1]
    if cmd == "capture": capture(sys.argv[2])
    elif cmd == "render": render(sys.argv[2])
    elif cmd == "all":
        for t in ITEMS: capture(t)
        for t in ITEMS: render(t)
