"""Cinematic render of the M3b balanced box jump (2026-10-20), studio kit engine/cinematic.py.
capture: replay the chosen search vector on FRESH plants at 1 ms and 0.5 ms through engine.boxjump2.rollout (the validation path),
         recording qpos and floor/box normal force every physics step; the replayed verdict and touchdown numbers are compared with
         the stored validation row (refused on mismatch).
render:  kinematic replay of the 1 ms record -> results/demo/cinematic/M3b_box_jump/<tag>.mp4 (+ stills), 1280x720, slow motion
         through the dip, push, flight and landing, label with height, robot, 'valid at 1 ms and 0.5 ms' and robustness.
usage (from src/): python cinematic_m3b.py capture <tag> <validation.json> | render <tag>"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = f"{ROOT}/results/demo/cinematic/cache"; OUT = f"{ROOT}/results/demo/cinematic/M3b_box_jump"
import numpy as np, mujoco

REC = dict(on=False, rows=[], ids=None)
_orig_step = mujoco.mj_step
def _step(m, d):
    _orig_step(m, d)
    if not REC["on"]: return
    if REC["ids"] is None: REC["ids"] = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor"), mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box"))
    fl, bx = REC["ids"]; ff = fb = 0.0; fv = np.zeros(6)
    for ci in range(d.ncon):
        c = d.contact[ci]
        if fl in (c.geom1, c.geom2): mujoco.mj_contactForce(m, d, ci, fv); ff += fv[0]
        elif bx in (c.geom1, c.geom2): mujoco.mj_contactForce(m, d, ci, fv); fb += fv[0]
    REC["rows"].append((float(d.time), ff, fb, d.qpos.copy(), d.subtree_com[1].copy()))
mujoco.mj_step = _step


def capture(tag, vfile):
    from engine import boxjump2 as J, boxjump as BJ
    V = json.load(open(vfile)); robot, h = V["robot"], V["h"]; u = np.asarray(V["u"], float); info = dict(tag=tag, vfile=vfile, robot=robot, h=h, lag=V["lag"])
    for dt, key in ((0.001, "m1"), (0.0005, "m05")):
        S = J.make_plant(robot, h, dt); REC.update(on=True, rows=[], ids=None); o = J._clean(J.rollout(S, J.par_from(u))); REC["on"] = False
        st = V["nominal"][key]; mism = [k for k in ("valid", "failed", "touch_on", "hops") if o[k] != st[k]]
        mism += [f"td.{k}" for k in ("vx", "cp_to_toe", "heel_margin") if abs(o["td"][k] - st["td"][k]) > 1e-6]
        info[key] = dict(replay=o, match=not mism, mismatches=mism)
        if dt == 0.001:
            rows = REC["rows"]; t = np.array([r[0] for r in rows]); br = np.where(np.diff(t) < 0)[0]; rows = rows[br[-1] + 1:] if len(br) else rows
            np.savez_compressed(f"{CACHE}/{tag}_rec.npz", T=np.array([r[0] for r in rows]), FF=np.array([r[1] for r in rows]), FB=np.array([r[2] for r in rows]),
                                Q=np.array([r[3] for r in rows]), COM=np.array([r[4] for r in rows]))
            info["scene"] = BJ.scene(robot, h, J.geometry(robot)["D"])[0]; info["edge"] = float(S["edge_x"]); info["T_PRE"] = J.T_PRE; info["t_dip"] = J.par_from(u)["t_dip"]
    info["robust"] = V["robust"]; info["n_pass"] = V["n_pass"]; info["perturb"] = [(r["name"], r["valid_both"]) for r in V["perturb"]]
    json.dump(info, open(f"{CACHE}/{tag}.json", "w"), indent=1, default=float)
    print(tag, "match 1ms", info["m1"]["match"], info["m1"]["mismatches"], "0.5ms", info["m05"]["match"], info["m05"]["mismatches"])


def ease(x):
    x = float(np.clip(x, 0, 1)); return x*x*(3 - 2*x)


def render(tag):
    from engine import cinematic as K
    I = json.load(open(f"{CACHE}/{tag}.json")); assert I["m1"]["match"] and I["m05"]["match"], "replay does not reproduce the stored validation row"
    z = np.load(f"{CACHE}/{tag}_rec.npz"); T, Q, COM, FF, FB = z["T"], z["Q"], z["COM"], z["FF"], z["FB"]
    r1, r2 = I["m1"]["replay"], I["m05"]["replay"]; robot, h = I["robot"], I["h"]; to = r1["takeoff"]; td = r1["td"]; w = r1["window"]
    t_td = r1["touch_t"]; t_to = r1["takeoff_t"]; t_end = float(T[-1]); t_dip = I["T_PRE"]
    st = K.Studio(I["scene"], fog=(10.0, 50.0))
    who = {"G1": "Unitree G1 (stock Menagerie model)", "H1": "Unitree H1 (stock Menagerie model)"}[robot]
    task = f"{who}  ·  BOX JUMP onto a {h:.2f} m rigid box, land on top, stand balanced"
    value = f"held {w['T']:.1f} s on the box  ·  foot travel {1000*w['foot_travel']:.1f} mm  ·  takeoff vx {to['vx']:.2f} / vz {to['vz']:.2f} m/s"
    rob = f"ROBUST {I['n_pass']}/4 fresh perturbations" if I["robust"] else f"NOT ROBUST: {I['n_pass']}/4 fresh perturbations (needs 3/4)"
    valid = f"valid at 1 ms and 0.5 ms (fresh plants)  ·  {rob}  ·  10 ms actuator lag on every joint"
    cav = [f"toes {100*0.075:.1f} cm behind the edge (scene standard)", f"touchdown vx {td['vx']:.2f} m/s, capture point {100*td['cp_to_toe']:.1f} cm inside the toe",
           f"heel {100*td['heel_margin']:.1f} cm past the edge", {"G1": "published ratings (knee 120, ankle/waist 35, hip roll 88 N·m)", "H1": "published ratings (URDF effort: knee 300, hip 200, ankle 40 N·m)"}[robot],
           "in-flight foot placement at the capture point", "no hop, no hands/shins on the box, no joint stop > 0.05 rad"]
    windows = [(t_dip - 0.05, t_td + 0.9, 0.25), (t_td + 2.5, t_end, 2.0)]
    def at(A, t): return K.track_q(A, T, t)
    def cam(t):
        c = at(COM, t); az = 125.0 - 20*ease((t - t_dip)/2.0); zc = float(COM[0][2])
        return [0.5*(c[0] + I["edge"] + 0.3), 0.0, h + 0.95*zc], 2.4 + 1.8*zc, az, -8.0
    def frame(t, fac):
        st.set(at(Q, t)); la, d0, az, el = cam(t)
        im = K.finish(st.frame(la, d0, az, el, key_target=[la[0], 0, 0.8]))
        im = K.title(im, f"BALANCED BOX JUMP  ·  {robot} onto {h:.2f} m", "replayed on a fresh plant at 1 ms (0.5 ms replay identical in verdict)")
        im = K.lower_third(im, task, value, valid, cav, scale=0.70); im = K.badge(im, fac, t)
        ph = "stand" if t < t_dip else ("dip + push" if t < t_to else ("flight" if t < t_td else ("absorb + stand up" if t < t_td + 1.6 else "balance hold")))
        fb = float(np.interp(t, T, FB)); ff = float(np.interp(t, T, FF))
        im = K.note(im, f"{ph}   ·   floor force {ff:5.0f} N   box force {fb:5.0f} N", (40, 92), color=(140, 200, 255), size=16)
        return im
    os.makedirs(OUT, exist_ok=True); Wr = K.Writer(f"{OUT}/{tag}.mp4"); stills = {}; im = None
    Wr.add(frame(0.0, 1.0), hold=20)
    for t, fac in K.timeline(t_end, windows=windows, base=1.0):
        im = frame(t, fac); Wr.add(im)
        for k, tt in (("dip", t_to - 0.1), ("flight", 0.5*(t_to + t_td)), ("touchdown", t_td + 0.02), ("hold", t_td + 2.0)):
            if k not in stills and t >= tt: stills[k] = im
    stills["end"] = im; Wr.add(im, hold=60); out, dur = Wr.close()
    for k, v in stills.items(): v.convert("RGB").save(f"{OUT}/{tag}_{k}.jpg", quality=88)
    print(out, f"{dur:.1f} s")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
    if sys.argv[1] == "capture": capture(sys.argv[2], sys.argv[3])
    else: render(sys.argv[2])
