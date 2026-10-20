"""Render the movement progression (every stage replayed faithfully at 1 ms from its stored solution) as one labelled
clip per stage, plus progression.json with what changed at each stage and the replayed metrics.
Tracks: G1 jump, H1 jump, G1 throw. Needs MUJOCO_GL=glfw (desktop display)."""
import sys, os, json, subprocess, shutil, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, branch_i as BI, mocap_baseline as MB, aprior_standard as AS
from PIL import Image, ImageDraw, ImageFont
R = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results"); OUT = f"{R}/viz/progression"; FPS = 30; SLOW = 0.5
TMP = "/tmp/prog_frames"


def logged(fn, log, n_out):
    def f(t, d, S_):
        log.append((t, d.qpos.copy()))
        return fn(t, d, S_) if fn is not None else np.zeros(n_out)
    return f


def rec_baseline9(r):
    c = MB.load_clip(r["clip"], "retargeted_g1_all.pkl"); ref = MB.analytic_clip(c) if r["method"] == "APRIOR" else c
    S = BI.setup(c["start"]); BI.apply_dt(S, 0.001); log = []
    tr = MB.Tracker(S, ref, omega=r["omega"], tscale=r["tscale"])
    o = BI.rollout(np.array(r["x"]), set(S["types"]), None, S=S, ctrl_fn=logged(tr, log, 0), horizon=ref["horizon"], score="best")
    BI.apply_dt(S, 0.004); return log, o, S["m"]


def rec_margin(r):
    import jump13 as J13
    c = AS.clip0(); S = BI.setup(c["start"]); BI.apply_dt(S, 0.004); ref = MB.analytic_clip(c)
    om = MB.tune_gains(S, ref, 1.0)[0]; nT = len(S["types"]); ts = 0.87 + 0.3*r["z"][nT*ref["nk"]]
    BI.apply_dt(S, 0.001); log = []; tr = MB.Tracker(S, ref, omega=om, tscale=ts)
    o = BI.rollout(np.array(r["x"]), set(S["types"]), None, S=S, ctrl_fn=logged(tr, log, 0), horizon=ref["horizon"],
                   score="best", knot_dt=J13.KNOT_DT, stop_margin=J13.MARGIN)
    BI.apply_dt(S, 0.004); return log, o, S["m"]


def rec_landed(r, robot="G1"):
    import render_landing as RL
    T, Q, o, m = RL.record(r, robot); return list(zip(T, Q)), o, m


def rec_h1(r):
    import aprior_h1 as AH
    c = AH.clip0(); S = BI.setup(start=AH.Q0, robot="H1"); BI.apply_dt(S, 0.001)
    ref = MB.analytic_clip(c, robot="H1") if r["method"] == "APRIOR" else c; log = []
    tr = MB.Tracker(S, ref, omega=r["omega"], tscale=r["tscale"]) if r["omega"] else None
    o = MB.run(S, ref, np.array(r["x"]), r["omega"], r["tscale"]) if False else None
    o = BI.rollout(np.array(r["x"]), set(S["types"]), None, S=S, ctrl_fn=logged(tr, log, len(S["jarr"]["aid"])),
                   horizon=ref["horizon"], score="best")
    BI.apply_dt(S, 0.002); return log, o, S["m"]


def rec_throw(x, driven=None, prior=None):
    import throw as T
    S = T.setup("G1", driven=driven); BI.apply_dt(S, 0.001); log = []
    fn = T.tracker(S, T.scaled_ref(prior[0], prior[3]), prior[1], prior[2]) if prior else None
    o = T.rollout(np.array(x), S, ctrl_fn=logged(fn, log, len(S["types"])))
    BI.apply_dt(S, T.DT_OPT); return log, o, S["m"]


def render(m, log, cam, title, sub, path, t_end=None, w=560, h=620):
    T = np.array([t for t, _ in log]); Q = np.array([q for _, q in log])
    m.vis.global_.offheight = max(m.vis.global_.offheight, h); m.vis.global_.offwidth = max(m.vis.global_.offwidth, w)
    r = mujoco.Renderer(m, h, w); d = mujoco.MjData(m); c = mujoco.MjvCamera(); c.type = mujoco.mjtCamera.mjCAMERA_FREE
    q0 = Q[0][3:7]; yaw = np.degrees(np.arctan2(2*(q0[0]*q0[3] + q0[1]*q0[2]), 1 - 2*(q0[2]**2 + q0[3]**2)))
    c.lookat[:] = [Q[0][0], Q[0][1], cam.get("z", 1.0)]; c.distance = cam.get("dist", 4.2); c.azimuth = yaw + cam.get("az", 90); c.elevation = cam.get("el", -5)
    try: fb = ImageFont.truetype("DejaVuSans-Bold.ttf", 21); fs = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception: fb = fs = ImageFont.load_default()
    if os.path.isdir(TMP): shutil.rmtree(TMP)
    os.makedirs(TMP); t_end = t_end or T[-1]; k = 0
    for t in np.arange(0, t_end + 0.35, 1.0/FPS):
        i = min(int(np.searchsorted(T, min(t, T[-1]))), len(Q) - 1); d.qpos[:] = Q[i]; mujoco.mj_forward(m, d)
        r.update_scene(d, camera=c); im = Image.fromarray(r.render()); dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, w, 54], fill=(10, 16, 26)); dr.text((12, 6), title, fill=(240, 244, 250), font=fb); dr.text((12, 32), sub, fill=(170, 186, 205), font=fs)
        dr.rectangle([0, h - 28, w, h], fill=(10, 16, 26)); dr.text((12, h - 24), f"t = {min(t, T[-1]):.2f} s   ({SLOW:g}x speed)", fill=(255, 214, 140), font=fs)
        im.save(f"{TMP}/f{k:04d}.png"); k += 1
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS*SLOW), "-i", f"{TMP}/f%04d.png",
                    "-vf", "scale=560:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "31", path], check=True)


def main():
    os.makedirs(OUT, exist_ok=True); stages = []
    B9 = json.load(open(f"{R}/phase9/baseline9.json")); clip = "obstacles5_subject3@1585"
    best = lambda meth: max([r for r in B9 if r["method"] == meth and r["clip"] == clip and r["valid"]], key=lambda r: r["v"])
    J13 = json.load(open(f"{R}/phase13/jump13.json")); J14 = json.load(open(f"{R}/phase14/jump14.json"))
    jcam = dict(z=1.0, dist=4.4, az=90, el=-5)
    G1J = [
        ("g1j_1", "Human motion, retargeted and tracked", "Phase 9", "Start: a recorded human jump (LAFAN1) mapped onto the G1 by GMR, then followed by a joint tracker on the real robot dynamics. Nothing is optimised.", lambda: rec_baseline9(best("B0"))),
        ("g1j_2", "Native optimiser (no strategy given)", "Phase 9", "Changed: the human reference is removed; CMA-ES searches the joint-torque schedule from the robot's own physics. Finds a stiff, no-crouch jump that relies on a big arm swing.", lambda: rec_baseline9(best("OURS"))),
        ("g1j_3", "Optimiser started from the human motion", "Phase 9", "Changed: the same optimiser, but seeded with the retargeted human jump and free to leave it. At 8 seeds this ties with native search (+0.006 m/s): human motion is one starting strategy among many.", lambda: rec_baseline9(best("B1"))),
        ("g1j_4", "Optimiser started from the robot's own deep crouch", "Phase 9", "Changed: the starting strategy is computed from the robot's specification (its certified, work-maximising crouch), with no human data. The jump doubles: deep crouch, then full extension. Replicates in a second simulator (Drake).", lambda: rec_baseline9(best("APRIOR"))),
        ("g1j_5", "Deep crouch, joints kept off their stops", "Phase 13", "Changed: every joint must stay 0.10 rad away from its mechanical stop (hardware safety). Costs about 20% of takeoff speed. From the standard knees-bent start.", lambda: rec_margin(max([r for r in J13 if r["kind"] == "margin" and r["ok_all"]], key=lambda r: r["v"]))),
        ("g1j_6", "Deep crouch + balance/landing controller", "Phase 14", "Changed: at takeoff a balance controller takes over: soles kept parallel to the floor, quick knee bend to absorb the impact, trunk kept vertical, capture point kept over the feet. The push-off is re-optimised so only landed jumps count. Lands at the 1 ms step; not yet at 0.5 ms.", lambda: rec_landed(max([r for r in J14 if r["method"] == "APRIOR" and r["meas"][0]["ok"]], key=lambda r: r["meas"][0]["v"]))),
    ]
    H1 = json.load(open(f"{R}/phase10/aprior_h1.json"))
    h1best = lambda meth, task: max([r for r in H1 if r["method"] == meth and r["task"] == task and r["valid"]], key=lambda r: r["v"])
    H1J = [
        ("h1j_1", "H1: native optimiser", "Phase 10", "Second robot (Unitree H1: heavier, stronger, slower joints, no waist pitch). Native search from its certified standing start.", lambda: rec_h1(h1best("NATIVE", "T1"))),
        ("h1j_2", "H1: started from its own deep crouch", "Phase 10", "Changed: the H1's analytic crouch (from its own specification) as the starting strategy. +0.71 m/s; the strategy transfers to a second robot.", lambda: rec_h1(h1best("APRIOR", "T1"))),
        ("h1j_3", "H1: deep crouch, landable flight", "Phase 10", "Changed: takeoff rotation limited to 0.35 rad as a hard constraint (a landing proxy). Still +0.45 m/s over native; 8/8 pass the landing proxy.", lambda: rec_h1(h1best("APRIOR", "T2H"))),
    ]
    TL = json.load(open(f"{R}/phase11/throw_g1_ladder2400.json")); TD = json.load(open(f"{R}/phase11/throw_demand2.json"))
    prior = pickle.load(open(f"{R}/phase11/prior_G1.pkl", "rb"))
    ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_pitch_joint"]
    arm = max([r for r in TD if r["target"] is None and r["iters_arm"] == 2400 and r["arm"]["dt_check"][0]["valid"] and r["arm"]["dt_check"][0]["in_cone"]], key=lambda r: r["arm"]["speed"])
    tn = max([r for r in TL if r["method"] == "NATIVE" and r["valid_all_dt"]], key=lambda r: r["speed"])
    ta = max([r for r in TL if r["method"] == "APRIOR" and r["valid_all_dt"]], key=lambda r: r["speed"])
    tcam = dict(z=0.9, dist=3.6, az=125, el=-8)
    THR = [
        ("g1t_1", "Throw: arm only", "Phase 11", "Standing right-arm throw of a 0.145 kg ball. Legs, trunk and other arm locked; only the throwing arm is optimised. Tops out near 9-13 m/s.", lambda: rec_throw(arm["x_arm"], driven=ARM)),
        ("g1t_2", "Throw: whole body, native optimiser", "Phase 11", "Changed: legs, trunk and counter-arm unlocked (17 joints). The legs do about half the positive work and speed builds hips -> hand. Releases almost straight from rest.", lambda: rec_throw(tn["x"])),
        ("g1t_3", "Throw: whole body, robot-derived wind-up", "Phase 11", "Changed: started from a wind-up computed from the specification (trunk and arm cocked back, then swept forward). Every seed valid; ~+4 m/s at the largest budget (few seeds).", lambda: rec_throw(ta["x"], prior=(prior, ta["omega"], ta["tscale"], ta["amp"]))),
    ]
    for track, items, cam in (("G1 vertical jump", G1J, jcam), ("H1 vertical jump", H1J, jcam), ("G1 overarm throw", THR, tcam)):
        for sid, title, phase, changed, fn in items:
            log, o, m = fn()
            if "speed" in o:
                metric = f"ball {o['speed']:.1f} m/s at release"; extra = dict(ball_speed=o["speed"], t_release=o["t_rel"])
            else:
                landed = o.get("landing", {}).get("ok") if o.get("landing") else None
                metric = f"takeoff {o['v']:.2f} m/s, crouch {100*o['mech']['crouch_depth']:.0f} cm" + ("" if landed is None else f", landed {'yes' if landed else 'no'}")
                extra = dict(takeoff=o["v"], crouch_cm=100*o["mech"]["crouch_depth"], landed=landed)
            render(m, log, cam, title, f"{phase} · {metric}", f"{OUT}/{sid}.mp4")
            stages.append(dict(id=sid, track=track, title=title, phase=phase, changed=changed, metric=metric, **extra))
            print(sid, metric, flush=True)
    json.dump(stages, open(f"{OUT}/progression.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
