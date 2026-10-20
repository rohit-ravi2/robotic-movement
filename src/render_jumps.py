"""Render the G1 jump strategies side by side (same fixed side camera, aligned at takeoff):
NATIVE optimum, APRIOR (analytic-prior) optimum, and a HUMAN jump (LAFAN1 jumps1_subject1, GMR retarget, kinematic
playback on G1). Physics runs are replayed faithfully at 1 ms from the stored baseline9 solutions. Also writes
kinematic descriptors per strategy. Needs MUJOCO_GL=glfw (desktop display)."""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, branch_i as BI, mocap_baseline as MB
from PIL import Image, ImageDraw, ImageFont
OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/viz"); CLIP = "obstacles5_subject3@1585"; FPS = 30

def record_physics(r):
    c = MB.load_clip(r["clip"], "retargeted_g1_all.pkl"); ref = MB.analytic_clip(c) if r["method"] == "APRIOR" else c
    S = BI.setup(c["start"]); BI.apply_dt(S, 0.001)
    tr = MB.Tracker(S, ref, omega=r["omega"], tscale=r["tscale"]); Q = []
    def fn(t, d, S_): Q.append(d.qpos.copy()); return tr(t, d, S_)
    o = BI.rollout(np.array(r["x"]), set(S["types"]), None, S=S, ctrl_fn=fn, horizon=ref["horizon"], score="best", record=True)
    BI.apply_dt(S, 0.004)
    return np.array(Q), 0.001, o["takeoff_t"], o

def record_human():
    D = pickle.load(open(f"{MB.BASE}/retargeted_g1_all.pkl", "rb"))[CLIP]
    Q = np.array(D["qpos"]); fps = D["fps"]; a = D["takeoff_frame"] - D["frame0"]
    return Q, 1.0/fps, a/fps

def descriptors(m, Q, dt, t_to, label):
    j = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
    d = mujoco.MjData(m); com = []
    for q in Q:
        d.qpos[:] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); com.append(d.subtree_com[0].copy())
    com = np.array(com); i_to = min(int(round(t_to/dt)), len(Q)-1); i0 = max(0, i_to - int(round(1.0/dt)))   # last 1.0 s before takeoff
    pre = slice(i0, i_to+1)
    knee = Q[pre, j("left_knee_joint")]; hip = Q[pre, j("left_hip_pitch_joint")]; sh = Q[:i_to+1, j("left_shoulder_pitch_joint")]
    zmin_i = int(np.argmin(com[pre, 2]))
    vz = np.gradient(com[:, 2], dt)
    # push-off = from the lowest COM point to takeoff
    zmin_i += i0
    return dict(label=label, crouch_depth_cm=100*float(com[i0, 2] - com[pre, 2].min()), max_knee_flex_deg=float(np.degrees(knee.max())),
                max_hip_flex_deg=float(np.degrees(-hip.min())), shoulder_swing_deg=float(np.degrees(sh.max() - sh.min())),
                pushoff_ms=1000*float((i_to - zmin_i)*dt), takeoff_vz=float(vz[i_to]), countermovement=bool(zmin_i > i0 + 2 and com[i0, 2] - com[pre, 2].min() > 0.02))

def render(m, Q, dt, t_to, w=420, h=560, pre=0.9, post=0.25):
    m.vis.global_.offheight = max(m.vis.global_.offheight, h); m.vis.global_.offwidth = max(m.vis.global_.offwidth, w)
    r = mujoco.Renderer(m, h, w); d = mujoco.MjData(m)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    q0 = Q[0][3:7]; yaw = np.degrees(np.arctan2(2*(q0[0]*q0[3] + q0[1]*q0[2]), 1 - 2*(q0[2]**2 + q0[3]**2)))
    cam.lookat[:] = [Q[0][0], Q[0][1], 0.95]; cam.distance = 3.8; cam.azimuth = yaw + 90; cam.elevation = -6
    opt = mujoco.MjvOption()
    frames = []
    for t in np.arange(-pre, post + 1e-9, 1.0/FPS):
        i = int(np.clip(round((t_to + t)/dt), 0, len(Q)-1)); d.qpos[:] = Q[i]; mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam, scene_option=opt); frames.append(r.render().copy())
    return frames

def label(img, txt, sub):
    im = Image.fromarray(img); dr = ImageDraw.Draw(im)
    try: f = ImageFont.truetype("DejaVuSans-Bold.ttf", 20); fs = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception: f = fs = ImageFont.load_default()
    dr.rectangle([0, 0, im.width, 52], fill=(0, 0, 0)); dr.text((10, 5), txt, fill=(255, 255, 255), font=f); dr.text((10, 30), sub, fill=(200, 200, 200), font=fs)
    return np.array(im)

if __name__ == "__main__":
    B = json.load(open(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase9/baseline9.json")))
    nat = max([r for r in B if r["method"] == "OURS" and r["clip"] == CLIP and r["valid"] and r["task"] == "T2"], key=lambda r: r["v"])
    apr_c = [r for r in B if r["method"] == "APRIOR" and r["clip"] == CLIP and r["valid"]]
    apr = max([r for r in apr_c if r["rot"] <= 0.35] or apr_c, key=lambda r: r["v"])
    S = BI.setup(); m = S["m"]
    runs = []
    for name, r in (("NATIVE optimum", nat), ("ANALYTIC-PRIOR optimum", apr)):
        Q, dt, tto, o = record_physics(r); runs.append((name, Q, dt, tto, f"takeoff {o['v']:.2f} m/s, crouch {100*o['mech']['crouch_depth']:.0f} cm"))
    Qh, dth, ttoh = record_human(); runs.append(("HUMAN jump (LAFAN1)", Qh, dth, ttoh, "retargeted to G1, kinematic playback"))
    desc = [descriptors(m, Q, dt, tto, name) for name, Q, dt, tto, _ in runs]
    json.dump(desc, open(f"{OUT}/jump_descriptors.json", "w"), indent=1)
    for dd in desc: print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in dd.items()})
    panels = [[label(f, name, sub) for f in render(m, Q, dt, tto)] for name, Q, dt, tto, sub in runs]
    tt = np.arange(-0.9, 0.25 + 1e-9, 1.0/FPS)
    n = min(len(p) for p in panels); os.makedirs(f"{OUT}/frames", exist_ok=True)
    for i in range(n):
        row = Image.fromarray(np.concatenate([p[i] for p in panels], axis=1)); dr = ImageDraw.Draw(row)
        try: fb = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        except Exception: fb = ImageFont.load_default()
        msg = f"t = {tt[i]:+.2f} s  " + ("(push-off)" if tt[i] < -0.02 else "TAKEOFF" if abs(tt[i]) < 0.02 else "(early flight; flight is not optimized)")
        dr.rectangle([0, row.height - 36, row.width, row.height], fill=(0, 0, 0)); dr.text((12, row.height - 32), msg, fill=(255, 220, 120), font=fb)
        row.save(f"{OUT}/frames/f{i:04d}.png")
    print("frames", n)
