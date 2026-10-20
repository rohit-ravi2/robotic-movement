"""Render a sprint ladder rung as a clip with a tracking side camera (2026-10-16).
usage: python engine/render_sprint.py <ladder json> <rung index> <out.mp4> [title]
Replays the rung's stored z at 1 ms on the same plant (SPRINT_LAG / SCALE / SPRINT_SCALE_ANKLE must match the ladder's; act_tau and
scale are read from the json and checked), records qpos every 2 ms, and renders at 0.5x speed with the camera following the pelvis."""
import os, sys, json, shutil, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont
L = json.load(open(sys.argv[1])); ix = int(sys.argv[2]); out = sys.argv[3]
os.environ.setdefault("SPRINT_LAG", str(L["supplied"]["act_tau"]))
from engine import sprint as SP, core
assert abs(SP.ACT_TAU - float(L["supplied"]["act_tau"])) < 1e-9, "SPRINT_LAG differs from the ladder's act_tau"
rw = L["rungs"][ix]; robot = L["robot"]; scale = float(L.get("scale", 1.0))
st_ = rw["stride"] if rw.get("stride_extrapolated") else None
if st_ is not None and os.environ.get("RENDER_EXACT_STRIDE"):    # K1: the exact extrapolated stride (planner on the knee-139 body at extrapolated_from), not the rounded record
    from engine import sprint_ceiling as SC
    st_ = dict(SP.select_stride(SC.body_spec(robot), float(rw["stride"]["extrapolated_from"])), extrapolated_from=rw["stride"]["extrapolated_from"])
Pb = SP.Sprint(robot, rw["v"], scale, stride=st_)
for k_ in ("tc", "tf"): assert abs(Pb.st[k_] - rw["stride"][k_]) < 1e-3, f"stride {k_} differs from the stored rung (SCALE / SPRINT_SCALE_JOINTS must match the ladder)"
r = Pb.run(np.asarray(rw["z"]), core.DT_OPT, record=True); tr = r["trace"]; Q = np.asarray(tr["qpos"]); Tm = np.asarray(tr["t"])
print(f"replay {robot} rung {rw['v']}: value {r['value']:.4f} (stored {rw['meas'][0]['value']}) valid {r['valid']} reasons {r['reasons']} frames {len(Q)} ({Tm[-1]:.2f} s)")
assert abs(r["value"] - rw["meas"][0]["value"]) < 1e-3, "replay does not reproduce the stored 1 ms value"
r2 = Pb.run(np.asarray(rw["z"]), core.DT_CHECK); print(f"replay 0.5 ms: value {r2['value']:.4f} (stored {rw['meas'][1]['value']}) valid {r2['valid']}")
assert abs(r2["value"] - rw["meas"][1]["value"]) < 1e-3 and r2["valid"] == rw["meas"][1]["valid"], "replay does not reproduce the stored 0.5 ms value"
m = Pb.S[core.DT_OPT]["m"]; W, H, FPS, SLOW = 720, 480, 30, 0.5
m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, H)
rd = mujoco.Renderer(m, H, W); d = mujoco.MjData(m); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance = 3.2 if robot == "G1" else 4.0; cam.azimuth = 90; cam.elevation = -8
try: fb = ImageFont.truetype("DejaVuSans-Bold.ttf", 20); fs = ImageFont.truetype("DejaVuSans.ttf", 14)
except Exception: fb = fs = ImageFont.load_default()
title = sys.argv[4] if len(sys.argv) > 4 else f"{robot} sprint, body-derived construction"
sub = f"{min(m_['value'] for m_ in rw['meas']):.2f} m/s, valid at 1 and 0.5 ms · actuator lag {1000*SP.ACT_TAU:.0f} ms · " + (f"ratings {SP.RATINGS}" + (f", flight gate {1000*SP.FLIGHT_GATE:.0f} ms" if SP.FLIGHT_GATE > 0 else "") if SP.RATINGS != "peak" else f"{'/'.join(SP.SCALED_JOINTS)} rating x{scale:g}")
tmp = f"/tmp/render_sprint_{os.getpid()}"; os.makedirs(tmp, exist_ok=True); k = 0
for t in np.arange(0, Tm[-1], SLOW/FPS):
    i = min(int(np.searchsorted(Tm, t)), len(Q) - 1); d.qpos[:] = Q[i]; mujoco.mj_forward(m, d)
    cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.75*d.qpos[2]]; rd.update_scene(d, camera=cam); im = Image.fromarray(rd.render()); dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, W, 50], fill=(10, 16, 26)); dr.text((12, 5), title, fill=(240, 244, 250), font=fb); dr.text((12, 30), sub, fill=(170, 186, 205), font=fs)
    dr.rectangle([0, H - 26, W, H], fill=(10, 16, 26)); dr.text((12, H - 22), f"t = {t:.2f} s   x = {d.qpos[0]:.1f} m   ({SLOW:g}x speed)", fill=(255, 214, 140), font=fs)
    im.save(f"{tmp}/f{k:04d}.png"); k += 1
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{tmp}/f%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", out], check=True)
shutil.rmtree(tmp); print("wrote", out)
