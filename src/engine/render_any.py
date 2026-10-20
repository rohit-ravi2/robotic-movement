"""Render ANY qpos trace (2026-10-16, leverage probes): mp4 with a tracking (pelvis) or fixed camera + a 6-frame contact sheet PNG.
usage (python): render(scene, Q, dt, out_mp4, title=..., sub=...). scene = an MJCF path, or a robot key ("G1", "G1H", "H1"); the model
is loaded only for kinematics, so its nq must match the trace. Conventions follow engine/render_sprint.py (720x480, 30 fps, title bar).
CLI: python engine/render_any.py <scene|robot> <trace.npy> <dt> <out.mp4> [title]"""
import os, sys, shutil, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont


def _model(scene):
    from engine import world as W
    return mujoco.MjModel.from_xml_path(W.BASE.get(scene, scene))


def render(scene, Q, dt, out, title="", sub="", camera="track", slow=1.0, W_=720, H_=480, fps=30, azimuth=135, distance=3.0, hands_mass=0, sheet=True, elevation=-12):
    m = _model(scene); Q = np.asarray(Q)
    if Q.shape[1] != m.nq: raise ValueError(f"trace nq {Q.shape[1]} != model nq {m.nq} ({scene})")
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W_); m.vis.global_.offheight = max(m.vis.global_.offheight, H_)
    rd = mujoco.Renderer(m, H_, W_); d = mujoco.MjData(m); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = distance; cam.azimuth = azimuth; cam.elevation = elevation; cam.lookat[:] = [Q[0, 0], Q[0, 1], 0.8]
    try: fb = ImageFont.truetype("DejaVuSans-Bold.ttf", 20); fs = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception: fb = fs = ImageFont.load_default()
    T = (len(Q) - 1)*dt; tmp = tempfile.mkdtemp(prefix="render_any_", dir=os.environ.get("TMPDIR", "/tmp")); frames = []
    for k, t in enumerate(np.arange(0, T + 1e-9, slow/fps)):
        i = min(int(round(t/dt)), len(Q) - 1); d.qpos[:] = Q[i]; mujoco.mj_forward(m, d)
        if camera == "track": cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.8]
        rd.update_scene(d, camera=cam); im = Image.fromarray(rd.render()); dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, W_, 50], fill=(10, 16, 26)); dr.text((12, 5), title, fill=(240, 244, 250), font=fb); dr.text((12, 30), sub, fill=(170, 186, 205), font=fs)
        dr.rectangle([0, H_ - 26, W_, H_], fill=(10, 16, 26)); dr.text((12, H_ - 22), f"t = {t:.2f} s" + (f"   ({slow:g}x speed)" if slow != 1 else ""), fill=(255, 214, 140), font=fs)
        im.save(f"{tmp}/f{k:04d}.png"); frames.append(f"{tmp}/f{k:04d}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{tmp}/f%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", out], check=True)
    if sheet and frames:
        pick = [frames[int(round(j))] for j in np.linspace(0, len(frames) - 1, 6)]; ims = [Image.open(p).resize((W_//2, H_//2)) for p in pick]
        cs = Image.new("RGB", (3*W_//2, 2*H_//2));
        for j, im in enumerate(ims): cs.paste(im, ((j % 3)*W_//2, (j//3)*H_//2))
        cs.save(os.path.splitext(out)[0] + "_sheet.png")
    shutil.rmtree(tmp); return out


if __name__ == "__main__":
    a = sys.argv; render(a[1], np.load(a[2]), float(a[3]), a[4], title=(a[5] if len(a) > 5 else ""))
