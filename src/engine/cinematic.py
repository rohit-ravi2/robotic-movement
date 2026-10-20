"""Cinematic render kit (2026-10-18): one studio look for every replayed result.
1280x720 / 30 fps, neutral studio floor (the model's floor texture remapped to two greys, fog into a plain backdrop), the same
three-light rig on every robot (shadow-casting key spot that follows the subject, fill, soft headlight), tracking cameras,
variable-speed timelines with the factor on screen, and a lower-third label. Rendering is kinematic only (qpos traces from
replays; nothing here steps physics). Visual-only extras (a ball, its trail, a landing marker) are scene user geoms.
Encoding: raw frames piped to ffmpeg (libx264, yuv420p)."""
import os, subprocess
import numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 30
BACK = np.array([0.50, 0.51, 0.54])                       # backdrop / fog colour
FLOOR = (np.array([0.78, 0.78, 0.79]), np.array([0.70, 0.70, 0.71]))   # the two floor greys
INK, SUB, ACC, WARN = (245, 247, 250), (190, 198, 210), (255, 176, 64), (255, 120, 100)
BALL_RGBA = np.array([0.98, 0.42, 0.08, 1.0])


def font(sz, bold=False):
    try: return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", sz)
    except Exception: return ImageFont.load_default()


def studio_model(path, fog=(8.0, 40.0)):
    """Load a replay's model (.mjb) and restyle only its visual fields."""
    m = mujoco.MjModel.from_binary_path(path) if path.endswith(".mjb") else mujoco.MjModel.from_xml_path(path)
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    m.vis.quality.shadowsize = 4096; m.vis.quality.offsamples = 8
    m.vis.headlight.ambient[:] = 0.40; m.vis.headlight.diffuse[:] = 0.10; m.vis.headlight.specular[:] = 0.05
    m.vis.rgba.fog[:3] = BACK; m.vis.rgba.fog[3] = 1.0; m.vis.rgba.haze[:3] = BACK; m.vis.rgba.haze[3] = 1.0; ext = float(m.stat.extent)
    m.vis.map.fogstart = fog[0]/ext; m.vis.map.fogend = fog[1]/ext; m.vis.map.zfar = 400.0/ext; m.vis.map.znear = 0.01
    for t in range(m.ntex):
        a, w, h, nc = int(m.tex_adr[t]), int(m.tex_width[t]), int(m.tex_height[t]), int(m.tex_nchannel[t]); n = w*h*nc
        px = m.tex_data[a:a+n].reshape(h*w, nc).astype(float)
        if m.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_SKYBOX: px[:, :3] = 255*BACK
        else: continue
        m.tex_data[a:a+n] = np.clip(px, 0, 255).astype(np.uint8).ravel()
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE: continue
        mat = int(m.geom_matid[g]); tid = -1
        if mat >= 0: tid = max(int(x) for x in m.mat_texid[mat])
        if tid >= 0 and m.tex_type[tid] == mujoco.mjtTexture.mjTEXTURE_2D:
            a, w, h, nc = int(m.tex_adr[tid]), int(m.tex_width[tid]), int(m.tex_height[tid]), int(m.tex_nchannel[tid]); n = w*h*nc
            px = m.tex_data[a:a+n].reshape(h*w, nc).astype(float); L = px[:, :3].mean(1, keepdims=True); L = (L - L.min())/max(L.max() - L.min(), 1e-6)
            px[:, :3] = 255*(FLOOR[1] + (FLOOR[0] - FLOOR[1])*L); m.tex_data[a:a+n] = np.clip(px, 0, 255).astype(np.uint8).ravel()
            m.mat_rgba[mat] = [1, 1, 1, 1]; m.mat_reflectance[mat] = 0.0; m.mat_texrepeat[mat] = [1.0, 1.0]   # 1 m tiles (texuniform)
            m.mat_texuniform[mat] = 1
        else:
            if mat >= 0: m.mat_rgba[mat] = list(FLOOR[0]) + [1]
            m.geom_rgba[g] = list(FLOOR[0]) + [1]
    return m


class Studio:
    """Renders qpos frames of one model with the studio rig; extra visual geoms per frame."""
    def __init__(self, path, w=W, h=H, fog=(8.0, 40.0)):
        self.m = studio_model(path, fog); self.fog = fog; self.d = mujoco.MjData(self.m); self.w, self.h = w, h
        self.r = mujoco.Renderer(self.m, h, w); self.cam = mujoco.MjvCamera(); self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.opt = mujoco.MjvOption(); self.opt.sitegroup[:] = 0
    def set(self, q):
        self.d.qpos[:] = q; mujoco.mj_kinematics(self.m, self.d); mujoco.mj_comPos(self.m, self.d); mujoco.mj_camlight(self.m, self.d)
    def frame(self, lookat, distance, azimuth, elevation, geoms=(), key_target=None):
        c = self.cam; c.lookat[:] = lookat; c.distance = distance; c.azimuth = azimuth; c.elevation = elevation
        self.r._gl_context.make_current() if hasattr(self.r, "_gl_context") and self.r._gl_context else None
        self.r.update_scene(self.d, camera=c, scene_option=self.opt); s = self.r.scene
        s.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0; s.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1; s.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
        tgt = np.asarray(key_target if key_target is not None else lookat, float)
        s.nlight = 2; L0, L1 = s.lights[0], s.lights[1]          # identical rig everywhere (replaces the headlight and model lights): key spot following the subject + fill carrying the ambient
        kp = tgt + np.array([-2.0, -2.6, 5.5]); L0.pos[:] = kp; L0.dir[:] = (tgt - kp)/np.linalg.norm(tgt - kp); L0.directional = 0; L0.headlight = 0
        L0.castshadow = 1; L0.cutoff = 55.0; L0.exponent = 1.0; L0.attenuation[:] = [1, 0, 0]; L0.ambient[:] = 0.0; L0.diffuse[:] = 0.60; L0.specular[:] = 0.35; L0.bulbradius = 0.15
        L1.pos[:] = tgt + np.array([3, 4, 4]); L1.dir[:] = np.array([-0.45, -0.6, -0.66]); L1.directional = 1; L1.headlight = 0; L1.castshadow = 0
        L1.ambient[:] = 0.38; L1.diffuse[:] = 0.30; L1.specular[:] = 0.05; L1.attenuation[:] = [1, 0, 0]; L1.cutoff = 180; L1.exponent = 0
        for g in geoms: add_geom(s, *g)
        rgb = self.r.render().astype(np.float32); self.r.enable_depth_rendering(); dep = self.r.render(); self.r.disable_depth_rendering()
        f = np.clip((dep - self.fog[0])/(self.fog[1] - self.fog[0]), 0, 1); f = (f*f*(3 - 2*f))[..., None]    # depth fog into the backdrop (post-process:
        return (rgb*(1 - f) + 255*BACK*f).astype(np.uint8)                                                       # MuJoCo's own fog left a dark band on the plane)


def add_geom(scn, kind, pos, size, rgba, mat=None):
    if scn.ngeom >= scn.maxgeom: return
    g = scn.geoms[scn.ngeom]; t = {"sphere": mujoco.mjtGeom.mjGEOM_SPHERE, "cyl": mujoco.mjtGeom.mjGEOM_CYLINDER, "box": mujoco.mjtGeom.mjGEOM_BOX}[kind]
    mujoco.mjv_initGeom(g, t, np.asarray(size, float), np.asarray(pos, float), (np.eye(3) if mat is None else np.asarray(mat)).ravel(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def ballistic(p0, v0, z_floor, dt=1/600, g=9.81):
    """Drag-free flight from (p0, v0) until the centre reaches z_floor; returns times and positions."""
    p0, v0 = np.asarray(p0, float), np.asarray(v0, float); a = 0.5*g; b = -v0[2]; c = z_floor - p0[2]
    tl = (-b + np.sqrt(max(b*b - 4*a*c, 0.0)))/(2*a); t = np.append(np.arange(0, tl, dt), tl)
    P = p0 + np.outer(t, v0); P[:, 2] -= 0.5*g*t**2; return t, P


def speed_fn(windows, base=1.0, ramp=0.03):
    """windows: [(t0, t1, factor)] in sim time; returns s(t) (sim s per video s) and the nominal factor to display."""
    def s(t):
        lg, nom, wmax = np.log(base), base, 0.0
        for t0, t1, f in windows:
            w = 1.0 if t0 <= t <= t1 else max(0.0, 1 - min(abs(t - t0), abs(t - t1))/ramp)
            w = w*w*(3 - 2*w)
            if w > wmax: wmax = w; lg = np.log(base)*(1 - w) + np.log(f)*w; nom = f if w > 0.5 else base
        return float(np.exp(lg)), nom
    return s


def timeline(t_end, windows=(), base=1.0, t_start=0.0):
    """Video frame -> (sim time, displayed factor)."""
    s = speed_fn(windows, base); out = []; t = t_start
    while t <= t_end + 1e-9:
        f, nom = s(t); out.append((t, nom)); t += f/FPS
    return out


def track_q(Q, T, t):
    """Nearest recorded frame at time t (clamped to the record)."""
    i = int(np.clip(np.searchsorted(T, t - 1e-9), 0, len(Q) - 1)); return Q[i]


_VIG = {}
def vignette(w, h, k=0.28):
    if (w, h) not in _VIG:
        y, x = np.mgrid[0:h, 0:w]; r = np.hypot((x - w/2)/(w/2), (y - h/2)/(h/2))/np.sqrt(2); _VIG[(w, h)] = (1 - k*r**2.2)[..., None].astype(np.float32)
    return _VIG[(w, h)]


def finish(img):
    """Vignette (uint8 HxWx3 -> PIL RGBA)."""
    a = (img.astype(np.float32)*vignette(img.shape[1], img.shape[0])).clip(0, 255).astype(np.uint8); return Image.fromarray(a).convert("RGBA")


def lower_third(im, robot_task, value, validity, caveats, x=40, y=None, scale=1.0, width=None):
    """Clean lower-third: accent bar, robot / task, the measured value, the validity line, and caveat chips."""
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    f1, f2, f3, f4 = font(int(24*scale), True), font(int(34*scale), True), font(int(17*scale)), font(int(15*scale))
    chips = []; full = width or (im.size[0] - 2*x); cw = full - 40*scale; line = []; lw = 0
    for c in caveats:
        tw = d.textlength(c, font=f4) + 20*scale
        if line and lw + tw > cw: chips.append(line); line, lw = [], 0
        line.append((c, tw)); lw += tw + 8*scale
    if line: chips.append(line)
    hgt = int((30 + 42 + 26)*scale + len(chips)*28*scale + 18*scale); y = (im.size[1] - hgt - 26*scale) if y is None else y
    wbox = int(min(full, max(d.textlength(robot_task, font=f1), d.textlength(value, font=f2), d.textlength(validity, font=f3), max((sum(t for _, t in l) + 8*scale*len(l) for l in chips), default=0)) + 40*scale))
    d.rounded_rectangle([x, y, x + wbox, y + hgt], radius=int(10*scale), fill=(12, 16, 24, 205)); d.rectangle([x, y + int(10*scale), x + int(5*scale), y + hgt - int(10*scale)], fill=ACC + (255,))
    xx = x + int(20*scale); yy = y + int(10*scale)
    d.text((xx, yy), robot_task, font=f1, fill=INK + (255,)); yy += int(30*scale)
    d.text((xx, yy), value, font=f2, fill=ACC + (255,)); yy += int(42*scale)
    d.text((xx, yy), validity, font=f3, fill=SUB + (255,)); yy += int(28*scale)
    for l in chips:
        cx = xx
        for c, tw in l:
            d.rounded_rectangle([cx, yy, cx + tw, yy + int(22*scale)], radius=int(11*scale), outline=(255, 214, 140, 230), width=1, fill=(60, 44, 20, 160))
            d.text((cx + 10*scale, yy + 3*scale), c, font=f4, fill=(255, 226, 170, 255)); cx += tw + 8*scale
        yy += int(28*scale)
    return Image.alpha_composite(im, ov)


def badge(im, factor, t, x=None, y=20, extra=None, scale=1.0):
    """Top-right: time and the playback factor (slow motion / fast forward / real time)."""
    d = ImageDraw.Draw(im); f = font(int(18*scale), True); fs = font(int(15*scale))
    txt = "REAL TIME" if abs(factor - 1) < 1e-6 else (f"SLOW MOTION {factor:g}x" if factor < 1 else f"FAST FORWARD {factor:g}x")
    col = INK if abs(factor - 1) < 1e-6 else ((120, 200, 255) if factor < 1 else (255, 200, 120))
    tw = max(d.textlength(txt, font=f), d.textlength(f"t = {t:5.2f} s", font=fs)) + 28*scale; x = im.size[0] - tw - 20*scale if x is None else x
    d.rounded_rectangle([x, y, x + tw, y + 52*scale], radius=int(8*scale), fill=(12, 16, 24, 190))
    d.text((x + 14*scale, y + 6*scale), txt, font=f, fill=col); d.text((x + 14*scale, y + 30*scale), f"t = {t:5.2f} s" + (f"   {extra}" if extra else ""), font=fs, fill=SUB)
    return im


def title(im, text, sub=None, x=40, y=20, scale=1.0):
    d = ImageDraw.Draw(im); f = font(int(22*scale), True); fs = font(int(15*scale))
    tw = max(d.textlength(text, font=f), d.textlength(sub or "", font=fs)) + 28*scale; hh = (56 if sub else 36)*scale
    d.rounded_rectangle([x, y, x + tw, y + hh], radius=int(8*scale), fill=(12, 16, 24, 190)); d.text((x + 14*scale, y + 6*scale), text, font=f, fill=INK)
    if sub: d.text((x + 14*scale, y + 33*scale), sub, font=fs, fill=SUB)
    return im


def note(im, text, xy, color=WARN, size=18, bg=(12, 16, 24, 200)):
    d = ImageDraw.Draw(im); f = font(size, True); tw = d.textlength(text, font=f)
    d.rounded_rectangle([xy[0], xy[1], xy[0] + tw + 24, xy[1] + size + 14], radius=8, fill=bg); d.text((xy[0] + 12, xy[1] + 6), text, font=f, fill=color); return im


class Writer:
    def __init__(self, out, w=W, h=H):
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True); self.out = out; self.n = 0
        self.p = subprocess.Popen(["nice", "-n", "10", "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
                                   "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    def add(self, im, hold=1):
        b = np.asarray(im.convert("RGB"), np.uint8).tobytes()
        for _ in range(hold): self.p.stdin.write(b); self.n += 1
    def close(self):
        self.p.stdin.close(); self.p.wait(); return self.out, self.n/FPS
