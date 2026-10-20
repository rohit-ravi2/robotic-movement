"""Render landed jumps (Phase 14, 1 ms): deep-crouch (analytic-prior) and native, with the balance/landing controller
taking over at takeoff. Logs qpos through the push-off tracker (every step) and the landing controller (500 Hz)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, branch_i as BI, mocap_baseline as MB, aprior_standard as AS, balance as B, jump13 as J13
from PIL import Image, ImageDraw, ImageFont
OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/viz"); FPS = 30
PAR = json.load(open(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase14/balance_tuning.json")))["best"]

def record(r, robot="G1"):
    if robot == "G1": c = AS.clip0(); S = BI.setup(c["start"]); par = PAR
    else:
        import aprior_h1 as AH; c = AH.clip0(); S = BI.setup(start=AH.Q0, robot="H1")
        par = json.load(open(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase14/balance_tuning_h1.json")))["best"]
    BI.apply_dt(S, 0.001); ref = MB.analytic_clip(c, robot=robot) if r["method"] == "APRIOR" else c
    log = []
    tr = MB.Tracker(S, ref, omega=r["omega"], tscale=r["tscale"]) if r["omega"] else None
    def fn(t, d, S_):
        log.append((t, d.qpos.copy())); return tr(t, d, S_) if tr else np.zeros(len(S["jarr"]["aid"]))
    ctl = B.BalanceController(S, np.array(S["q_settled"]), params=par); orig = ctl.targets
    def tg(t, d, c_):
        log.append((None, d.qpos.copy())); return orig(t, d, c_)
    ctl.targets = tg
    o = BI.rollout(np.array(r["x"]), set(S["types"]), None, S=S, ctrl_fn=fn, horizon=J13.H_LAND, score="best", knot_dt=J13.KNOT_DT, landing=dict(controller=ctl))
    # rebuild a uniform time axis: push-off entries carry t; landing entries are every 2 ms after the last push-off t
    t_last = max(t for t, _ in log if t is not None); Q, T = [], []; k = 0
    for t, q in log:
        if t is None: k += 1; t = t_last + 0.002*k
        T.append(t); Q.append(q)
    BI.apply_dt(S, 0.004)
    return np.array(T), np.array(Q), o, S["m"]

def frames(m, T, Q, label, sub, w=480, h=600):
    m.vis.global_.offheight = max(m.vis.global_.offheight, h); m.vis.global_.offwidth = max(m.vis.global_.offwidth, w)
    r = mujoco.Renderer(m, h, w); d = mujoco.MjData(m); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [Q[0][0], Q[0][1], 1.0]; cam.distance = 4.2; cam.azimuth = 90; cam.elevation = -5
    try: f = ImageFont.truetype("DejaVuSans-Bold.ttf", 20); fs = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception: f = fs = ImageFont.load_default()
    out = []
    for t in np.arange(0, min(T[-1], 2.6), 1.0/FPS):
        i = int(np.searchsorted(T, t)); i = min(i, len(Q)-1); d.qpos[:] = Q[i]; mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam); im = Image.fromarray(r.render()); dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, w, 52], fill=(0, 0, 0)); dr.text((10, 5), label, fill=(255, 255, 255), font=f); dr.text((10, 30), sub, fill=(200, 200, 200), font=fs)
        out.append(np.array(im))
    return out

if __name__ == "__main__":
    R = json.load(open(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase14/jump14.json")))
    apr = max([r for r in R if r["method"] == "APRIOR" and r["meas"][0]["ok"]], key=lambda r: r["meas"][0]["v"])
    nat = max([r for r in R if r["method"] == "NATIVE" and r["ok_all"]], key=lambda r: r["v"])
    P = []
    for r, lab in ((apr, "DEEP-CROUCH jump + landing"), (nat, "NATIVE jump + landing")):
        T, Q, o, m = record(r); L = o["landing"]
        P.append(frames(m, T, Q, lab, f"takeoff {o['v']:.2f} m/s, landed {'yes' if L['ok'] else 'no'} (1 ms)")); print(lab, round(o["v"], 2), L["ok"])
    n = min(len(p) for p in P); os.makedirs(f"{OUT}/land_frames", exist_ok=True)
    for i in range(n):
        Image.fromarray(np.concatenate([p[i] for p in P], axis=1)).save(f"{OUT}/land_frames/f{i:04d}.png")
    print("frames", n)
