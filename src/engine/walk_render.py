"""Render a stored M5 walking gait (render_any conventions: 720x480, 30 fps, tracking camera, title bar, 6-frame contact sheet).
usage: python engine/walk_render.py <gait json> <out.mp4> [dt] [perturbation name, e.g. shove_back]"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from engine import walk as W, render_any as RA, core

if __name__ == "__main__":
    g = json.load(open(sys.argv[1])); out = sys.argv[2]; dt = float(sys.argv[3]) if len(sys.argv) > 3 else core.DT_OPT
    P = W.Walk(g["robot"], g["v"], g.get("turn", 0.0), stride=g["stride"]); z = np.asarray(g["z"], float)
    if len(z) < P.n(): z = np.concatenate([z, np.zeros(P.n() - len(z))])
    pn = sys.argv[4] if len(sys.argv) > 4 else None; kw = dict(W.PERTURB[pn]) if pn else {}
    if kw.get("noise") == "start": rng = np.random.default_rng(11); kw["noise"] = dict(q=rng.normal(0, kw.pop("q_sd", 0.02), len(P.S[dt]["types"])), vx=0.05)
    r = P.run(z, dt, record=True, **kw); Q = np.asarray(r["trace"]["qpos"])
    S = P.S[dt]; m = S["m"]
    if Q.shape[1] != m.nq: raise SystemExit("nq mismatch")
    import mujoco
    # the plant model (scene + stock contact) is the one rendered; render_any loads a scene by key, so pass the plant's own model file
    scene = S.get("mjcf") or g["robot"]
    title = f"{g['robot']} walk {g['v']:.2f} m/s" + (f", turn {np.degrees(g.get('turn', 0.0)):.0f} deg" if g.get("turn") else "") + f"  (measured {r['value']:.3f} m/s, {r['cadence_spm']:.0f} steps/min)"
    if pn and "shove" in pn: title += f"  [{pn}: 0.1 BW for 0.2 s at t = {kw['shove'][2]*P.R['Tc']*P.unpack(z)['tscale']:.2f} s]"
    sub = f"stock body, published ratings, {W.LAG*1000:.0f} ms actuator lag, stock contact; dt {dt*1000:g} ms; valid={r['valid']} {r['reasons']}"
    RA.render(scene, Q, 4*dt, out, title=title, sub=sub, azimuth=120, distance=3.2)
    json.dump(dict(source=sys.argv[1], dt=dt, summary=W.summarize(r)), open(os.path.splitext(out)[0] + ".json", "w"), indent=1, default=float)
    print(out, W.summarize(r))
