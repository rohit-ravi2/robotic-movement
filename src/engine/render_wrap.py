"""Render the J27 wrap pick -> wipe clip from the composed trace (results/atlas/g1h_wrapwipe_trace.npy, qpos every 10 ms).
usage: render_wrap.py [out.mp4] [snap_dir]"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import numpy as np, mujoco
from engine import core; from tasks.wipe import WipeTask; import engine.g1h_wrap_wipe as P; import progression as PG
out = sys.argv[1] if len(sys.argv) > 1 else f"{core.RES}/demo/g1h_wrap_wipe.mp4"; snap = sys.argv[2] if len(sys.argv) > 2 else None
sc, objs, surfs, br = P.scene(); wt = WipeTask('G1H', scene=sc, objects=objs, surfaces=surfs); S = wt.plant(P.wipe_demand(wt)); m = S["m"]
Q = np.load(f"{core.RES}/atlas/g1h_wrapwipe_trace.npy"); frames = [(i*0.001, q) for i, q in enumerate(Q)]; print("frames", len(Q), "duration %.1f s" % (0.001*len(Q)))
os.makedirs(os.path.dirname(out), exist_ok=True)
PG.render(m, frames, dict(z=0.85, dist=2.1, az=60, el=-18), "Pick up the brush, wipe the counter",
          "G1H (Dex3 hands) · angled-handle brush, 0.76 m counter · side-approach fist wrap · force-controlled wipe 4.7 N mean · head held level", out)
print("wrote", out)
if snap:
    from engine.grasp_lab import snapshot; os.makedirs(snap, exist_ok=True); d = mujoco.MjData(m); sid = S["site"]
    for t_ in (0.0, 2.5, 4.5, 5.6, 7.0, 10.0, 11.5, 13.0, 14.5):
        i = min(int(t_/0.001), len(Q) - 1); d.qpos[:] = Q[i]; mujoco.mj_forward(m, d); snapshot(m, d, sid, f"{snap}/wrap_t{t_:.1f}.png", az=120, el=-20, dist=1.1)
    print("snapshots in", snap)
