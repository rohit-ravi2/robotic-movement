"""Render the wipe clip from a saved wipe solution (results/atlas/wipe_v0.json): replay the rollout with the trace on."""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import numpy as np
from engine import world as W, core; from tasks.wipe import WipeTask; import progression as PG
src = sys.argv[1] if len(sys.argv) > 1 else f"{core.RES}/atlas/wipe_v0.json"; out = sys.argv[2] if len(sys.argv) > 2 else f"{core.RES}/demo/wipe_v0.mp4"
sol = json.load(open(src)); d = sol["demand"]
objs = [W.obj("sponge", kind="box", size=(0.05, 0.04, 0.02), mass=0.1, pos=(0.053, -0.244, 0.621), rgba=(0.9, 0.8, 0.2, 1))]; surfs = [W.surface("counter", pos=(0.45, 0, 0.56), size=(0.35, 0.4, 0.02))]
scene = W.build_scene("G1", objs, surfs); wt = WipeTask("G1", scene=scene, objects=objs, surfaces=surfs); S = wt.plant(d)
rec = wt.rollout(S, d, wt.unpack(S, d, np.array(sol["z"])), record=True); e = rec["extra"]; m = S["m"]
print("replay:", rec["reasons"], "coverage %.2f gap %.3f force %.1f N cost %.0f J" % (e["coverage"], e["gap_mean"], e["force_mean"], rec["cost"]))
tr = e["trace"]; dt = m.opt.timestep*10; frames = [(i*dt, np.asarray(q)) for i, q in enumerate(tr)]
os.makedirs(os.path.dirname(out), exist_ok=True)
PG.render(m, frames, dict(z=0.75, dist=2.3, az=55, el=-22), "Wipe the counter", "G1 · sponge on a 0.58 m counter · squat on demand · path coverage %.0f%% · contact %.1f N (band 2-8 N open)" % (100*e["coverage"], e["force_mean"]), out)
print("wrote", out)
