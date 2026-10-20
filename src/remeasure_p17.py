"""P17 impact: re-measure every gate_a10_main solution under the sustained-flight takeoff rule."""
import sys, os, json, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gate_a10 import SPECS
R = json.load(open(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase10/gate_a10_main.json")))
for sp in SPECS:
    os.environ["MLAW_HUMAN_SPEC"] = SPECS[sp]; import gate_a2 as GA; importlib.reload(GA); h = GA.make_h()
    for r in [x for x in R if x["extra"]["spec"] == sp]:
        o = GA.rollout(h, np.array(r["x"]))
        r["height_p17"] = o["height"]; r["valid_p17"] = bool(o["valid"] and (not r["squat"] or o["dip"] <= 0.0105))
json.dump(R, open(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase10/gate_a10_main.json"), "w"), default=float)
for sp in SPECS:
    g = [x for x in R if x["extra"]["spec"] == sp]
    dh = np.array([(x["height_p17"] - x["height"])*100 for x in g])
    print(sp, "n", len(g), "height change under P17 rule: median %+.2f cm, min %+.2f, max %+.2f; changed >0.1 cm: %d" % (np.median(dh), dh.min(), dh.max(), int((np.abs(dh) > 0.1).sum())))
