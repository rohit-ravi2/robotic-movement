"""Phase 10: Bobbert & van Soest strength-without-retuning direction test on the P15/P16-corrected model,
both specs. Replays the converged 19.2k squat-jump solutions (gate_a10_main) with every actuator x1.5 and the
SAME control. Direction only (published magnitude not verified)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, importlib
from gate_a10 import SPECS
R = json.load(open(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase10/gate_a10_p17.json"))); out = []
for sp in SPECS:
    os.environ["MLAW_HUMAN_SPEC"] = SPECS[sp]
    import gate_a2 as GA; importlib.reload(GA)
    g = [r for r in R if r["extra"]["spec"] == sp and r["extra"].get("tag") == "ladder" and r["budget"] == 19200 and r["valid"]]
    hb, hs = GA.make_h(), GA.make_h([1.5]*5)
    for r in g:
        b = GA.rollout(hb, np.array(r["x"])); s = GA.rollout(hs, np.array(r["x"]))
        out.append(dict(spec=sp, seed=r["seed"], base=b["height"], strong=s["height"], strong_valid=s["valid"]))
    d = [o for o in out if o["spec"] == sp]
    print(f"{sp}: height falls in {sum(o['strong'] < o['base'] for o in d)}/{len(d)}; median {np.median([o['base'] for o in d])*100:.1f} -> {np.median([o['strong'] for o in d])*100:.1f} cm")
json.dump(out, open(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase10/bobbert10_p17.json"), "w"))
