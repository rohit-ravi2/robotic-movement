"""M5 side study (labelled): search at a commanded speed with a STRIDE OVERRIDE (step length s, double-support fraction delta, hip ratio hr)
instead of the spec-preferred stride, to separate 'this speed is not walkable by the construction' from 'the predicted stride is not'.
usage: python engine/walk_stride_probe.py ROBOT v s delta hr iters pop workers seed [warm_json]"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from engine import walk as W

if __name__ == "__main__":
    robot, v, s, dl, hr = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    it, pop, wk, seed = int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9])
    z0 = json.load(open(sys.argv[10]))["z"] if len(sys.argv) > 10 else None
    B = W.walk_body(robot); st = W.gait_check(B, v, s, dl, hr, strict=False)
    if st is None: raise SystemExit("stride geometrically infeasible")
    st = W.reach_margin(B, st); st["override"] = dict(s=s, delta=dl, hr=hr, note="side study: stride override, not the spec-preferred stride")
    r = W.search(robot, v, it, pop, wk, seed, z0=z0, stride=st, tag=f"_stride_s{s}_d{dl}")
    print(robot, v, "override", s, dl, hr, "->", r["score"], r["r1"]["reasons"], r["r1"]["value"], r["r2"]["reasons"], flush=True)
