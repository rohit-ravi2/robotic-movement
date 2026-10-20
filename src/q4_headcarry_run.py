"""Q4 held-out head-carry runs: the T2 driver (t2_headcarry_run.run_mass: CMA on the track A 14-D vector, slip-gated validity at
1 ms AND 0.5 ms on fresh plants, T1 support gate inside engine.headcarry.gate, law-off ablation) UNCHANGED, with the Q4 object table
registered and output to results/overnight/Q4. Pre-registered in results/overnight/Q4/PREDICTIONS.md before this runs.
usage: python q4_headcarry_run.py OBJ m1,m2,... [budget] [workers]"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, multiprocessing as mp
import t2_headcarry_run as T
from engine import dynpred_headcarry_v2 as V
T.OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "overnight", "Q4")
_t2_register = T.D.register_objects


def _register():
    """T2 objects + the Q4 objects (engine.dynpred_headcarry_v2.OBJ_Q4). No robot change."""
    _t2_register()
    for k, v in V.OBJ_Q4.items():
        T.HC.OBJ.setdefault(k, v)
        import leverage_probes as LP; LP.OBJ.setdefault(k, v)


T.D.register_objects = _register      # used by the workers (ev, gated): fork inherits it
if __name__ == "__main__":
    obj = sys.argv[1]; masses = [float(x) for x in sys.argv[2].split(",")]; budget = int(sys.argv[3]) if len(sys.argv) > 3 else 180
    W = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    _register(); x0 = np.zeros(T.HC.DIM["head"])
    with mp.get_context("fork").Pool(W) as pool:
        for mass in masses:
            r = T.run_mass(pool, obj, mass, budget, x0)
            p = f"{T.OUT}/head_{obj}_{mass:g}kg.json"; J = json.load(open(p))
            if J.get("track") != "Q4": J["track"] = "Q4"; J["driver"] = "t2_headcarry_run.run_mass (unchanged)"; json.dump(J, open(p, "w"), indent=1)
            if r["valid"]: x0 = np.asarray(r["best_z"])
