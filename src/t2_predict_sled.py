"""T2: dynamic sled predictor (engine.dynpred_sled), one (direction, sled mu) per call -> results/overnight/T2/pred_sled_<dir>_mu<mu>.json.
usage: python t2_predict_sled.py push|pull mu"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leverage_probes as LP
from engine import dynpred_sled as DS
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "overnight", "T2")
if __name__ == "__main__":
    nm, mu = sys.argv[1], float(sys.argv[2]); sgn = 1 if nm == "push" else -1
    S = LP.l3_plant(20.0, 0.001); t = time.time()
    r = DS.predict(S, sgn, mu_sled=mu, interface="weld"); r["time_s"] = time.time() - t; r["direction"] = nm
    json.dump(r, open(f"{OUT}/pred_sled_{nm}_mu{mu:g}.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(nm, mu, r.get("status"), "m_max_total", r.get("m_max_total"), "why heavier", r.get("why_heavier_fails"), "lean", r.get("lean_pred_mean"), round(r["time_s"]), flush=True)
