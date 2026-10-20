"""T2: dynamic stride predictor (engine.dynpred_sprint) -> results/overnight/T2/pred_sprint_<robot>_ank<scale>_lag<ms>.json.
usage: python t2_predict_sprint.py ROBOT [ankle_scale] [lag_s]   (ankle_scale != 1: WHAT-IF BODY, dose-response only)"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import dynpred_sprint as D
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "overnight", "T2")
if __name__ == "__main__":
    r = sys.argv[1]; sc = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0; lag = float(sys.argv[3]) if len(sys.argv) > 3 else 0.002
    t = time.time(); B = D.spec(r); out = D.v_max(B, lag=lag, ankle_scale=sc)
    res = dict(robot=r, ankle_scale=sc, what_if_body=(sc != 1.0), lag=lag, spec={k: v for k, v in B.items() if not isinstance(v, dict)},
               ratings={k: B[k] for k in ("hip", "knee", "ankle", "shoulder")}, **out, time_s=time.time() - t)
    json.dump(res, open(f"{OUT}/pred_sprint_{r}_ank{sc:g}_lag{lag*1000:g}ms.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(r, sc, lag, "v_max", out["v_max"], "binding above", out.get("binding"), out.get("fails_above"), "cadence", out.get("at", {}).get("step_hz"), round(res["time_s"]), flush=True)
