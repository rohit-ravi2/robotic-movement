"""Q4: sled predictor v2 (engine.dynpred_sled_v2) for one direction at every scored / new sled mu -> results/overnight/Q4/pred_sled_v2_<dir>.json.
usage: python q4_predict_sled.py push|pull [mu1,mu2,...]"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leverage_probes as LP
from engine import dynpred_sled_v2 as V
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "overnight", "Q4")
if __name__ == "__main__":
    nm = sys.argv[1]; sgn = 1 if nm == "push" else -1
    mus = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0.25,0.3,0.4,0.5,0.6").split(",")]
    S = LP.l3_plant(20.0, 0.001); t = time.time(); fam = V.build_family(S, sgn); res = dict(direction=nm, family_s=[s for s, _ in fam], family_time_s=time.time() - t)
    print(nm, "family", [(s, len(f)) for s, f in fam], round(time.time() - t), flush=True)
    for mu in mus:
        t = time.time(); r = V.predict_v2(S, sgn, mu, fam); r["time_s"] = time.time() - t; res[f"mu{mu:g}"] = r
        print(nm, mu, "v2 m_max_total", r.get("m_max_total"), "ramp1.5", r.get("m_max_total_ramp1.5"), "breakaway", round(r.get("m_breakaway_total", 0), 2),
              "why heavier", r.get("t_ramp_0", {}).get("why_heavier_fails"), "at max", {k: v for k, v in (r.get("t_ramp_0", {}).get("at_max") or {}).items() if k in ("t", "s_stop", "b_stop", "M_eff")}, round(r["time_s"]), flush=True)
        json.dump(res, open(f"{OUT}/pred_sled_v2_{nm}.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
