"""Track A summary (2026-10-17): predicted (spec only) vs found (search, gated at both dt) maximum load per strategy and object,
the law-off counterfactual, and the arm technique of each found hold. Writes results/overnight/A/summary.json."""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import headcarry as HC

R = HC.RES


def case(f):
    J = json.load(open(f)); b = J["best"]; c = J["check_dt0005"]; ab = J["law_off_ablation"]
    top = sorted(b.get("arm_util_mean", {}).items(), key=lambda kv: -kv[1])[:4]
    return dict(tag=os.path.basename(f)[:-5], strategy=J["strategy"], obj=J["obj"], mass=J["mass"], disturbance=J.get("sway"), valid=J["valid"],
                prior_valid_1ms=J["prior_gate"], solved=J.get("solved"), evals=J["evals"], wall_min=round(J["wall_s"]/60, 1),
                held_1ms=b["t_held"], reason_1ms=b["reason"], held_05ms=c["t_held"], reason_05ms=c["reason"],
                law_off_held=ab["t_held"], law_off_reason=ab["reason"], law_off_tilt_max=ab.get("tilt_max_deg"),
                head_frac=b.get("head_frac"), palm_N=b.get("palm_N_mean"), tilt_max_deg=b.get("tilt_max_deg"), e_max_m=b.get("e_max"),
                com_y_range=b.get("com_y_range"), util_rms=b.get("util_rms"), util_p95=b.get("util_p95"), top_arm_joints=top,
                gains=b.get("gains"), params={k: v for k, v in J["best_params"].items()})


if __name__ == "__main__":
    P = json.load(open(f"{R}/predictor.json")); cases = [case(f) for f in sorted(glob.glob(f"{R}/*.json")) if not any(k in f for k in ("_pass1", "predictor", "summary"))]
    comp = {}
    for st in ("head", "hands"):
        for ob in ("box", "pot"):
            cs = [c for c in cases if c["strategy"] == st and c["obj"] == ob and not c["disturbance"]]
            ok = [c["mass"] for c in cs if c["valid"]]; bad = [c["mass"] for c in cs if not c["valid"]]
            pr = P[f"{st}_{ob}"]
            comp[f"{st}_{ob}"] = dict(predicted_m_max=pr["m_max"], predicted_binding=pr["binding"], predicted_requirements={k: v["m_max"] for k, v in pr["requirements"].items()},
                                      found_max_valid=(max(ok) if ok else 0.0), tested_valid=sorted(ok), tested_failed=sorted(bad),
                                      first_failure_above_max=(min([m for m in bad if m > (max(ok) if ok else 0)], default=None)),
                                      disturbed={d: sorted(c["mass"] for c in cases if c["strategy"] == st and c["obj"] == ob and c["disturbance"] and c["disturbance"][0] == d and c["valid"]) for d in ("sway", "push")})
    out = dict(track="A", note="G1H, simulator only, free objects (no weld). valid = held 10 s at 1 ms AND 0.5 ms, tilt <= 45 deg, never on the floor; head: COM within the 7 cm head patch and the head carries >= 50% of the weight",
               comparison=comp, cases=cases)
    json.dump(out, open(f"{R}/summary.json", "w"), indent=1, default=str)
    for k, v in comp.items(): print(k, v)
