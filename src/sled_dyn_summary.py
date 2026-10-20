"""Track B summary: predicted (engine.leverage_dyn, spec only) vs found (sled_dyn search) max sled mass and lean -> results/overnight/B/summary.json."""
import os, json, glob
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "overnight", "B"))


def main():
    P = json.load(open(f"{OUT}/predictor.json")); rows = {}
    for f in sorted(g_ for g_ in glob.glob(f"{OUT}/*kg.json") if not os.path.basename(g_).startswith("seeded_")):
        J = json.load(open(f)); b = J["search"]["best"]; nm = J["direction"]
        rows.setdefault(nm, []).append(dict(payload=J["mass"], total=J["sled_total_mass"], disp_1ms=b["disp"], disp_05ms=b["dt0005"]["disp"], disp_min=b.get("disp_min"),
                                            ok_both=b.get("ok_both"), reason_1ms=b["reason"], reason_05ms=b["dt0005"]["reason"], moved_0p2_both=J["moved_0p2_both"],
                                            robust_moved_0p2=J["robust_moved_0p2"], robust=[(r["seed"], round(r["disp_min"], 3), r["reason"], r["reason05"]) for r in J["robust"]],
                                            lean_found=b["com_lean_mean"], lean_minmax=b.get("com_lean_minmax"), weld_err_max_m=[b["weld_err_max"], b["dt0005"].get("weld_err_max")],
                                            weld_ang_max_rad=[b["weld_ang_max"], b["dt0005"].get("weld_ang_max")], util_rms=b["util_rms"], evals=J["search"]["evals"],
                                            s_per_eval_cpu=J["search"]["s_per_eval_cpu"], wall_s=J["wall_s"], warm_start=J["warm_start"], decoded=J["decoded"]))
    out = dict(meta=dict(units="sled masses: payload label (search) and TOTAL (payload + 1 kg post, what the predictor reports); disp in m; lean = robot COM x minus feet-centre x (m), + = towards +x (the sled side)",
                         success="moved >= 0.2 m at BOTH 1 ms and 0.5 ms, no fall, feet slid <= 3 cm, T = 4 s", attachment="right hand WELDED to the sled post (simulator abstraction); free sled box mu 0.4"))
    for nm in ("push", "pull"):
        R = sorted(rows.get(nm, []), key=lambda r: r["payload"]); pw = P[f"{nm}_weld"]; pp = P[f"{nm}_point"]
        found_max = max([r["total"] for r in R if r["moved_0p2_both"]], default=None)
        found_any = max([r["total"] for r in R if r["ok_both"] and (r["disp_min"] or 0) > 0.05], default=None)
        out[nm] = dict(predicted_total_max_weld=pw.get("m_max_total"), predicted_quasistatic_weld=pw.get("m_quasistatic_total"), predicted_lunge_bound_weld=pw.get("m_max_total_lunge_bound"),
                       predicted_total_max_point=pp.get("m_max_total"), predicted_lean_mean=pw.get("lean_pred_mean"), predicted_lean_start=pw.get("lean_pred_start"), predicted_lean_end=pw.get("lean_pred_end"),
                       predicted_binding_start=pw["best_static_start"]["binding_joints"], predicted_binding_point_start=pp["best_static_start"]["binding_joints"],
                       predicted_heavier_fails=pw.get("why_heavier_fails"), static_brace_total=P["static_brace_overnight"]["m_total_equiv"],
                       found_total_max_moved_0p2_both=found_max, found_total_max_moved_gt_5cm_both=found_any, configs=R)
    out["seeded"] = []
    for f in sorted(glob.glob(f"{OUT}/seeded_*kg.json")):
        J = json.load(open(f)); b = J["search"]["best"]; se = J["seed_eval"]
        out["seeded"].append(dict(direction=J["direction"], payload=J["mass"], seed_disp_min=se["disp_min"], seed_reason=[se["reason"], se["dt0005"]["reason"]],
                                  best_disp_min=b.get("disp_min"), best_ok_both=b.get("ok_both"), best_reason=[b["reason"], b["dt0005"]["reason"]], evals=J["search"]["evals"], robust_moved_0p2=J["robust_moved_0p2"]))
    if os.path.exists(f"{OUT}/mechanism.json"):
        out["mechanism_fresh_1ms_replay"] = [{k: v for k, v in r.items() if k != "samples"} for r in json.load(open(f"{OUT}/mechanism.json"))]
    out["reproducibility"] = ("stored numbers reproduce exactly with the search workers' plant history (built at 1 ms, re-timed to 0.5 ms, back to 1 ms; "
                              "render_best does this). A plant built fresh at 1 ms settles ~1e-5 m differently and gives push 20 kg 0.129 m FEET_SLID, push 30 kg 0.090, "
                              "pull 30 kg 0.254, pull 40 kg 0.138 (mechanism.json): the found strategies sit near the slip limit and are sensitive; see robust_moved_0p2.")
    json.dump(out, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
    for nm in ("push", "pull"):
        o = out[nm]; print(nm, "pred weld", o["predicted_total_max_weld"], "point", o["predicted_total_max_point"], "lunge", o["predicted_lunge_bound_weld"], "lean", round(o["predicted_lean_mean"], 3),
                           "| found >=0.2 m both:", o["found_total_max_moved_0p2_both"], ">5 cm:", o["found_total_max_moved_gt_5cm_both"])
        for r in o["configs"]: print("   ", r["payload"], "disp", round(r["disp_1ms"], 3), round(r["disp_05ms"], 3), r["reason_1ms"], r["reason_05ms"], "lean", r["lean_found"] and round(r["lean_found"], 3), "robust", r["robust_moved_0p2"], "weld", [round(x or 0, 3) for x in r["weld_err_max_m"]], [round(x or 0, 2) for x in r["weld_ang_max_rad"]])


if __name__ == "__main__":
    main()

