"""Collect the held-out embodiment results into results/heldout/summary.json (2026-10-16)."""
import os, sys, json, glob
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
def J(p): return json.load(open(os.path.join(R, p)))
def first_valid(h, pop): return next((1 + g*pop for g, s in h if s < 0), None)      # generation-resolution (ladder hist keeps the best per generation)
def ladder(p):
    if not os.path.exists(os.path.join(R, p)): return None
    L = J(p); out = []
    for r in L["rungs"]:
        if "meas" not in r: out.append(dict(v=r["v"], ok=False, note=r.get("no_scaffold"))); continue
        u = r["utilisation"]; out.append(dict(v=r["v"], ok=r["ok_both_dt"], v_meas=[m["value"] for m in r["meas"]], reasons=[m["reasons"] for m in r["meas"]], evals=r["evals"],
            evals_to_first_valid_gen_res=first_valid(r["hist"], L["budget"]["pop"]), stride=dict((k, r["stride"][k]) for k in ("tc", "tf", "h0", "step_hz")), params=r["params"],
            binding_torque_p95=sorted(((k, v["torque_p95"]) for k, v in u.items()), key=lambda kv: -kv[1])[:4], max_speed_ratio=sorted(((k, v["speed"]) for k, v in u.items()), key=lambda kv: -kv[1])[:3],
            technique=r["technique"]))
    ok = [x for x in out if x["ok"]]
    return dict(scale=L.get("scale"), act_tau=L["supplied"]["act_tau"], rungs=out, top_valid=(max(ok, key=lambda x: x["v"]) if ok else None))
out = dict(ceiling={k: {c: v["caps"][c]["v_max"] for c in v["caps"]} for k, v in J("heldout/ceiling.json").items()}, documented=J("heldout/documented.json"),
           onboarding={r: dict(accept=J(f"heldout/accept_{r}.json")["accepted"], stand=J(f"heldout/accept_{r}.json")["stand"]) for r in ("Apollo", "T1")})
out["ladders"] = {k: ladder(p) for k, p in dict(Apollo_lag2ms="sprint/ladder_Apollo_heldout_lag2ms.json", Apollo_lag10ms="sprint/ladder_Apollo_heldout_lag10ms.json",
    Apollo_lag2ms_extrap="sprint/ladder_Apollo_heldout_lag2ms_extrap.json", T1_lag2ms="sprint/ladder_T1_heldout_lag2ms.json",
    T1_cf_x1p5_lag2ms="sprint/ladder_T1_heldout_cf_x1.5_lag2ms.json", T1_cf_x1p5_lag10ms="sprint/ladder_T1_heldout_cf_x1.5_lag10ms.json",
    T1_cf_x1p5ank_lag2ms="sprint/ladder_T1_heldout_cf_x1.5ank_lag2ms.json", T1_cf_x1p5ank_lag10ms="sprint/ladder_T1_heldout_cf_x1.5ank_lag10ms.json").items()}
ab = {}
for f in sorted(glob.glob(os.path.join(R, "heldout", "ablation_*.json"))):
    d = json.load(open(f)); ab[os.path.basename(f)[9:-5]] = dict(arm=d["arm"], seed=d["seed"], scale=d["scale"], evals=d["evals"], evals_to_first_valid=d["evals_to_first_valid"], best_speed=d["best_speed"],
                                                               t_run=[m["t_run"] for m in d["meas"]], reasons=[m["reasons"] for m in d["meas"]])
out["ablation"] = ab
out["movement_code_changes"] = "NONE. Registration only (names, ratings, feet, keyframe, contact pairs): src/heldout_onboard.py, additive entries in branch_i.ROBOTS, throw.CFG, sprint_ceiling.CHAINS, sprint.LEG/CENTRE/ARM. No _adapted variant was made: T1's failure is a body verdict (no scaffold stride at its MJCF ratings), probed with counterfactual bodies through the existing SCALE / SPRINT_SCALE_ANKLE switches."
out["notes"] = ["pycma seed=0 means a time-based random seed: seed-0 runs (these and the G1/H1 ladders) are not bit-reproducible; the z vectors are stored and replay deterministically",
                "Apollo omega is ASSUMED (G1/H1 power-per-kg rule); its result is conditional on that rule"]
json.dump(out, open(os.path.join(R, "heldout", "summary.json"), "w"), indent=1, default=float)
for k, v in out["ladders"].items():
    if v: t = v["top_valid"]; print(k, "top valid", (t["v"], t["v_meas"]) if t else None, [(r["v"], r["ok"]) for r in v["rungs"]])
for k, v in ab.items(): print(k, v["evals_to_first_valid"], v["best_speed"])
