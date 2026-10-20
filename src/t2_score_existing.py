"""T2: score the dynamic predictors (and the old static ones) against the EXISTING verified found values.
Found values are copied from the verified synthesis (results/overnight/BALANCE_LEVERAGE_SPRINT_2026-10-17.md) and the T2 slip-gated
re-score of track A (results/overnight/T2/rescore_A.json). ratio = predicted / found (> 1: optimistic). A found value that is a lower
bound (search budget, ladder spacing) makes the ratio an upper bound on the true error; a found value '< x' makes it a lower bound.
-> results/overnight/T2/score_existing.json (+ markdown table on stdout)"""
import os, json
T2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "overnight", "T2")
A = json.load(open(f"{T2}/pred_headcarry.json"))
rs = json.load(open(f"{T2}/rescore_A.json"))


def slip_max(prefix):
    v = [r["mass"] for k, r in rs.items() if k.startswith(prefix) and r["disturbance"] is None and r["valid_slip_both"]]
    return max(v) if v else None


def sp(name):
    p = f"{T2}/pred_sprint_{name}.json"
    return json.load(open(p))["v_max"] if os.path.exists(p) else None


rows = [
    # (task, case, predicted_dyn, predicted_static_old, found, found_kind, evidence)
    ("head-carry", "head pot (G1H)", A["head_pot"]["m_max"], 62.57, slip_max("head_pot_"), "slip-gated, both dt; 38 kg fails (OFF_PATCH 0.5 ms, wrist yaw saturated)", "results/overnight/T2/rescore_A.json"),
    ("head-carry", "head box (G1H)", A["head_box"]["m_max"], 65.21, slip_max("head_box_"), "slip-gated, both dt (16-32 kg skate 0.36-0.74 m at 0.5 ms: lower bound; old gate 32)", "results/overnight/T2/rescore_A.json"),
    ("head-carry", "head box (old gate, 1 ms clean)", A["head_box"]["m_max"], 65.21, 32.0, "track A gate (no slip check)", "results/overnight/A/head_box_32kg_warm.json"),
    ("head-carry", "hands box (G1H)", A["hands_box"]["m_max"], 3.73, slip_max("hands_box_"), "slip-gated, both dt", "results/overnight/T2/rescore_A.json"),
    ("head-carry", "hands pot (G1H)", A["hands_pot"]["m_max"], 3.70, slip_max("hands_pot_"), "slip-gated, both dt (strictly hands-only 2 kg: the 4 kg pot is a passive clamp)", "results/overnight/T2/rescore_A.json"),
]
for d, sd in (("push", "push"), ("pull", "pull")):
    p = json.load(open(f"{T2}/pred_sled_{d}_mu0.4.json"))
    rows.append(("sled mu 0.4 (G1, weld)", f"{d} total kg", p["m_max_total"], 30.0 if d == "push" else 83.5,
                 (21.0 if d == "pull" else "<21"), ("robust both dt 2/4 (31 kg single plant history)" if d == "pull" else "no mass moved 0.2 m (21 kg: 0.177 m)"),
                 "results/overnight/B/summary.json, VERIFY_B"))
for name, robot, found, kind, ev in (("G1_ank1_lag2ms", "G1", 2.2143, "valid 2.0 rung (2.5 rung FELL at 2.36)", "results/sprint/summary.json"),
                                     ("H1_ank1_lag2ms", "H1", 3.2147, "valid 3.0 rung (3.5 rung OVERSPEED+SLIP at 3.78)", "results/sprint/trackC/summary.json"),
                                     ("Apollo_ank1_lag2ms", "Apollo", 2.4438, "valid 1.5 rung (2.0 rung FELL)", "results/sprint/ladder_Apollo_heldout_lag2ms.json"),
                                     ("T1_ank1_lag2ms", "T1", None, "no valid gait (no feasible scaffold stride at 1.0)", "results/sprint/ladder_T1_heldout_lag2ms.json"),
                                     ("G1_ank1_lag10ms", "G1 lag 10 ms", 2.2525, "valid 2.0 rung, worse dt", "results/sprint/ladder_G1_lagsweep_0.010.json")):
    old = {"G1": 4.75, "H1": 6.0, "Apollo": 5.0, "T1": 1.5}.get(robot.split()[0])
    rows.append(("sprint (stock)", robot, sp(name), old, found, kind, ev))
for sc, found, src in ((1.25, 3.8049, "_lag2ms_cont_x1.25ank (3.0 rung only)"), (1.5, 3.6441, "_lag2ms_cont_x1.5ank"), (2.0, 3.732, "_night_S2ankle_x2.0 (3.0 rung only)"),
                       (2.8, 3.1677, "_lag2ms_cont_x2.8ank (3.0 rung only)"), (3.0, 4.6154, "_C_ank3_climb"), (4.0, 5.0829, "_C_ank4.0_climb"), (5.0, 5.0832, "_C_ank5.0_climb")):
    old = 5.25 if sc >= 3 else 0.0
    rows.append(("sprint WHAT-IF body (H1 ankle x scale, dose-response only)", f"H1 ankle x{sc:g}", sp(f"H1_ank{sc:g}_lag2ms"), old, found, "found top valid (lower bound: rungs/budget)", f"results/sprint/ladder_H1{src.split(' ')[0]}.json"))
out = []
print("| task | case | dynamic pred | ratio | old static pred | old ratio | found | found note |\n|---|---|---|---|---|---|---|---|")
for t, c, pdyn, pst, f, kind, ev in rows:
    fv = f if isinstance(f, (int, float)) else (float(f[1:]) if isinstance(f, str) and f.startswith("<") else None)
    r = (pdyn/fv if (pdyn is not None and fv) else None); ro = (pst/fv if (pst is not None and fv) else None)
    rr = (f">= {r:.2f}" if isinstance(f, str) and r is not None else (f"{r:.2f}" if r is not None else "n/a"))
    rro = (f">= {ro:.2f}" if isinstance(f, str) and ro is not None else (f"{ro:.2f}" if ro is not None else "n/a"))
    out.append(dict(task=t, case=c, predicted_dynamic=pdyn, predicted_static_old=pst, found=f, found_note=kind, ratio_dynamic=r, ratio_static_old=ro, evidence=ev))
    print(f"| {t} | {c} | {pdyn} | {rr} | {pst} | {rro} | {f} | {kind} |")
json.dump(out, open(f"{T2}/score_existing.json", "w"), indent=1)
