"""Re-measure (1 ms) the landability10 runs flagged by ENERGY_CLOSURE under the recalibrated tolerance."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, branch_i as BI, mocap_baseline as MB, aprior_standard as AS
P = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase10/landability10.json")
R = json.load(open(P)); n = 0
for r in R:
    if "ENERGY_CLOSURE" not in r["reasons"]: continue
    c = AS.clip0() if r["clip"] == "knees_bent" else MB.load_clip(r["clip"], "retargeted_g1_all.pkl")
    ref = MB.analytic_clip(c) if r["method"] == "APRIOR" else c
    S = BI.setup(c["start"]); BI.apply_dt(S, 0.001); o = MB.run(S, ref, np.array(r["x"]), r["omega"], r["tscale"]); BI.apply_dt(S, 0.004)
    ok = not [z for z in o["validity"]["reasons"] if z != "CONTACT_ENERGY"] and not o.get("bound_violation")
    r.update(v=o["v"], valid=bool(ok), reasons=o["validity"]["reasons"], landable=bool(ok and o["mech"]["flight_rotation"] <= BI.ROT_MAX),
             closure=o["energy_audit"]["closure"], rescored="closure_tol_v2"); n += 1
    print(r["method"], r["clip"][:10], r["seed"], round(r["v"], 2), r["valid"], r["landable"], r["reasons"], flush=True)
json.dump(R, open(P, "w"), default=float); print("rescored", n)
for cl in sorted({r["clip"] for r in R}):
    for m in ("APRIOR", "NATIVE"):
        g = [r for r in R if r["clip"] == cl and r["method"] == m]; L = [r["v"] for r in g if r["landable"]]
        print(f"{cl[:14]:14s} {m:6s} landable {len(L)}/{len(g)} median {np.median(L) if L else float('nan'):.2f}")
