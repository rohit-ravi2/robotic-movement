"""Held-out sprint CEILING before any search (2026-10-16): engine.sprint_ceiling's planar stance/swing feasibility, unchanged, on the
held-out bodies, at three cadence caps (step frequency 1/(t_c+t_f): any, <= 6 Hz, <= 4.5 Hz). Same grid as sprint_ceiling.feasible; a
stride above the cap is skipped. G1/H1 recomputed with this code for reference. -> results/heldout/ceiling.json"""
import os, sys, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from engine import sprint_ceiling as SC


def feasible(B, v, cap):
    L = B["L1"] + B["L2"]; fails = {}
    for tc, tf, hr, dr in itertools.product(np.linspace(0.06, 0.40, 18), np.linspace(0.0, 0.20, 9), np.linspace(0.70, 0.98, 8), (0.0, 0.03, 0.06, 0.10)):
        if cap and 1.0/(tc + tf) > cap + 1e-9: fails["cadence_cap"] = fails.get("cadence_cap", 0) + 1; continue
        ok, why, st = SC.stance_ok(B, v, tc, tf, hr*L, dr*L)
        if not ok: fails[why] = fails.get(why, 0) + 1; continue
        ok, why, sw = SC.swing_ok(B, tc, tf, hr*L, st)
        if not ok: fails[why] = fails.get(why, 0) + 1; continue
        return True, dict(tc=float(tc), tf=float(tf), h0=float(hr*L), dz=float(dr*L), stride=float(v*(tc + tf)), step_hz=float(1/(tc + tf)), stance=st, swing=sw)
    return False, fails


if __name__ == "__main__":
    robots = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Apollo", "T1", "G1", "H1"]; out = {}
    for robot in robots:
        B = SC.body_spec(robot); out[robot] = dict(body=B, caps={})
        for cap in (None, 6.0, 4.5):
            best = None; fail = None
            for v in np.arange(0.5, 16.01, 0.25):
                ok, info = feasible(B, float(v), cap)
                if ok: best = dict(v=float(v), stride=info)
                else: fail = dict(v=float(v), fails={k: n for k, n in info.items() if k != "cadence_cap"}); break
            binding = max(fail["fails"], key=fail["fails"].get) if fail and fail["fails"] else None
            out[robot]["caps"]["any" if cap is None else f"<={cap:g}Hz"] = dict(v_max=best["v"] if best else 0.0, at_vmax=best, first_infeasible=fail, binding=binding)
            print(f"{robot} cap {cap}: v_max {best['v'] if best else 0:.2f} m/s, stride {({k: round(x, 3) for k, x in best['stride'].items() if isinstance(x, float)} if best else None)}, binding next {binding}", flush=True)
    os.makedirs("../results/heldout", exist_ok=True); json.dump(out, open("../results/heldout/ceiling.json", "w"), indent=1, default=float)
