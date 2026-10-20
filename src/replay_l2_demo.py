"""Replay a stored L2 sheet-lift result at 1 ms and 0.5 ms, check the numbers against the stored ones, render the 1 ms trace (2026-10-17, track C).
usage: python replay_l2_demo.py <L2 json> <out.mp4> [azimuth=135] [distance=5.5] [elevation=-25]
Abstraction (as in the probe): both palms WELDED to the sheet at the searched grips; the initial vertical support is withdrawn over 1 s; simulator only."""
import sys, json, numpy as np
A = sys.argv
import leverage_probes as LP
J = json.load(open(sys.argv[1])); kind, mass, T = J["kind"], J["mass"], J["T"]; z = np.asarray(J["search"]["best_z"])
r1 = LP.l2_eval((kind, mass, z, T, 0.001, True)); r2 = LP.l2_eval((kind, mass, z, T, 0.0005, False))
s1, s2 = J["search"]["best"], J["check_dt0005"]
for name, r, s in (("1 ms", r1, s1), ("0.5 ms", r2, s2)):
    print(f"{name}: t_held {r['t_held']} (stored {s['t_held']})  util_rms {r['util_rms']:.4f} (stored {s['util_rms']:.4f})  util_p95 {r['util_p95']} (stored {s['util_p95']})  reason {r['reason']}")
    assert abs(r["t_held"] - s["t_held"]) < 1e-6 and abs(r["util_rms"] - s["util_rms"]) < 1e-3, f"{name} replay differs from the stored result"
from engine import render_any as RA
RA.render(r1["scene"], r1["trace"], r1["trace_dt"], sys.argv[2], title=f"G1H holds a {mass:g} kg {kind} sheet (grip found by search)",
          sub=f"held {r1['t_held']:.1f}/{T:g} s at 1 ms and {r2['t_held']:.1f} s at 0.5 ms · palms WELDED to the sheet · simulator only", azimuth=float(A[3]) if len(A) > 3 else 135, distance=float(A[4]) if len(A) > 4 else 5.5, elevation=float(A[5]) if len(A) > 5 else -25)
print("wrote", sys.argv[2])
