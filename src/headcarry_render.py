"""Render Track A cases (2026-10-17): re-run a case's best z at 1 ms with recording (law on, and optionally law off) and write
mp4 + contact sheet under results/overnight/A/. usage: python headcarry_render.py <case tag> [--ablation]"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from engine import headcarry as HC, render_any as RA

if __name__ == "__main__":
    tag = sys.argv[1]; C = json.load(open(f"{HC.RES}/{tag}.json")); z = np.asarray(C["best_z"]); sway = C.get("sway")
    for law in ([True, False] if "--ablation" in sys.argv else [True]):
        sw_, pu_ = HC.disturbance(sway); r = HC.rollout(C["obj"], C["mass"], C["strategy"], z, C["T"], 0.001, True, sw_, law=law, push=pu_)
        out = f"{HC.RES}/{tag}{'' if law else '_lawoff'}.mp4"; np.save(out.replace(".mp4", "_trace.npy"), r["trace"])
        what = ("on the head, palms balancing" if C["strategy"] == "head" else "between the palms") + ("" if law else " - LAW OFF (ablation)")
        RA.render(r["scene"], r["trace"], r["trace_dt"], out, title=f"G1H: {C['mass']:g} kg {C['obj']} {what}" + ({"sway": " + lateral sway", "push": " + 3 torso pushes"}.get(sway[0], "") if sway else ""),
                  sub=f"held {r['t_held']:.1f}/{C['T']:g} s ({r['reason'] or 'gates pass'}), head carries {100*r['head_frac']:.0f}%, palms {r['palm_N_mean']['l']:.0f}/{r['palm_N_mean']['r']:.0f} N, tilt max {r['tilt_max_deg']:.1f} deg",
                  azimuth=150, distance=2.6, elevation=-8)
        print(out, r["t_held"], r["reason"], flush=True)
