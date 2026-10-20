"""Render a K2 hold (lag-aware stand, lag on) at 1 ms with recording; writes mp4 + contact sheet + trace under results/overnight/K2/.
usage: python k2_render.py <obj> <lag s> [refine]   (default: the ladder maximum; 'refine': the refine.json rung if valid)"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import k2_lag as K

if __name__ == "__main__":
    obj, lag = sys.argv[1], float(sys.argv[2]); key = f"{obj}_lag{int(round(lag*1000))}ms"
    if len(sys.argv) > 3: R = json.load(open(f"{K.OUT}/refine.json"))[key]; mass, z = R["mass"], R["z"]
    else: L = json.load(open(f"{K.OUT}/ladder_{key}.json")); mass, z = L["max"], L["max_z"]
    HC = K.setmode(lag, K.COMP, True); from engine import render_any as RA
    r = HC.rollout(obj, mass, "head", np.asarray(z[:14]), 10.0, 0.001, True)
    out = f"{K.OUT}/head_{obj}_{mass:g}kg_lag{int(round(lag*1000))}ms.mp4"; np.save(out.replace(".mp4", "_trace.npy"), r["trace"])
    RA.render(r["scene"], r["trace"], r["trace_dt"], out, title=f"G1H stock: {mass:g} kg {obj} on the head, {lag*1000:g} ms actuator lag (lag-aware stand)",
              sub=f"held {r['t_held']:.1f}/10 s ({r['reason'] or 'gates pass'}), head carries {100*r['head_frac']:.0f}%, tilt max {r['tilt_max_deg']:.1f} deg, pelvis drift {1000*r['pelvis_drift']:.1f} mm",
              azimuth=150, distance=2.6, elevation=-8)
    print(out, r["t_held"], r["reason"], r["pelvis_drift"], r["foot_travel_max"], flush=True)
