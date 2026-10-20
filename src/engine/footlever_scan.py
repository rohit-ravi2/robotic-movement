"""Foot-lever ceiling (engine/sprint_ceiling.py, CEIL_FOOT model) as a function of the ANKLE-ONLY rating scale (tau and omega x k),
hip/knee stock, for H1 (2026-10-17, track C). Computed from the spec only; compared afterwards to the searched ladders.
usage: python engine/footlever_scan.py [robot] [out.json]"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import sprint_ceiling as C
robot = sys.argv[1] if len(sys.argv) > 1 else "H1"; out = sys.argv[2] if len(sys.argv) > 2 else f"../results/sprint/trackC/footlever_scan_{robot}.json"
B = C.foot_spec(robot); rows = []
for k in (1.0, 1.5, 2.0, 2.5, 2.91, 3.0, 4.0, 5.0):
    row = dict(ankle_scale=k)
    for cap in (None, 6.0, 5.5, 4.5):
        best, fail = C.ceiling_ext(B, foot=True, max_hz=cap, ankle_scale=k)
        row["any" if cap is None else f"le_{cap}Hz"] = dict(v_max=best["v"] if best else 0.0, binding=(max(fail["fails"], key=fail["fails"].get) if fail and fail["fails"] else None),
                                                          step_hz=(round(best["stride"]["step_hz"], 3) if best else None), tc=(best["stride"]["tc"] if best else None), tf=(best["stride"]["tf"] if best else None))
    rows.append(row); print(k, {c: (row[c]["v_max"], row[c]["binding"]) for c in row if c != "ankle_scale"}, flush=True)
json.dump(dict(robot=robot, model="foot-lever ceiling (CoP heel->toe, toe-pivot push-off, point-foot GRF), hip/knee stock, ankle tau and omega x k", rows=rows), open(out, "w"), indent=1, default=float)
