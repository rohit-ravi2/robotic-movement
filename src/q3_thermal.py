"""Q3 (2026-10-17): thermal reading of the G1 sprint's torque. A peak rating bounds the instantaneous torque; a continuous
(thermal) rating bounds the RMS torque over the motor's thermal time constant. For a stored z, record the 1 ms rollout and report
per joint: max |tau|, RMS |tau| over the whole run and over the periodic window, against the peak (MJCF) and URDF-effort values.
Usage: SPRINT_LAG=<s> [SPRINT_RATINGS=...] python q3_thermal.py <ladder json> <rung idx> <out json>"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from engine import sprint as SP, core
L = json.load(open(sys.argv[1])); R = L["rungs"][int(sys.argv[2])]
st = dict(R["stride"], extrapolated_from=None) if R.get("stride_extrapolated") else None
P = SP.Sprint("G1", R["v"]) if st is None else SP.Sprint("G1", R["v"], stride=R["stride"])
r = P.run(np.asarray(R["z"]), core.DT_OPT, record=True); tr = r["trace"]
tq = np.asarray(tr["tq"]); t = np.asarray(tr["t"]); keep = t >= SP.N_TRANS*P.R["Tc"]
cont = SP.rating_table("G1") if SP.RATINGS == "continuous" else {k: float(v["tau_max_continuous"]["value"]) for k, v in __import__("yaml").safe_load(open(os.path.join(os.path.dirname(__file__), "..", "spec", "g1.yaml")))["joints"].items()}
S = P.S[core.DT_OPT]; out = {}
for i, ty in enumerate(S["types"]):
    out[ty] = dict(tau_plant=float(S["jarr"]["tau"][i]), urdf_effort=cont.get(ty), max=round(float(np.abs(tq[:, i]).max()), 2),
                   rms_run=round(float(np.sqrt(np.mean(tq[:, i]**2))), 2), rms_periodic=round(float(np.sqrt(np.mean(tq[keep, i]**2))), 2) if keep.any() else None)
res = dict(src=sys.argv[1], rung=int(sys.argv[2]), v=R["v"], act_tau=SP.ACT_TAU, ratings=SP.RATINGS, value=r["value"], valid=r["valid"], t_run=r["t_run"], joints=out)
json.dump(res, open(sys.argv[3], "w"), indent=1); print(json.dumps({k: v for k, v in res.items() if k != "joints"}))
for k, v in out.items():
    if any(s in k for s in ("ankle", "waist", "hip", "knee")): print(k, v)
