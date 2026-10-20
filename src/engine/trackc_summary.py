"""Collect track C ladders (2026-10-17) into results/sprint/trackC/summary.json: per ladder the rungs (valid at both dt, measured speeds, gate
reasons / periodicity), top valid speed, binding joints at the top (torque p95 utilisation), technique; plus the ankle-scale vs speed table
against the foot-lever ceiling (engine/footlever_scan.py)."""
import os, sys, json, glob
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "sprint"))
def load(p): return json.load(open(p))
def rung_row(r):
    if "meas" not in r: return dict(v=r["v"], ok=False, note=r.get("no_scaffold"))
    m1, m2 = r["meas"]; why = sorted(set(m1["reasons"] + m2["reasons"])) or ([] if m1["periodic"] and m2["periodic"] else [f"not periodic (per_n {m1['per_n']}/{m2['per_n']})"])
    return dict(v=r["v"], ok=r["ok_both_dt"], v_meas=[m1["value"], m2["value"]], why_not=None if r["ok_both_dt"] else why, evals=r["evals"], secs=r["secs"], stride_extrapolated=r.get("stride_extrapolated"))
def top_of(L):
    ok = [r for r in L["rungs"] if r.get("ok_both_dt")]
    if not ok: return None
    t = max(ok, key=lambda r: min(m["value"] for m in r["meas"])); u = t["utilisation"]
    bind = sorted(((k, v["torque_p95"], v["torque"], v["speed"]) for k, v in u.items()), key=lambda x: (-x[1], -x[3]))[:6]
    spd = sorted(((k, v["speed"]) for k, v in u.items()), key=lambda x: -x[1])[:4]; te = t["technique"]
    return dict(rung=t["v"], v_valid=min(m["value"] for m in t["meas"]), binding_torque_p95=bind, top_speed_util=spd, cadence_hz=te.get("cadence_hz"), stride_len=te.get("stride_len"),
                flight_mean=te.get("flight_mean"), flight_fraction=te.get("flight_fraction"), duty=te.get("duty"), stride_tc_tf=(t["stride"]["tc"], t["stride"]["tf"]))
out = {}
for p in sorted(glob.glob(f"{R}/ladder_H1_C_*.json")) + [f"{R}/ladder_H1_night_S2ankle_climbfree.json", f"{R}/ladder_H1_night_S2ankle_x1.5.json", f"{R}/ladder_H1_night_S2ankle_x2.0.json", f"{R}/ladder_H1_night_S2ankle_x3.0.json", f"{R}/ladder_H1_lag2ms_extrap.json"]:
    L = load(p); out[os.path.basename(p)[len("ladder_H1"):-5]] = dict(scale=L.get("scale"), budget=L.get("budget"), rungs=[rung_row(r) for r in L["rungs"]], top=top_of(L))
FL = load(f"{R}/trackC/footlever_scan_H1.json")
def fl(k):
    best = None
    for row in FL["rows"]:
        if row["ankle_scale"] <= k + 1e-9: best = row
    return best
tab = []
for k in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
    found = [(n, o["top"]["v_valid"]) for n, o in out.items() if o["top"] and abs((o["scale"] or 1.0) - k) < 1e-6 and ("ank" in n or "S2ankle" in n or (k == 1.0 and n == "_lag2ms_extrap"))]
    row = fl(k); tab.append(dict(ankle_scale=k, found_top_valid=max((v for _, v in found), default=None), found_from=[n for n, v in found if v == max((v for _, v in found), default=None)],
                                 footlever_any=row["any"]["v_max"], footlever_le6Hz=row["le_6.0Hz"]["v_max"], footlever_le5_5Hz=row["le_5.5Hz"]["v_max"]))
out["_ankle_vs_footlever"] = tab
json.dump(out, open(f"{R}/trackC/summary.json", "w"), indent=1, default=float)
for t in tab: print(t)
for n, o in out.items():
    if n == "_ankle_vs_footlever": continue
    print(n, "x", o["scale"], [(r["v"], r["ok"], r.get("v_meas"), r.get("why_not")) for r in o["rungs"]], "TOP", o["top"] and (o["top"]["v_valid"], o["top"]["binding_torque_p95"][:3], o["top"]["cadence_hz"], o["top"]["stride_len"], o["top"]["flight_mean"]))
