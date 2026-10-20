"""Cinematic renders (2026-10-18): replay a stored head-carry result (law on or off) at 1 ms with recording and at 0.5 ms, compare
t_held / reason / pelvis drift with the stored record, and save trace + model for engine/cinematic.py.
usage: python cinematic_capture_headcarry.py t1pot|q1box on|off  ->  results/demo/cinematic/cache/hc_<case>_<law>.*"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
case, law = sys.argv[1], sys.argv[2] == "on"
if case == "t1pot":     # T1: zero actuator lag, stored z, law-off ablation in the same file
    from engine import headcarry as HC
    C = json.load(open(f"{ROOT}/results/overnight/T1/head_pot_40kg_t1.json")); z = np.asarray(C["best_z"]); obj, mass, lag = "pot", 40.0, 0.0
    st = [C["best"], C["check_dt0005"]] if law else [C["law_off_ablation"], json.load(open(f"{ROOT}/results/overnight/T1/ablation_at_max.json"))["head_pot_40kg_t1"]["lawoff_0.0005"]]
else:                   # Q1: 10 ms lag, servo cap c from z[14], lag-robust maximum and its ablation
    import q1_lag as Q
    L = json.load(open(f"{ROOT}/results/overnight/Q1/lag_maxima.json"))["head_box_lag10ms"]; z15 = np.asarray(L["lag_max_z"]); obj, mass, lag = "box", float(L["lag_max"]), 0.010
    HC = Q._setlag(lag, Q.cap_of(z15)); z = z15[:14]; ab = L["ablation"]; st = [ab[f"{'law' if law else 'lawoff'}_{dt}"] for dt in (0.001, 0.0005)]
rs = [HC.rollout(obj, mass, "head", z, 10.0, dt, dt == 0.001, law=law) for dt in (0.001, 0.0005)]
chk = dict(stored=[[round(s["t_held"], 3), s["reason"], round(s["pelvis_drift"], 5)] for s in st], replay=[[round(r["t_held"], 3), r["reason"], round(r["pelvis_drift"], 5)] for r in rs])
chk["match"] = all(a[0] == b[0] and a[1] == b[1] and abs(a[2] - b[2]) < 1e-4 for a, b in zip(chk["stored"], chk["replay"]))
r = rs[0]; out = f"{ROOT}/results/demo/cinematic/cache/hc_{case}_{'on' if law else 'off'}"
np.savez(out + ".npz", Q=np.asarray(r["trace"]), t=np.arange(len(r["trace"]))*r["trace_dt"]); mujoco.mj_saveModel(mujoco.MjModel.from_xml_path(r["scene"]), out + ".mjb", None)
json.dump(dict(case=case, obj=obj, mass=mass, lag=lag, law=law, check=chk, head_frac=r.get("head_frac"), tilt_max_deg=r.get("tilt_max_deg")), open(out + ".json", "w"), indent=1, default=float); print(case, law, chk)
