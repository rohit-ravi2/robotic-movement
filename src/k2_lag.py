"""TRACK K2 (2026-10-18): LAG-AWARE STAND for head carry on stock G1H. See engine.headcarry LAG_COMP (internal lag model in the
servo law, gains from the plant). Jobs run on FRESH plants (t2_gate.eval_job -> HC.plant fresh), gate t2_gate.valid_slip.
job tuple: (key, strategy, obj, mass, z, dt, lag, comp, place, law, wcap)."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = f"{ROOT}/results/overnight/K2"
KEEP = ("t_held", "reason", "head_frac", "tilt_max_deg", "foot_travel_max", "pelvis_drift", "foot_slide_max", "com_drift_max", "gate_hc", "slip_ok", "valid_slip", "e_max", "util_p95", "act_lag", "joint_sat", "error")


def J(o): return json.loads(json.dumps(o, default=lambda x: x.tolist() if hasattr(x, "tolist") else (float(x) if isinstance(x, (np.floating, np.integer)) else str(x))))
def log(s, f="status.txt"):
    line = time.strftime("%H:%M:%S ") + s; print(line, flush=True); open(f"{OUT}/{f}", "a").write(line + "\n")


COMP = "pred"   # the K2 lag-aware stand (engine.headcarry LAG_COMP): PD on q + tau qdot, balance loop on the state integrated tau ahead


def setmode(lag, comp, place, wcap=0.0, tau_hat=1.0):
    from engine import headcarry as HC, dynpred_headcarry as D
    D.register_objects(); HC.ACT_LAG = float(lag); HC.LAG_WCAP = float(wcap); HC.LAG_COMP = comp; HC.PLACE = bool(place); HC.TAU_HAT_SCALE = float(tau_hat); return HC


def unloaded_ok(r):
    from engine import headcarry as HC
    return bool(r.get("t_held", 0) >= 10.0 - 1e-6 and r.get("reason") is None and HC.support_ok(r) and r.get("slip_ok"))


def job(a):
    """a = (key, strategy, obj, mass, z, dt, lag, comp, place, law, wcap[, dist[, tau_hat]])"""
    key, st, obj, mass, z, dt, lag, comp, place, law, wcap = a[:11]; dist = a[11] if len(a) > 11 else None; th = a[12] if len(a) > 12 else 1.0
    setmode(lag, comp, place, wcap, th); import t2_gate as G
    r = G.eval_job((obj, mass, st, np.asarray(z[:14]), 10.0, dt, dist, law)); r = {k: r.get(k) for k in KEEP}
    r["unloaded_ok"] = unloaded_ok(r) if not place else None; return key, r


def score(a):
    st, obj, mass, z, lag, comp = a; HC = setmode(lag, comp, True)
    r = HC.evaluate((obj, mass, st, np.asarray(z[:14]), 10.0, 0.001, None)); r["gate1"] = HC.gate(r, 10.0, st, mass) if "foot_travel_max" in r else False
    return {k: r.get(k) for k in ("score", "t_held", "reason", "foot_travel_max", "pelvis_drift", "gate1", "head_frac")}
