"""Behaviour descriptors (DESCRIPTIVE ONLY; used by selection for diversity, never as a technique filter). From the rollout
record's trace and the telemetry summary: duration, support-change count, flight fraction, contact-history signature (fraction
of time in each support state), joint-use distribution (entropy and group shares of positive work), base displacement, peak
COM speed, peak angular momentum, envelope utilisation. All scalars in a flat dict; missing channels are simply absent."""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np


def describe(rec: dict, S=None, telemetry: Optional[dict] = None, events: Optional[list] = None) -> Dict[str, float]:
    out: Dict[str, float] = {}; tr = rec.get("trace") or {}; ex = rec.get("extra") or {}
    if tr:
        dt = float(tr.get("dt", 0.001)); out["duration"] = float(tr.get("release_time", len(tr.get("q", []))*dt))
        if tr.get("foot_load_bw"):
            F = np.asarray(tr["foot_load_bw"], float); on = F >= 0.02; code = on[:, 0]*2 + on[:, 1]
            for k, nm in ((3, "both"), (2, "left"), (1, "right"), (0, "flight")): out[f"support_{nm}_frac"] = float(np.mean(code == k))
            out["support_changes"] = float(np.count_nonzero(np.diff(code)))
            out["load_asymmetry"] = float(np.mean(F[:, 0] - F[:, 1])) if len(F) else 0.0
        if tr.get("power") and S is not None:
            P = np.asarray(tr["power"], float); W = np.clip(P, 0, None).sum(0)*dt; tot = float(W.sum()) + 1e-9; sh = W/tot
            out["joint_use_entropy"] = float(-np.sum(sh[sh > 0]*np.log(sh[sh > 0]))/np.log(max(len(sh), 2))); out["n_joints_active"] = float(np.sum(sh > 0.02))
            g = {}
            for gi, w in zip(S["groups"], W): g[gi] = g.get(gi, 0.0) + float(w)
            for gi, w in g.items(): out[f"work_share_{gi}"] = w/tot
            out["work_total"] = tot
        if tr.get("jv"):
            jv = np.asarray(tr["jv"], float); out["peak_joint_speed"] = float(np.abs(jv).max()); out["t_peak_joint_speed_frac"] = float(np.argmax(np.abs(jv).max(1))/max(len(jv) - 1, 1))
        if tr.get("seg_w"):
            sw = np.asarray(tr["seg_w"], float); pk = np.argmax(sw, axis=0); out["chain_order_monotone"] = float(np.all(np.diff(pk) >= 0)); out["chain_lag_s"] = float((pk[-1] - pk[0])*dt) if len(pk) > 1 else 0.0
    for k in ("com_speed", "overspeed", "peak_util", "cp_err_rel", "trunk_pitch_max", "knee_max", "t_rel", "dmin_obj", "t_contact", "f_max_obj"):
        if ex.get(k) is not None and isinstance(ex[k], (int, float, np.floating)): out[k] = float(ex[k])
    if ex.get("base_displacement") is not None: out["base_disp_xy"] = float(np.hypot(*ex["base_displacement"][:2])); out["base_disp_z"] = float(ex["base_displacement"][2])
    if telemetry:
        for k in ("com_speed_max", "L_ang_max", "util_torque_max", "grf_max_bw", "disturbance_work"):
            if telemetry.get(k) is not None: out[k] = float(telemetry[k])
        if telemetry.get("com_disp") is not None: out["com_disp_xy"] = float(np.hypot(*telemetry["com_disp"][:2])); out["com_disp_z"] = float(telemetry["com_disp"][2])
    if events is not None: out["n_events"] = float(len(events)); out["n_contact_loss"] = float(sum(e["kind"] == "contact_loss" for e in events))
    return out


DEFAULT_KEYS = ("duration", "support_changes", "support_both_frac", "support_flight_frac", "joint_use_entropy", "work_share_legs", "work_share_trunk", "chain_lag_s", "com_speed", "base_disp_xy")
