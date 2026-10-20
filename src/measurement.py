"""Validated, model-independent summaries of recorded throw traces.

The simulator remains the source of truth for dynamics. This module only derives
diagnostics from ``throw.rollout(..., record=True)`` and refuses ambiguous inputs.
"""
from __future__ import annotations

import numpy as np


def summarize_trace(trace: dict, release_time: float | None = None) -> dict:
    """Return reproducible timing, speed, power and work summaries.

    ``seg_w`` is world-frame segment linear/angular speed magnitude as recorded by
    MuJoCo; ``jv`` and ``power`` are driven-joint quantities. These are intentionally
    reported separately: joint speed is not segment angular speed.
    """
    dt = float(trace["dt"])
    seg = np.asarray(trace["seg_w"], dtype=float)
    jv = np.asarray(trace["jv"], dtype=float)
    power = np.asarray(trace.get("power", []), dtype=float)
    if seg.ndim != 2 or seg.shape[0] == 0:
        raise ValueError("trace.seg_w must be a non-empty [time, segment] array")
    peak_i = np.argmax(seg, axis=0)
    peak_t = peak_i * dt
    out = {
        "dt": dt,
        "n_steps": int(seg.shape[0]),
        "segment_peak_speed": np.max(seg, axis=0).tolist(),
        "segment_peak_time": peak_t.tolist(),
        "segment_peak_time_relative_release": (peak_t - float(release_time)).tolist() if release_time is not None else None,
        "strict_peak_order": bool(np.all(np.diff(peak_i) >= 0)),
        "joint_peak_speed": np.max(np.abs(jv), axis=0).tolist() if jv.size else [],
    }
    if power.size:
        out["positive_work_by_joint"] = (np.maximum(power, 0).sum(axis=0) * dt).tolist()
        out["negative_work_by_joint"] = (np.minimum(power, 0).sum(axis=0) * dt).tolist()
        out["positive_work"] = float(np.maximum(power, 0).sum() * dt)
        out["negative_work"] = float(np.minimum(power, 0).sum() * dt)
    if "foot_load_bw" in trace:
        foot = np.asarray(trace["foot_load_bw"], dtype=float)
        out["foot_min_load_bw"] = np.min(foot, axis=0).tolist()
        out["foot_unloaded_fraction"] = np.mean(foot < 1e-3, axis=0).tolist()
    return out


def finite_difference(x: np.ndarray, dt: float) -> np.ndarray:
    """Second-order edge, centered interior derivative for a time-major trace."""
    x = np.asarray(x, dtype=float)
    if x.shape[0] < 2 or dt <= 0:
        raise ValueError("need at least two samples and positive dt")
    return np.gradient(x, dt, axis=0, edge_order=2 if x.shape[0] >= 3 else 1)
