"""Event detection from recorded traces: release, impact, grasp, contact loss / regain per foot, support-polygon changes,
phase handoffs. Input: an engine.task record (throw.rollout trace keys: foot_load_bw [n x 2], dt, release_step, phase_bounds;
reach extras: t_reach, t_grasp; impact extras: t_contact, f_max_obj) plus the telemetry observer's channels when present
(contacts, object states). Output: a list of dict(kind, t, **data), time-ordered. Descriptive only."""
from __future__ import annotations
from typing import List, Optional
import numpy as np

UNLOAD_BW = 0.02      # a foot is "off" below 2% body weight
MIN_DUR = 0.020       # s: shorter unload/reload runs are contact chatter (branch_i.FLIGHT_CONFIRM)


def _runs(mask: np.ndarray, dt: float, min_dur: float):
    out = []; i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]: j += 1
            if (j - i)*dt >= min_dur: out.append((i*dt, j*dt))
            i = j
        else: i += 1
    return out


def detect(rec: dict, telemetry: Optional[dict] = None) -> List[dict]:
    ev = []; tr = rec.get("trace") or {}; ex = rec.get("extra") or {}
    dt = float(tr.get("dt", 0.001)) if tr else None
    if tr and tr.get("foot_load_bw"):
        F = np.asarray(tr["foot_load_bw"], float)
        for i_f, name in enumerate(("left_foot", "right_foot")):
            off = F[:, i_f] < UNLOAD_BW
            for (t0, t1) in _runs(off, dt, MIN_DUR):
                ev.append(dict(kind="contact_loss", t=float(t0), foot=name))
                if t1 < len(F)*dt - dt: ev.append(dict(kind="contact_regain", t=float(t1), foot=name))
        sup = (F >= UNLOAD_BW).astype(int); code = sup[:, 0]*2 + sup[:, 1]; names = {0: "none", 1: "right", 2: "left", 3: "both"}
        runs = []; i = 0                                           # de-bounced: a support state counts once it persists >= MIN_DUR
        while i < len(code):
            j = i
            while j < len(code) and code[j] == code[i]: j += 1
            if (j - i)*dt >= MIN_DUR: runs.append((i, int(code[i])))
            i = j
        for (i0, c0), (i1, c1) in zip(runs, runs[1:]):
            if c0 != c1: ev.append(dict(kind="support_change", t=float(i1*dt), frm=names[c0], to=names[c1]))
    if tr and tr.get("release_step") is not None: ev.append(dict(kind="release", t=float(tr.get("release_time", tr["release_step"]*(dt or 0.001))), speed=rec.get("value")))
    for k, b in enumerate(tr.get("phase_bounds", [])[1:-1]): ev.append(dict(kind="phase_handoff", t=b*dt, phase=k + 1))
    if ex.get("t_reach") is not None: ev.append(dict(kind="reach", t=float(ex["t_reach"]), dist=ex.get("dmin")))
    if ex.get("t_grasp") is not None: ev.append(dict(kind="grasp", t=float(ex["t_grasp"])))
    if ex.get("t_contact") is not None and float(ex["t_contact"]) > 0: ev.append(dict(kind="impact", t=None, duration=float(ex["t_contact"]), f_max=ex.get("f_max_obj"), object_speed=ex.get("ball_speed")))
    if telemetry and telemetry.get("object_contact_t"):
        for t in telemetry["object_contact_t"][:1]: ev.append(dict(kind="object_contact", t=float(t)))
    ev.sort(key=lambda e: (e["t"] is None, e["t"] if e["t"] is not None else 0.0)); return ev


def summary(events: List[dict]) -> dict:
    kinds = {}
    for e in events: kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    return dict(n=len(events), kinds=kinds, support_changes=kinds.get("support_change", 0), contact_losses=kinds.get("contact_loss", 0))
