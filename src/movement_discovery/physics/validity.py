"""ValidityReport from a rollout record (engine.task record keys: reasons, violation, extra). ONE reason -> flag table (_FLAG, flags(),
classify(): the families' mapper, used by locomotion/aerial/balance/posture/negotiation, reaching, impact, projection and the OBJECT
families instead of their former copies) and a task-dependent POLICY (report()) that says which reasons are not violations for this
task (a jump may lose foot contact; a stationary throw may not; a push forbids an impulsive slap). Policies never add physics; they
only decide which simulator facts count."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional
import numpy as np
from movement_discovery.core.contracts import ValidityReport

def strip_phase(r: str) -> str:
    """P2:STANCE_LOST (ContactSchedule phase) or s1:NOT_REACHED (sequence step) -> STANCE_LOST / NOT_REACHED."""
    if ":" not in r: return r
    head, tail = r.split(":", 1)
    return tail if (head[:1] in ("P", "s") and head[1:].isdigit()) else r



REASONS = ("OVERSPEED", "STANCE_LOST", "FELL", "NONFOOT_CONTACT", "OUT_OF_CONE", "ENERGY_CLOSURE", "ENERGY_ACCOUNTING_VIOLATION", "NOT_HIT",
           "BALL_OFF_TARGET", "IMPULSIVE_CONTACT", "CONTACT_TOO_BRIEF", "NOT_RECOVERABLE", "START_NOT_AT_REST", "NOT_REACHED", "NOT_HELD",
           "NOT_GRASPED", "END_UNBALANCED", "NO_TAKEOFF", "CONTACT_ENERGY", "MOMENTUM_INCONSISTENT")
# ONE reason -> flags table for every family (the former per-family copies: force_interaction._base.validity, reaching/_local, impact/_local,
# projection/_local.effect_validity, locomotion/_local.base_validity). A reason may clear several flags (union of the families' conventions:
# IMPULSIVE_CONTACT was `contact` for the OBJECT families and `object_interaction` for the policies; both are cleared now). Reasons prefixed
# by a phase (P2:) or a sequence step (s1:) are stripped first. Unknown reasons clear no flag (the report still carries them).
_FLAG: Dict[str, tuple] = {
    # actuation / numerics
    "OVERSPEED": ("actuator_speed", "envelope"), "OVER_TORQUE": ("actuator_torque", "envelope"),
    "ENERGY_CLOSURE": ("numerical",), "ENERGY_ACCOUNTING_VIOLATION": ("numerical",), "CONTACT_ENERGY": ("numerical",), "MOMENTUM_INCONSISTENT": ("numerical",), "START_NOT_AT_REST": ("numerical",), "NO_TRACE": ("numerical",),
    # support / balance
    "STANCE_LOST": ("support",), "FELL": ("support",), "NOT_RECOVERABLE": ("support",), "END_UNBALANCED": ("support",), "NO_TAKEOFF": ("support",), "NOT_LANDED": ("support",), "NOT_RECOVERED": ("support",), "NOT_STANDING": ("support",),
    "FOOT_OFF_BEAM": ("support",), "TILTED": ("support",), "SEQUENCE_NOT_OK": ("support",), "NOT_UPRIGHT": ("support",), "NO_FLIGHT": ("support",),
    # contact with the world
    "NONFOOT_CONTACT": ("contact",), "OBSTACLE_HIT": ("contact",), "PENETRATION": ("contact",), "IMPULSIVE_CONTACT": ("contact", "object_interaction"), "CONTACT_TOO_BRIEF": ("contact", "object_interaction"), "BLOCKED": ("contact",),
    # direction / target
    "OUT_OF_CONE": ("direction",), "BALL_OFF_TARGET": ("direction", "object_interaction"), "TARGET_MISSED": ("direction",), "LANDS_SHORT": ("direction",), "ARRIVAL_TOO_FLAT": ("direction",), "ARRIVAL_TOO_FAST": ("direction",),
    "NO_REBOUND": ("direction", "object_interaction"), "NOT_ROLLED": ("direction", "object_interaction"), "TIP_OFF": ("direction",), "ORIENT_OFF": ("direction",), "PATH_NOT_COVERED": ("direction",), "NOT_IN_REGION": ("direction", "object_interaction"),
    "NOT_PAST_OBSTACLE": ("direction",), "NOT_CLEARED": ("direction",), "SHORT_OF_FAR_EDGE": ("direction",), "FORWARD_DRIFT": ("direction",),
    # object interaction
    "NOT_HIT": ("object_interaction",), "HIT_WITHOUT_TOOL": ("object_interaction",), "NOT_REACHED": ("object_interaction", "direction"), "NOT_HELD": ("object_interaction", "direction"), "NOT_GRASPED": ("object_interaction",),
    "NOT_MOVED": ("object_interaction",), "NOT_CAPTURED": ("object_interaction",), "DROPPED": ("object_interaction",), "NOT_LIFTED": ("object_interaction",), "NOT_PLACED": ("object_interaction",), "HOLD_LOST": ("object_interaction",),
    "TWO_HAND_LOST": ("object_interaction",), "NO_CONTACT": ("object_interaction",), "TOOL_OFF_SURFACE": ("object_interaction",), "FORCE_TOO_LOW": ("object_interaction",),
}
_ENERGY = ("ENERGY_CLOSURE", "ENERGY_ACCOUNTING_VIOLATION", "CONTACT_ENERGY")


def flags(reasons: Iterable[str]) -> Dict[str, bool]:
    """ValidityReport flag values (False where a reason clears the flag) for a reason list; phase / step prefixes stripped; a reason
    containing ENERGY clears `numerical` even when it is not in the table (the OBJECT families' rule)."""
    out: Dict[str, bool] = {}
    for r in reasons:
        rs = strip_phase(r)
        for f in _FLAG.get(rs, ()): out[f] = False
        if "ENERGY" in rs: out["numerical"] = False
    return out


def classify(rec: dict, extra_reasons: Iterable[str] = (), **overrides) -> ValidityReport:
    """THE record -> ValidityReport mapper of the families (no policy): reasons = the record's + extra ones not already there, violation
    = the record's (kept even when no reason remains), flags from the table; `overrides` preset flag values (e.g. contact=False) and the
    table's clearing still applies on top (the BODY families' base_validity semantics)."""
    base = list((rec or {}).get("reasons", []) or []); rs = base + [r for r in extra_reasons if r not in base]
    v = ValidityReport(reasons=rs, violation=float((rec or {}).get("violation", 0.0) or 0.0), **overrides)
    for f, val in flags(rs).items(): setattr(v, f, val)
    return v


@dataclass
class Policy:
    """Which simulator facts are violations for this task. allow_* lists reasons that are expected by the task's physics."""
    name: str = "stationary"
    allow: List[str] = field(default_factory=list)           # reasons that do not count (e.g. STANCE_LOST for a jump)
    require_hit: bool = False                                 # NOT_HIT counts even when the task's own record forgot it
    overspeed_tol: Optional[float] = None                     # None = the task's own (throw.OVERSPEED_TOL 1.10)
    notes: str = ""


POLICIES: Dict[str, Policy] = {
    "stationary": Policy("stationary", notes="feet of the support contract stay loaded; no non-foot contact"),
    "aerial": Policy("aerial", allow=["STANCE_LOST", "NO_TAKEOFF"], notes="flight is the task; NO_TAKEOFF is judged by the program"),
    "locomotion": Policy("locomotion", allow=["STANCE_LOST", "START_NOT_AT_REST"], notes="support changes are the task; phase handoffs are mid-motion"),
    "impact": Policy("impact", require_hit=True, notes="the object must be hit and leave in the cone"),
    "sustained_force": Policy("sustained_force", require_hit=True, notes="peak force cap and contact duration are gates (IMPULSIVE_CONTACT, CONTACT_TOO_BRIEF)"),
    "quasi_static": Policy("quasi_static", allow=[], notes="reach / hold gates; END_UNBALANCED is a violation"),
    "body_contact": Policy("body_contact", allow=["NONFOOT_CONTACT"], notes="the task puts a non-foot body part on the world (roll, crawl, brace with the hands)"),
    "posture": Policy("posture", allow=["NONFOOT_CONTACT", "STANCE_LOST", "FELL", "START_NOT_AT_REST"], notes="seated / kneeling / lying starts: body contact and an unloaded stance are the initial condition"),
}


def report(rec: dict, policy: Policy | str = "stationary", dt_check: Optional[bool] = None) -> ValidityReport:
    pol = POLICIES[policy] if isinstance(policy, str) else policy
    raw = list(rec.get("reasons", [])); kept = [r for r in raw if strip_phase(r) not in set(pol.allow)]
    if pol.require_hit and not raw and rec.get("extra", {}).get("dmin_obj", 0.0) > 0.05 and rec.get("extra", {}).get("ball_speed", 1.0) < 1e-3: kept.append("NOT_HIT")
    v = ValidityReport(reasons=kept, violation=float(rec.get("violation", 0.0)) if kept else 0.0, timestep_robust=dt_check)
    for f, val in flags(kept).items(): setattr(v, f, val)
    ex = rec.get("extra", {}) or {}
    if pol.overspeed_tol is not None and ex.get("overspeed") is not None and float(ex["overspeed"]) > pol.overspeed_tol: v.actuator_speed = False; v.reasons.append("OVERSPEED") if "OVERSPEED" not in v.reasons else None
    if ex.get("peak_util") is not None and float(ex["peak_util"]) > 1.0 + 1e-6: v.actuator_torque = False
    v.envelope = v.actuator_speed and v.actuator_torque
    return v


def policy_for(family: str) -> Policy:
    return POLICIES.get({"aerial": "aerial", "locomotion": "locomotion", "impact": "impact", "force_interaction": "sustained_force", "reaching": "quasi_static", "payload": "quasi_static", "negotiation": "locomotion", "posture": "posture", "balance": "stationary"}.get(family, "stationary"))


# ---------------------------------------------------------------- external work (disturbances, launched objects) and robot-only rest
def launch_energy(spec, m) -> float:
    """Kinetic energy injected by objects' qvel0 (WorldSpec objects): 0.5 m |v|^2 + 0.5 w^T I w for free bodies, 0.5 I w^2 for suspended hinges."""
    import mujoco
    E = 0.0
    for o in (spec.objects if spec is not None else []):
        if not o.get("qvel0"): continue
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["name"]); v0 = np.asarray(list(o["qvel0"]) + [0.0]*max(0, 6 - len(o["qvel0"])), float)
        if b < 0: continue
        if o["kind"] == "suspended": E += 0.5*float(np.sum(m.body_inertia[b]))*float(np.sum(v0**2))/3.0
        else: E += 0.5*float(m.body_mass[b])*float(v0[:3] @ v0[:3]) + 0.5*float(np.sum(m.body_inertia[b]*v0[3:6]**2))
    return float(E)


def external_work(rec: dict, telemetry: Optional[dict] = None, spec=None, m=None) -> float:
    """W_ext credited to the environment: disturbance work (telemetry Observer: xfrc_applied . body velocity) + launched-object energy."""
    W = float((telemetry or {}).get("disturbance_work", 0.0) or 0.0)
    if spec is not None and m is not None: W += launch_energy(spec, m)
    return W


def rejudge_energy_closure(rec: dict, W_ext: float, tol: Optional[float] = None) -> dict:
    """RE-JUDGE ENERGY_CLOSURE with external work (LABEL: energy_rejudged). throw.rollout closes energy as
    (E1 - E0) - W_act - W_con - W_pas = closure, crediting actuator work only; disturbances through xfrc_applied and objects launched
    after E0 add W_ext. Needs the rollout's closure (rec["extra"]["closure"] or rec["closure"]) and tolerance (closure_tol, default
    max(5 J, 2% of |W_act|)); without them the record is returned unchanged with a note. Removes the gate's 2.0 violation when it clears."""
    ex = rec.get("extra") or {}; cl = ex.get("closure", rec.get("closure")); out = dict(rec, extra=dict(ex, W_ext=float(W_ext), energy_rejudged=True))
    if cl is None: out["extra"]["energy_rejudged"] = "closure not recorded"; return out
    tol = float(tol if tol is not None else ex.get("closure_tol", max(5.0, 0.02*abs(float(ex.get("W_act", 0.0)))))); cl2 = float(cl) - float(W_ext); out["extra"]["closure_ext"] = cl2
    had = "ENERGY_CLOSURE" in rec.get("reasons", []); bad = abs(cl2) > tol
    reasons = [r for r in rec.get("reasons", []) if r != "ENERGY_CLOSURE"] + (["ENERGY_CLOSURE"] if bad else [])
    viol = float(rec.get("violation", 0.0)) + (2.0 if (bad and not had) else 0.0) - (2.0 if (had and not bad) else 0.0)
    out.update(reasons=reasons, violation=max(0.0, viol), valid=(viol <= 0 and not reasons)); return out


def robot_at_rest(rec: dict, S, tol: float = 0.05) -> Optional[bool]:
    """START_NOT_AT_REST judged on the ROBOT's hinge dofs only (objects given qvel0 are moving by design). None when the trace lacks qvel."""
    tr = rec.get("trace") or {}
    if not tr.get("qvel"): return None
    v0 = np.asarray(tr["qvel"][0], float); rd = [i for i in (S.get("robot_dofs") if S.get("robot_dofs") is not None else range(len(v0))) if i >= 6]
    return bool(np.max(np.abs(v0[rd])) < tol) if rd else None


def strip_object_rest(rec: dict, S, objects_launched: bool) -> dict:
    """With launched objects, drop START_NOT_AT_REST when the robot itself started at rest (LABEL: rest_rejudged)."""
    if not objects_launched or "START_NOT_AT_REST" not in rec.get("reasons", []): return rec
    ok = robot_at_rest(rec, S)
    if not ok: return rec
    return dict(rec, reasons=[r for r in rec["reasons"] if r != "START_NOT_AT_REST"], extra=dict(rec.get("extra") or {}, rest_rejudged="robot dofs only"))
