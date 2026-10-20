"""Campaign rules. PASS = valid at BOTH timesteps (the robust measure: every dt valid, violation <= 0) AND the program's `achieved`
flag; the reasons are recorded otherwise. Multi-step Plans (optimization.sequence) have no two-dt measure in the runner: their PASS
is valid at the faithful timestep AND achieved, always tagged SINGLE_DT_ONLY(sequence) so the report shows the weaker standard. Regime labels are DESCRIPTIVE (reporting only; never a filter, never a technique)."""
from __future__ import annotations
from typing import Dict, List, Tuple

REGIMES = ("velocity_from_stance", "one_foot", "sustained_force", "receiving", "stepping_base", "non_standing_start", "aerial", "compound")

# program id -> regime (what the physical problem asks of the support, not how to do it)
REGIME: Dict[str, str] = {
    **{f"P{n:02d}": "stepping_base" for n in (1, 2, 3, 4, 5, 6, 7, 8)},                       # locomotion
    **{f"P{n:02d}": "aerial" for n in (9, 10, 11, 12, 13, 14, 15, 16)},                      # aerial
    **{f"P{n:02d}": "sustained_force" for n in (17, 18, 19, 21, 22, 23)},                    # balance under load / disturbance
    "P20": "stepping_base", "P24": "stepping_base",                                           # trip, partial support loss
    **{f"P{n:02d}": "non_standing_start" for n in (25, 26, 27, 28, 29, 30, 31)},             # posture
    "P32": "one_foot",
    **{f"P{n:02d}": "velocity_from_stance" for n in (33, 34, 35, 36, 37, 38, 39, 40)},       # reaching (quasi-static effector motion from stance)
    **{f"P{n:02d}": "velocity_from_stance" for n in (41, 42, 43, 44, 45, 46, 47, 48)},       # projection
    **{f"P{n:02d}": "velocity_from_stance" for n in (49, 50, 51, 53, 54, 55, 56)},           # impact
    "P52": "one_foot",                                                                        # foot-mediated impact
    **{f"P{n:02d}": "sustained_force" for n in (57, 58, 59, 60, 61, 62, 63, 64)},            # force interaction
    **{f"P{n:02d}": "sustained_force" for n in (65, 66, 67, 68, 69, 71, 72)},                # payload lifts / placement
    "P70": "stepping_base",
    **{f"P{n:02d}": "receiving" for n in (73, 74, 75, 76, 77, 78, 79, 80)},                  # interception
    **{f"P{n:02d}": "stepping_base" for n in (81, 82, 83, 84, 87, 88)},                      # negotiation on foot
    "P85": "sustained_force", "P86": "velocity_from_stance",
    **{f"P{n:02d}": "compound" for n in range(89, 97)},
}


def regime_of(program_id: str) -> str:
    return REGIME.get(program_id.split("_")[0], "unlabelled")


def judge(res: dict) -> Tuple[bool, List[str]]:
    """(pass, reasons) from a worker result dict (worker._exec's JSON: outcome + measured)."""
    reasons: List[str] = []; meas = ((res.get("measured") or {}).get("meas")) or []; out = res.get("outcome") or {}
    if res.get("error"): return False, [f"ERROR:{res['error'].splitlines()[0][:120]}"]
    if res.get("sequence"):                                   # multi-step Plan: the sequence search replays at ONE timestep (no two-dt measure in the runner today)
        v = out.get("validity") or {}; ok_dt = not v.get("reasons") and float(v.get("violation", 0.0) or 0.0) <= 0
        reasons += [f"dt1:{r}" for r in (v.get("reasons") or [])] + ([] if out.get("achieved") else ["NOT_ACHIEVED"]) + [f"MISSING:{m.get('goal')}" for m in res.get("missing") or []]
        return bool(ok_dt and out.get("achieved")), sorted(set(reasons + ["SINGLE_DT_ONLY(sequence)"]))
    if not meas: reasons.append("NO_TIMESTEP_MEASURE")
    for m in meas:
        bad = (not m.get("valid")) or float(m.get("violation", 0.0) or 0.0) > 0
        if bad: reasons += [f"dt{m.get('dt')}:{r}" for r in (m.get("reasons") or ["INVALID"])]
    v = out.get("validity") or {}
    if v.get("reasons"): reasons += [f"eval:{r}" for r in v["reasons"] if f"dt{meas[0].get('dt') if meas else ''}:{r}" not in reasons]
    if not out.get("achieved"): reasons.append("NOT_ACHIEVED")
    for m in res.get("missing") or []: reasons.append(f"MISSING:{m.get('goal')}")          # recorded, not part of the pass rule (partial plan is reported as such)
    passed = bool(meas) and all(m.get("valid") and float(m.get("violation", 0.0) or 0.0) <= 0 for m in meas) and bool(out.get("achieved"))
    return passed, sorted(set(reasons))
