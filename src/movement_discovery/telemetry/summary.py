"""Mechanics summary: engine.emergence.analyze (chain peaks, release ownership, excursions vs limits and human ROM, positive work
by group, lag signature) with the effector resolved from the demand, guarded so a missing trace or a non-chain effector returns
None instead of raising. Descriptive, never a filter."""
from __future__ import annotations
from typing import Optional


def summarize(S, rec: dict, effector: Optional[str] = None) -> Optional[dict]:
    if not rec.get("trace") or not rec["trace"].get("q"): return None
    try:
        from engine import emergence as EM
        import mujoco
        site = effector or mujoco.mj_id2name(S["m"], mujoco.mjtObj.mjOBJ_SITE, int(S["site"]))
        return EM.analyze(S, rec, site)
    except Exception as e:                       # a trace without the chain bodies (servo tasks record q only) is not an error
        return dict(error=f"{type(e).__name__}: {e}")


def one_line(mech: Optional[dict]) -> str:
    if not mech or "error" in (mech or {}): return "mechanics n/a"
    return f"peaks {mech.get('peak_omega')} order {mech.get('peak_order')} | own {mech.get('ownership_group')} base {round(mech.get('ownership_base', 0.0), 2)} | beyond-human {mech.get('beyond_human')} | W {{{', '.join(f'{k} {round(v)}' for k, v in (mech.get('W_pos') or {}).items())}}}"
