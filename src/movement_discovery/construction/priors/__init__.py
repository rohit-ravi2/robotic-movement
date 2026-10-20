"""Preparation / reference OPTIONS derived from the body (never from a demonstration). Each returns a reference dict in the
tracker format (start, Q, fps, takeoff_ref, horizon, nk) a representation may track; none is mandatory and no ordering is
implied. windup: throw.windup_reference (capacity-weighted constrained-Jacobian wind-up -> release for a direction, one-foot
preparation under a one-foot contract). ascent / reach: engine.strategy generic ascent and target-seeking reach. brace: the static
configuration maximising the force the body can apply at a point along a direction (priors/brace.py). direction_field exposes the
raw field u(q) = c * Jc^T d for inspection."""
from __future__ import annotations
from typing import Callable, Optional, Sequence
import numpy as np
from .brace import brace_prior, force_capacity   # noqa  (imported first: the submodule name would otherwise shadow brace() below)


def direction_field(S, dvec: Sequence[float], capacity_mode: str = "power") -> Callable[[np.ndarray], np.ndarray]:
    import throw as T
    J = S["jarr"]; d = np.asarray(dvec, float); d = d/max(np.linalg.norm(d), 1e-12)
    c = J["tau"]*J["om"]/4.0 if capacity_mode == "power" else np.ones(len(J["tau"]))
    def u(q):
        g = c*(T.constrained_jacobian(S, q).T @ d); return g/max(np.linalg.norm(g), 1e-12)
    return u


def windup(S, direction: Optional[Sequence[float]] = None, capacity_mode: str = "power", **kw) -> dict:
    import throw as T
    return T.windup_reference(S, direction=(None if direction is None else (float(direction[0]), float(direction[1]))), capacity_mode=capacity_mode, **kw)


def ascent(S, effector, dvec, horizon: float, nk: int, **kw) -> dict:
    from engine import strategy as ST
    return ST.ascent_prior(S, effector, dvec, horizon, nk, **kw)


def reach(S, target_world, **kw) -> dict:
    from engine import strategy as ST
    return ST.reach_prior(S, np.asarray(target_world, float), **kw)


def brace(S, contact_point, direction, **kw) -> dict: return brace_prior(S, contact_point, direction, **kw)


def catalog() -> dict:
    return dict(windup="wind-up -> release along a direction (velocity effects)", ascent="generic capacity-weighted ascent of a body point", reach="target-seeking reach with balance projection", brace="max static force at a point along a direction under the support contract (sustained-force effects)")
