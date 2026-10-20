"""Representation base: a construction representation exposes the engine.task.Task interface (plant / n_params / unpack /
controller / rollout / objective / floor / warm / sigma0 / score / run) by delegating to a wrapped reference task (EffectTask,
ImpactTask, ReachTask, JumpTask, StandTask, ...), so engine.task.optimize and engine.task.measure (robust two-timestep) run on it
unchanged. Duck-typed (no MuJoCo at import). Subclasses change only the parameter meaning; a task the subclass does not know
natively is delegated to as-is (its own n_params / unpack / controller / rollout). Demand keys handled here for every task:
    effector in ("pelvis", "com", "base")  -> a plain EffectTask is swapped for construction.base_effect.BaseEffectTask
    initial_state=(qpos, qvel)             -> the rollout starts from that state (engine.core.handoff, hold torques recomputed)
    start_z=<m>                            -> the plant is re-settled with its feet on raised terrain (physics.settle)
Nothing here prescribes a movement: a representation is a space of controllers; a prior is an option inside it."""
from __future__ import annotations
from typing import Any, Dict, Optional
import contextlib, functools
import numpy as np


class Representation:
    name = "representation"; label = "representation"
    def __init__(self, task, horizon: Optional[float] = None):
        self.task = task; self.robot = getattr(task, "robot", "G1"); self.horizon = horizon; self._refs: Dict[Any, Any] = {}
    # ---- task binding
    def _t(self, demand):
        """The reference task for this demand: swaps a plain EffectTask for BaseEffectTask when the effect is on the base."""
        if demand is not None and str(demand.get("effector")) in ("pelvis", "com", "base") and type(self.task).__name__ == "EffectTask":
            from .base_effect import BaseEffectTask
            self.task = BaseEffectTask(self.robot)
        return self.task
    def native(self) -> bool:
        """True when the wrapped task is one whose parameter layout this representation re-interprets (EffectTask family)."""
        return hasattr(self.task, "one_foot_hold")
    # ---- engine.task.Task interface, delegated
    def demand_default(self): return self.task.demand_default()
    def plant(self, demand, dt=None, fresh=False):
        from engine import core
        S = self._t(demand).plant(demand, core.DT_OPT if dt is None else dt, fresh=fresh); dz = float(demand.get("start_z", 0.0) or 0.0)
        if dz > 0 and S.get("start_z") != dz:
            from movement_discovery.physics.settle import settle_on_terrain
            settle_on_terrain(S, dz)
        return S
    def floor(self, demand): return self._t(demand).floor(demand)
    def objective(self, rec, demand): return self._t(demand).objective(rec, demand)
    def sigma0(self, demand, warm): return self._t(demand).sigma0(demand, warm)
    def n_params(self, S, demand): return self._t(demand).n_params(S, demand)
    def unpack(self, S, demand, z): return self._t(demand).unpack(S, demand, z)
    def controller(self, S, demand, p): return self._t(demand).controller(S, demand, p)
    def rollout(self, S, demand, p, ctrl_fn=None, record=False):
        with self.start_ctx(S, demand), self.horizon_ctx(): return self._t(demand).rollout(S, demand, p, ctrl_fn=ctrl_fn, record=record)
    def warm(self, S, demand, stored): return self._t(demand).warm(S, demand, stored)
    def score(self, rec, demand):
        """Lexicographic (engine.task.Task.score verbatim): any violation ranks below every valid solution."""
        if rec["violation"] > 0 or not rec["valid"]: return 100.0 + max(rec["violation"], 1e-3) - 0.01*rec["value"]
        return self.objective(rec, demand)
    def run(self, S, demand, z, record=False):
        p = self.unpack(S, demand, z); fn = self.controller(S, demand, p)
        rec = self.rollout(S, demand, p, ctrl_fn=fn, record=record); rec["dt"] = float(S["m"].opt.timestep); rec["params"] = p
        return rec
    # ---- representation extras
    def z0(self, S, demand) -> np.ndarray:
        """Prior-only start point: zeros, except a trailing tscale parameter (0.87 + 0.3 z convention of JumpTask/ReachTask) set to 1.0."""
        z = np.zeros(self.n_params(S, demand))
        try:
            p0 = self.unpack(S, demand, z)
            if p0.get("tscale") is not None and abs(float(p0["tscale"]) - 0.87) < 1e-9 and len(z): z[-1] = (1.0 - 0.87)/0.3
        except Exception: pass
        return z
    def describe(self) -> dict:
        return dict(representation=self.name, task=getattr(self.task, "name", type(self.task).__name__), robot=self.robot, horizon=self.horizon, native=self.native())
    def nk(self) -> int:
        """Knot count for the (possibly variable) horizon: throw.NK scaled by horizon / throw.HOR (knot spacing preserved)."""
        import throw as T
        return int(T.NK) if self.horizon is None else max(2, int(round(T.NK*float(self.horizon)/T.HOR)))
    def horizon_ctx(self): return bind_horizon(self.horizon if self.native() else None)
    def start_ctx(self, S, demand):
        """demand["initial_state"] = (qpos, qvel): the rollout starts there (hold torques recomputed for the handed posture)."""
        ist = demand.get("initial_state") if demand else None
        if ist is None: return contextlib.nullcontext()
        from engine import core
        return core.handoff(S, np.asarray(ist[0], float), np.asarray(ist[1], float), hold=demand.get("hold_mode", "keep"))


@contextlib.contextmanager
def bind_horizon(H: Optional[float]):
    """VARIABLE HORIZON. The frozen ThrowTask/EffectTask/ImpactTask rollouts call throw.rollout without a horizon argument; this
    context binds horizon= on that one module attribute for the duration of the call (process-local; the rollout reads the knot
    count from x itself, so the representation must size its knots). No edit to throw.py or tasks/*."""
    if H is None: yield; return
    import throw as T
    orig = T.rollout
    if isinstance(orig, functools.partial) and orig.keywords.get("horizon") == float(H): yield; return
    T.rollout = functools.partial(orig, horizon=float(H))
    try: yield
    finally: T.rollout = orig
