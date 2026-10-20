"""Optimizer interface and budget tiers. Every optimizer offers ask()/tell() and run(objective, n, z0, bounds, budget) over
z in [-1, 1]^n with a scalar objective (lower is better; engine.task's lexicographic score). TaskObjective adapts a
Representation + demand to that scalar with the robust two-timestep measure of engine.task (worse of 1 ms and 0.5 ms).
    get("cmaes" | "random" | "sobol" | "cross_entropy")(seed=..., **cfg) -> Optimizer
Budget tiers (evaluations): SCREEN = 2,400 (feasibility), EMERGENCE = 38,400 (the g1t_3 budget). No campaign runs here."""
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type
import time
import numpy as np

SCREEN = 2400; EMERGENCE = 38400
TIERS = dict(screen=SCREEN, emergence=EMERGENCE)


class Optimizer:
    name = "optimizer"
    def __init__(self, seed: int = 0, pop: int = 16, **cfg): self.seed = int(seed); self.pop = int(pop); self.cfg = dict(cfg); self.n_evals = 0; self.best: Tuple[float, Optional[np.ndarray]] = (np.inf, None); self.history: List[dict] = []
    def start(self, n: int, z0: np.ndarray, bounds=(-1.0, 1.0), sigma: float = 0.1) -> None: raise NotImplementedError
    def ask(self) -> List[np.ndarray]: raise NotImplementedError
    def tell(self, X: Sequence[np.ndarray], F: Sequence[float]) -> None: self._note(X, F)
    def _note(self, X, F) -> None:
        """Bookkeeping only (best, history, eval count); tell() = strategy update + _note()."""
        k = int(np.argmin(F)); self.n_evals += len(F)
        if F[k] < self.best[0]: self.best = (float(F[k]), np.array(X[k]))
        self.history.append(dict(evals=self.n_evals, best=self.best[0], gen_best=float(F[k])))
    def stop(self) -> bool: return False
    def run(self, objective: Callable[[np.ndarray], float], n: int, z0: Optional[np.ndarray] = None, bounds=(-1.0, 1.0), budget: int = SCREEN, sigma: float = 0.1, log: Optional[Callable] = None) -> dict:
        z0 = np.zeros(n) if z0 is None else np.asarray(z0, float); self.start(n, z0, bounds, sigma); t0 = time.time()
        f0 = float(objective(z0)); self._note([z0], [f0])                     # the start point is always a candidate (engine.task.optimize)
        while self.n_evals < budget and not self.stop():
            X = self.ask(); F = [float(objective(z)) for z in X]; self.tell(X, F)
            if log: log(self.n_evals, self.best[0])
        return dict(z=self.best[1].tolist(), score=self.best[0], evals=self.n_evals, secs=time.time() - t0, optimizer=self.name, config=dict(self.cfg, seed=self.seed, pop=self.pop), history=self.history)


class TaskObjective:
    """Representation + demand -> scalar. robust=True: engine.task.robust_score over the 1 ms and 0.5 ms plants."""
    def __init__(self, rep, demand: dict, robust: bool = True):
        from engine import core
        self.rep, self.demand, self.robust = rep, dict(demand), robust
        self.S1 = rep.plant(demand, core.DT_OPT); self.S2 = rep.plant(demand, core.DT_CHECK, fresh=True) if robust else None; self.n = rep.n_params(self.S1, demand); self.n_evals = 0; self.best = (np.inf, None, None)
    def __call__(self, z: np.ndarray) -> float:
        from engine.task import robust_score
        r1 = self.rep.run(self.S1, self.demand, z); r2 = self.rep.run(self.S2, self.demand, z) if self.robust else None; s = robust_score(self.rep, self.demand, r1, r2); self.n_evals += 1
        if s < self.best[0]: self.best = (s, np.array(z), r1)
        return float(s)
    def z0(self) -> np.ndarray: return self.rep.z0(self.S1, self.demand)
    def measure(self, z) -> dict:
        from engine.task import measure
        return measure(self.rep, self.demand, z)


_REG: Dict[str, Type[Optimizer]] = {}


def register(cls: Type[Optimizer]) -> Type[Optimizer]: _REG[cls.name] = cls; return cls


def get(name: str) -> Type[Optimizer]:
    if not _REG: from . import cmaes, random_search, cross_entropy   # noqa
    return _REG[name]


def names() -> List[str]:
    get("cmaes"); return sorted(_REG)
