"""CMA-ES adapter (pycma), the reference search of engine.task.optimize: bounds [-1, 1], sigma0 from the representation
(0.10 with a prior / warm start, 0.35 without), popsize 16, start point always a candidate. optimize_task() delegates to
engine.task.optimize verbatim (robust two-timestep measure, faithful replay) for exact parity with engine/construct.py."""
from __future__ import annotations
from typing import Optional
import numpy as np
from . import Optimizer, register, SCREEN


@register
class CMAES(Optimizer):
    name = "cmaes"
    def start(self, n, z0, bounds=(-1.0, 1.0), sigma=0.1):
        import cma
        self.es = cma.CMAEvolutionStrategy(np.asarray(z0, float), float(sigma), {"bounds": list(bounds), "seed": self.seed, "verbose": -9, "popsize": self.pop, "maxiter": int(self.cfg.get("maxiter", 10**9))})
    def ask(self): return self.es.ask()
    def tell(self, X, F): self.es.tell(list(X), list(F)); self._note(X, F)
    def stop(self): return bool(self.es.stop())


def optimize_task(rep, demand: dict, seed: int, budget: int = SCREEN, pop: int = 16, z0: Optional[np.ndarray] = None, sigma: Optional[float] = None, robust: bool = True, warm=None, log=None) -> dict:
    """engine.task.optimize on a Representation: iters = budget // pop; returns its measured dict (z, value, ok_all, meas, params)."""
    from engine import task as TK
    iters = max(1, int(budget)//int(pop))
    if z0 is None and warm is None:
        from engine import core
        z0 = rep.z0(rep.plant(demand, core.DT_OPT), demand)
    out = TK.optimize(rep, demand, seed=seed, iters=iters, pop=pop, warm=warm, sigma=sigma, robust=robust, log=log, z0=z0)
    out.update(optimizer="cmaes", budget=iters*pop); return out
