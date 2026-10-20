"""Random and quasi-random (Sobol, scrambled) search in the box around z0: samples z0 + sigma_box * u with u uniform in
[-1, 1]^n (sigma_box = cfg "width", default 1.0 = the whole box), clipped to bounds. Baselines and feasibility screens."""
from __future__ import annotations
import numpy as np
from . import Optimizer, register


@register
class RandomSearch(Optimizer):
    name = "random"
    def start(self, n, z0, bounds=(-1.0, 1.0), sigma=0.1):
        self.n, self.z0, self.lo, self.hi = int(n), np.asarray(z0, float), float(bounds[0]), float(bounds[1]); self.width = float(self.cfg.get("width", 1.0)); self.rng = np.random.default_rng(self.seed)
    def _u(self, k): return self.rng.uniform(-1.0, 1.0, (k, self.n))
    def ask(self): return list(np.clip(self.z0 + self.width*self._u(self.pop), self.lo, self.hi))


@register
class SobolSearch(RandomSearch):
    name = "sobol"
    def start(self, n, z0, bounds=(-1.0, 1.0), sigma=0.1):
        super().start(n, z0, bounds, sigma)
        from scipy.stats import qmc
        self.q = qmc.Sobol(d=self.n, scramble=True, seed=self.seed)
    def _u(self, k): return 2.0*self.q.random(k) - 1.0
