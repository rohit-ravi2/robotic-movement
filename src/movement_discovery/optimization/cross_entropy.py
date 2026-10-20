"""Cross-entropy method: diagonal Gaussian over z, refit each generation on the elite fraction (cfg "elite", default 0.25),
smoothed (cfg "alpha", default 0.7), sigma floored at cfg "sigma_min" (default 0.01). Clipped to bounds."""
from __future__ import annotations
import numpy as np
from . import Optimizer, register


@register
class CrossEntropy(Optimizer):
    name = "cross_entropy"
    def start(self, n, z0, bounds=(-1.0, 1.0), sigma=0.1):
        self.mu = np.asarray(z0, float).copy(); self.sd = np.full(int(n), float(sigma)); self.lo, self.hi = float(bounds[0]), float(bounds[1]); self.rng = np.random.default_rng(self.seed)
        self.elite = max(2, int(round(float(self.cfg.get("elite", 0.25))*self.pop))); self.alpha = float(self.cfg.get("alpha", 0.7)); self.sd_min = float(self.cfg.get("sigma_min", 0.01)); self._X = None
    def ask(self): self._X = np.clip(self.mu + self.sd*self.rng.standard_normal((self.pop, len(self.mu))), self.lo, self.hi); return list(self._X)
    def tell(self, X, F):
        self._note(X, F); X = np.asarray(X); idx = np.argsort(F)[:self.elite]; E = X[idx]
        self.mu = (1 - self.alpha)*self.mu + self.alpha*E.mean(0); self.sd = np.maximum((1 - self.alpha)*self.sd + self.alpha*E.std(0), self.sd_min)
    def stop(self): return bool(np.all(self.sd <= self.sd_min + 1e-12))
