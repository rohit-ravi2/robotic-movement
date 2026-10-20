"""ServoKnots: the quasi-static servo construction of tasks/reach.py as a representation. The balance controller owns the legs
(capture point over the feet, trunk vertical); arm + trunk are torque-limited position servos tracking a reference (ascent /
reach prior, or a brace prior supplied as demand["ref"]) plus knot OFFSETS (rad, +-0.3); z = [knots nT x NK | tscale].
For reaching / precision / sustained-contact tasks. Horizon through demand["horizon"] (ReachTask reads it natively). Any other
servo-mode task (e.g. the balance family's StandTask) is delegated to as-is."""
from __future__ import annotations
from typing import Optional
import numpy as np
from .base import Representation


class ServoKnots(Representation):
    name = "servo_knots"; label = "servo targets (balance-controlled legs) + knot offsets"
    def __init__(self, task, horizon: Optional[float] = None, prior: str = "ascent"):
        super().__init__(task, None); self.prior = prior; self.H = horizon
    def native(self) -> bool: return hasattr(self.task, "target_world")          # tasks/reach.py ReachTask (and subclasses)
    def _d(self, demand):
        if not self.native(): return demand
        d = dict(demand)
        if "prior" not in d: d["prior"] = self.prior
        if self.H is not None: d["horizon"] = float(self.H)
        return d
    def plant(self, demand, dt=None, fresh=False): return super().plant(self._d(demand), dt, fresh)
    def n_params(self, S, demand): return self.task.n_params(S, self._d(demand))
    def unpack(self, S, demand, z): return self.task.unpack(S, self._d(demand), z)
    def controller(self, S, demand, p): return self.task.controller(S, self._d(demand), p)
    def brace_ref(self, S, demand):
        from .brace import servo_brace_ref
        d = self._d(demand); key = ("brace", id(S["m"]), np.round(np.asarray(S["q_settled"]), 3).tobytes(), str(d.get("target_world")), str(d.get("target_object")), str(d.get("push")), str(d.get("force_direction")))
        if key not in self._refs: self._refs[key] = servo_brace_ref(self.task, S, d)
        return self._refs[key]
    def rollout(self, S, demand, p, ctrl_fn=None, record=False):
        d = self._d(demand); ext = d.get("ref")
        if ext is None and self.native() and d.get("prior") == "brace": ext = d["ref"] = self.brace_ref(S, d); d["prior"] = "ascent"   # ReachTask: a truthy prior reads the pinned ref
        if ext is not None and self.native():     # an externally built reference (e.g. a brace prior) pinned into the task's cache
            if not hasattr(self.task, "_refs"): self.task._refs = {}
            self.task._refs[self.task._ref_key(S, d)] = ext
        return super().rollout(S, d, p, ctrl_fn=ctrl_fn, record=record)
    def objective(self, rec, demand): return self.task.objective(rec, self._d(demand))
    def floor(self, demand): return self.task.floor(self._d(demand))
    def sigma0(self, demand, warm): return self.task.sigma0(self._d(demand), warm)
    def z0(self, S, demand):
        z = np.zeros(self.n_params(S, demand))
        if self.native() and self._d(demand).get("prior"): z[-1] = (1.0 - 0.87)/0.3          # tscale 1.0
        return z
    def warm(self, S, demand, stored): return self.task.warm(S, self._d(demand), stored)
    def describe(self): return dict(super().describe(), prior=self.prior, horizon=self.H, params=("knot offsets nT x 10 (rad, +-0.3) + tscale; legs by the balance controller" if self.native() else "delegated to the task"))
