"""Torque-knot representations over the reference construction (tasks/effect.py, tasks/impact.py, base_effect), semantics kept:
  TorqueKnotsTrackedPrior: z = [knots (nT x nk) | release | amp | tscale]; controller = body-derived reference (the task's ref())
      tracked at the searched amplitude and time scale; knots are torque OFFSETS on the gravity hold; amp 0.2..1.0.
  TorqueKnots: the same without the prior (demand["prior"] = None): z = [knots | release].
Variable horizon: horizon=H scales the knots (Representation.nk) and maps the release onto [T_REL_MIN, H]; the reference holds its
last pose to H. ANY OTHER engine.task.Task (JumpTask, StandTask, ...) is delegated to as-is: its own n_params / unpack /
controller / rollout, the prior flag passed through (TorqueKnots sets prior=None), z0 = zeros with tscale 1.0."""
from __future__ import annotations
from typing import Optional
import numpy as np
from .base import Representation


class TorqueKnotsTrackedPrior(Representation):
    name = "torque_knots+tracked_prior"; label = "torque knots + tracked body-derived prior"
    def __init__(self, task, horizon: Optional[float] = None, prior: str = "windup"):
        super().__init__(task, horizon); self.prior = prior
    def _d(self, demand):
        self._t(demand)
        if not self.native(): return demand if demand.get("prior") else dict(demand, prior=demand.get("prior", True))
        pr = "brace" if (self.prior == "brace" or demand.get("prior") == "brace") else self.prior
        return dict(demand, prior=pr) if demand.get("prior") != pr else demand
    def is_brace(self, demand) -> bool: return self.prior == "brace" or demand.get("prior") == "brace"
    @staticmethod
    def _key(S, d): dr = d.get("direction"); return (id(S["m"]), None if dr is None else tuple(np.round(dr, 4)), d.get("capacity_mode", "power"))
    def brace_ref(self, S, demand):
        """The brace -> sustained-push reference (construction.brace), cached per plant/direction, PINNED into the task's ref cache."""
        import throw as T
        from .brace import torque_brace_ref
        d = self._d(demand); key = self._key(S, d); ck = ("brace",) + key
        if ck not in self._refs: self._refs[ck] = torque_brace_ref(self.task, S, d, float(self.horizon or T.HOR))
        self.task._refs[key] = self._refs[ck]; return self._refs[ck]
    def ref(self, S, demand):
        """The reference this representation tracks: brace (pinned) or the task's own (windup / base ascent), evicting a stale brace one."""
        d = self._d(demand)
        if self.is_brace(d): return self.brace_ref(S, d)
        key = self._key(S, d); r = self.task._refs.get(key)
        if r is not None and r.get("key") == "brace_prior": self.task._refs.pop(key)
        return self.task.ref(S, d)
    def n_params(self, S, demand):
        if not self.native(): return self.task.n_params(S, self._d(demand))
        return len(S["types"])*self.nk() + 1 + 2
    def unpack(self, S, demand, z):
        if not self.native(): return self.task.unpack(S, self._d(demand), z)
        z = np.asarray(z, float); nk = self.nk(); n = len(S["types"])*nk + 1; p = dict(x=z[:n])
        lo_, hi_ = demand.get("tscale_range", (0.57, 1.17))
        p["amp"] = 0.2 + 0.8*(float(z[n]) + 1)/2; p["tscale"] = lo_ + (hi_ - lo_)*(float(z[n+1]) + 1)/2; p["omega"] = float(demand.get("omega", 20.0)); p["nk"] = nk
        return p
    def controller(self, S, demand, p):
        d = self._d(demand)
        if not self.native(): return self.task.controller(S, d, p)
        ref = self.ref(S, d)                           # pins a brace reference; a longer horizon holds the final pose to H (Tracker clamps past tend)
        if self.horizon is not None and ref.get("horizon") != float(self.horizon): ref["horizon"] = float(self.horizon)
        fn = self.task.controller(S, d, p)
        if self.is_brace(d) and ref.get("a_ff") is not None:
            from .brace import push_feedforward
            fn = push_feedforward(fn, ref, p, len(S["types"]))
        return fn
    def rollout(self, S, demand, p, ctrl_fn=None, record=False): return super().rollout(S, self._d(demand), p, ctrl_fn=ctrl_fn, record=record)
    def z0(self, S, demand):
        """Native: release at the reference's own release time, amplitude mid (0.6), tscale mid-range (engine.construct); else generic."""
        if not self.native(): return super().z0(S, self._d(demand))
        import throw as T
        d = self._d(demand); ref = self.ref(S, d); H = float(self.horizon or T.HOR); z = np.zeros(self.n_params(S, demand)); n = len(S["types"])*self.nk()
        z[n] = float(np.clip(2*(ref["takeoff_ref"] - T.T_REL_MIN)/(H - T.T_REL_MIN) - 1, -1, 1))
        if self.is_brace(d): z[n + 1] = 1.0            # a brace is a static configuration: full amplitude (a scaled brace leaves the contact point)
        return z
    def warm(self, S, demand, stored):
        if not self.native(): return self.task.warm(S, self._d(demand), stored)
        z = list(np.asarray(stored["x"])); lo_, hi_ = demand.get("tscale_range", (0.57, 1.17))
        return np.array(z + [2*(stored["amp"] - 0.2)/0.8 - 1, 2*(stored["tscale"] - lo_)/(hi_ - lo_) - 1])
    def describe(self):
        return dict(super().describe(), prior=self.prior, params=("knots nT x nk (torque offsets on the gravity hold) + release + amp[0.2,1] + tscale" if self.native() else "delegated to the task"), nk=self.nk())


class TorqueKnots(Representation):
    name = "torque_knots"; label = "torque knots, no prior"
    def _d(self, demand): self._t(demand); return dict(demand, prior=None)
    def n_params(self, S, demand): return len(S["types"])*self.nk() + 1 if self.native() else self.task.n_params(S, self._d(demand))
    def unpack(self, S, demand, z):
        if not self.native(): return self.task.unpack(S, self._d(demand), z)
        z = np.asarray(z, float); return dict(x=z[:len(S["types"])*self.nk() + 1], nk=self.nk())
    def controller(self, S, demand, p): return self.task.controller(S, self._d(demand), p)      # None: EffectTask pops hold_sched
    def rollout(self, S, demand, p, ctrl_fn=None, record=False): return super().rollout(S, self._d(demand), p, ctrl_fn=ctrl_fn, record=record)
    def sigma0(self, demand, warm): return 0.35 if warm is None else 0.10
    def z0(self, S, demand):
        if not self.native(): return super().z0(S, self._d(demand))
        import throw as T
        z = np.zeros(self.n_params(S, demand)); z[-1] = float(np.clip(2*(0.55 - T.T_REL_MIN)/(float(self.horizon or T.HOR) - T.T_REL_MIN) - 1, -1, 1)); return z
    def warm(self, S, demand, stored): return np.asarray(stored["x"], float) if self.native() else self.task.warm(S, self._d(demand), stored)
    def describe(self): return dict(super().describe(), prior=None, params=("knots nT x nk + release" if self.native() else "delegated to the task"), nk=self.nk())
