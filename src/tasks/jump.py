"""Vertical jump as an engine Task. Demand: {"land": bool, "prior": bool}. value = takeoff speed (m/s).
Prior = the analytic pose-bank deep crouch (Phases 9-14); controller = tracker + the Phase 14 landing controller."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, branch_i as BI, mocap_baseline as MB, jump13 as J13
from engine import core
from engine.task import Task


class JumpTask(Task):
    name = "jump"
    def __init__(self, robot="G1"): self.robot = robot; self._clip = None; self._ref = None; self._om = {}
    def demand_default(self): return dict(land=True, prior=True)
    def clip(self):
        if self._clip is None: self._clip = core.standard_jump_clip(self.robot)
        return self._clip
    def ref(self, demand):
        """prior: False/None = native (no reference); True/'bank' = analytic pose-bank crouch (Phases 9-14);
        'ascent' = the unified capacity-weighted ascent generator (engine.strategy.jump_prior)."""
        pr = demand.get("prior", True)
        if not pr: return self.clip()
        if pr == "ascent":
            if not hasattr(self, "_ref_ascent"):
                from engine import strategy as ST, core as C
                S = C.plant(self.robot, start=self.clip()["start"]); self._ref_ascent = dict(ST.jump_prior(S), key=self.clip()["key"] + "|ascent_prior")
            return self._ref_ascent
        if self._ref is None: self._ref = MB.analytic_clip(self.clip(), robot=self.robot)
        return self._ref
    def plant(self, demand, dt=core.DT_OPT, fresh=False):
        c = self.clip(); S = BI._build(np.asarray(c["start"], float), {}, self.robot) if fresh else core.plant(self.robot, start=c["start"])
        BI.apply_dt(S, dt); BI.settle(S); return S
    def n_params(self, S, demand): return len(S["types"])*self.ref(demand)["nk"] + (1 if demand.get("prior", True) else 0)
    def unpack(self, S, demand, z):
        z = np.asarray(z); nT = len(S["types"]); nk = self.ref(demand)["nk"]
        return dict(x=z[:nT*nk], tscale=(0.87 + 0.3*float(z[nT*nk])) if demand.get("prior", True) else 1.0)
    def omega(self, S, demand):
        key = (id(S), demand.get("prior", True))
        if key not in self._om: self._om[key] = MB.tune_gains(S, self.ref(demand), 1.0)[0] if demand.get("prior", True) else 0.0
        return self._om[key]
    def controller(self, S, demand, p):
        om = self.omega(S, demand)
        return MB.Tracker(S, self.ref(demand), omega=om, tscale=p["tscale"]) if om > 0 else None
    def rollout(self, S, demand, p, ctrl_fn=None, record=False):
        land = demand.get("land", True)
        ctl = core.landing_controller(S, self.robot)[0] if land else None; land_log = []
        if ctl is not None:
            _orig = ctl.targets
            def _tg(t, d, c_, _o=_orig): land_log.append((t, d.qpos.copy(), d.qvel.copy())); return _o(t, d, c_)
            ctl.targets = _tg
        o = BI.rollout(np.asarray(p["x"]), set(S["types"]), None, S=S, ctrl_fn=ctrl_fn, horizon=J13.H_LAND if land else self.ref(demand)["horizon"],
                       score="best", knot_dt=J13.KNOT_DT, landing=(dict(controller=ctl) if land else None), record=record)
        sc = J13.score(o); v = o["v"]
        viol = max(0.0, sc + 0.01*v - 100.0) if sc >= 0 else 0.0
        reasons = [z for z in o["validity"]["reasons"] if z != "CONTACT_ENERGY"]
        if "OVERSPEED" in reasons: viol += 1.0 + 10.0*max(0.0, o.get("overspeed", 0.0) - BI.OVERSPEED_TOL)
        if land and not (o.get("landing") or {}).get("ok"): reasons.append("NOT_LANDED")
        cost = float(o.get("total_work", 0.0))
        return dict(value=float(v), valid=(viol <= 0), reasons=reasons, violation=float(viol), cost=cost,
                    extra=dict(crouch=o["mech"]["crouch_depth"], rot=o["mech"]["flight_rotation"], T_push=o["mech"]["T_push"], landing=o.get("landing"), omega=self.omega(S, demand), tscale=p["tscale"],
                               final_state=((land_log[-1][1], land_log[-1][2]) if land_log else None), land_trace=[(t, q) for t, q, _ in land_log[::10]]))
    def floor(self, demand): return 2.0 if demand.get("land", True) else None
    def warm(self, S, demand, stored):
        x = np.asarray(stored["x"]); z = x if not demand.get("prior", True) else np.concatenate([x, [(stored["tscale"] - 0.87)/0.3]])
        return z
