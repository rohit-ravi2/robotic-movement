"""EFFECT task: the g1t_3 construction for an arbitrary velocity effect (construct, 2026-10-11).
Demand: effector (site name), mass (kg merged at the effector), support ("both"|"left"|"right"), direction (elev, az) in rad
with elev_tol/az_tol, target_speed, prior ("windup" | None), capacity_mode, omega, recoverable, driven (joint list, default
the robot's throw set). value = effector speed at release; the wind-up -> release reference is computed from the body for the
requested direction and tracked at a searched amplitude; torque knots are refined on the true dynamics; the gates are the
throw's (stance of the support feet, cone, actuator envelope, two timesteps)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, numpy as np, mujoco, throw as T, branch_i as BI
from engine import core
from tasks.throw import ThrowTask


class EffectTask(ThrowTask):
    name = "effect"
    def __init__(self, robot="G1"): super().__init__(robot); self._refs = {}
    def demand_default(self):
        return dict(effector=T.CFG[self.robot]["hand_site"], mass=T.BALL, support="both", direction=None, elev_tol=np.radians(5.0), az_tol=np.radians(5.0),
                    target_speed=None, prior="windup", capacity_mode="power", omega=20.0, recoverable=False, driven=None, rom_scale=1.0)
    def plant(self, demand, dt=core.DT_OPT, fresh=False):
        jr = demand.get("jnt_ranges"); key = (dt, fresh, demand.get("effector"), float(demand.get("mass", 0.0)), demand.get("support", "both"), tuple(demand.get("driven") or ()), json.dumps(jr, sort_keys=True) if jr else None)
        if key in self._plants: return self._plants[key]
        kw = dict(site=demand.get("effector"), ball=False, mass=float(demand.get("mass", 0.0)), support=demand.get("support", "both"), driven=demand.get("driven"))
        S0 = T.setup(self.robot, **kw); S = T.setup(self.robot, start=np.array(S0["q0"]) + 1e-9*(1 + len(self._plants)), **kw) if (fresh or jr) else S0   # an uncached copy when ranges are modified (counterfactuals)
        if jr:   # COUNTERFACTUAL joint ranges (e.g. a human shoulder): applied to the model; the prior, the search and the gates all see them
            m = S["m"]
            for jn, (lo, hi) in jr.items():
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn); assert j >= 0, jn; m.jnt_range[j] = [float(lo), float(hi)]
        BI.apply_dt(S, dt); BI.settle(S); S["_setup_kw"] = kw; self._plants[key] = S; return S
    def caps(self, S, demand):
        """COUNTERFACTUAL torque capacity: demand["tau_scale"] = {group or joint name: factor}; the search, the gates and the prior's capacity weights see it."""
        ts = demand.get("tau_scale")
        if not ts: return None
        return np.array([float(ts.get(S["groups"][i], ts.get(t, 1.0))) for i, t in enumerate(S["types"])])
    def rollout(self, S, demand, p, ctrl_fn=None, record=False):
        c = self.caps(S, demand)
        if c is None: return super().rollout(S, demand, p, ctrl_fn=ctrl_fn, record=record)
        J = S["jarr"]; tau0 = J["tau"].copy(); J["tau"] = tau0*c           # the prior's weights and the envelope see the scaled capacity
        try: return super().rollout(S, dict(demand, _caps=c.tolist()), p, ctrl_fn=ctrl_fn, record=record)
        finally: J["tau"] = tau0
    def ref(self, S, demand):
        """Body-derived wind-up -> release reference for the requested direction (cached per plant and direction)."""
        dr = demand.get("direction"); key = (id(S["m"]), None if dr is None else tuple(np.round(dr, 4)), demand.get("capacity_mode", "power"))
        if key not in self._refs:
            self._refs[key] = T.windup_reference(S, direction=(None if dr is None else (float(dr[0]), float(dr[1]))), capacity_mode=demand.get("capacity_mode", "power"))   # (elev, az) rad
        return self._refs[key]
    def one_foot_hold(self, S, demand, ref):
        """Holding torques for the PREPARED one-foot stance (position-servo settle at the prepared pose), as activation fractions."""
        key = ("hold", id(S["m"]), tuple(np.round(ref["prep_qpos"], 4)))
        if key not in self._refs:
            kw = S.get("_setup_kw") or dict(site=demand.get("effector"), ball=False, mass=float(demand.get("mass", 0.0)), support=demand.get("support", "both"), driven=demand.get("driven"))
            Sp = T.setup(self.robot, start=np.asarray(ref["prep_qpos"], float), **kw); BI.apply_dt(Sp, S["m"].opt.timestep); BI.settle(Sp)   # same scene (world plants carry objects: nq differs)
            self._refs[key] = np.array([Sp["a_hold"][t] for t in S["types"]])
        return self._refs[key]
    def controller(self, S, demand, p):
        if not demand.get("prior"): S.pop("hold_sched", None); return None
        ref = self.ref(S, demand)
        if ref.get("prep_qpos") is not None:
            a0 = np.array([S["a_hold"][t] for t in S["types"]]); a1 = self.one_foot_hold(S, demand, ref); t0, t1 = 0.05*p["tscale"], ref["t_prep_end"]*p["tscale"]
            S["hold_sched"] = lambda tt, a0=a0, a1=a1, t0=t0, t1=t1: a0 + (a1 - a0)*float(np.clip((tt - t0)/max(t1 - t0, 1e-6), 0.0, 1.0))
        else: S.pop("hold_sched", None)
        return T.tracker(S, T.scaled_ref(ref, p["amp"]), p["omega"], p["tscale"])
    def n_params(self, S, demand): return len(S["types"])*T.NK + 1 + (2 if demand.get("prior") else 0)
    def unpack(self, S, demand, z):
        z = np.asarray(z); n = len(S["types"])*T.NK + 1; p = dict(x=z[:n])
        lo_, hi_ = demand.get("tscale_range", (0.57, 1.17))      # the throw's range; a leg effector needs a slower reference (torque-limited swing)
        if demand.get("prior"): p["amp"] = 0.2 + 0.8*(float(z[n]) + 1)/2; p["tscale"] = lo_ + (hi_ - lo_)*(float(z[n+1]) + 1)/2; p["omega"] = float(demand.get("omega", 20.0))
        return p
    def warm(self, S, demand, stored):
        z = list(np.asarray(stored["x"])); lo_, hi_ = demand.get("tscale_range", (0.57, 1.17))
        if demand.get("prior"): z += [2*(stored["amp"] - 0.2)/0.8 - 1, 2*(stored["tscale"] - lo_)/(hi_ - lo_) - 1]
        return np.array(z)
