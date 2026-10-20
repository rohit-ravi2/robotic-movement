"""ContactSchedule: a representation whose parameters include a SUPPORT CONTRACT per phase (both | left | right) and per-phase
horizons, executed as a chain of EffectTask-style torque-knot segments: one plant per phase (support contract baked in, as
throw.setup does), the body-derived reference of each phase built FROM THE HANDED-OFF STATE (one-foot phases get the one-foot
preparation of throw.prepare_support), state handed off through engine.core.handoff (hold torques recomputed for the handed
posture), traces concatenated. Needed for stepping / locomotion. z = per phase [knots nT x nk_k | (amp, tscale if prior)] +
one release parameter for the final phase (earlier phases run their full horizon). No phase sequence is prescribed: the
schedule is an input the program samples (support contracts and horizons), not a technique."""
from __future__ import annotations
from typing import Dict, List, Optional
import contextlib
import numpy as np
from .base import Representation, bind_horizon

DEFAULT_PHASES = [dict(support="both", horizon=0.4, prior="windup"), dict(support="left", horizon=0.5, prior="windup")]


class ContactSchedule(Representation):
    name = "contact_schedule+knots"; label = "per-phase support contract + horizons; chained torque-knot segments"
    def phases(self, demand) -> List[dict]: return [dict(ph) for ph in demand.get("phases", DEFAULT_PHASES)]
    def _nk(self, ph):
        import throw as T
        return max(2, int(round(T.NK*float(ph["horizon"])/T.HOR)))
    SUPPORT_ALIAS = {"any": "both", "feet_any": "both", "hands": "both"}      # plant support contract (feet that must stay loaded); hands-allowed is a validity policy, not a plant
    def _pd(self, demand, ph):
        self._t(demand); d = dict(demand, **{k: v for k, v in ph.items() if k not in ("horizon", "prior", "support", "goal", "initial_posture")}, prior=ph.get("prior"))
        d["support"] = self.SUPPORT_ALIAS.get(str(ph.get("support", "both")), str(ph.get("support", "both"))); return d
    def plants(self, demand, dt=None, fresh=False): return [self.plant(self._pd(demand, ph), dt, fresh) for ph in self.phases(demand)]
    def n_params(self, S, demand):
        nT = len(S["types"]); return sum(nT*self._nk(ph) + (2 if ph.get("prior") else 0) for ph in self.phases(demand)) + 1
    def unpack(self, S, demand, z):
        z = np.asarray(z, float); nT = len(S["types"]); i = 0; out = []
        for k, ph in enumerate(self.phases(demand)):
            nk = self._nk(ph); kn = z[i:i + nT*nk]; i += nT*nk; p = dict(knots=kn.tolist(), nk=nk, support=ph["support"], horizon=float(ph["horizon"]))
            if ph.get("prior"):
                lo_, hi_ = ph.get("tscale_range", demand.get("tscale_range", (0.57, 1.17)))
                p["amp"] = 0.2 + 0.8*(float(z[i]) + 1)/2; p["tscale"] = lo_ + (hi_ - lo_)*(float(z[i+1]) + 1)/2; p["omega"] = float(ph.get("omega", demand.get("omega", 20.0))); i += 2
            out.append(p)
        return dict(x=z.tolist(), release=float(z[i]), phases=out)
    def controller(self, S, demand, p): return None      # built per phase inside rollout (depends on the handed-off state)
    def _world(self, Sk, dk):
        """engine.world.World on the phase plant when the demand carries world objects (demand["world_objects"/"world_surfaces"], set by the
        runner from the WorldSpec): holds use its weld / finger conventions; None on a bare plant."""
        objs = dk.get("world_objects") or []
        if not objs: return None
        if "md_world" not in Sk:
            from engine import world as EW
            Sk["md_world"] = EW.World(Sk["m"], objs, dk.get("world_surfaces") or [], self.robot)
        return Sk["md_world"]
    def _ref(self, task, Sk, dk, k):
        """Reference for phase k from the CURRENT S["q_settled"] (handed-off), pinned into the task's cache under its own key."""
        import throw as T
        dr = dk.get("direction"); key = (id(Sk["m"]), None if dr is None else tuple(np.round(dr, 4)), dk.get("capacity_mode", "power"))
        ck = (k, np.round(np.asarray(Sk["q_settled"]), 3).tobytes(), key[1])
        if ck not in self._refs:                  # the task's own reference builder (windup / bare-robot impact / base ascent) from the current q_settled
            q0 = task._refs.pop(key, None); self._refs[ck] = task.ref(Sk, dk)
            if q0 is not None: task._refs[key] = q0
        task._refs[key] = self._refs[ck]; return self._refs[ck]
    def rollout(self, S, demand, p, ctrl_fn=None, record=False):
        from engine import core
        task = self._t(demand); phs = self.phases(demand); plants = self.plants(demand, S["m"].opt.timestep); recs = []
        state = None if demand.get("initial_state") is None else (np.asarray(demand["initial_state"][0], float), np.asarray(demand["initial_state"][1], float)); traces = []; t_off = 0.0; viol = 0.0; reasons = []; bounds = [0]
        for k, (ph, pk, Sk) in enumerate(zip(phs, p["phases"], plants)):
            dk = self._pd(demand, ph); nT = len(Sk["types"]); last = (k == len(phs) - 1)
            if not last and not ph.get("judge_cone"): dk["ignore_cone"] = True          # the direction demand is on the FINAL effect; earlier phases are support/transport phases
            x = np.concatenate([np.asarray(pk["knots"], float), [p["release"] if last else 1.0]])
            pp = dict(x=x, nk=pk["nk"]); pp.update({kk: pk[kk] for kk in ("amp", "tscale", "omega") if kk in pk})
            if state is not None and k > 0:                   # phase plants may differ (support contract, objects): map the state by joint name
                from .state_map import map_to_plant
                q_, v_, _info = map_to_plant(plants[k - 1], state, Sk); state = (q_, v_)
            ctx = core.handoff(Sk, state[0], state[1], hold=demand.get("hold_mode", "keep")) if state is not None else contextlib.nullcontext()
            wd = self._world(Sk, dk); held = list(dk.get("held") or [])   # keep: the settle's realised hold torques (ground-supported); recompute = free-floating gravity torques (bent postures only)
            with ctx:
                if ph.get("prior"): self._ref(task, Sk, dk, k)
                fn = task.controller(Sk, dk, pp)
                hs = dict(held=[])
                def fn_k(tt, d, S_, fn=fn, t_off=t_off, wd=wd, held=held, hs=hs):
                    if tt <= 0.0 and wd is not None and held:               # carried objects: engage the hold at the phase start (weld, or fingers closed from the start)
                        if wd.fingers: wd.reset_fingers(held=held)
                        for nm in held:
                            if nm in wd.eq and not wd.held(d, nm):
                                if k == 0 and wd.palm_dist(d, nm) > 0.10:      # tasks/reach.py convention: an object declared IN HAND starts at the palm (the plant's settle let it fall); LABEL held_start_abstraction
                                    import mujoco as _mj; m_ = S_["m"]; j_ = _mj.mj_name2id(m_, _mj.mjtObj.mjOBJ_JOINT, nm + "_free"); a_ = m_.jnt_qposadr[j_]
                                    d.qpos[a_:a_ + 3] = d.site_xpos[wd.palm]; d.qpos[a_ + 3:a_ + 7] = [1, 0, 0, 0]; d.qvel[m_.jnt_dofadr[j_]:m_.jnt_dofadr[j_] + 6] = 0; _mj.mj_forward(m_, d); hs["abstraction"] = "held_start_abstraction"
                                wd.grasp(d, nm)
                    if wd is not None and wd.fingers: wd.finger_ctrl(d, tt)
                    if wd is not None and held: hs["held"] = [nm for nm in held if wd.held(d, nm)]      # weld state lives in d.eq_active: read it here, not from a fresh MjData
                    u = fn(tt, d, S_) if fn is not None else np.zeros(nT)
                    return u + (ctrl_fn(t_off + tt, d, S_) if ctrl_fn is not None else 0.0)
                with bind_horizon(float(ph["horizon"])): rec = task.rollout(Sk, dk, pp, ctrl_fn=fn_k, record=record)
            rs = [r for r in rec["reasons"] if not (k > 0 and r == "START_NOT_AT_REST")]       # handoffs are mid-motion by construction
            viol += float(rec["violation"]); reasons += [f"P{k}:{r}" for r in rs]
            state = rec["final_state"]
            if wd is not None and held: rec["extra"]["held_after"] = list(hs["held"]); rec["extra"]["held_note"] = hs.get("abstraction")
            recs.append(dict(phase=k, support=ph["support"], horizon=float(ph["horizon"]), value=float(rec["value"]), reasons=rs, violation=float(rec["violation"]), held_after=rec["extra"].get("held_after"), held_note=(rec["extra"].get("held_note") or (Sk.get("held_start_abstraction") and "held_start_abstraction")), extra={kk: rec["extra"].get(kk) for kk in ("elev", "az", "t_rel", "work", "com_speed", "feet_bw_rel", "overspeed")}))
            if rec.get("trace"): traces.append(rec["trace"]); bounds.append(bounds[-1] + int(rec["trace"]["release_step"]))
            t_off += float(rec["extra"].get("t_rel", ph["horizon"]))
        q0 = np.asarray(demand["initial_state"][0] if demand.get("initial_state") is not None else plants[0]["q_settled"]); qf = np.asarray(state[0]); com_dx = (qf[:3] - q0[:3]).tolist()
        last = recs[-1]; value = last["value"]
        if getattr(task, "name", "") == "base_effect":          # base mode: the effect is the WHOLE schedule's base displacement (projected on the direction if given)
            dv = task.dvec(demand); value = float(np.hypot(com_dx[0], com_dx[1])) if dv is None else float(np.asarray(com_dx) @ dv)
        out = dict(value=value, valid=(viol <= 0 and not reasons), reasons=reasons, violation=viol, cost=float(sum(sum((r["extra"].get("work") or {}).values()) for r in recs)),
                                    final_state=state, extra=dict(phases=recs, base_displacement=com_dx, t_total=t_off, n_phases=len(phs), elev=last["extra"].get("elev"), az=last["extra"].get("az"), t_rel=t_off, work=last["extra"].get("work")))
        if traces: out["trace"] = _concat(traces, bounds)
        return out
    def z0(self, S, demand):
        import throw as T
        z = np.zeros(self.n_params(S, demand)); ph = self.phases(demand)[-1]; H = float(ph["horizon"])
        z[-1] = float(np.clip(2*(0.55*min(1.0, H/T.HOR) - T.T_REL_MIN)/(H - T.T_REL_MIN) - 1, -1, 1)); return z
    def warm(self, S, demand, stored): return np.asarray(stored["x"], float)
    def describe(self, demand=None):
        return dict(super().describe(), base_effect=(getattr(self.task, "name", "") == "base_effect"), params="per phase: knots nT x nk_k (+amp, tscale with prior); final release", phases=(self.phases(demand) if demand else DEFAULT_PHASES))


def _concat(traces: List[dict], bounds: List[int]) -> dict:
    keys = ("seg_w", "jv", "torque", "power", "q", "qvel", "foot_load_bw"); out = {k: sum((list(t[k]) for t in traces), []) for k in keys}
    out.update(dt=traces[0]["dt"], release_step=bounds[-1], release_time=sum(float(t["release_time"]) for t in traces), phase_bounds=bounds); return out
