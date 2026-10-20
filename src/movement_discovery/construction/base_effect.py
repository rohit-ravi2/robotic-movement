"""BaseEffectTask: the EffectTask construction with the EFFECT ON THE BASE (pelvis / COM) instead of an effector site: value =
pelvis displacement (demand["measure"] = "displacement", default) or pelvis velocity ("speed") projected on the demanded direction
(elev, az in the body frame at the start; None = horizontal magnitude); the cone is judged on the base motion. The plant is the
throw plant (support contract, driven set, counterfactual ranges/torques) built on an optional world scene (demand["scene"] =
MJCF path), from an optional posed start (demand["initial_posture"] = qpos) and settled on raised terrain (demand["start_z"]).
The body-derived reference option is engine.strategy.ascent_prior with effector ("com",): capacity-weighted ascent of the COM along
the direction, both feet pinned (wind-up = counter-lean, release = lean/drive); legs move. A phase without prior is zero knots
(stand). Nothing here authors a gait: the representation (ContactSchedule) samples/searches the support contracts and horizons."""
from __future__ import annotations
import json
import numpy as np
from tasks.effect import EffectTask
from engine import core
import throw as T, branch_i as BI

BASE_EFFECTORS = ("pelvis", "com", "base")


def is_base_effector(name) -> bool: return str(name) in BASE_EFFECTORS


class BaseEffectTask(EffectTask):
    name = "base_effect"
    def demand_default(self):
        return dict(super().demand_default(), effector="pelvis", mass=0.0, measure="displacement", elev_tol=np.radians(90.0), az_tol=np.radians(180.0), prior=None)
    def plant(self, demand, dt=core.DT_OPT, fresh=False):
        import mujoco
        jr = demand.get("jnt_ranges"); st = demand.get("initial_posture"); st = None if st is None else np.asarray(st, float); dz = float(demand.get("start_z", 0.0) or 0.0)
        key = (dt, fresh, demand.get("support", "both"), tuple(demand.get("driven") or ()), demand.get("scene") or None, None if st is None else np.round(st, 6).tobytes(), dz, json.dumps(jr, sort_keys=True) if jr else None)
        if key in self._plants: return self._plants[key]
        kw = dict(site=T.CFG[self.robot]["hand_site"], ball=False, mass=0.0, support=demand.get("support", "both"), driven=demand.get("driven"), mjcf=(demand.get("scene") or None))
        S0 = T.setup(self.robot, start=st, **kw); own = fresh or bool(jr) or dz > 0 or st is not None
        S = T.setup(self.robot, start=np.array(S0["q0"]) + 1e-9*(1 + len(self._plants)), **kw) if own else S0     # uncached copy when the plant will be modified
        if jr:
            m = S["m"]
            for jn, (lo, hi) in jr.items():
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn); assert j >= 0, jn; m.jnt_range[j] = [float(lo), float(hi)]
        if dz > 0:
            from movement_discovery.physics.settle import settle_on_terrain
            settle_on_terrain(S, dz, resettle=False)
        BI.apply_dt(S, dt); BI.settle(S); S["_setup_kw"] = kw; S["base_effect"] = True; S["start_z"] = dz
        for nm in (demand.get("held") or []):      # a declared-held object starts IN THE HAND (initial condition; the settle let it fall): tasks/reach.py convention, LABEL held_start_abstraction
            m = S["m"]; j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{nm}_free")
            if j < 0: continue
            d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d); a = int(m.jnt_qposadr[j]); da = int(m.jnt_dofadr[j]); sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
            if np.linalg.norm(d.qpos[a:a + 3] - d.site_xpos[sid]) > 0.10:
                S["q_settled"][a:a + 3] = d.site_xpos[sid]; S["q_settled"][a + 3:a + 7] = [1, 0, 0, 0]; S["v_settled"][da:da + 6] = 0.0; S["held_start_abstraction"] = S.get("held_start_abstraction", []) + [nm]
        self._plants[key] = S; return S
    @staticmethod
    def dvec(demand):
        dr = demand.get("direction")
        if dr is None: return None
        e, a = map(float, dr); return np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    def ref(self, S, demand):
        dr = demand.get("direction"); key = (id(S["m"]), None if dr is None else tuple(np.round(dr, 4)), demand.get("capacity_mode", "power"))
        if key not in self._refs:
            from engine import strategy as ST
            dv = self.dvec(demand); dv = np.array([1.0, 0.0, 0.0]) if dv is None else dv
            qw, qx, qy, qz = np.asarray(S["q_settled"])[3:7]; yaw0 = float(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))
            c, s_ = np.cos(yaw0), np.sin(yaw0); dw = np.array([c*dv[0] - s_*dv[1], s_*dv[0] + c*dv[1], dv[2]])      # body-frame direction -> world
            r = ST.ascent_prior(S, ("com",), dw, float(demand.get("ref_horizon", T.HOR)), T.NK, t_w=0.40, t_ref=0.55, move_legs_in_windup=True, leg_cap=0.30, key="base_ascent_prior", hold_com_xy=False, integrate_base=True, limit_frac=0.05)
            r["prep_qpos"] = None; r["t_prep_end"] = 0.05; self._refs[key] = r
        return self._refs[key]
    def rollout(self, S, demand, p, ctrl_fn=None, record=False):
        rec = super().rollout(S, dict(demand, ignore_cone=True), p, ctrl_fn=ctrl_fn, record=record)
        q0 = np.asarray(S["q_settled"]); qf, vf = rec["final_state"]; disp = np.asarray(qf[:3]) - q0[:3]; vel = np.asarray(vf[:3])
        qw, qx, qy, qz = q0[3:7]; yaw0 = float(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))
        mv = disp if demand.get("measure", "displacement") == "displacement" else vel; dv = self.dvec(demand)
        horiz = float(np.hypot(mv[0], mv[1])); elev = float(np.arctan2(mv[2], max(horiz, 1e-9))); az = float((np.arctan2(mv[1], mv[0]) - yaw0 + np.pi) % (2*np.pi) - np.pi)
        if dv is None: value = horiz; in_cone = True
        else:
            c, s_ = np.cos(yaw0), np.sin(yaw0); dw = np.array([c*dv[0] - s_*dv[1], s_*dv[0] + c*dv[1], dv[2]]); value = float(mv @ dw)
            ec, ac = map(float, demand["direction"]); et, at = float(demand.get("elev_tol", 0.35)), float(demand.get("az_tol", 0.35))
            moved = (horiz > 0.02) if demand.get("measure", "displacement") == "displacement" else (horiz > 0.1)
            in_cone = (not moved) or (abs(elev - ec) <= et and abs((az - ac + np.pi) % (2*np.pi) - np.pi) <= at)
        reasons = [r_ for r_ in rec["reasons"] if r_ != "OUT_OF_CONE"]; viol = float(rec["violation"])
        if not in_cone: viol += 1.0 + 20.0*(max(0.0, abs(elev - ec) - et) + max(0.0, abs((az - ac + np.pi) % (2*np.pi) - np.pi) - at)); reasons.append("OUT_OF_CONE")
        out = dict(rec, value=float(value), valid=(viol <= 0 and not reasons), reasons=reasons, violation=viol)
        out["extra"] = dict(rec["extra"], base_disp=disp.tolist(), base_vel=vel.tolist(), base_elev=elev, base_az=az, eff_speed=rec["value"], elev=elev, az=az); return out
