"""Per-step telemetry beyond the frozen rollouts' trace: a ctrl_fn OBSERVER (returns zero control; chain it with the task's
controller through observe()) recording at `every` steps: time, COM position/velocity, linear and angular momentum about the
COM, contact points + normal forces (floor and object), support polygon (floor-contact points of the feet), object free-joint
states, actuator envelope utilisation (|w|/omega_rated, |tau|/tau_rated), per-joint positive work and disturbance work.
Channels come out as numpy arrays (Observer.arrays()) for archive.store_trace; events are derived in physics.events."""
from __future__ import annotations
from typing import Callable, Dict, List, Optional
import numpy as np


class Observer:
    def __init__(self, S, every: int = 1, objects: Optional[List[str]] = None):
        import mujoco
        self.S = S; m = S["m"]; self.m = m; self.every = int(every); self.J = S["jarr"]; self.nT = len(S["types"]); self.nA = len(self.J["aid"]); self.k = 0
        ti = np.asarray(self.J.get("ti", np.arange(self.nA))); self.names = [S["types"][int(ti[i])] if i < len(ti) and int(ti[i]) < self.nT else f"act{i}" for i in range(self.nA)]
        self.fl = S["floor"]; self.feet = S.get("feet") or _feet(m, S); self.robot_geoms = S.get("robot_geoms", set()); self.dt = float(m.opt.timestep)
        self.obj_j = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n + "_free") for n in (objects or [])}
        self.obj_j = {n: j for n, j in self.obj_j.items() if j >= 0}; self.obj_body = {n: int(m.jnt_bodyid[j]) for n, j in self.obj_j.items()}
        self.ch: Dict[str, List] = dict(t=[], com=[], vcom=[], p_lin=[], L_ang=[], support_poly=[], n_contacts=[], grf=[], contacts=[], util_speed=[], util_torque=[], work_joint=[], dist_work=[], objects=[], object_contact_t=[])
        self._work = np.zeros(self.nA); self._dw = 0.0; self._tq_prev = None
    def __call__(self, t: float, d, S) -> np.ndarray:
        self.k += 1
        if (self.k - 1) % self.every == 0: self._record(t, d)
        return np.zeros(self.nT)
    def _record(self, t, d):
        import mujoco, plant as P
        m = self.m; J = self.J; ch = self.ch; w = d.qvel[J["dof"]]; tq = d.actuator_force[J["aid"]]
        pw = tq*w; self._work += np.where(pw > 0, pw, 0.0)*self.dt*self.every
        com = d.subtree_com[0].copy(); vcom = P.com_velocity(m, d); M = float(self.S["M"])
        L = np.zeros(3); vel6 = np.zeros(6)
        for b in self.S.get("robot_bodies", range(1, m.nbody)):
            mb = float(m.body_mass[b])
            if mb <= 0: continue
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel6, 0); r = d.xipos[b] - com; L += mb*np.cross(r, vel6[3:]) + d.ximat[b].reshape(3, 3) @ (m.body_inertia[b]*(d.ximat[b].reshape(3, 3).T @ vel6[:3]))
        cons = []; grf = 0.0; poly = []; obj_hit = False; f6 = np.zeros(6)
        for ci in range(d.ncon):
            c = d.contact[ci]; mujoco.mj_contactForce(m, d, ci, f6); fn = float(f6[0]); g1, g2 = int(c.geom1), int(c.geom2)
            cons.append([float(c.pos[0]), float(c.pos[1]), float(c.pos[2]), fn, g1, g2])
            if self.fl in (g1, g2):
                grf += fn; o = g2 if g1 == self.fl else g1
                if any(o in fg for fg in self.feet): poly.append([float(c.pos[0]), float(c.pos[1])])
            bodies = (int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2]))
            if any(b in bodies for b in self.obj_body.values()) and (g1 in self.robot_geoms or g2 in self.robot_geoms): obj_hit = True
        if obj_hit: ch["object_contact_t"].append(float(t))
        self._dw += float(np.sum(d.xfrc_applied[:, :3]*np.array([_bvel(m, d, b) for b in range(m.nbody)])))*self.dt*self.every if np.any(d.xfrc_applied) else 0.0
        ch["t"].append(float(t)); ch["com"].append(com); ch["vcom"].append(np.asarray(vcom, float)); ch["p_lin"].append(M*np.asarray(vcom, float)); ch["L_ang"].append(L)
        ch["support_poly"].append(np.array(poly) if poly else np.zeros((0, 2))); ch["n_contacts"].append(len(cons)); ch["grf"].append(grf); ch["contacts"].append(np.array(cons) if cons else np.zeros((0, 6)))
        om_ = np.asarray(J["om"]); tau_ = np.asarray(J["tau"]); om_ = om_ if len(om_) == self.nA else om_[np.asarray(J["ti"])]; tau_ = tau_ if len(tau_) == self.nA else tau_[np.asarray(J["ti"])]
        ch["util_speed"].append(np.abs(w)/np.maximum(om_, 1e-9)); ch["util_torque"].append(np.abs(tq)/np.maximum(tau_, 1e-9)); ch["work_joint"].append(self._work.copy()); ch["dist_work"].append(self._dw)
        ch["objects"].append({n: np.concatenate([d.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 7], d.qvel[m.jnt_dofadr[j]:m.jnt_dofadr[j] + 6]]) for n, j in self.obj_j.items()})
    def arrays(self) -> dict:
        ch = self.ch; out = {k: np.asarray(ch[k]) for k in ("t", "com", "vcom", "p_lin", "L_ang", "n_contacts", "grf", "util_speed", "util_torque", "work_joint", "dist_work")}
        out["support_poly"] = ch["support_poly"]; out["contacts"] = ch["contacts"]; out["objects"] = ch["objects"]; out["object_contact_t"] = list(ch["object_contact_t"]); out["types"] = list(self.names); return out
    def summary(self) -> dict:
        a = self.arrays()
        if len(a["t"]) == 0: return {}
        return dict(duration=float(a["t"][-1]), com_disp=(a["com"][-1] - a["com"][0]).tolist(), com_speed_max=float(np.linalg.norm(a["vcom"], axis=1).max()), L_ang_max=float(np.linalg.norm(a["L_ang"], axis=1).max()),
                    util_speed_max={self.names[i]: round(float(v), 2) for i, v in enumerate(a["util_speed"].max(0)) if v > 0.5}, util_torque_max=float(a["util_torque"].max()), grf_max_bw=float(a["grf"].max()/(self.S["M"]*9.80665)),
                    work_joint={self.names[i]: round(float(v), 1) for i, v in enumerate(a["work_joint"][-1])}, disturbance_work=float(a["dist_work"][-1]), object_contact_t=(a["object_contact_t"][0] if a["object_contact_t"] else None), support_poly_min_pts=int(min(len(p) for p in a["support_poly"])))


def _feet(m, S):
    """Per-foot geom sets for plants that carry only foot_geoms (branch_i jump plants): split by foot body."""
    import mujoco, branch_i as BI
    fb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in BI.ROBOTS[S.get("robot", "G1")]["foot_bodies"]]
    return [{g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b} for b in fb]


def _bvel(m, d, b):
    import mujoco
    v = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0); return v[3:]


def observe(inner: Optional[Callable], obs: Observer) -> Callable:
    """Chain: the task's controller output plus the observer's zeros (the observer only reads)."""
    def fn(t, d, S):
        obs(t, d, S); return inner(t, d, S) if inner is not None else np.zeros(obs.nT)
    return fn
