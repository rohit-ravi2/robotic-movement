"""Balance-family helpers: the STAND bridge (balance.py capture-point controller in stand mode, Phase 14 tuned gains, under a
disturbance; prior only, zero knots) producing a format-C record, and disturbance conversion. MuJoCo only inside functions."""
from __future__ import annotations
import numpy as np
from ..locomotion._local import (BodyProgram, terrain, impulse, force, platform_motion, obstacle, CS_NAME, schedule_task, q_trace, pelvis_kin, upright, base_validity, no_trace, fail, schedule_demand, phase, foot_loads)  # noqa: F401


IMP_DT = 0.05

def push_of(spec):
    """WorldSpec disturbance dict -> balance.simulate push tuple (t0, duration, force_xyz)."""
    ds = [d for d in spec.disturbances if d.get("kind") in ("impulse", "force")]
    if not ds: return None
    d = ds[0]
    if d["kind"] == "impulse": return (float(d["t0"]), IMP_DT, np.asarray(d["vec"], float)/IMP_DT)      # N s -> N over IMP_DT (balance.simulate applies a force window)
    return (float(d["t0"]), float(d["t1"] - d["t0"]), np.asarray(d["vec"], float))

def stand_bridge(robot="G1", push=None, T=2.0, mass_at=None, friction=None, gains=None):
    """ONE execution of the standing controller from the settled stance (integration check only). mass_at=(site, kg) merges a payload;
    friction overrides the floor's sliding friction on an uncached plant copy."""
    import throw as TH, branch_i as BI, balance as B
    from engine import core
    kw = dict(ball=False) if mass_at is None else dict(site=mass_at[0], mass=float(mass_at[1]), ball=False)
    S = TH.setup(robot, **kw)
    if friction is not None: S = TH.setup(robot, start=np.array(S["q0"]) + 1e-9, **kw); S["m"].geom_friction[S["floor"], 0] = float(friction)
    BI.apply_dt(S, core.DT_OPT); BI.settle(S)
    ctl, par = core.landing_controller(S, robot); ctl.p["mode"] = "stand"; ctl.t_touch = -1e9; ctl.p.update(gains or {})
    o = B.simulate(S, ctl, np.asarray(S["q_settled"]), np.asarray(S["v_settled"]), T=T, push=push, record=True)
    reasons = (["NONFOOT_CONTACT"] if o["nonfoot"] else []); tr = o.pop("trace")
    return dict(value=None, valid=not reasons, reasons=reasons, violation=0.0, trace=dict(q=np.asarray(tr).tolist(), dt=10*float(S["m"].opt.timestep)), extra=dict(o, controller=dict(par, mode="stand")), plant=S)

def settle_time(q, dt, t_after, tilt_ok=np.radians(8), v_ok=0.15, hold=0.3):
    """First time after t_after at which pelvis tilt < tilt_ok and pelvis speed < v_ok hold for `hold` s; None if never."""
    xyz, yaw, tilt, v = pelvis_kin(q, dt); sp = np.linalg.norm(v, axis=1); ok = (tilt < tilt_ok) & (sp < v_ok); n = max(1, int(round(hold/dt))); i0 = int(t_after/dt)
    for i in range(i0, len(ok) - n + 1):
        if ok[i:i+n].all(): return float(i*dt - t_after)
    return None

def foot_sites_xy(S, q):
    import mujoco; m = S["m"]; d = mujoco.MjData(m); ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in ("left_foot", "right_foot")]; out = []
    for r in q: d.qpos[:] = r; mujoco.mj_kinematics(m, d); out.append([d.site_xpos[i][:2].copy() for i in ids])
    return np.asarray(out)


class StandTask:
    """engine.task.Task-shaped wrapper of the standing controller: the Phase 14 capture-point controller (stand mode) IS the prior; the two
    parameters scale its capture-point and trunk gains (0.5..1.5 x tuned). The plant is built on the world scene (`scene`, set by the
    program or the runner) and settled on raised terrain (demand["start_z"], physics.settle). The rollout is a LOCAL stand loop (the
    few lines of balance.simulate) that calls ctrl_fn(t, d, S) every step before mj_step, so physics.hooks.world_hooks delivers mocap
    platforms / trapdoors and wrench disturbances; when no ctrl_fn is given the hooks are built from `spec` here. Servo mode: the
    returned command is ignored (observers only). Record: q and foot loads (BW) every 10 steps."""
    name = "stand"
    def __init__(self, robot="G1", scene=None): self.robot = robot; self._plants = {}; self.scene = scene; self.spec = None
    def demand_default(self): return dict(T=2.0, mass_at=None, friction=None, gains=None)
    def sigma0(self, demand, warm): return 0.2
    def plant(self, demand, dt=None, fresh=False):
        import throw as TH, branch_i as BI; from engine import core
        ma = demand.get("mass_at"); fr = demand.get("friction"); dz = float(demand.get("start_z", 0.0) or 0.0); key = (ma and tuple(ma), fr, fresh, self.scene, dz)
        if key in self._plants: return self._plants[key]
        sc = {"mjcf": self.scene} if self.scene else {}
        kw = dict(ball=False, **sc) if ma is None else dict(site=ma[0], mass=float(ma[1]), ball=False, **sc); S = TH.setup(self.robot, **kw)
        if fr is not None or fresh or dz > 0: S = TH.setup(self.robot, start=np.array(S["q0"]) + 1e-9*(1 + len(self._plants)), **kw)
        if fr is not None: S["m"].geom_friction[S["floor"], 0] = float(fr)
        if dz > 0:
            from movement_discovery.physics.settle import settle_on_terrain; settle_on_terrain(S, dz, resettle=False)
        BI.apply_dt(S, core.DT_OPT if dt is None else dt); BI.settle(S); self._plants[key] = S; return S
    def n_params(self, S, demand): return 2
    def unpack(self, S, demand, z): z = np.asarray(z, float); return dict(x=z, k_cp_scale=1.0 + 0.5*float(z[0]), k_trunk_scale=1.0 + 0.5*float(z[1]))
    def controller(self, S, demand, p): return None
    def rollout(self, S, demand, p, ctrl_fn=None, record=False):
        import mujoco, balance as B, branch_i as BI, plant as P; from engine import core
        m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; d.qvel[:] = S["v_settled"]; mujoco.mj_forward(m, d)
        ctl, par = core.landing_controller(S, self.robot); ctl.p["mode"] = "stand"; ctl.t_touch = -1e9; ctl.p.update(demand.get("gains") or {})
        ctl.p["k_cp"] *= p.get("k_cp_scale", 1.0); ctl.p["k_trunk"] *= p.get("k_trunk_scale", 1.0); ctl._kc = None
        if ctrl_fn is None and self.spec is not None:
            from movement_discovery.physics.hooks import world_hooks; ctrl_fn = world_hooks(self.spec, len(S["types"]))
        restore = B.all_servos(S, ctl.p["stiff"]); fl = S["floor"]; feet = S["feet"]; foot = S["foot_geoms"]; tb = ctl.tb; M = S["M"]*B.G
        nonfoot = False; peak = 0.0; tilt_max = 0.0; trace = []; loads = []; T = float(demand.get("T", 2.0)); dt = m.opt.timestep; world_contact = False
        try:
            for (a_, dof, qad, tau, kp, kd) in S["held"]: d.ctrl[a_] = S["q_settled"][qad]
            for s_ in range(int(T/dt)):
                t = s_*dt; fz = 0.0; ff = [0.0, 0.0]
                for ci in range(d.ncon):
                    c = d.contact[ci]; g1, g2 = c.geom1, c.geom2
                    if fl in (g1, g2) or (g1 not in S["robot_geoms"]) != (g2 not in S["robot_geoms"]):      # floor or any world piece against the robot
                        fv = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fv); o = g2 if g1 in S["robot_geoms"] and g2 not in S["robot_geoms"] else g1
                        o = g1 if g1 in S["robot_geoms"] else g2; fz += fv[0]
                        if o in feet[0]: ff[0] += fv[0]
                        elif o in feet[1]: ff[1] += fv[0]
                        elif o in S["robot_geoms"]: nonfoot = True
                k_c = int(round(t/B.CTRL_DT, 6))
                if k_c != ctl._kc: ctl._kc = k_c; d.ctrl[:] = ctl.targets(k_c*B.CTRL_DT, d, fz > 1.0)
                d.xfrc_applied[:] = 0
                if ctrl_fn is not None: ctrl_fn(t, d, S)                       # hooks: mocap pieces, wrench disturbances; command ignored (servo mode)
                BI.backemf_brake(m, d, S); mujoco.mj_step(m, d)
                peak = max(peak, fz); tilt_max = max(tilt_max, float(np.arccos(np.clip(d.xmat[tb][8], -1, 1))))
                if s_ % 10 == 0: trace.append(d.qpos.copy()); loads.append([ff[0]/M, ff[1]/M])
        finally: restore()
        up = float(d.xmat[tb][8]); z = float(d.subtree_com[0][2]); v = float(np.linalg.norm(P.com_velocity(m, d)))
        reasons = (["NONFOOT_CONTACT"] if nonfoot else []) + (["FELL"] if up < 0.3 else [])
        o = dict(ok=bool((not nonfoot) and up > 0.85 and z > 0.6*ctl.z_stand and v < 0.5), nonfoot=nonfoot, up_end=up, z_end=z, v_end=v, peak_grf_bw=peak/M, tilt_max=tilt_max, controller=dict(ctl.p), final_state=(d.qpos.copy(), d.qvel.copy()), hooks=(getattr(ctrl_fn, "stats", None) if ctrl_fn is not None else None))
        return dict(value=float(-tilt_max), valid=not reasons, reasons=reasons, violation=0.0, cost=0.0, final_state=o["final_state"], trace=dict(q=np.asarray(trace).tolist(), foot_load_bw=loads, dt=10*float(dt)), extra=o, plant=S)
    def objective(self, rec, demand): return -rec["value"]
    def floor(self, demand): return None
    def warm(self, S, demand, stored): return np.asarray(stored["x"], float)


class BalanceProgram(BodyProgram):
    """Common: upright = pelvis tilt < 30 deg and pelvis height > 70% of the start height over the last `t_hold` s (declared per program)."""
    observe_site = "right_foot"
    def strategy_space(self, scenario): return ["servo_knots", "torque_knots"]
    def bridge_kwargs(self, scenario): return {}
    def demand(self, scenario):
        """Bridge: (StandTask, demand): the standing controller as the prior; disturbance and friction from the program's WorldSpec."""
        from movement_discovery import world as W
        w = self.world(scenario); robot = scenario.params.get("robot", "G1"); t = StandTask(robot, scene=(W.build(w, robot) or None)); t.spec = w
        d = dict(t.demand_default(), T=scenario.params["T"], friction=w.friction, **self.bridge_kwargs(scenario)); dz = float(W.start_z_offset(w)) if t.scene else 0.0
        if dz > 1e-3: d["start_z"] = dz
        return t, d
    def observe(self, rec, scenario):
        base = BodyProgram.observe(self, rec, scenario) or {}; ex = (rec or {}).get("extra") or {}
        return base | dict(tilt_max_deg=(float(np.degrees(ex["tilt_max"])) if ex.get("tilt_max") is not None else base.get("tilt_max_deg")), peak_grf_bw=ex.get("peak_grf_bw"), controller=ex.get("controller"), hooks=ex.get("hooks"))
