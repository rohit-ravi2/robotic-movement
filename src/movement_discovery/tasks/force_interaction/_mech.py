"""WorldReachTask: the quasi-static servo construction (tasks/reach.py: legs by the balance controller, arm+trunk servos on a
body-derived reference + knot offsets) on a WorldSpec compiled by movement_discovery.world (mechanisms, objects, surfaces) with
effector sphere colliders (world.colliders). Effects measured: a MECHANISM joint's displacement (door angle, drawer travel, crank angle,
lever angle) and every free OBJECT's displacement; sustained-force gates (f_max, t_contact_min) on effector<->mechanism contact as
tasks/impact.py. Holds: the engine.world WELD (G1, 'experimental simplification') through grasp_on_reach / held; on the G1H the
Dex3 fingers close on the handle (World.finger_ctrl). Demand (over ReachTask): mechanism=<name>, push=[dx,dy,dz] (palm target
relative to the handle site, world), min_disp, sign, f_max, t_contact_min. MuJoCo only inside methods."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import WorldSpec


def make(robot, spec: WorldSpec, effector="right_palm", colliders=("right_palm",), hand_collision=True):
    """hand_collision=False (payload welds): no effector sphere and the hand geoms stay non-colliding, the engine.world J14 convention
    (the weld engages at grasp_tol; a 4 cm palm sphere cannot get within 5.5 cm of a 1.5 cm handle). Labelled in the programs."""
    import mujoco
    from engine import core
    from tasks.reach import ReachTask
    from movement_discovery.world import colliders as CL
    mechs = {mk["name"]: mk for mk in spec.mechanisms}; objs = list(spec.objects); surfs = list(spec.surfaces)

    class WorldReachTask(ReachTask):
        name = "world_reach"
        def __init__(self):
            super().__init__(robot, scene=CL.scene(spec, robot, colliders), objects=objs, surfaces=surfs); self.spec = spec; self.mechanisms = mechs
        def plant(self, demand, dt=core.DT_OPT, fresh=False):
            S = super().plant(demand, dt, fresh)
            if not S.get("_eff_on"): S["_eff_on"] = CL.enable_effector_geoms(S) if hand_collision else 0
            so = demand.get("start_open")
            if so and S.get("_start_open") != float(so):     # an OPEN start: the mechanism joint's settled coordinate is world state (the drawer stands open), not a movement
                m_ = S["m"]; qa = int(m_.jnt_qposadr[mujoco.mj_name2id(m_, mujoco.mjtObj.mjOBJ_JOINT, mechs[demand["mechanism"]]["joint"])])
                S["q_settled"][qa] = float(so); S["q_stance0"] = np.array(S["q_settled"]); S["_start_open"] = float(so)
            return S
        def world(self, S):
            if "world" in S: return S["world"]
            wd = super().world(S); m = S["m"]; sites = {}
            for nm, mk in self.mechanisms.items():          # the moving link is the weld target; proximity/position are read at the HANDLE site
                wd.objects[nm] = dict(mk, graspable=mk.get("graspable", True), kind="mechanism", size=[0.0, 0.0, 0.0]); wd.bid[nm] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, mk["body"])
                e = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_{mk['body']}")
                if e >= 0: wd.eq[nm] = e
                sites[nm] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, mk["handle_site"])
            for o in objs:                                     # objects with a grip point (handles): proximity measured to the grip, not the body origin
                if o.get("grip_point") is not None: sites[o["name"]] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, o["name"] + "_grip")
            pd0, pos0 = wd.palm_dist, wd.pos
            wd.palm_dist = lambda d, name: float(np.linalg.norm(d.site_xpos[wd.palm] - d.site_xpos[sites[name]])) if name in sites else pd0(d, name)
            wd.grip_pos = lambda d, name: (d.site_xpos[sites[name]].copy() if name in sites else pos0(d, name))
            return wd
        def joint_value(self, S, qpos, name):
            m = S["m"]; j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, self.mechanisms[name]["joint"]); return float(np.asarray(qpos)[m.jnt_qposadr[j]])
        def handle_world(self, S, name):
            m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d); return self.world(S).grip_pos(d, name)
        def target_world(self, S, demand):
            if demand.get("push") is not None and (demand.get("mechanism") or demand.get("target_object")): return self.handle_world(S, demand.get("mechanism") or demand["target_object"]) + np.asarray(demand["push"], float)
            return super().target_world(S, demand)
        def rollout(self, S, demand, p, ctrl_fn=None, record=False):
            m = S["m"]; name = demand.get("mechanism"); mk = self.mechanisms.get(name); wd = self.world(S)
            watch = {wd.bid[name]} if mk else {wd.bid[o["name"]] for o in objs}; rg = S.get("robot_geoms") or {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == int(m.site_bodyid[S["site"]])}
            st = dict(f_max=0.0, t_contact=0.0, q_max=0.0, q_min=0.0, q0=None); dt_ = float(m.opt.timestep); every = max(1, int(round(0.002/dt_)))
            qa = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, mk["joint"])]) if mk else None
            def obs(tt, d, S_):
                if qa is not None:
                    if st["q0"] is None: st["q0"] = float(d.qpos[qa])
                    q = float(d.qpos[qa]) - st["q0"]; st["q_max"] = max(st["q_max"], q); st["q_min"] = min(st["q_min"], q)
                if int(round(tt/dt_)) % every == 0:
                    F = 0.0
                    for ci in range(d.ncon):
                        c = d.contact[ci]; b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                        if (b1 in watch or b2 in watch) and (c.geom1 in rg or c.geom2 in rg): f6 = np.zeros(6); mujoco.mj_contactForce(m, d, ci, f6); F += abs(float(f6[0]))
                    if F > 0.5: st["t_contact"] += every*dt_; st["f_max"] = max(st["f_max"], F)
                return ctrl_fn(tt, d, S_) if ctrl_fn is not None else None
            d_ = dict(demand, tol=float(demand.get("reach_tol", 0.5 if mk else demand.get("tol", 0.03))), hold=(0.05 if mk else demand.get("hold", 0.2)))   # mechanism: the reach gate is relaxed, the effect is the joint displacement
            lost = []
            if demand.get("held"):                                  # a handle/handled object declared held must still be at the hand (no teleport of a jointed body)
                dh = mujoco.MjData(m); dh.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, dh)
                lost = [nm for nm in demand["held"] if wd.palm_dist(dh, nm) > 0.12 and nm not in demand.get("start_in_hand", ())]; d_["held"] = [nm for nm in demand["held"] if nm not in lost]   # start_in_hand: the object is placed in the hand at t = 0 (tasks/reach.py labelled abstraction)
            rec = super().rollout(S, d_, p, ctrl_fn=obs, record=record); reasons = list(rec["reasons"]); viol = float(rec["violation"]); ex = rec["extra"]
            q0 = np.asarray(S["q_settled"]); qf_all = np.asarray(ex["final_state"][0]); disp_obj = {}
            for o in objs:
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, o["name"] + "_free")
                if j >= 0: a = int(m.jnt_qposadr[j]); disp_obj[o["name"]] = (qf_all[a:a + 3] - q0[a:a + 3]).tolist()
            ex.update(obj_disp=disp_obj, f_max_obj=st["f_max"], t_contact=st["t_contact"])
            value = rec["value"]
            if mk:
                qf = self.joint_value(S, qf_all, name) - (st["q0"] or 0.0); sgn = float(demand.get("sign", mk.get("open_sign", 1.0))); disp = sgn*qf
                reasons = [r for r in reasons if r not in ("NOT_REACHED", "NOT_HELD")]; md = float(demand.get("min_disp", 0.0))
                if disp < md: viol += 1.0 + 5.0*(md - disp); reasons.append("NOT_MOVED")
                ex.update(mech_disp=disp, mech_disp_max=max(sgn*st["q_max"], sgn*st["q_min"]), mech_joint=mk["joint"]); value = disp
            if demand.get("f_max") is not None and st["f_max"] > float(demand["f_max"]): viol += 1.0 + 0.01*(st["f_max"] - float(demand["f_max"])); reasons.append("IMPULSIVE_CONTACT")
            if demand.get("t_contact_min") is not None and st["t_contact"] < float(demand["t_contact_min"]): viol += 1.0 + 5.0*(float(demand["t_contact_min"]) - st["t_contact"]); reasons.append("CONTACT_TOO_BRIEF")
            if lost: reasons.append("HOLD_LOST"); viol += 2.0
            return dict(rec, value=value, valid=(viol <= 0 and not reasons), reasons=reasons, violation=viol)
        def objective(self, rec, demand): return (-rec["value"] if demand.get("mechanism") else rec["cost"]) + 0.5*rec["extra"].get("peak_util", 0.0)
    return WorldReachTask()


def demand(task, mech, push, horizon=2.5, grasp=None, **kw):
    """Servo demand on a mechanism: reach the handle (+push, world) with the reach prior; weld on arrival when grasp=<name>."""
    d = dict(task.demand_default(), mechanism=mech["name"], push=list(push), horizon=float(horizon), omega_servo=120.0, trunk_cap=0.05, hold=0.05, prior="ascent", approach="direct")
    if grasp: d.update(grasp_on_reach=mech["name"], grasp_tol=0.08)      # 8 cm: the palm sphere (4 cm) cannot touch a 1.2 cm knob closer than ~5.5 cm; the weld is a labelled abstraction anyway
    d.update(kw); return d


def plan(t, demands, events=(), settle_T=0.0, freeze_T=0.4, missing=()):
    """Plan of WorldReachTask steps (one demand each; held objects read from the demands)."""
    from movement_discovery.core.contracts import Plan, Step
    ev = list(events) + [None]*(len(demands) - len(events))
    return Plan(steps=[Step(task=t, demand=dict(d), held=list(d.get("held") or []), event=e) for d, e in zip(demands, ev)], robot=t.robot, settle_T=settle_T, freeze_T=freeze_T, missing=list(missing))


class MechProgram:
    """Mixin: shared embodiment/evaluate/validity/observe over WorldReachTask records (single rollout or a runner.run_sequence result)."""
    unit = "rad"; min_key = "min_disp"
    def embodiment(self, robot):
        from movement_discovery import embodiment as E; return E.resolve(robot, self)
    def run(self, scenario):
        """Thin wrapper: ONE prior-only rollout of the Plan through the runner (raw record)."""
        from movement_discovery import runner
        return runner.rollout(self.id, scenario, robot=scenario.params.get("robot", "G1"))["rec"]
    def _value(self, rec):
        last = rec["steps"][-1] if "steps" in rec else rec; return float(last["extra"].get("mech_disp", last["value"]))
    def evaluate(self, rec, scenario):
        from . import _base as B; req = float(scenario.params.get(self.min_key, 0.0)); v = self._value(rec)
        e = rec["steps"][-1]["extra"] if "steps" in rec else rec["extra"]
        m = dict(displacement=v, unit=self.unit, required=req, f_max=e.get("f_max_obj"), t_contact=e.get("t_contact"), held_after=e.get("held_after"), disp_max=e.get("mech_disp_max"))
        if "steps" in rec: return B.seq_outcome(rec, value=v, measures=m, achieved=v >= req)
        return B.outcome(rec, value=v, measures=B.reach_measures(rec) | m, achieved=v >= req)
    def validity(self, rec, scenario): return self.evaluate(rec, scenario).validity
    def observe(self, rec, scenario):
        e = rec["steps"][-1]["extra"] if "steps" in rec else rec.get("extra", {})
        return dict(work=e.get("work"), trunk_pitch_max=e.get("trunk_pitch_max"), knee_max=e.get("knee_max"), peak_util=e.get("peak_util"), cp_err_max=e.get("cp_err_max"))
