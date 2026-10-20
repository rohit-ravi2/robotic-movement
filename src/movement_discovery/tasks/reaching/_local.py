"""Local helpers for the reaching family (2026-10-12). Bridges to the quasi-static servo construction (tasks/reach.py
ReachTask, tasks/wipe.py WipeTask: balance controller owns the legs, arm+trunk servos track a body-derived reach prior + knot offsets)
= "servo_knots" representation. ADDITIVE observer gates through demand["observe"] (the base task's own gates stay):
  point        controlled point in the PALM frame (a held tool's tip); tip_target -> TIP_OFF when it never comes within tol
  moving_target  dict(name, v, radius): the object gets velocity v at t = 0 (set in the loop, before the first step); NO_CONTACT when
               the controlled point never comes within tol of the object's surface
  obstacle     dict(pos, size, margin): an axis-aligned box the controlled point and the arm links (obstacle_bodies) must not enter;
               OBSTACLE_HIT graded by the deepest penetration. The MJX G1 body has NO collision geoms, so this gate is geometric
               and declared as such (a physical barrier would need Capability.BODY_COLLISION).
  orient_hold  max orientation error allowed during the hold (rad) over demand["orient"] items -> ORIENT_OFF
MuJoCo only inside functions; the task class is built lazily."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Capability, WorldSpec, Outcome, ValidityReport, Status
from movement_discovery.physics import validity as V
PALM0 = np.array([0.059, -0.244, 0.620])      # right palm site in the settled G1 stance (probe 2026-10-12); programs place targets relative to it


def make_task(robot="G1", objects=(), surfaces=(), wipe=False):
    import mujoco
    from tasks.reach import ReachTask
    from tasks.wipe import WipeTask
    Base = WipeTask if wipe else ReachTask
    scene = None
    if objects or surfaces:                      # the ONE scene builder (world.build; same cached file the runner builds for the program's WorldSpec)
        from movement_discovery import world as WD; scene = WD.build(WorldSpec(objects=[dict(o) for o in objects], surfaces=[dict(s) for s in surfaces]), robot)

    class ReachTaskX(Base):
        name = Base.name + "_x"
        def rollout(self, S, demand, p, ctrl_fn=None, record=False):
            ob = demand.get("observe") or {}
            if not ob: return super().rollout(S, demand, p, ctrl_fn=ctrl_fn, record=record)
            m = S["m"]; sid = S["site"]; r_pt = np.asarray(ob.get("point", [0, 0, 0]), float); tol = float(demand.get("tol", 0.03)); hold = float(demand.get("hold", 0.2))
            mt = ob.get("moving_target"); box = ob.get("obstacle"); tip = ob.get("tip_target"); ori = demand.get("orient") if ob.get("orient_hold") is not None else None
            st = dict(tip_dmin=1e9, mv_dmin=1e9, t_contact=None, pen_max=0.0, launched=False, ori_hold_max=0.0, t_reach=None)
            mt_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, mt["name"]) if mt else -1
            mt_j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, mt["name"] + "_free") if mt else -1
            bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ob.get("obstacle_bodies", ["right_elbow_link", "right_wrist_yaw_link"])] if box else []
            ori_items = [(m.site_bodyid[sid] if it.get("body", "palm") == "palm" else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, it["body"]), np.asarray(it["axis"], float), np.asarray(it["target"], float)) for it in (ori or [])]
            tgt = np.asarray(self.target_world(S, demand), float)
            def fn(tt, d, S_):
                if mt and not st["launched"]: da = int(m.jnt_dofadr[mt_j]); d.qvel[da:da + 3] = np.asarray(mt["v"], float); st["launched"] = True
                pt = d.site_xpos[sid] + d.site_xmat[sid].reshape(3, 3) @ r_pt
                if tip is not None: st["tip_dmin"] = min(st["tip_dmin"], float(np.linalg.norm(pt - np.asarray(tip, float))))
                if mt:
                    dist = float(np.linalg.norm(pt - d.xpos[mt_b])) - float(mt.get("radius", 0.0)); st["mv_dmin"] = min(st["mv_dmin"], dist)
                    if dist <= tol and st["t_contact"] is None: st["t_contact"] = tt
                if box:
                    c = np.asarray(box["pos"], float); h = np.asarray(box["size"], float) + float(box.get("margin", 0.03))
                    for q_ in [pt] + [d.xpos[b] for b in bodies]:
                        pen = float(np.min(h - np.abs(q_ - c)))
                        if pen > 0: st["pen_max"] = max(st["pen_max"], pen)
                if ori_items:
                    if st["t_reach"] is None and float(np.linalg.norm(d.site_xpos[sid] - tgt)) <= tol: st["t_reach"] = tt
                    if st["t_reach"] is not None and tt <= st["t_reach"] + hold:
                        e = 0.0
                        for (b, a_l, a_t) in ori_items: e += float(np.linalg.norm(np.cross(d.xmat[b].reshape(3, 3) @ a_l, a_t)))
                        st["ori_hold_max"] = max(st["ori_hold_max"], e)
                if ctrl_fn is not None: ctrl_fn(tt, d, S_)
            rec = super().rollout(S, demand, p, ctrl_fn=fn, record=record); reasons = list(rec["reasons"]); viol = float(rec["violation"])
            if tip is not None and st["tip_dmin"] > tol: reasons.append("TIP_OFF"); viol += 1.0 + 20.0*(st["tip_dmin"] - tol)
            if mt and st["t_contact"] is None: reasons.append("NO_CONTACT"); viol += 1.0 + 20.0*max(0.0, st["mv_dmin"] - tol)
            if box and st["pen_max"] > 0: reasons.append("OBSTACLE_HIT"); viol += 1.0 + 50.0*st["pen_max"]
            if ori_items and st["ori_hold_max"] > float(ob["orient_hold"]): reasons.append("ORIENT_OFF"); viol += 1.0 + 5.0*(st["ori_hold_max"] - float(ob["orient_hold"]))
            out = dict(rec, valid=(viol <= 0 and not reasons), reasons=reasons, violation=viol); out["extra"] = dict(rec["extra"], tip_dmin=st["tip_dmin"], moving_dmin=st["mv_dmin"], t_contact=st["t_contact"], obstacle_pen_max=st["pen_max"], ori_hold_max=st["ori_hold_max"]); return out
    return ReachTaskX(robot, scene=scene, objects=list(objects), surfaces=list(surfaces)) if not wipe else ReachTaskX(robot, scene=scene, objects=list(objects), surfaces=list(surfaces))


def palm_frame(robot="G1"):
    """Settled palm site position and rotation (world) on the bare robot: the frame that palm-frame tool points refer to."""
    import mujoco, throw as T, branch_i as BI
    S = T.setup(robot); BI.settle(S); m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d)
    return np.array(d.site_xpos[S["site"]], float), np.array(d.site_xmat[S["site"]], float).reshape(3, 3)


class ReachProgram(TaskProgram):
    """Shared scaffolding for the reaching programs: adapter embodiment, servo representation, outcome from the record."""
    status = Status.SOURCE_IMPLEMENTED; objects = (); surfaces = ()
    def embodiment(self, robot):
        from movement_discovery.embodiment import resolve; return resolve(robot, self)
    def strategy_space(self, scenario): return ["servo_knots"]
    def world(self, scenario): return WorldSpec(objects=[dict(o) for o in self.scene_objects(scenario)], surfaces=[dict(s) for s in self.scene_surfaces(scenario)], tag=None) if (self.scene_objects(scenario) or self.scene_surfaces(scenario)) else WorldSpec(tag="bare")
    def scene_objects(self, scenario): return []
    def scene_surfaces(self, scenario): return []
    def task(self, scenario, wipe=False): return make_task(scenario.params.get("robot", "G1"), self.scene_objects(scenario), self.scene_surfaces(scenario), wipe=wipe)
    def evaluate(self, rec, scenario):
        ex = rec["extra"]; v = V.classify(rec)
        keys = ("dmin", "t_reach", "hold_ok", "hold_mean", "peak_util", "work", "trunk_pitch_max", "knee_max", "cp_err_max", "ori_err_max", "tip_dmin", "moving_dmin", "t_contact", "obstacle_pen_max", "ori_hold_max", "pen_max", "tilt_mean", "force_mean", "in_band")
        meas = {k: (ex[k].tolist() if hasattr(ex.get(k), "tolist") else ex[k]) for k in keys if k in ex}
        return Outcome(value=float(rec["value"]), achieved=v.ok, measures=meas, validity=v)
    def validity(self, rec, scenario): return self.evaluate(rec, scenario).validity
    def observe(self, rec, scenario):
        tr = rec.get("extra", {}).get("trace")
        if not tr: return None
        return dict(kind="servo_trace_qpos_every_10ms", n=len(tr), trunk_pitch_max=rec["extra"].get("trunk_pitch_max"), knee_max=rec["extra"].get("knee_max"), work=rec["extra"].get("work"))
