"""IMPACT effect (construct, 2026-10-11): the same construction as the velocity effect, but the EFFECT is on an object: a ball
on a tee (or on the floor) is struck by the effector; value = ball speed after impact; the direction cone applies to the BALL's
velocity; a miss is a violation. The prior is built for the effector's direction (the construction does not know where the ball
is); the search has to put the effector through it. Demand adds: object=dict(pos, radius, mass, tee=bool)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco, throw as T, branch_i as BI
from engine import core, world as W
from tasks.effect import EffectTask


class ImpactTask(EffectTask):
    name = "impact"
    def demand_default(self):
        return dict(super().demand_default(), mass=0.0, object=dict(pos=[0.42, -0.25, 1.0], radius=0.05, mass=0.1, tee=True), hit_min=0.5, elev_tol=np.radians(20.0), az_tol=np.radians(20.0))
    def scene(self, demand):
        o = demand["object"]; pos = list(o["pos"])
        if o.get("kind", "ball") == "box":      # PUSH target: a box standing on the floor (the effect is its displacement, not its speed)
            objs = [W.obj("ball", kind="box", size=tuple(o["size"]), mass=float(o["mass"]), pos=pos, graspable=False, rgba=(0.2, 0.4, 0.9, 1))]; surfs = []
        else:
            r = float(o["radius"])
            objs = [W.obj("ball", kind="ball", size=(r,), mass=float(o["mass"]), pos=pos, graspable=False, rgba=(0.9, 0.3, 0.2, 1))]
            surfs = [W.surface("tee", pos=(pos[0], pos[1], 0.5*(pos[2] - r)), size=(0.015, 0.015, 0.5*(pos[2] - r)))] if o.get("tee", True) and pos[2] - r > 0.02 else []
        path = W.build_scene(self.robot, objs, surfs)
        # EFFECTOR COLLIDER: a sphere at the effector site (the body's own collision capsule sits away from the palm site: the hand
        # passed within 2 mm of the ball centre with zero contacts). r = 4 cm (fist / forefoot), massless.
        out = path.replace(".xml", f"_eff_{demand.get('effector')}.xml")
        if not os.path.exists(out):
            spec = mujoco.MjSpec.from_file(path); mb = mujoco.MjModel.from_xml_path(path); sid = mujoco.mj_name2id(mb, mujoco.mjtObj.mjOBJ_SITE, demand.get("effector")); bname = mujoco.mj_id2name(mb, mujoco.mjtObj.mjOBJ_BODY, mb.site_bodyid[sid])
            body = spec.body(bname) if hasattr(spec, "body") else spec.find_body(bname)
            pos = np.array(mb.site_pos[sid], float); r_ = 0.04
            if "foot" in demand.get("effector"): pos = pos + np.array([0.0, 0.0, r_])*0 + (mb.site_quat[sid]*0)[:3] + np.array([0.0, 0.0, 0.0]); pos[2] += r_   # foot site is at the sole: raise the sphere so it is tangent to the sole (it stood on the sphere otherwise: NONFOOT_CONTACT)
            g = body.add_geom(name=f"eff_{demand.get('effector')}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[r_, 0, 0], pos=pos.tolist(), mass=0.0, group=3, rgba=[1, 0.6, 0.1, 0.6]); g.contype = 1; g.conaffinity = 1
            spec.compile(); open(out, "w").write(spec.to_xml())
        return out
    def plant(self, demand, dt=core.DT_OPT, fresh=False):
        key = (dt, fresh, demand.get("effector"), demand.get("support", "both"), tuple(demand.get("driven") or ()), str(demand["object"]))
        if key in self._plants: return self._plants[key]
        kw = dict(site=demand.get("effector"), ball=False, mass=0.0, support=demand.get("support", "both"), driven=demand.get("driven"), mjcf=self.scene(demand))
        S0 = T.setup(self.robot, **kw); S = T.setup(self.robot, start=np.array(S0["q0"]) + 1e-9*(1 + len(self._plants)), **kw) if fresh else S0
        # the MJX G1 model carries NO collision bits on its body geoms (floor contact comes from explicit pairs): enable the EFFECTOR body's
        # collision-class geoms (hand capsule / foot spheres) so it can hit the object; the rest of the body stays as modelled
        m = S["m"]; hb = int(m.site_bodyid[S["site"]]); n_on = 0
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) == hb and int(m.geom_group[g]) == 3: m.geom_contype[g] = 1; m.geom_conaffinity[g] = 1; n_on += 1
        S["effector_collision_geoms"] = n_on
        BI.apply_dt(S, dt); BI.settle(S); S["_setup_kw"] = kw; S["ball_dof"] = int(S["m"].jnt_dofadr[mujoco.mj_name2id(S["m"], mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]); self._plants[key] = S; return S
    def bare(self, S, demand):
        """The reference and the one-foot hold are built on the BARE robot (objects are kinematically irrelevant to the reference; a
        free ball in the world model made the static certificate's LP infeasible: its gravity has no contact in the LP)."""
        key = ("bare", S["m"].opt.timestep, demand.get("effector"), demand.get("support", "both"), tuple(demand.get("driven") or ()))
        if key not in self._refs:
            kw = dict(site=demand.get("effector"), ball=False, mass=0.0, support=demand.get("support", "both"), driven=demand.get("driven"))
            Sb = T.setup(self.robot, start=np.array(S["q_settled"][:T.setup(self.robot, **kw)["m"].nq]), **kw); BI.apply_dt(Sb, S["m"].opt.timestep); BI.settle(Sb); Sb["_setup_kw"] = kw; self._refs[key] = Sb
        return self._refs[key]
    def ref(self, S, demand):
        dr = demand.get("direction"); key = (id(S["m"]), None if dr is None else tuple(np.round(dr, 4)), demand.get("capacity_mode", "power"))
        if key not in self._refs:
            Sb = self.bare(S, demand); nqb = Sb["m"].nq
            r = T.windup_reference(Sb, direction=(None if dr is None else (float(dr[0]), float(dr[1]))), capacity_mode=demand.get("capacity_mode", "power"))
            pad = np.tile(np.asarray(S["q_settled"])[nqb:], (len(r["Q"]), 1)); r["Q"] = np.hstack([np.asarray(r["Q"]), pad]); r["start"] = np.array(S["q_settled"])
            if r.get("prep_qpos") is not None: r["prep_qpos_bare"] = list(r["prep_qpos"]); r["prep_qpos"] = list(r["prep_qpos"]) + list(np.asarray(S["q_settled"])[nqb:])
            self._refs[key] = r
        return self._refs[key]
    def one_foot_hold(self, S, demand, ref):
        key = ("hold", id(S["m"]), tuple(np.round(ref["prep_qpos"], 4)))
        if key not in self._refs:
            Sb = self.bare(S, demand); kw = Sb["_setup_kw"]
            Sp = T.setup(self.robot, start=np.asarray(ref["prep_qpos_bare"], float), **kw); BI.apply_dt(Sp, S["m"].opt.timestep); BI.settle(Sp)
            self._refs[key] = np.array([Sp["a_hold"][t] for t in S["types"]])
        return self._refs[key]
    def rollout(self, S, demand, p, ctrl_fn=None, record=False):
        # graded miss: the closest approach of the effector to the object over the rollout (a flat miss penalty gave the search no gradient: two seeds returned the initial point)
        m = S["m"]; bb = m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]; sid = S["site"]; nT = len(S["types"]); st = dict(dmin=1e9, f_max=0.0, t_contact=0.0)
        rg = S["robot_geoms"]; dt_ = float(m.opt.timestep)
        def fn2(tt, d, S_):
            if int(round(tt*1000)) % 10 == 0: st["dmin"] = min(st["dmin"], float(np.linalg.norm(d.site_xpos[sid] - d.xpos[bb])))
            F = 0.0
            for ci in range(d.ncon):
                c = d.contact[ci]
                if bb in (int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])) and (c.geom1 in rg or c.geom2 in rg): f6 = np.zeros(6); mujoco.mj_contactForce(m, d, ci, f6); F += abs(float(f6[0]))
            if F > 0.5: st["t_contact"] += dt_; st["f_max"] = max(st["f_max"], F)          # SUSTAINED-FORCE effects: contact duration and peak force on the object
            return ctrl_fn(tt, d, S_) if ctrl_fn is not None else np.zeros(nT)
        rec = super().rollout(S, dict(demand, ignore_cone=True), p, ctrl_fn=fn2, record=record)       # effector metrics, stance/envelope/energy gates; the cone is judged on the ball below
        qpos, qvel = rec["final_state"]
        if demand.get("measure", "speed") == "displacement":                   # push: the object's displacement from its start, in the horizontal plane
            qa0 = int(S["m"].jnt_qposadr[mujoco.mj_name2id(S["m"], mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]); v = np.asarray(qpos[qa0:qa0 + 3]) - np.asarray(S["q_settled"])[qa0:qa0 + 3]; v = np.array([v[0], v[1], 0.0])
        else: v = np.asarray(qvel[S["ball_dof"]:S["ball_dof"] + 3])
        spd = float(np.linalg.norm(v)); horiz = float(np.hypot(v[0], v[1])); elev = float(np.arctan2(v[2], horiz)); az = float(np.arctan2(v[1], v[0]))
        qw, qx, qy, qz = np.asarray(S["q_settled"])[3:7]; yaw0 = float(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))); az_rel = float((az - yaw0 + np.pi) % (2*np.pi) - np.pi)
        ec, ac = (map(float, demand["direction"]) if demand.get("direction") is not None else (0.35, 0.0)); et, at = float(demand.get("elev_tol", 0.35)), float(demand.get("az_tol", 0.35))
        reasons = [r_ for r_ in rec["reasons"] if r_ != "OUT_OF_CONE"]; viol = float(rec["violation"]); hit = spd >= float(demand.get("hit_min", 0.5))
        in_cone = hit and abs(elev - ec) <= et and abs(az_rel - ac) <= at
        if not hit: viol += 1.0 + 2.0*(float(demand.get("hit_min", 0.5)) - spd) + 5.0*st["dmin"]; reasons.append("NOT_HIT")
        elif not in_cone: viol += 1.0 + 20.0*(max(0.0, abs(elev - ec) - et) + max(0.0, abs(az_rel - ac) - at)); reasons.append("BALL_OFF_TARGET")
        # SUSTAINED-FORCE gates (push/lift): a peak contact force cap (a 1.5 kN slap is not a push) and a minimum contact duration
        if demand.get("f_max") is not None and st["f_max"] > float(demand["f_max"]): viol += 1.0 + 0.01*(st["f_max"] - float(demand["f_max"])); reasons.append("IMPULSIVE_CONTACT")
        if demand.get("t_contact_min") is not None and st["t_contact"] < float(demand["t_contact_min"]): viol += 1.0 + 5.0*(float(demand["t_contact_min"]) - st["t_contact"]); reasons.append("CONTACT_TOO_BRIEF")
        out = dict(rec, value=spd, valid=(viol <= 0 and not reasons), reasons=reasons, violation=viol)
        out["extra"] = dict(rec["extra"], ball_speed=spd, ball_elev=elev, ball_az=az_rel, eff_speed=rec["value"], dmin_obj=st["dmin"], f_max_obj=st["f_max"], t_contact=st["t_contact"]); return out
