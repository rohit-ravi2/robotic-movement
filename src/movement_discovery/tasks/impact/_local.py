"""Local helpers for the impact family (2026-10-12). Wraps tasks/impact.py (ImpactTask: object effect, effector
sphere collider, graded miss, sustained-force gates) with ADDITIVE scene and rollout extensions the family needs until WORLD's
factories land (code against the names world.objects.launched_ball / compliant_target / suspended_target / tool):
  object.qvel0   launched object: initial velocity set in the settled state (the object starts in the air where the scene puts it)
  object.soft    compliant target: contact time constant (s) on the object's geom (engine.world's per-geom compliance)
  tool           dict(length, radius, mass): a capsule fixed to the hand body from the palm site, forward along the palm's local z.
                 EXPERIMENTAL SIMPLIFICATION on G1 = a weld hold (no finger grasp; Capability.WELD_GRASP); the hit must be by the tool
  pendulum       dict(pivot, length, mass, radius): a hinge pendulum (suspended target); value = its peak swing energy (J)
  contact_required  a moving object that is never touched is NOT_HIT even though it has speed
MuJoCo only inside methods; the class is built lazily."""
from __future__ import annotations
import os, json, numpy as np
from movement_discovery.core.contracts import TaskProgram, Capability, WorldSpec, Outcome, ValidityReport, Status
from movement_discovery.physics import validity as V
G = 9.80665


def make_task(robot="G1"):
    import mujoco, throw as T, branch_i as BI
    from tasks.impact import ImpactTask
    from tasks.effect import EffectTask
    from engine import core

    class ImpactTaskX(ImpactTask):
        name = "impact_x"
        def scene(self, demand):
            path = super().scene(demand); o = demand["object"]; tool = demand.get("tool"); pend = demand.get("pendulum")
            if not (o.get("soft") or tool or pend): return path
            tag = json.dumps([o.get("soft"), tool, pend], sort_keys=True); import hashlib; out = path.replace(".xml", f"_x{hashlib.sha256(tag.encode()).hexdigest()[:8]}.xml")
            if os.path.exists(out): return out
            spec = mujoco.MjSpec.from_file(path); mb = mujoco.MjModel.from_xml_path(path)
            if o.get("soft"):
                g = [g_ for g_ in spec.geoms if g_.name == "ball_geom"][0]; g.solref = [float(o["soft"]), 1.0]; g.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]; g.priority = 1
            if tool:
                sid = mujoco.mj_name2id(mb, mujoco.mjtObj.mjOBJ_SITE, demand["effector"]); bname = mujoco.mj_id2name(mb, mujoco.mjtObj.mjOBJ_BODY, mb.site_bodyid[sid]); body = spec.body(bname) if hasattr(spec, "body") else spec.find_body(bname)
                R = np.zeros(9); mujoco.mju_quat2Mat(R, mb.site_quat[sid]); ax = R.reshape(3, 3) @ np.array([0.0, 0.0, 1.0]); p0 = np.array(mb.site_pos[sid], float); p1 = p0 + ax*float(tool["length"])
                g = body.add_geom(name="tool_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=list(p0) + list(p1), size=[float(tool["radius"]), 0, 0], mass=float(tool["mass"]), group=3, rgba=[0.3, 0.3, 0.3, 1]); g.contype = 1; g.conaffinity = 1
            if pend:
                pv = spec.worldbody.add_body(name="pendulum_pivot", pos=list(pend["pivot"])); j = pv.add_joint(name="pendulum_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], damping=float(pend.get("damping", 0.01)))
                L_ = float(pend["length"]); rod = pv.add_geom(name="pendulum_rod", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, 0, 0, -L_], size=[0.005, 0, 0], mass=0.02, rgba=[0.4, 0.4, 0.4, 1]); rod.contype = 0; rod.conaffinity = 0
                bob = pv.add_geom(name="pendulum_bob", type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[0, 0, -L_], size=[float(pend["radius"]), 0, 0], mass=float(pend["mass"]), rgba=[0.9, 0.5, 0.1, 1]); bob.solref = [0.002, 1.0]; bob.solimp = [0.99, 0.999, 0.001, 0.5, 2.0]
                for k in spec.keys:
                    k.qpos = list(k.qpos) + [0.0]
                    if len(k.qvel): k.qvel = list(k.qvel) + [0.0]
            spec.compile(); open(out, "w").write(spec.to_xml()); return out
        def plant(self, demand, dt=core.DT_OPT, fresh=False):
            o = demand["object"]; launched = o.get("qvel0") is not None
            S = super().plant(demand, dt=dt, fresh=(fresh or launched))
            if launched and not S.get("_launched"):
                m = S["m"]; j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"); qa = int(m.jnt_qposadr[j]); da = int(m.jnt_dofadr[j])
                S["q_settled"][qa:qa + 3] = np.asarray(o["pos"], float); S["q_settled"][qa + 3:qa + 7] = [1, 0, 0, 0]; S["v_settled"][da:da + 6] = 0.0; S["v_settled"][da:da + 3] = np.asarray(o["qvel0"], float); S["_launched"] = True
            return S
        def rollout(self, S, demand, p, ctrl_fn=None, record=False):
            m = S["m"]; bb = m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]; rg = S["robot_geoms"]; tg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "tool_geom")
            pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pendulum_pivot"); st = dict(hit_tool=False, hit_other=False, E_max=0.0, z_rest=None, n_contact=0); nT = len(S["types"]); v6 = np.zeros(6)
            pend = demand.get("pendulum"); bob_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pendulum_bob")
            def obs(tt, d, S_):
                for ci in range(d.ncon):
                    c = d.contact[ci]; g1, g2 = int(c.geom1), int(c.geom2); b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
                    tgt_hit = (bb in (b1, b2)) or (pend and bob_g in (g1, g2))
                    if tgt_hit and (g1 in rg or g2 in rg):
                        st["n_contact"] += 1
                        if tg in (g1, g2): st["hit_tool"] = True
                        else: st["hit_other"] = True
                if pend and int(round(tt*1000)) % 5 == 0:
                    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, bob_g, v6, 0); z = float(d.geom_xpos[bob_g][2]); st["z_rest"] = z if st["z_rest"] is None else st["z_rest"]
                    st["E_max"] = max(st["E_max"], 0.5*float(pend["mass"])*float(v6[3:] @ v6[3:]) + float(pend["mass"])*G*(z - st["z_rest"]))
                return ctrl_fn(tt, d, S_) if ctrl_fn is not None else np.zeros(nT)
            if pend:
                rec = EffectTask.rollout(self, S, dict(demand, ignore_cone=True), p, ctrl_fn=obs, record=record); reasons = list(rec["reasons"]); viol = float(rec["violation"])
                if st["n_contact"] == 0: reasons.append("NOT_HIT"); viol += 1.0 + 5.0*self._dmin_geom(S, rec, bob_g)
                out = dict(rec, value=st["E_max"], valid=(viol <= 0 and not reasons), reasons=reasons, violation=viol); out["extra"] = dict(rec["extra"], swing_energy=st["E_max"], eff_speed=rec["value"], n_contact=st["n_contact"]); return out
            rec = super().rollout(S, demand, p, ctrl_fn=obs, record=record); reasons = list(rec["reasons"]); viol = float(rec["violation"]); ex = rec["extra"]
            if demand["object"].get("qvel0") is not None and "START_NOT_AT_REST" in reasons and float(np.max(np.abs(np.asarray(S["v_settled"])[S["robot_dofs"][6:]]))) <= 0.05: reasons.remove("START_NOT_AT_REST")   # only the LAUNCHED object moves at t = 0
            if demand.get("contact_required") and st["n_contact"] == 0 and "NOT_HIT" not in reasons:
                if "BALL_OFF_TARGET" in reasons:      # an untouched moving ball is a miss, not an off-target hit: swap the cone term for the graded miss
                    ec, ac = map(float, demand["direction"]); et, at = float(demand.get("elev_tol", 0.35)), float(demand.get("az_tol", 0.35)); reasons.remove("BALL_OFF_TARGET")
                    viol -= 1.0 + 20.0*(max(0.0, abs(ex["ball_elev"] - ec) - et) + max(0.0, abs(ex["ball_az"] - ac) - at))
                reasons.append("NOT_HIT"); viol += 1.0 + 5.0*float(ex.get("dmin_obj", 1.0))
            if demand.get("tool") and st["hit_other"] and not st["hit_tool"]: reasons.append("HIT_WITHOUT_TOOL"); viol += 1.0
            out = dict(rec, valid=(viol <= 0 and not reasons), reasons=reasons, violation=viol); out["extra"] = dict(rec["extra"], hit_tool=st["hit_tool"], hit_other=st["hit_other"], n_contact=st["n_contact"]); return out
        def _dmin_geom(self, S, rec, gid):
            m = S["m"]; d = mujoco.MjData(m); q, v = rec["final_state"]; d.qpos[:] = q; mujoco.mj_forward(m, d); return float(np.linalg.norm(d.site_xpos[S["site"]] - d.geom_xpos[gid]))
    return ImpactTaskX(robot)


class ImpactProgram(TaskProgram):
    """Shared scaffolding: embodiment by the adapter, ImpactTaskX bridge, outcome from the record. Subclasses set `effector`,
    `support`, sample_scenario, demand_extra and the world."""
    effector = "right_palm"; support = "both"; status = Status.SOURCE_IMPLEMENTED
    def embodiment(self, robot):
        from movement_discovery.embodiment import resolve; return resolve(robot, self)
    def object_dict(self, p):
        o = p["object"]; d = dict(name="ball", kind=o.get("kind", "ball"), size=([o["radius"]] if o.get("kind", "ball") == "ball" else list(o["size"])), mass=o["mass"], pos=list(o["pos"]), graspable=False, rgba=[0.9, 0.3, 0.2, 1])
        for k in ("soft", "qvel0"):
            if o.get(k) is not None: d[k] = o[k]
        return d
    def world(self, scenario):
        p = scenario.params; o = p["object"]; surfs = [dict(name="tee", pos=[o["pos"][0], o["pos"][1], 0.5*(o["pos"][2] - o["radius"])], size=[0.015, 0.015, 0.5*(o["pos"][2] - o["radius"])], rgba=[0.6, 0.4, 0.2, 1], rigid=True)] if o.get("tee") else []
        return WorldSpec(objects=[self.object_dict(p)], surfaces=surfs)
    def demand_extra(self, p): return {}
    def demand(self, scenario):
        p = scenario.params; t = make_task(p.get("robot", "G1"))
        d = dict(t.demand_default(), effector=self.effector, support=self.support, direction=tuple(p["direction"]), elev_tol=p.get("elev_tol", np.radians(20.0)), az_tol=p.get("az_tol", np.radians(20.0)), object=dict(p["object"]), hit_min=p.get("hit_min", 0.5)); d.update(self.demand_extra(p)); return t, d
    def evaluate(self, rec, scenario):
        v = V.classify(rec)
        ex = rec["extra"]; keys = ("ball_speed", "ball_elev", "ball_az", "eff_speed", "dmin_obj", "f_max_obj", "t_contact", "t_rel", "hit_tool", "n_contact", "swing_energy", "com_speed", "cp_err_rel", "feet_bw_rel")
        return Outcome(value=float(rec["value"]), achieved=v.ok, measures={k: ex.get(k) for k in keys if k in ex}, validity=v)
    def validity(self, rec, scenario): return self.evaluate(rec, scenario).validity
    def observe(self, rec, scenario):
        if not rec.get("trace"): return None
        from engine import emergence as EM; t, d = self.demand(scenario); return EM.analyze(t.plant(d), rec, d["effector"])
