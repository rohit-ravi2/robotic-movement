"""CatchTask: tasks/impact.py ImpactTask turned toward INTERCEPTION. The object is a WORLD object with an initial velocity
(objects.launched_ball / rolling_ball / a sliding box: qvel0 applied to the plant's settled state, so the ball rests on a pedestal
during the settle and is launched at t = 0 with the recorded energy), the effector sphere collider(s) (world.colliders) receive it, and the
record carries: relative speed hand<->object at the end, closest approach, per-hand contact at the end, peak contact force, contact
duration (ImpactTask's sustained-force gates), captured = contact at the end with relative speed <= rel_speed_max (G1) or >= 2 Dex3
hand links in contact (G1H, fingers close when the object comes within close_dist of the palm: an EVENT, not a schedule).
Disturbances (dicts kind/t0/t1/vec/body, world.disturbances schema) are applied through physics.disturbance.apply_power (xfrc_applied;
the wrenches' power is integrated into W_ext for the energy re-judging). Effector colliders: world.colliders. Horizon = throw.HOR (0.8 s)."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import WorldSpec


def make(robot, spec: WorldSpec, colliders=("right_palm",), effector="right_palm"):
    import mujoco
    from engine import core, world as W
    from tasks.impact import ImpactTask
    from movement_discovery.world import apply_initial_velocities, colliders as CL
    from movement_discovery.physics import disturbance as D
    obj = spec.objects[0]; assert obj["name"] == "ball", "ImpactTask reads the object as joint 'ball_free'"

    class CatchTask(ImpactTask):
        name = "catch"
        def scene(self, demand): return CL.scene(spec, robot, colliders)
        def plant(self, demand, dt=core.DT_OPT, fresh=False):
            S = super().plant(demand, dt, fresh)
            if not S.get("_launched"):
                m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; d.qvel[:] = S["v_settled"]; apply_initial_velocities(m, d, spec); S["v_settled"] = d.qvel.copy(); S["_launched"] = True
                for s_ in colliders[1:]: CL.enable_effector_geoms(S, s_)
                S["_hand_bodies"] = {s_: int(m.site_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s_)]) for s_ in colliders}
                S["_wd"] = W.World(m, [dict(obj, graspable=False)], [], robot)
                S["_fingers"] = {a: W.FINGER_CLOSE[n] for n, a in S["_wd"].fingers.items()}
            return S
        def rollout(self, S, demand, p, ctrl_fn=None, record=False):
            m = S["m"]; bb = int(m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]); sid = S["site"]; dt_ = float(m.opt.timestep); every = max(1, int(round(0.002/dt_)))
            st = dict(t_last_contact=-1.0, hands_end=set(), links_end=0, rel_min=1e9, closed_at=None, F_end=0.0, W_ext=0.0); hb = S["_hand_bodies"]; dists = demand.get("disturbances"); wd = S["_wd"]
            def hook(tt, d, S_):
                if dists: st["W_ext"] += D.apply_power(dists, m, d, tt)*dt_
                if int(round(tt/dt_)) % every == 0:
                    hands = set(); F = 0.0
                    for ci in range(d.ncon):
                        c = d.contact[ci]; b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                        if bb in (b1, b2):
                            o = b2 if b1 == bb else b1
                            for s_, hb_ in hb.items():
                                if o == hb_ or o in wd.hand_bodies: hands.add(s_ if o == hb_ else "fingers"); f6 = np.zeros(6); mujoco.mj_contactForce(m, d, ci, f6); F += abs(float(f6[0]))
                    if hands: st["t_last_contact"] = tt; st["hands_end"] = hands; st["F_end"] = F
                    vel6 = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, sid, vel6, 0); vb = d.qvel[S["ball_dof"]:S["ball_dof"] + 3]
                    st["rel_min"] = min(st["rel_min"], float(np.linalg.norm(vb - vel6[3:])))
                    if S["_fingers"] and st["closed_at"] is None and np.linalg.norm(d.xpos[bb] - d.site_xpos[sid]) <= float(demand.get("close_dist", 0.12)): st["closed_at"] = tt
                    if st["closed_at"] is not None:
                        for a, q in S["_fingers"].items(): d.ctrl[a] = q
                    if S["_fingers"] and st["closed_at"] is not None: st["links_end"] = len(wd.hand_contacts(d, "ball")[0])
                return ctrl_fn(tt, d, S_) if ctrl_fn is not None else np.zeros(len(S["types"]))
            rec = super().rollout(S, dict(demand, hit_min=0.0, elev_tol=np.pi, az_tol=np.pi), p, ctrl_fn=hook, record=record)
            qpos, qvel = rec["final_state"]; d = mujoco.MjData(m); d.qpos[:] = qpos; d.qvel[:] = qvel; mujoco.mj_forward(m, d)
            vel6 = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, sid, vel6, 0); vb = np.asarray(qvel[S["ball_dof"]:S["ball_dof"] + 3]); rel = float(np.linalg.norm(vb - vel6[3:]))
            t_end = float(rec["trace"]["release_time"]) if rec.get("trace") else None; in_contact = st["t_last_contact"] >= 0 and (t_end is None or t_end - st["t_last_contact"] <= 0.01)
            n_hands = int(demand.get("hands", 1)); rmax = float(demand.get("rel_speed_max", 0.5))
            captured = bool(in_contact and (st["links_end"] >= 2 if S["_fingers"] else (len([h for h in st["hands_end"] if h != "fingers"]) >= n_hands and rel <= rmax)))
            reasons = [r for r in rec["reasons"] if r not in ("NOT_HIT", "BALL_OFF_TARGET")]; viol = float(rec["violation"])
            a0 = S["ball_dof"]; v0 = np.asarray(S["v_settled"]); robot_rest = float(np.max(np.abs(np.concatenate([v0[6:a0], v0[a0 + 6:]])))) <= 0.05 if len(v0) > a0 + 6 else float(np.max(np.abs(v0[6:a0]))) <= 0.05
            if robot_rest: reasons = [r for r in reasons if r != "START_NOT_AT_REST"]      # the launched OBJECT moves at t = 0 by design; the robot's own dofs are at rest (throw.penalty adds no term for this reason)
            energy_rejudged = False
            if dists and abs(st["W_ext"]) > 0 and any(r in reasons for r in ("ENERGY_ACCOUNTING_VIOLATION", "ENERGY_CLOSURE")):
                # the reference gate credits actuator work only; the external wrench's work W_ext is known here. Re-judged, LABELLED: the
                # residual is attributed to W_ext (not re-measured: E0/E1 are not in the record); throw.penalty's +2 for the energy reasons is removed
                reasons = [r for r in reasons if r not in ("ENERGY_ACCOUNTING_VIOLATION", "ENERGY_CLOSURE")]; viol = max(0.0, viol - 2.0); energy_rejudged = True
            if not captured: reasons.append("NOT_CAPTURED"); viol += 1.0 + 2.0*rel + 5.0*float(rec["extra"]["dmin_obj"])
            ex = dict(rec["extra"], rel_speed_end=rel, rel_speed_min=st["rel_min"], captured=captured, hands_end=sorted(st["hands_end"]), links_end=st["links_end"], t_last_contact=st["t_last_contact"], closed_at=st["closed_at"], ball_pos_end=np.asarray(qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'ball_free')]:][:3]).tolist(), ball_speed_end=float(np.linalg.norm(vb)), W_ext=st["W_ext"], energy_rejudged=energy_rejudged)
            return dict(rec, value=-(rel + 2.0*float(rec["extra"]["dmin_obj"])), valid=(viol <= 0 and not reasons), reasons=reasons, violation=viol, extra=ex)
    return CatchTask(robot)


def demand(task, effector="right_palm", support="both", **kw):
    d = dict(task.demand_default(), effector=effector, support=support, direction=(0.0, 0.0), object=dict(pos=[1.5, -0.2, 1.0], radius=0.05, mass=0.1, tee=True), rel_speed_max=0.5, hands=1); d.update(kw); return d


def pedestal(o):
    """Static slab under a launched object so it rests during the plant's settle (engine.world.surface dict)."""
    from engine import world as W
    r = o["size"][0] if o["kind"] == "ball" else o["size"][2]; z = o["pos"][2] - r
    return W.surface("pedestal", pos=(o["pos"][0], o["pos"][1], z/2), size=(0.06, 0.06, z/2))


class CatchProgram:
    """Mixin: evaluate/validity/observe over CatchTask records."""
    def embodiment(self, robot):
        from movement_discovery import embodiment as E; return E.resolve(robot, self)
    def strategy_space(self, scenario): return ["torque_knots+tracked_prior", "torque_knots"]
    def measures(self, rec):
        from movement_discovery.tasks.force_interaction import _base as B
        e = rec["extra"]; return B.impact_measures(rec) | {k: e.get(k) for k in ("rel_speed_end", "rel_speed_min", "captured", "hands_end", "links_end", "t_last_contact", "closed_at", "ball_speed_end", "ball_pos_end", "W_ext", "energy_rejudged")}
    def evaluate(self, rec, scenario):
        from movement_discovery.tasks.force_interaction import _base as B
        e = rec["extra"]; return B.outcome(rec, value=rec["value"], measures=self.measures(rec), achieved=bool(e.get("captured")))
    def validity(self, rec, scenario): return self.evaluate(rec, scenario).validity
    def observe(self, rec, scenario):
        from movement_discovery.tasks.force_interaction import _base as B
        t, d = self.demand(scenario); return B.observe_impact(t, d, rec, d["effector"])
