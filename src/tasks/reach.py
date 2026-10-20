"""J5: reach-with-load as an engine Task. Demand: {"target": [dx, dy, dz] (m, relative to the standing palm position),
"load": kg at the palm, "tol": 0.03, "hold": 0.20, "prior": False|"ascent"}.
Feasible iff the palm comes within tol of the target while both feet stay planted (Phase 11 stance rule), the validity
gates pass, and the palm then stays within 2*tol for `hold` seconds (a reach, not a fling-through).
value = -positive actuator work (J) (higher = cheaper); the demand sweep over height/distance/load is where the strategy
switch (arm-only / trunk bend / squat) should emerge. Plant = the throw plant (per-joint channels, right palm site) with
the ball replaced by the payload."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco, branch_i as BI, throw as T, plant as P
from validity import TrajectoryValidityReport
from engine import core
from engine.task import Task
G = 9.80665; HOR = 1.5; NK = 10


class ReachTask(Task):
    name = "reach"
    def __init__(self, robot="G1", scene=None, objects=(), surfaces=()):
        """scene: an engine.world scene path (objects + surfaces present, no merged payload); None = bare robot with a
        merged payload at the palm (demand 'load')."""
        self.robot = robot; self._plants = {}; self._palm0 = None; self.scene = scene; self.objects = list(objects); self.surfaces = list(surfaces)
    def demand_default(self): return dict(target=[0.30, 0.0, 0.0], load=0.0, tol=0.03, hold=0.20, prior="ascent")
    def sigma0(self, demand, warm): return 0.05 if demand.get("prior") else 0.35
    def world(self, S):
        from engine import world as W
        if "world" not in S: S["world"] = W.World(S["m"], self.objects, self.surfaces, self.robot)
        return S["world"]
    def plant(self, demand, dt=core.DT_OPT, fresh=False):
        load = float(demand.get("load", 0.0)) if self.scene is None else 0.0; key = (dt, load, fresh)
        if key in self._plants: return self._plants[key]
        if self.scene is None:
            S0 = T.setup(self.robot); S = T.setup(self.robot, start=np.array(S0["q0"]) + 1e-9*(1 + len(self._plants)))   # uncached copy
            m = S["m"]; hb = S["hand_body"]; sid = S["site"]; mb = float(m.body_mass[hb]); p = np.array(m.site_pos[sid])
            extra = load - T.BALL                                       # the throw plant carries the 0.145 kg ball; swap for the payload
            m.body_ipos[hb] = (mb*m.body_ipos[hb] + extra*p)/max(mb + extra, 1e-6); m.body_mass[hb] = max(mb + extra, 1e-6)
            mujoco.mj_setConst(m, mujoco.MjData(m))
        else:
            S0 = T.setup(self.robot, mjcf=self.scene, ball=False)
            S = T.setup(self.robot, mjcf=self.scene, ball=False, start=np.array(S0["q0"]) + 1e-9*(1 + len(self._plants)))
            m = S["m"]; sid = S["site"]
            if BI.ROBOTS[self.robot].get("settle_feedforward"):
                # certified holding torques from the BARE robot (S0 above is the scene plant): the whole-model certificate is
                # infeasible with a free object resting on furniture; the bare robot's torques apply (same actuator order)
                Sb = T.setup(self.robot, ball=False)
                if Sb.get("settle_cert_tau") is None:
                    m0 = Sb["m"]; d0 = mujoco.MjData(m0); d0.qpos[:] = Sb["q0"]; mujoco.mj_forward(m0, d0); rb0 = Sb["rb"]; rb0.d.qpos[:] = Sb["q0"]; mujoco.mj_forward(m0, rb0.d); d0.qpos[2] -= rb0.lowest_z(); mujoco.mj_forward(m0, d0)
                    cg0 = P.PlantContractVerifier(m0, d0, P.PlantContract(Sb["mjcf"], "torque")).collision_geoms(); c0 = P.static_certificate(m0, mujoco.MjData(m0), d0.qpos.copy(), coll_geoms=cg0)
                    if c0["status"] == "STATIC_FEASIBLE": Sb["settle_cert_tau"] = np.asarray(c0["tau"], float)
                if Sb.get("settle_cert_tau") is not None: S["settle_tau_ff"] = np.asarray(Sb["settle_cert_tau"])
        BI.apply_dt(S, dt); BI.settle(S); self._plants[key] = S
        S["q_stance0"] = np.array(S["q_settled"])        # the robot's NOMINAL stance: posture caps (trunk pitch, squat) are measured from it across handoffs, never from a handed-off pose (caps accumulated capability by capability otherwise: the pick bent the trunk 0.35, the wipe then bent it another 0.35)
        if self._palm0 is None:
            d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d); self._palm0 = d.site_xpos[sid].copy()
        return S
    def n_params(self, S, demand): return len(S["types"])*NK + (1 if demand.get("prior") else 0)
    def unpack(self, S, demand, z):
        z = np.asarray(z); n = len(S["types"])*NK
        return dict(x=z[:n], tscale=(0.87 + 0.3*float(z[n])) if demand.get("prior") else 1.0)
    def ref(self, S, demand):
        """Reference with optional AIM-OFF refinement (demand["refine"] = n): replay the reference once in pure tracking, measure the
        palm's end error, shift the planning target by -0.7 x error, rebuild (the extended arm sags 5-6 cm under gravity on the
        hands robot; the task gate is on the TRUE target)."""
        n_ref = int(demand.get("refine", 0)); aim = np.zeros(3); key0 = self._ref_key(S, demand)
        if not hasattr(self, "_refs"): self._refs = {}
        if key0 in self._refs: return self._refs[key0]
        for it in range(n_ref + 1):
            r = self._build_ref(S, demand, aim)
            if it == n_ref: break
            self._refs[key0] = r; p0 = dict(x=np.zeros(len(S["types"])*NK), tscale=1.0); rec = self.rollout(S, dict(demand, refine=0, grasp_on_reach=None), p0)
            err = np.asarray(rec["extra"].get("palm_hold", self.target_world(S, demand))) - self.target_world(S, demand)
            if not demand.get("aim_z", False): err[2] = 0.0                                              # xy only: a downward z aim drove the descent through the tool
            aim = np.clip(aim - 0.8*np.clip(err, -0.08, 0.08), -0.05, 0.05)   # error at the SETTLED hold after arrival; TOTAL aim capped at 5 cm (an uncapped aim diverged 19 cm at the reach limit)
        r["aim"] = aim.tolist(); self._refs[key0] = r
        return r
    def _ref_key(self, S, demand):
        H = float(demand.get("horizon", HOR))
        return (tuple(np.round(self.target_world(S, demand), 4)), float(demand.get("load", 0.0)), H, demand.get("trunk_cap"), demand.get("approach"), float(demand.get("clearance", 0.06)), json.dumps(demand.get("orient"), sort_keys=True) if demand.get("orient") else None, int(demand.get("refine", 0)), json.dumps([demand.get("via_offset"), demand.get("object_point"), demand.get("offset")]), np.round(np.asarray(S["q_settled"])[S["jarr"]["qad"]], 3).tobytes())   # start-dependent (handoff)
    def _build_ref(self, S, demand, aim):
        H = float(demand.get("horizon", HOR)); key = self._ref_key(S, demand)
        if True:
            from engine import strategy as ST
            if self.scene is None:
                return ST.reach_prior(S, self.target_world(S, demand) + aim, horizon=H, nk=NK, tol=min(0.01, 0.5*float(demand.get("tol", 0.03))), squat_if_short=float(demand.get("squat", 0.25)), trunk_cap=demand.get("trunk_cap"))
            else:
                # the prior (static certificate + constrained Jacobian) is computed on the BARE robot from the same robot
                # pose: objects are kinematically irrelevant to the reference and a welded object has no contact for the
                # certificate; the object columns are padded with their current (handed-off) values for the tracker
                Sb = T.setup(self.robot, ball=False); BI.apply_dt(Sb, core.DT_OPT); BI.settle(Sb)
                nqb = Sb["m"].nq; q = self.wrist_pose_for_orient(S, np.asarray(S["q_settled"]), demand); v = np.asarray(S["v_settled"]); tgt = self.target_world(S, demand) + aim
                via = self.approach_via(S, demand, tgt)
                with core.handoff(Sb, q[:nqb], v[:Sb["m"].nv]):
                    if via is None:
                        ref = ST.reach_prior(Sb, tgt, horizon=H, nk=NK, tol=min(0.01, 0.5*float(demand.get("tol", 0.03))), squat_if_short=float(demand.get("squat", 0.25)), trunk_cap=demand.get("trunk_cap"), cap_ref=np.asarray(S["q_stance0"])[:nqb])
                    else:
                        # APPROACH FROM ABOVE: an object resting on a surface is reached through a via point above it (the
                        # straight prior from the standing palm swept the hand into the slab edge and the robot fell)
                        segs = []; q_cur = q[:nqb].copy(); misses = []; legs = False
                        d_p = mujoco.MjData(Sb["m"]); d_p.qpos[:] = q[:nqb]; mujoco.mj_forward(Sb["m"], d_p); palm_now = d_p.site_xpos[Sb["site"]].copy()
                        # hand robots: LIFT straight up first (to the via height) so the hanging fingers do not sweep across objects on the way to the via point
                        pts_chain = ([np.array([palm_now[0], palm_now[1], via[2]])] if (bool(getattr(self.world(S), "fingers", None)) and via[2] > palm_now[2] + 0.01) else []) + [via, tgt]
                        arm_q = np.asarray(S["jarr"]["qad"])[[i for i, g in enumerate(S["groups"]) if g in ("shoulder", "elbow", "wrist")]]
                        for p_ in pts_chain:
                            Sb["q_settled"] = q_cur
                            r = ST.reach_prior(Sb, p_, horizon=1.0, nk=NK, t_ref=0.6, tol=min(0.01, 0.5*float(demand.get("tol", 0.03))), squat_if_short=float(demand.get("squat", 0.25)), trunk_cap=demand.get("trunk_cap"), cap_ref=np.asarray(S["q_stance0"])[:nqb])
                            i_end = int(0.6*r["fps"]); seg = np.array(r["Q"][:i_end + 1])
                            if demand.get("orient"):
                                # every waypoint (lift, via, target) ends ORIENTED: full-arm IK at the waypoint, blended over the segment's
                                # last 60% (straightening the hand beside the handle swept it off the counter)
                                q_full = np.concatenate([seg[-1], q[nqb:]]); q_ik, _, _ = self.arm_ik(S, q_full, p_, demand); i0 = int(0.4*(len(seg) - 1))
                                for i in range(i0, len(seg)): w = (i - i0)/max(len(seg) - 1 - i0, 1); seg[i, arm_q] = (1 - w)*seg[i, arm_q] + w*q_ik[arm_q]
                            segs.append(seg); q_cur = seg[-1].copy(); misses.append(r["miss"]); legs = legs or r["legs"]
                        dwell = float(demand.get("via_dwell", 0.6 if bool(getattr(self.world(S), "fingers", None)) else 0.0))
                        if dwell > 0 and len(segs) >= 2: segs.insert(len(segs) - 1, np.repeat(segs[-2][-1][None], 2, 0))    # DWELL at the via: the lateral tracking lag settles before the vertical descent (a finger swept the handle while still catching up in x)
                        fps = 100; n = int(H*fps) + 1; t = np.arange(n)/fps; T_ = 0.6*H
                        base_T = ([0.3*T_, 0.35*T_, 0.35*T_] if len(segs) - (1 if dwell > 0 else 0) == 3 else [0.6*T_, 0.4*T_])
                        seg_T = (base_T[:-1] + [dwell] + base_T[-1:]) if dwell > 0 else base_T
                        if dwell > 0: sc = (T_ - dwell)/T_; seg_T = [x*sc for x in seg_T[:-2]] + [dwell] + [seg_T[-1]*sc]   # keep the total reference time T_
                        bounds = np.cumsum([0] + seg_T); Qb = np.repeat(q[None, :nqb], n, 0)
                        for i, ti in enumerate(t):
                            k = min(int(np.searchsorted(bounds, ti, side="right")) - 1, len(segs) - 1); w = min(1.0, (ti - bounds[k])/seg_T[k]) if ti < T_ else 1.0
                            seg = segs[k]; Qb[i] = seg[min(int(w*(len(seg) - 1)), len(seg) - 1)]
                        ref = dict(key="reach_via_prior", Q=Qb, fps=fps, takeoff_ref=T_, horizon=H, nk=NK, path_steps=sum(len(s_) for s_ in segs), legs=legs, miss=misses[-1], misses=misses, via=np.asarray(via).tolist())
                Q = np.hstack([ref["Q"], np.tile(q[nqb:], (len(ref["Q"]), 1))])
                if demand.get("orient"):
                    # END-POSE IK: the reference's last frame is replaced by a full-arm solution (position + orientation), blended in
                    # over the final 40% of the reach so the tracked hand arrives at the target WITH the requested orientation
                    q_ik, pe, oe = self.arm_ik(S, Q[-1], tgt, demand); i_end = len(Q) - 1; i0 = int(0.6*i_end)
                    arm_q = np.asarray(S["jarr"]["qad"])[[i for i, g in enumerate(S["groups"]) if g in ("shoulder", "elbow", "wrist")]]
                    for i in range(i0, i_end + 1): w = (i - i0)/max(i_end - i0, 1); Q[i, arm_q] = (1 - w)*Q[i, arm_q] + w*q_ik[arm_q]
                    ref["ik_pos_err"] = pe; ref["ik_ori_err"] = oe
                q_act = np.asarray(S["q_settled"]); wq = np.asarray(S["jarr"]["qad"])[[i for i, g in enumerate(S["groups"]) if g == "wrist"]]
                if len(wq) and np.max(np.abs(q[wq] - q_act[wq])) > 1e-6:          # wrist pre-orientation: ramp from the actual wrist angles over 0.3 s
                    nr = min(int(0.3*ref["fps"]), len(Q) - 1)
                    for i in range(nr + 1): w = i/max(nr, 1); Q[i, wq] = (1 - w)*q_act[wq] + w*Q[i, wq]
                return dict(ref, Q=Q, start=q_act.copy())
    def wrist_pose_for_orient(self, S, q, demand, iters=30):
        """Solve the wrist joints (Gauss-Newton on the rotational Jacobian of the palm site) so the demand["orient"] axes hold at
        configuration q; returns the modified q. The reach reference is then planned from the ORIENTED wrist (a reference planned
        with the un-rotated wrist left the palm 10 cm off once the orientation loop had turned the hand)."""
        ori = demand.get("orient")
        if not ori: return q
        m = S["m"]; sid = S["site"]; wrist = [i for i, g in enumerate(S["groups"]) if g == "wrist"]
        if not wrist: return q
        J = S["jarr"]; wd_ = np.asarray(J["dof"])[wrist]; wq = np.asarray(J["qad"])[wrist]; wj = [int(m.dof_jntid[int(d_)]) for d_ in wd_]
        q = np.array(q, float); d = mujoco.MjData(m); jacr = np.zeros((3, m.nv)); jacp = np.zeros((3, m.nv)); wd0 = self.world(S) if self.scene else None
        items = []
        for it in ori:
            bname = it.get("body", "palm"); bid = m.site_bodyid[sid] if bname == "palm" else (wd0.bid[bname] if (wd0 is not None and bname in wd0.bid) else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname))
            items.append((int(bid), np.asarray(it.get("axis", [1, 0, 0]), float), np.asarray(it.get("target", [0, 0, -1]), float), float(it.get("k", 4.0))))
        for _ in range(iters):
            d.qpos[:] = q; mujoco.mj_forward(m, d); e = np.zeros(3)
            for (bid, a_l, a_t, k) in items: e += k*np.cross(d.xmat[bid].reshape(3, 3) @ a_l, a_t)
            if np.linalg.norm(e) < 1e-3: break
            mujoco.mj_jacSite(m, d, jacp, jacr, sid); Jw = jacr[:, wd_]
            dq = np.linalg.lstsq(Jw, 0.5*e, rcond=None)[0]
            for k_, (qa_, j_) in enumerate(zip(wq, wj)): q[qa_] = float(np.clip(q[qa_] + dq[k_], m.jnt_range[j_][0], m.jnt_range[j_][1]))
        return q
    def arm_ik(self, S, q0, target, demand, iters=60):
        """Full-arm IK (shoulder, elbow, wrist: position + demand["orient"] axes) by damped Gauss-Newton from q0, legs/trunk fixed.
        Used to make the END of a reach reference kinematically consistent with a requested hand orientation (the wrist alone
        cannot turn the palm down from the ready stance; it needs the shoulder and elbow too)."""
        m = S["m"]; sid = S["site"]; J = S["jarr"]; arm = [i for i, g in enumerate(S["groups"]) if g in ("shoulder", "elbow", "wrist")]
        dofs = np.asarray(J["dof"])[arm]; qad = np.asarray(J["qad"])[arm]; jids = [int(m.dof_jntid[int(d_)]) for d_ in dofs]
        q = np.array(q0, float); d = mujoco.MjData(m); jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv)); wd0 = self.world(S) if self.scene else None
        items = []
        for it in (demand.get("orient") or []):
            bname = it.get("body", "palm"); bid = m.site_bodyid[sid] if bname == "palm" else (wd0.bid[bname] if (wd0 is not None and bname in wd0.bid) else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname))
            items.append((int(bid), np.asarray(it.get("axis", [1, 0, 0]), float), np.asarray(it.get("target", [0, 0, -1]), float), float(it.get("k", 4.0))))
        r_pt = np.asarray(demand.get("ik_point", [0, 0, 0]), float)   # controlled point, PALM frame (a held tool's head: the IK then places the tool, through the hand)
        def ctrl_point(d_): return d_.site_xpos[sid] + d_.site_xmat[sid].reshape(3, 3) @ r_pt
        for _ in range(iters):
            d.qpos[:] = q; mujoco.mj_forward(m, d); ep = np.asarray(target, float) - ctrl_point(d); er = np.zeros(3)
            for (bid, a_l, a_t, k) in items: er += np.cross(d.xmat[bid].reshape(3, 3) @ a_l, a_t)
            if np.linalg.norm(ep) < 2e-3 and np.linalg.norm(er) < 2e-2: break
            mujoco.mj_jacSite(m, d, jacp, jacr, sid); rw = d.site_xmat[sid].reshape(3, 3) @ r_pt; Sk = np.array([[0, -rw[2], rw[1]], [rw[2], 0, -rw[0]], [-rw[1], rw[0], 0]])
            A = np.vstack([jacp[:, dofs] - Sk @ jacr[:, dofs], 0.3*jacr[:, dofs]]); b = np.concatenate([ep, 0.3*er])
            q_mid = np.array([0.5*(m.jnt_range[j_][0] + m.jnt_range[j_][1]) for j_ in jids]); w_mid = float(demand.get("ik_centering", 0.003))
            dq = np.linalg.solve(A.T @ A + (1e-3 + w_mid)*np.eye(len(dofs)), A.T @ b + w_mid*(q_mid - q[qad]))*0.7      # damped LS with a weak pull toward joint mid-ranges (keeps the wrist off its limits)
            for k_, (qa_, j_) in enumerate(zip(qad, jids)): q[qa_] = float(np.clip(q[qa_] + dq[k_], m.jnt_range[j_][0] + 0.02, m.jnt_range[j_][1] - 0.02))
        d.qpos[:] = q; mujoco.mj_forward(m, d); er = np.zeros(3)
        for (bid, a_l, a_t, k) in items: er += np.cross(d.xmat[bid].reshape(3, 3) @ a_l, a_t)
        return q, float(np.linalg.norm(np.asarray(target) - ctrl_point(d))), float(np.linalg.norm(er))
    def approach_via(self, S, demand, tgt):
        """Via point above a target that sits on a surface (object within 6 cm above a slab top and inside its xy extent):
        demand["approach"] = "above" | "direct" | None (auto). Returns None for a direct reach."""
        mode = demand.get("approach", None)
        if mode == "side": return np.asarray(tgt, float) + np.asarray(demand.get("via_offset", [-0.10, 0, 0]), float)   # SIDE approach (J27 wrap): via = target + via_offset (world), then a straight slide onto the target
        if mode == "direct" or not self.surfaces: return None
        clear = float(demand.get("clearance", 0.06)); tgt = np.asarray(tgt, float)
        for s_ in self.surfaces:
            top = s_["pos"][2] + s_["size"][2]; inside = all(abs(tgt[i] - s_["pos"][i]) <= s_["size"][i] + 0.02 for i in (0, 1))
            low = self._palm0 is not None and tgt[2] < self._palm0[2] + 0.02      # auto only when reaching DOWN toward the slab (a target above the standing palm is reached directly, as the pick-place was built)
            fingers = bool(getattr(self.world(S), "fingers", None)); hand_len = 0.10 if fingers else 0.0   # Dex3 fingers extend ~8.5 cm beyond the palm site: the HAND must clear the slab, not just the palm
            if inside and (mode == "above" or fingers or (low and -0.01 <= tgt[2] - top <= 0.06)): return np.array([tgt[0], tgt[1], max(tgt[2] + 0.08, top + hand_len + clear)])   # the via is a real descent above the grasp height (a tall tool's grip point can sit above slab + finger length)
        return None
    def controller(self, S, demand, p):
        return None            # servo mode: the reference enters through the servo targets, no torque tracker
    def target_world(self, S, demand):
        """'target_world' = absolute point; 'target_object' = an object's current position (+ 'offset'); else 'target' relative to the standing palm."""
        self.plant(demand)
        if demand.get("target_world") is not None: return np.asarray(demand["target_world"], float)
        if demand.get("target_object"):
            d = mujoco.MjData(S["m"]); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(S["m"], d); wd = self.world(S)
            pt = np.zeros(3) if demand.get("object_point") is None else d.xmat[wd.bid[demand["target_object"]]].reshape(3, 3) @ np.asarray(demand["object_point"], float)   # a point fixed in the OBJECT frame (e.g. a brush's mid-handle)
            return wd.pos(d, demand["target_object"]) + pt + np.asarray(demand.get("offset", [0, 0, 0]), float)
        return self._palm0 + np.asarray(demand["target"], float)
    def rollout(self, S, demand, p, ctrl_fn=None, record=False):
        """QUASI-STATIC MODE: the driven joints are torque-limited POSITION SERVOS (physical stiffness omega_s) tracking the
        prior path + knot OFFSETS (rad). Gravity, holding and stopping are then properties of the servo, not of a torque
        schedule with a lag: a reach ends at rest by construction and any posture can be handed off. (The torque-knot reach
        of v1 reached but could not be chained: it sagged or fell at every handoff.) Jumps and throws keep torque mode."""
        m = S["m"]; d = mujoco.MjData(m); BI.settle(S); BI._set_capacity_limits(m, S, None)
        m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
        d.qpos[:] = S["q_settled"]; d.qvel[:] = S["v_settled"]; mujoco.mj_forward(m, d)
        rd = S.get("robot_dofs"); rj = [i for i in (rd if rd is not None else range(m.nv)) if i >= 6]
        E0 = float(d.energy[0] + d.energy[1]); start_qdot = float(np.max(np.abs(d.qvel[rj])))
        nT = len(S["types"]); kn = np.asarray(p["x"]).reshape(nT, -1)*0.30; NKl = kn.shape[1]          # knots = joint offsets in [-0.3, 0.3] rad
        H = float(demand.get("horizon", HOR)); dt = m.opt.timestep; nst = int(round(H/dt)); J = S["jarr"]; tau_v = J["tau"]; knot_dt = H/NKl
        ref = self.ref(S, demand) if demand.get("prior") else None; ts = p.get("tscale", 1.0); fps = ref["fps"] if ref else 100
        q_s = np.asarray(S["q_settled"]); qa = J["qad"]
        # LEGS: the Phase 14 balance controller in stand mode (capture point over the feet, trunk vertical) owns every leg joint;
        # ARM + TRUNK: servos track the prior path + knot offsets. Open-loop leg targets from the prior tipped the robot (v2a).
        # stand pose = the NOMINAL stance (q_stance0), not the handed-off pose: the controller's capture-point and trunk-tilt
        # corrections are absolute offsets on the stand pose, and a handed-off pose already contains them (the previous step's
        # targets). Using it as the stand pose applied them twice: a 0.23 rad hip-pitch jump in 10 ms that threw the held brush (J27)
        q_st0 = np.asarray(S.get("q_stance0", S["q_settled"]))
        ctl = core.landing_controller(S, self.robot)[0]; ctl.p["mode"] = "stand"; ctl.t_touch = -1e9; ctl.q_stand = q_st0.copy(); ctl.q_land = q_st0.copy(); ctl._kc = None
        # quasi-static stand gains: the landing-tuned capture-point gain hops sideways when the arm shifts the COM slowly
        ctl.p["k_cp"] = float(demand.get("k_cp", 0.25)); ctl.p["k_trunk"] = float(demand.get("k_trunk", 0.5))
        # balance reference = the robot's OWN stance capture point (static COM projection of the nominal stance), not the
        # geometric feet centre; identical to within ~1 cm on G1, 8 cm apart on H1 (heel-margin stance)
        import balance as B_; feet_pts = B_.foot_points(S); d_st = mujoco.MjData(m); d_st.qpos[:] = S.get("q_stance0", S["q_settled"]); mujoco.mj_forward(m, d_st)
        cp_ref_off = d_st.subtree_com[0][:2] - 0.5*(B_.foot_xy(d_st, feet_pts[0]) + B_.foot_xy(d_st, feet_pts[1]))
        if demand.get("cp_ref", "stance") == "stance": ctl.cp_offset = cp_ref_off.copy()
        else: cp_ref_off = np.zeros(2)
        if "k_trunk_d" in demand: ctl.p["k_trunk_d"] = float(demand["k_trunk_d"])     # trunk-rate damping (lateral rocking under stiff leg servos: J15d)
        if "k_cp_d" in demand: ctl.p["k_cp_d"] = float(demand["k_cp_d"])
        task_mask = np.array([g in ("shoulder", "elbow", "wrist", "counter_arm", "trunk") for g in S["groups"]]); task_aid = J["aid"][task_mask]
        jlo = np.array([m.jnt_range[int(m.dof_jntid[int(dof)])][0] + 0.01 for dof in J["dof"]]); jhi = np.array([m.jnt_range[int(m.dof_jntid[int(dof)])][1] - 0.01 for dof in J["dof"]])
        omega_s = float(demand.get("omega_servo", 60.0)); omega_legs = float(demand.get("omega_legs", ctl.p["stiff"]))   # legs at the balance controller's TUNED stiffness (368 rad/s) unless the demand says otherwise; 60 collapsed
        ieff = np.array([1.0/float(S["Minv_key"][dof, dof]) for dof in J["dof"]])
        for i, aid in enumerate(J["aid"]):
            om = omega_s if task_mask[i] else omega_legs; kp = om*om*ieff[i]; BI._as_servo(m, int(aid), kp, 2.0*np.sqrt(kp*ieff[i]), float(tau_v[i]))
        for (a_, dof, qad, tau, kp, kd) in S["held"]: d.ctrl[a_] = S["q_settled"][qad]
        tgt = self.target_world(S, demand); tol = float(demand.get("tol", 0.03)); hold = float(demand.get("hold", 0.20))
        wd = self.world(S) if self.scene else None; t_grasp = None; grasp_name = demand.get("grasp_on_reach")
        sid = S["site"]
        if wd is not None and wd.fingers: wd.reset_fingers(held=demand.get("held", [])); wd.finger_ctrl(d, 0.0)
        if wd is not None:
            for nm in demand.get("held", []):
                if wd.fingers: continue                                   # finger robots: the object is physically in the hand (handed-off state), no teleport, no weld
                if wd.palm_dist(d, nm) > 0.10:
                    # object declared IN HAND but not at the palm (e.g. a tool that fell during the plant settle): it starts
                    # at the palm (labelled abstraction: "the tool is in the hand at the start")
                    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm + "_free"); a = m.jnt_qposadr[j]
                    d.qpos[a:a+3] = d.site_xpos[sid]; d.qpos[a+3:a+7] = [1, 0, 0, 0]; d.qvel[m.jnt_dofadr[j]:m.jnt_dofadr[j]+6] = 0; mujoco.mj_forward(m, d)
                wd.grasp(d, nm)
        W_act = W_con = W_pas = 0.0; work = np.zeros(nT); nonfoot = False; stance_lost = False; peak_grf = 0.0; run_unl = [0.0, 0.0]; cont_unl = [0.0, 0.0]; unl_run_max = 0.0; n_unl = 0
        fl = S["floor"]; feet = S["feet"]; z0 = float(d.subtree_com[0][2])
        dmin = 1e9; palm_dmin = None; palm_hold = None; t_reach = None; held_t = 0.0; hold_sum = 0.0; hold_ok = False; util = 0.0; trace = []; peak_speed_at_reach = None
        tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link"); trunk_pitch_max = 0.0; knee_max = 0.0
        kj = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("left_knee_joint", "right_knee_joint")]
        bal_n = 0; bal_bad = 0; cp_err_max = 0.0                     # terminal gate: capture point within 8 cm of the STANCE capture point (feet centre + cp_ref_off), both feet loaded
        ff_prev = 1e3; tg = np.array(d.ctrl); d_ref = mujoco.MjData(m); i_tilt = -1
        # ADMITTANCE on the reference (J15b): regulate the normal force between a held tool and a surface by moving the palm
        # target vertically through the arm joints (palm-z Jacobian pseudo-inverse): dz += k (F_ref - F) dt, |dz| <= dz_max,
        # active only in the time window the task declares (the wipe's pressing phase). The servo compliance turns dz into force.
        adm = demand.get("admittance"); F_tool = 0.0; dz_adm = 0.0; dq_adm = np.zeros(len(J["aid"]))
        # ORIENTATION control through the WRIST (J26): demand["orient"] = [dict(body=<body name>|"palm", axis=[local], target=[world], k=gain)]
        # Each control tick the rotation error e = sum cross(R a_local, a_target) is mapped to wrist-joint increments through the
        # rotational Jacobian of the palm site restricted to the wrist dofs (pseudo-inverse) and integrated into the servo targets.
        ori = demand.get("orient"); dq_ori = np.zeros(len(J["aid"])); ori_err_max = 0.0; ori_err_now = 0.0
        if ori:
            wrist_mask = np.array([g == "wrist" for g in S["groups"]]); wrist_dofs = np.asarray(J["dof"])[wrist_mask]
            if wrist_mask.sum() == 0: ori = None
            else:
                jacr = np.zeros((3, m.nv)); jacp_o = np.zeros((3, m.nv)); ori_items = []
                for it in ori:
                    bname = it.get("body", "palm"); bid_o = m.site_bodyid[sid] if bname == "palm" else (wd.bid[bname] if (wd is not None and bname in wd.bid) else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname))
                    ori_items.append((int(bid_o), np.asarray(it.get("axis", [1, 0, 0]), float), np.asarray(it.get("target", [0, 0, -1]), float), float(it.get("k", 4.0)), float(it.get("max", 0.6))))
        if adm:
            tool_g = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == wd.bid[adm["tool"]]}; surf_g = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == wd.bid[adm["surface"]]}
            arm_mask = np.array([g in ("shoulder", "elbow", "wrist") for g in S["groups"]]); arm_dofs = np.asarray(J["dof"])[arm_mask]
            jacp = np.zeros((3, m.nv)); t_on, t_off_ = adm.get("window", (0.0, 1e9)); F_ref = float(adm.get("F_ref", 4.0)); k_adm = float(adm.get("k", 0.004)); dz_max = float(adm.get("dz_max", 0.03))
            adm_log = []; F_filt = 0.0; tau_f = float(adm.get("tau", 0.05))     # first-order force filter: box-on-slab contact chatters (0 / 46 N spikes)
        try:
            for s in range(nst):
                tt = s*dt; k = min(int(tt/knot_dt + 1e-9), NKl-1); k2 = min(k+1, NKl-1); f = min(1.0, (tt - k*knot_dt)/knot_dt) if k < NKl-1 else 0.0
                off = kn[:, k] + (kn[:, k2]-kn[:, k])*f
                if ref is not None:
                    i_r = min(int(tt/ts*fps), len(ref["Q"]) - 1); q_ref = ref["Q"][i_r][qa]
                else: q_ref = q_s[qa]
                if t_reach is not None and tt > t_reach + hold: off = off*0.0 + off_hold          # after the hold: freeze the offsets (arrive and stay)
                elif t_reach is not None: off_hold = off
                fz_prev = ff_prev if s > 0 else 1e3
                k_c = int(round(tt/0.002, 6))
                if k_c != ctl._kc:
                    # SQUAT: when the reference moves the legs (low target), the balance controller's stand pose follows the
                    # reference leg posture; capture-point and trunk corrections are applied on top of it as usual
                    if ref is not None and ref.get("legs"): ctl.q_stand = q_st0 + (np.asarray(ref["Q"][i_r]) - q_s)    # reference leg motion as a delta from the handed pose, on the nominal stance
                    # the balance controller holds the REFERENCE torso tilt (a deliberately bent trunk is not an error to be
                    # fought with the hips: that fight pitched the pelvis and unloaded a foot on the wipe)
                    tr_mode = demand.get("tilt_ref", False)   # False (vertical torso target; the hip term doubles as a posture stabiliser), True (hold the reference tilt: tried, destabilised the squat wipe), "roll" (reference roll only)
                    if ref is not None and tr_mode and i_r != i_tilt:
                        d_ref.qpos[:] = ref["Q"][i_r]; mujoco.mj_kinematics(m, d_ref); tr = ctl._tilt(d_ref, ctl.tb); ctl.tilt_ref = tr if tr_mode is True else np.array([0.0, tr[1]]); i_tilt = i_r
                    ctl._kc = k_c; tg = ctl.targets(k_c*0.002, d, fz_prev > 1.0)
                    if ori and d.site_xpos[sid][2] >= float(demand.get("orient_min_z", -1e9)):   # height gate: do not pitch the fingers into a slab (orient only above top + hand length)
                        e_rot = np.zeros(3)
                        for (bid_o, a_l, a_t, k_o, mx) in ori_items:
                            Rb = d.xmat[bid_o].reshape(3, 3); a_w = Rb @ a_l; e_rot += k_o*np.cross(a_w, a_t)
                        ori_err_now = float(np.linalg.norm(e_rot)/max(k_o, 1e-9)); ori_err_max = max(ori_err_max, ori_err_now)
                        mujoco.mj_jacSite(m, d, jacp_o, jacr, sid); Jw = jacr[:, wrist_dofs]
                        Wsel = np.asarray(demand.get("orient_joints", [1.0]*Jw.shape[1]), float)   # per-wrist-joint weight (0 = do not use)
                        Jw_ = Jw*Wsel[None, :]; dq_w = Wsel*np.linalg.lstsq(Jw_, e_rot*0.002, rcond=None)[0]   # weighted least squares (pitch-only etc.)
                        leak = float(demand.get("orient_leak", 1.0))                  # leaky integrator (time constant ~0.2 s at 500 Hz): the reference carries the orientation via the IK end pose; the loop only corrects residuals
                        cur = leak*dq_ori[wrist_mask] + dq_w; dq_ori[wrist_mask] = np.clip(cur, -mx, mx)
                    if adm and t_on <= tt < t_off_:
                        dz_adm = float(np.clip(dz_adm - k_adm*(F_ref - F_filt)*0.002, -dz_max, dz_max))     # too little force -> palm target moves DOWN (negative z)
                        mujoco.mj_jacSite(m, d, jacp, None, sid); Jz = jacp[2, arm_dofs]
                        dq_adm[:] = 0.0; dq_adm[arm_mask] = Jz*dz_adm/max(float(Jz @ Jz), 1e-6)       # smallest arm-joint displacement that moves the palm by dz
                        pen_t = 0.0; tilt_t = 0.0
                        for ci in range(d.ncon):
                            c = d.contact[ci]
                            if (c.geom1 in tool_g and c.geom2 in surf_g) or (c.geom2 in tool_g and c.geom1 in surf_g): pen_t = min(pen_t, float(c.dist))
                        Rt = d.xmat[wd.bid[adm["tool"]]].reshape(3, 3); tilt_t = float(np.degrees(np.arccos(np.clip(Rt[2, 2], -1, 1))))
                        adm_log.append((tt, F_filt, dz_adm, -pen_t, tilt_t))          # (t, filtered force, dz, penetration m, tool tilt deg)
                tg[task_aid] = np.clip((q_ref + off + dq_adm + dq_ori)[task_mask], jlo[task_mask], jhi[task_mask]); d.ctrl[:] = tg   # servo targets inside the joint ranges (an out-of-range wrist target saturated the joint and left the hand offset)
                if ctrl_fn is not None: ctrl_fn(tt, d, S)                      # observers (composer logger); servo mode ignores the return
                if wd is not None and wd.fingers: wd.finger_ctrl(d, tt)
                BI.backemf_brake(m, d, S)
                qv0 = d.qvel.copy(); mujoco.mj_step(m, d); qm = 0.5*(qv0 + d.qvel)
                tq = d.actuator_force[J["aid"]]; util = max(util, float(np.max(np.abs(tq)/tau_v)))
                W_act += float(np.dot(d.actuator_force, qm[S["act_dof"]]))*dt; W_con += float(np.dot(d.qfrc_constraint, qm))*dt; W_pas += float(np.dot(d.qfrc_passive, qm))*dt + float(np.dot(d.qfrc_applied, qm))*dt
                pw = tq*d.qvel[J["dof"]]; work += np.where(pw > 0, pw, 0.0)*dt
                ff = [0.0, 0.0]; fz = 0.0; F_tool = 0.0
                for ci in range(d.ncon):
                    c = d.contact[ci]
                    if adm and ((c.geom1 in tool_g and c.geom2 in surf_g) or (c.geom2 in tool_g and c.geom1 in surf_g)):
                        fv_ = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fv_); F_tool += abs(float(fv_[0])); continue
                    if fl not in (c.geom1, c.geom2): continue
                    fv = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fv); fz += fv[0]
                    o = c.geom2 if c.geom1 == fl else c.geom1
                    if o in feet[0]: ff[0] += fv[0]
                    elif o in feet[1]: ff[1] += fv[0]
                    elif o in S.get("robot_geoms", {o}): nonfoot = True
                if adm: F_filt += (F_tool - F_filt)*dt/max(tau_f, dt)
                # quasi-static stance rule: the balance controller shifts weight, so a foot may unload briefly; a foot is LOST
                # if unloaded for more than 10% of the horizon cumulatively (the throw's 20 ms rule guarded a speed exploit)
                for i_f in (0, 1):
                    if s > 5 and ff[i_f] < 1.0: run_unl[i_f] += dt; cont_unl[i_f] += dt; unl_run_max = max(unl_run_max, cont_unl[i_f])
                    else: cont_unl[i_f] = 0.0
                    # quasi-static STANCE rule (revised 2026-10-10): a foot is LOST if it leaves the floor continuously for >= 0.25 s
                    # or cumulatively for >= 25% of the horizon. The earlier cumulative-10% rule was tripped by lateral ROCKING
                    # (feet alternately unloading at ~2 Hz with the capture point within 6 mm of centre) on a 5 s composite wipe:
                    # that is a servo-stiffness artefact (open item J15d), not a lost stance. Rocking is still penalised through n_unl.
                    if cont_unl[i_f] >= 0.25 or run_unl[i_f] >= 0.25*H: stance_lost = True
                if s > 5 and min(ff) < 1.0: n_unl += 1
                peak_grf = max(peak_grf, fz); ff_prev = fz
                if tt >= H - 0.2:
                    bal_n += 1; BW = S["M"]*G
                    com = d.subtree_com[0]; vcom = P.com_velocity(m, d); w0 = np.sqrt(G/max(com[2], 0.2)); cp = com[:2] + vcom[:2]/w0
                    feet_c = 0.5*(B_.foot_xy(d, feet_pts[0]) + B_.foot_xy(d, feet_pts[1])) + cp_ref_off; cp_err = float(np.linalg.norm(cp - feet_c))
                    cp_err_max = max(cp_err_max, cp_err)
                    if min(ff) < 0.10*BW or cp_err > 0.08: bal_bad += 1
                dist = float(np.linalg.norm(d.site_xpos[sid] - tgt))
                if dist < dmin: dmin = dist; palm_dmin = d.site_xpos[sid].copy()
                if ref is not None and palm_hold is None and tt >= float(ref.get("takeoff_ref", 0.0))*ts + 0.5: palm_hold = d.site_xpos[sid].copy()   # settled palm after the reference arrives (aim-off measures here)
                if t_reach is None and dist <= tol:
                    t_reach = tt; off_hold = off; vel6 = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, sid, vel6, 0); peak_speed_at_reach = float(np.linalg.norm(vel6[3:]))
                # WELD-ON-PROXIMITY (J14 abstraction): the grasp engages the first time the palm comes within grasp_tol of the object,
                # not only at the reach instant (H1's forearm pushed the cube 6 cm away while descending onto it, so the
                # reach "arrived" at a point the cube had left)
                ori_ok = (not ori) or (ori_err_now <= float(demand.get("close_ori_tol", 0.15)))        # fingers close only once the hand orientation is right (closing on a 30 deg-tilted hand missed the handle)
                if wd is not None and wd.fingers and ref is not None and tt < 0.95*float(ref.get("takeoff_ref", 0.0))*ts: ori_ok = False   # and only at the END of the reference (the palm passes within tolerance on the way down and closed mid-air)
                if wd is not None and grasp_name and t_grasp is None and not wd.held(d, grasp_name) and ori_ok and (wd.palm_dist(d, grasp_name) <= float(demand.get("grasp_tol", 2*tol)) or (wd.fingers and t_reach is not None)):   # fingers: close on ARRIVAL at the pre-grasp pose
                    wd.grasp(d, grasp_name)
                    if not wd.fingers: t_grasp = tt
                if wd is not None and wd.fingers and grasp_name and t_grasp is None and grasp_name in wd.closing and wd.held(d, grasp_name): t_grasp = tt   # fingers: held = >= 2 hand links in contact
                if t_reach is not None and tt < t_reach + hold + 1e-9: held_t += dt; hold_sum += dist*dt
                R = d.xmat[tb].reshape(3, 3); trunk_pitch_max = max(trunk_pitch_max, float(np.arccos(np.clip(R[2, 2], -1, 1))))
                knee_max = max(knee_max, float(max(d.qpos[kj])))
                if record and s % 10 == 0: trace.append(d.qpos.copy())
            if wd is not None and demand.get("release_at_end"):
                for nm in list(demand.get("held", [])) + ([grasp_name] if grasp_name else []):
                    if nm in wd.eq and wd.held(d, nm): wd.release(d, nm)
        finally:
            for i, aid in enumerate(J["aid"]): BI._as_motor(m, int(aid), float(tau_v[i]))
        E1 = float(d.energy[0] + d.energy[1]); resid = (E1 - E0) - W_act; closure = resid - W_con - W_pas
        fell = float(d.subtree_com[0][2]) < z0 - 0.35
        rep = TrajectoryValidityReport(start_qdot_max=(start_qdot if start_qdot >= 0.5 else min(start_qdot, 0.04)), energy_residual=resid, energy_tol=max(1.0, 0.02*abs(W_act)), actuator_work=W_act,
                                       contact_energy=0.0, ke_takeoff=0.0, nonfoot_contact=nonfoot, took_off=True, peak_grf_bw=peak_grf/(S["M"]*G),
                                       energy_closure=closure, closure_tol=max(5.0, 0.02*(abs(W_act) + abs(W_con)))).evaluate()
        reasons = list(rep.reasons) + (["STANCE_LOST"] if stance_lost else []) + (["FELL"] if fell else [])
        hold_mean = (hold_sum/max(held_t, 1e-9)) if t_reach is not None else None
        hold_ok = t_reach is not None and held_t >= hold - 1e-6 and hold_mean <= 2*tol
        if t_reach is None: reasons.append("NOT_REACHED")
        elif not hold_ok: reasons.append("NOT_HELD")
        if grasp_name and t_grasp is None and wd is not None: reasons.append("NOT_GRASPED")
        end_unbalanced = bal_n > 0 and bal_bad > 0.2*bal_n
        if end_unbalanced: reasons.append("END_UNBALANCED")
        end_qdot = float(np.max(np.abs(d.qvel[rj])))
        viol = 0.0
        if end_unbalanced: viol += 1.0 + 10.0*max(0.0, cp_err_max - 0.08) + 2.0*bal_bad/max(bal_n, 1)
        if grasp_name and t_grasp is None and wd is not None: viol += 1.0 + 20.0*max(0.0, (wd.palm_dist(d, grasp_name) if t_reach is None else 0.0))
        if t_reach is None: viol += 1.0 + 20.0*max(0.0, dmin - tol)
        elif not hold_ok: viol += 1.0 + 20.0*max(0.0, (hold_mean if hold_mean is not None else 1.0) - 2*tol) + 5.0*max(0.0, hold - held_t)
        if stance_lost: viol += 1.0 + 100.0*n_unl/max(nst, 1)
        if nonfoot: viol += 5.0
        if fell: viol += 5.0
        if any(z in rep.reasons for z in ("ENERGY_CLOSURE", "ENERGY_ACCOUNTING_VIOLATION")): viol += 2.0
        gw = {}
        for g, wv in zip(S["groups"], work): gw[g] = gw.get(g, 0.0) + float(wv)
        tot = float(sum(gw.values()))
        out = dict(value=-tot, valid=(viol <= 0), reasons=reasons, violation=float(viol), cost=tot,
                   extra=dict(palm_end=d.site_xpos[sid].copy(), palm_dmin=(palm_dmin if palm_dmin is not None else d.site_xpos[sid].copy()), palm_hold=(palm_hold if palm_hold is not None else d.site_xpos[sid].copy()), dmin=dmin, t_reach=t_reach, hold_ok=hold_ok, hold_mean=hold_mean, speed_at_reach=peak_speed_at_reach, peak_util=util, work=gw, unl_frac=n_unl/max(nst, 1), unl_run_max=unl_run_max, adm_dz_final=dz_adm, adm_log=(adm_log if adm else None), ori_err_max=ori_err_max,
                              trunk_pitch_max=trunk_pitch_max, knee_max=knee_max, W_act=W_act, closure=closure, peak_grf_bw=peak_grf/(S["M"]*G), final_state=(d.qpos.copy(), d.qvel.copy()),
                              end_qdot=end_qdot, t_grasp=t_grasp, cp_err_max=cp_err_max, end_unbalanced=end_unbalanced, held_after=([n for n in wd.objects if wd.held(d, n)] if wd is not None else []), world_after=(wd.state(d) if wd is not None else None)))
        if record: out["extra"]["trace"] = np.array(trace).tolist()
        return out

    def objective(self, rec, demand): return rec["cost"] + 20.0*rec["extra"].get("peak_util", 0.0)      # least work, light pressure off the torque limits
    def warm(self, S, demand, stored):
        z = list(np.asarray(stored["x"]))
        if demand.get("prior"): z.append((stored.get("tscale", 1.0) - 0.87)/0.3)
        return np.array(z)
