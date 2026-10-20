"""J15: surface contact with force regulation, as ONE parameterised family (wipe / scrub / sweep / mop / stir).
A grasped TOOL (an engine.world object, welded to the hand) is driven along a PATH over a SURFACE while the tool-surface
contact normal force must stay in a BAND. Built on the quasi-static reach (legs by the balance controller, arm+trunk servos
tracking a reference + knot offsets): the reference is the path at a height that presses the tool into the surface, the
servo compliance turns the pressing depth into force, and the optimiser's offsets tune depth and posture.
Demand: {"tool", "surface", "path": [[x,y],...] (world xy on the surface), "force": [lo, hi] N, "duration" s, "held": [tool]}.
Effect: path COVERAGE (fraction of path points passed within tol by the tool contact) with force in band."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco, branch_i as BI, plant as P
from validity import TrajectoryValidityReport
from engine import core
from tasks.reach import ReachTask, G, NK


class WipeTask(ReachTask):
    name = "wipe"
    def demand_default(self): return dict(tool="sponge", surface="table", path=[[0.30, -0.30], [0.45, -0.30], [0.45, -0.10], [0.30, -0.10]], force=[2.0, 8.0], duration=2.0, tol=0.04, prior="ascent", omega_servo=120.0, press=0.02, approach=1.2, clearance=0.06, retract=0.5, qd_max=0.8, omega_legs=150.0, refine=0, aim_z=False, trunk_cap=0.02, squat=0.25, cap_from_stance=True, tilt_ref=False)
    # omega_legs 150: softer leg servos than the landing-tuned 368 rad/s; at 368 the two legs fight through the floor and the
    # robot ROCKS laterally (feet alternately leaving the floor for >0.25 s with the capture point within 6 mm of centre) during
    # a long squatted wipe; at 150 the stance holds (handoff wipe from the pick: valid, cp 2 cm). J15d keeps the rocking open.
    # trunk_cap 0.02: waist yaw/roll essentially frozen (a rolled torso is "corrected" by the hip-roll balance term, which shifts the
    # pelvis and unloads the arm-side foot); squat 0.25: legs squat (sagittal, mirrored) only when arm+trunk cannot reach;
    # cap_from_stance: caps measured from the original stance across the chained segments (per-segment caps accumulated to a
    # 0.47 rad waist pitch on the 4-point path and the robot fell); tilt_ref False: holding the reference torso tilt destabilised
    def surface_top(self, demand): s = self.surfaces_by()[demand["surface"]]; return s["pos"][2] + s["size"][2]
    def surfaces_by(self): return {s["name"]: s for s in self.surfaces}
    def tool_half(self, demand):
        o = {x["name"]: x for x in self.objects}[demand["tool"]]
        if o.get("half_z") is not None: return float(o["half_z"])          # compound tools (brush): the head's half-height below the body origin
        return o["size"][2] if o["kind"] in ("box", "cylinder") else o["size"][0]
    def path_world(self, S, demand):
        """Reference path for the TOOL CENTRE: the xy waypoints at (surface top + tool half-height - press)."""
        z = self.surface_top(demand) + self.tool_half(demand) - float(demand.get("press", 0.01))
        return np.array([[x, y, z] for x, y in demand["path"]], float)
    def ref(self, S, demand):
        """Reference with AIM-OFF refinement: build the chained-prior reference, replay it once (pure tracking), measure the
        tool's xy error at each waypoint, shift the waypoint targets by the negative error (clipped 8 cm), rebuild; repeat
        demand["refine"] times. Corrects any systematic tracking offset (servo sag, the hips fighting a pitched trunk)
        without modelling its cause. Costs one pure rollout per refinement."""
        key = self._key(S, demand)
        if not hasattr(self, "_refs"): self._refs = {}
        if key in self._refs: return self._refs[key]
        n_ref = int(demand.get("refine", 0)); aim = np.zeros((len(demand["path"]), 3)); errs = []
        for it in range(n_ref + 1):
            r = self._build_ref(S, demand, aim)
            if it == n_ref: break
            self._refs[key] = r; err = self._waypoint_errors(S, demand, r); errs.append(np.round(err, 3).tolist())
            if not demand.get("aim_z", False): err[:, 2] = 0.0                          # z aim-off pushed the tool into the slab until it left the hand (table 0.70 test): xy only by default
            aim -= 0.6*np.clip(err, -0.08, 0.08)                                     # damped (unit gain overshot and oscillated)
        r["aim"] = aim; r["aim_errors"] = errs; self._refs[key] = r
        return r
    def _key(self, S, demand):
        return ("wipe", tuple(map(tuple, demand["path"])), float(demand.get("press", 0.01)), float(demand.get("approach", 0.8)), float(demand.get("duration", 2.0)), float(demand.get("clearance", 0.06)), float(demand.get("retract", 0.5)), float(demand.get("end_dwell", 0.2)), float(demand.get("qd_max", 0.8)), float(demand.get("trunk_cap", 0.10)), float(demand.get("squat", 0.25)), bool(demand.get("cap_from_stance", True)), np.round(np.asarray(S["q_settled"])[S["jarr"]["qad"]], 3).tobytes())
    def _waypoint_errors(self, S, demand, ref):
        """Pure-tracking replay: tool xy minus waypoint xy at the end of each pressing segment."""
        m = S["m"]; wd = self.world(S); pts = self.path_world(S, demand)
        ends = np.asarray(ref["bounds"])[3:3 + len(pts)]; seen = {}; top = self.surface_top(demand); half = self.tool_half(demand)
        def obs(t, d, S_):
            for k, te in enumerate(ends):
                if k not in seen and t >= te - 1e-9:
                    tp = wd.pos(d, demand["tool"]); seen[k] = np.array([tp[0], tp[1], max(0.0, tp[2] - half - top)])   # z error = height ABOVE the surface (0 when pressing)
        p0 = dict(x=np.zeros(len(S["types"])*NK), tscale=1.0)
        self.rollout(S, demand, p0, ctrl_fn=obs)        # ref() returns the reference under refinement (cached under the same key)
        return np.array([(seen[k] - np.array([pts[k][0], pts[k][1], 0.0])) if k in seen else np.zeros(3) for k in range(len(pts))])
    def hand_tool_frame(self, S, demand):
        """Finger robots holding the tool: the tool is RIGID in the hand (J27 wrap), so (a) the tool origin is a fixed point in the
        palm frame (rel) and (b) an orientation demand on the tool body is an orientation demand on a palm-frame axis. Returns
        (rel, orient_rewritten) from the handed state, or (None, demand["orient"]) when the tool is not held by fingers."""
        wd = self.world(S)
        if not (getattr(wd, "fingers", None) and demand["tool"] in demand.get("held", [])): return None, demand.get("orient")
        m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d); Rp = d.site_xmat[S["site"]].reshape(3, 3); bt = wd.bid[demand["tool"]]
        rel = Rp.T @ (d.xpos[bt] - d.site_xpos[S["site"]]); Rt = d.xmat[bt].reshape(3, 3); items = []
        for it in (demand.get("orient") or []):
            if it.get("body") == demand["tool"]: items.append(dict(it, body="palm", axis=(Rp.T @ (Rt @ np.asarray(it.get("axis", [0, 0, 1]), float))).tolist()))
            else: items.append(it)
        return rel, items
    def _build_ref(self, S, demand, aim):
        key = self._key(S, demand)
        if True:
            from engine import strategy as ST
            import throw as T
            m = S["m"]; wd = self.world(S); d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d)
            rel_tool, orient_ik = self.hand_tool_frame(S, demand); demand_ik = dict(demand, orient=orient_ik, **({"ik_point": rel_tool.tolist()} if rel_tool is not None else {}))   # waypoints are TOOL points for a held tool: the IK places the tool through the hand (the palm offset changes as the wrist orients the tool; a fixed offset left the head 2-4 cm high)
            off = d.site_xpos[S["site"]] - wd.pos(d, demand["tool"])            # palm relative to the held tool
            if np.linalg.norm(off) > 0.10 and not (getattr(wd, "fingers", None) and demand["tool"] in demand.get("held", [])): off = np.zeros(3)   # weld robots: tool not at the palm yet -> tool centre = palm. Finger robots hold a tool whose centre (brush head) is 16 cm below the palm: keep the real offset (the zeroed offset drove the palm 16 cm down and crushed the brush at 157 N)
            path = self.path_world(S, demand); path = path + aim   # aim-off: shifted waypoint TARGETS (coverage is still scored on the true path)
            # APPROACH: the tool rises above the surface and comes down onto the first waypoint before the pressing path
            # (a tool held below the table top otherwise drives into the slab edge)
            clear = self.surface_top(demand) + self.tool_half(demand) + float(demand.get("clearance", 0.06))
            tool0 = wd.pos(d, demand["tool"]) if np.linalg.norm(off) > 0 else d.site_xpos[S["site"]]
            # approach = UP first at the current xy (clear the slab edge), then OVER the first waypoint, then DOWN onto it
            Tr = float(demand.get("retract", 0.5))
            Td = float(demand.get("end_dwell", 0.2))      # finish the stroke: hold the last waypoint pressed before the retract (the last waypoint was scored at the retract transition)
            pts = np.vstack([[tool0[0], tool0[1], clear], [path[0][0], path[0][1], clear], path] + ([path[-1]] if Td > 0 else []) + ([[path[-1][0], path[-1][1], clear]] if Tr > 0 else [])) + off   # RETRACT: lift the tool off the surface at the end so the capability ENDS in a balanced, hand-off-able posture (the stretched end pose failed the terminal balance gate)
            T_ = float(demand.get("duration", 2.0)) + float(demand.get("approach", 0.8)) + Tr; H = T_ + 0.5
            # chain reach priors waypoint to waypoint on the bare robot from the handed pose; stitch the joint paths in time
            Sb = T.setup(self.robot, ball=False); BI.apply_dt(Sb, core.DT_OPT); BI.settle(Sb); nqb = Sb["m"].nq
            q = np.asarray(S["q_settled"]); v = np.asarray(S["v_settled"]); qa = S["jarr"]["qad"]; segs = []; q_cur = q[:nqb].copy(); legs = False; misses = []
            with core.handoff(Sb, q_cur, v[:Sb["m"].nv]):
                for p_ in pts:
                    Sb["q_settled"] = q_cur
                    r = ST.reach_prior(Sb, p_, horizon=1.0, nk=NK, t_ref=0.6, tol=0.01, trunk_cap=float(demand.get("trunk_cap", 0.10)), squat_if_short=float(demand.get("squat", 0.25)), cap_ref=(np.asarray(S["q_stance0"])[:nqb] if demand.get("cap_from_stance", True) else None)); Qs = r["Q"]; i_end = int(0.6*r["fps"])   # caps from the ORIGINAL stance across the chain; a wipe is an ARM task: trunk nearly still (lateral hopping otherwise); legs squat only when the arm cannot reach
                    seg = np.array(Qs[:i_end + 1])
                    if demand.get("orient"):
                        # ORIENTATION-CONSISTENT waypoints (the position-only prior rotated the hand back toward its natural pose and
                        # the wrench stripped the finger pinch within a second): full-arm IK at the waypoint, blended over the segment's last 60%
                        arm_q = np.asarray(S["jarr"]["qad"])[[i for i, g in enumerate(S["groups"]) if g in ("shoulder", "elbow", "wrist")]]
                        q_ik, _, _ = self.arm_ik(S, np.concatenate([seg[-1], q[nqb:]]), (p_ - off) if rel_tool is not None else p_, demand_ik); i0 = int(0.4*(len(seg) - 1))   # held tool: IK on the TOOL point (p_ - off), orientation on palm-frame axes
                        for i in range(i0, len(seg)): w_ = (i - i0)/max(len(seg) - 1 - i0, 1); seg[i, arm_q] = (1 - w_)*seg[i, arm_q] + w_*q_ik[arm_q]
                    segs.append(seg); q_cur = seg[-1].copy(); legs = legs or r["legs"]; misses.append(r["miss"])
            Ta = float(demand.get("approach", 0.8)); Tp = float(demand.get("duration", 2.0)); n_path = len(path); seg_T = [Ta/2, Ta/2] + [Tp/n_path]*n_path + ([Td] if Td > 0 else []) + ([Tr] if Tr > 0 else [])
            # JOINT SPEED LIMIT on the reference: a segment takes at least (largest joint change)/qd_max. From a bent handoff
            # posture the first "lift the tool" segment un-pitched the trunk 31 deg in 0.3 s and the robot hopped (feet 0.14 BW).
            qd_max = float(demand.get("qd_max", 0.8)); dq_seg = [float(np.max(np.sum(np.abs(np.diff(np.asarray(sg)[:, 7:nqb], axis=0)), axis=0))) if len(sg) > 1 else 0.0 for sg in segs]   # TOTAL VARIATION per joint (end-to-end change under-timed a segment whose IK blend doubles back: 1.7 rad/s shoulder under a 0.8 cap)
            seg_T = [max(T0, dq_/qd_max) for T0, dq_ in zip(seg_T, dq_seg)]; bounds = np.cumsum([0] + seg_T); T_ = float(bounds[-1]); H = T_ + 0.5
            fps = 100; n = int(H*fps) + 1; Q = np.repeat(q[None], n, 0); t = np.arange(n)/fps
            for i, ti in enumerate(t):
                k = min(int(np.searchsorted(bounds, ti, side="right")) - 1, len(segs) - 1); w = min(1.0, (ti - bounds[k])/seg_T[k]) if ti < T_ else 1.0
                seg = segs[k]; j = min(int(w*(len(seg) - 1)), len(seg) - 1); Q[i, :nqb] = seg[j]; Q[i, nqb:] = q[nqb:]
            return dict(key="wipe_prior", start=q.copy(), Q=Q, fps=fps, takeoff_ref=T_, horizon=H, nk=NK, path_steps=sum(len(s_) for s_ in segs), legs=legs, misses=misses,
                        bounds=bounds.tolist(), seg_T=seg_T, t_app=float(bounds[2]), t_press_end=float(bounds[2 + n_path + (1 if Td > 0 else 0)]), dq_seg=dq_seg)
    def rollout(self, S, demand, p, ctrl_fn=None, record=False):
        """Reach-style servo rollout driven by the wipe reference; measures tool-surface contact force and path coverage."""
        import tasks.reach as R_
        ref = self.ref(S, demand); H_old = R_.HOR; R_.HOR = float(ref["horizon"])            # the reference sets the horizon (joint speed limit may stretch it)
        try:
            d_ = dict(demand, target_world=self.path_world(S, demand)[-1].tolist(), tol=0.5, hold=0.05)   # the base rollout's reach gate is disabled (tol 0.5)
            p = dict(p, tscale=1.0)                                                                        # the wipe reference carries its own timing (speed limit, force window): the prior's time-scale knob is pinned (zeros gave 0.87 = reference 15% fast, window misaligned)
            # tool-surface normal force is measured IN THE LOOP every control tick (the earlier trace replay re-ran mj_forward with
            # zero velocities and under-read the force ~10x: 0.7 N vs 7 N); admittance (J15b) acts on it when force_control is set
            lo_, hi_ = demand["force"]
            d_["admittance"] = dict(tool=demand["tool"], surface=demand["surface"], F_ref=float(demand.get("F_ref", 0.5*(lo_ + hi_))), k=(float(demand.get("adm_k", 0.004)) if demand.get("force_control") else 0.0),
                                    dz_max=float(demand.get("adm_dz_max", 0.03)), tau=float(demand.get("adm_tau", 0.05)), window=(float(ref["t_app"]), float(ref["t_press_end"])))
            rec = super().rollout(S, d_, p, ctrl_fn=ctrl_fn, record=True)
        finally: R_.HOR = H_old
        # contact force + coverage from the recorded qpos trace (every 10 steps) by replaying forward kinematics + contact forces
        m = S["m"]; wd = self.world(S); tool_geoms = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == wd.bid[demand["tool"]]}
        surf_b = wd.bid[demand["surface"]]; surf_geoms = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == surf_b}
        pts = self.path_world(S, demand); tol = float(demand.get("tol", 0.04)); lo, hi = demand["force"]
        covered_xy = np.zeros(len(pts), bool); covered = np.zeros(len(pts), bool); f_in = 0; f_n = 0; forces = []; gaps = []
        top = self.surface_top(demand); half = self.tool_half(demand); t_app = float(ref["t_app"]); t_press_end = float(ref["t_press_end"])
        d = mujoco.MjData(m)
        for i, q in enumerate(rec["extra"]["trace"]):
            d.qpos[:] = q; mujoco.mj_forward(m, d); t_ = i*10*m.opt.timestep
            fz = 0.0
            for ci in range(d.ncon):
                c = d.contact[ci]
                if (c.geom1 in tool_geoms and c.geom2 in surf_geoms) or (c.geom2 in tool_geoms and c.geom1 in surf_geoms):
                    f = np.zeros(6); mujoco.mj_contactForce(m, d, ci, f); fz += abs(f[0])
            forces.append(fz); tp = wd.pos(d, demand["tool"]); dxy = np.linalg.norm(pts[:, :2] - tp[:2], axis=1)
            g_ = max(0.0, (tp[2] - half) - top)
            if t_app <= t_ < t_press_end: gaps.append(g_)                   # height gap over the pressing phase only (the retract lifts the tool by design)
            covered_xy |= dxy <= tol                                                                        # geometric coverage (xy) at ANY time (a waypoint passed during the approach counts: the composite wipe scored "on the surface at every waypoint" yet failed a pressing-phase-only xy rule)
            if fz > 0.5: f_n += 1; f_in += int(lo <= fz <= hi)
            if fz > 0.5 or g_ <= 0.01: covered |= dxy <= tol                                                 # covered = tool on the surface (contact or within 1 cm)
        coverage = float(covered.mean()); cov_xy = float(covered_xy.mean()); gap = float(np.mean(gaps)) if gaps else 0.0
        al = rec["extra"].get("adm_log") or []                                        # (t, filtered normal force, dz) every 2 ms over the pressing window
        Fw = np.array([r_[1] for r_ in al]) if al else np.array([]); pen_w = np.array([r_[3] for r_ in al]) if al else np.array([]); tilt_w = np.array([r_[4] for r_ in al]) if al else np.array([])
        if len(Fw):
            f_n = int(np.sum(Fw > 0.5)); fm = float(np.mean(Fw)); in_band = float(np.mean((Fw >= lo) & (Fw <= hi))); forces = Fw.tolist()
        else:
            in_band = (f_in/max(f_n, 1)) if f_n else 0.0; fm = float(np.mean([f for f in forces if f > 0.5])) if f_n else 0.0
        viol = rec["violation"]
        reasons = [r for r in rec["reasons"] if r not in ("NOT_REACHED", "NOT_HELD")]
        # graded so the search has a gradient toward the surface: xy coverage first, then the height gap, then force in band
        if cov_xy < 0.999: viol += 1.0 + 5.0*(1.0 - cov_xy); reasons.append("PATH_NOT_COVERED")
        if coverage < 0.999: viol += 1.0 + 3.0*(1.0 - coverage) + 20.0*gap; reasons.append("TOOL_OFF_SURFACE")
        pen_lim = float(demand.get("pen_max", 0.003)); tilt_lim = float(demand.get("tilt_max", 8.0))      # realism gates (2026-10-11): no sinking into the slab, tool flat
        if len(pen_w) and pen_w.max() > pen_lim: viol += 1.0 + 100.0*(pen_w.max() - pen_lim); reasons.append("PENETRATION")
        if len(tilt_w) and tilt_w.mean() > tilt_lim: viol += 1.0 + 0.1*(tilt_w.mean() - tilt_lim); reasons.append("TOOL_TILTED")
        # FORCE BAND: reported and pulled toward by a soft objective term, NOT a validity gate yet. Regulating the normal force
        # through a 5 N m wrist whose orientation the surface dictates needs admittance control (open item).
        self._force_pen = (0.0 if (f_n and lo <= fm <= hi) else 1.0 + 0.2*min(abs(fm - lo), abs(fm - hi)))
        al = rec["extra"].pop("adm_log", None); rec["extra"]["adm_trace"] = (al[::25] if al else None); rec["extra"].update(pen_max=(float(pen_w.max()) if len(pen_w) else 0.0), pen_mean=(float(pen_w.mean()) if len(pen_w) else 0.0), tilt_mean=(float(tilt_w.mean()) if len(tilt_w) else 0.0), tilt_max=(float(tilt_w.max()) if len(tilt_w) else 0.0), coverage=coverage, coverage_xy=cov_xy, gap_mean=gap, force_in_band=in_band, force_mean=fm, force_pen=self._force_pen, force_max=float(np.max(forces)) if forces else 0.0, n_contact=f_n)
        rec["extra"].pop("trace", None) if not record else None
        return dict(rec, value=coverage, valid=(viol <= 0), reasons=reasons, violation=float(viol))
    def objective(self, rec, demand): return rec["cost"] + 20.0*rec["extra"].get("peak_util", 0.0) + 10.0*rec["extra"].get("force_pen", 0.0)
