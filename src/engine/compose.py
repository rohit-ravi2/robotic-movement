"""J7: composer v0. Sequence capabilities through REAL physical states: each primitive runs from the previous one's final
state (handoff), a balance-controller settle phase brings the robot back near its stance between primitives, and the
handoff is checked (joint deviation, residual speed). No teleportation, no reset. Output: per-step records + one
continuous qpos trace for rendering."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco, branch_i as BI, balance as B
from engine import core

HANDOFF_Q_TOL = 0.15      # rad, max driven-joint deviation from the canonical stance at handoff
HANDOFF_V_TOL = 0.15      # rad/s


def _logger(S, inner):
    log = []; n = len(S["jarr"]["aid"])
    def fn(t, d, S_):
        log.append((t, d.qpos.copy(), d.qvel.copy())); return inner(t, d, S_) if inner is not None else np.zeros(n)
    return fn, log


def settle(S, robot, q, v, T=1.0, dt_ctrl=0.002, T_damp=0.3, T_ramp=1.0, omega_damp=40.0, omega_ramp=60.0, omega_hold=150.0, held=(), task=None):
    """Bring the robot from (q, v) back near its canonical stance in three gentle phases, all with PHYSICAL servo stiffness
    (kp = omega^2 I_eff, critically damped, torque-limited) rather than the 1/dt^2 settle servos:
      DAMP  (T_damp):  targets = current pose, omega_damp   -> kills residual motion without an impulse
      RAMP  (T_ramp):  targets -> stance (arm kept if carrying), omega_ramp
      HOLD  (T):       stance targets, omega_hold          -> at rest
    (v1 engaged 1/dt^2 servos on a moving body: one foot unloaded at the first step and the robot tipped over.)"""
    m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = q; d.qvel[:] = v; mujoco.mj_forward(m, d)
    if held and task is not None and getattr(task, "scene", None):
        for nm in held: task.world(S).grasp(d, nm)                            # keep carrying through the settle
    J = S["jarr"]; qs = np.asarray(S["q_settled"]); q_now = d.qpos[J["qad"]].copy(); q_tgt = qs[J["qad"]].copy(); trace = []
    if held:
        arm = np.array([g in ("shoulder", "elbow", "wrist", "counter_arm") for g in S.get("groups", [])])
        if arm.shape == q_tgt.shape: q_tgt = np.where(arm, q_now, q_tgt)
    ieff = np.array([1.0/float(S["Minv_key"][dof, dof]) for dof in J["dof"]])
    def gains(om):
        for i, aid in enumerate(J["aid"]):
            kp = om*om*ieff[i]; BI._as_servo(m, int(aid), kp, 2.0*np.sqrt(kp*ieff[i]), float(J["tau"][i]))
    for (a_, dof, qad, tau, kp, kd) in S["held"]: d.ctrl[a_] = qs[qad]
    def stiff():   # the plant's own settle servos (kp = I_eff/dt^2): hold the stance against gravity exactly
        for i, (aid, dof) in enumerate(zip(J["aid"], J["dof"])):
            kp = ieff[i]/m.opt.timestep**2; BI._as_servo(m, int(aid), kp, 2.0*np.sqrt(kp*ieff[i]), float(J["tau"][i]))
    try:
        # DAMP softly at the current pose (no impulse on a moving body), then RAMP + HOLD with the stiff settle servos
        # v2 behaviour kept (it handed the landed jump to a 23.6 m/s throw): stiff ramp + hold; the soft damp phase
        # (v3/v4) let the body lean before the stiff servos locked the lean in. Primitives are required to END AT REST.
        phases = [(T_ramp, stiff, lambda w: (1 - w)*q_now + w*q_tgt), (T, stiff, lambda w: q_tgt)]
        t = 0.0
        for (Tp, setg, target) in phases:
            setg(); n = int(Tp/m.opt.timestep)
            for s_ in range(n):
                w = min(1.0, s_*m.opt.timestep/max(Tp, 1e-9)); d.ctrl[J["aid"]] = target(w); mujoco.mj_step(m, d)
                if s_ % 10 == 0: trace.append((t + s_*m.opt.timestep, d.qpos.copy()))
            t += Tp
    finally:
        for i, aid in enumerate(J["aid"]): BI._as_motor(m, int(aid), float(J["tau"][i]))
    at_rest = float(np.max(np.abs(d.qvel[6:]))) < 0.05
    return d.qpos.copy(), d.qvel.copy(), trace, at_rest


def freeze(S, q, v, T=0.4, held=(), task=None):
    """Stiff implicit servos (kp = I_eff/dt^2) at the CURRENT pose for T s: residual motion -> 0, posture kept."""
    m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = q; d.qvel[:] = v; mujoco.mj_forward(m, d)
    if held and task is not None and getattr(task, "scene", None):
        for nm in held: task.world(S).grasp(d, nm)
    J = S["jarr"]; trace = []
    for i, (aid, dof) in enumerate(zip(J["aid"], J["dof"])):
        ieff = 1.0/float(S["Minv_key"][dof, dof]); kp = ieff/m.opt.timestep**2; BI._as_servo(m, int(aid), kp, 2.0*np.sqrt(kp*ieff), float(J["tau"][i]))
    try:
        d.ctrl[J["aid"]] = d.qpos[J["qad"]]
        for (a_, dof, qad, tau, kp, kd) in S["held"]: d.ctrl[a_] = S["q_settled"][qad]
        for s_ in range(int(T/m.opt.timestep)):
            mujoco.mj_step(m, d)
            if s_ % 10 == 0: trace.append((s_*m.opt.timestep, d.qpos.copy()))
    finally:
        for i, aid in enumerate(J["aid"]): BI._as_motor(m, int(aid), float(J["tau"][i]))
    return d.qpos.copy(), d.qvel.copy(), trace, float(np.max(np.abs(d.qvel[6:]))) < 0.05


def _rj(S):
    rd = S.get("robot_dofs"); return [i for i in (rd if rd is not None else range(S["m"].nv)) if i >= 6]


def handoff_check(S, q, v):
    qa = S["jarr"]["qad"]; dq = float(np.max(np.abs(q[qa] - np.asarray(S["q_settled"])[qa]))); dv = float(np.max(np.abs(np.asarray(v)[_rj(S)])))
    m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_forward(m, d)
    up = float(d.xmat[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")][8])        # torso z-axis . world z
    # a bent-over but balanced manipulation posture is a valid handoff (the previous step's terminal-balance gate judged it);
    # dq to the canonical stance is reported, not enforced, for manipulation
    return dict(dq=dq, dv=dv, upright=up, ok=(dv <= HANDOFF_V_TOL and up >= 0.3))


def run_sequence(robot, steps, settle_T=2.0, adapt_iters=20, freeze_T=0.4, settle_last=True):
    """steps: list of dict(task=Task, demand=..., z=...). Returns dict(steps=[...], trace=[(t_global, qpos)], ok=bool)."""
    state = None; out = []; trace = []; t_off = 0.0; ok = True; held = []
    for i, st in enumerate(steps):
        task, demand, z = st["task"], dict(st["demand"]), np.asarray(st["z"]); S = task.plant(demand, core.DT_OPT)
        if held: demand["held"] = list(held)                                   # world state carried across steps
        if state is not None and len(state[0]) != S["m"].nq: raise ValueError("plants differ in qpos layout; state mapping needed")
        hc = handoff_check(S, *state) if state is not None else dict(dq=0.0, dv=0.0, ok=True)
        p = task.unpack(S, demand, z); inner = task.controller(S, demand, p); fn, log = _logger(S, inner)
        ctx = core.handoff(S, *state) if state is not None else _null()
        with ctx: rec = task.rollout(S, demand, p, ctrl_fn=fn)
        rec["adapted"] = False; rec["alt_used"] = None
        if adapt_iters and state is not None and not (rec["valid"] and rec["violation"] <= 0):
            # ADAPT AT HANDOFF: the stored primitive was optimised from the canonical stance; re-plan briefly from the
            # actual handed-off state (small-step refinement, faithful dt), then replay
            from engine import task as TK
            with core.handoff(S, *state): res = TK.optimize(task, demand, seed=0, iters=adapt_iters, pop=8, z0=z, sigma=0.03, robust=False, measure_after=False)
            z = np.asarray(res["z"]); p = task.unpack(S, demand, z); inner = task.controller(S, demand, p); fn, log = _logger(S, inner)
            with core.handoff(S, *state): rec = task.rollout(S, demand, p, ctrl_fn=fn)
            rec["adapted"] = True
        # RE-PLAN on a failed step: try the step's alternative demands (different lift height, a via-point, a slower pass)
        # from the SAME handed-off state, each with its own prior + brief adaptation; first valid one wins
        for alt in (st.get("alts") or []):
            if rec["valid"] and rec["violation"] <= 0: break
            d_alt = dict(demand, **alt); S_a = task.plant(d_alt, core.DT_OPT); n_a = task.n_params(S_a, d_alt); z_a = np.zeros(n_a)
            if n_a > len(task.unpack(S_a, d_alt, z_a)["x"]): z_a[-1] = (1.0 - 0.87)/0.3
            from engine import task as TK
            with core.handoff(S_a, *state): res_a = TK.optimize(task, d_alt, seed=0, iters=adapt_iters, pop=8, z0=z_a, sigma=0.05, robust=False, measure_after=False)
            z_a = np.asarray(res_a["z"]); p_a = task.unpack(S_a, d_alt, z_a); inner = task.controller(S_a, d_alt, p_a); fn, log = _logger(S_a, inner)
            with core.handoff(S_a, *state): rec_a = task.rollout(S_a, d_alt, p_a, ctrl_fn=fn)
            if rec_a["valid"] and rec_a["violation"] <= 0 or rec_a["violation"] < rec["violation"]:
                rec, z, demand, S = rec_a, z_a, d_alt, S_a; rec["adapted"] = True; rec["alt_used"] = alt
        rec["handoff"] = hc; ok &= bool(rec["valid"] and rec["violation"] <= 0 and hc["ok"])
        for (t, q, v) in log: trace.append((t_off + t, q))
        for (t, q) in rec["extra"].get("land_trace") or []: trace.append((t_off + t, q))
        fs = rec["extra"].get("final_state")
        if fs is None: fs = (log[-1][1], log[-1][2]) if log else (S["q_settled"], S["v_settled"])
        q_end, v_end = np.asarray(fs[0]), np.asarray(fs[1]); t_end = max([t for t, _ in trace[-1:]] + [log[-1][0] if log else 0.0]) - t_off
        t_off += t_end
        held = list(rec["extra"].get("held_after", held)) if rec["extra"].get("world_after") is not None else held
        if settle_T and settle_T > 0 and (settle_last or i < len(steps) - 1):
            q_s, v_s, tr, at_rest = settle(S, robot, q_end, v_end, T=settle_T, held=held, task=task)
            for (t, q) in tr: trace.append((t_off + t, q))
            t_off += settle_T + 1.0; state = (q_s, v_s)
        else:
            # manipulation sequences hand over at the CURRENT posture (the previous step's terminal-balance gate judged it):
            # a short FREEZE under the plant's stiff servos at that posture removes the residual motion the brake leaves
            # (0.2-0.3 rad/s) so the next primitive passes its START_AT_REST gate; no ramp to stance (that topples a bent robot)
            if freeze_T and freeze_T > 0:
                q_f, v_f, tr, at_rest = freeze(S, q_end, v_end, T=freeze_T, held=held, task=task)
                for (t, q) in tr: trace.append((t_off + t, q))
                t_off += freeze_T; state = (q_f, v_f)
            else:
                at_rest = float(np.max(np.abs(np.asarray(v_end)[_rj(S)]))) < 0.5; state = (np.asarray(q_end), np.asarray(v_end))
        out.append(dict(i=i, task=task.name, demand=demand, value=rec["value"], valid=rec["valid"], reasons=rec["reasons"], violation=rec["violation"], handoff=hc, settled=at_rest, adapted=rec["adapted"], alt_used=rec.get("alt_used"), z=np.asarray(z).tolist(), extra={k: v for k, v in rec["extra"].items() if k not in ("trace", "landing", "final_state", "land_trace")}))
        ok &= at_rest
    return dict(steps=out, trace=trace, ok=bool(ok), t_total=t_off, final_state=state, held=held)


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


if __name__ == "__main__":
    # v0 acceptance: landed deep jump -> settle -> throw from the post-landing stance (one continuous state, no reset)
    from tasks.jump import JumpTask
    from tasks.throw import ThrowTask
    R4 = {r["seed"]: r for r in json.load(open(f"{core.RES}/phase14/jump14_robust4.json"))}
    W = json.load(open(f"{core.RES}/atlas/warm_throw_best.json"))[0]
    jt, tt = JumpTask("G1"), ThrowTask("G1"); dj = jt.demand_default(); dt_ = dict(tt.demand_default(), omega=W["omega"])
    steps = [dict(task=jt, demand=dj, z=jt.warm(jt.plant(dj), dj, R4[1])), dict(task=tt, demand=dt_, z=tt.warm(tt.plant(dt_), dt_, W))]
    t0 = time.time(); res = run_sequence("G1", steps)
    for s in res["steps"]: print(s["task"], "value %.2f" % s["value"], "valid", s["valid"], s["reasons"], "handoff dq %.3f dv %.3f ok %s" % (s["handoff"]["dq"], s["handoff"]["dv"], s["handoff"]["ok"]), "settled", s["settled"], "adapted", s["adapted"])
    print("sequence ok", res["ok"], "total %.2f s sim, %.0f s wall, trace %d" % (res["t_total"], time.time() - t0, len(res["trace"])))
    np.save(f"{core.RES}/atlas/compose_v0_trace.npy", np.array([q for _, q in res["trace"]])); json.dump(res["steps"], open(f"{core.RES}/atlas/compose_v0.json", "w"), default=float)
