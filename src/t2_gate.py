"""TRACK T2 (2026-10-17): slip-gated validity for the head-carry hold (and the instrumentation the dynamic predictors are scored with).

The track A hold gate (engine.headcarry.gate) had no foot-slip or COM-drift check: the 32 kg box "held" at 0.5 ms while the feet skated
0.36 m. This module re-runs engine.headcarry.rollout UNCHANGED (same plant, controller, law) with a per-step hook on the universal
back-EMF brake (called exactly once per physics step) that records, from the moment the load is placed:
  foot_slide_max  max over both foot sites of the horizontal displacement from their position at placement (m)
  com_drift_max   max horizontal displacement of the whole-system COM (robot + load) from placement (m)
  joint_sat       per actuated joint, the fraction of hold steps at |tau| >= 0.98 tau_max (where the hold saturates)
Gate (fixed before any T2 run, same 3 cm threshold as the sled/plywood FEET_SLID gate): slip_ok = foot_slide_max <= 0.03 m and
com_drift_max <= 0.10 m; valid_slip = engine.headcarry.gate(...) and slip_ok. NOTE: since commit 23ecf95 (track T1, 12:46, before
the first T2 run) engine.headcarry.gate itself includes T1's support gate (loaded-foot travel <= 2 cm, pelvis drift <= 5 cm) and
engine.headcarry.plant builds a fresh plant per rollout dt; "gate_hc" below is that gate, so valid_slip = both checks.
Nothing about the robot model is changed.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco

SLIP_MAX = 0.03
DRIFT_MAX = 0.10


def rollout_gated(obj, mass, strategy, z, T=10.0, dt=0.001, sway=None, push=None, law=True, t_pre=1.5):
    import branch_i as BI
    from engine import headcarry as HC, leverage as LV
    orig = BI.backemf_brake; st = dict(k=0, f0=None, c0=None, slide=0.0, drift=0.0, sat=None, nh=0, fs=None, tm=None, slide_t=None)
    n_pre = int(round(t_pre/dt))

    def hook(m, d, S, ramp=0.1):
        orig(m, d, S, ramp)
        k = st["k"]; st["k"] += 1
        if st["fs"] is None:
            st["fs"] = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in ("left_foot", "right_foot")]
            st["tm"] = LV.tau_max(m); st["sat"] = np.zeros(m.nu)
            st["ok"] = np.array([np.isfinite(st["tm"][a]) and "hand" not in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a][0])) or "") for a in range(m.nu)])
        if k < n_pre + 1: return            # the object is placed at step n_pre (before this step's integration)
        fp = np.array([d.site_xpos[i][:2] for i in st["fs"]]); c = np.array(d.subtree_com[0][:2])
        if st["f0"] is None: st["f0"] = fp.copy(); st["c0"] = c.copy()
        sl = float(np.max(np.linalg.norm(fp - st["f0"], axis=1)))
        if sl > st["slide"]: st["slide"] = sl
        if sl > SLIP_MAX and st["slide_t"] is None: st["slide_t"] = (k - n_pre)*dt
        st["drift"] = max(st["drift"], float(np.linalg.norm(c - st["c0"])))
        tm = st["tm"]; ok = st["ok"]
        st["sat"][ok] += (np.abs(d.actuator_force[ok]) >= 0.98*tm[ok]); st["nh"] += 1

    BI.backemf_brake = hook
    try:
        r = HC.rollout(obj, mass, strategy, z, T, dt, False, sway, law=law, t_pre=t_pre, push=push)
    finally:
        BI.backemf_brake = orig
    r.pop("trace", None)
    S, _ = HC.plant(obj, mass, dt); m = S["m"]
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a][0])) for a in range(m.nu)]
    sat = (st["sat"]/max(1, st["nh"])) if st["sat"] is not None else np.zeros(m.nu)
    r["joint_sat"] = {names[a]: round(float(sat[a]), 4) for a in np.argsort(-sat)[:6] if sat[a] > 0}
    r.update(foot_slide_max=st["slide"], foot_slide_t=st["slide_t"], com_drift_max=st["drift"],
             slip_ok=bool(st["slide"] <= SLIP_MAX and st["drift"] <= DRIFT_MAX))
    r["gate_hc"] = HC.gate(r, T, strategy, mass); r["valid_slip"] = bool(r["gate_hc"] and r["slip_ok"])
    return r


def eval_job(args):
    """args = (obj, mass, strategy, z, T, dt, disturbance spec or None, law) -> slip-gated result (pool-friendly)."""
    obj, mass, strategy, z, T, dt, dist, law = args
    from engine import headcarry as HC
    try:
        sw, pu = HC.disturbance(dist)
        r = rollout_gated(obj, mass, strategy, np.asarray(z), T, dt, sw, pu, law)
    except Exception as ex:
        return dict(error=f"{type(ex).__name__}: {ex}", valid_slip=False, t_held=0.0, reason="ERROR")
    return r
