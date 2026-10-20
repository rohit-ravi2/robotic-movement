"""Per-subset pose optimization.

The frontier must not be computed against a fixed heuristic pose pair. When a joint
is locked, the poses available to the body change, so crouch and takeoff are solved
PER SUBSET using only that subset's joints; everything else stays at standing.

Constraints on the crouch: every joint centre above ground, and the whole-body centre
of mass horizontally inside the foot support polygon (static balance). Takeoff is
required to be ground-clear only.
"""
import numpy as np, itertools
from scipy.optimize import minimize
import bound as B

G = 9.80665
STANDING_ABS = dict(shank=0.0, thigh=0.0, trunk=0.0, uarm=0.0, farm=0.0)


def standing_joint(b):
    return b.joint_from_abs(STANDING_ABS)


def _pose_penalty(b, q, ankle_h, require_balance, clear=0.03, bal_margin=0.015):
    """Zero when feasible; positive and smooth when not."""
    a = b.abs_from_joint(q)
    b.com_height_abs(a, ankle_h)
    z = b._last_joints
    pen = 0.0
    for v in z.values():
        if v < clear: pen += (clear - v)**2 * 400.0
    for n in B.JOINTS:
        lo, hi = b.j[n]["lo"], b.j[n]["hi"]
        if q[n] < lo: pen += (lo - q[n])**2 * 400.0
        if q[n] > hi: pen += (q[n] - hi)**2 * 400.0
    if require_balance:
        heel, toe = b.support_polygon()
        cx = b.com_x_abs(a, ankle_h)
        if cx < heel + bal_margin: pen += (heel + bal_margin - cx)**2 * 900.0
        if cx > toe - bal_margin:  pen += (cx - (toe - bal_margin))**2 * 900.0
    return pen


def evaluate(b, subset, q0, qT, T_push):
    """Bound value for an explicit pose pair. Mirrors bound.subset_bound's ledger."""
    W = 0.0; per = {}
    for n in subset:
        w_r = b.work_rom(n, q0[n], qT[n])
        w_p = b.peak_power(n, q0[n], qT[n]) * T_push
        w_c = b.work_conditional(n, q0[n], qT[n], T_push)
        u = min(w_r, w_p, w_c)
        W += u
        bd = "rom" if u == w_r else ("power" if u == w_p else "conditional")
        per[n] = dict(rom=w_r, power=w_p, conditional=w_c, used=u, binding=bd,
                      excursion=abs(qT[n]-q0[n]))
    a0, aT = b.abs_from_joint(q0), b.abs_from_joint(qT)
    h0 = B.ANKLE_H_STAND
    hT = B.ANKLE_H_TAKEOFF if "ankle" in subset else B.ANKLE_H_STAND
    z0 = b.com_height_abs(a0, h0); zT = b.com_height_abs(aT, hT)
    dz = zT - z0
    KE = W - b.total_mass * G * dz
    v = float(np.sqrt(max(0.0, 2.0 * KE / b.total_mass)))
    return dict(W_used=W, PE=b.total_mass*G*dz, dz=dz, KE=max(0.0, KE),
                v_takeoff=v, height=v*v/(2*G), per_joint_work=per,
                n_power_bound=sum(1 for x in per.values() if x["binding"] == "power"))


def _feasible(b, q, ankle_h, require_balance, clear=0.03, bal_margin=0.015):
    a = b.abs_from_joint(q)
    b.com_height_abs(a, ankle_h)
    if min(b._last_joints.values()) < clear: return False
    for n in B.JOINTS:
        if not (b.j[n]["lo"] - 1e-9 <= q[n] <= b.j[n]["hi"] + 1e-9): return False
    if require_balance:
        heel, toe = b.support_polygon()
        cx = b.com_x_abs(a, ankle_h)
        if not (heel + bal_margin <= cx <= toe - bal_margin): return False
    return True


def takeoff_pose_for(b, subset, clear=0.03, n_scan=6000, seed0=17):
    """Takeoff is not a free variable. The body extends until extension runs out and the
    feet leave, so the takeoff pose is the maximum-COM-height configuration reachable
    using only this subset's joints, with everything else at standing."""
    subset = tuple(subset); stand = standing_joint(b)
    hT = B.ANKLE_H_TAKEOFF if "ankle" in subset else B.ANKLE_H_STAND
    if not subset:
        return dict(stand), b.com_height_abs(b.abs_from_joint(stand), hT)
    idx = list(subset)
    bnds = [(b.j[n]["lo"], b.j[n]["hi"]) for n in idx]

    def unpack(x):
        q = dict(stand)
        for i, n in enumerate(idx): q[n] = x[i]
        return q

    def negz(x):
        q = unpack(x); a = b.abs_from_joint(q)
        z = b.com_height_abs(a, hT)
        pen = 0.0
        m = min(b._last_joints.values())
        if m < clear: pen += (clear-m)**2*500.0
        return -z + pen

    rng = np.random.default_rng(seed0)
    best = None
    starts = [np.array([0.0]*len(idx))] + [np.array([rng.uniform(lo, hi) for lo, hi in bnds])
                                           for _ in range(12)]
    for x0 in starts:
        res = minimize(negz, x0, method="L-BFGS-B", bounds=bnds,
                       options=dict(maxiter=500, ftol=1e-12))
        q = unpack(res.x); a = b.abs_from_joint(q); z = b.com_height_abs(a, hT)
        if min(b._last_joints.values()) < clear: continue
        if best is None or z > best[1]: best = (q, z)
    if best is None: return dict(stand), b.com_height_abs(b.abs_from_joint(stand), hT)
    return best


def optimize_subset(b, subset, T_push, n_restart=8, seed0=0, n_scan=4000, maxiter=400):
    """Crouch and takeoff poses for this subset, by constrained optimization.

    Feasible starts are found by a cheap kinematic scan (the balance-feasible set is a
    thin manifold that random restarts miss), then refined with SLSQP under explicit
    clearance and balance constraints.
    """
    subset = tuple(subset)
    stand = standing_joint(b)
    if not subset:
        r = evaluate(b, subset, dict(stand), dict(stand), T_push)
        return dict(**r, subset=subset, q_crouch=dict(stand), q_takeoff=dict(stand),
                    restarts=0, n_feasible=1, n_seeds=1, spread=0.0, converged=True)
    idx = list(subset)
    hT = B.ANKLE_H_TAKEOFF if "ankle" in subset else B.ANKLE_H_STAND
    qT_fixed, zT_fixed = takeoff_pose_for(b, subset)

    def unpack(x):
        q0 = dict(stand)
        for i, n in enumerate(idx): q0[n] = x[i]
        return q0, dict(qT_fixed)

    def negKE(x):
        q0, qT = unpack(x)
        return -evaluate(b, subset, q0, qT, T_push)["KE"]

    heel, toe = b.support_polygon(); BM = 0.015; CL = 0.03

    def cons_crouch_clear(x):
        q0, _ = unpack(x); a = b.abs_from_joint(q0); b.com_height_abs(a, B.ANKLE_H_STAND)
        return min(b._last_joints.values()) - CL

    def cons_bal_lo(x):
        q0, _ = unpack(x); a = b.abs_from_joint(q0)
        return b.com_x_abs(a, B.ANKLE_H_STAND) - (heel + BM)

    def cons_bal_hi(x):
        q0, _ = unpack(x); a = b.abs_from_joint(q0)
        return (toe - BM) - b.com_x_abs(a, B.ANKLE_H_STAND)

    def cons_rise(x):
        """Takeoff COM must be at or above crouch COM. Without this the optimizer picks a
        standing 'crouch' and a squatting 'takeoff', letting gravity donate energy, which
        is not a jump."""
        q0, qT = unpack(x)
        z0 = b.com_height_abs(b.abs_from_joint(q0), B.ANKLE_H_STAND)
        zT = b.com_height_abs(b.abs_from_joint(qT), hT)
        return zT - z0

    cons = [dict(type="ineq", fun=f) for f in
            (cons_crouch_clear, cons_bal_lo, cons_bal_hi, cons_rise)]
    bnds = [(b.j[n]["lo"], b.j[n]["hi"]) for n in idx]

    # --- cheap kinematic scan for feasible seeds
    rng = np.random.default_rng(seed0)
    seeds = []
    for _ in range(n_scan):
        x = np.array([rng.uniform(lo, hi) for lo, hi in bnds])
        q0, qT = unpack(x)
        if _feasible(b, q0, B.ANKLE_H_STAND, True):
            e = evaluate(b, subset, q0, qT, T_push)
            if e["dz"] >= 0.0:
                seeds.append((e["KE"], x))
    n_seeds = len(seeds)
    seeds.sort(key=lambda t: -t[0])
    seeds = [x for _, x in seeds[:n_restart]]
    if not seeds:
        r = evaluate(b, subset, dict(stand), dict(stand), T_push)
        return dict(**r, subset=subset, q_crouch=dict(stand), q_takeoff=dict(stand),
                    restarts=n_restart, n_feasible=0, n_seeds=0, spread=float("nan"),
                    converged=False, note="no balance-feasible pose pair found in scan")

    vals = []; best = None
    for x0 in seeds:
        res = minimize(negKE, x0, method="SLSQP", bounds=bnds, constraints=cons,
                       options=dict(maxiter=maxiter, ftol=1e-9))
        q0, qT = unpack(res.x)
        ok = (_feasible(b, q0, B.ANKLE_H_STAND, True)
              and evaluate(b, subset, q0, qT, T_push)["dz"] >= -1e-9)
        if not ok:
            q0, qT = unpack(x0)                       # refinement left the feasible set
        r = evaluate(b, subset, q0, qT, T_push)
        vals.append(r["v_takeoff"])
        if best is None or r["v_takeoff"] > best["v_takeoff"]:
            best = dict(**r, subset=subset, q_crouch=q0, q_takeoff=qT)
    v = np.array(vals)
    best.update(restarts=len(seeds), n_feasible=len(vals), n_seeds=n_seeds,
                spread=float(v.max()-v.min()), sd=float(v.std()),
                converged=bool(len(vals) >= 2 and (v.max()-v.min()) < 0.05*max(1e-9, v.max())))
    return best


def frontier(b, T_push, joints=None, n_restart=10):
    joints = list(B.JOINTS if joints is None else joints)
    out = []
    for r in range(1, len(joints)+1):
        for s in itertools.combinations(joints, r):
            out.append(optimize_subset(b, s, T_push, n_restart=n_restart))
    return out


def best_by_k(rows):
    best = {}
    for r in rows:
        k = len(r["subset"])
        if k not in best or r["v_takeoff"] > best[k]["v_takeoff"]: best[k] = r
    ks = sorted(best)
    marg = {k: best[k]["v_takeoff"] - (best[k-1]["v_takeoff"] if k-1 in best else 0.0) for k in ks}
    return best, marg
