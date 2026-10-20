"""Pose bank: decouple the expensive part (pose optimization, which depends on geometry)
from the cheap part (work ledger, which depends on capacities and masses).

For each subset we store the optimized crouch/takeoff joint angles, the per-body COM
positions at both poses, and the per-joint excursions. Any capacity vector or mass vector
can then be evaluated by arithmetic, with no MuJoCo calls and no re-optimization.

Caveat recorded in every use: the stored poses were optimized under the BASELINE capacities.
Re-evaluating them under different capacities gives a LOWER bound on that variant's true
optimum. To reduce that gap, every subset is evaluated against the pose pairs of all its
subsets as well as its own, and the maximum is taken.
"""
import numpy as np, itertools, json, mujoco
import robot_bound as R

G = 9.80665


def capture(rb, t0, tT, groups):
    """Record everything a later capacity/mass evaluation needs, with no capacities in it."""
    rec = dict(groups=tuple(groups), t0={k: float(v) for k, v in t0.items()},
               tT={k: float(v) for k, v in tT.items()})
    for tag, tv in (("c", t0), ("t", tT)):
        rb.set_pose(tv); rb.drop_to_ground()
        rec[f"xipos_{tag}"] = np.array(rb.d.xipos[1:], dtype=float).tolist()
    # per-joint excursion in each joint's own signed coordinate
    exc = {}
    for g in groups:
        for n in rb.groups[g]:
            t = rb.type_of(n); r = rb.J[n]
            s = -1.0 if (rb.mirror[t] and r["side_geom"] == "right") else 1.0
            q0 = float(np.clip(s*t0.get(t, 0.0), r["lo"], r["hi"]))
            q1 = float(np.clip(s*tT.get(t, 0.0), r["lo"], r["hi"]))
            exc[n] = abs(q1 - q0)
    rec["exc"] = exc
    return rec


def build(rb, T_push, n_scan=900, n_restart=6):
    """Optimize poses for every subset once. Returns a list of pose records."""
    gs = list(rb.groups)
    bank = []; solved = {}
    for r_ in range(1, len(gs)+1):
        for sub in itertools.combinations(gs, r_):
            inherited = []
            for drop in sub:
                key = tuple(x for x in sub if x != drop)
                if key in solved:
                    p = solved[key]
                    inherited.append((p.get("t_crouch", {}), p.get("t_takeoff", {})))
            res = R.optimize_subset(rb, sub, T_push, n_scan=n_scan, n_restart=n_restart,
                                    seed=abs(hash(sub)) % 9999, extra_seeds=inherited)
            solved[sub] = res
            bank.append(capture(rb, res.get("t_crouch", {}), res.get("t_takeoff", {}), sub))
    return bank


def _com(xipos, mass):
    x = np.asarray(xipos); m = np.asarray(mass)[:, None]
    return (m*x).sum(0)/m.sum()


def eval_pose(rec, caps, mass, T_push, joints_in_subset):
    """Work ledger for one stored pose under a capacity map and a mass vector.
    caps: {joint_name: (tau, omega)}. Only joints in `joints_in_subset` may contribute."""
    W = 0.0; per = {}
    for n, dth in rec["exc"].items():
        if n not in joints_in_subset or dth <= 0: continue
        tau, om = caps[n]
        wr = tau*dth
        wp = tau*om/4.0*T_push
        wbar = dth/T_push
        wc = tau*max(0.0, 1.0 - wbar/om)*dth
        u = min(wr, wp, wc); W += u
        per[n] = dict(rom=wr, power=wp, conditional=wc, used=u,
                      binding=("rom" if u == wr else "power" if u == wp else "conditional"))
    M = float(np.sum(mass))
    dz = float(_com(rec["xipos_t"], mass)[2] - _com(rec["xipos_c"], mass)[2])
    KE = W - M*G*dz
    v = float(np.sqrt(max(0.0, 2.0*KE/M)))
    return dict(W=W, dz=dz, PE=M*G*dz, KE=max(0.0, KE), v_takeoff=v, height=v*v/(2*G), per=per)


def frontier_from_bank(bank, groups_all, caps, mass, T_push):
    """Best value for every subset, evaluating each subset against its own pose pair and
    those of all its subsets. Monotone by construction, since a subset's poses are always
    admissible for a superset."""
    by_sub = {}
    for r_ in range(1, len(groups_all)+1):
        for sub in itertools.combinations(groups_all, r_):
            S = set(sub); best = None
            for rec in bank:
                if not set(rec["groups"]) <= S: continue
                joints = set(rec["exc"])
                e = eval_pose(rec, caps, mass, T_push, joints)
                if best is None or e["v_takeoff"] > best["v_takeoff"]:
                    best = dict(e, subset=sub, pose_from=rec["groups"])
            if best is None:
                best = dict(W=0.0, dz=0.0, PE=0.0, KE=0.0, v_takeoff=0.0, height=0.0,
                            per={}, subset=sub, pose_from=None)
            by_sub[sub] = best
    return list(by_sub.values())


def caps_from(rb, scale=None):
    """Baseline capacity map, optionally scaled per GROUP: scale={'knees':(tau_x, om_x)}."""
    out = {}
    for g, names in rb.groups.items():
        st, so = (scale or {}).get(g, (1.0, 1.0))
        for n in names:
            out[n] = (rb.J[n]["tau"]*st, rb.J[n]["omega"]*so)
    return out


def mass_of(rb):
    return np.array(rb.m.body_mass[1:], dtype=float)


def _group_names(d, g):
    """Joint names in a group, from a saved bank dict."""
    return sorted({n for rec in d["bank"] for n in rec["exc"]
                   if n in d.get("group_index", {}).get(g, [])})


def caps_from_scaled(d, scale):
    """Capacity map from a saved bank dict, scaling named groups by (tau_x, omega_x)."""
    gi = d["group_index"]
    out = {}
    for g, names in gi.items():
        st, so = scale.get(g, (1.0, 1.0))
        for n in names:
            t, o = d["caps"][n]
            out[n] = (t*st, o*so)
    return out
