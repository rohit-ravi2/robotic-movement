"""Analytic capacity bound for a real robot model. No optimizer, no contact model.

Same two-branch ledger as the planar human bound:
    W_i <= min( integral of tau_max_i over the traversed range , (tau_i*omega_i/4)*T )
Takeoff kinetic energy is bounded by summed work minus the potential-energy cost of
raising the centre of mass from crouch to takeoff.

Difference from the human model: electric actuators have no muscle length-tension
relation, so the torque-angle shape function is flat (f_angle = 1). This is stated
rather than assumed silently.
"""
import numpy as np, yaml, itertools, mujoco
from scipy.optimize import minimize

G = 9.80665
# Optional hard static-equilibrium constraint on the crouch (Gate 1). When set, it is a
# callable q -> bool; crouch_ok() then also requires the start pose to be holdable against
# gravity within actuator torque and friction limits. Set by rebuild_bank.py.
CROUCH_CERTIFIER = None
CROUCH_MARGIN = 0.015     # COM-inside-support tolerance; sensitivity reported in Task 2d


class RobotBody:
    SAGITTAL = ("pitch", "knee", "elbow")   # DOF that can move the COM vertically

    def __init__(self, mjcf, spec, capacity="peak", lock_groups=("wrists",), task="sagittal_jump"):
        self.m = mujoco.MjModel.from_xml_path(mjcf)
        self.d = mujoco.MjData(self.m)
        self.spec = yaml.safe_load(open(spec))
        self.capacity = capacity
        self.M = float(sum(self.m.body_mass))
        self.lock_groups = set(lock_groups)
        key = "tau_max_peak" if capacity == "peak" else "tau_max_continuous"
        self.J = {}
        for jn, rec in self.spec["joints"].items():
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0: continue
            self.J[jn] = dict(id=jid, qadr=int(self.m.jnt_qposadr[jid]),
                              lo=float(self.m.jnt_range[jid][0]), hi=float(self.m.jnt_range[jid][1]),
                              tau=float(rec[key]["value"]), omega=float(rec["omega_max"]["value"]),
                              group=rec["group"], side=rec["side"],
                              axis=[float(x) for x in self.m.jnt_axis[jid]])
        self.qpos_home0 = self._keyframe("home")
        self.task = task
        self.groups = {g: [n for n, r in self.J.items() if r["group"] == g]
                       for g in self.spec["joint_groups"] if g not in self.lock_groups}
        if task == "sagittal_jump":
            # With bilateral symmetry enforced, roll and yaw contributions to VERTICAL COM
            # motion cancel between left and right, so they carry no work toward takeoff
            # velocity. Sagittal DOF are identified by JOINT AXIS (rotation about body y),
            # not by name: H1's ankle pitch joint is simply called "ankle".
            self.groups = {g: [n for n in v if abs(self.J[n]["axis"][1]) > 0.9]
                           for g, v in self.groups.items()}
        self.excluded_dof = sorted({n for g, v in
                                    {gg: [n for n, r in self.J.items() if r["group"] == gg]
                                     for gg in self.spec["joint_groups"] if gg not in self.lock_groups}.items()
                                    for n in v} - {n for v in self.groups.values() for n in v})
        self.empty_groups = [g for g, v in self.groups.items() if not v]
        self.groups = {g: v for g, v in self.groups.items() if v}
        # Bilateral pairing from DECLARED GEOMETRY, not from name prefixes.
        # Two joints pair iff they share a rotation axis and sit at mirrored y with matching
        # x and z in the home pose. Names are used only as labels afterwards.
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = self.qpos_home0
        mujoco.mj_forward(self.m, self.d)
        anchor = {}
        for n, r in self.J.items():
            a = self.d.xanchor[r["id"]]
            anchor[n] = (round(float(a[0]), 3), round(float(a[1]), 3), round(float(a[2]), 3))
            r["anchor"] = [float(x) for x in a]
            r["side_geom"] = ("left" if a[1] > 1e-4 else "right" if a[1] < -1e-4 else "center")
        self.types = {}
        keyof = {}
        for n, r in self.J.items():
            ax = tuple(round(float(x), 2) for x in r["axis"])
            x0, y0, z0 = anchor[n]
            keyof[n] = (ax, round(abs(y0), 3), x0, z0)
        for n in self.J:
            self.types.setdefault(keyof[n], []).append(n)
        # label each geometric pair with a readable name taken from a member
        self.types = {min(v, key=len).replace("left_", "").replace("right_", ""): v
                      for v in self.types.values()}
        for t, v in self.types.items():
            for n in v: self.J[n]["type"] = t
        # Mirroring is decided by JOINT AXIS, not by whether the ranges look symmetric.
        # A sagittal (pitch, axis y) joint moves the same way on both sides in a symmetric
        # movement; roll and yaw must take opposite signs. Inferring this from range symmetry
        # was wrong: H1's hip pitch range is exactly +/-1.57, which made it look mirrored and
        # put the robot into a split whenever the hips were free.
        self.mirror = {}
        for t, names in self.types.items():
            if len(names) == 2:
                ax = self.J[names[0]]["axis"]
                self.mirror[t] = abs(ax[1]) <= 0.9      # non-sagittal -> mirrored
            else:
                self.mirror[t] = False
        self.qpos_home = self.qpos_home0
        self.foot_geoms = self._foot_geoms()
        self._assert_declared()

    def _keyframe(self, name):
        kid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, name)
        if kid >= 0: return np.array(self.m.key_qpos[kid])
        q = np.zeros(self.m.nq); q[3] = 1.0
        mujoco.mj_resetData(self.m, self.d); return np.array(self.d.qpos)

    def _collision_geoms(self):
        """MJX models set contype=0 and rely on explicit contact pairs, so contype is not
        a usable filter. Take every geom that appears in a pair with the floor; if the model
        defines no pairs (H1), fall back to contype."""
        fl = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        out = []
        for p in range(self.m.npair):
            gs = [int(self.m.pair_geom1[p]), int(self.m.pair_geom2[p])]
            if fl in gs: out.append([x for x in gs if x != fl][0])
        if not out:
            out = [g for g in range(self.m.ngeom)
                   if self.m.geom_contype[g] != 0 and self.m.geom_bodyid[g] != 0]
        return sorted(set(out))

    def _foot_geoms(self, tol=0.02):
        """Feet identified by ACTUAL GROUND CONTACT at the home pose, not by body name."""
        self.coll_geoms = self._collision_geoms()
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = self.qpos_home0
        mujoco.mj_forward(self.m, self.d)
        lo = min(self._geom_low(g) for g in self.coll_geoms)
        out = [g for g in self.coll_geoms if self._geom_low(g) <= lo + tol]
        self.foot_bodies = sorted({mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY,
                                                     self.m.geom_bodyid[g]) for g in out})
        return out

    def type_of(self, jn): return self.J[jn]["type"]

    def groups_of_type(self, t):
        return [n for n in self.J if self.J[n].get("type") == t
                and any(n in v for v in self.groups.values())]

    def _assert_declared(self):
        """Assertions for properties that cannot be read from a declared source."""
        self.assertions = []
        for n, r in self.J.items():
            ax = np.array(r["axis"])
            near = np.max(np.abs(ax)) > 0.99 and np.sum(np.abs(ax) > 0.01) == 1
            if not near:
                self.assertions.append(f"joint {n} axis {np.round(ax,3).tolist()} is not canonical; "
                                       f"the sagittal test |axis_y|>0.9 is an assertion here")
        for t, v in self.types.items():
            if len(v) == 2:
                s0, s1 = self.J[v[0]]["side_geom"], self.J[v[1]]["side_geom"]
                if {s0, s1} != {"left", "right"}:
                    self.assertions.append(f"pair {t} sides from geometry are {s0}/{s1}")
        return self.assertions

    def group_types(self, group):
        seen = []
        for n in self.groups.get(group, []):
            t = self.type_of(n)
            if t not in seen: seen.append(t)
        return seen

    def set_pose(self, tvals):
        """tvals: {joint_type: value}. Applies bilateral symmetry and clips to range."""
        q = np.array(self.qpos_home)
        for t, v in tvals.items():
            for n in self.types.get(t, []):
                r = self.J[n]
                s = -1.0 if (self.mirror[t] and r["side_geom"] == "right") else 1.0
                q[r["qadr"]] = float(np.clip(s*v, r["lo"], r["hi"]))
        self.d.qpos[:] = q; self.d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)
        return q

    def com(self):
        mm = self.m.body_mass[1:, None]
        return np.asarray((mm*self.d.xipos[1:]).sum(0)/self.M, dtype=float)

    def _geom_low(self, g):
        """Lowest world-z point of a geom, honouring box orientation rather than rbound."""
        t = self.m.geom_type[g]; c = self.d.geom_xpos[g]; sz = self.m.geom_size[g]
        Rm = self.d.geom_xmat[g].reshape(3, 3)
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            ext = abs(Rm[2, 0])*sz[0] + abs(Rm[2, 1])*sz[1] + abs(Rm[2, 2])*sz[2]
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            ext = sz[0]
        elif t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            import plant as _P
            return _P.geom_low(self.m, self.d, g)          # single audited implementation
        else:
            import plant as _P
            return _P.geom_low(self.m, self.d, g)          # exact (mesh/ellipsoid) or error; never geom_rbound
        return float(c[2] - ext)

    def lowest_z(self):
        return min((self._geom_low(g) for g in self.coll_geoms), default=0.0)

    def upright(self):
        """Cosine between the pelvis z-axis and world up. 1 = upright."""
        return float(self.d.xmat[1].reshape(3, 3)[2, 2])

    def drop_to_ground(self):
        """Translate the free base so the lowest collision geom sits on z=0."""
        lo = self.lowest_z()
        self.d.qpos[2] -= lo
        mujoco.mj_forward(self.m, self.d)

    def support_polygon(self, contact_tol=0.02):
        """Axis-aligned extent of foot geoms that are within contact_tol of the ground."""
        xs, ys = [], []
        for g in self.foot_geoms:
            if self._geom_low(g) > self.lowest_z() + contact_tol: continue
            c = self.d.geom_xpos[g]; sz = self.m.geom_size[g]
            Rm = self.d.geom_xmat[g].reshape(3, 3)
            t = self.m.geom_type[g]
            if t == mujoco.mjtGeom.mjGEOM_BOX:
                ex = abs(Rm[0, 0])*sz[0] + abs(Rm[0, 1])*sz[1] + abs(Rm[0, 2])*sz[2]
                ey = abs(Rm[1, 0])*sz[0] + abs(Rm[1, 1])*sz[1] + abs(Rm[1, 2])*sz[2]
            elif t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
                # true horizontal extent of the capsule axis, NOT the circumscribed sphere.
                # Using geom_rbound here inflated the G1 foot support by 45 mm on the heel
                # side and let the pose optimizer accept leaning crouches whose centre of
                # mass was outside the real support. Fourth occurrence of this bug class.
                # A capsule resting on a plane contacts along its AXIS SEGMENT; the radius
                # does not widen the contact patch, it only sets its height. Including it
                # (or using geom_rbound) inflated the G1 foot support by 40 mm.
                ex = abs(Rm[0, 2])*sz[1]
                ey = abs(Rm[1, 2])*sz[1]
            else:
                raise NotImplementedError("support extent for this geom type not implemented; geom_rbound is forbidden")
            xs += [c[0]-ex, c[0]+ex]; ys += [c[1]-ey, c[1]+ey]
        if not xs: return (-0.02, 0.02, -0.02, 0.02)
        return (min(xs), max(xs), min(ys), max(ys))

    # ---- per-joint work, flat torque-angle (electric actuator)
    def work_rom(self, jn, th0, th1):
        return float(self.J[jn]["tau"]*abs(th1-th0))

    def peak_power(self, jn):
        r = self.J[jn]; return float(r["tau"]*r["omega"]/4.0)

    def work_conditional(self, jn, th0, th1, t_contact):
        """Upper bound conditional on the movement completing in t_contact.

        The joint must sweep |dtheta| within t_contact, so its mean angular speed is at
        least |dtheta|/t_contact. Evaluating the torque-speed envelope at that mean speed
        instead of at omega -> 0 keeps this a rigorous upper bound, conditional on a
        quantity the power branch already assumes.
        """
        r = self.J[jn]; dth = abs(th1-th0)
        if t_contact <= 0 or dth == 0: return 0.0
        wbar = dth/t_contact
        frac = max(0.0, 1.0 - wbar/r["omega"])
        return float(r["tau"]*frac*dth)


def evaluate(rb, groups, t0, tT, T_push):
    """Energy ledger for an explicit type-angle pose pair."""
    W = 0.0; per = {}
    names = [n for g in groups for n in rb.groups[g]]
    for n in names:
        t = rb.type_of(n)
        a0 = t0.get(t, 0.0); aT = tT.get(t, 0.0)
        r = rb.J[n]
        s = -1.0 if (rb.mirror[t] and r["side_geom"] == "right") else 1.0
        q0 = float(np.clip(s*a0, r["lo"], r["hi"])); q1 = float(np.clip(s*aT, r["lo"], r["hi"]))
        wr = rb.work_rom(n, q0, q1); wp = rb.peak_power(n)*T_push
        wc = rb.work_conditional(n, q0, q1, T_push)
        u = min(wr, wp, wc); W += u
        bnd = "rom" if u == wr else ("power" if u == wp else "conditional")
        per[n] = dict(rom=wr, power=wp, conditional=wc, used=u, binding=bnd,
                      excursion=abs(q1-q0))
    rb.set_pose(t0); rb.drop_to_ground(); z0 = rb.com()[2]
    rb.set_pose(tT); rb.drop_to_ground(); zT = rb.com()[2]
    dz = zT - z0
    KE = W - rb.M*G*dz
    v = float(np.sqrt(max(0.0, 2.0*KE/rb.M)))
    return dict(W_used=W, PE=rb.M*G*dz, dz=dz, KE=max(0.0, KE), v_takeoff=v,
                height=v*v/(2*G), per_joint_work=per,
                n_power_bound=sum(1 for x in per.values() if x["binding"] == "power"),
                n_cond_bound=sum(1 for x in per.values() if x["binding"] == "conditional"))


def path_monotone_z(rb, t0, tT, n=12, tol=1e-4):
    """COM height must not decrease anywhere along the straight-line joint-space path
    from crouch to takeoff. This is what blocks a joint from harvesting work over a large
    sweep that produces no net rise."""
    keys = sorted(set(t0) | set(tT))
    zs = []
    for i in range(n+1):
        a = i/n
        tv = {k: (1-a)*t0.get(k, 0.0) + a*tT.get(k, 0.0) for k in keys}
        rb.set_pose(tv); rb.drop_to_ground()
        zs.append(rb.com()[2])
    d = np.diff(zs)
    return bool(np.all(d >= -tol)), zs


def _bounds_for(rb, groups):
    out = []
    for g in groups:
        for t in rb.group_types(g):
            n = rb.types[t][0]; r = rb.J[n]
            out.append((t, r["lo"], r["hi"]))
    return out


def takeoff_pose_for(rb, groups, n_start=10, seed=3, clear=0.0):
    """Max-COM-height configuration reachable with these groups; others at home."""
    bs = _bounds_for(rb, groups)
    if not bs:
        rb.set_pose({}); rb.drop_to_ground(); return {}, rb.com()[2]
    rng = np.random.default_rng(seed)
    UP = 0.85
    def negz(x):
        rb.set_pose({t: v for (t, _, _), v in zip(bs, x)}); rb.drop_to_ground()
        pen = 0.0
        u = rb.upright()
        if u < UP: pen += (UP-u)**2*50.0
        return -rb.com()[2] + pen
    best = None
    starts = [np.zeros(len(bs))] + [np.array([rng.uniform(lo, hi) for _, lo, hi in bs]) for _ in range(n_start)]
    for x0 in starts:
        res = minimize(negz, x0, method="L-BFGS-B",
                       bounds=[(lo, hi) for _, lo, hi in bs], options=dict(maxiter=300))
        tv = {t: v for (t, _, _), v in zip(bs, res.x)}
        rb.set_pose(tv); rb.drop_to_ground(); z = rb.com()[2]
        if rb.upright() < UP: continue
        if best is None or z > best[1]: best = (tv, z)
    return best


def optimize_subset(rb, groups, T_push, n_scan=1500, n_restart=6, seed=0, maxiter=250,
                    extra_seeds=None):
    """Crouch AND takeoff are free within two constraints that were established earlier:

      (1) takeoff COM height >= crouch COM height   -- blocks the direction exploit, where a
          high 'crouch' and a low 'takeoff' let gravity donate the energy;
      (2) COM height is non-decreasing along the path -- blocks the equal-height maximal-sweep
          exploit, where a joint harvests work over a large sweep producing no net rise.

    Takeoff is NOT forced to the maximum reachable height. Forcing it broke set inclusion:
    a superset was compelled to extend the added joint and so could not reproduce the
    subset's solution, which made the frontier non-monotone. With takeoff free, any superset
    can hold the extra joint at standing in both poses and reproduce the subset exactly, so
    monotonicity follows by construction.
    """
    groups = tuple(groups)
    if not groups:
        r = evaluate(rb, groups, {}, {}, T_push)
        return dict(**r, subset=groups, restarts=0, n_seeds=1, spread=0.0, converged=True,
                    t_crouch={}, t_takeoff={})
    bs = _bounds_for(rb, groups)

    def crouch_ok(tv):
        rb.set_pose(tv); rb.drop_to_ground()
        if rb.upright() < 0.85: return False
        c = rb.com(); xl, xh, yl_, yh_ = rb.support_polygon()
        mx = CROUCH_MARGIN
        if not ((xl+mx <= c[0] <= xh-mx) and (yl_+mx <= c[1] <= yh_-mx)): return False
        if CROUCH_CERTIFIER is not None:
            return bool(CROUCH_CERTIFIER(np.array(rb.d.qpos)))
        return True

    def takeoff_ok(tv):
        rb.set_pose(tv); rb.drop_to_ground()
        return rb.upright() >= 0.85

    def full_ok(t0, tT):
        if not crouch_ok(t0) or not takeoff_ok(tT): return False
        e = evaluate(rb, groups, t0, tT, T_push)
        if e["dz"] < 0.0: return False
        ok, _ = path_monotone_z(rb, t0, tT)
        return ok

    rng = np.random.default_rng(seed)
    stand = {t: 0.0 for t, _, _ in bs}
    tmax, _zmax = takeoff_pose_for(rb, groups)
    seeds = []
    # Seed takeoff by interpolating the crouch toward this subset's max-height pose. Random
    # pose PAIRS almost never satisfy balance and path-monotonicity simultaneously; this
    # generates physically sensible pairs. Takeoff stays FREE in the refinement, so the
    # set-inclusion guarantee is untouched.
    for _ in range(n_scan):
        t0 = {t: rng.uniform(lo, hi) for t, lo, hi in bs}
        if not crouch_ok(t0): continue
        for al in (0.25, 0.5, 0.75, 1.0):
            tT = {t: (1-al)*t0[t] + al*tmax.get(t, 0.0) for t, _, _ in bs}
            if full_ok(t0, tT):
                seeds.append((evaluate(rb, groups, t0, tT, T_push)["KE"], t0, tT))
    n_seeds = len(seeds)
    # Warm starts inherited from proper subsets: the extra joints sit at their subset values,
    # so bound(S) can never numerically fall below bound(S') for S' subset of S.
    for t0s, tTs in (extra_seeds or []):
        t0 = {t: t0s.get(t, 0.0) for t, _, _ in bs}
        tT = {t: tTs.get(t, 0.0) for t, _, _ in bs}
        if full_ok(t0, tT):
            seeds.append((evaluate(rb, groups, t0, tT, T_push)["KE"], t0, tT))
    # the standing/standing pair is always feasible and gives KE = 0; it guarantees that a
    # superset can never score below a subset
    seeds.append((0.0, dict(stand), dict(stand)))
    seeds.sort(key=lambda z: -z[0]); seeds = seeds[:n_restart]

    def negKE(x):
        n = len(bs)
        t0 = {t: v for (t, _, _), v in zip(bs, x[:n])}
        tT = {t: v for (t, _, _), v in zip(bs, x[n:])}
        if not full_ok(t0, tT): return 1e6
        return -evaluate(rb, groups, t0, tT, T_push)["KE"]

    vals = []; best = None
    for _, t0s, tTs in seeds:
        x0 = np.array([t0s[t] for t, _, _ in bs] + [tTs[t] for t, _, _ in bs])
        res = minimize(negKE, x0, method="Nelder-Mead",
                       options=dict(maxiter=maxiter, fatol=1e-8, xatol=1e-6))
        n = len(bs)
        t0 = {t: v for (t, _, _), v in zip(bs, res.x[:n])}
        tT = {t: v for (t, _, _), v in zip(bs, res.x[n:])}
        if not full_ok(t0, tT): t0, tT = t0s, tTs
        e = evaluate(rb, groups, t0, tT, T_push)
        vals.append(e["v_takeoff"])
        if best is None or e["v_takeoff"] > best["v_takeoff"]:
            best = dict(**e, subset=groups, t_crouch=t0, t_takeoff=tT)
    v = np.array(vals)
    best.update(restarts=len(seeds), n_seeds=n_seeds, spread=float(v.max()-v.min()),
                sd=float(v.std()),
                converged=bool(len(vals) >= 2 and (v.max()-v.min()) < 0.05*max(1e-9, v.max())))
    return best


def frontier(rb, T_push, n_scan=1200, n_restart=6, maxiter=250):
    """Subsets in ascending size, each warm-started from its (k-1)-subsets."""
    gs = list(rb.groups)
    solved = {}
    for r_ in range(1, len(gs)+1):
        for sub in itertools.combinations(gs, r_):
            inherited = []
            for drop in sub:
                key = tuple(x for x in sub if x != drop)
                if key in solved:
                    p = solved[key]
                    inherited.append((p.get("t_crouch", {}), p.get("t_takeoff", {})))
            res = optimize_subset(rb, sub, T_push, n_scan=n_scan, n_restart=n_restart,
                                  seed=abs(hash(sub)) % 9999, maxiter=maxiter,
                                  extra_seeds=inherited)
            solved[sub] = res
    return list(solved.values())


def enforce_monotone(rows):
    """Lift each subset's estimate to the best of itself and all its proper subsets.

    Justified by set inclusion: bound(S) >= bound(S') exactly whenever S' is a subset of S,
    so a numerical estimate that falls below a subset's estimate is an under-converged solve,
    not a property of the body. Lifting moves the estimate strictly closer to the true
    maximum. Returns (rows, lifted) where lifted records what moved and by how much.
    """
    by = {frozenset(r["subset"]): r for r in rows}
    lifted = []
    for key in sorted(by, key=len):
        r = by[key]
        for other in by:
            if other < key and by[other]["v_takeoff"] > r["v_takeoff"] + 1e-12:
                lifted.append(dict(subset=sorted(key), from_v=r["v_takeoff"],
                                   to_v=by[other]["v_takeoff"],
                                   source=sorted(other),
                                   delta=by[other]["v_takeoff"]-r["v_takeoff"]))
                r = dict(r); r["v_takeoff"] = by[other]["v_takeoff"]
                r["height"] = r["v_takeoff"]**2/(2*G); r["lifted_from"] = sorted(other)
                by[key] = r
    return list(by.values()), lifted


def required_depth(rows, demands):
    """PRIMARY OUTPUT. required_depth(D) = min |S| such that bound(S) >= D, with the
    identity of the minimal sufficient subset. Monotone in D by construction."""
    out = []
    for D in demands:
        ok = [r for r in rows if r["v_takeoff"] >= D]
        if not ok:
            out.append(dict(demand=float(D), depth=None, subset=None, v=None)); continue
        kmin = min(len(r["subset"]) for r in ok)
        cand = [r for r in ok if len(r["subset"]) == kmin]
        bestr = max(cand, key=lambda r: r["v_takeoff"])
        out.append(dict(demand=float(D), depth=kmin, subset=list(bestr["subset"]),
                        v=float(bestr["v_takeoff"])))
    return out


def check_monotone(rows, tol=1e-6):
    """Assert bound(S') >= bound(S) for every nested pair S subset of S'. A violation is a
    bug in the formulation, not a finding."""
    by = {frozenset(r["subset"]): r["v_takeoff"] for r in rows}
    bad = []
    for a, va in by.items():
        for b_, vb in by.items():
            if a < b_ and vb + tol < va:
                bad.append(("+".join(sorted(a)), va, "+".join(sorted(b_)), vb, va-vb))
    return bad


def best_by_k(rows):
    best = {}
    for r in rows:
        k = len(r["subset"])
        if k not in best or r["v_takeoff"] > best[k]["v_takeoff"]: best[k] = r
    ks = sorted(best)
    marg = {k: best[k]["v_takeoff"] - (best[k-1]["v_takeoff"] if k-1 in best else 0.0) for k in ks}
    return best, marg
