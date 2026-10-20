"""PlantContract, PlantContractVerifier, and StaticEquilibriumCertificate.

Phase 7 found that simulation semantics silently differed from experimental semantics for two
phases. Nothing here trusts a configuration string: every check interrogates the loaded model.
"""
import hashlib, json, os
import numpy as np, mujoco
from scipy.optimize import linprog

G = 9.80665
ACT_KIND = {0: "torque_motor", 1: "affine_position_servo"}


def file_hash(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


class PlantContract:
    """Declares what plant an experiment assumes. Verified against the loaded model."""

    def __init__(self, mjcf, actuator_semantics="torque", controlled=None, locked=None,
                 gravity=-G, penetration_tol=0.002, mu=None, timestep=None):
        self.mjcf = mjcf
        self.asset_hash = file_hash(mjcf)
        self.actuator_semantics = actuator_semantics
        self.controlled = list(controlled or [])
        self.locked = list(locked or [])
        self.gravity = gravity
        self.penetration_tol = penetration_tol
        self.mu = mu
        self.timestep = timestep

    def hash(self):
        d = dict(asset=self.asset_hash, sem=self.actuator_semantics,
                 ctl=sorted(self.controlled), lock=sorted(self.locked),
                 g=self.gravity, pen=self.penetration_tol, mu=self.mu, dt=self.timestep)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    def to_dict(self):
        return dict(mjcf=self.mjcf, asset_hash=self.asset_hash,
                    actuator_semantics=self.actuator_semantics, controlled=self.controlled,
                    locked=self.locked, gravity=self.gravity,
                    penetration_tol=self.penetration_tol, mu=self.mu,
                    timestep=self.timestep, contract_hash=self.hash())


def com_velocity(m, d):
    """Exact whole-body COM linear velocity, J_com(q) @ qdot.

    Defect P8 (found 2026-09-21): Phases 6-7 and Gate A computed it as
    sum_i m_i * cvel[i, 3:6] / M. cvel's linear part is the velocity of the point at the
    tree's COM rigidly attached to body i, NOT body i's own COM velocity, so fast limb rotation
    inflates it. The optimizer learned to swing the arms to game the metric: a solve reported
    2.54 m/s while the ground impulse implied 1.52 m/s."""
    jacp = np.zeros((3, m.nv))
    mujoco.mj_jacSubtreeCom(m, d, jacp, 0)
    return jacp @ d.qvel


def actuator_kinds(m):
    return [ACT_KIND.get(int(m.actuator_biastype[a]), f"other{int(m.actuator_biastype[a])}")
            for a in range(m.nu)]


def geom_low(m, d, g):
    """True lowest world-z of a geom. geom_rbound is the circumscribed sphere and is wrong
    for boxes and long capsules; using it caused the Phase 7 elevated-spawn defect."""
    t = m.geom_type[g]; c = d.geom_xpos[g]; sz = m.geom_size[g]
    R = d.geom_xmat[g].reshape(3, 3)
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        e = abs(R[2, 0])*sz[0]+abs(R[2, 1])*sz[1]+abs(R[2, 2])*sz[2]
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE: e = sz[0]
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        e = abs(R[2, 2])*sz[1] + sz[0]
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:          # flat end discs: rim, not a hemisphere
        e = abs(R[2, 2])*sz[1] + sz[0]*np.sqrt(max(0.0, 1.0 - R[2, 2]**2))
    elif t == mujoco.mjtGeom.mjGEOM_MESH:
        return mesh_low(m, d, g)
    elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        # support function of an ellipsoid along -z: sqrt(sum (R[2,k] * a_k)^2)
        e = float(np.sqrt(sum((R[2, k]*sz[k])**2 for k in range(3))))
    else:
        raise NotImplementedError(f"geom type {int(t)}: no exact extent; geom_rbound is forbidden (P3/P4/P11 class)")
    return float(c[2]-e)


def mesh_low(m, d, g):
    """Exact lowest world-z of a mesh geom from its vertices (Phase 9 inference audit: replaces the
    geom_rbound fallback, the bug family behind P3, P4 and G-a/G-b)."""
    mid = int(m.geom_dataid[g]); a = int(m.mesh_vertadr[mid]); n = int(m.mesh_vertnum[mid])
    V = m.mesh_vert[a:a+n]; R = d.geom_xmat[g].reshape(3, 3)
    return float(d.geom_xpos[g][2] + np.min(V @ R[2]))


class PlantContractVerifier:
    """PCTV-1..8. Each returns (pass, detail)."""

    def __init__(self, m, d, contract):
        self.m, self.d, self.c = m, d, contract
        self.results = {}

    def _rec(self, k, ok, detail):
        self.results[k] = dict(passed=bool(ok), detail=detail); return ok

    def pctv1_actuator_semantics(self):
        """Command a known value; check the realized generalized force is that torque."""
        m, d = self.m, self.d
        kinds = set(actuator_kinds(m))
        if isinstance(self.c.actuator_semantics, dict):
            # per-joint declaration: {"torque": [...], "implicit_position_hold": [...]}
            want = {}
            for sem, joints in self.c.actuator_semantics.items():
                for j in joints: want[j] = sem
            bad = []
            for a in range(m.nu):
                jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a][0]))
                k = actuator_kinds(m)[a]
                exp = {"torque": "torque_motor",
                       "implicit_position_hold": "affine_position_servo"}.get(want.get(jn), "?")
                if k != exp: bad.append((jn, k, want.get(jn)))
            if bad:
                return self._rec("PCTV-1", False, f"semantics mismatch {bad[:4]}")
            a0 = [a for a in range(m.nu) if actuator_kinds(m)[a] == "torque_motor"]
            if not a0:
                return self._rec("PCTV-1", True, "no torque motors declared")
            a = a0[0]; d.ctrl[a] = 0.5*float(m.actuator_ctrlrange[a][1])
            mujoco.mj_forward(m, d); got = float(d.actuator_force[a])
            ok = abs(got - d.ctrl[a]) < 1e-6
            return self._rec("PCTV-1", ok, f"mixed contract matches model; motor probe {got:.3f}")
        if self.c.actuator_semantics == "torque" and kinds != {"torque_motor"}:
            return self._rec("PCTV-1", False,
                             f"contract declares torque but model has {sorted(kinds)}")
        mujoco.mj_forward(m, d)
        probe = min(10.0, float(m.actuator_ctrlrange[0][1]))
        d.ctrl[:] = 0.0; d.ctrl[0] = probe
        mujoco.mj_forward(m, d)
        got = float(d.actuator_force[0])
        d.ctrl[:] = 0.0
        ok = abs(got - probe) < 1e-6
        return self._rec("PCTV-1", ok,
                         f"commanded {probe:.3f} -> actuator_force {got:.6f}; kinds {sorted(kinds)}")

    def pctv2_control_coverage(self):
        m = self.m
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a][0]))
                 for a in range(m.nu)]
        declared = set(self.c.controlled) | set(self.c.locked)
        missing = [n for n in names if n not in declared]
        unknown = [n for n in declared if n not in names]
        ok = not missing and not unknown
        return self._rec("PCTV-2", ok,
                         f"{len(names)} actuators; undeclared {missing[:4]}; unknown {unknown[:4]}")

    def collision_geoms(self):
        """Only geoms that can actually collide. MJX models set contype=0 and rely on explicit
        contact pairs, and they also carry visual meshes that hang below the collision hull;
        including those made ground placement look 107 mm wrong when it was correct."""
        m = self.m
        fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        out = []
        for p in range(m.npair):
            gs = [int(m.pair_geom1[p]), int(m.pair_geom2[p])]
            if fl in gs: out.append([x for x in gs if x != fl][0])
        if not out:
            out = [g for g in range(m.ngeom)
                   if m.geom_contype[g] != 0 and m.geom_bodyid[g] != 0]
        return sorted(set(out))

    def pctv3_ground_placement(self, q):
        m, d = self.m, self.d
        d.qpos[:] = q; d.qvel[:] = 0; mujoco.mj_forward(m, d)
        lows = [geom_low(m, d, g) for g in self.collision_geoms()]
        lo = min(lows)
        mujoco.mj_forward(m, d)
        pen = max([-float(d.contact[i].dist) for i in range(d.ncon)] + [0.0])
        ok = (abs(lo) < 0.005) and (pen < self.c.penetration_tol)
        return self._rec("PCTV-3", ok,
                         f"lowest geom {lo*1000:+.2f} mm, penetration {pen*1000:.2f} mm, "
                         f"tol {self.c.penetration_tol*1000:.1f} mm, ncon {d.ncon}")

    def pctv4_zero_control(self, q, T=0.3):
        m, d = self.m, self.d
        d.qpos[:] = q; d.qvel[:] = 0; mujoco.mj_forward(m, d)
        M = float(sum(m.body_mass[1:]))
        z0 = float((m.body_mass[1:, None]*d.xipos[1:]).sum(0)[2]/M)
        for _ in range(int(T/m.opt.timestep)):
            d.ctrl[:] = 0.0; mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        z1 = float((m.body_mass[1:, None]*d.xipos[1:]).sum(0)[2]/M)
        fmax = max([abs(float(d.actuator_force[a])) for a in range(m.nu)] + [0.0])
        ok = (z1 < z0 + 1e-3) and fmax < 1e-6
        return self._rec("PCTV-4", ok,
                         f"zero torque: COM {z0:.3f} -> {z1:.3f} m (must not rise), "
                         f"max |actuator force| {fmax:.2e} (must be 0)")

    def pctv6_known_torque(self, joint_idx=0):
        m, d = self.m, self.d
        mujoco.mj_forward(m, d)
        tau = 0.5*float(m.actuator_ctrlrange[joint_idx][1])
        d.ctrl[:] = 0.0; d.ctrl[joint_idx] = tau
        mujoco.mj_forward(m, d)
        dof = int(m.jnt_dofadr[int(m.actuator_trnid[joint_idx][0])])
        gf = float(d.qfrc_actuator[dof])
        d.ctrl[:] = 0.0
        ok = abs(gf - tau) < 1e-6
        return self._rec("PCTV-6", ok, f"commanded {tau:.2f} N*m -> qfrc_actuator {gf:.6f}")

    def pctv7_energy(self, validity):
        """Energy sanity on a completed rollout's TrajectoryValidityReport."""
        ok = validity["energy_residual"] <= validity["energy_tol"]
        return self._rec("PCTV-7", ok,
                         f"(E_takeoff-E0)-W_act = {validity['energy_residual']:+.2f} J, "
                         f"tol {validity['energy_tol']:.2f} J, W_act {validity['actuator_work']:.1f} J")

    def pctv8_bound_guard(self, v_takeoff, v_bound, tol=1e-3):
        """ENERGY_ACCOUNTING_VIOLATION if a trajectory beats a rigorous analytic bound."""
        ok = v_takeoff <= v_bound + tol
        return self._rec("PCTV-8", ok,
                         f"takeoff {v_takeoff:.3f} m/s vs rigorous bound {v_bound:.3f} m/s"
                         + ("" if ok else " -> ENERGY_ACCOUNTING_VIOLATION, quarantine"))

    def pctv9_capacity_realised(self, declared):
        """declared: {actuator_id: tau_limit}. Command 1.5x the limit and require the realised
        force to equal exactly the DECLARED limit (defect P6: scaled-up capacities were clamped
        to baseline by a stale ctrlrange)."""
        m, d = self.m, self.d; bad = []
        for a, lim in declared.items():
            d.ctrl[:] = 0.0; d.ctrl[a] = 1.5*lim; mujoco.mj_forward(m, d)
            # check at the JOINT (qfrc_actuator), after every clamp layer, not at the actuator
            dof = int(m.jnt_dofadr[int(m.actuator_trnid[a][0])])
            got = float(d.qfrc_actuator[dof])
            if abs(got - lim) > 1e-6*max(1.0, lim):
                bad.append((a, lim, got))
        d.ctrl[:] = 0.0
        return self._rec("PCTV-9", not bad, f"{len(declared)} actuators; mismatched {bad[:3]}")

    def run_all(self, q):
        self.pctv1_actuator_semantics()
        self.pctv2_control_coverage()
        self.pctv3_ground_placement(q)
        self.pctv4_zero_control(q)
        self.pctv6_known_torque()
        return self.results

    def summary(self):
        return all(v["passed"] for v in self.results.values())


# ----------------------------------------------------------------- static equilibrium
def potential_contacts(m, d, coll_geoms, tol=0.003, with_geom=False):
    """Candidate ground contact points built from GEOMETRY, not from MuJoCo's instantaneous
    contact set.

    A flat foot made of several capsules can have some of them a fraction of a millimetre above
    the plane, so MuJoCo reports a single contact per foot and the support region degenerates to
    a line. Static feasibility asks whether the pose CAN be held, so any collision geom within
    `tol` of the ground contributes its support points. 0.57 mm gaps are far below the contact
    model's own penetration scale and below any real manufacturing tolerance.

    Returns a list of world-frame contact points.
    """
    pts = []
    for g in coll_geoms:
        lo = geom_low(m, d, g)
        if lo > tol: continue
        t = m.geom_type[g]; c = np.array(d.geom_xpos[g]); sz = m.geom_size[g]
        R = np.array(d.geom_xmat[g]).reshape(3, 3)
        if t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            # each axis ENDPOINT counts only if that end itself touches the ground (defect P11:
            # a tilted capsule touching at one end had BOTH ends counted, inflating the support
            # region; found when a certified H1 start tipped over in simulation)
            ax = R[:, 2]*sz[1]
            for e in (c-ax, c+ax):
                if e[2] - sz[0] <= tol: pts.append([e[0], e[1], 0.0, g])
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz_ in (-1, 1):
                        e = c + R[:, 0]*sx*sz[0] + R[:, 1]*sy*sz[1] + R[:, 2]*sz_*sz[2]
                        if e[2] <= tol: pts.append([e[0], e[1], 0.0, g])  # only corners AT the ground (P11)
        else:
            pts.append([c[0], c[1], 0.0, g])
    # deduplicate points that coincide within a millimetre
    out = []
    for p_ in pts:
        if not any(abs(p_[0]-o[0]) < 1e-3 and abs(p_[1]-o[1]) < 1e-3 for o in out):
            out.append(p_)
    return out if with_geom else [p_[:3] for p_ in out]


def _support_margin(pts2, p):
    """Signed distance from p to the boundary of the convex hull of pts2 (+ inside)."""
    from scipy.spatial import ConvexHull
    try:
        h = ConvexHull(pts2)
    except Exception:
        return float("nan")
    # hull.equations: a.x + b <= 0 inside; distance to each facet = -(a.p + b)
    return float(np.min(-(h.equations[:, :2] @ p + h.equations[:, 2])))


def static_certificate(m, d, q, mu=0.9, tau_scale=None, coll_geoms=None, tol=0.003, tmax_override=None):
    """Solve the static equilibrium LP at pose q.

        qfrc_bias(q, 0) = S tau + sum_c J_c^T f_c

    subject to |tau_i| <= s * tau_max_i, unilateral normals f_n >= 0, and an inner pyramid
    friction cone. Minimise s, so s <= 1 means the pose can be held and 1 - s is the torque
    margin. This is the check the analytic pose optimizer never performed: it enforced ground
    clearance and centre-of-mass support but never torque feasibility against gravity.
    """
    d.qpos[:] = q; d.qvel[:] = 0; d.qacc[:] = 0
    mujoco.mj_forward(m, d)
    if coll_geoms is None:
        fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        coll_geoms = []
        for p_ in range(m.npair):
            gs = [int(m.pair_geom1[p_]), int(m.pair_geom2[p_])]
            if fl in gs: coll_geoms.append([x for x in gs if x != fl][0])
        if not coll_geoms:
            coll_geoms = [g for g in range(m.ngeom)
                          if m.geom_contype[g] != 0 and m.geom_bodyid[g] != 0]
    cpts_g = potential_contacts(m, d, sorted(set(coll_geoms)), tol, with_geom=True)
    cpts = [p_[:3] for p_ in cpts_g]
    nv, nu, nc = m.nv, m.nu, len(cpts)
    if nc == 0:
        return dict(status="CONTACT_INFEASIBLE", detail="no geom within tolerance of the ground")
    b = np.array(d.qfrc_bias, dtype=float)          # gravity generalised force (qvel = 0)

    S = np.zeros((nv, nu))
    tmax = np.zeros(nu)
    for a in range(nu):
        jid = int(m.actuator_trnid[a][0]); dof = int(m.jnt_dofadr[jid])
        S[dof, a] = float(m.actuator_gear[a][0])
        # torque limit from FORCErange (valid for motors AND servos); ctrlrange is an angle for a
        # position servo, and reading it here gave held joints limits of a few "N*m" (radians)
        if m.actuator_forcelimited[a]:
            t = float(m.actuator_forcerange[a][1])
        elif m.jnt_actfrclimited[jid]:
            t = float(m.jnt_actfrcrange[jid][1])
        else:
            t = float(m.actuator_ctrlrange[a][1]) if actuator_kinds(m)[a] == "torque_motor" else np.inf
        tmax[a] = t*(tau_scale[a] if tau_scale is not None else 1.0)
    if tmax_override is not None: tmax = np.asarray(tmax_override, float)

    # Map each candidate point to the nearest collision geom's body, then build its Jacobian.
    Jc = np.zeros((nv, 3*nc))
    jacp = np.zeros((3, nv)); jacr = np.zeros((3, nv))
    F = np.eye(3)[[2, 0, 1]]                         # rows: normal (+z), tangent x, tangent y
    for i, pt in enumerate(cpts):
        # each point belongs to the body of the geom that GENERATED it (defect P14: Phase 8 picked
        # the collision geom whose centre was nearest in xy, ignoring height, so a heel point could
        # be attached to a thigh or hip geom directly above it and the LP routed ground reaction
        # into the upper leg, needing no knee/ankle torque)
        best = int(m.geom_bodyid[int(cpts_g[i][3])])
        mujoco.mj_jac(m, d, jacp, jacr, np.array(pt, dtype=float), best)
        Jc[:, 3*i:3*i+3] = jacp.T @ F.T

    # variables x = [tau (nu), f (3nc), s]
    n = nu + 3*nc + 1
    Aeq = np.zeros((nv, n)); Aeq[:, :nu] = S; Aeq[:, nu:nu+3*nc] = Jc
    beq = b
    rows, rhs = [], []
    for i in range(nu):                              # |tau_i| <= s tmax_i
        r = np.zeros(n); r[i] = 1.0; r[-1] = -tmax[i]; rows.append(r); rhs.append(0.0)
        r = np.zeros(n); r[i] = -1.0; r[-1] = -tmax[i]; rows.append(r); rhs.append(0.0)
    k = mu/np.sqrt(2.0)
    for i in range(nc):
        o = nu + 3*i
        r = np.zeros(n); r[o] = -1.0; rows.append(r); rhs.append(0.0)          # f_n >= 0
        for t in (1, 2):                                                        # pyramid
            r = np.zeros(n); r[o+t] = 1.0; r[o] = -k; rows.append(r); rhs.append(0.0)
            r = np.zeros(n); r[o+t] = -1.0; r[o] = -k; rows.append(r); rhs.append(0.0)
    c = np.zeros(n); c[-1] = 1.0
    bnds = [(None, None)]*(nu+3*nc) + [(0, None)]
    res = linprog(c, A_ub=np.array(rows), b_ub=np.array(rhs), A_eq=Aeq, b_eq=beq,
                  bounds=bnds, method="highs")
    # support margin: signed distance of the COM ground projection to the contact hull edge
    com = np.array(d.subtree_com[0][:2])
    sup = _support_margin(np.array(cpts)[:, :2], com)
    lim_bad = [j for j in range(m.njnt) if m.jnt_limited[j] and m.jnt_type[j] in (2, 3)
               and not (m.jnt_range[j][0]-1e-6 <= q[m.jnt_qposadr[j]] <= m.jnt_range[j][1]+1e-6)]
    if lim_bad:
        return dict(status="JOINT_LIMIT_INFEASIBLE", support_margin=sup, ncon=int(nc),
                    detail=[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in lim_bad])
    if not res.success:
        # s is unbounded above, so infeasibility is never a torque-limit effect. Discriminate
        # support (no unilateral normal forces can balance) from friction (a cone too narrow).
        if mu < 1e5:
            r2 = static_certificate(m, d, q, mu=1e6, tau_scale=tau_scale, coll_geoms=coll_geoms, tol=tol,
                                    tmax_override=tmax_override)
            if r2["status"] in ("STATIC_FEASIBLE", "TORQUE_INFEASIBLE"):
                return dict(status="FRICTION_INFEASIBLE", support_margin=sup, ncon=int(nc),
                            detail=f"feasible only with mu=1e6 (s={r2.get('s'):.3f})")
        return dict(status="CONTACT_INFEASIBLE", support_margin=sup, detail=res.message, ncon=int(nc))
    s = float(res.x[-1])
    tau = res.x[:nu]; f = res.x[nu:nu+3*nc].reshape(nc, 3)
    resid = float(np.linalg.norm(Aeq @ res.x - beq))
    fn = f[:, 0]
    fric_margin = float(np.min([k*fn[i] - abs(f[i, 1]) for i in range(nc)] +
                               [k*fn[i] - abs(f[i, 2]) for i in range(nc)])) if nc else 0.0
    return dict(status="STATIC_FEASIBLE" if s <= 1.0 else "TORQUE_INFEASIBLE",
                s=s, support_margin=sup, tau=tau.tolist(), forces=f.tolist(), torque_margin=float(1.0-s), equilibrium_residual=resid,
                friction_margin=fric_margin, ncon=int(nc),
                max_norm_torque=float(np.max(np.abs(tau)/np.maximum(tmax, 1e-9))),
                total_normal_force=float(fn.sum()),
                weight=float(sum(m.body_mass[1:])*G),
                worst_joint=int(np.argmax(np.abs(tau)/np.maximum(tmax, 1e-9))))
