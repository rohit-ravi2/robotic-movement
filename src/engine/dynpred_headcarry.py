"""DYNAMIC MAX-LOAD PREDICTOR for the head carry with arm balance (track T2, 2026-10-17). Stock G1H spec only; no constant is fitted
to a found load. Replaces the static predictor of engine.headcarry (contact LP + 10 deg tilt + 0.5 x sway capacity: 63-65 kg
predicted vs 32-34 kg found).

The load rests on the head's dome at ONE contact point. Its free motion there is a ROCKING mode (rolling of the flat bottom /
curved side on the dome): with the load's COM a height h above the contact and the dome's radius of curvature R_a about axis a
(fitted to the head collision mesh's top 2 cm, separately in the sagittal and frontal planes), the gravity moment about the contact
at COM offset e is  m g (h - R_a) e / h  (destabilising when h > R_a), and the mode diverges at
lambda_a = sqrt(g (h - R_a) / rho_a^2), rho_a^2 = I_a/m + h^2 (radius of gyration about the contact). Only the palms (on the
load's sides) can supply the opposing moment.

  D1 TOPPLE vs ARM AUTHORITY (the mechanism the track A 0.5 ms failures show: wrist yaw saturated while the load tipped off): the
     arms must be able to return the load from ANYWHERE inside the allowed patch (the gate's 7 cm, a task definition) while it is
     diverging:  M_arm,a(v) >= m g (h - R_a) e_max / h,  so  m_max,a = M_arm,a h / (g (h - R_a) e_max).
     M_arm,a = the largest corrective moment about the head contact the two palms can apply (an LP over the 8 palm-patch contact
     forces, unilateral normals, friction pyramid at the model's palm-object mu, and the 14 arm joint torques J^T f + arm gravity
     inside each joint's torque-SPEED envelope tau_max (1 - |w|/omega) at the joint speeds w = J^+ v_p needed to follow the load at
     the divergence speed v_p = lambda e_max h_p/h, h_p = palm contact height above the head contact). Per axis, worse sign.
     h <= R_a: the mode is passively stable about that axis (no D1 bound).
  D2 STATIC whole body: the track A contact LP without tilt (a_static, every joint within its rating). Spec-only, reused.
  D3 COM / ZMP margin with the load: when the arms push the load back with their saturated force the reaction reaches the head;
     together with the load's COM offset at the patch edge the system ZMP moves by
        dx = (m e_max + F_corr z_head / g) / (M + m)    (F_corr = M_arm,a / h_p)
     which must stay inside the CoP authority: the support polygon AND |CoP - ankle| <= 2 tau_ankle / ((M + m) g) per axis
     (ankle pitch / roll ratings). The binding axis is reported.
  D4 FOOT FRICTION / SLIP: the same corrective force is horizontal at the feet: F_corr <= mu_foot (M + m) g (mu from the model).
Prediction m_max = min(D1, D2, D3, D4); the binding term, axis and joint are reported.
HANDS ONLY (no head): the load hangs in the palms' friction: D1h = the largest m for which the palms' LP (vertical weight + the
corrective moment for the grip's own slip tolerance) is feasible, i.e. the squeeze-friction capacity under the arms' torque
envelope; D3/D4 as above with the load ahead of the torso.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
from scipy.optimize import linprog

G = 9.80665
E_MAX = 0.07         # the hold gate's head patch radius (OFF_PATCH): a task definition
OBJ_EXTRA = {"tallcyl": dict(kind="cylinder", size=(0.08, 0.20)), "crate": dict(kind="box", size=(0.20, 0.15, 0.06))}


def register_objects():
    """Add the T2 held-out objects to the track A object tables (engine.headcarry.OBJ and leverage_probes.OBJ). No robot change."""
    from engine import headcarry as HC
    import leverage_probes as LP
    for k, v in OBJ_EXTRA.items(): HC.OBJ.setdefault(k, v); LP.OBJ.setdefault(k, v)


def obj_inertia(obj, mass=1.0):
    """Principal inertias (x, y, z about the COM) of the uniform object."""
    from engine import headcarry as HC
    o = HC.OBJ[obj]
    if o["kind"] == "box":
        a, b, c = [2*s for s in o["size"]]; return mass/12*np.array([b*b + c*c, a*a + c*c, a*a + b*b])
    r, hh = o["size"]; H = 2*hh; return mass*np.array([(3*r*r + H*H)/12, (3*r*r + H*H)/12, r*r/2])


def dome(m, d, H):
    """Radii of curvature of the head collision mesh at its top: least-squares paraboloid z = z0 - x^2/(2Rx) - y^2/(2Ry) over the
    vertices within 2 cm of the top (world frame at the settled pose)."""
    from engine import leverage as LV
    hg = H["head_geom"]; mid = m.geom_dataid[hg]; a = m.mesh_vertadr[mid]
    V = m.mesh_vert[a:a + m.mesh_vertnum[mid]] @ d.geom_xmat[hg].reshape(3, 3).T + d.geom_xpos[hg]
    top = LV.head_top(m, d, hg); P = V[V[:, 2] > top[2] - 0.02] - top
    A = np.c_[np.ones(len(P)), -P[:, 0]**2/2, -P[:, 1]**2/2, P[:, 0], P[:, 1]]
    c = np.linalg.lstsq(A, P[:, 2], rcond=None)[0]
    return dict(Rx=float(1/c[1]), Ry=float(1/c[2]), n=int(len(P)), top=top.tolist())


def arm_moment_lp(m, q, strategy, obj, axis, sign, v_p, pivot, arm_ids, tm, om, mu_palm):
    """Max corrective moment (about `axis` through `pivot`, sign) from the 8 palm-patch contacts, arm torques inside the torque-speed
    envelope at palm speed v_p along the corrective direction. Returns (moment N m, binding joint)."""
    from engine import headcarry as HC
    d = mujoco.MjData(m); d.qpos[:] = q; d.qvel[:] = 0; mujoco.mj_forward(m, d)
    cs, com = HC.obj_contacts(m, q, strategy, obj); cs = cs[:8]
    dofs = [int(m.jnt_dofadr[int(m.actuator_trnid[a][0])]) for a in arm_ids]
    # joint speeds needed: each palm moves at v_p along the moment's tangential direction at its centre
    jp = np.zeros((3, m.nv)); w = np.zeros(len(arm_ids))
    for side in range(2):
        p = np.mean([c["point"] for c in cs[4*side:4*side + 4]], 0); bdy = cs[4*side]["body"]
        mujoco.mj_jac(m, d, jp, None, p, bdy); J = jp[:, dofs]
        tdir = np.cross(sign*axis, p - pivot); nt = np.linalg.norm(tdir)
        if nt > 1e-9: w += np.linalg.pinv(J) @ (v_p*tdir/nt)
    env = np.array([tm[a]*max(0.0, 1 - abs(w[i])/om[i]) for i, a in enumerate(arm_ids)])
    bias = np.array(d.qfrc_bias)[dofs]                      # arm gravity (the posture must still be held)
    nc = len(cs); n = 3*nc; A_t = np.zeros((len(arm_ids), n)); obj_m = np.zeros(n)
    def basis(nrm):
        nrm = nrm/np.linalg.norm(nrm); t1 = np.cross(nrm, [1, 0, 0] if abs(nrm[0]) < 0.9 else [0, 1, 0]); t1 /= np.linalg.norm(t1); return np.stack([nrm, t1, np.cross(nrm, t1)], 1)
    for k, c in enumerate(cs):
        Bk = basis(c["normal"]); mujoco.mj_jac(m, d, jp, None, c["point"], c["body"])
        A_t[:, 3*k:3*k+3] = jp[:, dofs].T @ Bk                 # torque needed to press with force Bk x on the object
        obj_m[3*k:3*k+3] = (sign*axis) @ np.array([np.cross(c["point"] - pivot, Bk[:, j]) for j in range(3)]).T
    rows = []; rhs = []
    for i in range(len(arm_ids)):
        rows.append(A_t[i]); rhs.append(env[i] - bias[i]); rows.append(-A_t[i]); rhs.append(env[i] + bias[i])
    for k in range(nc):
        o = 3*k; kk = mu_palm/np.sqrt(2)
        for tt in (1, 2):
            for sg in (1, -1):
                r = np.zeros(n); r[o+tt] = sg; r[o] = -kk; rows.append(r); rhs.append(0.0)
    bnds = [(0, None) if j % 3 == 0 else (None, None) for j in range(n)]
    res = linprog(-obj_m, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=bnds, method="highs")
    if not res.success: return 0.0, "infeasible", w
    tau = A_t @ res.x + bias; u = np.abs(tau)/np.maximum(env, 1e-9)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a][0])) for a in arm_ids]
    return float(-res.fun), names[int(np.argmax(u))], w


def static_capacity(S, strategy, obj, qa):
    """D2: the track A static contact LP (no tilt, no sway): largest mass with every joint within its rating (bisection, 0.05 kg)."""
    from engine import headcarry as HC
    m = S["m"]; H, wb, arm = HC.parts(m); fg = sorted(S["foot_geoms"])
    def ok(mm):
        cs, com = HC.obj_contacts(m, qa, strategy, obj); return HC.contact_lp(m, qa, cs, com, mm, fg, S["robot_bodies"], H["fingers"])["s"] <= 1
    lo, hi = 0.5, 200.0
    if ok(hi): return hi
    if not ok(lo): return 0.0
    while hi - lo > 0.05:
        mid = 0.5*(lo + hi); lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def predict(strategy, obj, S=None, static=None, gp=None):
    """Dynamic max-load prediction (kg) for (strategy in head/hands, object). static: the track A a_static value if known."""
    import throw as TH
    from engine import headcarry as HC, leverage as LV
    register_objects()
    S = S or TH.setup("G1H", ball=False); m = S["m"]; q = np.array(S["q_settled"])
    c, tg = HC.grip_geometry(m, q, strategy, obj, gp if gp is not None else (dict(dz=0.5) if strategy == "head" else {})); qa, err, perr = HC.grip_ik(m, q, tg)
    H, wb, arm = HC.parts(m); d = mujoco.MjData(m); d.qpos[:] = qa; mujoco.mj_forward(m, d)
    tm = LV.tau_max(m); arm_ids = [[a for a in range(m.nu) if int(m.actuator_trnid[a][0]) == j][0] for s in "rl" for j in arm[s]]
    om = [float(S["rb"].J[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a][0]))]["omega"]) for a in arm_ids]
    hw, hh = HC.half(obj); I = obj_inertia(obj)
    M = float(sum(m.body_mass[b] for b in S["robot_bodies"]))
    # palm-object friction: MuJoCo uses the max of the pair; the object gets the default geom friction 1 (engine.world)
    pg = [g for g in range(m.ngeom) if m.geom_contype[g] and HC._under(m, g, wb["r"])]
    mu_palm = max([float(m.geom_friction[g][0]) for g in pg] + [1.0])
    fg = sorted(S["foot_geoms"]); mu_foot = max(float(m.geom_friction[g][0]) for g in fg)
    # support polygon and ankle authority (settled stance)
    pts = []
    for g in fg:
        pts.append(d.geom_xpos[g][:2])
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            R = d.geom_xmat[g].reshape(3, 3); s = m.geom_size[g]
            pts += [(d.geom_xpos[g] + R @ np.array([i*s[0], j*s[1], 0]))[:2] for i in (-1, 1) for j in (-1, 1)]
    pts = np.array(pts); comw = d.subtree_com[1][:2]
    ank = {ax: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_ankle_{ax}_joint") for s in ("left", "right")] for ax in ("pitch", "roll")}
    tau_ank = {ax: float(S["rb"].J[f"left_ankle_{ax}_joint"]["tau"]) for ax in ("pitch", "roll")}
    if static is None: static = static_capacity(S, strategy, obj, qa)
    out = dict(strategy=strategy, obj=obj, ik_err=err, palm_err=perr, h=hh, half_width=hw, mu_palm=mu_palm, mu_foot=mu_foot, M_robot=M)
    if strategy == "head":
        dm = dome(m, d, H); pivot = np.array(dm["top"]); out["dome"] = dm
        pal = np.mean([cc["point"] for cc in HC.obj_contacts(m, qa, strategy, obj)[0][:8]], 0); h_p = float(pal[2] - pivot[2]); out["h_palm"] = h_p
        z_head = float(pivot[2]); D1 = {}
        for ax_name, axis, R_, Ia in (("roll(x-axis, lateral tip)", np.array([1.0, 0, 0]), dm["Ry"], I[0]), ("pitch(y-axis, fore-aft tip)", np.array([0, 1.0, 0]), dm["Rx"], I[1])):
            rho2 = Ia + hh*hh; lam2 = G*(hh - R_)/rho2
            if hh <= R_:
                D1[ax_name] = dict(R=R_, lam=None, stable=True, m_max=np.inf); continue
            lam = np.sqrt(lam2); v_p = lam*E_MAX*h_p/hh
            Ms = [arm_moment_lp(m, qa, strategy, obj, axis, sg, v_p, pivot, arm_ids, tm, om, mu_palm) for sg in (1, -1)]
            Mw, jw, _ = min(Ms, key=lambda t: t[0])
            mmax = Mw*hh/(G*(hh - R_)*E_MAX)
            # D3 ZMP: lateral tip -> y (ankle roll), fore-aft tip -> x (ankle pitch)
            iax = 1 if axis[0] == 1 else 0; tau_a = tau_ank["roll" if iax == 1 else "pitch"]
            span = (pts[:, iax].min() - comw[iax], pts[:, iax].max() - comw[iax]); poly = min(abs(span[0]), abs(span[1]))
            F_corr = Mw/max(h_p, 1e-3)
            # the palm-head forces are internal; what reaches the feet is the load's own acceleration while it is pushed back from
            # the patch edge against its divergence: m lambda^2 e_max (horizontal, applied at the head height)
            def zmp_ok(mm):
                cop_auth = min(poly, 2*tau_a/((M + mm)*G))
                return (mm*E_MAX + mm*lam2*E_MAX*z_head/G)/(M + mm) <= cop_auth
            lo, hi = 0.0, 400.0
            if zmp_ok(hi): m3 = hi
            elif not zmp_ok(0.0): m3 = 0.0
            else:
                for _ in range(50):
                    mid = 0.5*(lo + hi); lo, hi = (mid, hi) if zmp_ok(mid) else (lo, mid)
                m3 = lo
            m4 = np.inf if lam2*E_MAX <= mu_foot*G else 0.0     # m lam^2 e <= mu (M + m) g holds for every m iff lam^2 e <= mu g
            D1[ax_name] = dict(R=R_, lam=float(lam), v_palm=float(v_p), M_arm=float(Mw), M_arm_binding_joint=jw, m_max=float(mmax),
                               D3_zmp_m_max=float(m3), D3_axis=("y (ankle roll)" if iax == 1 else "x (ankle pitch)"), F_corr=float(F_corr), D4_friction_ok=bool(np.isinf(m4)))
        out["D1_topple"] = D1
        m1 = min(v["m_max"] for v in D1.values()); m3 = min(v.get("D3_zmp_m_max", np.inf) for v in D1.values())
        terms = dict(D1_topple=m1, D2_static=(static if static is not None else np.inf), D3_zmp=m3, D4_friction=min((np.inf if v.get("D4_friction_ok", True) else 0.0) for v in D1.values()))
    else:
        # hands: the load hangs in squeeze friction: max m such that the palms carry m g vertically (+ the arms' gravity), every arm joint
        # inside its static envelope, friction pyramid at the palm patches: an LP maximising the supported weight
        cs, com = HC.obj_contacts(m, qa, strategy, obj); cs = cs[:8]
        dofs = [int(m.jnt_dofadr[int(m.actuator_trnid[a][0])]) for a in arm_ids]; jp = np.zeros((3, m.nv))
        nc = len(cs); n = 3*nc + 1; A_t = np.zeros((len(arm_ids), n)); Aeq = np.zeros((6, n))
        def basis(nrm):
            nrm = nrm/np.linalg.norm(nrm); t1 = np.cross(nrm, [1, 0, 0] if abs(nrm[0]) < 0.9 else [0, 1, 0]); t1 /= np.linalg.norm(t1); return np.stack([nrm, t1, np.cross(nrm, t1)], 1)
        for k, cc in enumerate(cs):
            Bk = basis(cc["normal"]); mujoco.mj_jac(m, d, jp, None, cc["point"], cc["body"]); A_t[:, 3*k:3*k+3] = jp[:, dofs].T @ Bk
            r_ = cc["point"] - com; Aeq[:3, 3*k:3*k+3] = Bk; Aeq[3:, 3*k:3*k+3] = np.array([np.cross(r_, Bk[:, j]) for j in range(3)]).T
        Aeq[2, -1] = -G                                       # sum of contact forces z = m g
        bias = np.array(d.qfrc_bias)[dofs]; env = np.array([tm[a] for a in arm_ids]); rows, rhs = [], []
        for i in range(len(arm_ids)):
            rows.append(A_t[i]); rhs.append(env[i] - bias[i]); rows.append(-A_t[i]); rhs.append(env[i] + bias[i])
        for k in range(nc):
            for tt in (1, 2):
                for sg in (1, -1):
                    r = np.zeros(n); r[3*k+tt] = sg; r[3*k] = -mu_palm/np.sqrt(2); rows.append(r); rhs.append(0.0)
        c_ = np.zeros(n); c_[-1] = -1.0
        res = linprog(c_, A_ub=np.array(rows), b_ub=np.array(rhs), A_eq=Aeq, b_eq=np.zeros(6), bounds=[(0, None) if j % 3 == 0 else (None, None) for j in range(3*nc)] + [(0, None)], method="highs")
        m1 = float(res.x[-1]) if res.success else 0.0
        jw = None
        if res.success:
            tau = A_t @ res.x + bias; names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a][0])) for a in arm_ids]; jw = names[int(np.argmax(np.abs(tau)/env))]
        # D3: the load ahead of the torso shifts the system COM forward; sagittal CoP authority
        span = (pts[:, 0].min() - comw[0], pts[:, 0].max() - comw[0]); xl = float(com[0] - comw[0])
        def zmp_ok(mm): return (mm*xl)/(M + mm) <= min(span[1], 2*tau_ank["pitch"]/((M + mm)*G))
        lo, hi = 0.0, 400.0
        for _ in range(50):
            mid = 0.5*(lo + hi); lo, hi = (mid, hi) if zmp_ok(mid) else (lo, mid)
        out["hands"] = dict(m_squeeze=m1, binding_joint=jw, load_ahead_x=xl)
        terms = dict(D1_squeeze=m1, D2_static=(static if static is not None else np.inf), D3_zmp=lo, D4_friction=np.inf)
    out["terms"] = {k: (None if not np.isfinite(v) else round(float(v), 2)) for k, v in terms.items()}
    b = min(terms, key=lambda k: terms[k]); out["m_max"] = round(float(terms[b]), 2) if np.isfinite(terms[b]) else None; out["binding"] = b
    return out
