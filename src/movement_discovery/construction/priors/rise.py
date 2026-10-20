"""SUPPORT-TRANSITION prior (rise / lower): a path of STATICALLY CERTIFIED poses from a start posture to a goal posture, moving the
COM over the evolving support polygon and transferring load contact by contact. Body- and scene-derived; no routine is authored.

Pieces (MuJoCo inside functions only):
  surfaces(S)              support surfaces from the MODEL: the floor plane and every world box whose top face a robot geom is paired
                           with (a seat, a platform); a contact is a robot geom whose lowest point is within tol of a surface top and
                           inside its footprint.
  contact_groups(S)        robot collision geoms paired with a surface -> load-bearing GROUPS (pelvis, torso, <side>_thigh / _shin /
                           _foot / _arm / _hand) and LIMBS (a leg = thigh+shin+foot of one side; an arm = upper arm+hand): a limb is the
                           unit that can be lifted and re-placed.
  certificate(S, q, active) the static certificate GENERALISED TO ARBITRARY SUPPORT SETS: plant.static_certificate's LP (gravity
                           = actuator torques + unilateral pyramid-cone contact forces, minimise the max normalised torque s) over the
                           candidate points of the ACTIVE groups only, on any horizontal surface (not only z = 0); ok = s <= 1 and the
                           COM >= margin inside the active support polygon (throw._cert_ok's friction-artefact rule kept).
  support_path(S, q0, goal) greedy certified path: each step is a weighted least-squares (differential IK) step of the floating base +
                           driven joints with the active contact points pinned (no slip, on the surface), non-active candidate geoms kept
                           above every surface, the COM pulled toward the centroid of the support that must REMAIN, and the joints /
                           pelvis toward the goal; every accepted step is certified on the active set. A contact is RELEASED only when
                           the certificate holds without it; a goal contact (a foot) that is not placed is freed by releasing its limb
                           (when the rest certifies), swung clear of the surfaces and placed flat at the goal stance's offset; when no
                           limb can be freed (the remaining support has no area under the COM) a free arm is ACQUIRED as an auxiliary
                           support at the reachable floor point that maximises the remaining polygon's margin. Joint motion is
                           capacity-weighted (metric 1/(tau om)); joint limits at 95 %.
Goals come from the program's GoalSpec: final="standing" (the canonical settled stance, feet the only support), final="seated" (pelvis
lowered onto the named seat surface over seat_x, feet kept), kind com_displacement with pelvis_target (pelvis lowered to z*, feet only).
final="lying" (rolls) is NOT expressible here: an orientation goal with a rolling support is not a sequence of static poses."""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence
import numpy as np

G = 9.80665
TOL = 0.006                    # contact candidate tolerance above a surface (m): plant.potential_contacts uses 3 mm; settled body contacts sit 3-5 mm up
LIMIT_FRAC = 0.05


# ------------------------------------------------------------------------------------------------ model-derived surfaces and groups
def surfaces(S) -> List[dict]:
    """Horizontal support surfaces: the floor plane (z = 0, unbounded) + world boxes (top face) paired with robot geoms."""
    import mujoco
    m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d); rg = S["robot_geoms"]; out = []
    world = set()
    for p in range(m.npair):
        a, b = int(m.pair_geom1[p]), int(m.pair_geom2[p])
        if (a in rg) != (b in rg): world.add(b if a in rg else a)
    for g in sorted(world):
        t = int(m.geom_type[g]); nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}"
        if t == mujoco.mjtGeom.mjGEOM_PLANE: out.append(dict(geom=g, name=nm, z=float(d.geom_xpos[g][2]), c=np.zeros(2), R=np.eye(2), half=np.array([np.inf, np.inf])))
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            R = d.geom_xmat[g].reshape(3, 3)
            if abs(R[2, 2]) < 0.99: continue                                                    # tilted box: no horizontal top
            sz = m.geom_size[g]; out.append(dict(geom=g, name=nm, z=float(d.geom_xpos[g][2] + sz[2]), c=d.geom_xpos[g][:2].copy(), R=R[:2, :2].copy(), half=sz[:2].copy()))
    return out


def _part(name: str) -> str:
    n = name.replace("_collision", "").rstrip("0123456789")
    side = "left_" if n.startswith("left") else ("right_" if n.startswith("right") else "")
    if n in ("pelvis",) or n.endswith("_hip"): return "pelvis"
    if n in ("torso", "head"): return "torso"
    for k, grp in (("foot", "foot"), ("shin", "shin"), ("thigh", "thigh"), ("hand", "hand"), ("wrist", "hand"), ("elbow", "arm"), ("shoulder", "arm")):
        if k in n: return side + grp
    return side + n


def contact_groups(S) -> Dict[str, List[int]]:
    """{group: [robot geom ids paired with a support surface]} (foot box / linkage geoms paired only robot-robot are excluded)."""
    import mujoco
    m = S["m"]; rg = S["robot_geoms"]; sg = {s["geom"] for s in surfaces(S)}; out: Dict[str, List[int]] = {}
    for p in range(m.npair):
        a, b = int(m.pair_geom1[p]), int(m.pair_geom2[p])
        if a in sg and b in rg: g = b
        elif b in sg and a in rg: g = a
        else: continue
        k = _part(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}")
        if g not in out.setdefault(k, []): out[k].append(g)
    return out


def limb_of(group: str) -> List[str]:
    """The groups lifted together with `group` (one side's leg or arm); trunk groups are their own limb."""
    for side in ("left_", "right_"):
        if group.startswith(side):
            return [side + p for p in (("thigh", "shin", "foot") if group[len(side):] in ("thigh", "shin", "foot") else ("arm", "hand"))]
    return [group]


def _inside(s, xy, pad=0.0) -> bool:
    if not np.isfinite(s["half"][0]): return True
    loc = s["R"].T @ (np.asarray(xy) - s["c"]); return bool(np.all(np.abs(loc) <= s["half"] + pad))


def geom_points(m, d, g) -> List[np.ndarray]:
    """Candidate support points of one geom (world xyz): capsule/cylinder axis ends, box bottom corners, else the lowest point below the centre."""
    import mujoco, plant as P
    t = m.geom_type[g]; c = np.array(d.geom_xpos[g]); sz = m.geom_size[g]; R = np.array(d.geom_xmat[g]).reshape(3, 3)
    if t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        ax = R[:, 2]*sz[1]; return [e - np.array([0, 0, sz[0]]) for e in (c - ax, c + ax)]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return [c + R[:, 0]*sx*sz[0] + R[:, 1]*sy*sz[1] + R[:, 2]*sz_*sz[2] for sx in (-1, 1) for sy in (-1, 1) for sz_ in (-1, 1)]
    if t == mujoco.mjtGeom.mjGEOM_SPHERE: return [c - np.array([0, 0, sz[0]])]
    return [np.array([c[0], c[1], P.geom_low(m, d, g)])]


def contacts(S, d, groups: Dict[str, List[int]], surfs: List[dict], tol: float = TOL, active: Optional[Sequence[str]] = None) -> List[tuple]:
    """[(group, geom, point xyz on the surface, surface index, gap)] for the groups (all when active is None) within tol of a surface top."""
    m = S["m"]; out = []
    for k, gs in groups.items():
        if active is not None and k not in active: continue
        for g in gs:
            for p in geom_points(m, d, g):
                if any(_inside(s2, p[:2]) and p[2] < s2["z"] - 0.01 for s2 in surfs if np.isfinite(s2["half"][0])): continue   # embedded under a raised surface (a heel inside a seat): not a support
                for si, s in enumerate(surfs):
                    gap = float(p[2] - s["z"])
                    if -0.03 <= gap <= tol and _inside(s, p[:2]):
                        out.append((k, g, np.array([p[0], p[1], s["z"]]), si, gap)); break
    ded = []
    for c in out:
        if not any(c[0] == o[0] and np.linalg.norm(c[2][:2] - o[2][:2]) < 1e-3 for o in ded): ded.append(c)
    return ded


def lowest_gap(S, d, g, surfs) -> float:
    """Clearance of geom g above the surface under it (min over its candidate points; inf when over no surface)."""
    best = np.inf
    for p in geom_points(S["m"], d, g):
        for s in surfs:
            if _inside(s, p[:2]) and p[2] - s["z"] > -0.05: best = min(best, float(p[2] - s["z"]))
    return best


def support_margin(pts_xy, xy) -> float:
    import plant as P
    if len(pts_xy) < 3: return -1.0
    v = P._support_margin(np.asarray(pts_xy)[:, :2], np.asarray(xy, float))
    return -1.0 if not np.isfinite(v) else float(v)


# ------------------------------------------------------------------------------------------------ the generalised certificate
def certificate(S, q, groups, surfs, active: Sequence[str], mu: float = 0.9, margin: float = 0.02, tol: float = TOL, refine: bool = False) -> dict:
    """Static equilibrium LP of plant.static_certificate over the ACTIVE groups' candidate points on any surface. ok = s <= 1 and
    COM margin >= margin (or the LP's friction artefact with margin >= 2 cm, throw._cert_ok)."""
    import mujoco
    from scipy.optimize import linprog
    m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = q; d.qvel[:] = 0; mujoco.mj_forward(m, d)
    cs = contacts(S, d, groups, surfs, tol, active); com = d.subtree_com[1][:2].copy()
    mg = support_margin([c[2] for c in cs], com) if cs else -1.0
    used = sorted({c[0] for c in cs})
    if not cs: return dict(ok=False, status="CONTACT_INFEASIBLE", s=np.inf, margin=mg, groups=used, tau=None)
    nv, nu, nc = m.nv, m.nu, len(cs); b = np.array(d.qfrc_bias, float)
    Sa = np.zeros((nv, nu)); tmax = np.zeros(nu)
    for a in range(nu):
        jid = int(m.actuator_trnid[a][0]); Sa[int(m.jnt_dofadr[jid]), a] = float(m.actuator_gear[a][0])
        tmax[a] = float(m.actuator_forcerange[a][1]) if m.actuator_forcelimited[a] else float(m.jnt_actfrcrange[jid][1] or 1e3)
    Jc = np.zeros((nv, 3*nc)); jp = np.zeros((3, nv)); F = np.eye(3)[[2, 0, 1]]
    for i, c in enumerate(cs):
        mujoco.mj_jac(m, d, jp, None, c[2], int(m.geom_bodyid[c[1]])); Jc[:, 3*i:3*i+3] = jp.T @ F.T
    n = nu + 3*nc + 1; Aeq = np.zeros((nv, n)); Aeq[:, :nu] = Sa; Aeq[:, nu:nu+3*nc] = Jc
    rows = []; k = mu/np.sqrt(2.0)
    for i in range(nu):
        for sg in (1.0, -1.0): r = np.zeros(n); r[i] = sg; r[-1] = -tmax[i]; rows.append(r)
    for i in range(nc):
        o = nu + 3*i; r = np.zeros(n); r[o] = -1.0; rows.append(r)
        for t in (1, 2):
            for sg in (1.0, -1.0): r = np.zeros(n); r[o+t] = sg; r[o] = -k; rows.append(r)
    c_ = np.zeros(n); c_[-1] = 1.0
    # equality rows scaled by the body weight: unscaled, HiGHS returns status 4 (numerical difficulties) on certified stances; this, not
    # friction, is the "FRICTION_INFEASIBLE artefact" of GOTCHAS (a stance with 8 cm margin failed at mu 0.9 and 1e6 and passed scaled)
    W = max(float(S.get("M", 35.0))*G, 1.0); bnds = [(None, None)]*(nu+3*nc) + [(0, None)]
    for meth in ("highs", "highs-ipm"):
        res = linprog(c_, A_ub=np.array(rows), b_ub=np.zeros(len(rows)), A_eq=Aeq/W, b_eq=b/W, bounds=bnds, method=meth)
        if res.status != 4: break
    if not res.success:
        if mu < 1e5 and mg >= max(margin, 0.02):                                # throw._cert_ok: friction LP artefact with a clear support margin
            r2 = certificate(S, q, groups, surfs, active, mu=1e6, margin=margin, tol=tol)
            if r2["status"] == "STATIC_FEASIBLE": return dict(r2, ok=True, status="FRICTION_INFEASIBLE")
        return dict(ok=False, status="CONTACT_INFEASIBLE", s=np.inf, margin=mg, groups=used, tau=None)
    s = float(res.x[-1])
    if refine and s <= 1.0:
        # second stage (feedforward quality): among the solutions with s <= s* + 0.05, the least total normalised torque sum |tau|/tmax
        # (the min-max solution is not unique; an arbitrary one loaded one foot and pushed the COM sideways when tracked)
        n2 = n + nu; A2 = np.hstack([np.array(rows), np.zeros((len(rows), nu))]); extra = []
        for i in range(nu):
            for sg in (1.0, -1.0): r = np.zeros(n2); r[i] = sg/tmax[i]; r[n + i] = -1.0; extra.append(r)
        r = np.zeros(n2); r[n - 1] = 1.0; extra.append(r)
        A2 = np.vstack([A2] + [e[None] for e in extra]); b2 = np.concatenate([np.zeros(len(rows) + 2*nu), [min(1.0, s + 0.05)]])
        c2 = np.concatenate([np.zeros(n), np.ones(nu)]); Aeq2 = np.hstack([Aeq/W, np.zeros((nv, nu))])
        r2 = linprog(c2, A_ub=A2, b_ub=b2, A_eq=Aeq2, b_eq=b/W, bounds=bnds + [(0, None)]*nu, method="highs")
        if r2.status == 4: r2 = linprog(c2, A_ub=A2, b_ub=b2, A_eq=Aeq2, b_eq=b/W, bounds=bnds + [(0, None)]*nu, method="highs-ipm")
        if r2.success: res = r2
    f = res.x[nu:nu+3*nc].reshape(nc, 3); load = {}
    for i, c in enumerate(cs): load[c[0]] = load.get(c[0], 0.0) + float(f[i, 0])
    return dict(ok=bool(s <= 1.0 and mg >= margin), status=("STATIC_FEASIBLE" if s <= 1.0 else "TORQUE_INFEASIBLE"), s=s, margin=mg, groups=used, tau=res.x[:nu].copy(), load=load, ncon=nc)


# ------------------------------------------------------------------------------------------------ goals
def stance_q(S) -> np.ndarray:
    """The canonical settled stance of the same robot (bare scene): the standing goal posture."""
    import throw as TH, branch_i as BI
    from engine import core
    S0 = TH.setup(S.get("robot", "G1"), ball=False); BI.apply_dt(S0, core.DT_OPT); BI.settle(S0); return np.asarray(S0["q_settled"], float)


def goal_from_spec(S, goal) -> dict:
    """GoalSpec -> planner goal: kind standing | seated | lower | unsupported."""
    p = dict(getattr(goal, "params", {}) or {}); kind = getattr(goal, "kind", None)
    if p.get("final") == "standing": return dict(kind="standing", q=stance_q(S), z=float(p.get("z_stand", 0.72)))
    if p.get("final") == "seated": return dict(kind="seated", x=float(p.get("seat_x", 0.0)), seat_h=float(p.get("seat_h", 0.3)))
    if kind == "com_displacement" and p.get("pelvis_target") is not None: return dict(kind="lower", z=float(p["pelvis_target"]))
    return dict(kind="unsupported", why=f"goal {kind} {p.get('final')}: not a static support transition (orientation / rolling goals)")


# ------------------------------------------------------------------------------------------------ the path
def _embedded(p, surfs) -> bool:
    return any(_inside(s2, p[:2]) and p[2] < s2["z"] - 0.01 for s2 in surfs if np.isfinite(s2["half"][0]))


def _depth(m, b: int) -> int:
    n = 0
    while b > 0: b = int(m.body_parentid[b]); n += 1
    return n


def _level(qf):
    """The yaw-only orientation closest to qf (a foot laid flat, heading kept)."""
    w, x, y, z = qf; yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)); return np.array([np.cos(yaw/2), 0.0, 0.0, np.sin(yaw/2)])


def _quat_err(qa, qb):
    """Rotation vector taking orientation qa to qb (world frame)."""
    import mujoco
    r = np.zeros(3); qi = np.zeros(4); mujoco.mju_negQuat(qi, qa); qd = np.zeros(4); mujoco.mju_mulQuat(qd, qb, qi); mujoco.mju_quat2Vel(r, qd, 1.0); return r


def support_path(S, goal: dict, q0=None, max_steps: int = 1600, dq_max: float = 0.02, margin: float = 0.03, verbose: bool = False) -> dict:
    """Certified quasi-static path (see module doc). Returns dict(Q [N, nq], active [N] (sorted group tuples), s [N], margin [N],
    tau [N, nu], events [(i, kind, groups)], reached, reason, goal)."""
    import mujoco
    m = S["m"]; J = S["jarr"]; dofs = np.asarray(J["dof"]); qad = np.asarray(J["qad"]); nv = m.nv
    q = np.array(S["q_settled"] if q0 is None else q0, float); d = mujoco.MjData(m)
    surfs = surfaces(S); groups = contact_groups(S); floor_i = [i for i, s in enumerate(surfs) if not np.isfinite(s["half"][0])]
    jids = [int(m.dof_jntid[int(x)]) for x in dofs]; lo = np.array([m.jnt_range[j][0] for j in jids]); hi = np.array([m.jnt_range[j][1] for j in jids])
    lo, hi = lo + LIMIT_FRAC*(hi - lo), hi - LIMIT_FRAC*(hi - lo)
    cap = np.asarray(J["tau"], float)*np.asarray(J["om"], float); wreg = np.concatenate([np.full(6, 1.0/np.median(cap)), 1.0/cap])
    idx = np.concatenate([np.arange(6), dofs]); root = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis"))
    foot_body = {"left_foot": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"), "right_foot": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")}
    kind = goal["kind"]; goal_groups = {"left_foot", "right_foot"} | ({"pelvis"} if kind == "seated" else set())
    seat_i = None
    if kind == "seated":
        cand = [i for i, s in enumerate(surfs) if np.isfinite(s["half"][0])]
        seat_i = min(cand, key=lambda i: abs(surfs[i]["c"][0] - goal["x"])) if cand else None
    # stance geometry (standing goal): foot poses relative to each other, and to the stance COM
    st = None
    if kind == "standing":
        ds = mujoco.MjData(m); qs = np.array(q); nqr = len(goal["q"]); qs[:nqr] = goal["q"][:nqr]; ds.qpos[:] = qs; mujoco.mj_forward(m, ds)
        st = dict(foot={k: (ds.xpos[b].copy(), ds.xquat[b].copy()) for k, b in foot_body.items()}, com=ds.subtree_com[root].copy(), pelvis_z=float(ds.qpos[2]), quat=ds.qpos[3:7].copy())
    qg = np.clip(goal["q"][qad], lo + 0.03, hi - 0.03) if kind == "standing" else None; best_g = np.inf; since = 0
    qs_all = np.array(q)
    if kind == "standing": qs_all[:len(goal["q"])] = goal["q"]
    def fwd(q_): d.qpos[:] = q_; d.qvel[:] = 0; mujoco.mj_forward(m, d)
    def placed(k):
        if k not in foot_body: return k == "pelvis" and seat_i is not None and any(c[3] == seat_i for c in contacts(S, d, groups, surfs, TOL, ["pelvis"]))
        b = foot_body[k]; R = d.xmat[b].reshape(3, 3)
        # a foot wedged under a raised surface (P25: heels inside the seat box, ~25 kN contacts) is TRAPPED, not re-placeable: it stays a
        # support and only its non-embedded points count (contacts() drops embedded points); re-placing it was tried and cannot happen in physics
        return bool(R[2, 2] > np.cos(np.radians(15)) and len(contacts(S, d, groups, surfs, TOL, [k])) >= 2)      # >= 2 points: a planted, slightly tilted foot is levelled in place by the pin + level rows
    fwd(q)
    active = sorted({c[0] for c in contacts(S, d, groups, surfs, TOL)}); cert0 = certificate(S, q, groups, surfs, active, margin=0.0)   # every start contact is declared (released at once when not needed)
    Q = [q.copy()]; A = [tuple(active)]; Ss = [cert0["s"]]; Ms = [cert0["margin"]]; TAU = [cert0["tau"]]; events = [(0, "start", tuple(active))]
    targets = []; boost = 1.0; last_fail = None; swing = None; swing_target = None; stall = 0; aux_target = None; aux_tried = 0; free_iters = 0; aux_held = set(); reason = "max_steps"; reached = False; dqm = dq_max
    def foot_target(k):
        """Goal placement of foot k: the stance offset from the other foot when that one is placed, else from the COM projection."""
        other = "right_foot" if k == "left_foot" else "left_foot"; ps, qs_ = st["foot"][k] if st else (d.xpos[foot_body[k]].copy(), d.xquat[foot_body[k]].copy())
        if st is None: return ps, qs_
        yaw = np.arctan2(2*(d.qpos[3]*d.qpos[6] + d.qpos[4]*d.qpos[5]), 1 - 2*(d.qpos[5]**2 + d.qpos[6]**2)); c, s_ = np.cos(yaw), np.sin(yaw); Rz = np.array([[c, -s_], [s_, c]])
        if other in active and placed(other): base = d.xpos[foot_body[other]][:2] + Rz @ (ps[:2] - st["foot"][other][0][:2])
        else: base = d.subtree_com[root][:2] + Rz @ (ps[:2] - st["com"][:2])
        quat = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
        # the placement must be clear of raised surfaces: the sole outline (stance foot points, foot frame) moved by the least shift
        # that takes every point out of every raised footprint (+3 cm)
        ds_ = mujoco.MjData(m); ds_.qpos[:] = qs_all; mujoco.mj_forward(m, ds_); bS = foot_body[k]; RS = ds_.xmat[bS].reshape(3, 3)
        rel = [RS.T @ (p_ - ds_.xpos[bS]) for g in groups.get(k, []) for p_ in geom_points(m, ds_, g)]
        Rz3 = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]); tgt = np.array([base[0], base[1], ps[2]])
        for s_ in surfs:
            if not np.isfinite(s_["half"][0]): continue
            pts = [(tgt + Rz3 @ r_)[:2] for r_ in rel]
            if not any(_inside(s_, p_, pad=0.03) for p_ in pts): continue
            best_shift = None; cur = d.xpos[foot_body[k]][:2]
            for ax in range(2):
                for sg in (1.0, -1.0):
                    loc = [s_["R"].T @ (p_ - s_["c"]) for p_ in pts]; v = s_["R"][:, ax]*sg*max(0.0, max(s_["half"][ax] + 0.03 - sg*l_[ax] for l_ in loc))
                    if best_shift is None or np.linalg.norm(tgt[:2] + v - cur) < np.linalg.norm(tgt[:2] + best_shift - cur): best_shift = v     # the exit nearest the foot's current position (least travel)
            tgt[:2] += best_shift
        return tgt, quat
    for it in range(max_steps):
        fwd(q); com = d.subtree_com[root].copy()
        # ---- 1. releases: a non-goal (or unplaced) contact goes only when the certificate holds without it
        goals_done = all(k in active and placed(k) for k in goal_groups if k.endswith("foot")) if kind == "standing" else True
        if swing is None:
            for k in [k for k in active if k not in goal_groups]:
                if k in aux_held and not goals_done: continue                  # an acquired auxiliary support stays until the feet are placed
                rest = [a for a in active if a != k]; rp = [c[2][:2] for c in contacts(S, d, groups, surfs, TOL, rest)]
                need = max(margin, 0.6*support_margin(rp, np.mean(rp, 0))) if len(rp) >= 3 else margin     # the COM most of the way to the deepest point of what remains
                if rest and certificate(S, q, groups, surfs, rest, margin=need)["ok"]:
                    active = rest; events.append((len(Q) - 1, "release", (k,))); break
        # ---- 2. what must move: an unplaced goal foot (free its limb, swing, place) or an auxiliary arm
        unplaced = [k for k in ("left_foot", "right_foot") if not (k in active and placed(k))] if kind == "standing" else []
        com_target = None; keep = list(active)
        if swing is not None:
            keep = [a for a in active if a not in limb_of(swing)]
        elif unplaced:
            # free the limb of the unplaced foot whose remaining support has the largest margin under the COM's reach
            best = None
            for k in unplaced:
                rest = [a for a in active if a not in limb_of(k)]; cs_ = contacts(S, d, groups, surfs, TOL, rest)
                if len(cs_) < 3: continue
                pts = np.array([c[2][:2] for c in cs_]); cen = pts.mean(0); mg = support_margin(pts, cen)
                if mg > 0.02 and (best is None or mg > best[0]): best = (mg, k, rest, cen)
            if best is not None and aux_target is None:
                _, k, rest, cen = best; com_target = cen; free_iters += 1
                if certificate(S, q, groups, surfs, rest, margin=margin)["ok"]:
                    released = [a for a in active if a in limb_of(k)]; active = rest; swing = k; swing_target = foot_target(k); stall = 0; free_iters = 0
                    events.append((len(Q) - 1, "release_limb", tuple(released))); keep = list(active); targets.append((swing, swing_target[0].round(3).tolist()))
            if aux_target is None and swing is None and (best is None or free_iters > 150):
                # no limb can be freed (no area left under the COM, or the COM cannot get there): acquire a free arm as an auxiliary
                # support at the floor point maximising the margin of (rest + hand) at the current COM; ranked, next one on a stall
                cands = _aux_points(S, d, groups, surfs, active, unplaced, root, limb_of, floor_i)
                if aux_tried >= 2*min(len(cands), 6): reason = "no_auxiliary_support"; break
                ci_ = aux_tried//2; aux_target = dict(group=cands[ci_][1], point=cands[ci_][2], iters=0, proximal=(aux_tried % 2 == 1)); aux_tried += 1; free_iters = 0   # each point: the limb end's lowest point first, then its most proximal geom
                events.append((len(Q) - 1, "acquire_target", (aux_target["group"],)))
        if com_target is None:
            cs_ = contacts(S, d, groups, surfs, TOL, [a for a in keep if a in goal_groups and placed(a)] or keep)
            pts = np.array([c[2][:2] for c in cs_]) if cs_ else com[None, :2]; com_target = pts.mean(0)
        # ---- 3. differential IK step
        rows = []; rhs = []; wts = []
        def add(Jr, br, w):
            rows.append(Jr[:, idx] if Jr.shape[1] == nv else Jr); rhs.append(np.atleast_1d(br)); wts.append(np.full(len(np.atleast_1d(br)), w))
        jp = np.zeros((3, nv)); jr = np.zeros((3, nv))
        for (k, g, p, si, gap) in contacts(S, d, groups, surfs, TOL, keep):     # pinned support points (no slip, on the surface)
            pw = p.copy(); pw[2] = p[2] + gap; mujoco.mj_jac(m, d, jp, jr, pw, int(m.geom_bodyid[g])); add(jp.copy(), np.array([0.0, 0.0, -gap]), 30.0)
        for k in keep:                                                          # placed feet stay flat
            if k in foot_body and placed(k):
                mujoco.mj_jacBody(m, d, jp, jr, foot_body[k]); add(jr.copy(), np.clip(_quat_err(d.xquat[foot_body[k]], _level(d.xquat[foot_body[k]])), -0.01, 0.01), 30.0)
        for k, gs in groups.items():                                            # everything else stays clear of every surface
            if k in keep or (aux_target is not None and k == aux_target["group"]): continue
            if swing is not None and k == swing and np.linalg.norm(swing_target[0][:2] - d.xpos[foot_body[swing]][:2]) <= 0.03: continue   # touch-down
            for g in gs:
                for p in geom_points(m, d, g):
                    for s in surfs:
                        if _inside(s, p[:2], pad=(0.06 if k.endswith(("hand", "arm")) else 0.0)) and -0.05 < p[2] - s["z"] < 0.015:
                            mujoco.mj_jac(m, d, jp, jr, p, int(m.geom_bodyid[g])); add(jp[2:3].copy(), np.array([0.015 - (p[2] - s["z"])]), 50.0); break
        near = np.where((q[qad] < lo + 0.03) | (q[qad] > hi - 0.03))[0]            # joint limits as soft inequalities (a clamp after the step tilted the feet)
        if len(near):
            Jl = np.zeros((len(near), nv)); Jl[np.arange(len(near)), dofs[near]] = 1.0; add(Jl, np.clip(np.where(q[qad][near] < lo[near] + 0.03, lo[near] + 0.03, hi[near] - 0.03) - q[qad][near], -dqm, dqm), 30.0)
        mujoco.mj_jacSubtreeCom(m, d, jp, root); ec = np.clip(com_target - com[:2], -0.01, 0.01)
        add(jp[:2].copy(), ec, 10.0*boost)
        w_post = 1.0 if (swing is None and not unplaced and all(a in goal_groups for a in active)) else 0.15
        if kind == "standing":
            eq = np.clip(qg - q[qad], -dqm, dqm); Jq = np.zeros((len(qad), nv)); Jq[np.arange(len(qad)), dofs] = 1.0; add(Jq, eq, w_post)
            ro = np.clip(_quat_err(q[3:7], st["quat"]), -0.02, 0.02); mujoco.mj_jacBody(m, d, jp, jr, root)
            add(jr[:2].copy(), ro[:2], (3.0 if w_post >= 1.0 else 0.3))      # final phase: the pelvis upright leads (standing is judged on pelvis height and tilt)
        else:
            zt = goal["z"] if kind == "lower" else (surfs[seat_i]["z"] if seat_i is not None else 0.3)
            mujoco.mj_jacBody(m, d, jp, jr, root)
            if kind == "lower": add(jp[2:3].copy(), np.clip([zt - d.xpos[root][2]], -0.01, 0.01), 3.0)
            else:
                pel = contacts(S, d, groups, surfs, 0.2, ["pelvis"]); low = min([lowest_gap(S, d, g, [surfs[seat_i]]) for g in groups.get("pelvis", [])] + [np.inf]) if seat_i is not None else np.inf
                tx = np.array([goal["x"] - d.xpos[root][0]]); add(jp[0:1].copy(), np.clip(tx, -0.01, 0.01), 1.0)
                if np.isfinite(low): add(jp[2:3].copy(), np.clip([-low], -0.01, 0.01), 3.0)
                else: add(jp[2:3].copy(), np.array([-0.01]), 3.0)
            ro = np.clip(_quat_err(q[3:7], np.array([1.0, 0, 0, 0])), -0.02, 0.02); add(jr[:2].copy(), ro[:2], 0.1)
            Jq = np.zeros((len(qad), nv)); Jq[np.arange(len(qad)), dofs] = 1.0; add(Jq, np.zeros(len(qad)), 0.05)
        if swing is not None:                                                   # swing the freed foot: clear, travel, descend flat
            b = foot_body[swing]; pt, qt = swing_target; pc = d.xpos[b].copy(); hor = np.linalg.norm(pt[:2] - pc[:2])
            clear = min([lowest_gap(S, d, g, surfs) for g in groups.get(swing, [])] + [np.inf])
            zt = pt[2] + (0.06 if hor > 0.03 else 0.0); tgt = np.array([pt[0], pt[1], zt]) if clear > 0.02 or hor <= 0.03 else np.array([pc[0], pc[1], pc[2] + 0.02])
            mujoco.mj_jacBody(m, d, jp, jr, b); add(jp.copy(), np.clip(tgt - pc, -0.015, 0.015), 8.0); add(jr.copy(), np.clip(_quat_err(d.xquat[b], qt), -0.03, 0.03), 4.0)
        if aux_target is not None:                                               # the auxiliary hand: its lowest point to the floor point
            gs_ = groups[aux_target["group"]]
            if aux_target["proximal"]: gs_ = [min(gs_, key=lambda g: _depth(m, int(m.geom_bodyid[g])))]   # load through the most PROXIMAL geom of the limb end first (least lever on the weak distal joints: wrist pitch / yaw 5 N m on the G1)
            gl = min(((p_, g) for g in gs_ for p_ in geom_points(m, d, g)), key=lambda t: t[0][2]); aux_target["iters"] += 1
            mujoco.mj_jac(m, d, jp, jr, gl[0], int(m.geom_bodyid[gl[1]])); add(jp.copy(), np.clip(aux_target["point"] - gl[0], -0.015, 0.015), 8.0)
            if aux_target["iters"] > 200: events.append((len(Q) - 1, "acquire_failed", (aux_target["group"],))); aux_target = None
        Am = np.vstack(rows); bm = np.concatenate(rhs); W = np.concatenate(wts)
        H = (Am*W[:, None]**2).T @ Am + 1e-3*np.diag(wreg/np.max(wreg)); v = np.linalg.solve(H, (Am*W[:, None]**2).T @ bm)
        vj = v[6:]; sc = max(1.0, float(np.max(np.abs(vj)))/dqm, float(np.linalg.norm(v[:3]))/0.015); v = v/sc
        accepted = False
        for shrink in (1.0, 0.5, 0.25):
            vv = np.zeros(nv); vv[idx] = v*shrink; qn = q.copy(); mujoco.mj_integratePos(m, qn, vv, 1.0); qn[qad] = np.clip(qn[qad], lo, hi)
            c = certificate(S, qn, groups, surfs, active, margin=(margin if swing is None and aux_target is None else 0.01))
            if c["ok"] or (len(Q) == 1 and not cert0["ok"]): accepted = True; break
        if not accepted:
            stall += 1
            last_fail = {k: v for k, v in c.items() if k != "tau"}; boost = min(boost*2.0, 64.0)     # a rejected step: the COM term dominates the next try
            if stall > 25 and aux_target is not None:                          # an unreachable auxiliary point: abandon it, next candidate
                events.append((len(Q) - 1, "acquire_failed", (aux_target["group"],))); aux_target = None; stall = 0; boost = 1.0; continue
            if stall > 25: reason = "stalled_uncertified"; break
            continue
        q = qn; fwd(q); boost = max(1.0, boost*0.9)
        # ---- 4. touch-downs: the swing foot placed flat at its target, the auxiliary hand on the floor, the pelvis on the seat
        if swing is not None and placed(swing) and np.linalg.norm(d.xpos[foot_body[swing]][:2] - swing_target[0][:2]) < 0.05:
            new = sorted(set(active) | {swing}); events.append((len(Q), "place", (swing,))); active = new; swing = None; swing_target = None
        if aux_target is not None and contacts(S, d, groups, surfs, TOL, [aux_target["group"]]):
            active = sorted(set(active) | {aux_target["group"]}); aux_held.add(aux_target["group"]); events.append((len(Q), "acquire", (aux_target["group"],))); aux_target = None; aux_tried = 0
        if kind == "seated" and "pelvis" not in active and placed("pelvis"):
            active = sorted(set(active) | {"pelvis"}); events.append((len(Q), "acquire", ("pelvis",)))
        Q.append(q.copy()); A.append(tuple(active)); Ss.append(c["s"]); Ms.append(c["margin"]); TAU.append(c["tau"]); stall = 0
        # ---- 5. done?
        fwd(q)
        if kind == "standing" and set(active) <= goal_groups and all(placed(k) for k in ("left_foot", "right_foot")):
            gerr = float(np.max(np.abs(qg - q[qad])))
            if gerr < 0.05 and abs(d.qpos[2] - st["pelvis_z"]) < 0.02: reached = True; reason = "goal"; break
            best_g = min(best_g, gerr); since = 0 if gerr < best_g - 1e-3 else since + 1
            if since > 60 and abs(d.qpos[2] - st["pelvis_z"]) < 0.03: reached = True; reason = f"goal_posture_converged (max joint err {gerr:.2f})"; break
        if kind == "lower" and set(active) <= goal_groups and abs(d.xpos[root][2] - goal["z"]) < 0.005: reached = True; reason = "goal"; break
        if kind == "seated" and "pelvis" in active and abs(d.xpos[root][0] - goal["x"]) < 0.05: reached = True; reason = "goal"; break
        if verbose and it % 50 == 0: print(f"  it {it} active {active} swing {swing} z {d.qpos[2]:.3f} com {com.round(3)} s {c['s']:.2f} mg {c['margin']:.3f}", flush=True)
    return dict(Q=np.array(Q), active=A, s=np.array(Ss, float), margin=np.array(Ms, float), tau=TAU, events=events, reached=reached, reason=reason, goal=goal,
                groups=sorted(groups), surfaces=[s["name"] for s in surfs], n=len(Q), last_fail=last_fail, swing_targets=targets)


def _aux_points(S, d, groups, surfs, active, unplaced, root, limb_of, floor_i, max_r: float = 0.65) -> list:
    """Ranked [(margin, hand group, floor point)] for the free hands: floor points within max_r of the COM projection (reachability is
    left to the whole-body IK, which may bend the trunk; a stalled reach moves on to the next candidate), scored by the support margin
    at the current COM of (active minus the unplaced foot's limb) plus the point. Only positive-margin candidates."""
    com = d.subtree_com[root][:2]; out = []
    for arm in ("left_hand", "right_hand"):
        if arm in active or arm not in groups: continue
        for k in unplaced:
            rest = [a for a in active if a not in limb_of(k)]; base = [c[2][:2] for c in contacts(S, d, groups, surfs, TOL, rest)]
            for r in np.linspace(0.15, max_r, 11):
                for a in np.linspace(-np.pi, np.pi, 24, endpoint=False):
                    p = com + r*np.array([np.cos(a), np.sin(a)])
                    if (arm == "left_hand") != (np.sin(a) > -0.2) or not any(_inside(surfs[i], p) for i in floor_i): continue   # a hand on its own side
                    mg = support_margin(base + [p], com)
                    if mg > 0.0: out.append((mg - 0.05*r, arm, np.array([p[0], p[1], 0.0])))
    out.sort(key=lambda t: -t[0]); return out


def shortcut(S, P: dict, max_skip: int = 24, n_check: int = 3) -> list:
    """Indices of the path kept as tracking nodes: within a run of one support set, jump from node i to the farthest j (<= i + max_skip)
    whose straight interpolation (qpos, quaternion renormalised) is certified on that set at n_check interior points with the pinned
    contacts within 1 cm of their surfaces; every support change and the last node are kept."""
    import mujoco
    Q = P["Q"]; A = P["active"]; n = len(Q)
    if n <= 2: return list(range(n))
    groups = contact_groups(S); surfs = surfaces(S); m = S["m"]; d = mujoco.MjData(m)
    def interp(a, b, w):
        q = (1 - w)*Q[a] + w*Q[b]; q[3:7] /= np.linalg.norm(q[3:7]); return q
    def ok(a, b):
        for w in np.linspace(0, 1, n_check + 2)[1:-1]:
            q = interp(a, b, w); d.qpos[:] = q; mujoco.mj_forward(m, d)
            cs = contacts(S, d, groups, surfs, 0.01, A[a])
            if {c[0] for c in cs} != set(A[a]) or not certificate(S, q, groups, surfs, A[a], margin=0.01)["ok"]: return False
            for k, gs in groups.items():                                     # nothing outside the support set goes through a surface
                if k in A[a]: continue
                if any(lowest_gap(S, d, g, surfs) < -0.002 for g in gs): return False
        return True
    keep = [0]; i = 0
    while i < n - 1:
        j_best = i + 1; hi_ = i + 1
        while hi_ < min(n - 1, i + max_skip) and A[hi_ + 1] == A[i]: hi_ += 1
        lo_ = i + 1                                                          # bisection on the farthest certified jump (monotone in practice)
        while lo_ < hi_:
            mid = (lo_ + hi_ + 1)//2
            if ok(i, mid): lo_ = mid
            else: hi_ = mid - 1
        j_best = max(lo_, i + 1); keep.append(j_best); i = j_best
    return keep


def constrained_com_jacobian(S, q, active, groups=None, surfs=None) -> np.ndarray:
    """d COM / d q_driven with the ACTIVE support pinned (contact points fixed; placed feet also non-rotating): the floating-base
    velocity solved in least squares from the pins (throw.constrained_jacobian's construction over an arbitrary support set). 3 x nT."""
    import mujoco
    m = S["m"]; J = S["jarr"]; dofs = np.asarray(J["dof"]); d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_forward(m, d)
    groups = groups or contact_groups(S); surfs = surfs or surfaces(S); jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv)); rows = []
    for (k, g, p, si, gap) in contacts(S, d, groups, surfs, 0.01, active):
        mujoco.mj_jac(m, d, jp, jr, p, int(m.geom_bodyid[g])); rows.append(jp.copy())
        if k.endswith("foot"): rows.append(jr.copy())
    root = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")); jc = np.zeros((3, m.nv)); mujoco.mj_jacSubtreeCom(m, d, jc, root)
    if not rows: return jc[:, dofs]
    Jp = np.vstack(rows); B = np.linalg.lstsq(Jp[:, :6], -Jp[:, dofs], rcond=None)[0]        # base velocity per unit joint velocity
    return jc[:, :6] @ B + jc[:, dofs]


def constrained_ieff(S, q, active, groups=None, surfs=None, cap: float = 200.0) -> np.ndarray:
    """Effective joint inertias WITH THE ACTIVE SUPPORT PINNED: 1 / diag(M^-1 - M^-1 Jp^T (Jp M^-1 Jp^T)^-1 Jp M^-1) over the driven dofs
    (the Tracker's 1/diag(M^-1) is the free-floating value: an ankle roll then feels the foot only and its PD holds nothing). Capped
    at `cap` x the free value."""
    import mujoco
    m = S["m"]; J = S["jarr"]; dofs = np.asarray(J["dof"]); d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, M, d.qM); Mi = np.linalg.inv(M); free = 1.0/np.diag(Mi)[dofs]
    groups = groups or contact_groups(S); surfs = surfs or surfaces(S); jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv)); rows = []
    for (k, g, p, si, gap) in contacts(S, d, groups, surfs, 0.01, active):
        mujoco.mj_jac(m, d, jp, jr, p, int(m.geom_bodyid[g])); rows.append(jp.copy())
        if k.endswith("foot"): rows.append(jr.copy())
    if not rows: return free
    Jp = np.vstack(rows); A = Jp @ Mi @ Jp.T; Mc = Mi - Mi @ Jp.T @ np.linalg.pinv(A, rcond=1e-8) @ Jp @ Mi
    return np.clip(1.0/np.maximum(np.diag(Mc)[dofs], 1e-12), free, cap*free)
