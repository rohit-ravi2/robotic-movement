"""LUGGAGE probe (2026-10-17): move a 100 lb (45.4 kg) bag >= 1.0 m horizontally in <= 8 s on stock G1H (G1 + Dex3), robot never falls,
support gate (each loaded foot <= 2 cm travel), valid at 1 ms and 0.5 ms on fresh plants. The task defines only the outcome.

Bag (0.70 x 0.45 x 0.28 m; bag frame x = depth (front +), y = width, z = height; BACK face x = -0.14 carries the pull handle and wheels):
  shell box 45.0 kg (COM at the centre), bag-floor sliding mu 0.4 (bag geom priority 2), contact solref 0.005 / solimp 0.9-0.95;
  top handle (capsule along y on the top face), side handle (on the +y face), telescoping pull handle (two rods + a bar; bar 1.0 m above
  the floor with the bag upright; rigid, massless, non-colliding: a grip feature only);
  two fixed WHEELS (cylinders r 0.04, 0.2 kg each, hinge joints, no joint damping) on the back-bottom edge, their bottoms 5 mm ABOVE the
  base plane, so upright the bag rests on its base (drag) and tipped > ~12 deg it rests on the wheels only. ROLLING RESISTANCE comes
  from the contact model: wheel contacts condim 6, friction (slide 1.0, torsion 0.005, roll 0.002 m) -> rolling resistance coefficient
  0.002/0.04 = 0.05. Total 45.4 kg; the wheels shift the COM 1.5 mm back / 2.7 mm down.
GRIP (labelled): MuJoCo CONNECT equality (a point weld: unlimited force, transmits NO moment) palm site -> handle grip point, solref
0.005 / solimp 0.9-0.95 (the sled_dyn weld convention); hand-bag collisions excluded (the constraint is the grip contact). The force
it carries is logged and compared with the Dex3 rating (PREDICTIONS.md). Dex3 finger grip is NOT exercised here.
LOCOMOTION: none. The step construction (movement_discovery/construction/step.py) exists but its prior falls on G1H for any 2-step
sequence even unloaded (1 step of 0.2 m valid at both dt; 2 steps or a 1 m x_min walk FELL), and making it walk needs a search the
2.5 h budget cannot fund (about 104 CPU-h estimated for a quasi-static walk). So the robot pulls / pushes with its feet planted: the
displacement is capped by arm reach + lean (PREDICTIONS.md).
Strategies (start states, each a physical option): S1 both hands on the top handle, bag upright on the floor in front (must lift);
S2 both hands on the pull bar, bag upright (drag on its base; tilt > 10 deg = BAG_TIPPED, it became S3); S3 both hands on the pull
bar, bag tipped back onto its wheels at a searched tilt (start-state abstraction: the tip itself is predicted, not executed).
Search vector (23-D, same for every strategy): stand-controller gain scales (2); phase A symmetric posture offsets (waist pitch,
shoulder pitch, shoulder roll (mirrored), elbow, wrist pitch, hip/knee/ankle pitch) (8) + capture-point lean set-point (1); phase B
deltas on the same (8 + 1) switched at t_B in [1, 5] s, ramped 1 s (1); strategy: grip reach x (1), S3 tilt (1; unused otherwise).
COMBINED: one more leading dimension picks the strategy (3 bins). Score (min) = -min(disp, 1.2) + 0.1 util_rms if valid, else 10 + 2 (1 - t_held/T) - 0.5 min(disp, 1.2)
(fell / feet > 2 cm / strategy gate), the WORSE of 1 ms and 0.5 ms. S1 counts only displacement while the bag is fully off the floor (+ a lift-height
shaping term). CLI: python luggage.py predict | search S1|S2|S3|C [--iters --pop --workers] | render S? | final S?
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "luggage"))
G = 9.80665
MASS = 45.4; WHEEL_M = 0.2; SHELL_M = MASS - 2*WHEEL_M
HX, HY, HZ = 0.14, 0.225, 0.35
WR, WHL = 0.04, 0.02; WHEEL_C = np.array([-HX - 0.025, 0.18, -HZ + 0.005 + WR])      # wheel centre (bag frame, +-y)
BAR = np.array([-0.12, 0.0, 0.65]); BAR_HW = 0.15                                   # pull bar centre (bag frame): 0.35 + 0.65 = 1.0 m above the floor upright
TOP = np.array([0.0, 0.0, HZ + 0.03])                                             # top handle centre
GRIPS = {"S1": [TOP + [0, 0.08, 0], TOP + [0, -0.08, 0]], "S2": [BAR + [0, 0.10, 0], BAR + [0, -0.10, 0]], "S3": [BAR + [0, 0.10, 0], BAR + [0, -0.10, 0]]}   # (left, right)
MU_BASE = 0.4; ROLL = (1.0, 0.005, 0.002); SOFT = ([0.005, 1.0], [0.9, 0.95, 0.001, 0.5, 2.0])
SIDES = ("left", "right")
POST = ["waist_pitch_joint", "shoulder_pitch_joint", "shoulder_roll_joint", "elbow_joint", "wrist_pitch_joint"]
PSCALE = np.array([0.3, 1.0, 0.4, 1.0, 0.6, 0.3, 0.3, 0.3])     # rad per unit: waist, sh pitch, sh roll, elbow, wrist pitch, hip, knee, ankle
LEGS = ("hip_pitch_joint", "knee_joint", "ankle_pitch_joint")
LEAN_SCALE = 0.15; DIM = 23; T_DEFAULT = 8.0
STRATS = ("S1", "S2", "S3")


def scene_path():
    from engine import world as W
    path = f"{W.SCENE_DIRS['G1H']}/scene_luggage_v2.xml"
    if os.path.exists(path): return path
    spec = mujoco.MjSpec.from_file(W.BASE["G1H"])
    b = spec.worldbody.add_body(name="bag", pos=[2.0, 0, HZ + 0.01]); b.add_freejoint(name="bag_free")
    g = b.add_geom(name="bag_shell", type=mujoco.mjtGeom.mjGEOM_BOX, size=[HX, HY, HZ], mass=SHELL_M, rgba=[0.15, 0.25, 0.45, 1])
    g.friction = [MU_BASE, 0.005, 0.0001]; g.priority = 2; g.solref, g.solimp = SOFT
    cap = mujoco.mjtGeom.mjGEOM_CAPSULE; qy = [0.70710678, 0.70710678, 0, 0]       # capsule z -> bag y
    def vis(name, pos, half, quat=(1, 0, 0, 0)):
        gg = b.add_geom(name=name, type=cap, size=[0.012, half, 0], pos=list(pos), quat=list(quat), mass=0.0, rgba=[0.8, 0.8, 0.82, 1]); gg.contype = 0; gg.conaffinity = 0
    vis("handle_top", TOP, 0.10, qy); vis("handle_side", [0.0, HY + 0.03, 0.0], 0.07)
    for s in (1, -1): vis(f"pull_rod{'L' if s > 0 else 'R'}", [BAR[0], s*BAR_HW, (HZ + BAR[2])/2], (BAR[2] - HZ)/2)
    vis("pull_bar", BAR, BAR_HW, qy)
    for s, nm in ((1, "L"), (-1, "R")):
        wb = b.add_body(name=f"wheel{nm}", pos=[WHEEL_C[0], s*WHEEL_C[1], WHEEL_C[2]]); wb.add_joint(name=f"wheel{nm}_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
        wg = wb.add_geom(name=f"wheel{nm}_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[WR, WHL, 0], quat=qy, mass=WHEEL_M, rgba=[0.1, 0.1, 0.1, 1])
        wg.condim = 6; wg.friction = list(ROLL); wg.priority = 2; wg.solref, wg.solimp = SOFT
    for s in SIDES:
        e = spec.add_equality(name=f"grip_{s}", type=mujoco.mjtEq.mjEQ_CONNECT, objtype=mujoco.mjtObj.mjOBJ_BODY, name1=f"{s}_wrist_yaw_link", name2="bag"); e.active = False
    for bd in spec.bodies:
        if bd.name and ("wrist_yaw_link" in bd.name or "_hand_" in bd.name):
            for o in ("bag", "wheelL", "wheelR"): spec.add_exclude(bodyname1=bd.name, bodyname2=o)
    for k in spec.keys:
        k.qpos = list(k.qpos) + [2.0, 0, HZ + 0.01, 1, 0, 0, 0, 0, 0]
        if len(k.qvel): k.qvel = list(k.qvel) + [0.0]*8
    spec.compile(); tmp = f"{path}.{os.getpid()}.tmp"; open(tmp, "w").write(spec.to_xml()); os.replace(tmp, path); return path


_PL = {}
def plant(dt):
    """A FRESH plant per timestep (built at that dt; TH.setup caches one plant per start, so the start is offset by 1e-9 per dt)."""
    if dt in _PL: return _PL[dt]
    import throw as TH, branch_i as BI
    path = scene_path(); S0 = TH.setup("G1H", mjcf=path, ball=False)
    S = TH.setup("G1H", start=np.array(S0["q0"]) + 1e-9*(1 + int(round(1e4*dt))), mjcf=path, ball=False); BI.apply_dt(S, dt)
    # HELD leg joints (hip roll, ankle roll) get the dt-INDEPENDENT stiffness of the driven legs (construction/step.py's fix: the held
    # form kp ~ I/dt^2 is 4x stiffer at 0.5 ms and made the stand creep sideways 5-10 cm/s at 0.5 ms on G1H). A controller gain, not hardware.
    from engine import core
    om = float(core.landing_controller(S, "G1H")[0].p["stiff"]); m = S["m"]; held = []
    for (a_, dof, qad, tau, kp, kd) in S["held"]:
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.dof_jntid[dof]))
        if any(k in jn for k in ("hip", "knee", "ankle")):
            ie = 1.0/float(S["Minv_key"][dof, dof]); kp = om*om*ie; kd = 2.0*np.sqrt(kp*ie); BI._as_servo(m, int(a_), kp, kd, float(tau))
        held.append((a_, dof, qad, tau, float(kp), float(kd)))
    S["held"] = held; BI.settle(S)
    m = S["m"]; S["bag"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bag"); S["bag_qa"] = int(m.jnt_qposadr[m.body_jntadr[S["bag"]]])
    S["bag_da"] = int(m.jnt_dofadr[m.body_jntadr[S["bag"]]]); S["q_settled0"] = np.array(S["q_settled"]); S["v_settled0"] = np.array(S["v_settled"]); S["scene"] = path
    _PL[dt] = S; return S


def Ry(a): return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def tilt_of(x):                 # S3 tilt (rad, top towards the robot) from the strategy number in [-1, 1]: 15..50 deg
    return np.radians(32.5 + 17.5*float(np.clip(x, -1, 1)))


def bag_pose(strat, xr, xt):
    """Bag rotation, position and the two grip targets (world). The robot faces +x at the origin; the bag is in front, its back face
    (handles, wheels) towards the robot. xr in [-1,1] -> grip reach x 0.22..0.46 m (S1: top handle; S2/S3: bar)."""
    R = np.eye(3) if strat != "S3" else Ry(-tilt_of(xt))
    xg = 0.34 + 0.12*float(np.clip(xr, -1, 1))
    lift = 0.025 if strat == "S1" else 0.0          # S1: IK aims 2.5 cm above the top handle (the arms sag ~2.4 cm in the settle)
    if strat == "S3":
        wl = WHEEL_C*[1, 1, 1]; pz = WR - (R @ wl)[2]                      # wheels on the floor
    else: pz = HZ
    gl = GRIPS[strat]; mid = (gl[0] + gl[1])/2; p = np.array([xg - (R @ mid)[0], 0.0, pz])
    return R, p, [R @ g + p + [0, 0, lift] for g in gl]


def setup(S, strat, xr, xt):
    """IK (arms only) so the palms reach the grips; returns (q, info). info['infeasible'] = penalty or None."""
    from engine import leverage as LV
    m = S["m"]; H = LV.g1h_parts(m); R, p, gw = bag_pose(strat, xr, xt); q = np.array(S["q_settled0"])
    q, err = LV.site_ik(m, q, [(H["lpalm"][0], H["lpalm"][1], gw[0]), (H["rpalm"][0], H["rpalm"][1], gw[1])], LV.ARMS, iters=300)     # arms only: the trunk / legs are the search's
    qu = np.zeros(4); mujoco.mju_mat2Quat(qu, R.flatten()); a = S["bag_qa"]; q[a:a+3] = p; q[a+3:a+7] = qu; q[a+7:a+9] = 0
    info = dict(ik_err=float(err), bag_p=p.tolist(), tilt_deg=float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))), grips=[g.tolist() for g in gw], infeasible=None)
    if err > 0.03: info["infeasible"] = 20.0 + err; return q, info
    d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_forward(m, d); pen = 0.0; rb = S["robot_bodies"]; bb = {S["bag"], S["bag"] + 1, S["bag"] + 2}
    for i in range(d.ncon):
        c = d.contact[i]; b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if (b1 in bb and b2 in rb) or (b2 in bb and b1 in rb) or (b1 in rb and b2 in rb): pen = max(pen, -float(c.dist))
    info["start_pen"] = pen          # robot-bag or robot self-penetration at the IK start
    if pen > 0.005: info["infeasible"] = 12.0 + pen
    return q, info


PRESETTLE = 1.5
_START = {}
def _grip_on(m, d, anchors, lps):
    for s, lp, g in zip(SIDES, lps, anchors):
        e = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"grip_{s}"); m.eq_data[e, :] = 0; m.eq_data[e, :3] = lp; m.eq_data[e, 3:6] = g
        m.eq_solref[e] = SOFT[0]; m.eq_solimp[e] = SOFT[1]; d.eq_active[e] = 1


def start_state(S, strat, xr, xt):
    """The task's start: the robot STANDING AT REST holding the grips, the bag resting on the floor (S3: on its wheels, held at the
    tilt by the grips). Built, NOT gated: (1) arm IK pose, bag parked 3 m away, PRESETTLE s of the stand controller (robot COM): the
    servo switch-on transient (a 1-5 cm hop from a raw IK pose) happens here; (2) the bag placed under the ACTUAL palms (rotation from
    bag_pose, x/y shifted so the grip midpoint is at the palm midpoint, grip anchors = the actual palm points in the bag frame), grips
    on, 1.0 s more (the bag's contact settling and the load arriving on the arms). The gated task starts from that state with the
    same grips. Returns (q, v, anchors, info)."""
    key = (strat, round(float(xr), 3), round(float(xt), 3), S["m"].opt.timestep)
    if key in _START: return _START[key]
    import leverage_probes as LP; from engine import leverage as LV
    m = S["m"]; q, info = setup(S, strat, xr, xt)
    if info["infeasible"] is not None: return None, None, None, info
    H = LV.g1h_parts(m); lps = [H["lpalm"][1], H["rpalm"][1]]; last = {}; a = S["bag_qa"]
    def cb(t, d): last["q"] = d.qpos.copy(); last["v"] = d.qvel.copy()
    def run(qs, vs, T, init=None):
        S["q_settled"] = qs; v0 = S["v_settled"]; S["v_settled"] = vs
        try: return LP.stand_rollout(S, np.zeros(2), T, cb, posture=[], robot="G1H", robot_com=True, init_cb=init)
        finally: S["q_settled"] = S["q_settled0"]; S["v_settled"] = v0
    qp = np.array(q); qp[a:a+3] = [3.0, 0.0, HZ + 0.001]; qp[a+3:a+7] = [1, 0, 0, 0]
    r = run(qp, np.zeros(m.nv), PRESETTLE)
    if r["reason"]: info["infeasible"] = 25.0; info["presettle"] = r["reason"]; return None, None, None, info
    q1 = np.array(last["q"]); v1 = np.array(last["v"]); d = mujoco.MjData(m); d.qpos[:] = q1; mujoco.mj_forward(m, d)
    palms = [d.xpos[b] + d.xmat[b].reshape(3, 3) @ lp for b, lp in (H["lpalm"], H["rpalm"])]
    R, p, gw = bag_pose(strat, xr, xt); mid_g = (gw[0] + gw[1])/2; mid_p = (palms[0] + palms[1])/2; p = p + np.array([mid_p[0] - mid_g[0], mid_p[1] - mid_g[1], 0.0])
    anchors = [R.T @ (pp - p) for pp in palms]
    qu = np.zeros(4); mujoco.mju_mat2Quat(qu, R.flatten()); q1[a:a+3] = p; q1[a+3:a+7] = qu; q1[a+7:a+9] = 0; v1[S["bag_da"]:S["bag_da"]+8] = 0
    d.qpos[:] = q1; mujoco.mj_forward(m, d); pen = 0.0; rb = S["robot_bodies"]; bb = {S["bag"], S["bag"] + 1, S["bag"] + 2}
    for i in range(d.ncon):
        c = d.contact[i]; b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if (b1 in bb and b2 in rb) or (b2 in bb and b1 in rb): pen = max(pen, -float(c.dist))
    info.update(start_pen_bag=pen, anchor_err=float(max(np.linalg.norm(an - g) for an, g in zip(anchors, GRIPS[strat]))))
    if pen > 0.005: info["infeasible"] = 12.0 + pen; return None, None, None, info
    r = run(q1, v1, 1.0, init=lambda d_: _grip_on(m, d_, anchors, lps))
    if r["reason"]: info["infeasible"] = 25.0; info["presettle"] = "grip:" + r["reason"]; return None, None, None, info
    info.update(v_start_max=float(np.abs(last["v"][:S["bag_da"]]).max()), bag_p=last["q"][a:a+3].tolist())
    _START[key] = (last["q"], last["v"], anchors, info)
    if len(_START) > 64: _START.pop(next(iter(_START)))
    return last["q"], last["v"], anchors, info


def decode(z, strat=None):
    z = np.clip(np.asarray(z, float), -1, 1)
    if strat is None: strat = STRATS[min(2, int((z[0] + 1)/2*3))]; z = z[1:]          # combined: the leading number picks the strategy
    return strat, dict(gains=z[:2], A=z[2:10], leanA=LEAN_SCALE*z[10], dB=z[11:19], dleanB=LEAN_SCALE*z[19], tB=3.0 + 2.0*z[20], xr=z[21], xt=z[22])


def _eq_force(m, d, eq):
    rows = [i for i in range(d.nefc) if d.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY and d.efc_id[i] == eq]
    return float(np.linalg.norm(d.efc_force[rows])) if rows else 0.0


def evaluate(args):
    """args = (strat or None (combined), z, T, dt, record). dt='both' -> the worse of 1 ms and 0.5 ms (0.5 ms skipped if 1 ms fails)."""
    strat, z, T, dt, record = args
    if dt == "both":
        r1 = evaluate((strat, z, T, 0.001, record))
        r2 = evaluate((strat, z, T, 0.0005, False)) if r1["reason"] is None else dict(r1, reason="SKIPPED_1MS_FAILED")
        r1["dt0005"] = {k: r2.get(k) for k in ("score", "reason", "disp", "grip_peak_N", "foot_travel_max", "util_p95")}
        r1["score"] = max(r1["score"], r2["score"]); r1["disp_min"] = min(r1["disp"], r2["disp"]); r1["ok_both"] = r1["reason"] is None and r2["reason"] is None; return r1
    import leverage_probes as LP
    st, P = decode(z, strat); S = plant(dt); m = S["m"]; q, v_start, anchors, info = start_state(S, st, P["xr"], P["xt"]); info = dict(info)
    if info["infeasible"] is not None:
        return dict(strat=st, score=float(info["infeasible"]), reason="INFEASIBLE_START", disp=0.0, util_rms=None, util_p95=None, grip_peak_N=None, foot_travel_max=None, **info)
    bid = S["bag"]; eqs = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"grip_{s}") for s in SIDES]
    hands = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wrist_yaw_link") for s in SIDES]; from engine import leverage as LV; H = LV.g1h_parts(m)
    lps = [H["lpalm"][1], H["rpalm"][1]]
    acts = {}
    for j in POST:
        for s in SIDES: acts[(j, s)] = LP.actuator_of(m, j if j == "waist_pitch_joint" else f"{s}_{j}")
    legs = [(LP.actuator_of(m, f"{s}_{j}"), k) for k, j in enumerate(LEGS) for s in SIDES]
    offA = PSCALE*P["A"]; offB = PSCALE*P["dB"]
    fs = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in ("left_foot", "right_foot")]
    floor = S["floor"]; bag_geoms = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in (bid, bid + 1, bid + 2)}; shell = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "bag_shell")
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis"); rgeoms = S["robot_geoms"]
    st_ = dict(foot0=None, ftrav=0.0, x0=None, air_disp=0.0, lift=0.0, air_prev=None, tilt_max=0.0, tilt_min=90.0, grip=[0.0, 0.0], body_contact=None, base_contact_late=0.0, nlate=0, pel0=None)
    def wB(t): return float(np.clip((t - P["tB"])/1.0, 0.0, 1.0))
    def extra(t, d, w):
        off = w*offA + wB(t)*offB
        for i, j in enumerate(POST):
            for s in SIDES:
                a = acts[(j, s)]
                if a is None: continue
                if j == "waist_pitch_joint":
                    if s == "left": d.ctrl[a] += off[i]
                else: d.ctrl[a] += off[i]*(-1 if (j == "shoulder_roll_joint" and s == "right") else 1)
        for a, k in legs: d.ctrl[a] += off[5 + k]
    def init(d):
        _grip_on(m, d, anchors, lps)
        st_["foot0"] = [d.site_xpos[i][:2].copy() for i in fs]; st_["x0"] = d.xpos[bid][:2].copy(); st_["pel0"] = d.xpos[pel][:2].copy()
    def cb(t, d):
        tr = max(np.linalg.norm(d.site_xpos[i][:2] - f0) for i, f0 in zip(fs, st_["foot0"])); st_["ftrav"] = max(st_["ftrav"], tr)
        if tr > 0.02: return "FEET_SLID"
        if d.xpos[pel][2] < 0.45: return "FELL"
        k = int(round(t/dt))
        if k % max(1, int(round(0.01/dt))): return None
        tilt = float(np.degrees(np.arccos(np.clip(d.xmat[bid][8], -1, 1)))); st_["tilt_max"] = max(st_["tilt_max"], tilt); st_["tilt_min"] = min(st_["tilt_min"], tilt)
        on_floor = False; base = False
        for ci in range(d.ncon):
            c = d.contact[ci]; g1, g2 = c.geom1, c.geom2
            if floor in (g1, g2) and (g1 in bag_geoms or g2 in bag_geoms):
                on_floor = True; base = base or shell in (g1, g2)
            elif st_["body_contact"] is None and ((g1 in bag_geoms and g2 in rgeoms) or (g2 in bag_geoms and g1 in rgeoms)):
                st_["body_contact"] = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g2 if g1 in bag_geoms else g1)), round(t, 2))
        for h in (0, 1): st_["grip"][h] = max(st_["grip"][h], _eq_force(m, d, eqs[h])) if t > 0.3 else st_["grip"][h]
        xy = d.xpos[bid][:2].copy(); st_["disp_end"] = float(np.linalg.norm(xy - st_["x0"])); st_["pel_drift"] = float(np.linalg.norm(d.xpos[pel][:2] - st_["pel0"]))
        if st == "S1":
            st_["lift"] = max(st_["lift"], float(d.xpos[bid][2] - HZ))
            if not on_floor and st_["air_prev"] is not None: st_["air_disp"] += float(np.linalg.norm(xy - st_["air_prev"]))
            st_["air_prev"] = xy
        if t > 1.0:
            st_["nlate"] += 1; st_["base_contact_late"] += float(base)
        if st == "S2" and tilt > 10.0: return "BAG_TIPPED"
        if tilt > 75.0: return "BAG_TOPPLED"
    lean = lambda t, w: w*P["leanA"] + wB(t)*P["dleanB"]
    S["q_settled"] = q; v0 = S["v_settled"]; S["v_settled"] = v_start
    try: r = LP.stand_rollout(S, np.asarray(P["gains"]), T, cb, record=record, posture=[], robot="G1H", init_cb=init, extra_targets=extra, ramp=1.0, robot_com=True, lean_cb=lean)
    finally:
        S["q_settled"] = S["q_settled0"]; S["v_settled"] = v0
        for e in eqs: m.eq_data[e, :] = 0
    r.update(info); r["strat"] = st; r["scene"] = S["scene"]
    return _finish(r, st, st_, S, T)


def _finish(r, st, st_, S, T):
    r["pelvis_drift"] = st_.get("pel_drift")
    r["foot_travel_max"] = st_["ftrav"]; r["grip_peak_N"] = [round(g, 1) for g in st_["grip"]]; r["tilt_range_deg"] = [st_["tilt_min"], st_["tilt_max"]]
    r["body_contact"] = st_["body_contact"]; r["lift_max"] = st_["lift"]; r["base_on_floor_frac_late"] = st_["base_contact_late"]/max(1, st_["nlate"])
    disp = st_.get("disp_end", 0.0)
    if st == "S1": r["disp_total"] = disp; disp = st_["air_disp"]
    if st == "S3" and r["reason"] is None and r["base_on_floor_frac_late"] > 0.5: r["reason"] = "BAG_UNTIPPED"
    r["disp"] = float(disp)
    fail = (10.0 + 2.0*(1.0 - float(r["t_held"])/T) - 0.5*min(disp, 1.2)) if r["reason"] else None     # a failure: credit for time survived first
    r["score"] = (fail if fail is not None else -min(disp, 1.2) + 0.1*float(r["util_rms"] or 0)) - (0.3*min(st_["lift"], 0.05)/0.05 if st == "S1" else 0.0)
    return r


# ------------------------------------------------------------------------------------------------ PREDICTION (spec only, before search)
def predict():
    """Static LP (engine.leverage.solve) at each strategy's start pose + closed-form bag mechanics + reach cap + Dex3 ratings."""
    import throw as TH; from engine import leverage as LV
    path = scene_path(); S = plant(0.001); m = S["m"]; H = LV.g1h_parts(m); fg = sorted(S["foot_geoms"]); W = MASS*G; out = {}
    tm = LV.tau_max(m); fing = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a][0])): float(tm[a]) for a in H["fingers"]}
    out["dex3_ratings_Nm"] = {k: v for k, v in fing.items() if k.startswith("right")}
    # hook-grip bound: the bar loads the curled index + middle fingers at about the proximal phalanx length from their _0 joints
    i0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_hand_index_0_link"); i1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_hand_index_1_link")
    d = mujoco.MjData(m); d.qpos[:] = S["q_settled0"]; mujoco.mj_forward(m, d); r0 = float(np.linalg.norm(d.xpos[i1] - d.xpos[i0]))
    hook = 2*fing["right_hand_index_0_joint"]/max(r0, 1e-3)
    out["grip_capacity_per_hand_N"] = dict(hook_bound=hook, proximal_len_m=r0, wrap_measured_normal_N=[100, 109], note="hook = 2 fingers x 1.4 N m / proximal length; wrap = J27 grasp lab normal force sum at the 1.4 N m limits (x mu 1 -> ~100 N friction-limited). Range used: 60-100 N per hand.")
    def pose_util(strat, xr, xt, F_on_robot_per_hand):
        q, info = setup(S, strat, xr, xt); d.qpos[:] = q; mujoco.mj_forward(m, d)
        P_ = [d.xpos[b] + d.xmat[b].reshape(3, 3) @ lp for b, lp in (H["lpalm"], H["rpalm"])]
        loads = [LV.point_load(m, d, b, p_, force=F_on_robot_per_hand) for (b, _), p_ in zip((H["lpalm"], H["rpalm"]), P_)]
        r = LV.solve(m, q, loads, fg, robot_bodies=S["robot_bodies"], exclude_act=H["fingers"])
        return dict(status=r["status"], max_util=r.get("max_util"), worst=r.get("worst_joint"), ik_err=info["ik_err"], top=(sorted(r["util"].items(), key=lambda kv: -kv[1])[:4] if r["status"] == "OK" else None))
    # S1: lift 45.4 kg by the top handle, both hands (W/2 down at each palm)
    s1 = pose_util("S1", 0.0, 0.0, [0, 0, -W/2])
    cap = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "lift", "capacity.json")))["G1H"]["poses"]
    out["S1"] = dict(F_grip_per_hand_N=W/2, lp_at_start_pose=s1, capacity_json=dict(suitcase_kg=cap["suitcase (arms hanging at the sides)"]["max_kg"], chest_kg=cap["hug at chest"]["max_kg"]),
                     util_scaled_from_capacity=dict(suitcase=MASS/cap["suitcase (arms hanging at the sides)"]["max_kg"], chest=MASS/cap["hug at chest"]["max_kg"]))
    # S2: drag on the base. Slide needs F_x >= mu (W + F_z); no tip needs F_x h <= arm_w W + arm_z F_z about the leading bottom edge
    h = HZ + BAR[2]; tips = {}
    for nm, arm_w, arm_z in (("pull (towards the bar side)", HX, HX + BAR[0]), ("push", HX, HX - BAR[0])):
        # F_x_slide(Fz) = mu (W + Fz); F_x_tip(Fz) = (arm_w W + arm_z Fz)/h; feasible iff slide <= tip for some Fz >= 0 (pressing down) -> check slope and intercept
        ok = (MU_BASE*W <= arm_w*W/h) or (MU_BASE < arm_z/h)
        tips[nm] = dict(F_slide_N_at_Fz0=MU_BASE*W, F_tip_N_at_Fz0=arm_w*W/h, feasible=bool(ok), tip_ratio_needed=MU_BASE, available=arm_w/h, down_press_gain=arm_z/h)
    s2 = pose_util("S2", 0.0, 0.0, [MU_BASE*W/2, 0, 0])
    out["S2"] = dict(F_drag_N=MU_BASE*W, bar_height_m=h, tip_vs_slide=tips, lp_at_start_pose_with_drag_force=s2,
                     verdict="the bag TIPS before it slides at any push/pull with any downward press (mu 0.4 > half-depth / bar height 0.14) -> a drag on the base is statically infeasible from the pull bar")
    # S3: tip and roll. Bag-frame offsets from the wheel centre: COM (+0.165, 0.305), bar (+0.045, 0.955); tilt th = top towards the robot
    cx, cz = -WHEEL_C[0], -WHEEL_C[2]; bx, bz = BAR[0] - WHEEL_C[0], BAR[2] - WHEEL_C[2]; rows = []
    th_bal = float(np.degrees(np.arctan2(cx, cz)))
    F_tip0 = W*cx/(bz + WR)                       # horizontal pull at the bar to start the tip about the wheel contact (upright)
    for th in np.radians(np.arange(0, 61, 5.0)):
        com_h = cz*np.sin(th) - cx*np.cos(th); bar_h = bz*np.sin(th) - bx*np.cos(th); bar_z = WR + bz*np.cos(th) + bx*np.sin(th)
        Fv = W*com_h/bar_h if (th > 0 and bar_h > 0.05) else None; N = W - (Fv or 0.0)
        rows.append(dict(tilt_deg=float(np.degrees(th)), bar_height_m=float(bar_z), F_vertical_hold_N=Fv, F_roll_N=0.05*N))
    s3 = {}
    for xt in (-1.0, 0.0, 1.0):
        th = tilt_of(xt); com_h = cz*np.sin(th) - cx*np.cos(th); bar_h = bz*np.sin(th) - bx*np.cos(th); Fv = W*com_h/bar_h; Fr = 0.05*(W - Fv) + MASS*0.0625
        s3[f"{np.degrees(th):.1f}"] = dict(F_vertical_N=Fv, F_roll_plus_accel_N=Fr, lp_pull=pose_util("S3", 0.0, xt, [Fr/2, 0, -Fv/2]), lp_push=pose_util("S3", 0.0, xt, [-Fr/2, 0, -Fv/2]))
    out["S3"] = dict(F_tip_start_horizontal_N=F_tip0, balance_tilt_deg=th_bal, table=rows, rolling_resistance_coeff=ROLL[2]/WR, per_tilt=s3,
                     note="task brief said ~half the weight at the start: that is a bag lying flat lifted at one end; this upright bag needs a horizontal pull of W*0.165/1.0 at the bar to start the tip, ~0 at the balance tilt, then a vertical hold rising with tilt")
    # reach cap (feet planted): palm x range reachable by arms-only IK at the S3 grip height (bar), waist pitch -0.3 .. +0.5 (lean)
    reach = {}
    for xt in (-1.0, 0.0, 1.0):
        _, _, gw = bag_pose("S3", 0.0, xt); zt = gw[0][2]; xs = []
        for wp in (-0.3, 0.0, 0.25, 0.5):
            q0 = np.array(S["q_settled0"]); q0[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "waist_pitch_joint")]] = wp
            for x in np.arange(-0.1, 0.8, 0.02):
                q, e = LV.site_ik(m, q0, [(H["lpalm"][0], H["lpalm"][1], (x, 0.1, zt)), (H["rpalm"][0], H["rpalm"][1], (x, -0.1, zt))], LV.ARMS, iters=150)
                if e < 0.02: xs.append(float(x))
        reach[f"{np.degrees(tilt_of(xt)):.1f}"] = dict(bar_z=float(zt), x_min=min(xs) if xs else None, x_max=max(xs) if xs else None, span=(max(xs) - min(xs)) if xs else 0.0)
    out["reach_cap"] = dict(per_tilt=reach, note="bar x span reachable by both palms with the feet planted (arms IK, waist pitch -0.3..0.5); the bag can move at most this far (minus the body: the tipped bag's top cannot pass through the torso)")
    rank = [("S3", "feasible: hold <= ~80 N + roll ~20 N, within joint and Dex3 ratings; displacement capped by reach"),
            ("S1", f"torque-infeasible: 45.4 kg vs 26.2 kg suitcase / 7.9 kg chest capacity; grip {W/2:.0f} N per hand vs 60-100 N Dex3"),
            ("S2", "statically infeasible: tips before it slides (becomes S3); grip 89 N/hand at the Dex3 limit")]
    out["ranking"] = rank
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(f"{OUT}/predictions.json", "w"), indent=1, default=float); return out


def _clean(r): return {k: v for k, v in r.items() if k not in ("trace",)}


def cma_ckpt(mkjob, dim, iters, pop, workers, x0, log, ckpt, wall_s, sigma=0.35, seed=0):
    """leverage_probes.cma_search + a per-generation checkpoint (best z / record / history) and a wall-clock cap (evaluation cost
    varies 5-25 CPU-s with the strategy, so the budget is set in minutes)."""
    import cma, leverage_probes as LP
    es = cma.CMAEvolutionStrategy(x0, sigma, dict(popsize=pop, seed=seed + 1, bounds=[-1, 1], verbose=-9))
    best = (np.inf, None, None); hist = []; t0 = time.time(); nev = 0
    for it in range(iters):
        X = es.ask(); R = LP._pool_map(evaluate, [mkjob(np.asarray(x)) for x in X], workers); nev += len(X)
        f = [r["score"] for r in R]; es.tell(X, f); i = int(np.argmin(f))
        if f[i] < best[0]: best = (f[i], np.asarray(X[i]).tolist(), _clean(R[i]))
        hist.append(dict(it=it, best=float(best[0]), gen_best=float(f[i]), gen_med=float(np.median(f)), n_valid=sum(1 for r in R if r.get("ok_both")),
                         gen_strats={k: sum(1 for r in R if r.get("strat") == k) for k in STRATS}, wall=time.time() - t0))
        print(json.dumps(hist[-1]), file=log, flush=True)
        res = dict(best_score=best[0], best_z=best[1], best=best[2], hist=hist, evals=nev, wall_s=time.time() - t0, s_per_eval_cpu=(time.time() - t0)/nev*max(1, workers), iters_done=it + 1)
        json.dump(res, open(ckpt, "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
        if time.time() - t0 > wall_s: break
    return res


def search(tag, a):
    import leverage_probes as LP
    os.makedirs(OUT, exist_ok=True); strat = None if tag == "C" else tag; dim = DIM + (1 if strat is None else 0)
    log = open(f"{OUT}/search_{tag}.log", "w"); t0 = time.time()
    x0 = np.zeros(dim)
    if strat is None: x0[0] = 0.0                       # combined: starts in the middle bin (S2), not at the predicted winner
    res = cma_ckpt(lambda x: (strat, x, a.T, "both", False), dim, a.iters, a.pop, a.workers, x0, log, f"{OUT}/search_{tag}.ckpt.json", a.wall*60)
    b = res["best"]; J = dict(probe="luggage", tag=tag, robot="G1H (stock menagerie G1 + Dex3)", T=a.T, label=GRIP_LABEL, sim_choices=SIM_CHOICES, search=res,
                              best_strategy=b.get("strat"), decoded={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in decode(np.asarray(res["best_z"]), strat)[1].items()}, wall_s=time.time() - t0)
    json.dump(J, open(f"{OUT}/search_{tag}.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(tag, "best", round(res["best_score"], 3), "strat", b.get("strat"), "disp", round(b.get("disp", 0), 3), "disp_min", b.get("disp_min"), "ok_both", b.get("ok_both"), "reason", b.get("reason"),
          "evals", res["evals"], "wall", round(time.time() - t0), flush=True)


def final(tag, T=T_DEFAULT, render=True):
    """Fresh process: the stored best z at 1 ms and 0.5 ms (fresh plants built at each dt), record at 1 ms, render a clip."""
    J = json.load(open(f"{OUT}/search_{tag}.json")); z = np.asarray(J["search"]["best_z"]); strat = None if tag == "C" else tag
    r1 = evaluate((strat, z, T, 0.001, True)); r2 = evaluate((strat, z, T, 0.0005, False))
    F = dict(tag=tag, strat=r1["strat"], dt001=_clean(r1), dt0005=_clean(r2), valid_both=(r1["reason"] is None and r2["reason"] is None), disp_min=min(r1["disp"], r2["disp"]))
    tm_ = None
    if r1.get("trace") is not None and len(r1["trace"]):
        np.save(f"{OUT}/{tag}_trace.npy", r1["trace"])
        if render:
            from engine import render_any as RA
            RA.render(scene_path(), r1["trace"], r1["trace_dt"], f"{OUT}/{tag}.mp4", title=f"G1H moves a 45.4 kg bag: {r1['strat']} ({GRIP_SHORT})",
                      sub=f"bag moved {r1['disp']:.2f} m (1 ms) / {r2['disp']:.2f} m (0.5 ms), {r1['reason'] or 'valid'} / {r2['reason'] or 'valid'}, feet planted", azimuth=120, distance=3.2)
    json.dump(F, open(f"{OUT}/final_{tag}.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(tag, r1["strat"], "1ms", round(r1["disp"], 3), r1["reason"], "0.5ms", round(r2["disp"], 3), r2["reason"], "grip", r1["grip_peak_N"], r2["grip_peak_N"], "u95", r1["util_p95"], "ft", r1["foot_travel_max"], r2["foot_travel_max"], flush=True)
    return F


GRIP_SHORT = "connect welds, labelled"
GRIP_LABEL = "GRIP = MuJoCo CONNECT equality palm->handle per hand (point weld: unlimited force, no moment; solref 0.005, solimp 0.9/0.95); hand-bag collisions excluded. Dex3 fingers not used."
SIM_CHOICES = ["stock menagerie G1H (scene_hands_palm.xml); no change to ratings, masses, geometry, limits or robot physics",
               "grip: connect equality per hand (weld-class, unlimited force, labelled)", "zero actuator lag (stand-controller servos as in leverage_probes)",
               "bag contacts solref 0.005 / solimp 0.9-0.95; bag-floor mu 0.4 (bag priority); wheels condim 6, friction (1, 0.005, 0.002 m) -> rolling resistance coeff 0.05; wheel hinges undamped",
               "start state built (not gated): 1.5 s stand in the arm-IK pose without the bag + 1.0 s holding the bag; gate clock starts after",
               "S3 starts already tipped (tilt searched 15-50 deg): start-state abstraction; the tip itself is predicted, not executed",
               "no stepping: the step construction falls on G1H for >= 2 steps unloaded (prior only)",
               "support gate: foot site horizontal travel <= 2 cm over the 8 s; FELL = torso tilt > 35 deg or pelvis < 0.45 m",
               "both dt: 1 ms and 0.5 ms; the 0.5 ms run is skipped in the search when 1 ms already fails"]


def mechanism(tag, dt=0.001, T=T_DEFAULT):
    """Replay the stored best and log what binds: per 20 ms the actuator utilisations, grip forces, bag-robot body contacts (by body)."""
    import leverage_probes as LP; from engine import leverage as LV
    J = json.load(open(f"{OUT}/search_{tag}.json")); z = np.asarray(J["search"]["best_z"]); strat = None if tag == "C" else tag
    S = plant(dt); m = S["m"]; tm = LV.tau_max(m); keep = [a for a in range(m.nu) if np.isfinite(tm[a]) and "hand" not in m.actuator(a).name]
    eqs = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"grip_{s}") for s in SIDES]; bb = {S["bag"], S["bag"] + 1, S["bag"] + 2}; rows = []; sat = {}; bodyc = {}
    orig = LP.stand_rollout
    def wrap(*a_, **k_):
        cb = a_[3]
        def cb2(t, d):
            if int(round(t/dt)) % int(round(0.02/dt)) == 0 and t > 0.2:
                u = {m.actuator(a).name: float(abs(d.actuator_force[a])/tm[a]) for a in keep}
                for n_, v in u.items():
                    if v >= 0.98: sat[n_] = sat.get(n_, 0) + 1
                for ci in range(d.ncon):
                    c = d.contact[ci]; b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                    if (b1 in bb) != (b2 in bb) and (b1 in S["robot_bodies"] or b2 in S["robot_bodies"]):
                        f6 = np.zeros(6); mujoco.mj_contactForce(m, d, ci, f6); nb = m.body(b2 if b1 in bb else b1).name; bodyc[nb] = max(bodyc.get(nb, 0.0), float(f6[0]))
                rows.append(dict(t=round(t, 2), grip=[round(_eq_force(m, d, e), 1) for e in eqs], bag_vx=float(d.qvel[S["bag_da"]]), top=sorted(u.items(), key=lambda kv: -kv[1])[:3]))
            return cb(t, d)
        a_ = list(a_); a_[3] = cb2; return orig(*a_, **k_)
    LP.stand_rollout = wrap
    try: r = evaluate((strat, z, T, dt, False))
    finally: LP.stand_rollout = orig
    n = max(1, len(rows)); g = np.array([r_["grip"] for r_ in rows]) if rows else np.zeros((1, 2))
    out = dict(tag=tag, strat=r["strat"], disp=r["disp"], reason=r["reason"], saturated_frac=dict(sorted(((k, v/n) for k, v in sat.items()), key=lambda kv: -kv[1])[:6]),
               grip_mean_N=g.mean(0).tolist(), grip_p95_N=np.percentile(g, 95, axis=0).tolist(), bag_robot_body_contact_peak_N=bodyc, samples=rows[::25])
    json.dump(out, open(f"{OUT}/mechanism_{tag}.json", "w"), indent=1, default=float); print(json.dumps({k: v for k, v in out.items() if k != "samples"}, default=float)); return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("arg", nargs="?"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--iters", type=int, default=40); ap.add_argument("--pop", type=int, default=16); ap.add_argument("--T", type=float, default=T_DEFAULT); ap.add_argument("--wall", type=float, default=25.0)
    a = ap.parse_args()
    if a.cmd == "predict": print(json.dumps(predict(), indent=1, default=float)[:6000])
    elif a.cmd == "search": search(a.arg, a)
    elif a.cmd == "final": final(a.arg, a.T)
    elif a.cmd == "mech": mechanism(a.arg)
    elif a.cmd == "time":
        import leverage_probes as LP
        t = time.time(); R = LP._pool_map(evaluate, [(a.arg, np.random.default_rng(i).uniform(-0.3, 0.3, DIM), a.T, "both", False) for i in range(a.workers)], a.workers)
        print("s/eval (cpu, both dt)", (time.time() - t)/len(R)*a.workers, [(round(r["disp"], 3), r["reason"]) for r in R])
