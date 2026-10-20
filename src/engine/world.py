"""J13/J14: world state, objects, effect predicates, and the ABSTRACT GRASP.
Scenes are built on MuJoCo's spec API from the robot's own scene file: objects (free-joint rigid bodies), static surfaces,
and one INACTIVE weld per graspable object between the hand body and the object. Grasp = activate the weld at the current
relative pose (labelled abstraction: no finger control); release = deactivate. Keyframes are padded for the object joints.
Robot geoms have contype 0 (MJX model: contacts only via explicit pairs), so objects collide with the floor/surfaces but
not with the robot; the hand does not push objects, it only carries them through the weld."""
import sys, os, hashlib, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
MEN = os.environ.get("MENAGERIE_DIR", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "models", "mujoco_menagerie"))
SCENE_DIRS = {"G1": f"{MEN}/unitree_g1", "G1H": f"{MEN}/unitree_g1", "H1": f"{MEN}/unitree_h1"}; SCENE_DIR = SCENE_DIRS["G1"]
BASE = {"G1": f"{SCENE_DIRS['G1']}/scene_mjx.xml", "G1H": f"{SCENE_DIRS['G1']}/scene_hands_palm.xml", "H1": f"{SCENE_DIRS['H1']}/scene_palm.xml"}
HAND = {"G1": "right_wrist_yaw_link", "G1H": "right_wrist_yaw_link", "H1": "right_elbow_link"}
# Dex3 right hand (G1H): joint names, open and closed targets (rad). Closed = a thumb-opposed pinch; the servos' 1.4 N m limit sets the squeeze.
FINGER_JOINTS = ("right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint", "right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint", "right_hand_middle_1_joint")
FINGER_OPEN = {"right_hand_thumb_0_joint": 0.0, "right_hand_thumb_1_joint": -0.3, "right_hand_thumb_2_joint": 0.0, "right_hand_index_0_joint": 0.0, "right_hand_index_1_joint": 0.0, "right_hand_middle_0_joint": 0.0, "right_hand_middle_1_joint": 0.0}
FINGER_CLOSE = {"right_hand_thumb_0_joint": 0.0, "right_hand_thumb_1_joint": -0.3, "right_hand_thumb_2_joint": -1.0, "right_hand_index_0_joint": 1.2, "right_hand_index_1_joint": 1.3, "right_hand_middle_0_joint": 1.2, "right_hand_middle_1_joint": 1.3}
# closed: thumb pad at hand (0.023, 0.067, 0), fingertips at (0.035, 0.054, +-0.029): a 3 cm handle centred at hand (0.03, 0.06, 0) is squeezed between them
PINCH_CENTRE = (0.03, 0.06, 0.0)


def obj(name, kind="ball", size=(0.036,), mass=0.145, pos=(0.5, -0.2, 0.46), graspable=True, rgba=(0.9, 0.2, 0.2, 1), soft=None, quat=(1, 0, 0, 0), rigid=True):
    """soft: contact time constant (s) for a compliant object (0.1 = sponge-like); rigid=True (default since 2026-10-11): stiff contact
    (0.002 s, solimp 0.99-0.999) so objects do not sink into slabs. quat: initial orientation (w x y z)."""
    return dict(name=name, kind=kind, size=list(size), mass=float(mass), pos=list(pos), graspable=bool(graspable), rgba=list(rgba), soft=(None if soft is None else float(soft)), quat=list(quat), rigid=bool(rigid))


def brush(name="brush", pos=(0.3, -0.25, 0.655), head=(0.04, 0.025, 0.015), handle_r=0.015, handle_h=0.16, mass=0.15, rgba=(0.2, 0.6, 0.3, 1), lean=None, bristle=None, quat=(1, 0, 0, 0)):
    """A scrub brush: head box (half-extents) with a cylinder handle on top, ONE free body with its origin at the head centre.
    lean: unit vector (body frame) of the handle axis (None = vertical). An ANGLED handle (J27: lean = the Dex3 fist's settled
    groove axis, ~49 deg from the head normal) is held at equilibrium by the standard hand pose (fingers forward, palm z up) with the
    head flat. tool bottom = head half-height below the origin; grip_point = mid-handle (body frame); grip_top = handle tip."""
    o = obj(name, kind="compound", size=list(head), mass=mass, pos=pos, rgba=rgba, quat=quat)
    ax = np.array([0, 0, 1.0]) if lean is None else np.asarray(lean, float)/np.linalg.norm(lean); mid = np.array([0, 0, head[2]]) + ax*(handle_h/2)
    zq = np.zeros(4); mujoco.mju_quatZ2Vec(zq, ax)
    o["geoms"] = [dict(type="box", size=list(head), pos=[0, 0, 0], mass=0.8*mass, **({"soft": float(bristle)} if bristle else {})), dict(type="cylinder", size=[handle_r, handle_h/2], pos=mid.tolist(), quat=zq.tolist(), mass=0.2*mass)]   # head-heavy: it stands on its head; bristle = contact time constant of the HEAD only (bristles deflect a few mm under 5-10 N; the handle stays rigid)
    o["bristle"] = (float(bristle) if bristle else None)
    o["half_z"] = float(head[2]); o["grip_top"] = float(head[2] + handle_h); o["handle_axis"] = ax.tolist(); o["grip_point"] = mid.tolist(); o["grip_tip"] = (np.array([0, 0, head[2]]) + ax*handle_h).tolist(); return o


def surface(name, pos, size, rgba=(0.6, 0.4, 0.2, 1), rigid=True):
    return dict(name=name, pos=list(pos), size=list(size), rgba=list(rgba), rigid=bool(rigid))


def build_scene(robot="G1", objects=(), surfaces=(), tag=None):
    """Write <SCENE_DIR>/scene_world_<hash>.xml (next to the robot's assets) and return its path + object list."""
    base_h = hashlib.sha256(open(BASE[robot], "rb").read()).hexdigest()[:8]           # a changed base model (new keyframe, new site) must not reuse a stale scene file
    desc = json.dumps(dict(v=3, robot=robot, base=base_h, objects=list(objects), surfaces=list(surfaces)), sort_keys=True)
    h = tag or hashlib.sha256(desc.encode()).hexdigest()[:10]; path = f"{SCENE_DIRS[robot]}/scene_world_{h}.xml"
    if os.path.exists(path): return path
    spec = mujoco.MjSpec.from_file(BASE[robot]); nq0 = sum(len(k.qpos) for k in spec.keys[:1])
    for s in surfaces:
        # STATIC surfaces are massless: a default-density table would enter the world subtree COM (13 kg at the table
        # position) and corrupt every balance quantity (priors, gates, controller)
        b = spec.worldbody.add_body(name=s["name"], pos=s["pos"]); g = b.add_geom(name=s["name"] + "_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=s["size"], rgba=s["rgba"], mass=0.0)
        if s.get("rigid", True): g.solref = [0.002, 1.0]; g.solimp = [0.99, 0.999, 0.001, 0.5, 2.0]
    kinds = dict(ball=mujoco.mjtGeom.mjGEOM_SPHERE, box=mujoco.mjtGeom.mjGEOM_BOX, cylinder=mujoco.mjtGeom.mjGEOM_CYLINDER, capsule=mujoco.mjtGeom.mjGEOM_CAPSULE)
    for o in objects:
        b = spec.worldbody.add_body(name=o["name"], pos=o["pos"], quat=o.get("quat", [1, 0, 0, 0])); b.add_freejoint(name=o["name"] + "_free")
        glist = o.get("geoms") or [dict(type=o["kind"], size=o["size"], pos=[0, 0, 0])]; mass_each = o["mass"]/len(glist)
        for gi, gd in enumerate(glist):
            g = b.add_geom(name=o["name"] + ("_geom" if gi == 0 else f"_geom{gi}"), type=kinds[gd["type"]], size=list(gd["size"]) + [0]*(3 - len(gd["size"])), pos=gd.get("pos", [0, 0, 0]), quat=gd.get("quat", [1, 0, 0, 0]), mass=float(gd.get("mass", mass_each)), rgba=o["rgba"])
            if gd.get("soft"): g.solref = [float(gd["soft"]), 1.0]; g.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]; g.priority = 1   # per-geom compliance (brush bristles): its solref wins over the slab's
            elif o.get("soft"): g.solref = [float(o["soft"]), 1.0]; g.priority = 1          # compliant contact (its solref wins over the slab's)
            elif o.get("rigid", True): g.solref = [0.002, 1.0]; g.solimp = [0.99, 0.999, 0.001, 0.5, 2.0]
        if o["graspable"]:
            e = spec.add_equality(name=f"grasp_{o['name']}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY, name1=HAND[robot], name2=o["name"]); e.active = False
    # keyframes: pad with the objects' initial free-joint qpos (pos + identity quaternion)
    pad = [v for o in objects for v in (list(o["pos"]) + list(o.get("quat", [1, 0, 0, 0])))]
    for k in spec.keys:
        k.qpos = list(k.qpos) + pad
        if len(k.qvel): k.qvel = list(k.qvel) + [0.0]*(6*len(objects))
    spec.compile()                                   # validates
    open(path, "w").write(spec.to_xml())
    return path


class World:
    """Predicates and the abstract grasp on a compiled (m, d)."""
    def __init__(self, m, objects=(), surfaces=(), robot="G1"):
        self.m = m; self.objects = {o["name"]: o for o in objects}; self.surfaces = {s["name"]: s for s in surfaces}
        self.bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in list(self.objects) + list(self.surfaces)}
        self.eq = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_{n}") for n in self.objects if self.objects[n]["graspable"]}
        self.hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, HAND[robot]); self.palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
        pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis"); rb = set()
        for b in range(m.nbody):
            x = b
            while x > 0 and x != pel: x = int(m.body_parentid[x])
            if x == pel: rb.add(b)
        self.robot_bodies = rb; self.robot_mass = float(sum(m.body_mass[b] for b in rb))
        self.robot_geoms = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in rb}
        # FINGER GRASP (J20, robots with a right hand): no weld; the finger servos close on command (force-limited by their
        # actuators) and an object is HELD when it is in contact with at least two distinct hand links
        fj = [(n, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)) for n in FINGER_JOINTS]
        self.fingers = {n: [a for a in range(m.nu) if int(m.actuator_trnid[a][0]) == j][0] for n, j in fj if j >= 0 and any(int(m.actuator_trnid[a][0]) == j for a in range(m.nu))}
        self.hand_bodies = {b for b in rb if "right_hand" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")} | ({self.hand} if self.hand >= 0 else set())
        self.closing = {}; self.t_now = 0.0       # name -> time the close command started (None = open)
    def reset_fingers(self, held=()):
        """Per rollout: fingers closed from the start on objects declared held, open otherwise."""
        self.closing = {n: -1.0 for n in held}; self.t_now = 0.0
    def finger_ctrl(self, d, t, T_close=0.4):
        """Apply the finger servo targets for time t (ramp open -> closed over T_close after a close command)."""
        if not self.fingers: return
        self.t_now = t
        if self.closing:
            t0 = min(self.closing.values()); w = 1.0 if t0 < 0 else float(np.clip((t - t0)/T_close, 0.0, 1.0))
        else: w = 0.0
        for n, a in self.fingers.items(): d.ctrl[a] = (1 - w)*FINGER_OPEN[n] + w*FINGER_CLOSE[n]
    def hand_contacts(self, d, name):
        """Distinct hand links in contact with the object, and the summed normal force."""
        m = self.m; b_o = self.bid[name]; links = set(); F = 0.0
        for ci in range(d.ncon):
            c = d.contact[ci]; b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if b_o in (b1, b2):
                other = b2 if b1 == b_o else b1
                if other in self.hand_bodies: links.add(other); f = np.zeros(6); mujoco.mj_contactForce(m, d, ci, f); F += abs(float(f[0]))
        return links, F
    def pos(self, d, name): return d.xpos[self.bid[name]].copy()
    def held(self, d, name):
        if self.fingers:
            if name not in self.closing: return False
            links, F = self.hand_contacts(d, name); return len(links) >= 2 and F >= 1.0      # two distinct hand links with real force, not grazing
        return bool(d.eq_active[self.eq[name]]) if name in self.eq else False
    def palm_dist(self, d, name): return float(np.linalg.norm(d.site_xpos[self.palm] - d.xpos[self.bid[name]]))
    def on(self, d, name, surf, tol=0.03):
        o = self.objects[name]; s = self.surfaces[surf]; p = d.xpos[self.bid[name]]; r = o["size"][0] if o["kind"] == "ball" else o["size"][2]
        top = s["pos"][2] + s["size"][2]; inside = abs(p[0] - s["pos"][0]) <= s["size"][0] and abs(p[1] - s["pos"][1]) <= s["size"][1]
        return bool(inside and abs((p[2] - r) - top) <= tol)
    def grasp(self, d, name):
        """Finger robots: command the fingers to close. Others: activate the weld at the CURRENT relative pose of the object in the hand frame."""
        if self.fingers:
            if name not in self.closing: self.closing[name] = self.t_now
            return
        m = self.m; e = self.eq[name]; b1, b2 = self.hand, self.bid[name]
        R1 = d.xmat[b1].reshape(3, 3); p_rel = R1.T @ (d.xpos[b2] - d.xpos[b1])
        q1 = d.xquat[b1]; q2 = d.xquat[b2]; q1i = np.array([q1[0], -q1[1], -q1[2], -q1[3]]); q_rel = np.zeros(4); mujoco.mju_mulQuat(q_rel, q1i, q2)
        m.eq_data[e, :3] = 0.0; m.eq_data[e, 3:6] = p_rel; m.eq_data[e, 6:10] = q_rel; m.eq_data[e, 10] = 1.0
        # a STIFF weld: the default 0.02 s constraint time constant let the hand sink 5 cm relative to a pressed tool, so the
        # tool transmitted only its own weight (1 N) to the surface
        m.eq_solref[e] = [0.002, 1.0]; m.eq_solimp[e] = [0.99, 0.999, 0.001, 0.5, 2.0]
        d.eq_active[e] = 1
    def release(self, d, name):
        if self.fingers: self.closing.pop(name, None); return
        d.eq_active[self.eq[name]] = 0
    def state(self, d):
        return {n: dict(pos=self.pos(d, n).tolist(), held=self.held(d, n)) for n in self.objects}


def effect_place(world, d_before, d_after, name, target, tol=0.05, settle=0.5, surface=None):
    """Verified effect: after release the object is let SETTLE (free physics, robot frozen in place) and must rest within tol
    (xy, and z within 2*tol) of the target, released, and on `surface` if given."""
    m = world.m; d = mujoco.MjData(m); d.qpos[:] = d_after.qpos; d.qvel[:] = 0; mujoco.mj_forward(m, d)
    for e in world.eq.values(): d.eq_active[e] = 0
    # freeze the robot: all its joints as stiff servos at the current pose (the object settles under gravity alone)
    import branch_i as BI
    saved = []
    for a in range(m.nu):
        saved.append((int(m.actuator_gaintype[a]), m.actuator_gainprm[a].copy(), int(m.actuator_biastype[a]), m.actuator_biasprm[a].copy(), m.actuator_ctrlrange[a].copy(), m.actuator_forcerange[a].copy(), int(m.actuator_ctrllimited[a]), int(m.actuator_forcelimited[a])))
        jid = int(m.actuator_trnid[a][0]); dof = int(m.jnt_dofadr[jid]); ieff = 1.0/max(float(np.linalg.inv(_fullM(m, d))[dof, dof]), 1e-9); kp = ieff/m.opt.timestep**2
        BI._as_servo(m, a, kp, 2.0*np.sqrt(kp*ieff), 1e4); d.ctrl[a] = d.qpos[m.jnt_qposadr[jid]]
    try:
        for _ in range(int(settle/m.opt.timestep)): mujoco.mj_step(m, d)
    finally:
        for a, (gt, gp, bt, bp, cr, fr, cl, fl_) in enumerate(saved):
            m.actuator_gaintype[a] = gt; m.actuator_gainprm[a] = gp; m.actuator_biastype[a] = bt; m.actuator_biasprm[a] = bp; m.actuator_ctrlrange[a] = cr; m.actuator_forcerange[a] = fr; m.actuator_ctrllimited[a] = cl; m.actuator_forcelimited[a] = fl_
    p = world.pos(d, name); t = np.asarray(target, float); dxy = float(np.linalg.norm(p[:2] - t[:2])); dz = float(abs(p[2] - t[2]))
    on = world.on(d, name, surface) if surface else True
    return dict(ok=bool(dxy <= tol and dz <= 2*tol and on and not world.held(d, name)), dist=float(np.linalg.norm(p - t)), dxy=dxy, dz=dz, on_surface=bool(on), pos=p.tolist())


def _fullM(m, d):
    M = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, M, d.qM); return M
