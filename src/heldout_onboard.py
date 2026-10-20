"""HELD-OUT EMBODIMENT onboarding (2026-10-16): Apptronik Apollo and Booster T1 registered into the throw/branch_i plant.

REGISTRATION ONLY (no movement code): derived MJCF copies under results/heldout/models/ (the menagerie is never edited) and
spec yamls under spec/ (the format of spec/h1.yaml). What the copies change, every item listed here and in REG below:
  NAMES   joints and bodies renamed to the project's G1 naming convention (left_/right_ prefix; hip_pitch/hip_roll/hip_yaw/knee/
          ankle_pitch/ankle_roll/shoulder_pitch/shoulder_roll/elbow/waist_*), the floating-base body to "pelvis". The movement code
          finds joints by these names (mirror signs from "roll"/"yaw", leg columns from "hip"/"knee"/"ankle"), so the map is the
          whole of the registration. The name maps are in NAMEMAP.
  PALM    a "right_palm" site (throw.setup requires an effector site; unused by the sprint).
  APOLLO  (a) joint `damping` removed: the menagerie CHANGELOG 2025-05-02 "moved kv to damping" -> it is the position servo's
          D-gain, which the torque-motor conversion removes (G1's kv lives in biasprm and branch_i.to_torque_motors drops it the
          same way); frictionloss and armature (passive) kept. (b) the sole/floor pairs set to the project's contact protocol
          (condim 3, mu 1, as G1/H1; menagerie: condim 6, mu 2). (c) floor pairs ADDED for every collision capsule (pelvis, torso,
          arms, hands, thighs, shins): the menagerie model only lets the soles touch the floor, so the no-non-foot-contact gate
          could not see a knee or hand on the ground. (d) a "stand_level" keyframe: the menagerie "stand" with each ankle's pitch and
          roll re-solved so the soles are flat (the original's soles are tilted 2.2/1.2 deg; its held statue tips over on the edge).
  T1      unchanged apart from names/site (contype collision, condim 3, mu 1 already; frictionloss 0.1 / armature 0.005 kept).
RATINGS  tau = MJCF actuatorfrcrange (both). omega: T1 = Booster's T1_serial.urdf <limit velocity> (the URDF the menagerie model
          is derived from); Apollo: NONE PUBLISHED (no URDF, no actuator data; searched 2026-10-16) -> ASSUMPTION, one rule:
          the rated power per body kg (tau x omega / M) of the same joint type, mean of G1 and H1 (their URDF ratings agree
          within 6% for hip pitch 84.5/89.4 and knee 83.4/81.7 W/kg): omega = k_type M / tau. Flagged in the spec (tag ASSUMED).
usage: python heldout_onboard.py   (writes the copies + spec/apollo.yaml, spec/t1.yaml, results/heldout/onboard_sources.json)"""
import os, sys, json, re, copy
import xml.etree.ElementTree as ET
import numpy as np, yaml, mujoco

MEN = os.environ.get("MENAGERIE_DIR", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "models", "mujoco_menagerie"))
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = f"{ROOT}/results/heldout/models"

NAMEMAP = {
    "Apollo": dict(
        joints={"torso_yaw": "waist_yaw_joint", "torso_roll": "waist_roll_joint", "torso_pitch": "waist_pitch_joint",
                **{f"{s}_{a}": f"{S}_{b}" for s, S in (("l", "left"), ("r", "right")) for a, b in (
                    ("shoulder_aa", "shoulder_roll_joint"), ("shoulder_ie", "shoulder_yaw_joint"), ("shoulder_fe", "shoulder_pitch_joint"),
                    ("elbow_fe", "elbow_joint"), ("wrist_roll", "wrist_roll_joint"), ("wrist_yaw", "wrist_yaw_joint"), ("wrist_pitch", "wrist_pitch_joint"),
                    ("hip_ie", "hip_yaw_joint"), ("hip_aa", "hip_roll_joint"), ("hip_fe", "hip_pitch_joint"), ("knee_fe", "knee_joint"),
                    ("ankle_ie", "ankle_roll_joint"), ("ankle_pd", "ankle_pitch_joint"))}},
        bodies=lambda n: "pelvis" if n == "base_link" else re.sub(r"^l_", "left_", re.sub(r"^r_", "right_", n)),
        palm=("right_wrist_pitch_link", "0 0 -0.12")),
    "T1": dict(
        joints={"AAHead_yaw": "head_yaw_joint", "Head_pitch": "head_pitch_joint", "Waist": "waist_yaw_joint",
                **{f"{S}_{a}": f"{s}_{b}" for S, s in (("Left", "left"), ("Right", "right")) for a, b in (
                    ("Shoulder_Pitch", "shoulder_pitch_joint"), ("Shoulder_Roll", "shoulder_roll_joint"), ("Elbow_Pitch", "elbow_joint"),
                    ("Elbow_Yaw", "forearm_yaw_joint"), ("Hip_Pitch", "hip_pitch_joint"), ("Hip_Roll", "hip_roll_joint"), ("Hip_Yaw", "hip_yaw_joint"),
                    ("Knee_Pitch", "knee_joint"), ("Ankle_Pitch", "ankle_pitch_joint"), ("Ankle_Roll", "ankle_roll_joint"))}},
        bodies=lambda n: {"Trunk": "pelvis", "Waist": "waist_link", "H1": "head_yaw_link", "H2": "head_pitch_link",
                          "AL1": "left_shoulder_pitch_link", "AL2": "left_shoulder_roll_link", "AL3": "left_elbow_link", "left_hand_link": "left_hand_link",
                          "AR1": "right_shoulder_pitch_link", "AR2": "right_shoulder_roll_link", "AR3": "right_elbow_link", "right_hand_link": "right_hand_link",
                          **{f"{a}_{S}": f"{s}_{b}" for S, s in (("Left", "left"), ("Right", "right")) for a, b in (
                              ("Hip_Pitch", "hip_pitch_link"), ("Hip_Roll", "hip_roll_link"), ("Hip_Yaw", "hip_yaw_link"), ("Shank", "knee_link"),
                              ("Ankle_Cross", "ankle_pitch_link"))},
                          "left_foot_link": "left_ankle_roll_link", "right_foot_link": "right_ankle_roll_link"}.get(n, n),
        palm=("right_hand_link", "0 -0.2 0")),
}
SRC = {"Apollo": dict(dir="apptronik_apollo", robot="apptronik_apollo.xml", scene="scene.xml"),
       "T1": dict(dir="booster_t1", robot="t1.xml", scene="scene.xml")}
T1_URDF = "https://raw.githubusercontent.com/BoosterRobotics/booster_gym/main/resources/T1/T1_serial.urdf"
T1_URDF_SHA256 = "027a5333ce4ed0a1cf6fcbbb12e59709d34b31c926fb4bd682ea5e4cbbc36071"
T1_URDF_VEL = {"AAHead_yaw": 12.56, "Head_pitch": 12.56, "Left_Shoulder_Pitch": 18.84, "Left_Shoulder_Roll": 18.84, "Left_Elbow_Pitch": 18.84,
               "Left_Elbow_Yaw": 18.84, "Right_Shoulder_Pitch": 18.84, "Right_Shoulder_Roll": 18.84, "Right_Elbow_Pitch": 18.84, "Right_Elbow_Yaw": 18.84,
               "Waist": 10.88, "Left_Hip_Pitch": 12.5, "Left_Hip_Roll": 10.9, "Left_Hip_Yaw": 10.9, "Left_Knee_Pitch": 11.7, "Left_Ankle_Pitch": 18.8,
               "Left_Ankle_Roll": 12.4, "Right_Hip_Pitch": 12.5, "Right_Hip_Roll": 10.9, "Right_Hip_Yaw": 10.9, "Right_Knee_Pitch": 11.7,
               "Right_Ankle_Pitch": 18.8, "Right_Ankle_Roll": 12.4}     # copied from the URDF (sha256 above), downloaded 2026-10-16


def jtype(n):
    """joint type key shared across robots (for the Apollo power-per-kg rule)."""
    for k in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll", "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw", "waist_yaw", "waist_roll", "waist_pitch"):
        if k in n: return k
    return None


def power_per_kg():
    """k_type = tau x omega / M per joint type, mean over G1 and H1 (their manufacturer-URDF ratings)."""
    H1T = {"left_hip_pitch": "hip_pitch", "left_hip_roll": "hip_roll", "left_hip_yaw": "hip_yaw", "left_knee": "knee", "left_ankle": "ankle_pitch",
           "left_shoulder_pitch": "shoulder_pitch", "left_shoulder_roll": "shoulder_roll", "left_shoulder_yaw": "shoulder_yaw", "left_elbow": "elbow", "torso": "waist_yaw"}
    acc = {}
    for f, tmap in (("g1.yaml", None), ("h1.yaml", H1T)):
        s = yaml.safe_load(open(f"{ROOT}/spec/{f}")); M = float(s["meta"]["total_mass"])
        for j, r in s["joints"].items():
            if j.startswith("right"): continue
            t = tmap.get(j) if tmap else jtype(j)
            if t is None: continue
            acc.setdefault(t, {})[f[:2].upper()] = float(r["tau_max_peak"]["value"])*float(r["omega_max"]["value"])/M
    return {t: dict(v, mean=float(np.mean(list(v.values())))) for t, v in acc.items()}


def build(robot):
    cfg = NAMEMAP[robot]; src = SRC[robot]; d_in = f"{MEN}/{src['dir']}"; os.makedirs(OUT, exist_ok=True)
    tree = ET.parse(f"{d_in}/{src['robot']}"); r = tree.getroot()
    comp = r.find("compiler"); comp.set("meshdir", f"{d_in}/assets")
    if comp.get("texturedir"): comp.set("texturedir", f"{d_in}/assets")
    jm = cfg["joints"]; changes = []
    for el in r.iter():
        if el.tag == "joint" and el.get("name") in jm: el.set("name", jm[el.get("name")])
        if el.tag == "body" and el.get("name"): el.set("name", cfg["bodies"](el.get("name")))
        if el.get("joint") in jm: el.set("joint", jm[el.get("joint")])
        if el.tag == "site" and el.get("name", "").startswith(("l_", "r_")): el.set("name", cfg["bodies"](el.get("name")))
        if robot == "Apollo" and el.tag == "joint" and el.get("damping") is not None:
            changes.append((el.get("name"), "damping", el.get("damping"))); del el.attrib["damping"]
    pb, pp = cfg["palm"]
    for b in r.iter("body"):
        if b.get("name") == pb: ET.SubElement(b, "site", dict(name="right_palm", pos=pp, size="0.01"))
    base = f"{robot.lower()}_heldout"
    tree.write(f"{OUT}/{base}.xml")
    if robot == "Apollo":       # LEVEL-SOLE STANDING KEYFRAME: the menagerie "stand" key has both soles tilted (2.2 deg pitch, 1.2 deg roll) so they
        changes.append(level_keyframe(tree, f"{OUT}/{base}.xml", "stand", "stand_level"))   # touch on an edge and the held statue tips (the stock model falls from it too)
    # scene: the menagerie scene with the include pointed at the copy; Apollo's contact block set to the protocol + body pairs
    st = ET.parse(f"{d_in}/{src['scene']}"); sr = st.getroot(); sr.find("include").set("file", f"{base}.xml")
    if robot == "Apollo":
        con = sr.find("contact")
        for p in con.findall("pair"):
            if "floor" in (p.get("geom1"), p.get("geom2")):
                p.set("condim", "3"); p.set("friction", "1 1 0.005 0.0001 0.0001")
        m0 = mujoco.MjModel.from_xml_path(f"{d_in}/{src['scene']}")
        for g in range(m0.ngeom):
            n = mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_GEOM, g)
            if n and n.startswith("collision_") and "sole" not in n:
                ET.SubElement(con, "pair", dict(geom1=n, geom2="floor", condim="3", friction="1 1 0.005 0.0001 0.0001"))
    st.write(f"{OUT}/scene_{base}.xml")
    return f"{OUT}/scene_{base}.xml", changes


def level_keyframe(tree, path, src_key, new_key):
    """Copy of keyframe src_key with each ankle's pitch and roll solved (Newton, finite differences) so that the sole geom's normal is
    vertical; everything else unchanged. Appended to the copy as new_key."""
    m = mujoco.MjModel.from_xml_path(path); d = mujoco.MjData(m); k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, src_key); q = m.key_qpos[k].copy()
    qa = lambda n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]); res = {}
    for side, sole in (("left", "collision_l_sole"), ("right", "collision_r_sole")):
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, sole); ids = [qa(f"{side}_ankle_pitch_joint"), qa(f"{side}_ankle_roll_joint")]
        def err(qq): d.qpos[:] = qq; mujoco.mj_kinematics(m, d); return d.geom_xmat[g].reshape(3, 3)[:2, 2].copy()
        e0 = err(q).copy()
        for _ in range(20):
            e = err(q)
            if np.linalg.norm(e) < 1e-7: break
            Jm = np.zeros((2, 2))
            for i, a in enumerate(ids):
                qp = q.copy(); qp[a] += 1e-6; Jm[:, i] = (err(qp) - e)/1e-6
            dq = -np.linalg.solve(Jm, e)
            for i, a in enumerate(ids): q[a] += dq[i]
        res[side] = dict(tilt_before=e0.tolist(), tilt_after=err(q).tolist(), ankle=[float(q[a]) for a in ids])
    r = tree.getroot(); kf = r.find("keyframe"); ET.SubElement(kf, "key", dict(name=new_key, qpos=" ".join(f"{x:.6g}" for x in q))); tree.write(path)
    return ("keyframe", new_key, res)


def spec(robot, scene):
    m = mujoco.MjModel.from_xml_path(scene); inv = {v: k for k, v in NAMEMAP[robot]["joints"].items()}
    M = float(sum(m.body_mass)); kpk = power_per_kg() if robot == "Apollo" else None
    groups = {}; joints = {}
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE: continue
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j); t = jtype(n) or ("head" if "head" in n or "neck" in n else "forearm" if "forearm" in n else "other")
        g = ("hips" if t.startswith("hip") else "knees" if t == "knee" else "ankles" if t.startswith("ankle") else "waist" if t.startswith("waist")
             else "shoulders" if t.startswith("shoulder") else "elbows" if t in ("elbow", "forearm") else "wrists" if t.startswith("wrist") else "neck")
        groups.setdefault(g, []).append(n)
        tau = float(m.jnt_actfrcrange[j][1]); side = "left" if n.startswith("left") else "right" if n.startswith("right") else "center"
        if robot == "T1":
            om = dict(value=T1_URDF_VEL[inv[n]], tag="CITED", src="booster_t1_urdf")
        else:
            k = kpk.get(t, kpk.get({"wrist_roll": "elbow", "wrist_pitch": "elbow", "wrist_yaw": "elbow", "waist_roll": "waist_yaw", "waist_pitch": "waist_yaw"}.get(t, ""), None)) if t else None
            if k is None: k = kpk["elbow"]         # neck: the lightest available class (held, never driven)
            om = dict(value=round(k["mean"]*M/tau, 2), tag="ASSUMED", src="rule_power_per_kg",
                      note=f"no published Apollo velocity limit; omega = k M / tau, k = {k['mean']:.1f} W/kg = mean G1/H1 tau*omega/M for type {t} ({ {a: round(b, 1) for a, b in k.items() if a != 'mean'} })")
        joints[n] = dict(range=dict(value=[float(x) for x in m.jnt_range[j]], tag="CITED", src=f"menagerie_{robot.lower()}"),
                         armature=dict(value=float(m.dof_armature[m.jnt_dofadr[j]]), tag="CITED", src=f"menagerie_{robot.lower()}"),
                         axis=dict(value=[float(x) for x in m.jnt_axis[j]], tag="CITED", src=f"menagerie_{robot.lower()}"),
                         group=g, side=side, source_name=inv.get(n, n),
                         tau_max=dict(value=tau, tag="CITED", src=f"menagerie_{robot.lower()}", note="MJCF actuatorfrcrange"),
                         omega_max=om,
                         tau_max_continuous=dict(value=tau, tag="CITED", src=("booster_t1_urdf" if robot == "T1" else f"menagerie_{robot.lower()}"),
                                                 note=("URDF effort (equal to the MJCF)" if robot == "T1" else "no separate continuous rating published")),
                         tau_max_peak=dict(value=tau, tag="CITED", src=f"menagerie_{robot.lower()}", note="MJCF actuatorfrcrange"))
    sources = {f"menagerie_{robot.lower()}": f"MuJoCo Menagerie {SRC[robot]['dir']} MJCF (derived copy {os.path.relpath(scene, ROOT)}; names mapped, see src/heldout_onboard.py)"}
    if robot == "T1": sources["booster_t1_urdf"] = f"Booster Robotics booster_gym T1_serial.urdf ({T1_URDF}, sha256 {T1_URDF_SHA256}): effort and velocity limits"
    else: sources["rule_power_per_kg"] = "ASSUMPTION (no Apollo velocity limits published, 2026-10-16 search): rated power per body kg of the joint type = mean of G1 and H1 URDF ratings"
    meta = dict(name=robot, mjcf=os.path.relpath(scene, ROOT), total_mass=M, n_actuated=int(m.nu), timestep=float(m.opt.timestep),
                units=dict(angle="radian", torque="N*m", ang_vel="rad/s", mass="kg"), capacity_sets=dict(primary="peak", rationale="single published torque set (MJCF actuatorfrcrange)"),
                heldout=True)
    out = dict(meta=meta, sources=sources, joints=joints, joint_groups=groups)
    path = f"{ROOT}/spec/{robot.lower()}.yaml"; yaml.safe_dump(out, open(path, "w"), sort_keys=False, width=140); return path, out


if __name__ == "__main__":
    rec = {}
    for robot in ("Apollo", "T1"):
        scene, ch = build(robot); p, sp = spec(robot, scene)
        rec[robot] = dict(scene=scene, spec=p, removed=ch, name_map=NAMEMAP[robot]["joints"],
                          ratings={n: dict(tau=r["tau_max_peak"]["value"], omega=r["omega_max"]["value"], omega_src=r["omega_max"]["src"], omega_tag=r["omega_max"]["tag"], source_name=r["source_name"]) for n, r in sp["joints"].items()})
        print(robot, scene, p, "mass", round(sp["meta"]["total_mass"], 2))
    rec["Apollo"]["power_per_kg_rule"] = power_per_kg()
    os.makedirs(f"{ROOT}/results/heldout", exist_ok=True); json.dump(rec, open(f"{ROOT}/results/heldout/onboard_sources.json", "w"), indent=1)
