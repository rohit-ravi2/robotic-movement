"""Embodiment adapter: capabilities DETECTED from the robot model (sites, joints, actuators, collision bits, hand type), not from
a table. resolve() keeps the fixed interface (raises UnsupportedCapability with the missing items; never approximates).
chains() resolves kinematic chains for palm/foot effectors; driven_set() gives driven joint lists per role with the G1/G1H/H1
name differences handled; the perturbation helpers return the plain dicts EffectTask consumes as demand["jnt_ranges"] /
demand["tau_scale"] (first-class, pass-through, no hidden state). MuJoCo is imported lazily inside the model probe."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
from movement_discovery.core.contracts import EmbodimentSpec, Capability, TaskProgram
from . import names as N


class UnsupportedCapability(Exception):
    def __init__(self, robot, missing): super().__init__(f"{robot} lacks {[m.value for m in missing]}"); self.robot = robot; self.missing = missing


@dataclass
class ModelProbe:
    """What the model file says: names and collision facts. Built once per robot (cached), read by everything else."""
    robot: str; mjcf: str; sites: List[str]; joints: List[str]; bodies: List[str]; actuated: List[str]
    jnt_range: Dict[str, Tuple[float, float]]; has_free: bool; body_collision: bool; hand: str
    foot_bodies: Tuple[str, str] = ("", ""); hand_site: str = ""; notes: List[str] = field(default_factory=list)


_PROBES: Dict[str, ModelProbe] = {}


def mjcf_of(robot: str) -> str:
    import throw as T, branch_i as BI
    return T.CFG[robot].get("mjcf") or BI.ROBOTS[robot]["mjcf"]


def probe(robot: str) -> ModelProbe:
    if robot in _PROBES: return _PROBES[robot]
    import mujoco, throw as T, branch_i as BI
    path = mjcf_of(robot); m = mujoco.MjModel.from_xml_path(path); nm = lambda kind, i: mujoco.mj_id2name(m, kind, i) or ""
    sites = [nm(mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)]
    joints = [nm(mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    bodies = [nm(mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    actuated = sorted({joints[int(m.actuator_trnid[a][0])] for a in range(m.nu)} - {""})
    rng = {joints[j]: (float(m.jnt_range[j][0]), float(m.jnt_range[j][1])) for j in range(m.njnt) if m.jnt_limited[j] and joints[j]}
    has_free = any(int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE) for j in range(m.njnt))
    fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    body_coll = any((m.geom_contype[g] or m.geom_conaffinity[g]) and g != fl and int(m.geom_bodyid[g]) != 0 for g in range(m.ngeom))
    hand = "dex3" if N.finger_joints(joints) else "palm"
    notes = []
    if not body_coll: notes.append("body geoms carry no collision bits (floor contact via explicit pairs); impact scenes add an effector sphere")
    p = ModelProbe(robot, path, sites, joints, bodies, actuated, rng, has_free, body_coll, hand, tuple(BI.ROBOTS[robot]["foot_bodies"]), T.CFG[robot]["hand_site"], notes)
    _PROBES[robot] = p; return p


def capabilities(robot: str) -> Set[Capability]:
    p = probe(robot); C = Capability; js = set(p.joints); caps: Set[Capability] = set()
    if "right_palm" in p.sites: caps.add(C.PALM_SITE_RIGHT)
    if "left_palm" in p.sites: caps.add(C.PALM_SITE_LEFT)
    if "right_foot" in p.sites: caps.add(C.FOOT_SITE_RIGHT)
    if "left_foot" in p.sites: caps.add(C.FOOT_SITE_LEFT)
    if {"right_palm", "left_palm"} <= set(p.sites): caps.add(C.TWO_HANDS)
    if all(N.resolve_joint(f"wrist_{k}", "right", p.actuated) for k in ("roll", "pitch", "yaw")): caps.add(C.WRIST_3DOF)
    if N.resolve_joint("waist_pitch", None, js): caps.add(C.WAIST_PITCH)
    if N.resolve_joint("waist_yaw", None, js): caps.add(C.WAIST_YAW)
    if N.resolve_joint("hip_roll", "right", js): caps.add(C.HIP_ROLL)
    if N.resolve_joint("ankle_roll", "right", js): caps.add(C.ANKLE_ROLL)
    if p.hand == "dex3": caps.add(C.FINGER_GRASP)
    caps.add(C.WELD_GRASP)                                    # engine.world weld equality on any palm site
    if p.body_collision: caps.add(C.BODY_COLLISION)
    caps.add(C.EFFECTOR_COLLIDER)                             # tasks/impact.py adds a sphere at any site through MjSpec
    if p.actuated: caps.add(C.TORQUE_CONTROL); caps.add(C.SERVO_CONTROL)
    if p.has_free: caps.add(C.FLOATING_BASE)
    return caps


def chains(robot: str) -> Dict[str, List[str]]:
    """Proximal-to-distal body chains per effector role, resolved against the model: palm_r, palm_l, foot_r, foot_l."""
    p = probe(robot); out = {}
    for role, kind, side in (("palm_r", "palm", "right"), ("palm_l", "palm", "left"), ("foot_r", "foot", "right"), ("foot_l", "foot", "left")):
        c = N.chain(kind, side, p.bodies)
        if len(c) >= 2: out[role] = c
    return out


def driven_set(robot: str, role: str) -> List[str]:
    """Driven joints for a role. throw_arm_r = the robot's recorded throw set (throw.CFG); throw_arm_l its mirror; kick_leg_r/l = full
    legs + trunk + the ipsilateral arm + the counter shoulder (construct.KICK_DRIVEN, name-resolved); full_body = every actuated
    non-finger joint; arms_both = both arms + trunk; legs = legs + trunk. Only joints the model actuates are returned."""
    import throw as T
    p = probe(robot); act = p.actuated; A = set(act)
    legs = N.leg_joints("left", act) + N.leg_joints("right", act); trunk = N.trunk_joints(act)
    if role == "throw_arm_r": return [j for j in T.CFG[robot]["driven"] if j in A]
    if role == "throw_arm_l": return [N.mirror(j) for j in T.CFG[robot]["driven"] if N.mirror(j) in A]
    if role in ("kick_leg_r", "kick_leg_l"):
        s = "right" if role.endswith("_r") else "left"; o = "left" if s == "right" else "right"
        return legs + trunk + N.arm_joints(s, act, ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_pitch")) + N.arm_joints(o, act, ("shoulder_pitch", "shoulder_roll"))
    if role == "full_body": return [j for j in act if N.group_of(j) != "fingers"]
    if role == "arms_both": return trunk + N.arm_joints("right", act) + N.arm_joints("left", act)
    if role == "legs": return legs + trunk
    raise KeyError(f"unknown driven role {role!r}; roles: throw_arm_r, throw_arm_l, kick_leg_r, kick_leg_l, full_body, arms_both, legs")


def effectors(robot: str) -> Dict[str, str]:
    p = probe(robot); e = {}
    for role, site in (("hand_r", "right_palm"), ("hand_l", "left_palm"), ("foot_r", "right_foot"), ("foot_l", "left_foot")):
        if site in p.sites: e[role] = site
    return e


def resolve(robot: str, program: TaskProgram, driven: Optional[Sequence[str]] = None) -> EmbodimentSpec:
    caps = capabilities(robot); missing = [c for c in program.requires if c not in caps]
    if missing: raise UnsupportedCapability(robot, missing)
    p = probe(robot); drv = list(driven) if driven else driven_set(robot, "throw_arm_r")
    bad = [j for j in drv if j not in set(p.actuated)]
    if bad: raise UnsupportedCapability(robot, [Capability.TORQUE_CONTROL]) if not p.actuated else KeyError(f"{robot}: driven joints not actuated in the model: {bad}")
    feet = ("left_foot", "right_foot") if {"left_foot", "right_foot"} <= set(p.sites) else p.foot_bodies
    return EmbodimentSpec(robot=robot, mjcf=p.mjcf, driven=drv, effectors=effectors(robot), feet=tuple(feet), capabilities=sorted(caps, key=lambda c: c.value), notes="; ".join(p.notes + [f"hand={p.hand}"]))


# ---------------------------------------------------------------- morphology / capacity perturbations (plain dicts, pass-through)
def joint_ranges(robot: str, spec: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    """Absolute joint-range overrides, keys in any robot's naming (re-keyed onto this model); unknown joints dropped."""
    p = probe(robot); out = N.rename_map(spec, p.joints)
    return {k: (float(lo), float(hi)) for k, (lo, hi) in out.items()}


def scaled_ranges(robot: str, joints: Sequence[str], scale: float, about: Optional[Dict[str, float]] = None) -> Dict[str, Tuple[float, float]]:
    """Ranges shrunk/expanded by `scale` about a centre (default: range midpoint; `about` = {joint: angle} e.g. the stance)."""
    p = probe(robot); out = {}
    for j in N.rename_map({j: 1 for j in joints}, p.joints):
        lo, hi = p.jnt_range[j]; c = float((about or {}).get(j, 0.5*(lo + hi)))
        out[j] = (c + scale*(lo - c), c + scale*(hi - c))
    return out


HUMAN_SHOULDER = {"right_shoulder_pitch_joint": (-3.09, 1.05), "right_shoulder_yaw_joint": (-1.57, 1.57), "right_elbow_joint": (-0.17, 2.09)}   # extension <= 60, rotation <= 90, hyperextension <= 10 deg (construct throw_humanROM)


def human_rom(robot: str, side: str = "right") -> Dict[str, Tuple[float, float]]:
    src = HUMAN_SHOULDER if side == "right" else {N.mirror(k): v for k, v in HUMAN_SHOULDER.items()}
    return joint_ranges(robot, src)


def torque_scale(**groups: float) -> Dict[str, float]:
    """demand["tau_scale"]: {group or joint name: factor}; groups are throw.group_of's: legs, trunk, shoulder, elbow, wrist, counter_arm."""
    return {k: float(v) for k, v in groups.items()}


def mirror_driven(driven: Sequence[str]) -> List[str]:
    return [N.mirror(j) for j in driven]
