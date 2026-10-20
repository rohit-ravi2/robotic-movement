"""Robot naming conventions, resolved against the MODEL (never trusted blindly). G1/G1H joints end in "_joint" and have
ankle_pitch/ankle_roll + waist_{yaw,roll,pitch}; H1 joints carry no suffix, a single "ankle" per leg and a single "torso" yaw.
Everything here is pure string logic over a list of joint/site/body names supplied by the adapter; MuJoCo is never imported."""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence

SIDES = ("left", "right")
LEG_CANON = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
ARM_CANON = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
TRUNK_CANON = ("waist_yaw", "waist_roll", "waist_pitch")
# canonical -> candidate model names, in order of preference ({s} = side)
_CANDIDATES = {
    "ankle_pitch": ("{s}_ankle_pitch_joint", "{s}_ankle"), "ankle_roll": ("{s}_ankle_roll_joint",),
    "waist_yaw": ("waist_yaw_joint", "torso"), "waist_roll": ("waist_roll_joint",), "waist_pitch": ("waist_pitch_joint",),
}
CHAIN_PATTERNS = {   # effector role -> proximal-to-distal body-name patterns; missing bodies are dropped (H1 has no wrist link)
    "palm": ["pelvis", "torso_link", "{s}_shoulder_yaw_link", "{s}_elbow_link", "{s}_wrist_yaw_link"],
    "foot": ["pelvis", "{s}_hip_yaw_link", "{s}_knee_link", "{s}_ankle_roll_link", "{s}_ankle_link"],
}


def resolve_joint(canon: str, side: Optional[str], joints: Sequence[str]) -> Optional[str]:
    """Model joint name for a canonical joint ("hip_pitch", "waist_yaw", ...) on a side (None for trunk), or None if absent."""
    js = set(joints)
    cands = list(_CANDIDATES.get(canon, ()))
    if side is not None: cands = [c.format(s=side) for c in cands] + [f"{side}_{canon}_joint", f"{side}_{canon}"]
    else: cands = cands + [f"{canon}_joint", canon]
    for c in cands:
        if c in js: return c
    return None


def leg_joints(side: str, joints: Sequence[str], canon: Sequence[str] = LEG_CANON) -> List[str]:
    return [j for j in (resolve_joint(c, side, joints) for c in canon) if j]


def arm_joints(side: str, joints: Sequence[str], canon: Sequence[str] = ARM_CANON) -> List[str]:
    return [j for j in (resolve_joint(c, side, joints) for c in canon) if j]


def trunk_joints(joints: Sequence[str], canon: Sequence[str] = TRUNK_CANON) -> List[str]:
    return [j for j in (resolve_joint(c, None, joints) for c in canon) if j]


def finger_joints(joints: Sequence[str]) -> List[str]:
    return [j for j in joints if "hand_" in j and any(k in j for k in ("thumb", "index", "middle", "ring", "pinky"))]


def mirror(name: str) -> str:
    """left <-> right in a joint/site/body name (the counter-arm/left-hand sets are mirrors of the right-hand ones)."""
    if name.startswith("left_"): return "right_" + name[5:]
    if name.startswith("right_"): return "left_" + name[6:]
    return name


def chain(kind: str, side: str, bodies: Sequence[str]) -> List[str]:
    bs = set(bodies); return [n.format(s=side) for n in CHAIN_PATTERNS[kind] if n.format(s=side) in bs]


def side_of(name: str) -> Optional[str]:
    return "left" if name.startswith("left") else ("right" if name.startswith("right") else None)


def group_of(jn: str) -> str:
    """Functional group for a joint name (mirrors throw.group_of but symmetric in side): legs | trunk | shoulder | elbow | wrist | fingers."""
    if "hand_" in jn: return "fingers"
    if "shoulder" in jn: return "shoulder"
    if "elbow" in jn: return "elbow"
    if "wrist" in jn: return "wrist"
    if "waist" in jn or jn == "torso" or jn.startswith("torso_"): return "trunk"
    if any(k in jn for k in ("hip", "knee", "ankle")): return "legs"
    return "other"


def rename_map(src: Dict[str, object], joints: Sequence[str]) -> Dict[str, object]:
    """Re-key a dict written in G1 names ("right_shoulder_pitch_joint") onto the model's names (H1: "right_shoulder_pitch"). Keys
    that do not resolve are dropped (a perturbation of a joint the body lacks is a no-op, reported by the caller)."""
    js = set(joints); out = {}
    for k, v in src.items():
        if k in js: out[k] = v; continue
        base = k[:-6] if k.endswith("_joint") else k; sd = side_of(base); canon = base[len(sd) + 1:] if sd else base
        r = resolve_joint(canon, sd, joints)
        if r: out[r] = v
    return out
