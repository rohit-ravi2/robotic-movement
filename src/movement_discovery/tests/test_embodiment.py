"""Capability refusal and detection (model probe)."""
import pytest
from movement_discovery.core.contracts import Capability

pytestmark = pytest.mark.mujoco


def test_finger_grasp_refused_on_g1(registry, mujoco):
    from movement_discovery import embodiment as E
    p = registry.get("P75_one_handed_capture"); assert Capability.FINGER_GRASP in p.requires
    with pytest.raises(E.UnsupportedCapability) as ei: E.resolve("G1", p)
    assert Capability.FINGER_GRASP in ei.value.missing and ei.value.robot == "G1"
    spec = E.resolve("G1H", p); assert spec.robot == "G1H" and Capability.FINGER_GRASP in spec.capabilities


def test_h1_has_no_foot_sites(registry, mujoco):
    from movement_discovery import embodiment as E
    caps = E.capabilities("H1"); assert Capability.FOOT_SITE_LEFT not in caps and Capability.PALM_SITE_RIGHT in caps
    with pytest.raises(E.UnsupportedCapability): E.resolve("H1", registry.get("P17_impulse_recovery"))


def test_g1_detected_capabilities(mujoco):
    from movement_discovery import embodiment as E
    caps = E.capabilities("G1")
    assert {Capability.PALM_SITE_RIGHT, Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE, Capability.WELD_GRASP, Capability.EFFECTOR_COLLIDER} <= caps
    assert Capability.BODY_COLLISION not in caps and Capability.FINGER_GRASP not in caps      # MJX G1 body carries no collision bits; palm hands


def test_resolve_returns_spec_with_hash(registry, mujoco):
    from movement_discovery import embodiment as E
    s = E.resolve("G1", registry.get("P41_max_release_speed")); assert s.driven and s.effectors["hand_r"] == "right_palm" and len(s.hash()) == 12
    assert set(E.driven_set("G1", "throw_arm_r")) <= set(E.probe("G1").actuated)


def test_perturbation_helpers_are_plain_dicts(mujoco):
    from movement_discovery import embodiment as E
    ts = E.torque_scale(shoulder=0.5, legs=0.5); assert ts == {"shoulder": 0.5, "legs": 0.5}
    hr = E.human_rom("G1"); assert "right_shoulder_pitch_joint" in hr and hr["right_shoulder_pitch_joint"][1] == pytest.approx(1.05)
    sr = E.scaled_ranges("G1", ["right_elbow_joint"], 0.5); lo, hi = sr["right_elbow_joint"]; flo, fhi = E.probe("G1").jnt_range["right_elbow_joint"]; assert (hi - lo) == pytest.approx(0.5*(fhi - flo))
