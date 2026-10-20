"""RISE: the support-transition prior (construction/rise.py, construction/priors/rise.py). MuJoCo-free: registration, strategy-space
appends (first entry kept), the geom -> group / limb map, goal parsing. mujoco: the generalised certificate on the seated start (pelvis on
the seat + feet) and the scaled-LP regression on the canonical stance. execution: the certified sit-to-stand path reaches the goal; the
prior-only controlled lowering is achieved at 1 ms."""
from __future__ import annotations
import types
import pytest

NAME = "torque_knots+rise_prior"
RISE_PROGRAMS = ["P25_sit_to_stand", "P26_stand_to_sit", "P27_kneel_to_stand", "P28_prone_to_stand", "P29_supine_to_stand", "P31_controlled_lowering"]


def test_registered():
    from movement_discovery import construction as C
    cls = C.get(NAME); assert cls.name == NAME
    assert NAME in C.describe_all()


def test_strategy_space_appended(registry):
    for pid in RISE_PROGRAMS:
        p = registry._REG[pid](); ss = p.strategy_space(p.sample_scenario(0))
        assert ss[0] == "torque_knots" and ss[-1] == NAME, (pid, ss)
    p30 = registry._REG["P30_ground_roll"](); assert NAME not in p30.strategy_space(p30.sample_scenario(0))     # rolls: not a static support transition


def test_groups_and_limbs():
    from movement_discovery.construction.priors.rise import _part, limb_of
    assert _part("left_foot2_collision") == "left_foot" and _part("right_shin_collision") == "right_shin"
    assert _part("pelvis_collision") == "pelvis" and _part("left_hip_collision") == "pelvis" and _part("head_collision") == "torso"
    assert _part("right_wrist_collision") == "right_hand" and _part("left_elbow_yaw_collision") == "left_arm"
    assert limb_of("left_foot") == ["left_thigh", "left_shin", "left_foot"] and limb_of("right_hand") == ["right_arm", "right_hand"] and limb_of("pelvis") == ["pelvis"]


def test_goal_parsing_without_model():
    from movement_discovery.construction.priors.rise import goal_from_spec
    g = types.SimpleNamespace(kind="com_displacement", params=dict(axis="z", pelvis_target=0.45))
    assert goal_from_spec(None, g) == dict(kind="lower", z=0.45)
    g = types.SimpleNamespace(kind="pose", params=dict(final="seated", seat_h=0.3, seat_x=-0.2))
    assert goal_from_spec(None, g)["kind"] == "seated"
    g = types.SimpleNamespace(kind="pose", params=dict(final="lying", droll=3.14))
    assert goal_from_spec(None, g)["kind"] == "unsupported"


def _plant(pid):
    from movement_discovery import runner
    from engine import core
    asm = runner.assemble(pid, 0, "G1", NAME); rep = asm["rep"]; dm = asm["demand"]
    return rep, dm, rep.plant(dm, core.DT_OPT)


@pytest.mark.mujoco
@pytest.mark.slow
def test_certificate_seated_and_stance(mujoco):
    from movement_discovery.construction.priors import rise as PR
    rep, dm, S = _plant("P25_sit_to_stand")
    gr = PR.contact_groups(S); su = PR.surfaces(S)
    assert {"left_foot", "right_foot", "pelvis"} <= set(gr) and len(su) == 2              # floor + the seat's top face
    c = PR.certificate(S, S["q_settled"], gr, su, ["left_foot", "right_foot", "pelvis"], margin=0.0)
    assert c["status"] == "STATIC_FEASIBLE" and c["load"]["pelvis"] > 50.0                # the seat carries the pelvis
    rep, dm, S = _plant("P31_controlled_lowering")
    c = PR.certificate(S, S["q_settled"], PR.contact_groups(S), PR.surfaces(S), ["left_foot", "right_foot"])
    assert c["ok"] and c["status"] == "STATIC_FEASIBLE"                                   # unscaled, HiGHS returned status 4 here


@pytest.mark.execution
def test_sit_to_stand_path_certified(mujoco):
    from movement_discovery.construction.priors import rise as PR
    rep, dm, S = _plant("P25_sit_to_stand")
    P = PR.support_path(S, PR.goal_from_spec(S, dm["goal"]))
    assert P["reached"], P["reason"]
    assert any(e[1] == "release" and e[2] == ("pelvis",) for e in P["events"])            # the seat is released only once the feet certify alone
    assert P["Q"][-1][2] - P["Q"][0][2] > 0.25 and float(max(P["s"][1:])) <= 1.0


@pytest.mark.execution
def test_lowering_prior_only_achieved(mujoco):
    from movement_discovery import runner
    asm = runner.assemble("P31_controlled_lowering", 0, "G1", NAME); r = runner.simulate(asm, record=True)
    o = asm["program"].evaluate(r["rec"], asm["scenario"])
    assert o.achieved, (o.validity.reasons, o.measures)
