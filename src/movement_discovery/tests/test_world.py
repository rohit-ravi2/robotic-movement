"""World factory compile contracts: declarative dicts (MuJoCo-free) and MjSpec compilation (marker mujoco)."""
import os
import pytest
from movement_discovery.core.contracts import WorldSpec, Capability
from movement_discovery import world as W


def test_factories_are_plain_dicts():
    t = W.terrain.slope(0.2); assert t["kind"] == "slope"
    o = W.objects.ball(); assert o["kind"] == "ball" and o["graspable"]
    m = W.mechanisms.door(); assert m["kind"] == "door"
    d = W.disturbances.impulse(0.5, [10, 0, 0]); assert d["kind"] == "impulse" and d["t0"] == 0.5 and d["body"] == "torso_link"
    spec = WorldSpec(terrain=t, objects=[o], mechanisms=[m], disturbances=[d]); assert len(spec.hash()) == 12 and "terrain=slope" in W.describe(spec)


def test_required_capabilities_alternatives():
    spec = WorldSpec(objects=[W.objects.heavy_block()]); req = W.required_capabilities(spec); assert (Capability.EFFECTOR_COLLIDER, Capability.BODY_COLLISION) in req
    spec = WorldSpec(objects=[W.objects.box()]); assert (Capability.FINGER_GRASP, Capability.WELD_GRASP) in W.required_capabilities(spec)
    assert W.required_capabilities(WorldSpec(terrain=W.terrain.steps(3, 0.1, 0.25))) == []


def test_start_z_offset_platform():
    spec = WorldSpec(terrain=W.terrain.platform(size=(0.5, 0.5), height=0.12)); assert W.start_z_offset(spec) == pytest.approx(0.12, abs=1e-6)
    assert W.start_z_offset(WorldSpec()) == 0.0 and W.start_z_offset(WorldSpec(terrain=W.terrain.flat())) == 0.0


def test_unknown_kinds_refused():
    from movement_discovery.world.build import _pieces
    with pytest.raises(NotImplementedError): _pieces(WorldSpec(mechanisms=[dict(kind="teleporter", name="t")]))
    with pytest.raises(NotImplementedError): _pieces(WorldSpec(disturbances=[dict(kind="earthquake", t0=0.0)]))


@pytest.mark.mujoco
@pytest.mark.parametrize("robot", ["G1", "G1H", "H1"])
def test_build_compiles(robot, mujoco):
    spec = WorldSpec(terrain=W.terrain.steps(3, 0.1, 0.25), objects=[W.objects.ball(pos=(0.6, 0.0, 0.5))], mechanisms=[W.mechanisms.door(pos=(1.5, 0.0))], disturbances=[W.disturbances.impulse(0.5, [10, 0, 0])])
    path = W.build(spec, robot); assert path and os.path.exists(path) and spec.hash() in path
    m = mujoco.MjModel.from_xml_path(path); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    assert m.nkey >= 1 and m.key_qpos.shape[1] == m.nq        # keyframes padded for the added joints
    assert W.build(spec, robot) == path                        # cached


@pytest.mark.mujoco
def test_empty_spec_is_bare_scene(mujoco):
    assert W.build(WorldSpec(), "G1") == "" and W.build(WorldSpec(tag="bare"), "G1") == ""
