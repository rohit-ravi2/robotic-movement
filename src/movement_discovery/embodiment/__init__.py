"""Embodiment adapters. Fixed interface (kept):
    resolve(robot: str, program: TaskProgram, driven: list[str] | None = None) -> EmbodimentSpec   # raises UnsupportedCapability(missing=[...])
    capabilities(robot: str) -> set[Capability]
Capabilities are DETECTED from the model file (adapter.probe): sites, actuated joints, collision bits, hand type. Extras:
chains(robot), driven_set(robot, role), effectors(robot), joint_ranges / scaled_ranges / human_rom / torque_scale / mirror_driven.
capacity.apply(demand, scenario.params, robot) forwards a scenario's tau_scale / jnt_ranges into a demand (runner.assemble, every Plan step)."""
from .adapter import (UnsupportedCapability, ModelProbe, probe, capabilities, resolve, chains, driven_set, effectors,   # noqa
                      joint_ranges, scaled_ranges, human_rom, torque_scale, mirror_driven, mjcf_of)
from . import capacity   # noqa
