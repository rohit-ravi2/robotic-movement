"""World factories. Fixed interface (family modules code against this):
    build(spec: WorldSpec, robot) -> str        MJCF path next to the robot assets, cached as scene_md_<robot>_<spec.hash()>_<base6>.xml
    terrain.*  objects.*  mechanisms.*  obstacles.*  disturbances.*   -> plain dicts (serialisable; WorldSpec fields)
    start_z_offset(spec)                       terrain top under the start xy; re-add to pelvis z after the plant's settle
    apply_initial_velocities(m, d, spec)       objects' qvel0 (rolling/launched balls, pendulum rates) after reset
    apply_mocap(m, d, spec, t)                 kinematic supports/obstacles each step (physics.disturbance may delegate)
    describe(spec) -> str ; required_capabilities(spec) -> [tuple(Capability alternatives)]
    colliders.scene(spec, robot, sites) / effector_spheres(path, sites) / enable_effector_geoms(S)   the ONE effector-sphere helper
MuJoCo is imported only inside build/apply functions. Routing: obstacles go in WorldSpec.mechanisms (by kind); several terrain
features via terrain.compose(...); slip_patch disturbances become static friction patches."""
from __future__ import annotations
from movement_discovery.core.contracts import WorldSpec
from . import terrain, objects, mechanisms, obstacles, disturbances, colliders
from .build import build, describe, required_capabilities, start_z_offset, kinematic_pieces
from .objects import apply_initial_velocities
from .disturbances import apply_mocap

__all__ = ["WorldSpec", "build", "describe", "required_capabilities", "start_z_offset", "kinematic_pieces", "apply_initial_velocities", "apply_mocap", "terrain", "objects", "mechanisms", "obstacles", "disturbances", "colliders"]
