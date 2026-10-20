"""Program 75 (interception): ONE-HANDED capture with real fingers (G1H, Capability.FINGER_GRASP): a slow ball lobbed from 1 m must end held by the Dex3 hand (>= 2 distinct hand links in contact, engine.world.World.held rule). The fingers close when the ball comes within close_dist of the palm (an event on world state).
integration check: 2026-10-12 — prior-only rollout (results/md_build/checks_interception.json), plant built, value -4.93, achieved False, gates fired [NOT_CAPTURED]; rel_speed_end=4.14 t_contact=0 dmin_obj=0.393 captured=False (42 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from movement_discovery.tasks.force_interaction import _base as B
from . import _catch as C


@register
class OneHandedCapture(C.CatchProgram, TaskProgram):
    id = "P75_one_handed_capture"; family = Family.INTERCEPTION; title = "Capture a lobbed ball in one hand"
    requires = [Capability.PALM_SITE_RIGHT, Capability.FINGER_GRASP, Capability.WRIST_3DOF, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED
    distinct_from = "P73's capture is a contact-and-relative-speed condition on a sphere collider; here the hold must be a finger enclosure (two links), which bounds the ball size and demands the palm face the arrival direction."
    def obj(self, p): return O.launched_ball(pos=[p["dist"], -0.22, 0.9], vel=[-p["v"], 0.0, p["vz"]], name="ball", radius=p["radius"], mass=p["mass"])
    def world(self, scenario): o = self.obj(scenario.params); return WorldSpec(objects=[o], surfaces=[C.pedestal(o)])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); dist = float(rng.uniform(0.9, 1.3)); v = float(rng.uniform(1.8, 2.8)); tf = dist/v
        p = dict(robot="G1H", dist=dist, v=v, vz=float(0.5*9.81*tf + 0.1/tf), radius=float(rng.uniform(0.03, 0.045)), mass=float(rng.uniform(0.1, 0.3)), close_dist=0.10, f_max=100.0); return B.scenario(self, seed, p, ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="capture", params=dict(object="ball", hold="finger", links_min=2), horizon=0.8)
    def demand(self, scenario):
        p = scenario.params; t = C.make(p.get("robot", "G1H"), self.world(scenario)); o = self.obj(p)
        return t, C.demand(t, object=dict(pos=o["pos"], radius=p["radius"], mass=p["mass"], tee=True), close_dist=p["close_dist"], f_max=p["f_max"], mass=0.0)
