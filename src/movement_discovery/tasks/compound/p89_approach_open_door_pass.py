"""Program 89 (compound): APPROACH a door on foot, OPEN it, PASS through. Sub-goals with world events: (1) approach: COM within
`stand_off` of the door plane (locomotion family, P01-type goal); (2) open: door angle >= open_min with the stance held (P60 bridge,
WorldReachTask push); (3) pass: COM beyond the door plane by `pass_depth` (locomotion). Executable today: step 2 only; steps 1 and
3 need a compose-chainable locomotion task and the bare/world plant state mapping (CORE). Status SCAFFOLDED, partial run available.
integration check: 2026-10-12 — prior-only rollout, plant built, value 0.0882, achieved False, gates fired [s0:NOT_MOVED,SEQUENCE_NOT_OK,MISSING:locomotion:point_to_point,MISSING:locomotion:point_to_point]; door_angle=0.0882 (28 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, Plan
from movement_discovery.core.registry import register
from movement_discovery.world import mechanisms as MK
from movement_discovery.tasks.force_interaction import _base as B, _mech as M
from . import _seq as Q


@register
class ApproachOpenDoorPass(Q.CompoundProgram, TaskProgram):
    id = "P89_approach_open_door_pass"; family = Family.COMPOUND; title = "Walk to a door, open it, walk through"
    requires = [Capability.PALM_SITE_RIGHT, Capability.EFFECTOR_COLLIDER, Capability.SERVO_CONTROL, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE, Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT]; status = Status.SCAFFOLDED
    distinct_from = "P60 opens a door from a fixed stance; here the stance itself is the product of an approach and the opened door is only an intermediate state for the passage, so the open angle must leave room for the body."
    subgoals = [dict(goal="locomotion:point_to_point", event="com_within", of="door_plane", dist="stand_off", horizon=4.0), dict(goal="P60_open_articulated_door", event="mechanism_displacement", mechanism="door", min="open_min", horizon=2.5), dict(goal="locomotion:point_to_point", event="com_beyond", plane="door_plane", depth="pass_depth", horizon=4.0)]
    def mech(self, p): return MK.door(name="door", pos=[p["x_door"], 0.2], width=0.9, hinge_side="left", resist=2.0, frictionloss=1.0, handle_height=1.0, max_open=1.57)
    def world(self, scenario): return WorldSpec(mechanisms=[self.mech(scenario.params)])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(x_door=float(rng.uniform(1.5, 3.0)), stand_off=0.55, open_min=np.radians(75.0), pass_depth=0.6, f_max=400.0, t_contact_min=0.3); return B.scenario(self, seed, p, ov)
    def demand(self, scenario):
        p = scenario.params; mk = dict(self.mech(p)); mk["pos"] = [0.55, 0.2]      # the executable door step is built as if the approach had ended at stand_off from the door
        t = M.make(p.get("robot", "G1"), WorldSpec(mechanisms=[mk]))
        steps = [Q.step(t, M.demand(t, mk, push=[0.3, 0.0, 0.0], horizon=2.5, min_disp=p["open_min"], f_max=p["f_max"], t_contact_min=p["t_contact_min"], sign=mk["open_sign"]), event="mechanism_displacement")]
        return Plan(steps=steps, robot=t.robot, missing=[dict(goal=self.subgoals[0]["goal"], why=Q.LOCOMOTION + "; " + Q.NQ_MAPPING), dict(goal=self.subgoals[2]["goal"], why=Q.LOCOMOTION)], settle_T=0.0, freeze_T=0.4)
    def evaluate(self, rec, scenario):
        v = float(rec["steps"][-1]["extra"].get("mech_disp", 0.0)) if "steps" in rec else float(rec["extra"].get("mech_disp", 0.0))
        return B.seq_outcome(rec, value=v, measures=dict(door_angle=v, open_min=scenario.params["open_min"]), extra_reasons=Q.missing_reasons(rec) or ["MISSING:approach", "MISSING:pass"], achieved=False) if "steps" in rec else B.outcome(rec, value=v, extra_reasons=["MISSING:approach", "MISSING:pass"], achieved=False)
