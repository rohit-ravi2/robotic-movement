"""Program 92 (compound): RISE from a seat, MOVE to a box, apply a sustained CONTACT force to it. Sub-goals with events: (1) risen:
standing posture reached from a seated start (posture family); (2) moved: COM within reach of the box (locomotion);
(3) pushed: box displaced >= min_disp with the sustained-force gates (P58 bridge). Executable today: step 3 from the stance; 1 and 2
need the posture/locomotion tasks chained through compose and the plant state mapping (CORE). Status SCAFFOLDED, partial run.
integration check: 2026-10-12 — prior-only rollout, plant built, value 0, achieved False, gates fired [s0:NOT_HIT,s0:CONTACT_TOO_BRIEF,SEQUENCE_NOT_OK,MISSING:posture:sit_to_stand,MISSING:locomotion:point_to_point]; displacement=0 f_max=0 t_contact=0 (19 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, Plan
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from movement_discovery.tasks.force_interaction import _base as B
from . import _seq as Q


@register
class RiseMoveContact(Q.CompoundProgram, TaskProgram):
    id = "P92_rise_move_contact"; family = Family.COMPOUND; title = "Stand up, walk to a box, push it steadily"
    requires = [Capability.PALM_SITE_RIGHT, Capability.EFFECTOR_COLLIDER, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE, Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT]; status = Status.SCAFFOLDED
    distinct_from = "P58 pushes from the canonical stance; here the stance is inherited from a rise and an approach, so the push must be constructed from whatever posture those leave, and the push's bracing competes with the walk's momentum."
    subgoals = [dict(goal="posture:sit_to_stand", event="standing", horizon=3.0), dict(goal="locomotion:point_to_point", event="com_within", of="box", dist="stand_off", horizon=4.0), dict(goal="P58_sustained_force_application", event="object_displacement", object="ball", min="min_disp", horizon=0.8)]
    def world(self, scenario): p = scenario.params; return WorldSpec(objects=[O.box(name="ball", half=[0.15, 0.15, 0.30], mass=p["mass"], pos=[p["x_box"], -0.2, 0.30], graspable=False)])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(x_box=float(rng.uniform(1.5, 3.0)), stand_off=0.5, mass=float(rng.uniform(4.0, 10.0)), f_max=300.0, f_min=45.0, t_contact_min=0.3, min_disp=0.02); return B.scenario(self, seed, p, ov)
    def demand(self, scenario):
        p = scenario.params
        t, d = B.impact_task(p.get("robot", "G1"), effector="right_palm", support="both", direction=(0.0, 0.0), measure="displacement", hit_min=p["min_disp"], elev_tol=np.radians(89), az_tol=np.radians(30), f_max=p["f_max"], t_contact_min=p["t_contact_min"], object=dict(kind="box", pos=[0.5, -0.2, 0.3], size=[0.15, 0.15, 0.30], mass=p["mass"]))
        return Plan(steps=[Q.step(t, d, event="object_displacement", representation=Q.TORQUE)], robot=t.robot, missing=[dict(goal=self.subgoals[0]["goal"], why="posture task chained through the Plan runner (); " + Q.NQ_MAPPING), dict(goal=self.subgoals[1]["goal"], why=Q.LOCOMOTION)], settle_T=0.0, freeze_T=0.4)
    def evaluate(self, rec, scenario):
        e = rec["steps"][-1]["extra"] if "steps" in rec else rec["extra"]; disp = float(e.get("ball_speed", 0.0))
        if "steps" in rec: return B.seq_outcome(rec, value=disp, measures=dict(displacement=disp, f_max=e.get("f_max_obj"), t_contact=e.get("t_contact")), extra_reasons=Q.missing_reasons(rec) or ["MISSING:rise", "MISSING:move"], achieved=False)
        return B.outcome(rec, value=disp, measures=B.impact_measures(rec), extra_reasons=["MISSING:rise", "MISSING:move"], achieved=False)
