"""Program 95 (compound): CARRY a load while NEGOTIATING an obstacle (a step or a low bar). Sub-goals with events: (1) held; (2) obstacle
cleared: both feet beyond the obstacle's far edge with the load still held (P70's carry phase demand + world.obstacles); (3) load still
held and the stance settled. Needs the contact_schedule+knots representation with a held object (as P70) plus the obstacle terrain:
status SCAFFOLDED; this program supplies the world, the two-step Plan (grasp, then the carry schedule on the same scene) and the evaluation.
integration check: 2026-10-12 — Plan assembled (2 steps: WorldReachTask grasp -> contact_schedule+knots carry over the step terrain, P70's asset) but NOT rolled out; the carry step is listed as missing until a check runs"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, Plan
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from movement_discovery.tasks.force_interaction import _base as B
from movement_discovery.tasks.payload import _pay as P
from . import _seq as Q


@register
class CarryNegotiateObstacle(Q.CompoundProgram, TaskProgram):
    id = "P95_carry_negotiate_obstacle"; family = Family.COMPOUND; title = "Carry a load over a step"
    requires = [Capability.PALM_SITE_RIGHT, Capability.WELD_GRASP, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE, Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT]; status = Status.SCAFFOLDED
    distinct_from = "P70 carries on flat ground; the obstacle forces a swing-leg clearance and a support change under load, and the held mass shifts the COM the step negotiation must manage."
    subgoals = [dict(goal="payload:grasp", event="held", object="payload", horizon=3.0), dict(goal="negotiation:step", event="feet_beyond", obstacle="step", held=True, horizon=4.0), dict(goal="balance:settle", event="at_rest", held=True, horizon=2.0)]
    def world(self, scenario):
        p = scenario.params
        from movement_discovery.world import terrain as T_
        return WorldSpec(objects=[O.payload_with_handle(name="payload", half=[0.08, 0.08, 0.06], mass=p["mass"], pos=[0.35, -0.2, 0.06])], terrain=T_.steps(n=1, rise=p["rise"], run=0.4, x0=p["x_step"]) if hasattr(T_, "steps") else None)
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(mass=float(rng.uniform(1.0, 4.0)), rise=float(rng.uniform(0.08, 0.15)), x_step=float(rng.uniform(0.8, 1.2)), tol=0.1); return B.scenario(self, seed, p, ov)
    def demand(self, scenario):
        """Plan: (1) grasp the handle (WorldReachTask on the world plant: payload + step terrain); (2) the carry as a contact_schedule+knots
        step on the SAME scene (BaseEffectTask, support contracts alternating, the payload held through the schedule) handed off from
        the grasp's real state. Assembled, not yet checked: the carry step is P70's asset and is listed as missing until a check runs."""
        p = scenario.params; spec = self.world(scenario); t = P.task(p.get("robot", "G1"), spec); phases = [dict(support="left", horizon=0.6, held=["payload"]), dict(support="right", horizon=0.6, held=["payload"]), dict(support="both", horizon=1.0, held=["payload"])]
        d1 = P.reach_grasp(t, "payload", horizon=3.5, squat=0.5); t2, d2 = P.carry_task(p.get("robot", "G1"), phases, held=["payload"], direction=(0.0, 0.0))
        return Plan(steps=[Q.step(t, d1, event="held"), Q.step(t2, d2, event="feet_beyond", representation="contact_schedule+knots")], robot=t.robot, settle_T=0.0, freeze_T=0.4,
                    missing=[dict(goal=self.subgoals[1]["goal"], why="carry phases over the step terrain (contact_schedule+knots with the held payload: P70's asset, checked on flat ground) assembled but NOT checked here (grasp -> schedule handoff on a terrain scene); the feet_beyond event is judged from the base displacement")])
    def evaluate(self, rec, scenario):
        e = rec["steps"][-1]["extra"] if "steps" in rec else rec.get("extra", {}); bd = e.get("base_displacement"); x_end = float(bd[0]) if bd is not None else None   # pelvis x at the end of the carry (foot sites are not in the schedule record: LABEL base_for_feet)
        beyond = bool(x_end is not None and x_end >= scenario.params["x_step"] + 0.4); phases = e.get("phases") or []; held = ("payload" in (phases[-1].get("held_after") or [])) if phases else ("payload" in (e.get("held_after") or []))
        return B.outcome(rec, value=(x_end if x_end is not None else 0.0), measures=dict(base_x_end=x_end, held_after=(phases[-1].get("held_after") if phases else e.get("held_after"))), extra_reasons=([] if beyond else ["NOT_CLEARED"]) + ([] if held else ["DROPPED"]) + Q.missing_reasons(rec), achieved=beyond and held)
