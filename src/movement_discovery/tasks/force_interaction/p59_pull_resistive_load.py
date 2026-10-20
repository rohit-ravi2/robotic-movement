"""Program 59 (force_interaction): PULL a resistive load. A heavy box with a handle (world.objects.payload_with_handle, its own
floor friction mu) must be dragged toward the robot; resistance = mu*m*g, mass-proportional. The pull needs a HOLD on the handle:
on the G1 the hold is the engine.world WELD hand<->payload body (experimental simplification, Capability.WELD_GRASP); on the G1H the
Dex3 cylindrical wrap (engine/grasps U_EQ/C_EQ) is the declared hold (Capability.FINGER_GRASP) and is NOT exercised by the bridge yet.
External sub-goals: (1) handle held, (2) load displaced >= min_disp toward the robot. Bridge: WorldReachTask over compose.run_sequence.
integration check: 2026-10-12 — prior-only rollout, plant built, value -1.51e-06, achieved False, gates fired [s0:NOT_REACHED,s0:NOT_GRASPED,s1:NONFOOT_CONTACT,s1:STANCE_LOST,s1:NOT_REACHED,s1:END_UNBALANCED,s1:HOLD_LOST,SEQUENCE_NOT_OK]; displacement=-1.51e-06 f_max=0 t_contact=0 (68 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from . import _base as B, _mech as M


@register
class PullResistiveLoad(M.MechProgram, TaskProgram):
    id = "P59_pull_resistive_load"; family = Family.FORCE_INTERACTION; title = "Drag a friction-loaded box toward the body by its handle"
    requires = [Capability.PALM_SITE_RIGHT, Capability.WELD_GRASP, Capability.SERVO_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED
    unit = "m"; distinct_from = "P61's drawer is a guided slide with a fixed dry friction at counter height; the load here is free on the floor, its resistance scales with its mass (mu m g), and the pull is toward the body through a hold."
    def obj(self, p): return O.payload_with_handle(name="sled", half=[0.15, 0.12, 0.10], mass=p["mass"], pos=[p["x"], -0.22, 0.10], mu=p["mu"])
    def world(self, scenario): return WorldSpec(objects=[self.obj(scenario.params)])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(mass=float(rng.uniform(6.0, 20.0)), mu=float(rng.uniform(0.2, 0.5)), x=float(rng.uniform(0.35, 0.45)), min_disp=0.08, pull=0.22, f_max=500.0)
        return B.scenario(self, seed, p, ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="sequence", params=dict(subgoals=[dict(event="held", object="sled"), dict(event="object_displacement", object="sled", min=p["min_disp"], direction="-x", resistance_N=p["mu"]*(p["mass"] + 0.1)*9.81)]), horizon=5.0)
    def strategy_space(self, scenario): return ["servo_knots", "torque_knots+tracked_prior", "servo_knots+brace_prior", "torque_knots+brace_prior"]
    def _value(self, rec):
        last = rec["steps"][-1] if "steps" in rec else rec; return float(-last["extra"].get("obj_disp", {}).get("sled", [0.0])[0])
    def demand(self, scenario):
        """Plan: reach the handle (grip site, from above) and hold it, then pull by p['pull'] with the hold carried over (runner.run_sequence)."""
        p = scenario.params; o = self.obj(p); t = M.make(p.get("robot", "G1"), self.world(scenario), colliders=(), hand_collision=False)   # weld abstraction (no palm collider): labelled
        d1 = dict(t.demand_default(), target_object="sled", push=[0.0, 0.0, 0.01], tol=0.03, hold=0.3, grasp_on_reach="sled", grasp_tol=0.05, omega_servo=120.0, trunk_cap=0.05, horizon=3.5, approach="direct", refine=1, aim_z=True, squat=0.5)
        d2 = dict(t.demand_default(), target_object="sled", push=[-p["pull"], 0.0, 0.01], tol=0.05, hold=0.3, held=["sled"], omega_servo=120.0, trunk_cap=0.05, horizon=3.0, approach="direct", f_max=p["f_max"])
        return M.plan(t, [d1, d2], events=("held", "object_displacement"))
