"""Program 64 (force_interaction): CONTROLLED torque on a grasped object: a valve knob on a fixed axis must be turned through a
bounded angle while the torque transmitted stays in a band (too little: no motion; too much: the 'valve' is over-torqued, judged by
the peak handle force x radius). Hold = G1H weld to the knob (experimental simplification, Capability.WELD_GRASP); the turn is
commanded as a hand-orientation goal on the held body (needs the 3-DOF wrist). Outcome = valve angle with the torque band.
integration check: 2026-10-12 — prior-only rollout, plant built, value 0.613, achieved False, gates fired [s0:NOT_GRASPED,s0:NOT_MOVED,s1:START_NOT_AT_REST,SEQUENCE_NOT_OK,OVER_TORQUE]; displacement=0.613 f_max=319 t_contact=0.22 torque_peak=12.8 (36 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec
from movement_discovery.core.registry import register
from movement_discovery.world import mechanisms as MK
from . import _base as B, _mech as M


@register
class ControlledTorqueGrasped(M.MechProgram, TaskProgram):
    id = "P64_controlled_torque_grasped"; family = Family.FORCE_INTERACTION; title = "Turn a valve through a bounded angle inside a torque band"
    requires = [Capability.PALM_SITE_RIGHT, Capability.WRIST_3DOF, Capability.WELD_GRASP, Capability.SERVO_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED
    unit = "rad"; distinct_from = "P62 carries a knob around a circle with the whole arm under a constant load; here the hand stays on axis and the wrist supplies a BOUNDED torque through a small angle, with an upper torque limit as a validity gate."
    def mech(self, p): return MK.crank(name="valve", pos=p["pos"], height=p["height"], radius=0.04, axis=[1, 0, 0], resist=0.2, load=p["load_torque"], handle_len=0.10, mass=0.3)
    def world(self, scenario): return WorldSpec(mechanisms=[self.mech(scenario.params)])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); lt = float(rng.uniform(0.3, 1.5))
        p = dict(robot="G1H", pos=[0.42, -0.25], height=float(rng.uniform(0.75, 0.9)), load_torque=lt, angle=float(np.radians(rng.uniform(30, 90))), torque_max=3.0*lt + 1.0, min_disp=float(np.radians(20.0)))
        return B.scenario(self, seed, p, ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="mechanism_displacement", params=dict(mechanism="valve", joint="valve_hinge", min=p["min_disp"], target=p["angle"], torque_band=(p["load_torque"], p["torque_max"])), horizon=5.0)
    def strategy_space(self, scenario): return ["servo_knots", "servo_knots+brace_prior"]
    def demand(self, scenario):
        p = scenario.params; mk = self.mech(p); t = M.make(p.get("robot", "G1H"), self.world(scenario)); a = p["angle"]
        d1 = M.demand(t, mk, push=[-0.03, 0.0, 0.0], horizon=2.5, grasp=mk["name"], min_disp=0.0)
        # turn = an orientation goal on the HELD valve body: its -y arm axis rotated by `angle` about the valve axis (+x), the hand stays at the knob
        tgt = [0.0, float(-np.cos(a)), float(-np.sin(a))]
        d2 = M.demand(t, mk, push=[-0.03, 0.0, 0.0], horizon=2.5, min_disp=p["min_disp"], held=[mk["name"]], orient=[dict(body=mk["body"], axis=[0, -1, 0], target=tgt, k=6.0, max=1.5)])
        return M.plan(t, [d1, d2], events=("held", "mechanism_displacement"), freeze_T=0.3)
    def evaluate(self, rec, scenario):
        out = super().evaluate(rec, scenario); p = scenario.params; f = (out.measures.get("f_max") or 0.0); tq = f*0.04
        out.measures.update(torque_peak=tq, torque_band=(p["load_torque"], p["torque_max"]), target=p["angle"])
        if tq > p["torque_max"]: out.validity.reasons.append("OVER_TORQUE"); out.validity.violation += 1.0 + (tq - p["torque_max"]); out.validity.contact = False; out.achieved = False
        return out
