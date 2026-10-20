"""Program 57 (force_interaction): BREAKAWAY from static friction. A heavy block rests on the floor with a high friction
coefficient; the outcome is FIRST MOTION of the block (time of first displacement > 1 cm, then displacement). Distinct physics: the
force must exceed mu*m*g before anything moves (a threshold, not a proportional response). Bridge: tasks/impact.py ImpactTask
(effector sphere collider, bare-robot reference, measure='displacement') with the block's geom friction set per scenario.
integration check: 2026-10-12 — prior-only rollout, plant built, value 0, achieved False, gates fired [NOT_HIT,CONTACT_TOO_BRIEF]; displacement=0 t_contact=0 dmin_obj=0.582 (24 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, GoalSpec
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from . import _base as B


def friction_impact_task(robot, mu):
    from tasks.impact import ImpactTask
    class FrictionImpactTask(ImpactTask):
        name = "friction_impact"
        def plant(self, demand, dt=None, fresh=False):
            import mujoco; from engine import core
            S = super().plant(demand, dt if dt is not None else core.DT_OPT, fresh); m = S["m"]
            for g in range(m.ngeom):
                if int(m.geom_bodyid[g]) == int(m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]): m.geom_friction[g, 0] = float(mu)
            return S
    return FrictionImpactTask(robot)


@register
class BreakawayStaticFriction(TaskProgram):
    id = "P57_breakaway_static_friction"; family = Family.FORCE_INTERACTION; title = "Break a heavy block away from static friction"
    requires = [Capability.PALM_SITE_RIGHT, Capability.EFFECTOR_COLLIDER, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED
    distinct_from = "P58 regulates a force held over a window on a light box; here the object does not move at all until a friction threshold (mu m g) is exceeded, and the outcome is whether/when it first moves."
    def embodiment(self, robot): return B.embodiment(robot, self.requires)
    def world(self, scenario):
        p = scenario.params; return WorldSpec(objects=[O.heavy_block(mass=p["mass"], mu=p["mu"], name="ball", half=p["size"], pos=p["pos"])])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed)
        p = dict(mass=float(rng.uniform(12.0, 30.0)), mu=float(rng.uniform(0.5, 0.9)), size=[0.18, 0.18, 0.25], pos=[0.50, -0.20, 0.25], support="both", f_max=600.0, t_contact_min=0.15, min_disp=0.01)
        return B.scenario(self, seed, p, ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="object_displacement", params=dict(object="ball", min=p["min_disp"], threshold_N=p["mu"]*p["mass"]*9.81), tolerance=dict(az=np.radians(30)), horizon=0.8)
    def strategy_space(self, scenario): return ["torque_knots+tracked_prior", "torque_knots", "servo_knots", "servo_knots+brace_prior", "torque_knots+brace_prior"]
    def demand(self, scenario):
        p = scenario.params; t = friction_impact_task(p.get("robot", "G1"), p["mu"])
        return t, dict(t.demand_default(), effector="right_palm", support=p["support"], direction=(0.0, 0.0), measure="displacement", hit_min=p["min_disp"], elev_tol=np.radians(89), az_tol=np.radians(30), f_max=p["f_max"], t_contact_min=p["t_contact_min"], object=dict(kind="box", pos=p["pos"], size=p["size"], mass=p["mass"]))
    def evaluate(self, rec, scenario):
        e = rec.get("extra", {}); disp = float(e.get("ball_speed", rec.get("value", 0.0))); t_first = None; self._sc = scenario
        if rec.get("trace"):           # first motion from the recorded object trajectory
            q = np.asarray(rec["trace"]["q"]); dt = float(rec["trace"]["dt"]); qa = self._ball_qadr(rec)
            if qa is not None:
                dxy = np.linalg.norm(q[:, qa:qa + 2] - q[0, qa:qa + 2], axis=1); idx = np.nonzero(dxy > 0.01)[0]; t_first = float(idx[0]*dt) if len(idx) else None
        m = B.impact_measures(rec) | dict(displacement=disp, t_first_motion=t_first, threshold_N=scenario.params["mu"]*scenario.params["mass"]*9.81)
        return B.outcome(rec, value=disp, measures=m, achieved=disp >= scenario.params["min_disp"])
    def _ball_qadr(self, rec):
        if rec.get("ball_qadr") is not None: return int(rec["ball_qadr"])
        import mujoco; t, d = self.demand(self._sc); m = t.plant(d)["m"]; return int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")])
    def validity(self, rec, scenario): return self.evaluate(rec, scenario).validity
    def observe(self, rec, scenario): t, d = self.demand(scenario); return B.observe_impact(t, d, rec, "right_palm")
