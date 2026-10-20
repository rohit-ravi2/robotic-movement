"""Program 16 (aerial): airborne target contact. Bring the right palm within tolerance of a target point placed above the standing reach
while BOTH feet are unloaded, then land upright. Outcome = -(closest airborne palm distance). Bridge: JumpTask (palm positions by forward
kinematics of the post-takeoff pelvis/joint trace at 10 ms; airborne = both foot sites lifted off their settled height). integration check: 2026-10-12 one prior-only JumpTask rollout (ascent prior, zero knots, tscale 1; shared by P09-P16): flight 0.151 m (takeoff 1.72 m/s), landing controller did NOT recover (NOT_LANDED), 8 s; closest airborne palm distance 0.61 m."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import AerialProgram, terrain, flight, aerial_validity, no_trace, fail, site_xyz, airborne


@register
class AirborneContact(AerialProgram):
    id = "P16_airborne_contact"; family = Family.AERIAL; title = "Touch a point above standing reach while airborne"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.PALM_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "The flight and an arm reach are coupled through angular momentum conservation in the air; P09 has no effector goal."
    observe_site = "right_palm"
    def world(self, scenario): return WorldSpec(terrain=terrain("flat"), tag="flat")          # the target is a point in space (goal params), not a body
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); return self.scenario(seed, dict(robot="G1", prior="ascent", target=[float(rng.uniform(0.0, 0.25)), float(rng.uniform(-0.35, -0.05)), float(rng.uniform(1.35, 1.60))], tol=0.05, reach_standing=1.30), **ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="effector_position", params=dict(effector="hand_r", target=p["target"], while_="airborne", landing="upright"), tolerance=dict(pos=p["tol"]), horizon=2.6)
    def evaluate(self, rec, scenario):
        if not rec: return no_trace()
        p = scenario.params; f = flight(rec); val = aerial_validity(rec, f)
        if f["q"] is None: fail(val, "NO_LANDING_TRACE"); return Outcome(value=-1.0, achieved=False, validity=val)
        t, d = self.demand(scenario); S = t.plant(d); palm = site_xyz(S, f["q"], "right_palm"); air = airborne(S, f["q"])
        dist = np.linalg.norm(palm - np.asarray(p["target"]), axis=1); dmin = float(dist[air].min()) if air.any() else float(dist.min()) + 1.0
        if not air.any(): fail(val, "NEVER_AIRBORNE_IN_TRACE")
        return Outcome(value=-dmin, achieved=(dmin <= p["tol"] and val.ok), measures=dict(dmin_airborne=dmin, palm_zmax=float(palm[:, 2].max()), height=f["height"], landed=f["landed"]), validity=val)
