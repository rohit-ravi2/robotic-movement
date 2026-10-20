"""Program 13 (aerial): aerial reorientation. Change the base yaw by a commanded angle between takeoff and touchdown and land upright
facing the new heading. Outcome = -|yaw change - target|. Bridge: JumpTask. integration check: 2026-10-12 one prior-only JumpTask rollout (ascent prior, zero knots, tscale 1; shared by P09-P16): flight 0.151 m (takeoff 1.72 m/s), landing controller did NOT recover (up_end 0.16, NOT_LANDED), 8 s; dyaw 1.7 deg (target 70 deg), NOT_LANDED."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import AerialProgram, terrain, flight, landing_displacement, aerial_validity, no_trace, fail


@register
class AerialReorientation(AerialProgram):
    id = "P13_aerial_reorientation"; family = Family.AERIAL; title = "Turn about the vertical during flight"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE, Capability.WAIST_YAW]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "Angular momentum about z at takeoff (or its in-flight exchange between trunk and legs) is the physics here, not displacement."
    def world(self, scenario): return WorldSpec(terrain=terrain("flat"), tag="flat")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); return self.scenario(seed, dict(robot="G1", prior="ascent", dpsi=float(rng.choice([-1, 1])*np.radians(rng.uniform(30, 180))), tol=np.radians(10), drift_max=0.25), **ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="heading", params=dict(dpsi=p["dpsi"], during="flight", landing="upright"), tolerance=dict(psi=p["tol"]), horizon=2.6)
    def evaluate(self, rec, scenario):
        if not rec: return no_trace()
        p = scenario.params; f = flight(rec); val = aerial_validity(rec, f); d, dyaw = landing_displacement(f)
        if d is None: fail(val, "NO_LANDING_TRACE"); return Outcome(value=-abs(p["dpsi"]), achieved=False, validity=val)
        err = float(abs((dyaw - p["dpsi"] + np.pi) % (2*np.pi) - np.pi))
        if np.linalg.norm(d) > p["drift_max"]: fail(val, "LANDING_DRIFT", float(np.linalg.norm(d)) - p["drift_max"])
        return Outcome(value=-err, achieved=(err <= p["tol"] and val.ok), measures=dict(dyaw_deg=float(np.degrees(dyaw)), err_deg=float(np.degrees(err)), drift=float(np.linalg.norm(d)), height=f["height"], rot=f["rot"], landed=f["landed"]), validity=val)
