"""Program 15 (aerial): repeated hopping. Produce n confirmed flights (each with a minimum COM rise) within the horizon and finish upright.
Outcome = number of qualifying flights. Representation: torque_knots over a long horizon or contact_schedule+knots. Bridge:
JumpTask with the landing controller counts the FIRST flight only (its controller stands after landing): the check proves the flight
counter on a real record, not the task. integration check: 2026-10-12 one prior-only JumpTask rollout (ascent prior, zero knots, tscale 1; shared by P09-P16): flight 0.151 m (takeoff 1.72 m/s), landing controller did NOT recover (up_end 0.16, NOT_LANDED), 8 s; flights counted 1 (foot-site airborne test on the post-takeoff trace), NOT_UPRIGHT_AT_END."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, Status, WorldSpec, GoalSpec, Outcome
from movement_discovery.core.registry import register
from ._local import AerialProgram, terrain, flight, aerial_validity, no_trace, fail, upright, schedule_demand, phase, airborne, CS_NAME


@register
class RepeatedHopping(AerialProgram):
    id = "P15_repeated_hopping"; family = Family.AERIAL; title = "Several consecutive flights with a standing finish"
    requires = [Capability.FOOT_SITE_LEFT, Capability.FOOT_SITE_RIGHT, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.SOURCE_IMPLEMENTED
    distinct_from = "P09 is one flight from rest; here each landing is the next takeoff's initial state (energy must be recycled or re-supplied under the landing load)."
    def world(self, scenario): return WorldSpec(terrain=terrain("flat"), tag="flat")
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); return self.scenario(seed, dict(robot="G1", prior="ascent", n=int(rng.integers(2, 5)), h_min=0.02, horizon=[2.0, 5.0]), **ov)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="sequence", params=dict(flights=p["n"], h_min=p["h_min"], finish="upright"), tolerance=dict(), horizon=p["horizon"][1])
    def strategy_space(self, scenario): return ["torque_knots", CS_NAME]
    def schedule(self, scenario):
        """(EffectTask, demand) for the contact_schedule+knots representation; `demand` keeps the JumpTask bridge."""
        p = scenario.params; return schedule_demand(scenario, p["horizon"], [phase("feet_any", self.goal(scenario), flights_min=p["n"])], self.goal(scenario))
    def evaluate(self, rec, scenario):
        if not rec: return no_trace()
        p = scenario.params; f = flight(rec); val = aerial_validity(rec, f, need_landing=False)
        if f["q"] is None: fail(val, "NO_LANDING_TRACE"); return Outcome(value=0.0, achieved=False, validity=val)
        t, d = self.demand(scenario); air = airborne(t.plant(d), f["q"]); n = max(1, int(round(0.02/f["dt"])))      # confirmed flight: airborne >= 0.02 s
        runs = np.diff(np.concatenate([[0], air.astype(int), [0]])); starts, ends = np.where(runs == 1)[0], np.where(runs == -1)[0]; n_fl = int(np.sum((ends - starts) >= n))
        up, m = upright(f["q"], f["dt"], T=0.3)
        if not up: fail(val, "NOT_UPRIGHT_AT_END")
        return Outcome(value=float(n_fl), achieved=(n_fl >= p["n"] and val.ok), measures=dict(flights=int(n_fl), first_height=f["height"], **m), validity=val)
