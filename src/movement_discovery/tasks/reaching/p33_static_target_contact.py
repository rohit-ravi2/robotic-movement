"""Program 33 (reaching): bring the palm to a static point in space and hold it there (ReachTask: within tol, held 2*tol for
`hold` s, both feet planted, balance controller owning the legs). value = -positive actuator work (J; cheaper is better); the
load merged at the palm and the target's offset from the standing palm are the scenario. integration check: 2026-10-12 prior-only servo rollout (bare plant, seed 0): FELL, STANCE_LOST, NOT_REACHED; gates fire. The reference ReachTask at z=0 is valid only for ~0.2 m reaches (probe: 0.2 m fwd valid, 0.3 m fwd +0.1 falls); the knot search is what it relies on. Re-checked 2026-10-12 (Plan runner, validity through physics.validity.classify): value -468.3, gates [ENERGY_CLOSURE,NONFOOT_CONTACT,STANCE_LOST,FELL,NOT_REACHED,END_UNBALANCED], 40 s."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import Family, Capability, GoalSpec, Scenario
from movement_discovery.core.registry import register
from ._local import ReachProgram, PALM0


@register
class StaticTargetContact(ReachProgram):
    id = "P33_static_target_contact"; family = Family.REACHING; title = "Static spatial target contact"
    requires = [Capability.PALM_SITE_RIGHT, Capability.SERVO_CONTROL, Capability.FLOATING_BASE]
    distinct_from = "P36/P37 put the target where arm reach alone cannot (floor level / overhead) so legs or trunk must enter; here the target is in the arm's workspace and the question is the cheapest whole-body posture."
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(target=[float(rng.uniform(0.15, 0.40)), float(rng.uniform(-0.20, 0.25)), float(rng.uniform(-0.20, 0.30))], load=float(rng.choice([0.0, 0.5, 1.0, 2.0])), tol=0.03, hold=0.20, horizon=1.5); p.update(ov)
        return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=p)
    def goal(self, scenario): p = scenario.params; return GoalSpec(kind="pose", params=dict(effector="hand_r", target=(PALM0 + np.asarray(p["target"])).tolist(), load=p["load"]), tolerance=dict(pos=p["tol"], hold=p["hold"]), horizon=p["horizon"])
    def demand(self, scenario):
        p = scenario.params; t = self.task(scenario); return t, dict(t.demand_default(), target=list(p["target"]), load=p["load"], tol=p["tol"], hold=p["hold"], horizon=p["horizon"], prior="ascent")
