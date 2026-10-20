"""Program 94 (compound): BRACE, PUSH a heavy box with a sustained force, RECOVER to a settled stance. Sub-goals with events: (1) pushed:
box displaced >= min_disp with the sustained-force gates (P58 bridge: f_max cap, t_contact_min floor, so a slap is invalid); (2)
recovered: after the push the balance settle (engine.compose.settle, 2 s) ends AT REST with both feet loaded and the torso upright.
The brace is not scripted: whatever stance the search finds that satisfies (1) and survives (2) is the brace. Executable today.
integration check: 2026-10-12 — prior-only rollout, plant built, value 0, achieved False, gates fired [s0:NOT_HIT,s0:CONTACT_TOO_BRIEF,SEQUENCE_NOT_OK]; displacement=0 f_max=0 t_contact=0 recovered=True (18 s)"""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Family, Capability, Status, WorldSpec, Plan
from movement_discovery.core.registry import register
from movement_discovery.world import objects as O
from movement_discovery.tasks.force_interaction import _base as B
from . import _seq as Q


@register
class BracePushRecover(Q.CompoundProgram, TaskProgram):
    id = "P94_brace_push_recover"; family = Family.COMPOUND; title = "Push a heavy box steadily and come back to a quiet stance"
    requires = [Capability.PALM_SITE_RIGHT, Capability.EFFECTOR_COLLIDER, Capability.TORQUE_CONTROL, Capability.FLOATING_BASE]; status = Status.INTEGRATED
    distinct_from = "P58 scores the push alone and may end anywhere balanced-enough for its own gate; here the end state must hand off to a 2 s settle that reaches rest, which forbids a push that leans the body onto the box."
    subgoals = [dict(goal="P58_sustained_force_application", event="object_displacement", object="ball", min="min_disp", horizon=0.8), dict(goal="balance:settle", event="at_rest_upright", horizon=3.0)]
    def world(self, scenario): p = scenario.params; return WorldSpec(objects=[O.box(name="ball", half=[0.15, 0.15, 0.30], mass=p["mass"], pos=[0.5, -0.2, 0.30], graspable=False)])
    def sample_scenario(self, seed, **ov):
        rng = np.random.default_rng(seed); p = dict(mass=float(rng.uniform(8.0, 20.0)), f_max=350.0, t_contact_min=0.3, min_disp=0.02, settle_T=2.0); return B.scenario(self, seed, p, ov)
    def demand(self, scenario):
        p = scenario.params
        t, d = B.impact_task(p.get("robot", "G1"), effector="right_palm", support="both", direction=(0.0, 0.0), measure="displacement", hit_min=p["min_disp"], elev_tol=np.radians(89), az_tol=np.radians(30), f_max=p["f_max"], t_contact_min=p["t_contact_min"], object=dict(kind="box", pos=[0.5, -0.2, 0.3], size=[0.15, 0.15, 0.30], mass=p["mass"]))
        return Plan(steps=[Q.step(t, d, event="object_displacement", representation=Q.TORQUE)], robot=t.robot, settle_T=float(p["settle_T"]), freeze_T=0.0, settle_last=True)   # the recovery = the balance settle after the push (event at_rest_upright)
    def evaluate(self, rec, scenario):
        if "steps" in rec:        # sequence record: the runner ran the settle (Plan.settle_T) and judged rest
            s = rec["steps"][-1]; e = s["extra"]; disp = float(e.get("ball_speed", 0.0)); recovered = bool(s["settled"]); src = "settle"
            m = dict(displacement=disp, f_max=e.get("f_max_obj"), t_contact=e.get("t_contact"), recovered=recovered, recovered_from=src)
            return B.seq_outcome(rec, value=disp, measures=m, extra_reasons=([] if recovered else ["NOT_RECOVERED"]), achieved=recovered and disp >= scenario.params["min_disp"])
        # single-step record (runner.simulate runs a one-step Plan straight, without Plan.settle_T): recovery judged from the RELEASE
        # STATE (throw.rollout's J22 recoverability measures: both feet loaded >= 0.1 BW, capture point within 8 cm of the feet, no fall), labelled
        e = rec.get("extra", {}); disp = float(e.get("ball_speed", rec.get("value", 0.0))); fb = e.get("feet_bw") or [0.0, 0.0]; cp = e.get("cp_err_rel"); rs = rec.get("reasons", [])
        recovered = bool(min(fb) >= 0.1 and cp is not None and np.isfinite(cp) and cp <= 0.08 and not any(r in rs for r in ("FELL", "STANCE_LOST")))
        m = B.impact_measures(rec) | dict(displacement=disp, recovered=recovered, recovered_from="release_state (no settle run for a one-step Plan)", feet_bw=fb, cp_err_rel=cp)
        return B.outcome(rec, value=disp, measures=m, extra_reasons=([] if recovered else ["NOT_RECOVERED"]), extra_violation=(0.0 if recovered else 1.0 + 5.0*max(0.0, 0.1 - min(fb))), achieved=recovered and disp >= scenario.params["min_disp"])
