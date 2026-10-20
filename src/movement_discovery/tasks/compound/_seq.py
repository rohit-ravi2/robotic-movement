"""Compound-family helpers (). A compound program is an ORDERED LIST OF EXTERNAL SUB-GOALS with world-state EVENTS between
them (never a movement sequence); demand() returns a core.contracts.Plan whose Steps are the other families' bridges (WorldReachTask,
ImpactTask, CatchTask, ...) executed by runner.run_sequence from the real handed-off states (construction.state_map across plants,
freeze between steps, no reset), and whose `missing` names the sub-goals without a bridge today and why. MuJoCo-free at import."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import GoalSpec, Status, Plan, Step
from movement_discovery.tasks.force_interaction import _base as B

NQ_MAPPING = "bare plants (jump/throw/locomotion) and world plants (objects, mechanisms) differ in qpos layout: runner.run_sequence maps the state by joint name (construction.state_map), but the bare task must accept the world scene for the object to exist in its step"
LOCOMOTION = "a locomotion step task that chains through the Plan runner () is the missing asset"
TORQUE = "torque_knots+tracked_prior"


def step(task, demand, event=None, representation=None, **kw) -> Step:
    """One sub-goal: a reference task + its demand, the world event that ends it, the construction for it (torque for impact / catch
    steps, the program's default (servo) otherwise); held objects read from the demand."""
    return Step(task=task, demand=dict(demand), held=list(demand.get("held") or []), event=event, representation=representation, **kw)


class CompoundProgram:
    subgoals: list = []            # [dict(goal=<program id or family>, event=<world condition>, ...)]
    def embodiment(self, robot):
        from movement_discovery import embodiment as E; return E.resolve(robot, self)
    def goal(self, scenario): return GoalSpec(kind="sequence", params=dict(subgoals=self.subgoals, scenario=scenario.params), horizon=sum(float(s.get("horizon", 3.0)) for s in self.subgoals))
    def strategy_space(self, scenario): return ["servo_knots", "torque_knots+tracked_prior", "contact_schedule+knots"]
    def validity(self, rec, scenario): return self.evaluate(rec, scenario).validity
    def observe(self, rec, scenario):
        if "steps" not in rec: return None
        return [dict(task=s["task"], valid=s["valid"], reasons=s["reasons"], handoff=s["handoff"], settled=s["settled"], peak_util=s["extra"].get("peak_util"), work=s["extra"].get("work")) for s in rec["steps"]]
    def run(self, scenario, **kw):
        """Thin wrapper: ONE prior-only rollout of the Plan through the runner (the raw record, with rec["missing"] from the Plan)."""
        from movement_discovery import runner
        return runner.rollout(self.id, scenario, robot=scenario.params.get("robot", "G1"))["rec"]


def missing_reasons(rec): return [f"MISSING:{m['goal']}" for m in rec.get("missing", [])]
