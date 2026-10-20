"""Compound composition: ordered external sub-goals with events (never a movement sequence); demand() = core.contracts.Plan."""
import pytest
from movement_discovery.core.contracts import Family, Plan, Step
from movement_discovery.tasks.compound._seq import CompoundProgram, missing_reasons


def test_subgoals_and_sequence_goal(registry):
    for cls in registry.programs(Family.COMPOUND):
        p = cls(); assert isinstance(p, CompoundProgram) and len(p.subgoals) >= 2, p.id
        for s in p.subgoals: assert "goal" in s and "event" in s, (p.id, s)        # an external condition between steps, not a pose
        g = p.goal(p.sample_scenario(0)); assert g.kind == "sequence" and g.horizon == pytest.approx(sum(float(s.get("horizon", 3.0)) for s in p.subgoals))
        assert "contact_schedule+knots" in p.strategy_space(p.sample_scenario(0))


@pytest.mark.mujoco
def test_demand_shape_and_missing_assets(registry, mujoco):
    """Executable chains carry Steps; scaffolded programs name the missing asset instead of passing silently."""
    for cls in registry.programs(Family.COMPOUND):
        p = cls(); plan = p.plan(p.sample_scenario(0)); assert isinstance(plan, Plan) and isinstance(plan.robot, str) and plan is p.demand(p.sample_scenario(0)) or isinstance(p.demand(p.sample_scenario(0)), Plan), p.id
        if p.status.value == "scaffolded": assert plan.missing and all("why" in m for m in plan.missing), p.id
        for st in plan.steps: assert isinstance(st, Step) and st.task is not None and isinstance(st.demand, dict) and st.event, p.id
        assert missing_reasons(dict(missing=plan.missing)) == plan.missing_reasons() == [f"MISSING:{m['goal']}" for m in plan.missing]


def test_missing_never_silent():
    plan = Plan(steps=[], missing=[dict(goal="locomotion", why="no step task")]); assert not plan.executable and plan.missing_reasons() == ["MISSING:locomotion"]
    assert missing_reasons(dict(missing=plan.missing)) == ["MISSING:locomotion"]
