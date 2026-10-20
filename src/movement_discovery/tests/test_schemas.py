"""Schema round trips for Scenario and ArchiveEntry; the validator refuses malformed rows."""
import json
import pytest
from movement_discovery.core.contracts import Scenario, Candidate, Outcome, ValidityReport, SearchRecord, ArchiveEntry, Status, SCHEMA_VERSION
from movement_discovery.scenarios import schemas
from movement_discovery.archive import _plain


def test_scenario_round_trip():
    sc = Scenario(id="P41_max_release_speed_s0", program_id="P41_max_release_speed", seed=0, params=dict(mass=0.145, elev=0.2))
    row = json.loads(sc.to_json()); assert row["version"] == SCHEMA_VERSION and row["status"] == Status.DESIGNED.value
    assert schemas.validate(row, schemas.SCENARIO) == []


def test_scenario_schema_refuses():
    bad = dict(id="x", program_id="not_a_program", seed="0", params={}, version="md-0.9", status="executed_by_magic")
    errs = schemas.validate(bad, schemas.SCENARIO); assert any("program_id" in e for e in errs) and any("seed" in e for e in errs) and any("version" in e for e in errs) and any("status" in e for e in errs)
    with pytest.raises(ValueError): schemas.check(bad, schemas.SCENARIO)
    assert schemas.validate(dict(id="a", program_id="P01_x", seed=0, params={}, version=SCHEMA_VERSION, status="defined", extra=1), schemas.SCENARIO)   # additionalProperties false


def test_archive_entry_round_trip():
    cand = Candidate(representation="torque_knots", params=dict(z=[0.0, 0.1]), duration=0.4, provenance=dict(generator="random", seed=0))
    out = Outcome(value=1.5, achieved=False, measures=dict(t_rel=0.4), validity=ValidityReport(reasons=["OUT_OF_CONE"], violation=0.3, direction=False))
    e = ArchiveEntry(program_id="P41_max_release_speed", scenario_id="s0", embodiment_hash="abc", world_hash="def", candidate=cand, outcome=out, search=SearchRecord("random", {}, 0, 10, cand.id))
    row = json.loads(json.dumps(_plain(e))); assert schemas.validate(row, schemas.ARCHIVE_ENTRY) == []
    assert row["candidate"]["id"] == cand.id and len(cand.id) == 12
    row["outcome"]["validity"]["violation"] = "high"; assert schemas.validate(row, schemas.ARCHIVE_ENTRY)


def test_validity_ok_property():
    assert ValidityReport().ok and not ValidityReport(reasons=["FELL"]).ok and not ValidityReport(violation=0.1).ok


def test_plan_contract_and_coercion():
    """Every program's demand() is a Plan or the legacy pair; as_plan makes a one-step Plan of the pair; empty Plans name their missing assets."""
    from movement_discovery.core.contracts import Plan, Step, as_plan
    class _T:
        name = "t"; robot = "G1"
    pl = as_plan((_T(), dict(a=1, held=["box"])), robot="G1"); assert isinstance(pl, Plan) and pl.single and pl.steps[0].held == ["box"] and pl.task.name == "t" and pl.demand == dict(a=1, held=["box"])
    assert as_plan(pl) is pl and not Plan(steps=[], missing=[dict(goal="x", why="y")]).executable and Plan(steps=[]).missing_reasons() == []
    assert Plan(steps=[Step(task=_T(), demand={}), Step(task=_T(), demand={}, event="held")]).pairs()[1][0].name == "t"
    import pytest
    with pytest.raises(TypeError): as_plan(("robot", [], []))
