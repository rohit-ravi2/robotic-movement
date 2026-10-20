"""Archive round trip: entries.jsonl, references, traces, query/best/lineage, warm-start retrieval."""
import numpy as np
from movement_discovery.core.contracts import Candidate, Outcome, ValidityReport, SearchRecord
from movement_discovery.archive import make_entry
from movement_discovery.archive.retrieval import nearest, warm_start


def _entry(pid, sid, z, value, reasons=(), parent=None, params=None):
    c = Candidate(representation="torque_knots", params=dict(z=list(z)), duration=0.4, provenance=dict(generator="random", seed=0, parent=parent, scenario_params=params or dict(mass=0.3, elev=0.2)))
    o = Outcome(value=value, achieved=not reasons, validity=ValidityReport(reasons=list(reasons), violation=float(bool(reasons))))
    return c, make_entry(pid, sid, "emb", "wld", c, o, SearchRecord("random", {}, 0, 10, c.id), dict(duration=0.4, work_total=value), software=dict(git="test", mujoco="x"))


def test_append_query_best(tmp_archive):
    A = tmp_archive; c1, e1 = _entry("P41", "s0", [0.0, 0.0], 5.0); c2, e2 = _entry("P41", "s0", [0.1, 0.0], 7.0, reasons=["OUT_OF_CONE"]); c3, e3 = _entry("P42", "s1", [0.2, 0.0], 1.0)
    for e in (e1, e2, e3): A.append(e)
    assert len(list(A.entries())) == 3 and A.stats()["programs"] == {"P41": 2, "P42": 1}
    assert len(A.query("P41")) == 2 and len(A.query("P41", valid_only=True)) == 1 and A.query(representation="servo_knots") == []
    assert A.best("P41")["candidate"]["id"] == c1.id and A.best("P41", valid_only=False)["candidate"]["id"] == c2.id


def test_reference_and_trace_side_files(tmp_archive):
    A = tmp_archive; c, e = _entry("P41", "s0", [0.0], 1.0); A.append(e)
    A.store_reference(c.id, dict(nk=10, q=np.zeros(3))); ref = A.load_reference(c.id); assert ref["nk"] == 10 and np.all(ref["q"] == 0)
    A.store_trace(c.id, dict(q=np.ones((4, 2)), dt=0.001, note="x")); tr = A.load_trace(c.id); assert tr["q"].shape == (4, 2) and tr["dt"] == 0.001 and tr["note"] == "x"
    assert A.load_trace("nope") is None and A.load_reference("nope") is None


def test_lineage_and_cells(tmp_archive):
    A = tmp_archive; c1, e1 = _entry("P41", "s0", [0.0], 1.0); A.append(e1); c2, e2 = _entry("P41", "s0", [0.5], 2.0, parent=c1.id); A.append(e2)
    lin = A.lineage(c2.id); assert [e["candidate"]["id"] for e in lin] == [c2.id, c1.id]
    cells = A.best_per_cell("P41", ["duration", "work_total"], bins=2); assert cells and max(e["outcome"]["value"] for e in cells.values()) == 2.0


def test_warm_start_nearest_by_scenario_distance(tmp_archive):
    A = tmp_archive
    for sid, m, v in (("a", 0.1, 1.0), ("b", 0.5, 2.0), ("c", 2.0, 3.0)): A.append(_entry("P41", sid, [m], v, params=dict(mass=m, elev=0.2))[1])
    r = nearest(A, "P41", dict(mass=0.45, elev=0.2), "torque_knots"); assert r and r[0]["scenario_id"] == "b"
    w = warm_start(A, "P41", dict(mass=0.45, elev=0.2), "torque_knots", exclude_scenario="b"); assert w["scenario_id"] in ("a", "c") and "parent" in w
    assert nearest(A, "P41", dict(mass=0.45, elev=0.2), "servo_knots") is None
