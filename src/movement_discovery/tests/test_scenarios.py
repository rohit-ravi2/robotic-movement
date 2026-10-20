"""Deterministic scenario generation and the committed bank (scenarios/manifest.jsonl)."""
import collections
import pytest
from movement_discovery.scenarios import generate, manifest, schemas
from movement_discovery.scenarios.manifest import _plain
from movement_discovery.scenarios.generators import generate_program, RECIPES, to_scenario


def test_sample_scenario_deterministic(registry):
    for cls in registry.programs():
        p = cls(); a, b = p.sample_scenario(3), p.sample_scenario(3); assert a.to_json() == b.to_json(), p.id
        assert a.program_id == p.id and a.seed == 3 and a.id.endswith("_s3"), p.id


def test_overrides_apply(registry):
    p = registry.get("P41_max_release_speed"); sc = p.sample_scenario(0, mass=9.0, robot="G1H"); assert sc.params["mass"] == 9.0 and sc.params["robot"] == "G1H"


def test_generate_deterministic_and_deduplicated():
    rows1 = generate(["P41_max_release_speed", "P17_impulse_recovery"], robots=("G1",)); rows2 = generate(["P41_max_release_speed", "P17_impulse_recovery"], robots=("G1",))
    assert [r["id"] for r in rows1] == [r["id"] for r in rows2] and rows1 == rows2
    hashes = [r["params_hash"] for r in rows1]; assert len(hashes) == len(set(hashes))
    assert {r["recipe"] for r in rows1} >= {"seed", "direction", "mass", "impulse", "capacity"}


def test_recipes_only_when_keys_exist(registry):
    p = registry.get("P09_max_vertical"); rows, _ = generate_program(p, robots=("G1",)); assert {r["recipe"] for r in rows} == {"seed"}   # no mass/direction knobs -> seeds only
    for name, applies, variants in RECIPES:
        assert callable(applies) and callable(variants) and name


def test_robot_gating_uses_adapter(registry):
    """A robot that the adapter refuses yields no rows and a note; a fake check never touches MuJoCo."""
    p = registry.get("P75_one_handed_capture")
    rows, notes = generate_program(p, robots=("G1", "G1H", "H1"), robot_check=lambda r, prog: (r == "G1H", "" if r == "G1H" else "missing finger_grasp"))
    assert {r["robot"] for r in rows} == {"G1", "G1H"} and notes == {"H1": "missing finger_grasp"}


def test_committed_manifest():
    rows = manifest.read(); assert len(rows) >= 500, "the bank must hold 500+ scenarios"
    ids = [r["id"] for r in rows]; assert len(ids) == len(set(ids))
    for r in rows: assert schemas.validate(r, schemas.SCENARIO) == [] and r["status"] == "defined"
    per = collections.Counter(r["program_id"] for r in rows); assert len(per) == 96
    sc = to_scenario(rows[0]); assert sc.program_id == rows[0]["program_id"] and sc.params == rows[0]["params"]


def test_manifest_write_read_round_trip(tmp_path):
    rows = generate(["P41_max_release_speed"], robots=("G1",)); p = str(tmp_path / "m.jsonl")
    assert manifest.write(rows, p) == len(rows) and manifest.read(p) == [__import__("json").loads(__import__("json").dumps(r, sort_keys=True, default=_plain)) for r in rows]
    assert manifest.summary(rows)["n"] == len(rows)
