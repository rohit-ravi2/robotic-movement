"""Registry completeness: 96 programs, 8 per family, unique ids, every program carries the anti-duplicate sentence."""
from movement_discovery.core.contracts import Family, Status, Capability, TaskProgram, SCHEMA_VERSION


def test_registry_counts(registry):
    rows = registry.manifest(); assert len(rows) == 96
    for fam in Family: assert len(registry.programs(fam)) == 8, fam


def test_ids_unique_and_well_formed(registry):
    ids = [r["id"] for r in registry.manifest()]; assert len(ids) == len(set(ids))
    import re
    for i in ids: assert re.match(r"^P\d{2}_[a-z0-9_]+$", i), i
    nums = sorted(int(i[1:3]) for i in ids); assert nums == list(range(1, 97))


def test_contract_fields(registry):
    for cls in registry.programs():
        p = cls(); assert isinstance(p, TaskProgram)
        assert p.title and p.distinct_from.strip(), p.id
        assert p.requires and all(isinstance(c, Capability) for c in p.requires), p.id
        assert isinstance(p.status, Status) and p.status not in (Status.EXECUTED, Status.VALIDATED), (p.id, "build phase: nothing executed/validated")
        assert Capability.FLOATING_BASE in p.requires, p.id
        assert Capability.TORQUE_CONTROL in p.requires or Capability.SERVO_CONTROL in p.requires, p.id


def test_serialize_round_trip(registry):
    import json
    for r in registry.manifest():
        s = json.loads(json.dumps(r)); assert s["version"] == SCHEMA_VERSION and s["cls"].endswith(type(registry.get(r["id"])).__name__)


def test_import_is_mujoco_free():
    """Rule 4: importing every family must not import MuJoCo."""
    import subprocess, sys, os
    code = "import sys; from movement_discovery.core import registry as R; R.load_all(); assert not R.IMPORT_ERRORS, R.IMPORT_ERRORS; print('mujoco' in sys.modules)"
    out = subprocess.check_output([sys.executable, "-c", code], cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).decode().strip()
    assert out == "False"
