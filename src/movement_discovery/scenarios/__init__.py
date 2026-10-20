"""Scenario bank (): deterministic, declarative configurations per program; nothing here rolls out.
    generators.generate(programs=None, robots=("G1","G1H","H1"), base_seeds=4, alt_seeds=2) -> list[dict]   manifest rows (status "defined")
    manifest.write(rows, path) / read(path) / summary(rows)                                                    scenarios/manifest.jsonl
    schemas.SCENARIO, schemas.ARCHIVE_ENTRY, schemas.validate(obj, schema)                                     JSON schemas (no jsonschema dep)
Scenarios are produced by program.sample_scenario(seed, **overrides) plus explicit factorial overrides (generators.RECIPES) and
de-duplicated by (program, robot, params) hash. Robots other than G1 are included only when embodiment.resolve accepts the program."""
from . import schemas, generators, manifest   # noqa
from .generators import generate, scenario_row   # noqa
from .manifest import write, read, summary, MANIFEST_PATH   # noqa
