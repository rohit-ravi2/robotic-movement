"""JSON schemas for the serialised contracts (core.contracts): Scenario rows of scenarios/manifest.jsonl and ArchiveEntry lines of
archive/entries.jsonl. validate(obj, schema) is a small dependency-free checker (type, required, properties, enum, items,
additionalProperties, nullable via a type list); it exists so tests and the CLI can refuse malformed rows without jsonschema."""
from __future__ import annotations
from typing import Any, List
from movement_discovery.core.contracts import SCHEMA_VERSION, Status, Family

_STATUS = [s.value for s in Status]; _FAMILY = [f.value for f in Family]
_NUM = ["number", "integer"]

SCENARIO = {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "title": "Scenario", "type": "object",
    "required": ["id", "program_id", "seed", "params", "version", "status"],
    "properties": {
        "id": {"type": "string", "minLength": 1}, "program_id": {"type": "string", "pattern": "^P[0-9]{2}_"},
        "seed": {"type": "integer"}, "params": {"type": "object"}, "version": {"type": "string", "const": SCHEMA_VERSION},
        "status": {"type": "string", "enum": ["defined"] + _STATUS},     # "defined": in the bank, never assembled (below DESIGNED on the ladder)
        # manifest-row extras (scenarios/manifest.jsonl)
        "family": {"type": "string", "enum": _FAMILY}, "robot": {"type": "string", "enum": ["G1", "G1H", "H1"]},
        "overrides": {"type": "object"}, "recipe": {"type": "string"}, "params_hash": {"type": "string"}, "base_id": {"type": "string"},
    },
    "additionalProperties": False,
}

VALIDITY_REPORT = {
    "title": "ValidityReport", "type": "object", "required": ["reasons", "violation"],
    "properties": {k: {"type": "boolean"} for k in ("numerical", "joint_limits", "actuator_torque", "actuator_speed", "envelope", "contact", "support", "direction", "object_interaction")}
                  | {"timestep_robust": {"type": ["boolean", "null"]}, "reasons": {"type": "array", "items": {"type": "string"}}, "violation": {"type": _NUM}},
}
CANDIDATE = {"title": "Candidate", "type": "object", "required": ["representation", "params", "duration", "id"],
             "properties": {"representation": {"type": "string"}, "params": {"type": "object"}, "duration": {"type": _NUM}, "provenance": {"type": "object"}, "id": {"type": "string"}}}
OUTCOME = {"title": "Outcome", "type": "object", "required": ["value", "achieved", "validity"],
           "properties": {"value": {"type": _NUM}, "achieved": {"type": "boolean"}, "measures": {"type": "object"}, "validity": VALIDITY_REPORT,
                          "trace_ref": {"type": ["string", "null"]}, "mechanics": {"type": ["object", "null"]}}}
SEARCH_RECORD = {"title": "SearchRecord", "type": ["object", "null"], "required": ["optimizer", "seed", "budget_evals", "best_id"],
                 "properties": {"optimizer": {"type": "string"}, "config": {"type": "object"}, "seed": {"type": "integer"}, "budget_evals": {"type": "integer"},
                                "best_id": {"type": "string"}, "history": {"type": "array"}, "secs": {"type": _NUM}}}
ARCHIVE_ENTRY = {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "title": "ArchiveEntry", "type": "object",
    "required": ["program_id", "scenario_id", "embodiment_hash", "world_hash", "candidate", "outcome", "version"],
    "properties": {"program_id": {"type": "string"}, "scenario_id": {"type": "string"}, "embodiment_hash": {"type": "string"}, "world_hash": {"type": "string"},
                   "candidate": CANDIDATE, "outcome": OUTCOME, "search": SEARCH_RECORD, "descriptors": {"type": "object"}, "software": {"type": "object"},
                   "version": {"type": "string"}},
}
SCHEMAS = {"scenario": SCENARIO, "archive_entry": ARCHIVE_ENTRY, "candidate": CANDIDATE, "outcome": OUTCOME, "validity_report": VALIDITY_REPORT, "search_record": SEARCH_RECORD}

_PY = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def _is(v: Any, t: str) -> bool:
    if t == "integer": return isinstance(v, int) and not isinstance(v, bool)
    if t == "number": return isinstance(v, (int, float)) and not isinstance(v, bool)
    return isinstance(v, _PY[t])


def validate(obj: Any, schema: dict, path: str = "$") -> List[str]:
    """List of violations (empty = valid)."""
    errs: List[str] = []; types = schema.get("type")
    if types is not None:
        ts = [types] if isinstance(types, str) else list(types)
        if not any(_is(obj, t) for t in ts): return [f"{path}: expected {ts}, got {type(obj).__name__}"]
    if "const" in schema and obj != schema["const"]: errs.append(f"{path}: expected const {schema['const']!r}, got {obj!r}")
    if "enum" in schema and obj not in schema["enum"]: errs.append(f"{path}: {obj!r} not in {schema['enum']}")
    if isinstance(obj, str):
        if "minLength" in schema and len(obj) < schema["minLength"]: errs.append(f"{path}: shorter than {schema['minLength']}")
        if "pattern" in schema:
            import re
            if not re.search(schema["pattern"], obj): errs.append(f"{path}: {obj!r} does not match {schema['pattern']}")
    if isinstance(obj, dict):
        for k in schema.get("required", []):
            if k not in obj: errs.append(f"{path}: missing required {k!r}")
        props = schema.get("properties", {})
        for k, v in obj.items():
            if k in props: errs += validate(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties", True) is False: errs.append(f"{path}: unexpected property {k!r}")
    if isinstance(obj, list) and "items" in schema:
        for i, v in enumerate(obj): errs += validate(v, schema["items"], f"{path}[{i}]")
    return errs


def check(obj: Any, schema: dict) -> None:
    e = validate(obj, schema)
    if e: raise ValueError("; ".join(e[:8]) + (f" (+{len(e) - 8} more)" if len(e) > 8 else ""))
