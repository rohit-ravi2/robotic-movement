"""scenarios/manifest.jsonl: one line per scenario row (schemas.SCENARIO). write() validates every row; read() returns the rows;
summary() counts by program / family / robot / recipe. Regenerate with `python -m movement_discovery.cli scenarios --write`."""
from __future__ import annotations
from typing import Dict, Iterable, List
import json, os, collections
from . import schemas

MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.jsonl")


def write(rows: Iterable[dict], path: str = MANIFEST_PATH) -> int:
    rows = list(rows); n = 0
    for r in rows: schemas.check(r, schemas.SCENARIO)
    ids = [r["id"] for r in rows]; assert len(ids) == len(set(ids)), "duplicate scenario ids"
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r, sort_keys=True, default=_plain) + "\n"); n += 1
    return n


def _plain(o):
    if hasattr(o, "tolist"): return o.tolist()
    if hasattr(o, "item"): return o.item()
    if hasattr(o, "value"): return o.value
    return str(o)


def read(path: str = MANIFEST_PATH) -> List[dict]:
    if not os.path.exists(path): return []
    with open(path) as f: return [json.loads(l) for l in f if l.strip()]


def summary(rows: List[dict]) -> Dict[str, dict]:
    c = lambda k: dict(sorted(collections.Counter(r[k] for r in rows).items()))
    return dict(n=len(rows), by_family=c("family"), by_robot=c("robot"), by_recipe=c("recipe"), by_status=c("status"), by_program=c("program_id"))


def rows_for(program_id: str, rows: List[dict] | None = None, robot: str | None = None) -> List[dict]:
    rows = read() if rows is None else rows
    return [r for r in rows if r["program_id"] == program_id and (robot is None or r["robot"] == robot)]
