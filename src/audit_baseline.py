"""Replay and fingerprint the throw baselines without modifying historical results.

This is deliberately an audit command: it loads saved controls, replays them through
the current ThrowTask wrapper, and writes a new manifest describing any drift.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tasks.throw import ThrowTask  # noqa: E402
from engine import task as task_engine  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception as e:  # audit should remain useful in exported trees
        return f"ERROR: {e}"


def replay_task(rec: dict) -> dict:
    task = ThrowTask("G1")
    measured = task_engine.measure(task, rec["demand"], np.asarray(rec["z"], dtype=float))
    return {
        "stored_value": rec.get("value"),
        "stored_ok_all": rec.get("ok_all"),
        "replay_value": measured.get("value"),
        "replay_ok_all": measured.get("ok_all"),
        "replay_dt_check": [{"dt": x.get("dt"), "value": x.get("value"), "valid": x.get("valid"), "reasons": x.get("reasons", [])} for x in measured.get("meas", [])],
        "replay_reasons": measured.get("meas", [{}])[0].get("reasons", []),
        "replay_overspeed": measured.get("meas", [{}])[0].get("extra", {}).get("overspeed"),
        "replay_extra": measured.get("meas", [{}])[0].get("extra", {}),
        "z_hash": measured.get("z_hash"),
    }


def main() -> None:
    atlas_path = ROOT / "results/atlas/throw_gated_v1.json"
    old_path = ROOT / "results/phase11/throw_g1_main.json"
    ladder_path = ROOT / "results/phase11/throw_g1_ladder2400.json"
    atlas = json.loads(atlas_path.read_text())
    manifest = {
        "schema": "movement-law-baseline-audit-v1",
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_head": git(["rev-parse", "HEAD"]),
        "git_status_short": git(["status", "--short"]).splitlines(),
        "replay_environment": {"mujoco": __import__("mujoco").__version__},
        "files": {},
        "replays": {"corrected_atlas": replay_task(atlas)},
    }
    for p in [
        "models/mujoco_menagerie/unitree_g1/scene_mjx.xml",
        "models/mujoco_menagerie/unitree_g1/scene_hands_palm.xml",
        "src/throw.py", "src/tasks/throw.py", "src/engine/task.py", "src/branch_i.py",
        "results/atlas/throw_gated_v1.json", "results/phase11/throw_g1_main.json",
        "results/phase11/throw_g1_ladder2400.json",
    ]:
        q = ROOT / p
        manifest["files"][p] = {"exists": q.exists(), "sha256": sha(q) if q.exists() else None}
    if old_path.exists():
        old = json.loads(old_path.read_text())
        rows = old if isinstance(old, list) else [old]
        manifest["historical_pre_envelope"] = {
            "entries": len(rows),
            "stored_speeds": [x.get("speed", x.get("value")) for x in rows],
            "stored_valid_all_dt": [x.get("valid_all_dt") for x in rows],
            "stored_reasons": [x.get("reasons", []) for x in rows],
            "note": "Historical artifact retained; current source replays it through the corrected envelope and may reject it.",
        }
    if ladder_path.exists():
        ladder = json.loads(ladder_path.read_text())
        manifest["ladder_summary"] = {
            "entries": len(ladder) if isinstance(ladder, list) else None,
            "stored_speeds": [x.get("speed") for x in ladder] if isinstance(ladder, list) else None,
            "stored_valid": [x.get("valid_all_dt") for x in ladder] if isinstance(ladder, list) else None,
        }
    out = ROOT / "results/audit_baseline_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps(manifest["replays"], indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
