"""JSONL archive of ArchiveEntry records with side files: pickled references (the reference a search tracked, so later edits to
the reference builder cannot change a stored result: engine.construct's rule) and compressed traces. Layout under root:
    entries.jsonl   one ArchiveEntry per line (dataclasses -> dicts; numpy -> lists)
    refs/<id>.pkl   pickled reference dicts        traces/<id>.npz  per-ms channels
query by program / scenario / representation; best_per_cell over descriptor bins; lineage through candidate.provenance["parent"].
Default root: results/md_build/archive (never results/construct*)."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json, os, pickle
import numpy as np
from movement_discovery.core.contracts import ArchiveEntry, Candidate, Outcome, SearchRecord, ValidityReport

ROOT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "results/md_build/archive")


def _plain(o: Any):
    if is_dataclass(o): return _plain(asdict(o))
    if isinstance(o, dict): return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_plain(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if hasattr(o, "value") and not isinstance(o, (int, float, str)): return o.value
    return o


class Archive:
    def __init__(self, root: str = ROOT):
        self.root = root; os.makedirs(f"{root}/refs", exist_ok=True); os.makedirs(f"{root}/traces", exist_ok=True); self.path = f"{root}/entries.jsonl"
    # ---- write
    def append(self, e: ArchiveEntry) -> str:
        with open(self.path, "a") as f: f.write(json.dumps(_plain(e), sort_keys=True) + "\n")
        return e.candidate.id
    def store_reference(self, cid: str, ref: dict) -> str:
        p = f"{self.root}/refs/{cid}.pkl"; pickle.dump(ref, open(p, "wb")); return p
    def load_reference(self, cid: str) -> Optional[dict]:
        p = f"{self.root}/refs/{cid}.pkl"; return pickle.load(open(p, "rb")) if os.path.exists(p) else None
    def store_trace(self, cid: str, trace: dict) -> str:
        p = f"{self.root}/traces/{cid}.npz"; arrs = {k: np.asarray(v) for k, v in trace.items() if isinstance(v, (list, np.ndarray))}; meta = {k: v for k, v in trace.items() if k not in arrs}
        np.savez_compressed(p, _meta=json.dumps(_plain(meta)), **arrs); return p
    def load_trace(self, cid: str) -> Optional[dict]:
        p = f"{self.root}/traces/{cid}.npz"
        if not os.path.exists(p): return None
        z = np.load(p, allow_pickle=False); out = {k: z[k] for k in z.files if k != "_meta"}; out.update(json.loads(str(z["_meta"]))); return out
    # ---- read
    def entries(self) -> Iterable[dict]:
        if not os.path.exists(self.path): return
        with open(self.path) as f:
            for line in f:
                if line.strip(): yield json.loads(line)
    def query(self, program_id: Optional[str] = None, scenario_id: Optional[str] = None, representation: Optional[str] = None, valid_only: bool = False) -> List[dict]:
        out = []
        for e in self.entries():
            if program_id and e["program_id"] != program_id: continue
            if scenario_id and e["scenario_id"] != scenario_id: continue
            if representation and e["candidate"]["representation"] != representation: continue
            if valid_only and (e["outcome"]["validity"]["reasons"] or e["outcome"]["validity"]["violation"] > 0): continue
            out.append(e)
        return out
    def best(self, program_id: str, scenario_id: Optional[str] = None, valid_only: bool = True) -> Optional[dict]:
        es = self.query(program_id, scenario_id, valid_only=valid_only); return max(es, key=lambda e: e["outcome"]["value"]) if es else None
    def best_per_cell(self, program_id: str, keys: Sequence[str], bins: int = 6) -> Dict[Tuple[int, ...], dict]:
        es = self.query(program_id, valid_only=True)
        if not es: return {}
        X = np.array([[float(e["descriptors"].get(k, 0.0)) for k in keys] for e in es]); lo, hi = X.min(0), X.max(0); cells: Dict[Tuple[int, ...], dict] = {}
        for e, x in zip(es, X):
            c = tuple(np.clip(((x - lo)/(hi - lo + 1e-9)*bins).astype(int), 0, bins - 1))
            if c not in cells or e["outcome"]["value"] > cells[c]["outcome"]["value"]: cells[c] = e
        return cells
    def lineage(self, cid: str) -> List[dict]:
        by_id = {e["candidate"]["id"]: e for e in self.entries()}; out = []; cur = by_id.get(cid)
        while cur is not None and cur not in out:
            out.append(cur); cur = by_id.get((cur["candidate"].get("provenance") or {}).get("parent"))
        return out
    def stats(self) -> dict:
        es = list(self.entries()); progs = {}
        for e in es: progs[e["program_id"]] = progs.get(e["program_id"], 0) + 1
        return dict(n=len(es), programs=progs, root=self.root)


def make_entry(program_id: str, scenario_id: str, embodiment_hash: str, world_hash: str, candidate: Candidate, outcome: Outcome, search: Optional[SearchRecord], descriptors: Dict[str, float], software: Optional[Dict[str, str]] = None) -> ArchiveEntry:
    return ArchiveEntry(program_id=program_id, scenario_id=scenario_id, embodiment_hash=embodiment_hash, world_hash=world_hash, candidate=candidate, outcome=outcome, search=search, descriptors=descriptors, software=software or _software())


def _software() -> Dict[str, str]:
    import subprocess
    try: sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(ROOT), stderr=subprocess.DEVNULL).decode().strip()
    except Exception: sha = "unknown"
    try: import mujoco; mj = mujoco.__version__
    except Exception: mj = "unknown"
    return dict(git=sha, mujoco=mj)
