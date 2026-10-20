"""Warm-start retrieval (OPTIONAL, never mandatory): the nearest archived VALID solution of the same program and representation
by scenario-parameter distance (numeric params standardised over the archive; non-numeric must match). Returns the stored
z and its entry, or None. A warm start is only a candidate start point: engine.task.optimize keeps z0 as a candidate and the
search may leave it; nothing is copied into the objective."""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
from . import Archive


def _num(params: Dict) -> Dict[str, float]:
    out = {}
    for k, v in params.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool): out[k] = float(v)
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v):
            for i, x in enumerate(v): out[f"{k}[{i}]"] = float(x)
    return out


def nearest(archive: Archive, program_id: str, params: Dict, representation: str, exclude_scenario: Optional[str] = None, max_dist: float = 2.0) -> Optional[Tuple[dict, float]]:
    es = [e for e in archive.query(program_id, representation=representation, valid_only=True) if e["scenario_id"] != exclude_scenario and e["candidate"]["params"].get("z") is not None]
    if not es: return None
    q = _num(params); keys = sorted(q); P = []
    for e in es:
        sp = e["candidate"]["provenance"].get("scenario_params") or {}; pn = _num(sp)
        if any(k not in pn for k in keys) or any(sp.get(k) != params.get(k) for k in params if k not in q and k in sp): P.append(None); continue
        P.append([pn[k] for k in keys])
    ok = [i for i, p in enumerate(P) if p is not None]
    if not ok: return None
    X = np.array([P[i] for i in ok], float); x = np.array([q[k] for k in keys]); s = X.std(0) + 1e-9 if len(X) > 1 else np.ones(len(keys))
    d = np.linalg.norm((X - x)/s, axis=1); j = int(np.argmin(d))
    return (es[ok[j]], float(d[j])) if d[j] <= max_dist else None


def warm_start(archive: Archive, program_id: str, params: Dict, representation: str, exclude_scenario: Optional[str] = None) -> Optional[Dict]:
    r = nearest(archive, program_id, params, representation, exclude_scenario)
    if r is None: return None
    e, d = r; return dict(z=np.asarray(e["candidate"]["params"]["z"], float), parent=e["candidate"]["id"], distance=d, scenario_id=e["scenario_id"], value=e["outcome"]["value"])
