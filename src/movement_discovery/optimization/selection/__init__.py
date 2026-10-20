"""Diversity-preserving selection over BEHAVIOUR DESCRIPTORS supplied by telemetry (contact history, joint-use distribution,
duration, support changes): novelty = mean distance to the k nearest archive members in standardised descriptor space. Selection
never looks at technique templates; it only keeps the archive spread in what the body DID. NoveltyArchive.add() keeps an entry
when it is novel enough (threshold) or better than the occupant of its descriptor cell; select() ranks candidates by
score + w * novelty (lower score is better)."""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


def _vec(desc: Dict[str, float], keys: Sequence[str]) -> np.ndarray: return np.array([float(desc.get(k, 0.0)) for k in keys])


class NoveltyArchive:
    def __init__(self, keys: Sequence[str], k: int = 5, threshold: float = 0.5, cell_bins: int = 8):
        self.keys = list(keys); self.k = int(k); self.thr = float(threshold); self.bins = int(cell_bins); self.items: List[dict] = []
    def _X(self): return np.array([it["x"] for it in self.items]) if self.items else np.zeros((0, len(self.keys)))
    def _scale(self):
        X = self._X(); return (X.std(0) + 1e-9) if len(X) > 1 else np.ones(len(self.keys))
    def novelty(self, desc: Dict[str, float]) -> float:
        X = self._X()
        if len(X) == 0: return float("inf")
        s = self._scale(); d = np.linalg.norm((X - _vec(desc, self.keys))/s, axis=1); return float(np.sort(d)[:min(self.k, len(d))].mean())
    def cell(self, desc: Dict[str, float]) -> Tuple[int, ...]:
        X = self._X(); x = _vec(desc, self.keys)
        if len(X) < 2: return tuple(np.zeros(len(self.keys), int))
        lo, hi = X.min(0), X.max(0); return tuple(np.clip(((x - lo)/(hi - lo + 1e-9)*self.bins).astype(int), 0, self.bins - 1))
    def add(self, cid: str, desc: Dict[str, float], score: float) -> bool:
        nov = self.novelty(desc); c = self.cell(desc); occ = [i for i, it in enumerate(self.items) if it["cell"] == c]
        if nov >= self.thr or not occ: self.items.append(dict(id=cid, x=_vec(desc, self.keys), score=float(score), cell=c, novelty=nov)); return True
        j = min(occ, key=lambda i: self.items[i]["score"])
        if score < self.items[j]["score"]: self.items[j] = dict(id=cid, x=_vec(desc, self.keys), score=float(score), cell=c, novelty=nov); return True
        return False
    def summary(self) -> dict: return dict(n=len(self.items), cells=len({it["cell"] for it in self.items}), keys=self.keys)


def select(candidates: Sequence[dict], k: int, keys: Sequence[str], w_novelty: float = 1.0, archive: Optional[NoveltyArchive] = None) -> List[dict]:
    """candidates: dicts with "score" (lower better) and "descriptors". Greedy: each pick maximises -score + w * novelty w.r.t. the
    archive plus the already-picked set, so the k picks cover different behaviours instead of k copies of the best."""
    arc = archive or NoveltyArchive(keys); picked: List[dict] = []; pool = list(candidates)
    tmp = NoveltyArchive(keys, k=arc.k, threshold=0.0); [tmp.add(it["id"], dict(zip(keys, it["x"])), it["score"]) for it in arc.items]
    sc = np.array([c["score"] for c in pool]); s_scale = sc.std() + 1e-9
    while pool and len(picked) < k:
        nov = np.array([tmp.novelty(c["descriptors"]) for c in pool]); nov = np.where(np.isfinite(nov), nov, nov[np.isfinite(nov)].max() if np.isfinite(nov).any() else 1.0)
        util = -(np.array([c["score"] for c in pool]) - sc.mean())/s_scale + w_novelty*nov; i = int(np.argmax(util))
        c = pool.pop(i); picked.append(c); tmp.add(c.get("id", str(len(picked))), c["descriptors"], c["score"])
    return picked
