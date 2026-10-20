"""JSONL job queue with an fcntl-locked, atomic claim. One file per phase (queue.jsonl + queue.lock next to it); every state change
rewrites the file under the lock (temp + rename), so a killed driver or worker leaves a consistent queue and `recover` puts the
jobs whose process is gone back to `queued`. A job is a plain dict (Job.to_dict) so the file is greppable."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import fcntl, json, os, re, time, contextlib
from . import STATES

MAX_ATTEMPTS = 2
REP_SHORT = {"torque_knots+tracked_prior": "tkp", "torque_knots": "tk", "servo_knots": "sk", "contact_schedule+knots": "csk", "torque_knots+brace_prior": "tkb", "servo_knots+brace_prior": "skb"}


def rep_short(rep: str) -> str: return REP_SHORT.get(rep) or re.sub(r"[^A-Za-z0-9]+", "", rep)[:12]


@dataclass
class Job:
    phase: str; program_id: str; scenario_id: str; robot: str; representation: str; budget_tier: str; budget: int; seed: int
    scenario_seed: int = 0; scenario_params: Dict = field(default_factory=dict); arm: Optional[str] = None
    family: str = ""; regime: str = ""; n_steps: Optional[int] = None; missing: List[dict] = field(default_factory=list)
    state: str = "queued"; reason: str = ""; worker: str = ""; pid: int = 0; attempts: int = 0
    queued_at: float = 0.0; started_at: float = 0.0; finished_at: float = 0.0; secs: float = 0.0; passed: Optional[bool] = None; value: Optional[float] = None
    id: str = ""
    def __post_init__(self):
        if not self.id: self.id = f"{self.scenario_id}_{rep_short(self.representation)}_{self.budget_tier}_s{self.seed}" + (f"_{self.arm}" if self.arm else "")
        if not self.queued_at: self.queued_at = time.time()
    def to_dict(self) -> dict: return asdict(self)
    @staticmethod
    def from_dict(d: dict) -> "Job": return Job(**d)


def queue_path(pdir: str) -> str: return os.path.join(pdir, "queue.jsonl")


@contextlib.contextmanager
def locked(pdir: str):
    """Exclusive fcntl lock on <pdir>/queue.lock for the duration of the block."""
    os.makedirs(pdir, exist_ok=True); lp = os.path.join(pdir, "queue.lock")
    with open(lp, "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _read(pdir: str) -> List[Job]:
    p = queue_path(pdir)
    if not os.path.exists(p): return []
    with open(p) as f: return [Job.from_dict(json.loads(l)) for l in f if l.strip()]


def _write(pdir: str, jobs: List[Job]) -> None:
    p = queue_path(pdir); tmp = p + ".tmp"
    with open(tmp, "w") as f:
        for j in jobs: f.write(json.dumps(j.to_dict(), sort_keys=True, default=str) + "\n")
    os.replace(tmp, p)


def load(pdir: str) -> List[Job]:
    with locked(pdir): return _read(pdir)


def write_jobs(pdir: str, jobs: List[Job], replace: bool = False) -> dict:
    """Merge generated jobs into the queue: an existing job keeps its state (resumable re-plan); a new one is added; with replace=True
    the queue is rebuilt from scratch (states lost). Returns counts."""
    with locked(pdir):
        old = {} if replace else {j.id: j for j in _read(pdir)}; out = []; added = 0; kept = 0
        for j in jobs:
            if j.id in old: out.append(old[j.id]); kept += 1
            else: out.append(j); added += 1
        ids = {j.id for j in jobs}; stale = [j for j in old.values() if j.id not in ids]; out += stale      # jobs no longer generated stay (recorded), never silently dropped
        _write(pdir, out)
    return dict(n=len(out), added=added, kept=kept, stale=len(stale))


def append_skipped(pdir: str, jobs: List[Job]) -> dict:
    """Re-evaluate ONLY the queue's skipped jobs against freshly generated ones: a skipped job whose regenerated twin is now queued is
    replaced (state queued) under the lock; every other job is untouched (safe on a live queue: running workers claim under the same
    lock). Returns counts."""
    gen = {j.id: j for j in jobs}; requeued = []; still = 0
    with locked(pdir):
        cur = _read(pdir); out = []
        for j in cur:
            g = gen.get(j.id)
            if j.state == "skipped" and g is not None and g.state == "queued": g.reason = f"requeued (was skipped: {j.reason[:80]})"; out.append(g); requeued.append(j.id)
            else:
                out.append(j); still += (j.state == "skipped")
        have = {j.id for j in out}; added = [g for g in jobs if g.id not in have]; out += added      # brand-new jobs (e.g. a new program) are appended too
        _write(pdir, out)
    return dict(requeued=len(requeued), still_skipped=still, added=len(added), ids=requeued)


def _is_code_error(pdir: str, j: "Job") -> Optional[str]:
    """A failed job counts as a CODE error when its result file carries an exception (error/traceback) or, with no result file, its log
    ends in a Python traceback / the driver recorded a non-zero rc. Physics failures (pass False with gate reasons) are `done`, never
    here; wall-cap and recover failures are left alone. Returns the one-line error or None."""
    rp = result_path(pdir, j.id)
    if os.path.exists(rp):
        try:
            with open(rp) as f: r = json.load(f)
        except Exception as e: return f"unreadable result: {e}"
        if str(r.get("error") or "").startswith("UnsupportedCapability"): return None   # a declared capability refusal (e.g. P75 on G1), not a code error
        return (r.get("error") or None) if (r.get("error") or r.get("traceback")) else None
    lp = log_path(pdir, j.id); tail = ""
    if os.path.exists(lp):
        with open(lp, errors="replace") as f: tail = f.read()[-3000:]
    if "Traceback (most recent call last)" in tail: return "traceback in log: " + (tail.strip().splitlines() or ["?"])[-1][:160]
    if j.reason.startswith("rc ") and "no result file" in j.reason: return j.reason[:160]
    return None


def _next_failed_index(pdir: str, job_id: str) -> int:
    n = 1
    while os.path.exists(os.path.join(pdir, f"{job_id}.failed-{n}.json")) or os.path.exists(os.path.join(pdir, f"{job_id}.failed-{n}.log")): n += 1
    return n


def requeue_failed(pdir: str, include_wallcap: bool = False) -> dict:
    """Under the lock: every `failed` job whose failure is a CODE error (see _is_code_error) goes back to `queued` (attempts reset);
    its result/log are renamed <job>.failed-<n>.json/.log for the record. Other failed jobs (wall cap, process gone) stay unless
    include_wallcap=True, which also requeues WALL_CLOCK_CAP failures (e.g. after the per-job cap rule changed)."""
    out = dict(requeued=[], left=[], errors={})
    with locked(pdir):
        jobs = _read(pdir)
        for j in jobs:
            if j.state != "failed": continue
            err = _is_code_error(pdir, j)
            if err is None and include_wallcap and j.reason.startswith("WALL_CLOCK_CAP"): err = j.reason
            if err is None: out["left"].append(j.id); continue
            n = _next_failed_index(pdir, j.id)
            for ext in ("json", "log"):
                src = os.path.join(pdir, f"{j.id}.{ext}")
                if os.path.exists(src): os.replace(src, os.path.join(pdir, f"{j.id}.failed-{n}.{ext}"))
            j.state = "queued"; j.attempts = 0; j.pid = 0; j.worker = ""; j.passed = None; j.value = None; j.finished_at = 0.0; j.secs = 0.0; j.reason = f"requeued after code error (failed-{n}): {err[:120]}"
            out["requeued"].append(j.id); out["errors"][j.id] = err[:200]
        _write(pdir, jobs)
    return out


def append_jobs(pdir: str, jobs: List[Job]) -> dict:
    """Under the lock: append the jobs whose ids are not in the queue; nothing existing is touched (live-safe)."""
    with locked(pdir):
        cur = _read(pdir); have = {j.id for j in cur}; new = [j for j in jobs if j.id not in have]; _write(pdir, cur + new)
    return dict(added=len(new), already=len(jobs) - len(new), ids=[j.id for j in new])


def counts(pdir: str) -> Dict[str, int]:
    c = {s: 0 for s in STATES}
    for j in load(pdir): c[j.state] = c.get(j.state, 0) + 1
    return c


def _alive(pid: int) -> bool:
    if not pid: return False
    try: os.kill(pid, 0); return True
    except ProcessLookupError: return False
    except PermissionError: return True


def recover(pdir: str, wall_cap_s=None) -> List[str]:
    """Jobs marked running whose worker process is gone (driver killed) go back to queued (or failed past MAX_ATTEMPTS). Returns the ids.
    wall_cap_s: seconds, or a callable job -> seconds (per-job caps)."""
    out = []
    with locked(pdir):
        jobs = _read(pdir)
        for j in jobs:
            if j.state != "running": continue
            cap = wall_cap_s(j) if callable(wall_cap_s) else wall_cap_s
            dead = not _alive(j.pid); stale = cap is not None and j.started_at and (time.time() - j.started_at) > cap
            if not (dead or stale): continue
            if stale and not dead:
                with contextlib.suppress(Exception): os.kill(j.pid, 15)
            j.pid = 0; j.worker = ""                                   # the claim already counted this attempt
            if j.attempts >= MAX_ATTEMPTS: j.state = "failed"; j.reason = f"recover: process {'gone' if dead else 'over wall cap'} after {j.attempts} attempts"
            else: j.state = "queued"; j.reason = f"recovered ({'process gone' if dead else 'over wall cap'})"
            out.append(j.id)
        _write(pdir, jobs)
    return out


def claim(pdir: str, worker: str, pid: int = 0, limit_ids: Optional[set] = None) -> Optional[Job]:
    """Atomically take the first queued job (optionally restricted to limit_ids) and mark it running."""
    with locked(pdir):
        jobs = _read(pdir)
        for j in jobs:
            if j.state != "queued" or (limit_ids is not None and j.id not in limit_ids): continue
            j.state = "running"; j.worker = worker; j.pid = int(pid); j.started_at = time.time(); j.attempts += 1; j.reason = ""
            _write(pdir, jobs); return j
    return None


def update(pdir: str, job_id: str, **fields) -> Optional[Job]:
    with locked(pdir):
        jobs = _read(pdir); hit = None
        for j in jobs:
            if j.id == job_id:
                for k, v in fields.items(): setattr(j, k, v)
                hit = j
        _write(pdir, jobs); return hit


def get(pdir: str, job_id: str) -> Optional[Job]:
    return next((j for j in load(pdir) if j.id == job_id), None)


def result_path(pdir: str, job_id: str) -> str: return os.path.join(pdir, f"{job_id}.json")
def log_path(pdir: str, job_id: str) -> str: return os.path.join(pdir, f"{job_id}.log")


def results(pdir: str) -> List[dict]:
    """Every <job>.json in the phase directory (the on-disk truth the next phase reads)."""
    out = []
    for fn in sorted(os.listdir(pdir)) if os.path.isdir(pdir) else []:
        if fn.endswith(".json") and fn != "queue.jsonl" and not re.search(r"\.failed-\d+\.json$", fn):      # *.failed-<n>.json are kept for the record, not results
            try:
                with open(os.path.join(pdir, fn)) as f: out.append(json.load(f))
            except Exception: continue
    return out
