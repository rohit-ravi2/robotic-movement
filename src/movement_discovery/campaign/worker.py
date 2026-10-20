"""Worker: `_exec` runs ONE job inside its own process (runner.run execute=True, the phase's Archive) and writes <job>.json;
`run_phase` is the driver: claims jobs under the queue lock, launches one `nice` subprocess per job (a crash cannot take the
driver down), enforces a per-job wall-clock cap and a campaign --max-hours, and records done/failed with the reason."""
from __future__ import annotations
from typing import List, Optional
import json, os, signal, subprocess, sys, time, traceback, socket
from . import phase_dir, RUN2
from . import queue as Q
from .rules import judge, regime_of

WALL_CAP_MIN = dict(screen=90.0, emergence=480.0)          # per-job wall-clock cap by budget tier (minutes); --wall-cap-min overrides the base
SLOW_FACTOR = 4.0                                            # multi-step Plans (sequence search) and servo representations (long rollouts) get 4x
PY = sys.executable


def job_cap_s(job: Q.Job, override_min: Optional[float] = None) -> float:
    """Per-job wall-clock cap in seconds, computed at LAUNCH from the job: tier base (or --wall-cap-min) x 4 for multi-step Plans or
    servo representations (screen 360 / emergence 1920 min), x 1 for single-step torque jobs (90 / 480). Code is read once per driver
    process: a changed rule needs a driver restart, a changed job does not."""
    base = float(override_min) if override_min else WALL_CAP_MIN.get(job.budget_tier, 90.0)
    slow = (job.n_steps or 1) > 1 or job.representation.startswith("servo") or "sequence" in job.representation
    return 60.0*base*(SLOW_FACTOR if slow else 1.0)


def _safe(o, depth: int = 0):
    """JSON-safe, size-bounded view of a runner result (numpy -> lists, long arrays summarised, unknown objects -> type name)."""
    from movement_discovery.archive import _plain
    o = _plain(o)
    if isinstance(o, dict): return {str(k): _safe(v, depth + 1) for k, v in o.items() if k not in ("trace", "trace_q", "history", "demand", "land_trace")}
    if isinstance(o, (list, tuple)):
        if len(o) > 64 and all(isinstance(x, (int, float, list, tuple)) for x in o[:8]): return f"<{len(o)} items>"
        return [_safe(v, depth + 1) for v in o]
    if isinstance(o, (int, float, str, bool)) or o is None:
        if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))): return str(o)
        return o
    return f"<{type(o).__name__}>"


def _capacity_note(pid: str, sc, robot: str, rep: str) -> Optional[list]:
    """Which capacity keys each step's task consumes (so an inert counterfactual arm is visible, never silent)."""
    try:
        from movement_discovery import runner
        asm = runner.assemble(pid, sc, robot, rep); return [st["capacity"] for st in asm["steps"]]
    except Exception as e: return [dict(error=f"{type(e).__name__}: {str(e)[:120]}")]


def exec_job(pdir: str, job: Q.Job, budget_override: Optional[int] = None) -> dict:
    """The one-job body (runs in the subprocess). Writes <pdir>/<job>.json; returns the result dict."""
    from movement_discovery import runner
    from movement_discovery.archive import Archive
    from .phases import to_scenario
    t0 = time.time(); budget = int(budget_override or job.budget); sc = to_scenario(job)
    res = dict(job=job.id, phase=job.phase, program_id=job.program_id, family=job.family, regime=regime_of(job.program_id), scenario_id=job.scenario_id, robot=job.robot, representation=job.representation,
               budget_tier=job.budget_tier, budget=budget, budget_override=bool(budget_override), seed=job.seed, scenario_seed=job.scenario_seed, scenario_params=_safe(job.scenario_params), arm=job.arm,
               n_steps=job.n_steps, missing=list(job.missing), host=socket.gethostname(), started=t0, error=None)
    try:
        if job.arm: res["capacity"] = _capacity_note(job.program_id, sc, job.robot, job.representation)
        r = runner.run(job.program_id, sc, robot=job.robot, representation=job.representation, optimizer="cmaes", budget=budget, execute=True, seed=job.seed, archive=Archive(os.path.join(pdir, "archive")), robust=(job.budget_tier != "screen"))   # screen tier: single-timestep search, two-timestep check at the end (5x cheaper on long-horizon tasks)
        o = r["outcome"]; m = r.get("measured") or {}
        res.update(outcome=_safe(dict(value=o.value, achieved=o.achieved, measures=o.measures, validity=o.validity, trace_ref=o.trace_ref)), mechanics=_safe(o.mechanics), measured=_safe(dict(ok_all=m.get("ok_all"), meas=m.get("meas"), value=m.get("value"), score=m.get("score"), evals=m.get("budget", m.get("evals")), secs=m.get("secs"), z_hash=m.get("z_hash"))),
                   candidate_id=r.get("candidate_id"), sequence=bool(r.get("step_entries") is not None or (r.get("n_steps") or 1) > 1), step_entries=r.get("step_entries"), step_budgets=r.get("budgets"), step_searches=_safe(r.get("searches")), descriptors=_safe(r.get("descriptors")), events=_safe(r.get("events")), embodiment_hash=r.get("embodiment_hash"), world_hash=r.get("world_hash"), n_params=r.get("n_params"), runner_status=r.get("status"), archive=os.path.join(pdir, "archive", "entries.jsonl"))
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"; res["traceback"] = traceback.format_exc()[-4000:]
    res["secs"] = time.time() - t0; res["pass"], res["reasons"] = judge(res)
    res["valid_both_dt"] = None if res.get("sequence") else (bool(((res.get("measured") or {}).get("ok_all")) is True) or (bool((res.get("measured") or {}).get("meas")) and not any(x.startswith("dt") for x in res["reasons"])))
    tmp = Q.result_path(pdir, job.id) + ".tmp"
    with open(tmp, "w") as f: json.dump(res, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, Q.result_path(pdir, job.id)); return res


def result_line(res: dict) -> str:
    o = res.get("outcome") or {}; v = o.get("value")
    return (f"[{res['phase']} {res['job']}] " + (f"ERROR {res['error']}" if res.get("error") else f"value {v:.3f} achieved {o.get('achieved')} valid_both_dt {res.get('valid_both_dt')}") + f" pass {res.get('pass')} reasons {res.get('reasons')} evals {res.get('budget')} | {res.get('secs', 0):.0f} s")


# ---------------------------------------------------------------- driver
def run_phase(phase: str, workers: int = 14, max_hours: Optional[float] = None, budget_override: Optional[int] = None, limit: Optional[int] = None, wall_cap_min: Optional[float] = None, poll: float = 2.0, nice: int = 10) -> dict:
    from .phases import generate
    smoke = budget_override is not None; pdir = phase_dir(phase, smoke); workers = max(1, min(int(workers), (os.cpu_count() or 2) - 1))
    if not os.path.exists(Q.queue_path(pdir)):
        jobs = generate(phase, smoke=smoke); Q.write_jobs(pdir, jobs); print(f"[driver] no queue in {pdir}: planned {len(jobs)} jobs", flush=True)
    cap_of = (lambda j: min(job_cap_s(j, wall_cap_min), 60.0*(wall_cap_min or 20.0))) if smoke else (lambda j: job_cap_s(j, wall_cap_min))
    rec = Q.recover(pdir, cap_of)
    if rec: print(f"[driver] recovered {len(rec)} running jobs: {rec}", flush=True)
    t_start = time.time(); deadline = t_start + 3600.0*max_hours if max_hours else None; running = []; launched = 0; done = failed = 0; me = f"{socket.gethostname()}:{os.getpid()}"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    stop = {"flag": False}
    def _sig(signum, frame): stop["flag"] = True; print(f"[driver] signal {signum}: stop claiming, terminating {len(running)} running jobs", flush=True)
    signal.signal(signal.SIGINT, _sig); signal.signal(signal.SIGTERM, _sig)
    print(f"[driver] phase {phase}{' SMOKE' if smoke else ''} dir {pdir} workers {workers} wall-cap per job (base {wall_cap_min or WALL_CAP_MIN} min, x{SLOW_FACTOR:.0f} multi-step/servo) max-hours {max_hours} budget-override {budget_override} limit {limit} queue {Q.counts(pdir)}", flush=True)
    while True:
        while not stop["flag"] and len(running) < workers and (limit is None or launched < limit) and (deadline is None or time.time() < deadline):
            j = Q.claim(pdir, me)
            if j is None: break
            log = open(Q.log_path(pdir, j.id), "w"); cmd = ["nice", "-n", str(nice), PY, "-m", "movement_discovery.campaign", "_exec", phase, j.id] + (["--smoke"] if smoke else []) + ([f"--budget-override={budget_override}"] if budget_override else [])
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=os.path.join(RUN2, "src"), env=env, start_new_session=True)
            cap_s = cap_of(j); Q.update(pdir, j.id, pid=p.pid); running.append((j, p, time.time(), log, cap_s)); launched += 1
            print(f"[driver] launched {j.id} pid {p.pid} cap {cap_s/60:.0f} min ({launched} launched, {len(running)} running)", flush=True)
        for r in list(running):
            j, p, t0, log, cap_s = r; rc = p.poll(); over = (time.time() - t0) > cap_s
            if rc is None and not over and not stop["flag"]: continue
            if rc is None:
                try: os.killpg(p.pid, signal.SIGTERM); p.wait(timeout=15)
                except Exception:
                    try: os.killpg(p.pid, signal.SIGKILL)
                    except Exception: pass
                rc = p.poll()
            log.close(); running.remove(r)
            if stop["flag"] and rc is not None and not os.path.exists(Q.result_path(pdir, j.id)):
                Q.update(pdir, j.id, state="queued", pid=0, worker="", reason="driver stopped (signal); requeued"); print(f"[driver] requeued {j.id}", flush=True); continue
            if over and not os.path.exists(Q.result_path(pdir, j.id)):
                Q.update(pdir, j.id, state="failed", pid=0, finished_at=time.time(), secs=time.time() - t0, reason=f"WALL_CLOCK_CAP {cap_s/60:.0f} min"); failed += 1; print(f"[driver] FAILED (wall cap) {j.id}", flush=True); continue
            rp = Q.result_path(pdir, j.id)
            if os.path.exists(rp):
                res = json.load(open(rp)); ok = res.get("error") is None
                Q.update(pdir, j.id, state=("done" if ok else "failed"), pid=0, finished_at=time.time(), secs=res.get("secs", time.time() - t0), passed=bool(res.get("pass")), value=((res.get("outcome") or {}).get("value")), reason=("" if ok else res["error"][:300]))
                done += ok; failed += (not ok); print(result_line(res), flush=True)
            else:
                tail = ""
                try: tail = open(Q.log_path(pdir, j.id)).read()[-600:].strip().splitlines()[-1:]
                except Exception: pass
                Q.update(pdir, j.id, state="failed", pid=0, finished_at=time.time(), secs=time.time() - t0, reason=f"rc {rc}; no result file; {tail}"); failed += 1; print(f"[driver] FAILED rc {rc} {j.id} {tail}", flush=True)
        if not running and (stop["flag"] or (limit is not None and launched >= limit) or (deadline is not None and time.time() >= deadline) or Q.counts(pdir)["queued"] == 0): break
        time.sleep(poll)
    c = Q.counts(pdir); summ = dict(phase=phase, dir=pdir, launched=launched, done=done, failed=failed, hours=(time.time() - t_start)/3600.0, queue=c)
    print(f"[driver] end {summ}", flush=True); return summ
