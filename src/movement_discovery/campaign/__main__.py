"""python -m movement_discovery.campaign <command> (from run2/src):
    plan <phase> [--replace | --append-skipped | --requeue-failed [--include-wallcap] | --append --representation R --programs P57,...]      generate the phase's jobs -> results/md_campaign/<phase>/queue.jsonl (merge: existing states kept;
                                                                     --append-skipped: on a LIVE queue, requeue only the skipped jobs that became runnable)
    run <phase> [--workers 14] [--max-hours H] [--wall-cap-min M]     claim and execute jobs, one subprocess per job (runner.run execute=True)
    status [<phase>]                                                 queue counts per phase, running jobs with elapsed time
    report <phase>                                                   -> results/md_campaign/<phase>/REPORT.md
    ledger                                                           append new rows to results/md_campaign/LEDGER.md
Hidden (smoke only): `run ... --budget-override N --limit K` executes K jobs at N evaluations under results/md_campaign/_smoke/<phase>/."""
from __future__ import annotations
import argparse, json, os, sys, time
from . import PHASES, ROOT, phase_dir
from . import queue as Q


def cmd_plan(a):
    from .phases import generate, plan_search_available
    if a.requeue_failed:
        r = Q.requeue_failed(phase_dir(a.phase, a.smoke), include_wallcap=a.include_wallcap); print(f"[plan {a.phase}] requeued {len(r['requeued'])} failed jobs with a code error{' or wall cap' if a.include_wallcap else ''}; left {len(r['left'])} failed (wall cap / process gone / no error)")
        for k, v in r["errors"].items(): print(f"  {k}: {v}")
        return 0
    if a.append:
        from .phases import gen_append
        assert a.representation and a.programs, "--append needs --representation and --programs"
        jobs = gen_append(a.phase, a.programs.split(","), a.representation); pdir = phase_dir(a.phase, a.smoke); c = Q.append_jobs(pdir, [j for j in jobs if j.state == "queued"])
        print(f"[plan {a.phase} --append {a.representation}] {len(jobs)} generated, {sum(j.state == 'skipped' for j in jobs)} skipped, appended {c['added']} (already present {c['already']}) -> {Q.queue_path(pdir)}")
        for j in jobs:
            if j.state == "skipped": print(f"  skipped {j.id}: {j.reason[:160]}")
        return 0
    jobs = generate(a.phase, smoke=a.smoke, programs=(a.programs.split(",") if a.programs else None)); pdir = phase_dir(a.phase, a.smoke)
    c = Q.append_skipped(pdir, jobs) if a.append_skipped else Q.write_jobs(pdir, jobs, replace=a.replace)
    sk = {}; 
    for j in jobs:
        if j.state == "skipped": sk[j.reason.split(":")[0][:70]] = sk.get(j.reason.split(":")[0][:70], 0) + 1
    print(f"[plan {a.phase}] {len(jobs)} jobs generated ({sum(1 for j in jobs if j.state == 'queued')} queued, {sum(1 for j in jobs if j.state == 'skipped')} skipped; {len({j.program_id for j in jobs if j.state == 'queued'})} programs runnable); queue {c} -> {Q.queue_path(pdir)}; sequence search available: {plan_search_available()}")
    for k, v in sorted(sk.items(), key=lambda kv: -kv[1]): print(f"  skipped x{v}: {k}")
    return 0


def cmd_run(a):
    from .worker import run_phase
    s = run_phase(a.phase, workers=a.workers, max_hours=a.max_hours, budget_override=a.budget_override, limit=a.limit, wall_cap_min=a.wall_cap_min)
    print(json.dumps(s, indent=1)); return 0


def cmd_exec(a):
    from .worker import exec_job, result_line
    pdir = phase_dir(a.phase, a.smoke); j = Q.get(pdir, a.job_id)
    if j is None: print(f"no job {a.job_id} in {pdir}", file=sys.stderr); return 2
    res = exec_job(pdir, j, budget_override=a.budget_override); print(result_line(res), flush=True); return 0 if res.get("error") is None else 1


def cmd_status(a):
    for ph in ([a.phase] if a.phase else PHASES):
        for smoke in (False, True):
            d = os.path.join(ROOT, "_smoke", ph) if smoke else os.path.join(ROOT, ph)
            if not os.path.exists(Q.queue_path(d)): continue
            jobs = Q.load(d); c = {s: sum(1 for j in jobs if j.state == s) for s in ("queued", "running", "done", "failed", "skipped")}; passed = sum(1 for j in jobs if j.passed)
            print(f"{ph}{' (smoke)' if smoke else ''}: {c} passed {passed} | CPU {sum(j.secs for j in jobs)/3600:.2f} h")
            for j in jobs:
                if j.state == "running": print(f"   running {j.id} pid {j.pid} {(time.time() - j.started_at)/60:.1f} min")
    return 0


def cmd_report(a):
    from .report import write_report
    print("wrote", write_report(a.phase, a.smoke)); return 0


def cmd_ledger(a):
    from .report import append_ledger, LEDGER
    print(f"appended {append_ledger()} rows -> {LEDGER}"); return 0


def parser():
    P = argparse.ArgumentParser(prog="python -m movement_discovery.campaign", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); S = P.add_subparsers(dest="cmd", required=True)
    s = S.add_parser("plan"); s.add_argument("phase", choices=PHASES); s.add_argument("--replace", action="store_true", help="rebuild the queue (states lost)"); s.add_argument("--requeue-failed", action="store_true", help="LIVE queue: failed jobs whose result/log shows a code ERROR (traceback) go back to queued; their files are kept as *.failed-<n>.json/.log"); s.add_argument("--include-wallcap", action="store_true", help="with --requeue-failed: also requeue WALL_CLOCK_CAP failures"); s.add_argument("--append-skipped", action="store_true", help="LIVE queue: re-evaluate only skipped jobs and requeue the now-runnable ones under the lock"); s.add_argument("--programs", help="comma-separated program ids or numbers (P57,...)"); s.add_argument("--append", action="store_true", help="LIVE queue: append jobs for --representation on --programs (nominal scenario, the phase's seeds); existing jobs untouched"); s.add_argument("--representation", help="with --append: the construction name (must be in the program's strategy_space)"); s.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS); s.set_defaults(fn=cmd_plan)
    s = S.add_parser("run"); s.add_argument("phase", choices=PHASES); s.add_argument("--workers", type=int, default=14); s.add_argument("--max-hours", type=float); s.add_argument("--wall-cap-min", type=float, help="per-job wall-clock cap BASE (default 90 min screen tier, 480 emergence); multi-step Plans and servo representations get 4x")
    s.add_argument("--budget-override", type=int, help=argparse.SUPPRESS); s.add_argument("--limit", type=int, help=argparse.SUPPRESS); s.set_defaults(fn=cmd_run)
    s = S.add_parser("_exec"); s.add_argument("phase", choices=PHASES); s.add_argument("job_id"); s.add_argument("--smoke", action="store_true"); s.add_argument("--budget-override", type=int); s.set_defaults(fn=cmd_exec)
    s = S.add_parser("status"); s.add_argument("phase", nargs="?", choices=PHASES); s.set_defaults(fn=cmd_status)
    s = S.add_parser("report"); s.add_argument("phase", choices=PHASES); s.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS); s.set_defaults(fn=cmd_report)
    s = S.add_parser("ledger"); s.set_defaults(fn=cmd_ledger)
    return P


def main(argv=None) -> int:
    a = parser().parse_args(argv); return int(a.fn(a) or 0)


if __name__ == "__main__": sys.exit(main())
