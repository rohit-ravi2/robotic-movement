"""argparse front end; every command is a thin call into the packages it reports on (nothing here authors movement or search)."""
from __future__ import annotations
import argparse, json, sys, os
from movement_discovery.core import registry
from movement_discovery.core.contracts import Family


def _j(o) -> str: return json.dumps(o, indent=1, sort_keys=True, default=_plain)


def _plain(o):
    if hasattr(o, "tolist"): return o.tolist()
    if hasattr(o, "value"): return o.value
    if hasattr(o, "__dict__"): return vars(o)
    return str(o)


# ---------------------------------------------------------------- commands
def cmd_list(a):
    rows = registry.manifest()
    if a.family: rows = [r for r in rows if r["family"] == a.family]
    if registry.IMPORT_ERRORS: print("IMPORT ERRORS:", registry.IMPORT_ERRORS, file=sys.stderr)
    if a.json: print(_j(rows)); return 0
    for r in rows: print(f"{r['id']:<36} {r['family']:<18} {r['status']:<19} {r['title']}")
    print(f"{len(rows)} programs"); return 0


def cmd_inspect(a):
    from movement_discovery import world as W
    p = registry.get(a.program_id); sc = p.sample_scenario(a.seed, robot=a.robot)
    out = dict(program=p.serialize(), requires=[c.value for c in p.requires], distinct_from=p.distinct_from, scenario=dict(id=sc.id, seed=sc.seed, params=sc.params),
               goal=vars(p.goal(sc)), world=W.describe(p.world(sc)), world_required_capabilities=[[c.value for c in alt] for alt in W.required_capabilities(p.world(sc))],
               strategy_space=p.strategy_space(sc), doc=(sys.modules[type(p).__module__].__doc__ or type(p).__doc__ or "").strip())
    if a.robots:
        from movement_discovery.scenarios.generators import robot_ok
        out["robots"] = {r: ("ok" if ok else note) for r, (ok, note) in ((r, robot_ok(r, p)) for r in ("G1", "G1H", "H1"))}
    print(_j(out)); return 0


def cmd_scenarios(a):
    from movement_discovery.scenarios import generate, manifest, summary
    robots = (a.robot,) if a.robot else ("G1", "G1H", "H1")
    if a.program_id == "all":
        rows = generate(robots=robots)
        if a.write: print("wrote", manifest.write(rows), "rows to", manifest.MANIFEST_PATH)
        print(_j({k: v for k, v in summary(rows).items() if k != "by_program"})); return 0
    rows = generate([a.program_id], robots=robots)
    if a.n: rows = rows[:a.n]
    for r in rows: print(json.dumps(r, sort_keys=True, default=_plain))
    print(f"{len(rows)} scenarios for {a.program_id}", file=sys.stderr); return 0


def cmd_check(a):
    from movement_discovery import runner
    res = runner.check(a.program_id, a.seed, robot=a.robot, representation=a.representation)
    if a.json: print(_j(dict(line=res["line"], outcome=res["outcome"], n_params=res["n_params"], representation=res["representation"])))
    return 0


def cmd_run(a):
    from movement_discovery import runner
    from movement_discovery.optimization import TIERS
    budget = TIERS.get(str(a.budget), None) or int(a.budget)
    if not a.execute:
        plan = runner.run(a.program_id, a.seed, robot=a.robot, representation=a.representation, optimizer=a.optimizer, budget=budget, execute=False, seed=a.seed, pop=a.pop)
        print(_j(plan)); print("execution DISABLED (plan only). Re-run with --execute to start an optimisation campaign of", budget, "evaluations.", file=sys.stderr); return 0
    print(f"EXECUTING {a.program_id} seed {a.seed} on {a.robot}: {a.optimizer} x {budget} evaluations (archive -> results/md_build/archive)", file=sys.stderr)
    res = runner.run(a.program_id, a.seed, robot=a.robot, representation=a.representation, optimizer=a.optimizer, budget=budget, execute=True, seed=a.seed, pop=a.pop)
    print(_j({k: v for k, v in res.items() if k not in ("telemetry", "events")})); return 0


def cmd_replay(a):
    from movement_discovery.archive import Archive
    arc = Archive(a.archive) if a.archive else Archive(); es = list(arc.entries())
    hit = [e for e in es if e["candidate"]["id"] == a.key] or [e for e in es if e["scenario_id"] == a.key] or [e for e in es if e["program_id"] == a.key]
    if not hit: print(f"no archive entry matches {a.key!r} in {arc.root} ({len(es)} entries)", file=sys.stderr); return 1
    e = max(hit, key=lambda e: e["outcome"]["value"]); print(_j(dict(program_id=e["program_id"], scenario_id=e["scenario_id"], candidate=e["candidate"], outcome=e["outcome"], search=(e.get("search") or {}) | {"history": "..."} if e.get("search") else None, descriptors=e.get("descriptors"), software=e.get("software"),
                                                                 trace=arc.load_trace(e["candidate"]["id"]) is not None, reference=arc.load_reference(e["candidate"]["id"]) is not None)))
    if not a.rollout: print("stored entry only; --rollout re-runs ONE rollout of the stored candidate (no search)", file=sys.stderr); return 0
    from movement_discovery import runner
    import numpy as np
    prov = e["candidate"].get("provenance") or {}; sc = dict(prov.get("scenario_params") or {}); sc["seed"] = prov.get("seed", 0)
    asm = runner.assemble(e["program_id"], sc, robot=sc.get("robot", "G1"), representation=e["candidate"]["representation"]); rep, demand = asm["rep"], asm["demand"]
    S = rep.plant(demand); rec = rep.run(S, demand, np.asarray(e["candidate"]["params"]["z"], float), record=False)
    print(_j(dict(stored_value=e["outcome"]["value"], replay_value=rec["value"], reasons=rec["reasons"], violation=rec["violation"]))); return 0


def cmd_manifest(a):
    from .manifest_build import build
    out = build(robots=not a.no_robots, out_dir=a.out); print("wrote", out["md"], "and", out["json"], f"({out['n']} programs, {out['n_scenarios']} scenarios)"); return 0


# ---------------------------------------------------------------- parser
def parser() -> argparse.ArgumentParser:
    P = argparse.ArgumentParser(prog="python -m movement_discovery.cli", description=__doc__); S = P.add_subparsers(dest="cmd", required=True)
    s = S.add_parser("list"); s.add_argument("--family", choices=[f.value for f in Family]); s.add_argument("--json", action="store_true"); s.set_defaults(fn=cmd_list)
    s = S.add_parser("inspect"); s.add_argument("program_id"); s.add_argument("--seed", type=int, default=0); s.add_argument("--robot", default="G1"); s.add_argument("--robots", action="store_true", help="resolve G1/G1H/H1 (loads MuJoCo models)"); s.set_defaults(fn=cmd_inspect)
    s = S.add_parser("scenarios"); s.add_argument("program_id", help="program id or 'all'"); s.add_argument("--n", type=int, default=0); s.add_argument("--robot"); s.add_argument("--write", action="store_true", help="with 'all': regenerate scenarios/manifest.jsonl"); s.set_defaults(fn=cmd_scenarios)
    s = S.add_parser("check"); s.add_argument("program_id"); s.add_argument("--seed", type=int, default=0); s.add_argument("--robot", default="G1"); s.add_argument("--representation"); s.add_argument("--json", action="store_true"); s.set_defaults(fn=cmd_check)
    s = S.add_parser("run"); s.add_argument("program_id"); s.add_argument("seed", type=int); s.add_argument("--representation"); s.add_argument("--optimizer", default="cmaes"); s.add_argument("--budget", default="screen", help="evaluations or tier name: screen (2400) | emergence (38400)")
    s.add_argument("--robot", default="G1"); s.add_argument("--pop", type=int, default=16); s.add_argument("--execute", action="store_true", help="actually run the search (an optimisation campaign)"); s.set_defaults(fn=cmd_run)
    s = S.add_parser("replay"); s.add_argument("key", help="candidate id, scenario id or program id (best valid entry)"); s.add_argument("--archive"); s.add_argument("--rollout", action="store_true"); s.set_defaults(fn=cmd_replay)
    s = S.add_parser("manifest"); s.add_argument("--no-robots", action="store_true", help="skip the embodiment probe (no MuJoCo)"); s.add_argument("--out"); s.set_defaults(fn=cmd_manifest)
    return P


def main(argv=None) -> int:
    a = parser().parse_args(argv); return int(a.fn(a) or 0)
