"""MANIFEST.md / MANIFEST.json: one row per program from registry.manifest() + required capabilities, which robots resolve
(embodiment.resolve in try/except; the model probe loads MuJoCo once per robot), implementation file, scenario count (from
scenarios/manifest.jsonl), status, the "integration check: ..." line of the module docstring (fallback: the class docstring, then
the family's results/md_build/checks_<family>.json record), known limitations (docstring sentences carrying limitation markers),
and a dependency / unsupported-capability report. Regenerate: `python -m movement_discovery.cli manifest`."""
from __future__ import annotations
from typing import Dict, List, Optional
import collections, json, os, re, sys, datetime
from movement_discovery.core import registry
from movement_discovery.core.contracts import TaskProgram
from movement_discovery.scenarios import manifest as SM

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); SRC = os.path.dirname(PKG); RUN2 = os.path.dirname(SRC)
ROBOTS = ("G1", "G1H", "H1")
LIMIT_WORDS = ("simplification", "labelled", "labeled", "OPEN", "missing asset", "not yet", "cannot", "no collision", "geometric", "fallback", "experimental", "needs ", "NOT ", "unmodelled", "today", "until ", "blocked", "scaffold")


def _doc(prog) -> str:
    mod = sys.modules.get(type(prog).__module__); return " ".join(((mod.__doc__ if mod else "") or "").split()) or " ".join((type(prog).__doc__ or "").split())


def integration_line(prog) -> str:
    d = _doc(prog) or " ".join((type(prog).__doc__ or "").split()); m = re.search(r"integration check:\s*(.+)$", d, flags=re.I)
    if m: return m.group(1).strip()[:400]
    fam = prog.family.value; p = f"{RUN2}/results/md_build/checks_{fam}.json"
    if os.path.exists(p):
        try:
            r = json.load(open(p)).get(prog.id)
            if r: return f"(from results/md_build/checks_{fam}.json) value {r.get('value')} achieved {r.get('achieved')} reasons {r.get('reasons')}"[:400]
        except Exception: pass
    return "none recorded"


def limitations(prog) -> str:
    d = _doc(prog); d = re.split(r"integration check:", d, flags=re.I)[0]
    sents = [s.strip() for s in re.split(r"(?<=[.;])\s+", d) if s.strip()]
    keep = [s for s in sents if any(w in s for w in LIMIT_WORDS)]
    return (" | ".join(keep))[:600] if keep else ""


def robots_of(prog, probe: bool) -> Dict[str, str]:
    if not probe: return {r: "not probed" for r in ROBOTS}
    from movement_discovery.scenarios.generators import robot_ok
    return {r: ("ok" if ok else note) for r, (ok, note) in ((r, robot_ok(r, prog)) for r in ROBOTS)}


def rows(probe: bool = True) -> List[dict]:
    registry.load_all(); sc = SM.read(); nsc = collections.Counter(r["program_id"] for r in sc); nsc_robot = collections.defaultdict(collections.Counter)
    for r in sc: nsc_robot[r["program_id"]][r["robot"]] += 1
    out = []
    for m in registry.manifest():
        p = registry.get(m["id"]); s0 = p.sample_scenario(0); mod = sys.modules[type(p).__module__]
        try: ss = p.strategy_space(s0)
        except Exception as e: ss = [f"ERR {type(e).__name__}"]
        try:
            from movement_discovery import world as W; wd = W.describe(p.world(s0))
        except Exception as e: wd = f"ERR {type(e).__name__}: {str(e)[:80]}"
        out.append(dict(id=m["id"], family=m["family"], title=m["title"], status=m["status"], requires=m["requires"], distinct_from=m["distinct_from"], robots=robots_of(p, probe),
                        file=os.path.relpath(mod.__file__, SRC), scenarios=int(nsc.get(m["id"], 0)), scenarios_by_robot=dict(nsc_robot.get(m["id"], {})), strategy_space=ss, world=wd,
                        integration_check=integration_line(p), limitations=limitations(p), has_demand=(type(p).demand is not TaskProgram.demand), has_run=hasattr(p, "run")))
    return out


def unsupported_report(rs: List[dict]) -> dict:
    by_robot = {r: {} for r in ROBOTS}; by_cap = collections.Counter()
    for row in rs:
        for r, note in row["robots"].items():
            if note not in ("ok", "not probed"):
                by_robot[r][row["id"]] = note
                for c in note.replace("missing ", "").split(","): by_cap[(r, c.strip())] += 1
    return dict(by_robot=by_robot, counts={r: len(v) for r, v in by_robot.items()}, by_capability={f"{r}:{c}": n for (r, c), n in sorted(by_cap.items())})


def build(robots: bool = True, out_dir: Optional[str] = None) -> dict:
    rs = rows(probe=robots); rep = unsupported_report(rs); out_dir = out_dir or PKG; sc = SM.read()
    by_status = collections.Counter(r["status"] for r in rs); by_family = collections.Counter(r["family"] for r in rs)
    meta = dict(generated=datetime.date.today().isoformat(), n_programs=len(rs), n_scenarios=len(sc), by_status=dict(by_status), by_family=dict(by_family), import_errors=dict(registry.IMPORT_ERRORS), scenario_summary={k: v for k, v in SM.summary(sc).items() if k != "by_program"})
    json.dump(dict(meta=meta, programs=rs, unsupported=rep), open(f"{out_dir}/MANIFEST.json", "w"), indent=1, sort_keys=True)
    L = [f"# movement_discovery MANIFEST ({meta['generated']})", "", f"{len(rs)} programs, {len(sc)} scenarios (status defined, none executed). Statuses: " + ", ".join(f"{k} {v}" for k, v in sorted(by_status.items())) + ".",
         "Regenerate: `python -m movement_discovery.cli manifest` (from run2/src). Integration-check lines are quoted from module docstrings (never claims of performance).", "",
         "| id | family | status | requires | G1 | G1H | H1 | file | scenarios | strategy space | integration check | known limitations |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    esc = lambda s: str(s).replace("|", "\\|")
    for r in rs:
        rb = [("ok" if r["robots"][x] == "ok" else esc(r["robots"][x])) for x in ROBOTS]
        L.append(f"| {r['id']} | {r['family']} | {r['status']} | {esc(', '.join(r['requires']))} | {rb[0]} | {rb[1]} | {rb[2]} | `{r['file']}` | {r['scenarios']} | {esc(', '.join(r['strategy_space']))} | {esc(r['integration_check'])} | {esc(r['limitations'])} |")
    L += ["", "## Unsupported-capability report", "", "Programs that do not resolve on a robot (embodiment.resolve refuses; nothing is approximated):", ""]
    for rb in ROBOTS:
        L.append(f"### {rb}: {rep['counts'][rb]} unavailable"); L.append("")
        for pid, note in sorted(rep["by_robot"][rb].items()): L.append(f"- {pid}: {note}")
        L.append("")
    L += ["### Blocking capabilities (robot:capability -> programs blocked)", ""] + [f"- {k}: {v}" for k, v in rep["by_capability"].items()] + [""]
    if registry.IMPORT_ERRORS: L += ["## Import errors", ""] + [f"- {k}: {v}" for k, v in registry.IMPORT_ERRORS.items()] + [""]
    open(f"{out_dir}/MANIFEST.md", "w").write("\n".join(L))
    return dict(md=f"{out_dir}/MANIFEST.md", json=f"{out_dir}/MANIFEST.json", n=len(rs), n_scenarios=len(sc), unsupported=rep["counts"])
