"""Job generators for the four phases. Every generator reads the scenario bank (scenarios/manifest.jsonl) and, for the phases after
SCREEN, the previous phase's <job>.json results on disk (passers), so `plan <phase>` can be re-run at any time and sees exactly
what has been executed. Multi-step Plans are skipped with a reason until the runner offers a sequence search (plan_search_available)."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import collections, math
from movement_discovery.core import registry
from movement_discovery.core.contracts import Scenario, _hash
from movement_discovery.scenarios import manifest as SM, generators as G
from movement_discovery.optimization import SCREEN, EMERGENCE
from . import phase_dir
from .queue import Job, results
from .rules import regime_of

SCREEN_SEEDS = (0, 1); EMERGENCE_SEEDS = (0, 1, 2)
SEQUENCE_SEARCH_NAMES = ("run_plan_search", "search_sequence", "run_sequence_search", "optimize_sequence", "SEQUENCE_SEARCH")
ARMS = ("tau_arm_0.5", "tau_legs_0.5", "human_rom", "mirror")


def plan_search_available() -> bool:
    """True when the runner searches multi-step Plans: optimization.sequence.search_sequence exists AND runner.run's source calls it
    (wiring, 2026-10-13); older names are still accepted. The campaign never edits runner."""
    import inspect
    from movement_discovery import runner
    try:
        from movement_discovery.optimization.sequence import search_sequence   # noqa
        if "search_sequence" in inspect.getsource(runner.run): return True
    except Exception: pass
    return any(getattr(runner, n, None) for n in SEQUENCE_SEARCH_NAMES)


def nominal_row(program_id: str, rows: Optional[List[dict]] = None) -> Optional[dict]:
    """The nominal scenario: recipe 'seed', seed 0, G1 if the program resolves on G1 else the first robot in the bank."""
    rs = [r for r in SM.rows_for(program_id, rows) if r["recipe"] == "seed" and int(r["seed"]) == 0]
    return next((r for r in rs if r["robot"] == "G1"), rs[0] if rs else None)


def probe(program_id: str, row: dict) -> dict:
    """n_steps / missing / first representation of the program's Plan for this scenario (program.plan is MuJoCo-free and ~ms)."""
    try:
        p = registry.get(program_id); sc = G.to_scenario(row); pl = p.plan(sc); ss = p.strategy_space(sc)
        return dict(n_steps=len(pl.steps), missing=[dict(goal=m.get("goal"), why=m.get("why")) for m in pl.missing], representation=ss[0], family=p.family.value, error="")
    except Exception as e:
        return dict(n_steps=None, missing=[], representation=None, family=getattr(registry._REG.get(program_id), "family", None) and registry._REG[program_id].family.value or "", error=f"{type(e).__name__}: {str(e)[:160]}")


def _job(phase: str, row: dict, tier: str, budget: int, seed: int, pr: dict, arm: Optional[str] = None, params: Optional[dict] = None, scenario_id: Optional[str] = None, skip: str = "") -> Job:
    j = Job(phase=phase, program_id=row["program_id"], scenario_id=scenario_id or row["id"], robot=row["robot"], representation=pr.get("representation") or "torque_knots+tracked_prior",
            budget_tier=tier, budget=int(budget), seed=int(seed), scenario_seed=int(row["seed"]), scenario_params=dict(params if params is not None else row["params"]), arm=arm,
            family=pr.get("family") or row.get("family", ""), regime=regime_of(row["program_id"]), n_steps=pr.get("n_steps"), missing=list(pr.get("missing") or []))
    if skip: j.state = "skipped"; j.reason = skip
    return j


def _skip_reason(pr: dict, seq_ok: bool) -> str:
    if pr.get("error"): return f"plan probe failed: {pr['error']}"
    if pr.get("n_steps") == 0: return "no executable step (Plan.missing: " + "; ".join(f"{m['goal']}: {m['why']}" for m in pr["missing"]) + ")"
    if (pr.get("n_steps") or 0) > 1 and not seq_ok: return f"multi-step Plan ({pr['n_steps']} steps): runner.run(execute=True) has no sequence search yet (missing asset)"
    return ""


def passers(prev_phase: str, smoke: bool = False) -> Dict[str, List[dict]]:
    """program_id -> its PASSED result dicts in <prev_phase>/ (a program passes a phase when ANY of its jobs passed; that is the
    documented input rule for the next phase, and the per-seed detail stays in the result files)."""
    out: Dict[str, List[dict]] = collections.defaultdict(list)
    for r in results(phase_dir(prev_phase, smoke)):
        if r.get("pass"): out[r["program_id"]].append(r)
    return dict(out)


def passing_reps(results_: List[dict], pr: dict) -> List[str]:
    """The representation(s) a program PASSED with in the previous phase (a passer found through an appended representation, e.g. the
    brace prior, carries THAT representation forward; the program's first representation is the fallback for results without one)."""
    reps = sorted({r.get("representation") for r in results_ if r.get("representation")})
    return reps or [pr.get("representation") or "torque_knots+tracked_prior"]


# ---------------------------------------------------------------- generators
def gen_screen(programs: Optional[List[str]] = None, **_) -> List[Job]:
    rows = SM.read(); seq_ok = plan_search_available(); registry.load_all(); jobs = []
    for m in registry.manifest():
        pid = m["id"]
        if programs and pid not in programs: continue
        row = nominal_row(pid, rows)
        if row is None:
            jobs.append(Job(phase="screen", program_id=pid, scenario_id=f"{pid}_none", robot="G1", representation="?", budget_tier="screen", budget=SCREEN, seed=0, family=m["family"], regime=regime_of(pid), state="skipped", reason="no nominal scenario in the bank")); continue
        pr = probe(pid, row); skip = _skip_reason(pr, seq_ok)
        for s in SCREEN_SEEDS: jobs.append(_job("screen", row, "screen", SCREEN, s, pr, skip=skip))
    return jobs


def gen_breadth(smoke: bool = False, **_) -> List[Job]:
    rows = SM.read(); seq_ok = plan_search_available(); jobs = []
    P = passers("screen", smoke)
    for pid in sorted(P):
        for row in SM.rows_for(pid, rows):
            pr = probe(pid, row)
            for rep in passing_reps(P[pid], pr): jobs.append(_job("breadth", row, "screen", SCREEN, 0, dict(pr, representation=rep), skip=_skip_reason(pr, seq_ok)))
    return jobs


def gen_emergence(smoke: bool = False, **_) -> List[Job]:
    rows = SM.read(); seq_ok = plan_search_available(); jobs = []
    P = passers("screen", smoke)
    for pid in sorted(P):
        row = nominal_row(pid, rows); pr = probe(pid, row); skip = _skip_reason(pr, seq_ok)
        for rep in passing_reps(P[pid], pr):
            for s in EMERGENCE_SEEDS: jobs.append(_job("emergence", row, "emergence", EMERGENCE, s, dict(pr, representation=rep), skip=skip))
    return jobs


# ---------------------------------------------------------------- counterfactual arms (embodiment helpers; the scenario carries them)
MIRROR_KEYS = ("support", "site", "az", "direction", "dir", "offset", "side", "hand", "foot")


def _mirror_val(k: str, v):
    swap = {"left": "right", "right": "left", "left_palm": "right_palm", "right_palm": "left_palm", "left_foot": "right_foot", "right_foot": "left_foot", "hand_l": "hand_r", "hand_r": "hand_l", "foot_l": "foot_r", "foot_r": "foot_l"}
    if isinstance(v, str): return swap.get(v, v)
    if k == "az" and isinstance(v, (int, float)): return -float(v)
    if k == "dir" and isinstance(v, (int, float)): return -float(v)
    if k == "offset" and isinstance(v, (int, float)): return -float(v)
    if k == "direction" and isinstance(v, (list, tuple)) and len(v) == 2: return [float(v[0]), -float(v[1])]
    return v


def mirror_params(params: dict) -> Tuple[dict, List[str]]:
    """Left<->right mirror of the scenario's lateral parameters (support foot, hand site, azimuth sign, lateral offset). Returns the
    new params and the keys that changed; no change = the mirror arm has no bridge for this program (the driven-set mirror,
    embodiment.mirror_driven, needs a scenario->demand['driven'] bridge the runner does not offer today)."""
    out = dict(params); changed = []
    for k in MIRROR_KEYS:
        if k in out:
            nv = _mirror_val(k, out[k])
            if nv != out[k]: out[k] = nv; changed.append(k)
    if "Jvec" in out and "dir" in changed and isinstance(out.get("J"), (int, float)): out["Jvec"] = [float(out["J"])*math.cos(out["dir"]), float(out["J"])*math.sin(out["dir"]), 0.0]
    if "F" in out and "dir" in changed and isinstance(out.get("Fmag"), (int, float)): out["F"] = [float(out["Fmag"])*math.cos(out["dir"]), float(out["Fmag"])*math.sin(out["dir"]), 0.0]
    return out, changed


def arm_params(arm: str, row: dict) -> Tuple[dict, str]:
    """Scenario params for a counterfactual arm (embodiment helpers' plain-dict conventions: runner.assemble forwards tau_scale /
    jnt_ranges through embodiment.capacity into every Plan step's demand). Returns (params, skip reason or '')."""
    from movement_discovery.embodiment import torque_scale
    from movement_discovery.embodiment.adapter import HUMAN_SHOULDER
    p = dict(row["params"])
    if arm == "tau_arm_0.5": return dict(p, tau_scale=torque_scale(shoulder=0.5, elbow=0.5, wrist=0.5)), ""
    if arm == "tau_legs_0.5": return dict(p, tau_scale=torque_scale(legs=0.5)), ""
    if arm == "human_rom":
        side = "left" if str(p.get("site", p.get("support", ""))).startswith("left") and "palm" in str(p.get("site", "")) else "right"
        from movement_discovery.embodiment import names as N
        src = HUMAN_SHOULDER if side == "right" else {N.mirror(k): v for k, v in HUMAN_SHOULDER.items()}
        return dict(p, jnt_ranges={k: list(v) for k, v in src.items()}), ""
    if arm == "mirror":
        q, changed = mirror_params(p)
        return q, ("" if changed else "mirror: no lateral parameter in the scenario (driven-set mirroring needs a scenario->demand bridge the runner lacks)")
    raise ValueError(arm)


def gen_counterfactual(smoke: bool = False, **_) -> List[Job]:
    rows = SM.read(); seq_ok = plan_search_available(); jobs = []
    P = passers("emergence", smoke)
    for pid in sorted(P):
        row = nominal_row(pid, rows); pr = probe(pid, row); base_skip = _skip_reason(pr, seq_ok)
        for rep in passing_reps(P[pid], pr):
            for arm in ARMS:
                params, skip = arm_params(arm, row); sid = f"{pid}_{row['robot']}_{_hash([pid, row['robot'], params])}"
                jobs.append(_job("counterfactual", row, "emergence", EMERGENCE, 0, dict(pr, representation=rep), arm=arm, params=params, scenario_id=sid, skip=base_skip or skip))
    return jobs


def resolve_programs(names: List[str]) -> List[str]:
    """'P57' or 'P57_breakaway_static_friction' -> registered ids (unknown names raise)."""
    registry.load_all(); ids = [m["id"] for m in registry.manifest()]; out = []
    for n in names:
        hit = [i for i in ids if i == n or i.split("_")[0] == n]
        if not hit: raise KeyError(f"unknown program {n!r}")
        out += hit
    return out


def gen_append(phase: str, programs: List[str], representation: str, **_) -> List[Job]:
    """Extra jobs for ONE named representation (e.g. a brace prior listed LAST in strategy_space, which the screen's first-representation
    rule never reaches): nominal scenario, the phase's seeds and budget tier; skipped with a reason when the program does not list it."""
    rows = SM.read(); seq_ok = plan_search_available(); jobs = []
    seeds = {"screen": SCREEN_SEEDS, "breadth": (0,), "emergence": EMERGENCE_SEEDS, "counterfactual": (0,)}[phase]
    tier, budget = ("emergence", EMERGENCE) if phase in ("emergence", "counterfactual") else ("screen", SCREEN)
    for pid in resolve_programs(programs):
        row = nominal_row(pid, rows); pr = probe(pid, row); skip = _skip_reason(pr, seq_ok)
        try: ss = registry.get(pid).strategy_space(G.to_scenario(row))
        except Exception as e: ss = []; skip = skip or f"strategy_space failed: {type(e).__name__}: {e}"
        if not skip and representation not in ss: skip = f"representation {representation!r} not in strategy_space {ss}"
        for s in seeds: jobs.append(_job(phase, row, tier, budget, s, dict(pr, representation=representation), skip=skip))
    return jobs


GENERATORS = dict(screen=gen_screen, breadth=gen_breadth, emergence=gen_emergence, counterfactual=gen_counterfactual)


def generate(phase: str, smoke: bool = False, programs: Optional[List[str]] = None) -> List[Job]:
    return GENERATORS[phase](smoke=smoke, programs=programs)


def to_scenario(job: Job) -> Scenario:
    return Scenario(id=job.scenario_id, program_id=job.program_id, seed=int(job.scenario_seed), params=dict(job.scenario_params))
