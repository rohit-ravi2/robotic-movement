"""Sequential per-step search over a multi-step Plan (runner.run execute=True on n_steps > 1): step i is optimised with its own budget
from the REAL state handed off by the best solution of step i-1 (runner.simulate_step: construction.state_map across plants,
engine.core.handoff, held objects carried, engine.compose freeze / settle between steps as runner.run_sequence applies them),
then replayed once with recording. Budgets per step = total * weights / sum(weights) (Plan.weights or demand["budget_weight"]
per step, else equal). Objective = the step representation's lexicographic score at the faithful timestep (no two-dt robust measure
inside the loop: a sequence search is already n_steps x the cost; the final replay is recorded). One ArchiveEntry per step
(scenario_id "<id>#s<i>") is written when an Archive is given; the runner adds the sequence entry whose provenance lists them."""
from __future__ import annotations
from typing import List, Optional
import time
import numpy as np


def step_budgets(plan, steps: List[dict], budget: int) -> List[int]:
    w = getattr(plan, "weights", None) or [float(st["demand"].get("budget_weight", 1.0)) for st in steps]; w = np.asarray(w, float); w = w/max(w.sum(), 1e-9)
    return [max(1, int(round(budget*wi))) for wi in w]


def search_sequence(asm: dict, optimizer: str = "cmaes", budget: int = 2400, seed: int = 0, pop: int = 16, archive=None, record: bool = True) -> dict:
    from movement_discovery import runner as R, optimization as O
    from movement_discovery.core.contracts import Candidate, Outcome, SearchRecord
    from movement_discovery.physics import validity as V
    from movement_discovery.telemetry import describe
    from movement_discovery.archive import make_entry
    from engine import core
    plan, steps, sc = asm["plan"], asm["steps"], asm["scenario"]; budgets = step_budgets(plan, steps, int(budget)); searches = []; entries = []; t0 = time.time()
    def step_fn(i, st, state, S_prev, held):
        rep, demand = st["rep"], dict(st["demand"])
        if held: demand["held"] = list(held)
        S = rep.plant(demand, core.DT_OPT); z0 = R._z_for(st, S, demand); n_ev = [0]
        def obj(z):
            r = R.simulate_step(asm, st, state=state, S_prev=S_prev, z=z, record=False, held=held); n_ev[0] += 1; return float(rep.score(r["rec"], r["demand"]))
        opt = O.get(optimizer)(seed=int(seed) + i, pop=int(pop)); ts = time.time()
        res = opt.run(obj, len(z0), z0=z0, budget=budgets[i], sigma=float(rep.sigma0(demand, None)))
        r = R.simulate_step(asm, st, state=state, S_prev=S_prev, z=np.asarray(res["z"]), record=record, held=held); rec = r["rec"]
        srec = SearchRecord(optimizer=optimizer, config=dict(pop=pop, step=i, robust=False), seed=int(seed) + i, budget_evals=int(res["evals"]), best_id="", history=res["history"], secs=time.time() - ts)
        cand = Candidate(representation=st["representation"], params=dict(z=np.asarray(res["z"]).tolist()), duration=float((rec.get("extra") or {}).get("t_rel", 0.0) or 0.0), provenance=dict(generator=optimizer, seed=int(seed) + i, budget=budgets[i], step=i, scenario_params=sc.params, event=st["event"]))
        srec.best_id = cand.id; r["search"] = dict(evals=res["evals"], score=res["score"], secs=srec.secs, budget=budgets[i], candidate_id=cand.id)
        searches.append(r["search"])
        if archive is not None:
            out = Outcome(value=float(rec["value"]), achieved=bool(rec["valid"] and rec["violation"] <= 0), measures={k: v for k, v in (rec.get("extra") or {}).items() if isinstance(v, (int, float, str, bool))}, validity=V.classify(rec))
            if rec.get("trace"): out.trace_ref = archive.store_trace(cand.id, rec["trace"])
            e = make_entry(asm["program"].id, f"{sc.id}#s{i}", asm["embodiment"].hash(), st["world"].hash(), cand, out, srec, describe(rec, r["S"])); archive.append(e); entries.append(cand.id)
        return r
    seq = R.run_sequence(asm, record=record, step_fn=step_fn)
    return dict(S=seq["S"], rec=seq["rec"], zs=[s["z"] for s in seq["rec"]["steps"]], searches=searches, budgets=budgets, step_entries=entries, secs=time.time() - t0)
