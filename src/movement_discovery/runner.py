"""Runner: assembles embodiment -> world -> Plan (steps) -> representation per step -> (search) -> evaluate -> validity -> observe -> archive.
    assemble(program_id, scenario, robot, representation, horizon)    # everything up to simulation; asm["steps"] one entry per Plan step
    rollout(program_id, scenario, robot, representation)              # ONE prior-only rollout of the Plan (single step or SEQUENCE); raw record
    check(program_id, seed, robot="G1", representation=None)          # INTEGRATION CHECK: plant + ONE prior-only rollout + evaluate + validity, one line
    run(program_id, scenario, robot, representation, optimizer, budget, execute=False)   # execute=False: plan only (no simulation)
Programs expose plan(scenario) -> core.contracts.Plan (demand() returns a Plan or the legacy (reference task, demand dict) pair). A multi-step
Plan is executed by run_sequence: the ONE sequence runner (replaces tasks/force_interaction/_base.run_steps over engine.compose.run_sequence):
each step starts from the previous step's REAL final state, mapped by joint name across differing plants (construction.state_map; the
NQ_MAPPING gap of engine.compose), handed off through engine.core.handoff (or demand["initial_state"] for the chained ContactSchedule),
held objects carried forward, engine.compose's freeze / settle between steps, no adaptation search (build phase). World specs with
objects/surfaces go through world.build and are passed to the task as its scene when the task accepts one; capacity perturbations of the
scenario (tau_scale / jnt_ranges) reach every step's demand through embodiment.capacity. Nothing here authors a movement."""
from __future__ import annotations
from typing import List, Optional, Union
import contextlib, os, time, json
import numpy as np
from movement_discovery.core.contracts import Scenario, Candidate, Outcome, SearchRecord, Status, Plan, Step
from movement_discovery.core import registry
from movement_discovery import embodiment as E, world as W, construction as C
from movement_discovery.embodiment import capacity as CAP
from movement_discovery.physics import validity as V, events as EV
from movement_discovery.telemetry import Observer, observe, describe, summarize, one_line
from movement_discovery.archive import Archive, make_entry

SCRATCH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "results/md_build")


FAMILIES = ("locomotion", "aerial", "balance", "posture", "reaching", "projection", "impact", "force_interaction", "payload", "interception", "negotiation", "compound")
IMPORT_ERRORS = registry.IMPORT_ERRORS          # shared with core.registry.load_all


class NotExecutable(NotImplementedError):
    """The Plan has no executable step: the program names its missing assets (Plan.missing) instead of passing silently."""
    def __init__(self, program_id, missing): super().__init__(f"{program_id}: no executable step; missing: " + "; ".join(f"{m.get('goal')}: {m.get('why')}" for m in missing)); self.missing = list(missing)


def program_(program_id: str):
    """registry.get with DEFENSIVE family loading: a family package that fails to import (mid-edit) is recorded in
    IMPORT_ERRORS and skipped instead of taking every other program down with it."""
    if program_id not in registry._REG: registry.load_all()
    if program_id not in registry._REG: raise KeyError(f"program {program_id!r} not registered; family import errors: {IMPORT_ERRORS}")
    return registry._REG[program_id]()


def _scenario(program, scenario: Union[int, Scenario, dict], robot: str) -> Scenario:
    if isinstance(scenario, Scenario): return scenario
    if isinstance(scenario, dict): return program.sample_scenario(int(scenario.pop("seed", 0)), robot=robot, **scenario)
    return program.sample_scenario(int(scenario), robot=robot)


def _glue(task, demand: dict, wspec, wpath: str, posed: bool) -> float:
    """World glue for one step: the MJCF goes to the task as `scene` (attribute when the task has one, demand["scene"] otherwise);
    objects/surfaces are listed for the representations that hold objects (ContactSchedule); raised terrain under the start becomes
    demand["start_z"] (physics.settle; not for posed starts: the seat is under the pelvis, not the feet). Returns start_z."""
    if wpath:
        if hasattr(task, "scene") and getattr(task, "scene", None) is None: task.scene = wpath
        demand.setdefault("scene", wpath)
        if wspec.objects: demand.setdefault("world_objects", list(wspec.objects)); demand.setdefault("world_surfaces", list(wspec.surfaces))
    dz = float(W.start_z_offset(wspec)) if (wpath and not posed) else 0.0
    if dz > 1e-3: demand.setdefault("start_z", dz)
    if "driven" not in demand or demand.get("driven") is None: demand["driven"] = None       # the task's own default (throw.CFG set) unless the program says otherwise
    return dz


def assemble(program_id: str, scenario, robot: str = "G1", representation: Optional[str] = None, horizon: Optional[float] = None) -> dict:
    """Everything up to (not including) simulation: program, scenario, embodiment spec, world path, the Plan and, per step, task + demand
    (world glue, posture start, capacity perturbations) + representation. asm["task"/"demand"/"rep"] are the FIRST step's (the single-step
    view every caller had before); asm["steps"] carries all of them. A Plan without steps raises NotExecutable (its missing assets)."""
    program = program_(program_id); sc = _scenario(program, scenario, robot); emb = E.resolve(robot, program, driven=None)
    wspec = program.world(sc); wpath = W.build(wspec, robot); plan = program.plan(sc)
    if not plan.steps: raise NotExecutable(program_id, plan.missing)
    rep_name = representation or program.strategy_space(sc)[0]; steps = []
    for i, st in enumerate(plan.steps):
        d = dict(st.demand); ws = st.world if st.world is not None else wspec; wp = W.build(ws, robot) if st.world is not None else wpath
        if i == 0 and hasattr(program, "initial_plant") and getattr(program, "start_posture", lambda s: None)(sc) is not None and d.get("initial_posture") is None:
            S0, info = program.initial_plant(sc); d["initial_posture"] = np.asarray(S0["q0"], float); d["initial_posture_info"] = info
        dz = _glue(st.task, d, ws, wp, posed=d.get("initial_posture") is not None)
        d = CAP.apply(d, sc.params, robot)                                                      # tau_scale / jnt_ranges of the scenario -> the demand (every step)
        rn = (st.representation if (st.representation and representation is None) else rep_name); rep = C.get(rn)(st.task, horizon=horizon)
        steps.append(dict(i=i, step=st, task=st.task, demand=d, rep=rep, representation=rn, world=ws, world_path=wp, start_z=dz, held=list(st.held), event=st.event, hooks=None, capacity=CAP.consumed_by(st.task, d)))
    s0 = steps[0]
    return dict(program=program, scenario=sc, embodiment=emb, world=wspec, world_path=wpath, plan=plan, steps=steps, n_steps=len(steps),
                task=s0["task"], demand=s0["demand"], rep=s0["rep"], representation=rep_name, start_z=s0["start_z"], hooks=None)


def controller_chain(asm: dict, S, fn, obs=None, step: Optional[dict] = None):
    """hooks (mocap / initial velocities / wrenches from the step's WorldSpec) -> task controller -> telemetry observer."""
    from movement_discovery.physics import hooks as HK
    st = step if step is not None else asm["steps"][0]
    nT = len(S["types"]); hk = HK.world_hooks(st["world"], nT, inner=fn); st["hooks"] = hk; asm["hooks"] = hk
    return observe(hk, obs) if obs is not None else hk


def _z_for(st: dict, S, demand, z=None) -> np.ndarray:
    if z is not None: return np.asarray(z, float)
    if st["step"].z is not None: return np.asarray(st["step"].z, float)
    return st["rep"].z0(S, demand)


def _final_state(rec: dict, log: list, S):
    fs = rec.get("final_state") if rec.get("final_state") is not None else (rec.get("extra") or {}).get("final_state")
    if fs is None: fs = (log[-1][1], log[-1][2]) if log else (S["q_settled"], S["v_settled"])
    return np.asarray(fs[0], float), np.asarray(fs[1], float)


def simulate_step(asm: dict, st: dict, state=None, S_prev=None, z=None, record: bool = True, held: Optional[list] = None) -> dict:
    """ONE rollout of one Plan step from `state` (None = the plant's settled start): plant, z (given | stored | prior z0), hooks +
    observer + logger chain, handoff (construction.state_map across differing plants; engine.core.handoff, or demand["initial_state"]
    for a chained ContactSchedule, which builds its own phase plants), record. Returns dict(S, rec, z, obs, log, handoff, mapped)."""
    from engine import core, compose as CP
    from movement_discovery.construction.state_map import map_to_plant
    from movement_discovery.construction.contact_schedule import ContactSchedule
    rep, demand = st["rep"], dict(st["demand"])
    if held: demand["held"] = list(held)                                                        # world state carried across steps (engine.compose convention)
    S = rep.plant(demand, core.DT_OPT); CAP.ensure_ranges(S, demand); mapped = None
    if state is not None:
        q, v, mapped = map_to_plant(S_prev, state, S) if S_prev is not None else (np.asarray(state[0], float), np.asarray(state[1], float), dict(copied="all", kept=[], dropped=[])); state = (q, v)
    hc = CP.handoff_check(S, *state) if state is not None else dict(dq=0.0, dv=0.0, upright=1.0, ok=True)
    chained = isinstance(rep, ContactSchedule)
    if state is not None and chained: demand["initial_state"] = (state[0], state[1])            # the schedule hands the state into its own phase plants
    z = _z_for(st, S, demand, z); obs = Observer(S); p = rep.unpack(S, demand, z); fn = rep.controller(S, demand, p); chain = controller_chain(asm, S, fn, obs, st); log = []
    def fn_log(t, d, S_):
        log.append((t, d.qpos.copy(), d.qvel.copy())); return chain(t, d, S_)
    ctx = core.handoff(S, state[0], state[1], hold=demand.get("hold_mode", "keep")) if (state is not None and not chained) else contextlib.nullcontext()
    with ctx: rec = rep.rollout(S, demand, p, ctrl_fn=fn_log, record=record)
    rec["dt"] = float(S["m"].opt.timestep); rec["params"] = p
    return dict(S=S, rec=rec, z=z, obs=obs, log=log, handoff=hc, mapped=mapped, demand=demand)


def run_sequence(asm: dict, zs: Optional[List] = None, record: bool = True, step_fn=None) -> dict:
    """THE sequence runner for multi-step Plans (engine.compose.run_sequence's loop, re-owned here with the state mapping it lacked and
    without its adaptation search). Record = compose's format (steps=[...], trace_q=[(t, qpos)], ok, t_total, final_state, held) PLUS the
    single-rollout keys the evaluator and the family policy read (value = last step's, reasons prefixed s<i>:, violation summed, extra =
    last step's extra + steps summary, trace None). plan.missing is attached as rec["missing"] (never a silent pass)."""
    from engine import compose as CP
    plan, steps = asm["plan"], asm["steps"]; state = None; S_prev = None; S = None; out = []; trace = []; t_off = 0.0; ok = True; held = []; reasons = []; viol = 0.0
    for i, st in enumerate(steps):
        r = step_fn(i, st, state, S_prev, held) if step_fn is not None else simulate_step(asm, st, state=state, S_prev=S_prev, z=(zs[i] if zs else None), record=record, held=held)   # step_fn: a per-step SEARCH (optimization.sequence) returning simulate_step's dict
        S, rec, log, hc = r["S"], r["rec"], r["log"], r["handoff"]
        task = st["task"]; ex = rec.get("extra") or {}
        ok &= bool(rec["valid"] and rec["violation"] <= 0 and hc["ok"])
        for (t, q, v) in log: trace.append((t_off + t, q))
        for (t, q) in ex.get("land_trace") or []: trace.append((t_off + t, q))
        q_end, v_end = _final_state(rec, log, S); t_end = max([t for t, _ in trace[-1:]] + [log[-1][0] if log else 0.0]) - t_off; t_off += t_end
        held = list(ex.get("held_after", held)) if ex.get("world_after") is not None else held
        last = (i == len(steps) - 1)
        if plan.settle_T and plan.settle_T > 0 and (plan.settle_last or not last):
            q_s, v_s, tr, at_rest = CP.settle(S, plan.robot, q_end, v_end, T=plan.settle_T, held=held, task=task)
            for (t, q) in tr: trace.append((t_off + t, q))
            t_off += plan.settle_T + 1.0; state = (q_s, v_s)
        elif plan.freeze_T and plan.freeze_T > 0:      # manipulation sequences hand over at the CURRENT posture: a short freeze removes the residual motion (engine.compose)
            q_f, v_f, tr, at_rest = CP.freeze(S, q_end, v_end, T=plan.freeze_T, held=held, task=task)
            for (t, q) in tr: trace.append((t_off + t, q))
            t_off += plan.freeze_T; state = (q_f, v_f)
        else:
            at_rest = float(np.max(np.abs(np.asarray(v_end)[CP._rj(S)]))) < 0.5; state = (np.asarray(q_end), np.asarray(v_end))
        out.append(dict(i=i, task=getattr(task, "name", type(task).__name__), demand=r["demand"], value=rec["value"], valid=rec["valid"], reasons=list(rec["reasons"]), violation=rec["violation"], handoff=hc, settled=at_rest, adapted=False, alt_used=None, z=np.asarray(r["z"]).tolist(),
                        extra={k: v for k, v in ex.items() if k not in ("trace", "landing", "final_state", "land_trace")}, mapped=r["mapped"], representation=st["representation"], event=st["event"], telemetry=r["obs"].summary(), hooks=dict(st["hooks"].stats, active=st["hooks"].active) if st["hooks"] is not None else None, search=r.get("search")))
        reasons += [f"s{i}:{rr}" for rr in rec["reasons"]]; viol += float(rec["violation"]); ok &= at_rest; S_prev = S
        asm["obs_summary"] = r["obs"].summary()
    lx = out[-1]["extra"]
    rec_seq = dict(steps=out, trace=None, trace_q=trace, ok=bool(ok), t_total=t_off, final_state=state, held=held, missing=list(plan.missing), value=float(out[-1]["value"]),
                   reasons=reasons + ([] if ok else ["SEQUENCE_NOT_OK"]), violation=float(viol), valid=bool(ok and not reasons), cost=float(sum(float(s["extra"].get("cost", 0.0) or 0.0) for s in out)),
                   extra=dict(lx, n_steps=len(out), t_total=t_off, steps=[dict(i=s["i"], task=s["task"], value=s["value"], valid=s["valid"], reasons=s["reasons"], settled=s["settled"], handoff=s["handoff"], event=s["event"]) for s in out]), dt=float(S["m"].opt.timestep), params=dict(steps=[s["z"] for s in out]))
    return dict(S=S, rec=rec_seq)


def simulate(asm: dict, z=None, record: bool = True) -> dict:
    """ONE rollout of the assembled Plan: a single step straight (z = given | stored | prior z0), several steps through run_sequence.
    Returns dict(S, rec) with asm["obs_summary"] set (telemetry of the (last) step)."""
    if asm["n_steps"] == 1:
        r = simulate_step(asm, asm["steps"][0], z=z, record=record); asm["obs_summary"] = r["obs"].summary(); r["rec"].setdefault("missing", list(asm["plan"].missing)); return dict(S=r["S"], rec=r["rec"], z=r["z"])
    if z is not None: raise NotImplementedError("a multi-step Plan takes per-step z lists (zs), not one vector; the search over a sequence is a missing asset (per-step adaptation at the handoff, engine.compose adapt_iters)")
    return run_sequence(asm, record=record)


def rollout(program_id: str, scenario, robot: str = "G1", representation: Optional[str] = None, record: bool = True) -> dict:
    """ONE prior-only rollout of a program's Plan (the families' former program.run(scenario)): dict(asm, S, rec); rec is the raw record
    program.evaluate consumes (a sequence record for multi-step Plans)."""
    asm = assemble(program_id, scenario, robot, representation); r = simulate(asm, record=record); return dict(asm=asm, S=r["S"], rec=r["rec"])


def evaluate(asm: dict, S, rec: dict, with_mechanics: bool = True) -> dict:
    program, sc = asm["program"], asm["scenario"]; pol = V.policy_for(program.family.value); spec = asm["steps"][-1]["world"]
    launched = any(o.get("qvel0") for o in spec.objects); rec = V.strip_object_rest(rec, S, launched)                 # rest judged on robot dofs when objects are launched
    hk = asm["hooks"].stats if asm.get("hooks") is not None else {}; W_ext = V.external_work(rec, asm.get("obs_summary"), spec if launched else None, S["m"])
    if launched or hk.get("wrench_steps"): rec = V.rejudge_energy_closure(rec, W_ext)                                   # LABEL energy_rejudged: closure with external work credited
    out = program.evaluate(rec, sc)
    out.measures["policy_validity"] = V.report(rec, pol).reasons     # the PROGRAM owns validity (rule 7); the family policy is attached for comparison
    ev = EV.detect(rec); desc = describe(rec, S, events=ev)
    if with_mechanics and out.mechanics is None: out.mechanics = summarize(S, rec, asm["steps"][-1]["demand"].get("effector"))
    return dict(outcome=out, events=ev, descriptors=desc, policy=pol.name)


def _hooks_note(asm: dict) -> str:
    notes = []
    for st in asm["steps"]:
        hk = st["hooks"]
        if hk is None: continue
        s = (f"hooks {hk.stats}" if hk.active else "hooks none") + (" (ctrl_fn not called by the task: hooks NOT applied)" if hk.active and hk.stats["calls"] == 0 else "")
        notes.append(s if asm["n_steps"] == 1 else f"s{st['i']}: {s}")
    return " | ".join(notes)


def check(program_id: str, seed: Union[int, Scenario, dict] = 0, robot: str = "G1", representation: Optional[str] = None, verbose: bool = True) -> dict:
    """The allowed INTEGRATION CHECK: build the plant(s), ONE prior-only rollout (z0; no search) of the Plan, evaluate, validity, one line."""
    from engine import core
    t0 = time.time(); asm = assemble(program_id, seed, robot, representation); r = simulate(asm, record=True); S, rec = r["S"], r["rec"]
    ev = evaluate(asm, S, rec); o = ev["outcome"]; tele = asm.get("obs_summary") or {}
    n_params = int(sum(st["rep"].n_params(st["rep"].plant(st["demand"], core.DT_OPT), st["demand"]) for st in asm["steps"]))
    cap = [st["capacity"] for st in asm["steps"] if st["capacity"].get("requested")]; cap_note = f" | capacity {cap[0]}" if cap else ""
    steps_note = "" if asm["n_steps"] == 1 else f" | steps {asm['n_steps']} " + " ".join(f"s{s['i']}:{s['task']}:{'ok' if (s['valid'] and s['settled']) else 'x'}" for s in rec["steps"]) + (f" missing {rec['missing']}" if rec.get("missing") else "")
    reps = asm["representation"] if asm["n_steps"] == 1 else "+".join(st["representation"] for st in asm["steps"])
    line = (f"[check {program_id} s{asm['scenario'].seed} {robot} {reps}] value {o.value:.3f} achieved {o.achieved} reasons {o.validity.reasons} viol {o.validity.violation:.2f} "
            f"| policy {ev['policy']} | events {EV.summary(ev['events'])['kinds']} | {one_line(o.mechanics)} | n_params {n_params} | start_z {asm['start_z']:.3f} | {_hooks_note(asm)}{cap_note}{steps_note} | {time.time() - t0:.0f} s")
    if verbose: print(line, flush=True)
    return dict(line=line, outcome=o, events=ev["events"], descriptors=ev["descriptors"], telemetry=tele, embodiment=asm["embodiment"], representation=asm["rep"].describe(), n_params=n_params, n_steps=asm["n_steps"], rec=rec)


def run(program_id: str, scenario, robot: str = "G1", representation: Optional[str] = None, optimizer: str = "cmaes", budget: int = 2400, execute: bool = False, seed: int = 0, pop: int = 16, archive: Optional[Archive] = None, horizon: Optional[float] = None, warm: Optional[dict] = None, robust: bool = True) -> dict:
    """robust: search with the worse-of-two-timesteps measure on EVERY candidate (emergence tier; 5x the cost on long-horizon tasks) or at 1 ms only with the two-timestep check at the end (screen tier)."""
    """Full pipeline. execute=False (default) returns the PLAN (what would run, parameter count, budget) without simulating.
    execute=True runs the search (an optimisation campaign: not for the build phase), measures (two timesteps), observes, archives.
    Multi-step Plans: sequential per-step search (optimization.sequence): budgets per step = total * weights (Plan.weights /
    demand["budget_weight"], else equal), each step optimised from the best previous end state, one ArchiveEntry per step + one for the sequence."""
    from movement_discovery import optimization as O
    from engine import core
    asm = assemble(program_id, scenario, robot, representation, horizon); rep, demand, sc = asm["rep"], asm["demand"], asm["scenario"]
    plan = dict(program_id=program_id, scenario=sc.to_json(), robot=robot, representation=rep.describe(), optimizer=optimizer, budget=int(budget), pop=pop, seed=seed, embodiment_hash=asm["embodiment"].hash(), world_hash=asm["world"].hash(), execute=execute,
                n_steps=asm["n_steps"], steps=[dict(i=st["i"], task=getattr(st["task"], "name", ""), representation=st["representation"], event=st["event"], held=st["held"]) for st in asm["steps"]], missing=list(asm["plan"].missing))
    if not execute:
        ns = [int(st["rep"].n_params(st["rep"].plant(st["demand"], core.DT_OPT), st["demand"])) for st in asm["steps"]]; plan["n_params"] = int(sum(ns)); plan["n_params_steps"] = ns; plan["status"] = "planned (execute=False)"; return plan
    t0 = time.time(); arc = archive or Archive()
    if asm["n_steps"] > 1:                                   # SEQUENTIAL PER-STEP SEARCH (optimization.sequence): one entry per step + the sequence entry
        from movement_discovery.optimization.sequence import search_sequence
        sq = search_sequence(asm, optimizer=optimizer, budget=budget, seed=seed, pop=pop, archive=arc); S, rec = sq["S"], sq["rec"]
        ev = evaluate(asm, S, rec); o = ev["outcome"]; o.validity.timestep_robust = None; o.measures["steps_search"] = sq["searches"]
        cand = Candidate(representation="sequence:" + "+".join(st["representation"] for st in asm["steps"]), params=dict(steps=sq["zs"]), duration=float(rec.get("t_total", 0.0)), provenance=dict(generator=optimizer, seed=seed, budget=budget, budgets=sq["budgets"], scenario_params=sc.params, parents=sq["step_entries"]))
        srec = SearchRecord(optimizer=optimizer, config=dict(pop=pop, per_step=True, robust=False), seed=seed, budget_evals=int(sum(s_["evals"] for s_ in sq["searches"])), best_id=cand.id, history=[], secs=float(sq["secs"]))
        entry = make_entry(program_id, sc.id, asm["embodiment"].hash(), asm["world"].hash(), cand, o, srec, ev["descriptors"]); arc.append(entry)
        return dict(plan, outcome=o, candidate_id=cand.id, step_entries=sq["step_entries"], budgets=sq["budgets"], searches=sq["searches"], events=ev["events"], descriptors=ev["descriptors"], telemetry=asm.get("obs_summary"), secs=time.time() - t0, status="executed (sequence: per-step entries + sequence entry written; program status unchanged)")
    S = rep.plant(demand); CAP.ensure_ranges(S, demand)
    if optimizer == "cmaes":
        from movement_discovery.optimization.cmaes import optimize_task
        res = optimize_task(rep, demand, seed=seed, budget=budget, pop=pop, warm=warm, robust=robust); z = np.asarray(res["z"]); hist = []
    else:
        obj = O.TaskObjective(rep, demand, robust=robust); opt = O.get(optimizer)(seed=seed, pop=pop); r = opt.run(obj, obj.n, z0=(np.asarray(warm["x"]) if warm else obj.z0()), budget=budget, sigma=rep.sigma0(demand, warm)); z = np.asarray(r["z"]); res = obj.measure(z); res.update(secs=r["secs"], optimizer=optimizer); hist = r["history"]
    r1 = simulate(asm, z=z, record=True); S, rec, p = r1["S"], r1["rec"], r1["rec"]["params"]
    ev = evaluate(asm, S, rec); o = ev["outcome"]; o.validity.timestep_robust = bool(res.get("ok_all")); o.measures["dt_check"] = res.get("meas")
    cand = Candidate(representation=asm["representation"], params=dict(z=z.tolist(), **{k: v for k, v in p.items() if k not in ("x", "knots")}), duration=float(rec["extra"].get("t_rel", 0.0) or 0.0), provenance=dict(generator=optimizer, seed=seed, budget=budget, scenario_params=sc.params, parent=(warm or {}).get("parent")))
    srec = SearchRecord(optimizer=optimizer, config=dict(pop=pop), seed=seed, budget_evals=int(budget), best_id=cand.id, history=hist, secs=float(res.get("secs", time.time() - t0)))
    o.trace_ref = arc.store_trace(cand.id, rec["trace"]) if rec.get("trace") else None
    if hasattr(rep.task, "ref") and hasattr(rep.task, "_refs"):
        try: arc.store_reference(cand.id, rep.task.ref(S, dict(demand, prior=getattr(rep, "prior", demand.get("prior")))))
        except Exception: pass
    entry = make_entry(program_id, sc.id, asm["embodiment"].hash(), asm["world"].hash(), cand, o, srec, ev["descriptors"]); arc.append(entry)
    return dict(plan, outcome=o, candidate_id=cand.id, events=ev["events"], descriptors=ev["descriptors"], telemetry=asm.get("obs_summary"), measured=res, secs=time.time() - t0, status="executed (archive entry written; program status unchanged)")


if __name__ == "__main__":
    import sys
    check(sys.argv[1] if len(sys.argv) > 1 else "P41_max_release_speed", int(sys.argv[2]) if len(sys.argv) > 2 else 0)
